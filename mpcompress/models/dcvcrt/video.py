# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import json
import io
import time
import torch
from torch import nn
import torch.nn.functional as F

from compressai.models.base import CompressionModel
from mpcompress.latent_codecs.dcvc_base import DmcCompressionModel
from mpcompress.utils.registery import instantiate_class, register

from mpcompress.models.dcvcrt.pframe import DMCP
from mpcompress.models.dcvcrt.iframe import DMCI
from mpcompress.utils.stream_helper import (
    SpsManager,
    NalType,
    write_sps,
    read_vps,
    read_sps_remaining,
    read_picture_remaining,
    write_picture,
    write_picture_pps,
)
from mpcompress.datasets.video_reader import PngSequenceVideoReader, YUV420VideoReader
from mpcompress.datasets.video_writer import PngSequenceVideoWriter, YUV420VideoWriter


def replicate_pad(x, pad_b, pad_r):
    if pad_b == 0 and pad_r == 0:
        return x
    return F.pad(x, (0, pad_r, 0, pad_b), mode="replicate")


def np_image_to_tensor(img, device):
    image = torch.from_numpy(img).to(device=device).to(dtype=torch.float32) / 255.0
    image = image.unsqueeze(0)
    return image


def get_src_reader(args):
    if args["src_type"] == "png":
        src_reader = PngSequenceVideoReader(
            args["src_path"], args["src_width"], args["src_height"]
        )
    elif args["src_type"] == "yuv420":
        src_reader = YUV420VideoReader(
            args["src_path"], args["src_width"], args["src_height"]
        )
    return src_reader


@register("DCVC_RT_Video")
class DCVC_RT_Video(CompressionModel):
    # just for inference
    def __init__(
        self,
        dmci_codec={},
        dmcp_codec={},
        use_fp16=True,
        **kwargs,
    ):
        super().__init__()
        self.i_frame_net: DMCI = instantiate_class(dmci_codec)
        self.p_frame_net: DMCP = instantiate_class(dmcp_codec)
        self.qp_offset_map = [0, 1, 0, 2, 0, 2, 0, 2]
        self.use_fp16 = use_fp16
        if self.use_fp16:
            self.i_frame_net.half()
            self.p_frame_net.half()

    def update(self, force_zero_thres=0.12):
        self.i_frame_net.update(force_zero_thres)
        self.p_frame_net.update(force_zero_thres)

    def compress_video(self, src_reader, codec_args):
        self.init_compress(codec_args)
        frame_num = codec_args["frame_num"]
        with torch.no_grad():
            for _ in range(frame_num):
                _frame = src_reader.read_one_frame()
                x = np_image_to_tensor(_frame.yuv444, self.device)
                if self.use_fp16:
                    x = x.half()
                res = self.compress_frame_then_write(x)

        src_reader.close()
        total_bytes = self.close_write()
        total_kbps = int(
            total_bytes * 8 / (self.frame_num / 30) / 1000
        )  # assume 30 fps
        return {"bits": total_bytes * 8}

    def init_compress(self, args):
        self.frame_num = args["frame_num"]
        self.reset_interval = args["reset_interval"]
        self.intra_period = args["intra_period"]
        self.src_height = args["src_height"]
        self.src_width = args["src_width"]
        self.qp_i = args["qp_i"]
        self.qp_p = args["qp_p"]
        if "out_bin_path" in args and args["out_bin_path"]:
            self.out_bin_path = args["out_bin_path"]
            os.makedirs(os.path.dirname(self.out_bin_path), exist_ok=True)
        else:
            self.out_bin_path = "temporary_video.bin"

        self.device = next(self.i_frame_net.parameters()).device
        self.padding_r, self.padding_b = DMCI.get_padding_size(
            self.src_height, self.src_width, 16
        )

        self.use_two_entropy_coders = self.src_height * self.src_width > 1280 * 720
        self.i_frame_net.set_use_two_entropy_coders(self.use_two_entropy_coders)
        self.p_frame_net.set_use_two_entropy_coders(self.use_two_entropy_coders)
        self.p_frame_net.set_curr_poc(0)

        self.output_buff = io.BytesIO()
        self.sps_manager = SpsManager()
        self.frame_idx = 0
        self.last_qp = 0

    def compress_frame_then_write(self, x):
        res = self.compress_frame(x)
        res1 = self.write_frame_by_syntax(res["bit_stream"], res["pstate"])
        res.update(res1)
        return res

    def compress_frame(self, x):
        torch.cuda.synchronize(device=self.device)

        frame_type = 0
        use_ada_i = 0

        if self.frame_idx == 0 or (
            self.intra_period > 0 and self.frame_idx % self.intra_period == 0
        ):
            frame_type = 0
            curr_qp = self.qp_i
        else:
            frame_type = 1
            fa_idx = self.qp_offset_map[self.frame_idx % 8]
            curr_qp = self.p_frame_net.shift_qp(self.qp_p, fa_idx)
            if self.reset_interval > 0 and self.frame_idx % self.reset_interval == 1:
                use_ada_i = 1

        pstate = {
            "qp": curr_qp,
            "height": self.src_height,
            "width": self.src_width,
            "frame_type": frame_type,
            "use_ada_i": use_ada_i,
            "ec_part": 1 if self.use_two_entropy_coders else 0,
        }

        # pad if necessary
        x_padded = replicate_pad(x, self.padding_b, self.padding_r)
        if pstate["frame_type"] == 0:
            encoded = self.i_frame_net.compress(x=x_padded, **pstate)
            self.p_frame_net.clear_dpb()
            self.p_frame_net.add_ref_frame(None, encoded["x_hat"])
        elif pstate["frame_type"] == 1:
            if pstate["use_ada_i"] == 1:
                self.p_frame_net.prepare_feature_adaptor_i(self.last_qp)
            encoded = self.p_frame_net.compress(x=x_padded, qp=pstate["qp"])
            self.last_qp = pstate["qp"]

        torch.cuda.synchronize(device=self.device)
        self.frame_idx += 1
        # encoded.keys(): bit_stream, x_hat
        encoded["pstate"] = pstate
        return encoded

    def decompress_video(self, codec_args, **kwargs):
        self.init_decompress(codec_args)
        frame_num = codec_args["frame_num"]
        decoded_frame_number = 0
        results = []
        with torch.no_grad():
            while decoded_frame_number < frame_num:
                res = self.decompress_frame_by_read()
                results.append(res)
                decoded_frame_number += 1
        self.close_read()
        return results

    def init_decompress(self, args):
        self.src_height = args["src_height"]
        self.src_width = args["src_width"]
        self.device = next(self.i_frame_net.parameters()).device
        self.p_frame_net.set_curr_poc(0)

        self.sps_manager = SpsManager()
        with open(self.out_bin_path, "rb") as input_file:
            self.input_buff = io.BytesIO(input_file.read())

    def decompress_frame_by_read(self):
        res = self.read_frame_by_syntax()
        res = self.decompress_frame(res["bit_stream"], res["pstate"])
        return res

    def decompress_frame(self, bit_stream, pstate):
        torch.cuda.synchronize(device=self.device)

        if pstate["frame_type"] == 0:
            decoded: dict = self.i_frame_net.decompress(bit_stream, pstate)
            self.p_frame_net.clear_dpb()
            self.p_frame_net.add_ref_frame(None, decoded["x_hat"])
        elif pstate["frame_type"] == 1:
            if pstate["use_ada_i"] == 1:
                self.p_frame_net.reset_ref_feature()
            decoded = self.p_frame_net.decompress(bit_stream, pstate)

        recon_frame = decoded["x_hat"]
        x_hat = recon_frame[:, :, : self.src_height, : self.src_width]

        torch.cuda.synchronize(device=self.device)
        return {
            "x_hat": x_hat,
        }

    def write_frame_by_syntax(self, bit_stream, pstate):
        sps = {
            "sps_id": -1,
            "height": pstate["height"],
            "width": pstate["width"],
            "ec_part": pstate["ec_part"],
            "use_ada_i": pstate["use_ada_i"],
        }
        sps, is_new_sps = self.sps_manager.resue_or_insert(sps)
        sps_bytes = 0
        if is_new_sps:
            sps_bytes = write_sps(self.output_buff, sps)
        pps = {
            "nal_type": NalType.NAL_I if pstate["frame_type"] == 0 else NalType.NAL_P,
            "sps_id": sps["sps_id"],
            "qp": pstate["qp"],
        }
        stream_bytes = write_picture_pps(self.output_buff, pps, bit_stream)
        frame_bits = stream_bytes * 8 + sps_bytes * 8
        return {
            "bits": frame_bits,
        }

    def read_frame_by_syntax(self):
        header = read_vps(self.input_buff)
        while header["nal_type"] == NalType.NAL_SPS:
            sps = read_sps_remaining(self.input_buff, header["sps_id"])
            self.sps_manager.update_or_insert(sps)
            header = read_vps(self.input_buff)
            continue
        sps_id = header["sps_id"]

        sps = self.sps_manager.find_sps_by_id(sps_id)
        qp, bit_stream = read_picture_remaining(self.input_buff)

        pstate = sps.copy()
        pstate["frame_type"] = 0 if header["nal_type"] == NalType.NAL_I else 1
        pstate["qp"] = qp
        pstate["x_format"] = "ycbcr"

        return {
            "pstate": pstate,
            "bit_stream": bit_stream,
        }

    def close_write(self):
        with open(self.out_bin_path, "wb") as output_file:
            bytes_buffer = self.output_buff.getbuffer()
            output_file.write(bytes_buffer)
            total_bytes = bytes_buffer.nbytes
            bytes_buffer.release()
        self.output_buff.close()
        return total_bytes

    def close_read(self):
        self.input_buff.close()
        return
