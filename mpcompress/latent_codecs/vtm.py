import os
import sys
import json
import time
import subprocess
import numpy as np
from omegaconf import OmegaConf
from tempfile import mkstemp


def truncation(feat, trun_low, trun_high):
    trun_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(trun_low, list):
        for idx in range(len(trun_low)):
            trun_feat[:, idx, :, :] = np.clip(
                feat[:, idx, :, :], trun_low[idx], trun_high[idx]
            )
    else:
        trun_feat = np.clip(feat, trun_low, trun_high)

    return trun_feat


def load_quantization_points(file_path: str or list[str]):
    """
    Load quantization points from a file or a list of files.

    Parameters:
        file_path (Union[str, List[str]]): Path to load the quantization points from.
            Can be a single file path (str) or a list of file paths (List[str]).

    Returns:
        Union[numpy.ndarray, List[numpy.ndarray]]: Loaded quantization points. If `file_path`
            is a single path, returns a single numpy.ndarray. If `file_path` is a list of paths,
            returns a list of numpy.ndarray.
    """

    def load_file(path):
        with open(path, "r") as f:
            quantization_points = np.array(json.load(f))
        # print(f"Quantization points loaded from {path}")
        return quantization_points

    if isinstance(file_path, list):
        # Load quantization points from each file in the list
        return [load_file(path) for path in file_path]
    elif isinstance(file_path, str):
        # Load quantization points from a single file
        return load_file(file_path)
    else:
        raise ValueError("file_path must be a string or a list of strings.")


def uniform_quantization(feat, min_v, max_v, bit_depth):
    quant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) - 1) / (max_v[idx] - min_v[idx])
            quant_feat[:, idx, :, :] = (feat[:, idx, :, :] - min_v[idx]) * scale
    else:
        scale = ((2**bit_depth) - 1) / (max_v - min_v)
        quant_feat = (feat - min_v) * scale

    quant_feat = (
        quant_feat.astype(np.uint16) if bit_depth > 8 else quant_feat.astype(np.uint8)
    )
    return quant_feat


def uniform_dequantization(feat, min_v, max_v, bit_depth):
    feat = feat.astype(np.float32)
    dequant_feat = np.zeros_like(feat).astype(np.float32)
    if isinstance(min_v, list):
        for idx in range(len(min_v)):
            scale = ((2**bit_depth) - 1) / (max_v[idx] - min_v[idx])
            dequant_feat[:, idx, :, :] = feat[:, idx, :, :] / scale + min_v[idx]
    else:
        scale = ((2**bit_depth) - 1) / (max_v - min_v)
        dequant_feat = feat / scale + min_v
    return dequant_feat


def nonlinear_quantization(data, quantization_points, bit_depth):
    """
    Apply quantization to data using a single or multiple sets of quantization points.

    Parameters:
        data (numpy.ndarray): Original floating-point array with shape (N, C, H, W).
        quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]):
            A single numpy array of quantization points or a list of numpy arrays,
            one for each channel (C).

    Returns:
        numpy.ndarray: Quantized integer array with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        num_levels = len(quantization_points)
        data_flat = data.flatten()
        quantized_data_flat = np.digitize(data_flat, quantization_points) - 1
        quantized_data_flat = np.clip(quantized_data_flat, 0, num_levels - 1)
        quantized_data = quantized_data_flat.reshape(data.shape)
    elif isinstance(quantization_points, list):
        if len(quantization_points) != data.shape[1]:
            raise ValueError(
                "Length of quantization_points list must match the number of channels (C) in data."
            )

        quantized_data = np.zeros_like(data, dtype=int)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            num_levels = len(qp)
            channel_data = data[:, i, :, :]
            channel_data_flat = channel_data.flatten()
            quantized_channel_flat = np.digitize(channel_data_flat, qp) - 1
            quantized_channel_flat = np.clip(quantized_channel_flat, 0, num_levels - 1)
            quantized_data[:, i, :, :] = quantized_channel_flat.reshape(
                channel_data.shape
            )
    else:
        raise ValueError(
            "quantization_points must be a numpy array or a list of numpy arrays."
        )

    quantized_data = (
        quantized_data.astype(np.uint16)
        if bit_depth > 8
        else quantized_data.astype(np.uint8)
    )
    return quantized_data


def nonlinear_dequantization(quantized_data, quantization_points):
    """
    Dequantize quantized data back to its approximate original floating-point values.

    Parameters:
        quantized_data (numpy.ndarray): Quantized integer array with shape (N, C, H, W).
        quantization_points (Union[numpy.ndarray, List[numpy.ndarray]]):
            A single numpy array of quantization points or a list of numpy arrays,
            one for each channel (C).

    Returns:
        numpy.ndarray: Dequantized floating-point array with the same shape as the input data.
    """
    if isinstance(quantization_points, np.ndarray):
        # If quantization_points is a single array, apply it to all channels
        quantization_points = np.sort(quantization_points)  # Ensure points are sorted
        dequantized_data = quantization_points[quantized_data]
    elif isinstance(quantization_points, list):
        if len(quantization_points) != quantized_data.shape[1]:
            raise ValueError(
                "Length of quantization_points list must match the number of channels (C) in quantized_data."
            )

        dequantized_data = np.zeros_like(quantized_data, dtype=np.float32)
        # Apply different quantization points to each channel
        for i, qp in enumerate(quantization_points):
            qp = np.sort(qp)  # Ensure points are sorted
            channel_data = quantized_data[:, i, :, :]
            dequantized_data[:, i, :, :] = qp[channel_data]
    else:
        raise ValueError(
            "quantization_points must be a numpy array or a list of numpy arrays."
        )

    # print(dequantized_data.dtype)
    dequantized_data = dequantized_data.astype(np.float32)
    return dequantized_data


def packing(feat, model_type):
    N, C, H, W = feat.shape
    if model_type == "llama3":
        feat = feat[0, 0, :, :]
    elif model_type == "dinov2":
        feat = feat.transpose(0, 2, 1, 3).reshape(N * H, C * W)
    elif model_type == "sd3":
        feat = (
            feat.reshape(int(C / 4), int(C / 4), H, W)
            .transpose(0, 2, 1, 3)
            .reshape(int(C / 4 * H), int(C / 4 * W))
        )
    return feat


def unpacking(feat, shape, model_type):
    N, C, H, W = shape
    if model_type == "llama3":
        feat = np.expand_dims(feat, axis=0)
        feat = np.expand_dims(feat, axis=0)
    elif model_type == "dinov2":
        feat = feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)
    elif model_type == "sd3":
        feat = (
            feat.reshape(int(C / 4), H, int(C / 4), W)
            .transpose(0, 2, 1, 3)
            .reshape(N, C, H, W)
        )
    return feat


def run_shell(cmd, ignore_returncodes=None):
    if isinstance(cmd, list):
        cmd = " ".join(cmd)
    try:
        rv = subprocess.check_output(cmd, shell=True)
        return rv.decode("ascii")
    except subprocess.CalledProcessError as err:
        if ignore_returncodes is not None and err.returncode in ignore_returncodes:
            return err.output
        print(err.output.decode("utf-8"))
        sys.exit(1)


def filesize(filepath: str) -> int:
    """Return file size in bytes of `filepath`."""
    if not os.path.isfile(filepath):
        raise ValueError(f'Invalid file "{filepath}".')
    return os.stat(filepath).st_size


class VtmCodec:
    def __init__(self, repo_dir):
        self.encoder_path = os.path.join(repo_dir, "bin", "EncoderAppStatic")
        self.decoder_path = os.path.join(repo_dir, "bin", "DecoderAppStatic")
        self.config_path = os.path.join(repo_dir, "cfg", "encoder_intra_vtm.cfg")

        self.version = run_shell(f"{self.encoder_path} |grep Version").split()[4]
        self.description = f"VTM-{self.version}"

    def compress(
        self,
        raw_path,
        bin_path,
        width,
        height,
        quality: int,
        bitdepth: int = 8,
        chroma_format: str = "400",
    ):
        cmd = (
            f"{self.encoder_path} -c {self.config_path} "
            f'-i {raw_path} -o "" -b {bin_path} -q {quality} --ConformanceWindowMode=1 '
            f"-wdt {width} -hgt {height} -f 1 -fr 1 "
            f"--InternalBitDepth={bitdepth} --InputBitDepth={bitdepth} "
            f"--InputChromaFormat={chroma_format} --OutputBitDepth={bitdepth} "
        )
        print(cmd)
        run_shell(cmd)

    def decompress(self, bin_path, rec_path, bit_depth=8):
        cmd = f"{self.decoder_path} -b {bin_path} -o {rec_path} -d {bit_depth}"
        print(cmd)
        run_shell(cmd)


class VtmFeatureCodec:
    def __init__(self, cfg):
        self.cfg = cfg
        self.codec = VtmCodec(cfg.vtm_path)

    def forward_test( # just for debug
        self,
        org_feat,
        quality: int,
    ):
        cfg = self.cfg
        org_feat_shape = org_feat.shape

        # Truncation
        if cfg.trun_flag is True:
            feat = truncation(org_feat, cfg.trun_low, cfg.trun_high)
        # Quantization
        feat = uniform_quantization(feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)
        # Packing
        pack_feat = packing(feat, cfg.model_type)
        fd, raw_path = mkstemp(suffix=".yuv")
        bin_path = os.path.splitext(raw_path)[0] + ".bin"
        rec_path = os.path.splitext(raw_path)[0] + "_rec.yuv"
        with open(raw_path, "wb") as f:
            pack_feat.tofile(f)

        # VTM encoding
        start = time.time()
        self.codec.compress(
            raw_path,
            bin_path,
            pack_feat.shape[1],
            pack_feat.shape[0],
            quality,
            cfg.bit_depth,
            "400",
        )
        enc_time = time.time() - start

        # VTM decoding
        start = time.time()
        self.codec.decompress(bin_path, rec_path, cfg.bit_depth)
        dec_time = time.time() - start

        # Load decoded YUV
        with open(rec_path, "rb") as f:
            decoded_yuv = np.fromfile(
                f, dtype=np.uint16 if cfg.bit_depth == 10 else np.uint8
            )
            decoded_yuv = decoded_yuv.reshape(pack_feat.shape)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, org_feat_shape, cfg.model_type)

        # Dequantization
        dequant_feat = uniform_dequantization(
            unpack_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth
        )

        # Save features
        if cfg.model_type == "sd3":
            dequant_feat = dequant_feat.astype(np.float16)

        out = {
            "bits": {"vtm": filesize(bin_path) * 8.0},
            "encoding_time": enc_time,
            "decoding_time": dec_time,
            "h_hat": dequant_feat,
        }

        # cleanup encoder input
        os.close(fd)
        os.unlink(raw_path)
        os.unlink(bin_path)
        os.unlink(rec_path)

        return out

    def compress(
        self,
        org_feat,
        quality: int,
    ):
        # expected feature: (N_crop, N_layer, H*W+1, C)
        cfg = self.cfg
        print(org_feat.shape)

        # Truncation
        if cfg.trun_flag is True:
            feat = truncation(org_feat, cfg.trun_low, cfg.trun_high)
        # Quantization
        feat = uniform_quantization(feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth)
        # Packing
        pack_feat = packing(feat, cfg.model_type)
        fd, raw_path = mkstemp(suffix=".yuv")
        bin_path = os.path.splitext(raw_path)[0] + ".bin"
        with open(raw_path, "wb") as f:
            pack_feat.tofile(f)

        # VTM encoding
        self.codec.compress(
            raw_path,
            bin_path,
            pack_feat.shape[1],
            pack_feat.shape[0],
            quality,
            cfg.bit_depth,
            "400",
        )
        os.close(fd)
        os.unlink(raw_path)
        return {
            "bits": {"vtm": filesize(bin_path) * 8.0},
            "bin_path": bin_path,
            "pack_shape": pack_feat.shape,
            "feat_shape": org_feat.shape,
            "bit_depth": cfg.bit_depth,
        }

    def decompress(self, bin_path, pack_shape, feat_shape, **kwargs):
        cfg = self.cfg
        rec_path = os.path.splitext(bin_path)[0] + "_rec.yuv"
        # VTM decoding
        self.codec.decompress(bin_path, rec_path, cfg.bit_depth)

        # Load decoded YUV
        with open(rec_path, "rb") as f:
            decoded_yuv = np.fromfile(
                f, dtype=np.uint16 if cfg.bit_depth == 10 else np.uint8
            )
            decoded_yuv = decoded_yuv.reshape(pack_shape)

        # Postprocessing
        unpack_feat = unpacking(decoded_yuv, feat_shape, cfg.model_type)

        # Dequantization
        dequant_feat = uniform_dequantization(
            unpack_feat, cfg.trun_low, cfg.trun_high, cfg.bit_depth
        )

        # Save features
        if cfg.model_type == "sd3":
            dequant_feat = dequant_feat.astype(np.float16)

        out = {
            "h_hat": dequant_feat,
        }
        os.unlink(bin_path)
        os.unlink(rec_path)

        return out
