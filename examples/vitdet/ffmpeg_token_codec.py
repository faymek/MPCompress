import os
import numpy as np
import subprocess
import torch
from einops import rearrange


class FFmpegVitTokenCodec:
    def __init__(
        self,
        ffmpeg_path,
        work_dir,
        codec="libx264",
        qp=None,
        preset="placebo",
        maxQuantCoeff=255,
    ):
        self.ffmpeg_path = ffmpeg_path
        self.work_dir = work_dir
        self.codec = codec
        self.qp = qp
        self.preset = preset
        self.maxQuantCoeff = maxQuantCoeff
        self.counter = 0  # 新增帧编号

        yuv_dir = os.path.join(work_dir, "yuv")
        rec_dir = os.path.join(work_dir, "bin")
        size_log_path = os.path.join(work_dir, "size_stats.txt")
        error_log_path = os.path.join(work_dir, "mse_log.txt")

        self.yuv_dir = yuv_dir
        self.rec_dir = rec_dir
        self.size_log_path = size_log_path
        self.error_log_path = error_log_path

        cmd = (
            f"rm -rf {yuv_dir}/*\n"
            + f"rm -rf {rec_dir}/*\n"
            + f"rm -rf {size_log_path}\n"
            + f"rm -rf {error_log_path}"
        )
        os.system(cmd)

        os.makedirs(yuv_dir, exist_ok=True)
        os.makedirs(rec_dir, exist_ok=True)

    def __call__(self, token_tensor, frame_id=None):
        """
        token_tensor: [B, C, H, W]
        返回: 同尺寸的 [B, C, H, W]，每个样本独立压缩+重建
        """
        B = token_tensor.shape[0]
        output_list = []
        total_bytes = 0
        assert B == 1, "Batch size must be 1"

        for b in range(B):
            if frame_id is None:
                fid = self.counter
                self.counter += 1
            else:
                fid = frame_id + b  # 若外部给定 frame_id，则连续编号

            feat = token_tensor[b : b + 1].detach().cpu().numpy()  # shape: [1, C, H, W]
            _, yuv_path, meta = self.ndarray_to_yuv(fid, feat)  # quant, pack
            num_bytes = self.compress_yuv(yuv_path, fid, meta)
            rec_feat = self.yuv_to_ndarray(fid, meta)  # dequant, unpack

            # # 误差计算
            # mse = np.mean((feat - rec_feat) ** 2)
            # # 写入误差日志
            # with open(self.error_log_path, "a") as f:
            #     f.write(f"{fid:06d},{mse:.6f}\n")
            output_list.append(torch.from_numpy(rec_feat))
            total_bytes += num_bytes

        rec_tensor = torch.cat(output_list, dim=0).to(token_tensor.device)
        return {
            "x_hat": rec_tensor,
            "num_bytes": total_bytes,
        }

    def ndarray_to_yuv(self, frame_id, feat):  # global minmax
        min_val, max_val = feat.min(), feat.max()
        quant_feat = np.rint(
            (feat - min_val) / (max_val - min_val + 1e-5) * self.maxQuantCoeff
        ).astype(np.uint8)
        _, C, H, W = feat.shape
        # nh, nw = self.closest_divisors(C)
        nh, nw = 12, 64
        packed_feat = rearrange(
            quant_feat, "1 (nh nw) H W -> (nh H) (nw W)", nh=nh, nw=nw
        )

        yuv_path = os.path.join(self.yuv_dir, f"{frame_id:06d}.yuv")
        with open(yuv_path, "wb") as f:
            f.write(packed_feat.tobytes())
        meta = (max_val, min_val, nh, nw, H, W)
        return frame_id, yuv_path, meta

    def compress_yuv(self, yuv_path, frame_id, meta):
        maxVal, minVal, nh, nw, H, W = meta
        w, h = nw * W, nh * H
        ext = ".h264" if self.codec == "libx264" else ".h265"
        bin_path = os.path.join(self.rec_dir, f"{frame_id:06d}{ext}")
        rec_yuv = os.path.join(self.rec_dir, f"{frame_id:06d}.yuv")

        # 以 shell 字符串形式构造命令，便于阅读和调试
        qp_part = f"-qp {self.qp} " if self.qp is not None else ""
        cmd = (
            f"{self.ffmpeg_path} -s {w}x{h} -pix_fmt gray -i {yuv_path} "
            f"-c:v {self.codec} {qp_part} -preset {self.preset} "
            f"-loglevel quiet -y {bin_path}"
        )
        subprocess.call(cmd, shell=True)

        # 解码部分同样使用紧凑的 shell 字符串形式
        cmd_dec = (
            f"{self.ffmpeg_path} -i {bin_path} -pix_fmt gray -loglevel quiet {rec_yuv}"
        )
        subprocess.call(cmd_dec, shell=True)
        # with open(self.size_log_path, "a") as f:
        #     original_size = os.path.getsize(yuv_path)
        #     compressed_size = os.path.getsize(mp4_path)
        #     f.write(f"{frame_id:06d},{original_size},{compressed_size}\n")
        num_bytes = os.path.getsize(bin_path)
        return num_bytes

    def yuv_to_ndarray(self, frame_id, meta):  # global minmax
        maxVal, minVal, nh, nw, H, W = meta
        rec_yuv = os.path.join(self.rec_dir, f"{frame_id:06d}.yuv")
        with open(rec_yuv, "rb") as f:
            raw = f.read()
        # 读取 YUV 数据并 reshape 为平铺矩阵
        packed_feat = np.frombuffer(raw, dtype=np.uint8).reshape(nh * H, nw * W)
        # 使用 unpack 方法将平铺矩阵解包回特征图
        quant_feat = rearrange(
            packed_feat, "(nh H) (nw W) -> 1 (nh nw) H W", nh=nh, nw=nw, H=H, W=W
        )
        # 反量化：将 uint8 转换回 float32
        feat = (
            quant_feat.astype(np.float32) / self.maxQuantCoeff * (maxVal - minVal)
            + minVal
        )
        return feat.astype(np.float32)
