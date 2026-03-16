from pathlib import Path
import time
import subprocess
import torch
import math


class HMCodecFFmpeg:
    def __init__(self, ffmpeg_path='ffmpeg', verbose=True):
        """
        初始化 FFmpeg 处理器
        :param ffmpeg_path: ffmpeg 可执行文件的路径
        :param verbose: 控制是否打印 Python 层面的进度信息（例如"开始处理..."），
                        不影响 FFmpeg 内部日志的静默设置。
        """
        self.ffmpeg_path = ffmpeg_path
        self.verbose = verbose

    def _run_command(self, command):
        """
        内部方法：执行 FFmpeg 命令并统计时间
        关键修改：加入了 quiet 参数并捕获输出，除非报错否则不打印 ffmpeg 日志
        """
        # --- 修改点 1: 注入静默参数 ---
        # 1. -hide_banner: 隐藏版权和版本头部信息
        # 2. -loglevel error: 只有发生错误时才输出日志，屏蔽进度条和常规信息
        # 我们把这些参数插在 ffmpeg 路径之后，其他参数之前
        cmd_with_quiet = [command[0], '-hide_banner', '-loglevel', 'error'] + command[1:]

        if self.verbose:
            print(f"执行: {' '.join(command)}") # 仅打印简洁的命令，不打印 ffmpeg 输出

        start_time = time.time()
        
        try:
            # --- 修改点 2: 捕获输出 ---
            # capture_output=True 会将 stdout 和 stderr 存入 result 对象，而不是直接打印到屏幕
            result = subprocess.run(
                cmd_with_quiet, 
                check=True, 
                capture_output=True, 
                text=True
            )
            
            elapsed_time = time.time() - start_time
            return True, elapsed_time
            
        except subprocess.CalledProcessError as e:
            # --- 修改点 3: 仅在出错时打印日志 ---
            print(f"\n[错误] FFmpeg 执行失败，文件可能已损坏或参数错误。")
            print(f"命令: {' '.join(command)}")
            print(f"错误详情:\n{e.stderr}") # 打印捕获到的错误日志
            return False, 0
            
        except Exception as e:
            print(f"\n[未知错误]: {e}")
            return False, 0

    def compress(self, input_video_path, output_bitstream_path, bitrate='2488k', preset='veryfast', extra_params=None):
        """
        压缩方法
        """
        Path(output_bitstream_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            '-i', input_video_path,
            '-c:v', 'libx265',
            '-b:v', bitrate,
            '-preset', preset,
            '-vsync', '0',
            '-avoid_negative_ts', 'make_zero',
            '-y',
            output_bitstream_path
        ]

        if extra_params:
            cmd[8:8] = extra_params # 插入到 output 之前

        success, elapsed = self._run_command(cmd)
        
        if success and self.verbose:
            print(f" -> 压缩完成: {output_bitstream_path} ({elapsed:.2f}s)")
        
        return output_bitstream_path

    def decompress(self, input_bitstream_path, output_video_path):
        """
        解压方法
        """
        Path(output_video_path).parent.mkdir(parents=True, exist_ok=True)

        cmd = [
            self.ffmpeg_path,
            '-i', input_bitstream_path,
            '-c:v', 'copy', # 仅封装，若需完全解码请移除此行或改为 rawvideo
            '-y',
            output_video_path
        ]

        success, elapsed = self._run_command(cmd)
        
        if success and self.verbose:
            print(f" -> 解压完成: {output_video_path} ({elapsed:.2f}s)")
            
        return success