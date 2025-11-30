# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from pathlib import Path
from mpcompress.utils.transforms import ycbcr420_to_444_np


class Frame:
    """Frame 数据结构，包含 y, u, v, yuv444, rgb 属性"""

    def __init__(self, y=None, u=None, v=None, yuv444=None, rgb=None):
        self.y = y
        self.u = u
        self.v = v
        self.yuv444 = yuv444
        self.rgb = rgb


class SingleImageVideoReader:
    # just same as Image Folder
    def __init__(self, src_path, **kwargs):
        self.src_path = src_path

    def read_one_frame(self):
        # rgb: 3xhxw uint8 numpy array
        rgb = Image.open(self.src_path).convert("RGB")
        rgb = np.asarray(rgb).astype(np.uint8).transpose(2, 0, 1)
        return Frame(rgb=rgb)

    def close(self):
        pass


class PngSequenceVideoReader:
    def __init__(self, src_path, width, height, start_num=1):
        self.eof = False
        self.src_path = src_path
        self.width = width
        self.height = height
        pngs = os.listdir(self.src_path)
        if "im1.png" in pngs:
            self.padding = 1
        elif "im00001.png" in pngs:
            self.padding = 5
        else:
            raise ValueError("unknown image naming convention; please specify")
        self.start_num = start_num
        self.current_frame_index = start_num

    def restart(self):
        self.current_frame_index = self.start_num
        self.eof = False

    def read_one_frame(self):
        # rgb: 3xhxw uint8 numpy array
        if self.eof:
            return None

        png_path = os.path.join(
            self.src_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png"
        )
        if not os.path.exists(png_path):
            self.eof = True
            return None

        rgb = Image.open(png_path).convert("RGB")
        rgb = np.asarray(rgb).astype(np.uint8).transpose(2, 0, 1)
        _, height, width = rgb.shape
        assert height == self.height
        assert width == self.width

        self.current_frame_index += 1
        return Frame(rgb=rgb)

    def close(self):
        self.current_frame_index = 1


class YUV420VideoReader:
    def __init__(self, src_path, width, height, skip_frame=0):
        self.eof = False
        if not src_path.endswith(".yuv"):
            src_path = src_path + ".yuv"
        self.src_path = src_path

        self.y_size = width * height
        self.y_width = width
        self.y_height = height
        self.uv_size = width * height // 2
        self.uv_width = width // 2
        self.uv_height = height // 2
        # pylint: disable=R1732
        self.file = open(src_path, "rb")
        # pylint: enable=R1732
        skipped_frame = 0
        while not self.eof and skipped_frame < skip_frame:
            y = self.file.read(self.y_size)
            uv = self.file.read(self.uv_size)
            if not y or not uv:
                self.eof = True
            skipped_frame += 1
        
    def restart(self):
        if self.file.closed:
            self.file = open(self.src_path, "rb")
        self.file.seek(0)
        self.eof = False

    def read_one_frame(self):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        if self.eof:
            return None
        y = self.file.read(self.y_size)
        uv = self.file.read(self.uv_size)
        if not y or not uv:
            self.eof = True
            return None
        y = (
            np.frombuffer(y, dtype=np.uint8)
            .copy()
            .reshape(1, self.y_height, self.y_width)
        )
        uv = (
            np.frombuffer(uv, dtype=np.uint8)
            .copy()
            .reshape(2, self.uv_height, self.uv_width)
        )

        # 分离 u 和 v
        u = uv[0, :, :]
        v = uv[1, :, :]

        # 转换为 yuv444
        yuv444 = ycbcr420_to_444_np(y, uv)

        return Frame(y=y[0, :, :], u=u, v=v, yuv444=yuv444)

    def close(self):
        self.file.close()
