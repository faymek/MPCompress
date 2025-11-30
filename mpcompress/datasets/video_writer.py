# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

from PIL import Image
import torch
from mpcompress.utils.transforms import (
    rgb2ycbcr,
    ycbcr2rgb,
    yuv_444_to_420,
    ycbcr420_to_444_np,
)


class PngSequenceVideoWriter:
    def __init__(self, dst_path, width, height):
        self.dst_path = dst_path
        self.width = width
        self.height = height
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb):
        # rgb: 3xhxw uint8 numpy array
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(
            self.dst_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png"
        )
        Image.fromarray(rgb).save(png_path)

        self.current_frame_index += 1

    def write_one_frame_from_tensor(self, x, format="yuv444"):
        if format == "yuv444":
            rgb = ycbcr2rgb(x)
            rgb = torch.clamp(rgb * 255, 0, 255).round().to(dtype=torch.uint8)
            rgb = rgb.squeeze(0).cpu().numpy()
            self.write_one_frame(rgb)
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

    def close(self):
        self.current_frame_index = 1


class YUV420VideoWriter:
    def __init__(self, dst_path, width, height):
        if not dst_path.endswith(".yuv"):
            dst_path = dst_path + "/out.yuv"
        self.dst_path = dst_path
        self.width = width
        self.height = height

        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, y, uv):
        # y: 1xhxw uint8 numpy array
        # uv: 2x(h/2)x(w/2) uint8 numpy array
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def write_one_frame_from_tensor(self, x, format="yuv444"):
        if format == "yuv444":
            y, uv = yuv_444_to_420(x)
            y = torch.clamp(y * 255, 0, 255).round().to(dtype=torch.uint8)
            y = y.squeeze(0).cpu().numpy()
            uv = torch.clamp(uv * 255, 0, 255).to(dtype=torch.uint8)
            uv = uv.squeeze(0).cpu().numpy()
            self.write_one_frame(y, uv)
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

    def close(self):
        self.file.close()
