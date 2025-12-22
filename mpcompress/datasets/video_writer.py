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
    """Video writer for PNG image sequences.

    This writer saves frames as a sequence of PNG images with naming
    convention "im00001.png", "im00002.png", etc. The output directory
    is created automatically if it doesn't exist.

    Args:
        dst_path (str): Path to the output directory for PNG images.
        width (int): Width of each frame in pixels.
        height (int): Height of each frame in pixels.
    """

    def __init__(self, dst_path, width, height):
        """Initialize PNG sequence video writer.

        Args:
            dst_path (str): Path to the output directory for PNG images.
            width (int): Width of each frame in pixels.
            height (int): Height of each frame in pixels.
        """
        self.dst_path = dst_path
        self.width = width
        self.height = height
        self.padding = 5
        self.current_frame_index = 1
        os.makedirs(dst_path, exist_ok=True)

    def write_one_frame(self, rgb):
        """Write a single frame as a PNG image.

        Args:
            rgb (numpy.ndarray): RGB image data of shape (3, H, W) as uint8
                numpy array.
        """
        # RGB: 3xHxW uint8 numpy array
        rgb = rgb.transpose(1, 2, 0)

        png_path = os.path.join(
            self.dst_path, f"im{str(self.current_frame_index).zfill(self.padding)}.png"
        )
        Image.fromarray(rgb).save(png_path)

        self.current_frame_index += 1

    def write_one_frame_from_tensor(self, x, format="yuv444"):
        """Write a single frame from a PyTorch tensor.

        Converts the tensor from the specified format to RGB and writes it
        as a PNG image.

        Args:
            x (torch.Tensor): Input tensor in the specified format.
            format (str): Input format. Currently only "yuv444" is supported.
                Defaults to "yuv444".

        Raises:
            NotImplementedError: If the specified format is not supported.
        """
        if format == "yuv444":
            rgb = ycbcr2rgb(x)
            rgb = torch.clamp(rgb * 255, 0, 255).round().to(dtype=torch.uint8)
            rgb = rgb.squeeze(0).cpu().numpy()
            self.write_one_frame(rgb)
        else:
            raise NotImplementedError(f"Unsupported format: {format}")

    def close(self):
        """Close the writer and reset frame index."""
        self.current_frame_index = 1


class YUV420VideoWriter:
    """Video writer for YUV420 format video files.

    This writer saves frames to a raw YUV420 format video file. YUV420 is a
    chroma-subsampled format where the Y (luminance) channel is full resolution
    and the U and V (chrominance) channels are subsampled by a factor of 2
    in both dimensions.

    Args:
        dst_path (str): Path to the output YUV file (with or without .yuv extension).
            If a directory path is provided, it will create "out.yuv" in that directory.
        width (int): Width of each frame in pixels.
        height (int): Height of each frame in pixels.
    """

    def __init__(self, dst_path, width, height):
        """Initialize YUV420 video writer.

        Args:
            dst_path (str): Path to the output YUV file (with or without .yuv extension).
                If a directory path is provided, it will create "out.yuv" in that directory.
            width (int): Width of each frame in pixels.
            height (int): Height of each frame in pixels.
        """
        if not dst_path.endswith(".yuv"):
            dst_path = dst_path + "/out.yuv"
        self.dst_path = dst_path
        self.width = width
        self.height = height

        # pylint: disable=R1732
        self.file = open(dst_path, "wb")
        # pylint: enable=R1732

    def write_one_frame(self, y, uv):
        """Write a single frame in YUV420 format.

        Args:
            y (numpy.ndarray): Luminance channel of shape (H, W) as uint8 numpy array.
            uv (numpy.ndarray): Chrominance channels of shape (2, H//2, W//2) as uint8
                numpy array, where uv[0] is U and uv[1] is V.
        """
        # Y: HxW uint8 numpy array
        # UV: 2x(H/2)x(W/2) uint8 numpy array
        self.file.write(y.tobytes())
        self.file.write(uv.tobytes())

    def write_one_frame_from_tensor(self, x, format="yuv444"):
        """Write a single frame from a PyTorch tensor.

        Converts the tensor from the specified format to YUV420 and writes it
        to the file.

        Args:
            x (torch.Tensor): Input tensor in the specified format.
            format (str): Input format. Currently only "yuv444" is supported.
                Defaults to "yuv444".

        Raises:
            NotImplementedError: If the specified format is not supported.
        """
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
        """Close the file handle."""
        self.file.close()
