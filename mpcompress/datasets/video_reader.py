# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import os

import numpy as np
from PIL import Image
from mpcompress.utils.transforms import ycbcr420_to_444_np


class Frame:
    """Frame data structure for video frames.

    This class stores frame data in various color space formats including
    YUV420, YUV444, and RGB. All attributes are optional and can be set
    independently.

    Attributes:
        y (numpy.ndarray, optional): Luminance channel of shape (H, W) in uint8.
        u (numpy.ndarray, optional): U chrominance channel of shape (H//2, W//2)
            for YUV420 or (H, W) for YUV444, in uint8.
        v (numpy.ndarray, optional): V chrominance channel of shape (H//2, W//2)
            for YUV420 or (H, W) for YUV444, in uint8.
        yuv444 (numpy.ndarray, optional): YUV444 format data of shape (3, H, W)
            in uint8.
        rgb (numpy.ndarray, optional): RGB format data of shape (3, H, W) in uint8.
    """

    def __init__(self, y=None, u=None, v=None, yuv444=None, rgb=None):
        """Initialize a Frame object.

        Args:
            y (numpy.ndarray, optional): Luminance channel. Defaults to None.
            u (numpy.ndarray, optional): U chrominance channel. Defaults to None.
            v (numpy.ndarray, optional): V chrominance channel. Defaults to None.
            yuv444 (numpy.ndarray, optional): YUV444 format data. Defaults to None.
            rgb (numpy.ndarray, optional): RGB format data. Defaults to None.
        """
        self.y = y
        self.u = u
        self.v = v
        self.yuv444 = yuv444
        self.rgb = rgb


class SingleImageVideoReader:
    """Video reader for single image files.

    This reader treats a single image file as a one-frame video sequence.
    It is functionally equivalent to an Image Folder reader for single images.

    Args:
        src_path (str): Path to the image file.
        **kwargs: Additional keyword arguments (unused).
    """

    def __init__(self, src_path, **kwargs):
        """Initialize single image video reader.

        Args:
            src_path (str): Path to the image file.
            **kwargs (dict): Additional keyword arguments (unused).
        """
        self.src_path = src_path

    def read_one_frame(self):
        """Read the single frame from the image file.

        Returns:
            frame (Frame): Frame object containing RGB data of shape (3, H, W)
                as uint8 numpy array.
        """
        # RGB: 3xHxW uint8 numpy array
        rgb = Image.open(self.src_path).convert("RGB")
        rgb = np.asarray(rgb).astype(np.uint8).transpose(2, 0, 1)
        return Frame(rgb=rgb)

    def close(self):
        """Close the reader (no-op for single image)."""
        pass


class PngSequenceVideoReader:
    """Video reader for PNG image sequences.

    This reader loads frames from a sequence of PNG images with naming
    conventions like "im1.png", "im2.png" or "im00001.png", "im00002.png".
    The padding width is automatically detected from the first image found.

    Args:
        src_path (str): Path to the directory containing PNG images.
        width (int): Expected width of each frame in pixels.
        height (int): Expected height of each frame in pixels.
        start_num (int): Starting frame number. Defaults to 1.
    """

    def __init__(self, src_path, width, height, start_num=1):
        """Initialize PNG sequence video reader.

        Args:
            src_path (str): Path to the directory containing PNG images.
            width (int): Expected width of each frame in pixels.
            height (int): Expected height of each frame in pixels.
            start_num (int): Starting frame number. Defaults to 1.

        Raises:
            ValueError: If the image naming convention cannot be determined.
        """
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
        """Reset the reader to the starting frame."""
        self.current_frame_index = self.start_num
        self.eof = False

    def read_one_frame(self):
        """Read the next frame from the PNG sequence.

        Returns:
            frame (Frame or None): Frame object containing RGB data of shape
                (3, H, W) as uint8 numpy array. Returns None if end of file
                is reached or frame cannot be read.
        """
        # RGB: 3xHxW uint8 numpy array
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
        """Close the reader and reset frame index."""
        self.current_frame_index = 1


class YUV420VideoReader:
    """Video reader for YUV420 format video files.

    This reader loads frames from raw YUV420 format video files. YUV420 is a
    chroma-subsampled format where the Y (luminance) channel is full resolution
    and the U and V (chrominance) channels are subsampled by a factor of 2
    in both dimensions.

    Args:
        src_path (str): Path to the YUV file (with or without .yuv extension).
        width (int): Width of each frame in pixels.
        height (int): Height of each frame in pixels.
        skip_frame (int): Number of frames to skip at the beginning. Defaults to 0.
    """

    def __init__(self, src_path, width, height, skip_frame=0):
        """Initialize YUV420 video reader.

        Args:
            src_path (str): Path to the YUV file (with or without .yuv extension).
            width (int): Width of each frame in pixels.
            height (int): Height of each frame in pixels.
            skip_frame (int): Number of frames to skip at the beginning.
                Defaults to 0.
        """
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
        """Reset the reader to the beginning of the file."""
        if self.file.closed:
            self.file = open(self.src_path, "rb")
        self.file.seek(0)
        self.eof = False

    def read_one_frame(self):
        """Read the next frame from the YUV420 file.

        Returns:
            frame (Frame or None): Frame object containing Y, U, V channels and
                YUV444 converted data. Returns None if end of file is reached
                or frame cannot be read.

                Frame attributes:

                    - y: Luminance channel of shape (H, W) as uint8 numpy array.
                    - u: U chrominance channel of shape (H//2, W//2) as uint8 numpy array.
                    - v: V chrominance channel of shape (H//2, W//2) as uint8 numpy array.
                    - yuv444: YUV444 format data of shape (3, H, W) as uint8 numpy array.
        """
        # Y: 1xHxW uint8 numpy array
        # UV: 2x(H/2)x(W/2) uint8 numpy array
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

        # Separate U and V channels
        u = uv[0, :, :]
        v = uv[1, :, :]

        # Convert to YUV444
        yuv444 = ycbcr420_to_444_np(y, uv)

        return Frame(y=y[0, :, :], u=u, v=v, yuv444=yuv444)

    def close(self):
        """Close the file handle."""
        self.file.close()
