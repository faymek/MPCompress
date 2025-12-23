"""
Color space conversion utilities.
Adapted from https://github.com/microsoft/DCVC/blob/main/src/utils/transforms.py
"""

import numpy as np
import scipy.ndimage
import torch
import torch.nn.functional as F
from torch import Tensor

# YCbCr color space conversion weights.
# Dictionary mapping color space standards to their RGB to YCbCr conversion weights.
# Each entry contains (K_r, K_g, K_b) where K_g = 1 - K_r - K_b.
YCBCR_WEIGHTS = {"ITU-R_BT.709": (0.2126, 0.7152, 0.0722)}


def ycbcr420_to_444_np(y, uv, order=0, separate=False):
    """
    Convert YCbCr 4:2:0 format to 4:4:4 format using numpy.

    Upsamples the chroma channels (UV) from half resolution to full resolution
    to match the luma channel (Y) resolution.

    Args:
        y (np.ndarray): Y channel float numpy array with shape (1, H, W).
        uv (np.ndarray): UV channels float numpy array with shape (2, H/2, W/2).
        order (int): Interpolation order for upsampling. 0 for nearest neighbor (default),
            1 for bilinear interpolation.
        separate (bool): If True, return Y and UV separately. If False, return
            concatenated YCbCr array. Defaults to False.

    Returns:
        out (np.ndarray or tuple(np.ndarray, np.ndarray)): If separate is False, return yuv with shape (3, H, W).
            If separate is True, return (y, uv) with shapes (1, H, W) and (2, H/2, W/2).
    """
    uv = scipy.ndimage.zoom(uv, (1, 2, 2), order=order)
    if separate:
        return y, uv
    yuv = np.concatenate((y, uv), axis=0)
    return yuv


def rgb2ycbcr(rgb: Tensor, is_bgr=False):
    """
    Convert RGB tensor to YCbCr color space.

    Converts RGB values to YCbCr using ITU-R BT.709 standard weights.
    The output values are clamped to [0, 1] range.

    Args:
        rgb (torch.Tensor): RGB tensor with shape (..., 3, H, W) or (..., C, H, W)
            where the last 3 channels are RGB.
        is_bgr (bool): If True, interpret input as BGR instead of RGB. Defaults to False.

    Returns:
        ycbcr (torch.Tensor): YCbCr tensor with shape (..., 3, H, W), values clamped to [0, 1].
    """
    if is_bgr:
        b, g, r = rgb.chunk(3, -3)
    else:
        r, g, b = rgb.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    y = Kr * r + Kg * g + Kb * b
    cb = 0.5 * (b - y) / (1 - Kb) + 0.5
    cr = 0.5 * (r - y) / (1 - Kr) + 0.5
    ycbcr = torch.cat((y, cb, cr), dim=-3)
    ycbcr = torch.clamp(ycbcr, 0.0, 1.0)
    return ycbcr


def ycbcr2rgb(ycbcr: Tensor, is_bgr=False, clamp=True):
    """
    Convert YCbCr tensor to RGB color space.

    Converts YCbCr values to RGB using ITU-R BT.709 standard weights.
    Optionally clamps output values to [0, 1] range.

    Args:
        ycbcr (torch.Tensor): YCbCr tensor with shape (..., 3, H, W) or (..., C, H, W)
            where the last 3 channels are YCbCr.
        is_bgr (bool): If True, output as BGR instead of RGB. Defaults to False.
        clamp (bool): If True, clamp output values to [0, 1] range. Defaults to True.

    Returns:
        rgb (torch.Tensor): RGB tensor with shape (..., 3, H, W), optionally clamped to [0, 1].
    """
    y, cb, cr = ycbcr.chunk(3, -3)
    Kr, Kg, Kb = YCBCR_WEIGHTS["ITU-R_BT.709"]
    r = y + (2 - 2 * Kr) * (cr - 0.5)
    b = y + (2 - 2 * Kb) * (cb - 0.5)
    g = (y - Kr * r - Kb * b) / Kg
    if is_bgr:
        rgb = torch.cat((b, g, r), dim=-3)
    else:
        rgb = torch.cat((r, g, b), dim=-3)
    if clamp:
        rgb = torch.clamp(rgb, 0.0, 1.0)
    return rgb


def yuv_444_to_420(yuv):
    """
    Convert YUV 4:4:4 format to 4:2:0 format.

    Downsamples the chroma channels (UV) from full resolution to half resolution
    using average pooling, while keeping the luma channel (Y) at full resolution.

    Args:
        yuv (torch.Tensor): YUV tensor with shape (B, 3, H, W) where channels are (Y, Cb, Cr).

    Returns:
        y (torch.Tensor): Y channel with shape (B, 1, H, W).
        uv (torch.Tensor): Downsampled UV channels with shape (B, 2, H/2, W/2).
    """

    def _downsample(tensor):
        return F.avg_pool2d(tensor, kernel_size=2, stride=2)

    y = yuv[:, :1, :, :]
    uv = yuv[:, 1:, :, :]

    return y, _downsample(uv)
