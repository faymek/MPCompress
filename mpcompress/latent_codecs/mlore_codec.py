"""
MLoRE Feature Compression Codec for MPCompress Framework

This module wraps the RFC's FeatCompression module to provide standard
LatentCodec interfaces compatible with the MPCompress framework.

The codec uses a hyperprior-style architecture with Gaussian conditional
entropy model for compressing ViT intermediate features.
"""

import sys
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from compressai.models.base import CompressionModel
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.layers import (
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
)
from compressai.registry import register_module

# RFC code has been migrated to mpcompress, no need for external RFC dependency


__all__ = [
    "MLoREFeatureCodec",
    "MLoREFeatureCodecLight",
]


def conv(in_channels, out_channels, kernel_size=5, stride=1):
    """Helper function for convolution with same padding."""
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )


def ste_round(x):
    """Straight-through estimator rounding."""
    return torch.round(x) - x.detach() + x


SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min_val=SCALES_MIN, max_val=SCALES_MAX, levels=SCALES_LEVELS):
    """Generate scale table for Gaussian conditional."""
    return torch.exp(torch.linspace(math.log(min_val), math.log(max_val), levels))


@register_module("MLoREFeatureCodec")
class MLoREFeatureCodec(CompressionModel):
    """
    Feature compression codec using hyperprior architecture.
    
    This codec compresses ViT intermediate features using a learned
    transform coding approach with Gaussian conditional entropy model.
    
    Architecture:
        - g_a: Analysis transform (encoder)
        - g_s: Synthesis transform (decoder)
        - h_a: Hyper analysis (side info encoder)
        - h_s: Hyper synthesis (mean/scale prediction)
    
    Args:
        feat_dim: Input feature dimension (e.g., 768 for ViT-B)
        N: Number of channels in transforms
        M: Bottleneck dimension
    """
    
    def __init__(self, feat_dim=768, N=1024, M=512):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.N = N
        self.M = M
        
        # Analysis transform
        self.g_a = nn.Sequential(
            ResidualBlockWithStride(feat_dim, N, 1),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, M, 1),
        )
        
        # Synthesis transform
        self.g_s = nn.Sequential(
            ResidualBlockUpsample(M, N, 1),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlockUpsample(N, N, 1),
            ResidualBlockUpsample(N, feat_dim, 1),
        )
        
        # Hyper transforms
        self.entropy_bottleneck = EntropyBottleneck(384)
        self.gaussian_conditional = GaussianConditional(None)
        
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M, N, 2),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, 384, 2),
        )
        
        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample(384, N, 2),
            ResidualBlockUpsample(N, N, 2),
            conv(N, N // 2, stride=1, kernel_size=3),
            nn.GELU(),
            conv(N // 2, M, stride=1, kernel_size=3),
        )
        
        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample(384, N, 2),
            ResidualBlockUpsample(N, N, 2),
            conv(N, N // 2, stride=1, kernel_size=3),
            nn.GELU(),
            conv(N // 2, M, stride=1, kernel_size=3),
        )
    
    def forward(self, x, target=None):
        """
        Forward pass for training.
        
        Args:
            x: Input feature tensor (B, C, H, W)
            target: Optional target for MSE loss (default: use x)
            
        Returns:
            dict with keys:
                - bpp_loss: Bits per pixel loss
                - mse_loss: Reconstruction MSE loss
                - x_hat: Reconstructed features
                - likelihoods: Dict of likelihood tensors
        """
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3] * 256
        
        y = self.g_a(x)
        z = self.h_a(y)
        
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)
        
        _, y_likelihood = self.gaussian_conditional(y, latent_scales, latent_means)
        y_hat = ste_round(y - latent_means) + latent_means
        x_hat = self.g_s(y_hat)
        
        y_bpp = torch.log(y_likelihood).sum() / (-math.log(2) * num_pixels)
        z_bpp = torch.log(z_likelihood).sum() / (-math.log(2) * num_pixels)
        bpp_loss = y_bpp + z_bpp
        
        if target is not None:
            mse_loss = F.mse_loss(x_hat, target.detach())
        else:
            mse_loss = F.mse_loss(x_hat, x.detach())
        
        return {
            "bpp_loss": bpp_loss,
            "mse_loss": mse_loss,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
        }
    
    def update(self, scale_table=None, force=False):
        """Update entropy model parameters."""
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def compress(self, x):
        """
        Compress features to bitstream.
        
        Args:
            x: Input feature tensor (B, C, H, W)
            
        Returns:
            dict following MPCompress coded_unit format:
                - strings: Dict of compressed bitstreams
                - shape: Tensor shape for decompression
        """
        y = self.g_a(x)
        y_shape = y.shape[2:]
        
        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        scale = self.h_scale_s(z_hat)
        mean = self.h_mean_s(z_hat)
        
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        
        index = self.gaussian_conditional.build_indexes(scale)
        y_q = self.gaussian_conditional.quantize(y, "symbols", mean)
        
        symbols_list.extend(y_q.reshape(-1).tolist())
        indexes_list.extend(index.reshape(-1).tolist())
        
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        
        return {
            "strings": {"y": [[y_string]], "z": z_strings},
            "shape": z.size()[-2:],
            "pstate": {
                "y_shape": y_shape,
                "input_shape": x.shape,
            }
        }
    
    def decompress(self, strings, shape, pstate=None):
        """
        Decompress bitstream to features.
        
        Args:
            strings: Dict of compressed bitstreams
            shape: Hyper latent shape
            pstate: Additional state (y_shape, input_shape)
            
        Returns:
            dict with keys:
                - x_hat: Reconstructed features
        """
        z_hat = self.entropy_bottleneck.decompress(strings["z"], shape)
        scales = self.h_scale_s(z_hat)
        means = self.h_mean_s(z_hat)
        
        y_shape = pstate["y_shape"] if pstate else [z_hat.shape[2] * 4, z_hat.shape[3] * 4]
        
        y_string = strings["y"][0][0]
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        
        decoder = RansDecoder()
        decoder.set_stream(y_string)
        
        index = self.gaussian_conditional.build_indexes(scales)
        rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
        y_hat = self.gaussian_conditional.dequantize(rv, means)
        x_hat = self.g_s(y_hat)
        
        return {"x_hat": x_hat}


@register_module("MLoREFeatureCodecLight")
class MLoREFeatureCodecLight(CompressionModel):
    """
    Lightweight feature compression codec.
    
    Uses simpler transforms with fewer parameters for faster
    encoding/decoding, suitable for real-time applications.
    
    Args:
        feat_dim: Input feature dimension
        N: Number of channels in transforms
        M: Bottleneck dimension
    """
    
    def __init__(self, feat_dim=768, N=512, M=256):
        super().__init__()
        
        self.feat_dim = feat_dim
        self.N = N
        self.M = M
        
        # Simpler analysis transform
        self.g_a = nn.Sequential(
            conv(feat_dim, N, kernel_size=3, stride=1),
            nn.GELU(),
            conv(N, N, kernel_size=3, stride=1),
            nn.GELU(),
            conv(N, M, kernel_size=3, stride=1),
        )
        
        # Simpler synthesis transform
        self.g_s = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.GELU(),
            conv(N, N, kernel_size=3, stride=1),
            nn.GELU(),
            conv(N, feat_dim, kernel_size=3, stride=1),
        )
        
        # Entropy model
        self.entropy_bottleneck = EntropyBottleneck(M)
        self.gaussian_conditional = GaussianConditional(None)
        
        # Hyper transforms
        self.h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=2),
            nn.GELU(),
            conv(N, M, kernel_size=3, stride=2),
        )
        
        self.h_s = nn.Sequential(
            nn.ConvTranspose2d(M, N, kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            nn.ConvTranspose2d(N, M * 2, kernel_size=4, stride=2, padding=1),
        )
    
    def forward(self, x, target=None):
        """Forward pass for training."""
        num_pixels = x.shape[0] * x.shape[2] * x.shape[3] * 256
        
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_hat, z_likelihood = self.entropy_bottleneck(z)
        
        params = self.h_s(z_hat)
        scales, means = params.chunk(2, dim=1)
        scales = F.relu(scales) + 0.11
        
        y_hat, y_likelihood = self.gaussian_conditional(y, scales, means)
        x_hat = self.g_s(y_hat)
        
        y_bpp = torch.log(y_likelihood).sum() / (-math.log(2) * num_pixels)
        z_bpp = torch.log(z_likelihood).sum() / (-math.log(2) * num_pixels)
        bpp_loss = y_bpp + z_bpp
        
        if target is not None:
            mse_loss = F.mse_loss(x_hat, target.detach())
        else:
            mse_loss = F.mse_loss(x_hat, x.detach())
        
        return {
            "bpp_loss": bpp_loss,
            "mse_loss": mse_loss,
            "x_hat": x_hat,
            "likelihoods": {"y": y_likelihood, "z": z_likelihood},
        }
    
    def update(self, scale_table=None, force=False):
        """Update entropy model parameters."""
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated
    
    def compress(self, x):
        """Compress features to bitstream."""
        y = self.g_a(x)
        z = self.h_a(y)
        
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        
        params = self.h_s(z_hat)
        scales, means = params.chunk(2, dim=1)
        scales = F.relu(scales) + 0.11
        
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_strings = self.gaussian_conditional.compress(y, indexes, means)
        
        return {
            "strings": {"y": y_strings, "z": z_strings},
            "shape": z.size()[-2:],
            "pstate": {
                "y_shape": y.shape[2:],
                "input_shape": x.shape,
            }
        }
    
    def decompress(self, strings, shape, pstate=None):
        """Decompress bitstream to features."""
        z_hat = self.entropy_bottleneck.decompress(strings["z"], shape)
        
        params = self.h_s(z_hat)
        scales, means = params.chunk(2, dim=1)
        scales = F.relu(scales) + 0.11
        
        indexes = self.gaussian_conditional.build_indexes(scales)
        y_hat = self.gaussian_conditional.decompress(strings["y"], indexes, means=means)
        x_hat = self.g_s(y_hat)
        
        return {"x_hat": x_hat}


class MLoREFeatureCodecWrapper(nn.Module):
    """
    Wrapper to use RFC's original FeatCompression class.
    
    This wrapper imports the original FeatCompression from RFC
    and provides the same interface as MLoREFeatureCodec.
    
    Args:
        feat_dim: Input feature dimension
        N: Number of channels
        M: Bottleneck dimension
    """
    
    def __init__(self, feat_dim=768, N=1024, M=512):
        super().__init__()
        
        # Import migrated FeatCompression
        from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom import FeatCompression
        
        self.codec = FeatCompression(feat_dim, N=N, M=M)
    
    def forward(self, x, target=None):
        """Forward pass using RFC's implementation."""
        bpp_loss, mse_loss, x_hat = self.codec(x, target)
        return {
            "bpp_loss": bpp_loss,
            "mse_loss": mse_loss,
            "x_hat": x_hat,
        }
    
    def update(self, scale_table=None, force=False):
        """Update entropy model parameters."""
        return self.codec.update(scale_table, force)
    
    def compress(self, x):
        """Compress features."""
        return self.codec.compress(x)
    
    def decompress(self, strings, shape):
        """Decompress features."""
        return self.codec.decompress(strings, shape)




