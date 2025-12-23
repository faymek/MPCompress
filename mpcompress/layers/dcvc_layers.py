# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

import torch
from torch import nn
from einops import rearrange
from mpcompress.layers.dcvc_cuda_inference import CUSTOMIZED_CUDA_INFERENCE

if CUSTOMIZED_CUDA_INFERENCE:
    from mpcompress.layers.dcvc_cuda_inference import DepthConvProxy, SubpelConv2xProxy


class WSiLU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.sigmoid(4.0 * x) * x


class WSiLUChunkAdd(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):
        x1, x2 = self.silu(x).chunk(2, 1)
        return x1 + x2


class WSiLUChunkAdd_blc(nn.Module):
    def __init__(self):
        super().__init__()
        self.silu = WSiLU()

    def forward(self, x):  # x: (b,l,c)
        x1, x2 = self.silu(x).chunk(2, dim=2)
        return x1 + x2


class SubpelConv2x(nn.Module):
    """Sub-pixel convolution layer for 2x upsampling.

    This layer performs 2x upsampling using sub-pixel convolution (also known as
    pixel shuffle). It uses a convolution followed by PixelShuffle to achieve
    efficient upsampling. Supports both PyTorch and CUDA implementations.
    """

    def __init__(self, in_ch, out_ch, kernel_size, padding=0):
        """Initialize SubpelConv2x layer.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            kernel_size (int): Size of the convolution kernel.
            padding (int, optional): Padding size for the convolution. Defaults to 0.
        """
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=kernel_size, padding=padding),
            nn.PixelShuffle(2),
        )
        self.padding = padding

        self.proxy = None

    def forward(self, x, to_cat=None, cat_at_front=True):
        """Forward pass with optional tensor concatenation.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            to_cat (torch.Tensor, optional): Tensor to concatenate with output.
                If None, only upsampled output is returned. Defaults to None.
            cat_at_front (bool, optional): If True, concatenate to_cat before output.
                If False, concatenate after output. Defaults to True.

        Returns:
            out (torch.Tensor): Upsampled tensor of shape [B, out_ch, 2*H, 2*W], or
                concatenated tensor if to_cat is provided.
        """
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, to_cat, cat_at_front)

        return self.forward_cuda(x, to_cat, cat_at_front)

    def forward_torch(self, x, to_cat=None, cat_at_front=True):
        """PyTorch implementation of forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            to_cat (torch.Tensor, optional): Tensor to concatenate with output.
                Defaults to None.
            cat_at_front (bool, optional): Concatenation order. Defaults to True.

        Returns:
            out (torch.Tensor): Upsampled or concatenated tensor.
        """
        out = self.conv(x)
        if to_cat is None:
            return out
        if cat_at_front:
            return torch.cat((to_cat, out), dim=1)
        return torch.cat((out, to_cat), dim=1)

    def forward_cuda(self, x, to_cat=None, cat_at_front=True):
        """CUDA-optimized implementation of forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            to_cat (torch.Tensor, optional): Tensor to concatenate with output.
                Defaults to None.
            cat_at_front (bool, optional): Concatenation order. Defaults to True.

        Returns:
            out (torch.Tensor): Upsampled or concatenated tensor.
        """
        if self.proxy is None:
            self.proxy = SubpelConv2xProxy()
            self.proxy.set_param(self.conv[0].weight, self.conv[0].bias, self.padding)

        if to_cat is None:
            return self.proxy.forward(x)

        return self.proxy.forward_with_cat(x, to_cat, cat_at_front)


class DepthConvBlock(nn.Module):
    """Depthwise convolution block with feed-forward network.

    This block implements a residual block using depthwise separable convolutions
    and a feed-forward network. It consists of:
    - Optional channel adaptor (1x1 conv) for dimension matching
    - Depthwise convolution path with residual connection
    - Feed-forward network with residual connection
    - Optional shortcut connection from input
    - Optional quantization step scaling
    - Optional tensor concatenation

    Supports both PyTorch and CUDA implementations for efficient inference.
    """

    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        """Initialize DepthConvBlock.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
            shortcut (bool, optional): Whether to add shortcut connection from input
                to final output. Defaults to False.
            force_adaptor (bool, optional): Whether to force use of channel adaptor
                even when in_ch == out_ch. Defaults to False.
        """
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Conv2d(in_ch, out_ch, 1)
        self.shortcut = shortcut
        self.dc = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Conv2d(out_ch, out_ch * 4, 1),
            WSiLUChunkAdd(),
            nn.Conv2d(out_ch * 2, out_ch, 1),
        )

        self.proxy = None

    def forward(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        """Forward pass with optional quantization and concatenation.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            quant_step (float, optional): Quantization step for scaling output.
                If provided, output is multiplied by quant_step. Defaults to None.
            to_cat (torch.Tensor, optional): Tensor to concatenate with output.
                Defaults to None.
            cat_at_front (bool, optional): If True, concatenate to_cat before output.
                If False, concatenate after output. Defaults to True.

        Returns:
            out (torch.Tensor): Processed tensor of shape [B, out_ch, H, W], or
                concatenated tensor if to_cat is provided.
        """
        if not CUSTOMIZED_CUDA_INFERENCE or not x.is_cuda:
            return self.forward_torch(x, quant_step, to_cat, cat_at_front)

        return self.forward_cuda(x, quant_step, to_cat, cat_at_front)

    def forward_torch(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        """PyTorch implementation of forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            quant_step (float, optional): Quantization step for scaling. Defaults to None.
            to_cat (torch.Tensor, optional): Tensor to concatenate. Defaults to None.
            cat_at_front (bool, optional): Concatenation order. Defaults to True.

        Returns:
            out (torch.Tensor): Processed or concatenated tensor.
        """
        if self.adaptor is not None:
            x = self.adaptor(x)
        out = self.dc(x) + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        if quant_step is not None:
            out = out * quant_step
        if to_cat is not None:
            if cat_at_front:
                out = torch.cat((to_cat, out), dim=1)
            else:
                out = torch.cat((out, to_cat), dim=1)
        return out

    def forward_cuda(self, x, quant_step=None, to_cat=None, cat_at_front=True):
        """CUDA-optimized implementation of forward pass.

        Args:
            x (torch.Tensor): Input tensor of shape [B, C, H, W].
            quant_step (float, optional): Quantization step for scaling. Defaults to None.
            to_cat (torch.Tensor, optional): Tensor to concatenate. Defaults to None.
            cat_at_front (bool, optional): Concatenation order. Defaults to True.

        Returns:
            out (torch.Tensor): Processed or concatenated tensor.
        """
        if self.proxy is None:
            self.proxy = DepthConvProxy()
            if self.adaptor is not None:
                self.proxy.set_param_with_adaptor(
                    self.dc[0].weight,
                    self.dc[0].bias,
                    self.dc[2].weight,
                    self.dc[2].bias,
                    self.dc[3].weight,
                    self.dc[3].bias,
                    self.ffn[0].weight,
                    self.ffn[0].bias,
                    self.ffn[2].weight,
                    self.ffn[2].bias,
                    self.adaptor.weight,
                    self.adaptor.bias,
                    self.shortcut,
                )
            else:
                self.proxy.set_param(
                    self.dc[0].weight,
                    self.dc[0].bias,
                    self.dc[2].weight,
                    self.dc[2].bias,
                    self.dc[3].weight,
                    self.dc[3].bias,
                    self.ffn[0].weight,
                    self.ffn[0].bias,
                    self.ffn[2].weight,
                    self.ffn[2].bias,
                    self.shortcut,
                )

        if quant_step is not None:
            return self.proxy.forward_with_quant_step(x, quant_step)
        if to_cat is not None:
            return self.proxy.forward_with_cat(x, to_cat, cat_at_front)

        return self.proxy.forward(x)


class Conv2d_blc0(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_ch, out_ch, kernel_size, stride=stride, padding=kernel_size // 2
        )
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x, resolution):
        # x: (b, 1+h*w, c)
        x_cls = x[:, 0:1]
        x_cls = self.linear(x_cls)

        x_patch = x[:, 1:]
        H, W = resolution
        x_patch = rearrange(x_patch, "b (h w) c -> b c h w", h=H, w=W)
        x_patch = self.conv(x_patch)
        x_patch = rearrange(x_patch, "b c h w -> b (h w) c")

        x_trans = torch.cat([x_cls, x_patch], dim=1).contiguous()
        return x_trans


class Deconv2d_blc0(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride=1):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(
            in_ch,
            out_ch,
            kernel_size,
            stride=stride,
            output_padding=stride - 1,
            padding=kernel_size // 2,
        )
        self.linear = nn.Linear(in_ch, out_ch)

    def forward(self, x, resolution):
        # x: (b, 1+h*w, c)
        x_cls = x[:, 0:1]
        x_cls = self.linear(x_cls)

        x_patch = x[:, 1:]
        H, W = resolution
        x_patch = rearrange(x_patch, "b (h w) c -> b c h w", h=H, w=W)
        x_patch = self.deconv(x_patch)
        x_patch = rearrange(x_patch, "b c h w -> b (h w) c")

        x_trans = torch.cat([x_cls, x_patch], dim=1).contiguous()
        return x_trans


class DepthConvBlock_blc0(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Linear(in_ch, out_ch)
        self.shortcut = shortcut
        self.cls_linear = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            WSiLU(),
            nn.Linear(out_ch, out_ch),
        )
        self.patch_dconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 1),
            WSiLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(out_ch, out_ch * 4),
            WSiLUChunkAdd_blc(),
            nn.Linear(out_ch * 2, out_ch),
        )

    def forward(self, x, resolution):
        if self.adaptor is not None:
            x = self.adaptor(x)

        x_cls = x[:, 0:1]
        x_cls = self.cls_linear(x_cls)

        x_patch = x[:, 1:]
        H, W = resolution
        x_patch = rearrange(x_patch, "b (h w) c -> b c h w", h=H, w=W)
        x_patch = self.patch_dconv(x_patch)
        x_patch = rearrange(x_patch, "b c h w -> b (h w) c")

        x_trans = torch.cat([x_cls, x_patch], dim=1).contiguous()
        out = x_trans + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out


class BlockStack(nn.Module):
    def __init__(self, cls_name, *args, blocks=1, **kwargs) -> None:
        super().__init__()
        self.blocks = nn.Sequential(*[cls_name(*args, **kwargs) for _ in range(blocks)])

    def forward(self, x, resolution):
        for block in self.blocks:
            x = block(x, resolution)
        return x


class DepthConvBlock_blc1(nn.Module):
    def __init__(self, in_ch, out_ch, shortcut=False, force_adaptor=False):
        super().__init__()
        self.adaptor = None
        if in_ch != out_ch or force_adaptor:
            self.adaptor = nn.Linear(in_ch, out_ch)
        self.shortcut = shortcut
        self.share_linear = nn.Sequential(
            nn.Linear(out_ch, out_ch),
            WSiLU(),
        )
        self.cls_linear = nn.Linear(out_ch, out_ch)
        self.patch_dconv = nn.Sequential(
            nn.Conv2d(out_ch, out_ch, 3, padding=1, groups=out_ch),
            nn.Conv2d(out_ch, out_ch, 1),
        )
        self.ffn = nn.Sequential(
            nn.Linear(out_ch, out_ch * 4),
            WSiLUChunkAdd_blc(),
            nn.Linear(out_ch * 2, out_ch),
        )

    def forward(self, x, resolution):
        if self.adaptor is not None:
            x = self.adaptor(x)

        x_share = self.share_linear(x)
        x_cls = x_share[:, 0:1]
        x_cls = self.cls_linear(x_cls)

        x_patch = x_share[:, 1:]
        H, W = resolution
        x_patch = rearrange(x_patch, "b (h w) c -> b c h w", h=H, w=W)
        x_patch = self.patch_dconv(x_patch)
        x_patch = rearrange(x_patch, "b c h w -> b (h w) c")

        x_share = torch.cat([x_cls, x_patch], dim=1).contiguous()
        out = x_share + x
        out = self.ffn(out) + out
        if self.shortcut:
            out = out + x
        return out


class ResidualBlockWithStride2(nn.Module):
    """Residual block with 2x downsampling.

    This block performs 2x spatial downsampling followed by depthwise convolution
    processing. It combines a strided convolution for downsampling with a
    DepthConvBlock for feature refinement.
    """

    def __init__(self, in_ch, out_ch):
        """Initialize ResidualBlockWithStride2.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super().__init__()
        self.down = nn.Conv2d(in_ch, out_ch, 2, stride=2)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        """Forward pass with 2x downsampling.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_ch, H, W].

        Returns:
            out (torch.Tensor): Downsampled and processed tensor of shape [B, out_ch, H//2, W//2].
        """
        x = self.down(x)
        out = self.conv(x)
        return out


class ResidualBlockUpsample(nn.Module):
    """Residual block with 2x upsampling.

    This block performs 2x spatial upsampling followed by depthwise convolution
    processing. It combines a sub-pixel convolution for upsampling with a
    DepthConvBlock for feature refinement.
    """

    def __init__(self, in_ch, out_ch):
        """Initialize ResidualBlockUpsample.

        Args:
            in_ch (int): Number of input channels.
            out_ch (int): Number of output channels.
        """
        super().__init__()
        self.up = SubpelConv2x(in_ch, out_ch, 1)
        self.conv = DepthConvBlock(out_ch, out_ch, shortcut=True)

    def forward(self, x):
        """Forward pass with 2x upsampling.

        Args:
            x (torch.Tensor): Input tensor of shape [B, in_ch, H, W].

        Returns:
            out (torch.Tensor): Upsampled and processed tensor of shape [B, out_ch, 2*H, 2*W].
        """
        out = self.up(x)
        out = self.conv(out)
        return out
