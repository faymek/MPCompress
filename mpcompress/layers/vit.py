"""Vision Transformer (ViT) in PyTorch

A PyTorch implement of Vision Transformers as described in:

'An Image Is Worth 16 x 16 Words: Transformers for Image Recognition at Scale'
    - https://arxiv.org/abs/2010.11929

`How to train your ViT? Data, Augmentation, and Regularization in Vision Transformers`
    - https://arxiv.org/abs/2106.10270

`FlexiViT: One Model for All Patch Sizes`
    - https://arxiv.org/abs/2212.08013

The official jax code is released and available at
  * https://github.com/google-research/vision_transformer
  * https://github.com/google-research/big_vision

Acknowledgments:
  * The paper authors for releasing code and weights, thanks!
  * I fixed my class token impl based on Phil Wang's https://github.com/lucidrains/vit-pytorch
  * Simple transformer style inspired by Andrej Karpathy's https://github.com/karpathy/minGPT
  * Bert reference code checks against Huggingface Transformers and Tensorflow Bert

Hacked together by / Copyright 2020, Ross Wightman
"""

import logging
import math
from functools import partial
from typing import Optional


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch.jit import Final

from timm.layers import Mlp, DropPath, use_fused_attn


def init_1d_freqs(dim: int, end: int, theta: float = 10000.0):
    """
    Precompute the frequency tensor for complex exponentials (cis) with given dimensions.

    This function calculates a frequency tensor with complex exponentials using the given dimension 'dim'
    and the end index 'end'. The 'theta' parameter scales the frequencies.
    The returned tensor contains complex values in complex64 data type.

    Args:
        dim (int): Dimension of the frequency tensor.
        end (int): End index for precomputing frequencies.
        theta (float, optional): Scaling factor for frequency computation. Defaults to 10000.0.

    Returns:
        torch.Tensor: Precomputed frequency tensor with complex exponentials.
    """
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    return freqs_cis


def init_2d_freqs(dim: int, num_heads: int, theta: float = 10.0, rotate: bool = True):
    freqs_x = []
    freqs_y = []
    mag = 1 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    for i in range(num_heads):
        angles = torch.rand(1) * 2 * torch.pi if rotate else torch.zeros(1)
        fx = torch.cat(
            [mag * torch.cos(angles), mag * torch.cos(torch.pi / 2 + angles)], dim=-1
        )
        fy = torch.cat(
            [mag * torch.sin(angles), mag * torch.sin(torch.pi / 2 + angles)], dim=-1
        )
        freqs_x.append(fx)
        freqs_y.append(fy)
    freqs_x = torch.stack(freqs_x, dim=0)
    freqs_y = torch.stack(freqs_y, dim=0)
    freqs = torch.stack([freqs_x, freqs_y], dim=0)
    return freqs


def init_t_xy(end_x: int, end_y: int):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x, t_y


def compute_mixed_cis(
    freqs: torch.Tensor, t_x: torch.Tensor, t_y: torch.Tensor, num_heads: int
):
    N = t_x.shape[0]
    # No float 16 for this range
    with torch.cuda.amp.autocast(enabled=False):
        freqs_x = (
            (t_x.unsqueeze(-1) @ freqs[0].unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_y = (
            (t_y.unsqueeze(-1) @ freqs[1].unsqueeze(-2))
            .view(N, num_heads, -1)
            .permute(1, 0, 2)
        )
        freqs_cis = torch.polar(torch.ones_like(freqs_x), freqs_x + freqs_y)
    return freqs_cis


def compute_axial_cis(dim: int, end_x: int, end_y: int, theta: float = 100.0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))

    t_x, t_y = init_t_xy(end_x, end_y)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis: torch.Tensor, x: torch.Tensor):
    ndim = x.ndim
    assert 0 <= 1 < ndim
    if freqs_cis.shape == (x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    elif freqs_cis.shape == (x.shape[-3], x.shape[-2], x.shape[-1]):
        shape = [d if i >= ndim - 3 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_emb(xq: torch.Tensor, xk: torch.Tensor, freqs_cis: torch.Tensor):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq).to(xq.device), xk_out.type_as(xk).to(xk.device)


class Attention(nn.Module):
    """Multi-head self-attention mechanism.

    This module implements scaled dot-product attention with optional QK normalization
    and fused attention support. It computes attention over the input sequence using
    query, key, and value projections.

    The attention mechanism follows: Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    where d_k is the head dimension.
    """

    fused_attn: Final[bool]

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
        norm_layer: nn.Module = nn.LayerNorm,
        **kwargs,
    ) -> None:
        """Initialize multi-head attention layer.

        Args:
            dim (int): Embedding dimension of input tokens. Must be divisible by num_heads.
            num_heads (int, optional): Number of attention heads. Defaults to 8.
            qkv_bias (bool, optional): Whether to use bias in QKV projection.
                Defaults to False.
            qk_norm (bool, optional): Whether to apply normalization to Q and K.
                Defaults to False.
            attn_drop (float, optional): Dropout probability for attention weights.
                Defaults to 0.0.
            proj_drop (float, optional): Dropout probability for output projection.
                Defaults to 0.0.
            norm_layer (nn.Module, optional): Normalization layer for QK normalization.
                Defaults to nn.LayerNorm.
            **kwargs (dict): Additional keyword arguments (unused).
        """
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**-0.5
        self.fused_attn = use_fused_attn()

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through attention layer.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C] where B is batch size,
                N is sequence length, and C is embedding dimension.
            attn_mask (torch.Tensor, optional): Attention mask tensor of shape [B, N, N]
                or broadcastable shape. Values are added to attention scores before softmax.
                Defaults to None.

        Returns:
            torch.Tensor: Output tensor of the same shape as input [B, N, C].
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv.unbind(0)
        q, k = self.q_norm(q), self.k_norm(k)

        if self.fused_attn:
            x = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_drop.p if self.training else 0.0,
                attn_mask=attn_mask,
            )
        else:
            q = q * self.scale
            attn = q @ k.transpose(-2, -1)
            if attn_mask is not None:
                attn += attn_mask
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = attn @ v

        x = x.transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RoPEAttention(Attention):
    """Multi-head attention with rotary position embeddings (RoPE).

    This attention mechanism extends standard multi-head attention by applying
    rotary position embeddings to query and key vectors. It supports two modes:

    - Mixed mode: Learnable 2D frequencies for image tokens and 1D frequencies for latent tokens
    - Axial mode: Fixed 2D axial frequencies for image tokens
    """

    def __init__(
        self,
        *args,
        num_prefix_tokens=1,
        num_latent_tokens=32,
        num_image_tokens=256,
        rope_theta=10.0,
        rope_mixed=True,
        **kwargs,
    ):
        """Initialize RoPE attention layer.

        Args:
            *args (tuple): Positional arguments passed to parent Attention class.
            num_prefix_tokens (int, optional): Number of prefix tokens (e.g., CLS token)
                that do not receive positional embeddings. Defaults to 1.
            num_latent_tokens (int, optional): Number of latent tokens that receive
                1D positional embeddings. Defaults to 32.
            num_image_tokens (int, optional): Number of image tokens that receive
                2D positional embeddings. Defaults to 256.
            rope_theta (float, optional): Base frequency parameter for RoPE.
                Higher values result in lower frequencies. Defaults to 10.0.
            rope_mixed (bool, optional): If True, use learnable mixed 2D frequencies.
                If False, use fixed axial 2D frequencies. Defaults to True.
            **kwargs (dict): Additional keyword arguments passed to parent Attention class.
        """
        super().__init__(*args, **kwargs)

        self.rope_mixed = rope_mixed
        self.num_prefix_tokens = num_prefix_tokens
        self.num_latent_tokens = num_latent_tokens
        self.num_image_tokens = num_image_tokens
        self.num_axis_tokens = int(num_image_tokens**0.5)

        if self.rope_mixed:
            self.compute_cis = partial(compute_mixed_cis, num_heads=self.num_heads)

            freqs = init_2d_freqs(
                dim=self.head_dim,
                num_heads=self.num_heads,
                theta=rope_theta,
                rotate=True,
            ).view(2, -1)
            self.freqs = nn.Parameter(freqs, requires_grad=True)

            t_x, t_y = init_t_xy(end_x=self.num_axis_tokens, end_y=self.num_axis_tokens)
            self.register_buffer("freqs_t_x", t_x)
            self.register_buffer("freqs_t_y", t_y)
        else:
            self.compute_cis = partial(
                compute_axial_cis, dim=self.head_dim, theta=rope_theta
            )
            freqs_cis = self.compute_cis(
                end_x=self.num_axis_tokens, end_y=self.num_axis_tokens
            )
            self.freqs_cis = freqs_cis

        # get pre-compted 1d rope
        freqs_1d = init_1d_freqs(dim=self.head_dim, end=self.num_latent_tokens)
        self.freqs_1d = nn.Parameter(freqs_1d, requires_grad=True)

    def forward(self, x, attn_mask=None):
        """Forward pass with rotary position embeddings.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C] where B is batch size,
                N is sequence length (1 + num_image_tokens + num_latent_tokens),
                and C is embedding dimension.
            attn_mask (torch.Tensor, optional): Attention mask tensor. Defaults to None.

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input [B, N, C].
        """
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        ###### Apply rotary position embedding
        w = h = math.sqrt(x.shape[1] - 1)
        if self.rope_mixed:
            t_x, t_y = self.freqs_t_x, self.freqs_t_y
            if (
                self.freqs_t_x.shape[0]
                != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens
            ):
                t_x, t_y = init_t_xy(end_x=w, end_y=h)
                t_x, t_y = t_x.to(x.device), t_y.to(x.device)
            freqs_cis = self.compute_cis(self.freqs, t_x, t_y)
        else:
            freqs_cis = self.freqs_cis
            if (
                self.freqs_cis.shape[0]
                != x.shape[1] - self.num_prefix_tokens - self.num_latent_tokens
            ):
                freqs_cis = self.compute_cis(end_x=w, end_y=h)
            freqs_cis = freqs_cis.to(x.device)

        # apply rotary position embedding to image tokens
        dtype = x.dtype
        with torch.cuda.amp.autocast(enabled=False):
            (
                q[:, :, self.num_prefix_tokens : -self.num_latent_tokens],
                k[:, :, self.num_prefix_tokens : -self.num_latent_tokens],
            ) = apply_rotary_emb(
                q[:, :, self.num_prefix_tokens : -self.num_latent_tokens],
                k[:, :, self.num_prefix_tokens : -self.num_latent_tokens],
                freqs_cis=freqs_cis,
            )
            q[:, :, -self.num_latent_tokens :], k[:, :, -self.num_latent_tokens :] = (
                apply_rotary_emb(
                    q[:, :, -self.num_latent_tokens :],
                    k[:, :, -self.num_latent_tokens :],
                    freqs_cis=self.freqs_1d,
                )
            )
        q, k = q.to(dtype), k.to(dtype)
        #########

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class LayerScale(nn.Module):
    """Layer scaling module for stabilizing deep networks.

    This module scales the input by a learnable parameter gamma. It is commonly used
    in Vision Transformers to stabilize training of very deep networks. The scaling
    factor is initialized to a small value (e.g., 1e-5) and learned during training.

    Reference: "Going deeper with Image Transformers" (Touvron et al., 2021)
    """

    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        """Initialize LayerScale module.

        Args:
            dim (int): Dimension of the input tensor (last dimension).
            init_values (float, optional): Initial value for the scaling parameter gamma.
                Defaults to 1e-5.
            inplace (bool, optional): Whether to perform in-place multiplication.
                Defaults to False.
        """
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Scale input tensor by learnable parameter.

        Args:
            x (torch.Tensor): Input tensor of shape [..., dim] where dim matches
                the dimension used in initialization.

        Returns:
            out (torch.Tensor): Scaled tensor of the same shape as input.
        """
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class Block(nn.Module):
    """Vision Transformer block with attention and MLP layers.

    This block implements a standard Transformer block for Vision Transformers,
    consisting of:

    - Multi-head self-attention with optional layer scaling
    - Feed-forward MLP with optional layer scaling
    - Residual connections with optional drop path regularization
    - Layer normalization before each sub-layer

    The block follows the architecture: x = x + DropPath(LayerScale(Attn(Norm(x))))
    followed by x = x + DropPath(LayerScale(MLP(Norm(x)))).
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_norm: bool = False,
        proj_drop: float = 0.0,
        attn_drop: float = 0.0,
        init_values: Optional[float] = None,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm,
        mlp_layer: nn.Module = Mlp,
        attn_layer: nn.Module = Attention,
    ) -> None:
        """Initialize Vision Transformer block.

        Args:
            dim (int): Embedding dimension of the input tokens.
            num_heads (int): Number of attention heads.
            mlp_ratio (float, optional): Ratio of MLP hidden dimension to embedding
                dimension. Defaults to 4.0.
            qkv_bias (bool, optional): Whether to use bias in QKV projection.
                Defaults to False.
            qk_norm (bool, optional): Whether to apply normalization to Q and K.
                Defaults to False.
            proj_drop (float, optional): Dropout probability for projection layers.
                Defaults to 0.0.
            attn_drop (float, optional): Dropout probability for attention weights.
                Defaults to 0.0.
            init_values (Optional[float], optional): Initial value for layer scaling.
                If None, layer scaling is disabled. Defaults to None.
            drop_path (float, optional): Drop path probability for stochastic depth.
                Defaults to 0.0.
            act_layer (nn.Module, optional): Activation function for MLP.
                Defaults to nn.GELU.
            norm_layer (nn.Module, optional): Normalization layer to use.
                Defaults to nn.LayerNorm.
            mlp_layer (nn.Module, optional): MLP layer class to use.
                Defaults to Mlp.
            attn_layer (nn.Module, optional): Attention layer class to use.
                Defaults to Attention.
        """
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_layer(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_norm=qk_norm,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            norm_layer=norm_layer,
        )
        self.ls1 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path1 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

        self.norm2 = norm_layer(dim)
        self.mlp = mlp_layer(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=proj_drop,
        )
        self.ls2 = (
            LayerScale(dim, init_values=init_values) if init_values else nn.Identity()
        )
        self.drop_path2 = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        """Forward pass through the Transformer block.

        Args:
            x (torch.Tensor): Input tensor of shape [B, N, C] where B is batch size,
                N is sequence length, and C is embedding dimension.
            attn_mask (torch.Tensor, optional): Attention mask tensor. If provided,
                will be applied to the attention computation. Defaults to None.

        Returns:
            out (torch.Tensor): Output tensor of the same shape as input [B, N, C].
        """
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x), attn_mask)))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        return x
