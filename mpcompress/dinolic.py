# Copyright (c) 2021-2024, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import torch
import torch.nn as nn

from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
    HyperLatentCodecWithCtx,
    HyperpriorLatentCodecWithCtx,
)
from compressai.layers import (
    AttentionBlock,
    CheckerboardMaskedConv2d,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv1x1,
    conv3x3,
    sequential_channel_ramp,
    subpel_conv3x3,
)
from compressai.registry import register_model

from compressai.models.base import SimpleVAECompressionModel
from compressai.models.utils import conv, deconv
from compressai.models.sensetime import ResidualBottleneckBlock
from einops import rearrange
import timm
import torchvision.transforms as transforms

# import timm.models.vision_transformer.Block as Block
from compressai.models.sensetime import Elic2022Official
from mpcompress.vit import Block
from mpcompress.layers import DepthConvBlock, SubpelConv2x
import torch.nn.functional as F

from mpcompress.vq_model import VQModel

def border_padding(x, patch_size):
    B, C, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
    return x


def center_padding(x, patch_size):
    B, C, H, W = x.shape
    pad_h = (patch_size - H % patch_size) % patch_size
    pad_w = (patch_size - W % patch_size) % patch_size
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    # print("center padding: ", pad_left, pad_right, pad_top, pad_bottom)
    if pad_h > 0 or pad_w > 0:
        x = torch.nn.functional.pad(x, (pad_left, pad_right, pad_top, pad_bottom))
    return x


def center_crop(x, org_size):
    img_h, img_w = org_size
    crop_h = x.shape[2] - img_h
    crop_w = x.shape[3] - img_w
    crop_left = crop_w // 2
    crop_right = crop_w - crop_left
    crop_top = crop_h // 2  
    crop_bottom = crop_h - crop_top
    print("tail crop:    ", crop_left, crop_right, crop_top, crop_bottom)
    x = x[:, :, crop_top:crop_top+img_h, crop_left:crop_left+img_w]
    return x


class Dinov2TimmBackbone(nn.Module):
    def __init__(self, model_size="small", img_size=256, patch_size=16, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float):
        super().__init__()
        self.n_last_blocks = n_last_blocks
        self.autocast_ctx = autocast_ctx
        assert model_size in ["small", "base", "large", "giant"]
        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.dynamic_size = dynamic_size
        self.feature_model = self.load_timm_model()

    def load_timm_model(self):
        feature_model = timm.create_model(
            f"vit_{self.model_size}_patch14_dinov2.lvd142m",
            pretrained=True,
            img_size=self.img_size,
            patch_size=self.patch_size,
            drop_path_rate=0.0,
            dynamic_img_size=self.dynamic_size
        )
        feature_model.eval()
        return feature_model

    def forward(
        self,
        x,
        slot=-4,
        n=None,  # Layers or n last layers to take
        reshape=False,
        return_class_token=True,
        norm=True,
    ):
        if n is None:
            n = self.n_last_blocks
        with torch.inference_mode():
            with self.autocast_ctx():
                # features = self.feature_model.get_intermediate_layers(
                #     x, n, reshape=reshape, return_prefix_tokens=return_class_token, norm=norm)
                # features = [(f, c[:, 0, :]) for (f, c) in features]
                features = self.feature_startpart(x, slot)
                # features += torch.randn_like(features) * 1
                features = self.feature_endpart(
                    x=features,
                    slot=slot,
                    n=n,
                    reshape=reshape,
                    return_class_token=return_class_token,
                    norm=norm,
                )
                return features

    def feature_startpart(
        self,
        x,
        slot=-4,
    ):
        # for compressed model, we only need the output of the -4 block
        dino = self.feature_model
        x = dino.patch_embed(x)
        x = dino._pos_embed(x)
        x = dino.patch_drop(x)
        x = dino.norm_pre(x)
        for i, blk in enumerate(dino.blocks[:slot]):
            x = blk(x)
        return x

    def feature_output(  # for training purpose
        self,
        x,  # from get_intermediate_layers_start, the output of the -4 block
        slot=-4,
        n=1,  # Layers or n last layers to take
        norm=True,
    ):
        dino = self.feature_model

        # If n is an int, take the n last blocks. If it's a list, take them
        outputs = []
        total_block_len = len(dino.blocks)
        if isinstance(n, int):
            blocks_to_take = range(total_block_len - n, total_block_len)
        else:
            blocks_to_take = n

        for i, blk in enumerate(dino.blocks[slot:]):
            x = blk(x)
            block_num = total_block_len + slot + i
            if block_num in blocks_to_take:
                outputs.append(x)
        assert len(outputs) == len(blocks_to_take), (
            f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
        )

        if norm:
            outputs = [dino.norm(out) for out in outputs]
        return outputs

    def feature_endpart(
        self,
        x,  # from get_intermediate_layers_start, the output of the -4 block
        slot=-4,
        n=1,  # Layers or n last layers to take
        token_res=None,
        reshape=False,
        return_class_token=False,
        norm=True,
    ):
        dino = self.feature_model

        # If n is an int, take the n last blocks. If it's a list, take them
        outputs = []
        if slot is None:
            outputs.append(x)  # x is just the last layer
        else:
            total_block_len = len(dino.blocks)
            if isinstance(n, int):
                blocks_to_take = range(total_block_len - n, total_block_len)
            else:
                blocks_to_take = n

            for i, blk in enumerate(dino.blocks[slot:]):
                x = blk(x)
                block_num = total_block_len + slot + i
                if block_num in blocks_to_take:
                    outputs.append(x)
            assert len(outputs) == len(blocks_to_take), (
                f"only {len(outputs)} / {len(blocks_to_take)} blocks found"
            )

        if norm:
            outputs = [dino.norm(out) for out in outputs]

        class_tokens = [out[:, 0] for out in outputs]
        outputs = [out[:, dino.num_prefix_tokens :] for out in outputs]
        if reshape:
            h, w = token_res
            outputs = [
                rearrange(out, "b (h w) c -> b c h w", h=h, w=w) for out in outputs
            ]
        if return_class_token:
            return tuple(zip(outputs, class_tokens))
        return tuple(outputs)


@register_model("DinoElicBase")
class DinoElicBase(Elic2022Official):
    def __init__(self, N=192, M=256, D_DINO=384, groups=None, **kwargs):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups, **kwargs)

        self.slot = -4

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = nn.Sequential(
            AttentionBlock(D_DINO),
            deconv(D_DINO, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)
        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        x_hat = self.g_s(h_hat.detach())
        # x_hat = self.g_s(h.detach())
        # x_hat = self.conv(x)

        return {
            "x_hat": x_hat,
            "h_dino_hat": h_dino_hat,
            "h_dino": h_dino,
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)
            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out

        # return {
        #     "feat": feat_out,
        #     "likelihoods": y_out["likelihoods"],
        # }

    def compress(self, x):
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
        }


@register_model("DinoL44")
class DinoL44(Elic2022Official):
    def __init__(self, N=192, M=256, D_DINO=384, groups=None, **kwargs):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = -4

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

        self.num_out_layers = 4

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_dino_hat": h_dino_hat,
            "h_dino": h_dino,
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)
            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("DinoL41Norm")
class DinoL41Norm(Elic2022Official):
    def __init__(self, N=192, M=256, D_DINO=384, groups=None, **kwargs):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = -4

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        self.num_out_layers = 4

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls(N=192, M=120, groups=5)
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)
            o_dino = self.dino.feature_output(
                h_dino,
                slot=self.slot,
                n=1,
                norm=True,
            )[0]

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        o_dino_hat = self.dino.feature_output(
            h_dino_hat,
            slot=self.slot,
            n=1,
            norm=True,
        )[0]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)
            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("DinoL41NormUpdate")
class DinoL41NormUpdate(Elic2022Official):
    def __init__(
        self,
        N=192,
        M=256,
        D_DINO=384,
        groups=None,
        dynamic_size=True,
        patch_size=16,
        **kwargs,
    ):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = -4
        self.patch_size = patch_size

        self.dino_input_transform = transforms.Compose(
            [
                # transforms.CenterCrop(img_size),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=256, patch_size=patch_size, dynamic_size=True, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        self.num_out_layers = 4

    @classmethod
    def from_state_dict(cls, state_dict, img_size=512, patch_size=16, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls(N=192, M=120, groups=5, img_size=img_size, patch_size=patch_size)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dino.")}
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)
            o_dino = self.dino.feature_output(
                h_dino,
                slot=self.slot,
                n=1,
                norm=True,
            )[0]

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        token_H = x.shape[2] // self.patch_size
        token_W = x.shape[3] // self.patch_size
        h = rearrange(h, "B (H W) C -> B C H W", H=token_H, W=token_W)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        o_dino_hat = self.dino.feature_output(
            h_dino_hat,
            slot=self.slot,
            n=1,
            norm=True,
        )[0]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear_no_compression(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)
            feat_out = self.dino.feature_endpart(
                feat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
        return feat_out

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            B, C, H, W = x.shape
            # 如果图像尺寸不是lic_patch的整数倍，需要进行填充
            lic_patch = 128
            pad_h = (lic_patch - H % lic_patch) % lic_patch
            pad_w = (lic_patch - W % lic_patch) % lic_patch
            if pad_h > 0 or pad_w > 0:
                x = torch.nn.functional.pad(x, (0, pad_w, 0, pad_h))
            # 设置dino输入尺寸
            B, C, H, W = x.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)

            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                token_res=None,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out

    def eval_bpp(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)
            x_dino = self.dino_input_transform(x)

            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size  # token resolution

            feat = self.dino.feature_startpart(x_dino, slot=self.slot)
            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            y = self.f_a(h)
            y_out = self.latent_codec(y)

        return {
            "likelihoods": y_out["likelihoods"],
        }
    
    def dino_eval_mmseg_no_compression(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=16)
            x_dino = self.dino_input_transform(x)
            
            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size  # token resolution

            feat = self.dino.feature_startpart(x_dino, slot=self.slot)
            feat_hat = feat

            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                token_res=(LH, LW),
                reshape=True,
                return_class_token=False,
                norm=True,
            )

        return feat_out

    def dino_eval_mmseg(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)
            x_dino = self.dino_input_transform(x)

            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size  # token resolution

            feat = self.dino.feature_startpart(x_dino, slot=self.slot)
            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x_dino.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)

            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                token_res=(LH, LW),
                reshape=True,
                return_class_token=False,
                norm=True,
            )

        return feat_out



class HyperEncoderWithCtx(nn.Module):
    def __init__(self, N, M, D_VQGAN):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(M + D_VQGAN, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

    def forward(self, x, ctx):
        return self.h_a(torch.cat([x, ctx], dim=1))

class HyperDecoderWithCtx(nn.Module):
    def __init__(self, N, M, D_VQGAN):
        super().__init__()
        self.h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
        )
        self.fusion = nn.Sequential(
            conv(N * 2 + D_VQGAN, N * 2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N * 2, N * 2, kernel_size=3, stride=1),
        )

    def forward(self, x, ctx):
        h = self.h_s(x)
        h = self.fusion(torch.cat([h, ctx], dim=1))
        return h
            

@register_model("MPC12")
class MPC12(Elic2022Official):
    def __init__(
        self,
        N=192,
        M=256,
        D=256,
        D_DINO=384,
        D_VQGAN=256,
        groups=None,
        dynamic_size=True,
        patch_size=16,
        vqgan_config=None,
        **kwargs,
    ):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = -4
        self.patch_size = patch_size

        self.g_a = None
        self.g_s = None

        self.vqgan = VQModel(**vqgan_config)
        self.dino_input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=256, patch_size=patch_size, dynamic_size=True, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )


        self.mp12_enc = nn.Sequential(
            conv(D_DINO+D_VQGAN, D_DINO, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_DINO, D_DINO, kernel_size=3, stride=1),
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.mp12_dec = nn.Sequential(
            conv(D_DINO+D_VQGAN, D_DINO, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_DINO, D_DINO, kernel_size=3, stride=1),
        )

        self.mp12_vqgan_down = conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=2)

        self.num_out_layers = 4

        h_a = HyperEncoderWithCtx(N, M, D_VQGAN)
        h_s = HyperDecoderWithCtx(N, M, D_VQGAN)

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodecWithCtx(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodecWithCtx(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    @classmethod
    def from_state_dict(cls, state_dict, config, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls(**config.ee_model.params)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith("dino.") and not k.startswith("vqgan.")}
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vqgan.quantize(h_vqgan)

            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            o_dino = self.dino.feature_output(
                h_dino,
                slot=self.slot,
                n=1,
                norm=True,
            )[0]

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        h_vqgan_ctx_down = self.mp12_vqgan_down(h_vqgan_ctx)

        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        token_H = x.shape[2] // self.patch_size
        token_W = x.shape[3] // self.patch_size
        h = rearrange(h, "B (H W) C -> B C H W", H=token_H, W=token_W)
        h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
        y = self.f_a(h)
        y_out = self.latent_codec(y, h_vqgan_ctx_down)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        h_hat = self.mp12_dec(torch.cat([h_hat, h_vqgan_ctx], dim=1))
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        o_dino_hat = self.dino.feature_output(
            h_dino_hat,
            slot=self.slot,
            n=1,
            norm=True,
        )[0]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear_no_compression(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)
            feat_out = self.dino.feature_endpart(
                feat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
        return feat_out

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)
            
            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vqgan.quantize(h_vqgan)

            x_dino = self.dino_input_transform(x)
            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan_ctx_down = self.mp12_vqgan_down(h_vqgan_ctx)

            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
            y = self.f_a(h)
            y_out = self.latent_codec(y, h_vqgan_ctx_down)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            h_hat = self.mp12_dec(torch.cat([h_hat, h_vqgan_ctx], dim=1))
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            h_dino_hat = self.post_vit_blocks(_h_hat)

            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=n,
                token_res=None,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out

    def eval_bpp(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)
            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vqgan.quantize(h_vqgan)

            x_dino = self.dino_input_transform(x)
            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan_ctx_down = self.mp12_vqgan_down(h_vqgan_ctx)

            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
            y = self.f_a(h)
            y_out = self.latent_codec(y, h_vqgan_ctx_down)

        return {
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_mmseg(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)

            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vqgan.quantize(h_vqgan)

            x_dino = self.dino_input_transform(x)
            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan_ctx_down = self.mp12_vqgan_down(h_vqgan_ctx)

            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
            y = self.f_a(h)
            y_out = self.latent_codec(y, h_vqgan_ctx_down)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            h_hat = self.mp12_dec(torch.cat([h_hat, h_vqgan_ctx], dim=1))
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            h_dino_hat = self.post_vit_blocks(_h_hat)

            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=n,
                token_res=(LH, LW),
                reshape=True,
                return_class_token=False,
                norm=True,
            )

        return feat_out


@register_model("DinoL41")
class DinoL41(Elic2022Official):
    def __init__(self, N=192, M=256, D_DINO=384, groups=None, **kwargs):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = -4

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        self.num_out_layers = 4

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)
            o_dino = self.dino.feature_output(
                h_dino,
                slot=self.slot,
                n=1,
                norm=False,
            )[0]

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        o_dino_hat = self.dino.feature_output(
            h_dino_hat,
            slot=self.slot,
            n=1,
            norm=False,
        )[0]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)
            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("DinoL11")
class DinoL11(Elic2022Official):
    def __init__(self, N=192, M=256, D_DINO=384, groups=None, **kwargs):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = None

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        self.num_out_layers = 1

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_dino_hat": h_dino_hat,
            "h_dino": h_dino.clone(),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=1):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)
            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("DinoL11Norm")
class DinoL11Norm(Elic2022Official):
    def __init__(self, N=192, M=256, D_DINO=384, groups=None, **kwargs):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.slot = None

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        self.num_out_layers = 1

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)

        # feat: (B, 257, 384)
        h_dino = h_dino.clone()
        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_dino_hat": self.dino.feature_model.norm(h_dino_hat),
            "h_dino": self.dino.feature_model.norm(h_dino.clone()),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=1):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            _h_hat = torch.cat(
                [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            )
            feat_hat = self.post_vit_blocks(_h_hat)
            feat_out = self.dino.feature_endpart(
                feat_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("VqganDinoCtx")
class VqganDinoCtx(Elic2022Official):
    def __init__(
        self,
        N=192,
        M=320,
        D=256,
        D_DINO=384,
        D_VQGAN=256,
        n_embed=1024,
        embed_dim=256,
        groups=None,
        **kwargs,
    ):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups, **kwargs)

        self.D = D
        self.D_DINO = D_DINO
        self.D_VQGAN = D_VQGAN
        self.slot = -4

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.dino_pre_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.dino_post_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.dino_post_registers = nn.Parameter(
            torch.randn(1, D_DINO), requires_grad=True
        )

        # self.f_a1 = conv(P, P, kernel_size=3, stride=1)
        self.f_a = nn.Sequential(
            DepthConvBlock(D_DINO + D_VQGAN, D),
            DepthConvBlock(D, D),
            DepthConvBlock(D, D),
        )
        self.f_down = conv(D, M, kernel_size=3, stride=2)

        self.f_up = SubpelConv2x(M, D, kernel_size=3, padding=1)
        self.f_s = nn.Sequential(
            DepthConvBlock(D + D_VQGAN, D),
            DepthConvBlock(D, D),
            DepthConvBlock(D, D),
        )
        self.f_expand = conv(D, D_DINO + D_VQGAN, kernel_size=1, stride=1)

        self.vqgan_post_convs = nn.Sequential(
            DepthConvBlock(D_VQGAN, D_VQGAN),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None

        # self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x, quant_step=1.0):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vq_model.quantize(h_vqgan)

        h_vqgan = h_vqgan.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        h_dino = h_dino.clone()
        h = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
        h = torch.cat([h, h_vqgan_ctx], dim=1)
        h = self.f_a(h)
        # h = h * quant_step
        y = self.f_down(h)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        h_hat = self.f_up(y_hat)

        h_hat = torch.cat([h_hat, h_vqgan_ctx], dim=1)
        h_hat = self.f_s(h_hat)
        # h_hat = h_hat * quant_step
        h_hat = self.f_expand(h_hat)

        h_dino_hat, h_vqgan_hat = h_hat.split([self.D_DINO, self.D_VQGAN], dim=1)
        h_dino_hat = rearrange(h_dino_hat, "B C H W -> B (H W) C")
        h_dino_hat = torch.cat(
            [self.dino_post_registers.expand(x.shape[0], -1, -1), h_dino_hat], dim=1
        )
        h_dino_hat = self.dino_post_blocks(h_dino_hat)

        # h_vqgan_hat = self.vqgan_post_convs(h_vqgan_hat)
        # with torch.inference_mode():
        #     _h = self.vq_model.decoder(self.vq_model.post_quant_conv(h_vqgan_hat))
        #     x_hat = (_h + 1) / 2

        return {
            "h_dino": h_dino,
            "h_dino_hat": h_dino_hat,
            # "h_vqgan": h_vqgan,
            # "h_vqgan_hat": h_vqgan_hat,
            # "h_vqgan_ctx": h_vqgan_ctx,
            # "x_hat": x_hat.clone(),
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vq_model.quantize(h_vqgan)
            # print(x.shape, x_dino.shape, h_dino.shape, h_vqgan_ctx.shape)
            # torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 224, 224]) torch.Size([1, 257, 384]) torch.Size([1, 256, 16, 16])

            h = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            h = torch.cat([h, h_vqgan_ctx], dim=1)
            h = self.f_a(h)
            # h = h * quant_step
            y = self.f_down(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]
            h_hat = self.f_up(y_hat)

            h_hat = torch.cat([h_hat, h_vqgan_ctx], dim=1)
            h_hat = self.f_s(h_hat)
            # h_hat = h_hat * quant_step
            h_hat = self.f_expand(h_hat)

            h_dino_hat, h_vqgan_hat = h_hat.split([self.D_DINO, self.D_VQGAN], dim=1)
            h_dino_hat = rearrange(h_dino_hat, "B C H W -> B (H W) C")
            h_dino_hat = torch.cat(
                [self.dino_post_registers.expand(x.shape[0], -1, -1), h_dino_hat], dim=1
            )
            h_dino_hat = self.dino_post_blocks(h_dino_hat)

            # h_dino_hat[:, 0, :] = h_dino[:, 0, :]
            # dino_loss = F.mse_loss(h_dino_hat.contiguous(), h_dino.contiguous())

            # h_dino_hat = h_dino + torch.randn_like(h_dino) * 0.076
            # dino_loss = F.mse_loss(h_dino_hat.contiguous(), h_dino.contiguous())
            # print("dino loss:", dino_loss)

            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out

    def ctx_for_elic(self, x):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vq_model.quantize(h_vqgan)
            # print(x.shape, x_dino.shape, h_dino.shape, h_vqgan_ctx.shape)
            # torch.Size([1, 3, 256, 256]) torch.Size([1, 3, 224, 224]) torch.Size([1, 257, 384]) torch.Size([1, 256, 16, 16])

            h = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=16, W=16)
            h = torch.cat([h, h_vqgan_ctx], dim=1)
            h = self.f_a(h)
            # h = h * quant_step
            y = self.f_down(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            return y_hat, h_vqgan_ctx


@register_model("VqganDinoShare")
class VqganDinoShare(Elic2022Official):
    def __init__(
        self,
        N=192,
        M=320,
        D=256,
        D_DINO=384,
        D_VQGAN=256,
        n_embed=1024,
        embed_dim=256,
        groups=None,
        **kwargs,
    ):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups, **kwargs)

        self.D = D
        self.D_DINO = D_DINO
        self.D_VQGAN = D_VQGAN
        self.slot = -4
        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.dino_pre_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.dino_post_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.dino_post_registers = nn.Parameter(
            torch.randn(1, D_DINO), requires_grad=True
        )

        # self.f_a1 = conv(P, P, kernel_size=3, stride=1)
        # self.f_a = nn.Sequential(
        #     DepthConvBlock(D_DINO+D_VQGAN, D),
        #     DepthConvBlock(D, D),
        #     DepthConvBlock(D, D),
        # )
        # self.f_down = conv(D, M, kernel_size=3, stride=2)

        # self.f_up = SubpelConv2x(M, D, kernel_size=3, padding=1)
        # self.f_s = nn.Sequential(
        #     DepthConvBlock(D+D_VQGAN, D),
        #     DepthConvBlock(D, D),
        #     DepthConvBlock(D, D),
        # )
        # self.f_expand = conv(D, D_DINO+D_VQGAN, kernel_size=1, stride=1)

        # self.vqgan_post_convs = nn.Sequential(
        #     DepthConvBlock(D_VQGAN, D_VQGAN),
        #     conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        # )

        self.g_a = None
        self.g_s = None
        # self.latent_codec = None
        self.dino_bind_embed = nn.Embedding(n_embed, D_DINO)
        self.dino_bind_embed.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

        # self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x, quant_step=1.0):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
            B, _, _, _ = h_vqgan_ctx.shape
            idxs = idxs.view(B, -1)

        h_vqgan = h_vqgan.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        h_dino = h_dino.clone()
        h_share = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        # h_share = rearrange(h_share, "B (H W) C -> B C H W", H=16, W=16)

        h_share_hat = self.dino_bind_embed(idxs.clone())
        # h_res = h - h_share
        # y = self.f_down(h_res)
        # y_out = self.latent_codec(y)
        # y_hat = y_out["y_hat"]
        # h_res_hat = self.f_up(y_hat)
        # h_dino_hat = h_share + h_res_hat

        # h_dino_hat = rearrange(h_share_hat, "B C H W -> B (H W) C")
        h_dino_hat = torch.cat(
            [self.dino_post_registers.expand(x.shape[0], -1, -1), h_share_hat], dim=1
        )
        h_dino_hat = self.dino_post_blocks(h_dino_hat)

        # h_vqgan_hat = self.vqgan_post_convs(h_vqgan_hat)
        # with torch.inference_mode():
        #     _h = self.vq_model.decoder(self.vq_model.post_quant_conv(h_vqgan_ctx))
        #     x_hat = (_h + 1) / 2

        return {
            "h_share": h_share,
            "h_share_hat": h_share_hat,
            "h_dino": h_dino,
            "h_dino_hat": h_dino_hat,
            # "h_vqgan": h_vqgan,
            # "h_vqgan_hat": h_vqgan_ctx,
            # "h_vqgan_ctx": h_vqgan_ctx,
            "x_hat": x,
            # "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
            B, _, _, _ = h_vqgan_ctx.shape
            idxs = idxs.view(B, -1)

            h_share = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h_share_hat = self.dino_bind_embed(idxs.clone())
            h_dino_hat = torch.cat(
                [self.dino_post_registers.expand(x.shape[0], -1, -1), h_share_hat],
                dim=1,
            )
            h_dino_hat = self.dino_post_blocks(h_dino_hat)

            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("VqganDinoShareSupp")
class VqganDinoShareSupp(Elic2022Official):
    def __init__(
        self,
        N=192,
        M=320,
        D=256,
        D_DINO=384,
        D_VQGAN=256,
        n_embed=1024,
        embed_dim=256,
        groups=None,
        **kwargs,
    ):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups, **kwargs)

        self.D = D
        self.D_DINO = D_DINO
        self.D_VQGAN = D_VQGAN
        self.slot = -4
        self.n_embed = n_embed
        self.embed_dim = embed_dim

        self.dino_input_transform = transforms.Compose(
            [
                transforms.CenterCrop(224),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(model_size="small", img_size=224, patch_size=14, dynamic_size=False, n_last_blocks=4, autocast_ctx=torch.float)
        self.dino_pre_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.dino_post_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.dino_post_registers = nn.Parameter(
            torch.randn(1, D_DINO), requires_grad=True
        )

        self.f_a = nn.Sequential(
            conv(D_DINO, M, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(M, M, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(M, D_DINO, kernel_size=3, stride=1),
        )

        self.g_a = None
        self.g_s = None
        # self.latent_codec = None
        self.dino_bind_embed = nn.Embedding(n_embed, D_DINO)
        self.dino_bind_embed.weight.data.uniform_(-1.0 / n_embed, 1.0 / n_embed)

        # self.conv = nn.Conv2d(3, 3, kernel_size=3, stride=1, padding=1)

    @classmethod
    def from_state_dict(cls, state_dict, strict=True):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def forward(self, x, quant_step=1.0):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
            B, _, _, _ = h_vqgan_ctx.shape
            idxs = idxs.view(B, -1)

        h_vqgan = h_vqgan.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        h_dino = h_dino.clone()
        h_share = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        # h_share = rearrange(h_share, "B (H W) C -> B C H W", H=16, W=16)

        h_share_vq = self.dino_bind_embed(idxs.clone())
        h_share_res = h_share - h_share_vq
        h_share_res = rearrange(h_share_res, "B (H W) C -> B C H W", H=16, W=16)
        y = self.f_a(h_share_res)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        h_share_res_sq = self.f_s(y_hat)
        h_share_res_sq = rearrange(h_share_res_sq, "B C H W -> B (H W) C", H=16, W=16)
        h_share_hat = h_share_vq + h_share_res_sq

        # h_dino_hat = rearrange(h_share_hat, "B C H W -> B (H W) C")
        h_dino_hat = torch.cat(
            [self.dino_post_registers.expand(x.shape[0], -1, -1), h_share_hat], dim=1
        )
        h_dino_hat = self.dino_post_blocks(h_dino_hat)

        # h_vqgan_hat = self.vqgan_post_convs(h_vqgan_hat)
        # with torch.inference_mode():
        #     _h = self.vq_model.decoder(self.vq_model.post_quant_conv(h_vqgan_ctx))
        #     x_hat = (_h + 1) / 2

        return {
            "h_share": h_share,
            "h_share_vq": h_share_vq,
            "h_dino": h_dino,
            "h_dino_hat": h_dino_hat,
            # "h_vqgan": h_vqgan,
            # "h_vqgan_hat": h_vqgan_ctx,
            # "h_vqgan_ctx": h_vqgan_ctx,
            "x_hat": x,
            "likelihoods": y_out["likelihoods"],
        }

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x_dino = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
            B, _, _, _ = h_vqgan_ctx.shape
            idxs = idxs.view(B, -1)

            h_share = self.dino_pre_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h_share_vq = self.dino_bind_embed(idxs.clone())
            h_share_res = h_share - h_share_vq
            h_share_res = rearrange(h_share_res, "B (H W) C -> B C H W", H=16, W=16)
            y = self.f_a(h_share_res)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]
            h_share_res_sq = self.f_s(y_hat)
            h_share_res_sq = rearrange(
                h_share_res_sq, "B C H W -> B (H W) C", H=16, W=16
            )
            h_share_hat = h_share_vq + h_share_res_sq

            # h_dino_hat = rearrange(h_share_hat, "B C H W -> B (H W) C")
            h_dino_hat = torch.cat(
                [self.dino_post_registers.expand(x.shape[0], -1, -1), h_share_hat],
                dim=1,
            )
            h_dino_hat = self.dino_post_blocks(h_dino_hat)

            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=n,
                reshape=False,
                return_class_token=True,
                norm=True,
            )

        return feat_out


@register_model("VqganDinoElicS3")
class VqganDinoElicS3(SimpleVAECompressionModel):
    def __init__(self, N=192, M=320, D_DINO=120, D_VQGAN=256, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a1 = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
        )
        self.g_a2 = nn.Sequential(
            # insert
            conv(N + D_VQGAN, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.vqgan_up = deconv(D_VQGAN, D_VQGAN, kernel_size=5, stride=2)
        # self.dino_up = nn.Sequential(
        #     deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        #     nn.ReLU(inplace=True),
        #     deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        # )

        # self.f_a = nn.Sequential(
        #     DepthConvBlock(M+D_DINO+D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        #     # conv(M, M, kernel_size=1, stride=1)
        # )

        # self.f_s = nn.Sequential(
        #     DepthConvBlock(M + D_DINO + D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        # )

        self.g_s1 = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
        )
        self.g_s2 = nn.Sequential(
            # insert
            conv(N + D_VQGAN, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def forward(self, x):
        with torch.inference_mode():
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
        vqgan_ctx = h_vqgan_ctx.clone()
        vqgan_ctx = self.vqgan_up(vqgan_ctx)

        y = self.g_a1(x)
        y = torch.cat([y, vqgan_ctx], dim=1)
        y = self.g_a2(y)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        y_hat = self.g_s1(y_hat)
        y_hat = torch.cat([y_hat, vqgan_ctx], dim=1)
        x_hat = self.g_s2(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


@register_model("VqganDinoElicS4")
class VqganDinoElicS4(SimpleVAECompressionModel):
    def __init__(self, N=192, M=320, D_DINO=120, D_VQGAN=256, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a1 = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
        )
        self.g_a2 = nn.Sequential(
            # insert
            conv(N + D_VQGAN, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.vqgan_up = deconv(D_VQGAN, D_VQGAN, kernel_size=5, stride=2)
        # self.dino_up = nn.Sequential(
        #     deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        #     nn.ReLU(inplace=True),
        #     deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        # )

        # self.f_a = nn.Sequential(
        #     DepthConvBlock(M+D_DINO+D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        #     # conv(M, M, kernel_size=1, stride=1)
        # )

        # self.f_s = nn.Sequential(
        #     DepthConvBlock(M + D_DINO + D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        # )

        self.g_s1 = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
        )
        self.g_s2 = nn.Sequential(
            # insert
            conv(N + D_VQGAN, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )



        h_a = HyperEncoderWithCtx(N, M, D_VQGAN)
        h_s = HyperDecoderWithCtx(N, M, D_VQGAN)

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodecWithCtx(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodecWithCtx(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def forward(self, x):
        with torch.inference_mode():
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
        vqgan_ctx = h_vqgan_ctx.clone()
        vqgan_ctx_up = self.vqgan_up(vqgan_ctx)

        y = self.g_a1(x)
        y = torch.cat([y, vqgan_ctx_up], dim=1)
        y = self.g_a2(y)
        y_out = self.latent_codec(y, vqgan_ctx)
        y_hat = y_out["y_hat"]
        y_hat = self.g_s1(y_hat)
        y_hat = torch.cat([y_hat, vqgan_ctx_up], dim=1)
        x_hat = self.g_s2(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


# use vqgan_ctx in hyperprior decoder, load backbone
@register_model("VqganDinoElicV0")
class VqganDinoElicV0(SimpleVAECompressionModel):
    def __init__(self, N=192, M=320, D_DINO=120, D_VQGAN=256, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        # self.vqgan_up = deconv(D_VQGAN, D_VQGAN, kernel_size=5, stride=2)
        # self.dino_up = nn.Sequential(
        #     deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        #     nn.ReLU(inplace=True),
        #     deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        # )

        # self.f_a = nn.Sequential(
        #     DepthConvBlock(M+D_DINO+D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        #     # conv(M, M, kernel_size=1, stride=1)
        # )

        # self.f_s = nn.Sequential(
        #     DepthConvBlock(M + D_DINO + D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        # )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        class HyperEncoderWithCtx(nn.Module):
            def __init__(self, N, M, D_VQGAN):
                super().__init__()
                self.h_a = nn.Sequential(
                    conv(M + D_VQGAN, N, kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N, N, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(N, N, kernel_size=5, stride=2),
                )

            def forward(self, x, ctx):
                return self.h_a(torch.cat([x, ctx], dim=1))

        class HyperDecoderWithCtx(nn.Module):
            def __init__(self, N, M, D_VQGAN):
                super().__init__()
                self.h_s = nn.Sequential(
                    deconv(N, N, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(N, N * 3 // 2, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
                )
                self.fusion = nn.Sequential(
                    conv(N * 2 + D_VQGAN, N * 2, kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N * 2, N * 2, kernel_size=3, stride=1),
                )

            def forward(self, x, ctx):
                h = self.h_s(x)
                h = self.fusion(torch.cat([h, ctx], dim=1))
                return h

        h_a = HyperEncoderWithCtx(N, M, D_VQGAN)
        h_s = HyperDecoderWithCtx(N, M, D_VQGAN)

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodecWithCtx(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodecWithCtx(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def forward(self, x):
        # with torch.inference_mode():
        #     h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
        #     h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
        # _h = self.vq_model.decoder(self.vq_model.post_quant_conv(h_vqgan_ctx))
        # x_hat = (_h + 1) / 2
        # vqgan_ctx = h_vqgan_ctx.clone()
        y = self.g_a(x)
        B, _, H, W = y.shape
        vqgan_ctx = torch.ones(B, 256, H, W).to(y.device)
        y_out = self.latent_codec(y, vqgan_ctx)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net


# use vqgan_ctx in hyperprior decoder, load backbone
@register_model("VqganDinoElicV1")
class VqganDinoElicV1(SimpleVAECompressionModel):
    def __init__(self, N=192, M=320, D_DINO=120, D_VQGAN=256, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.g_s = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        class HyperEncoderWithCtx(nn.Module):
            def __init__(self, N, M, D_VQGAN):
                super().__init__()
                self.h_a = nn.Sequential(
                    conv(M + D_VQGAN, N, kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N, N, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    conv(N, N, kernel_size=5, stride=2),
                )

            def forward(self, x, ctx):
                return self.h_a(torch.cat([x, ctx], dim=1))

        class HyperDecoderWithCtx(nn.Module):
            def __init__(self, N, M, D_VQGAN):
                super().__init__()
                self.h_s = nn.Sequential(
                    deconv(N, N, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(N, N * 3 // 2, kernel_size=5, stride=2),
                    nn.ReLU(inplace=True),
                    deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
                )
                self.fusion = nn.Sequential(
                    conv(N * 2 + D_VQGAN, N * 2, kernel_size=3, stride=1),
                    nn.ReLU(inplace=True),
                    conv(N * 2, N * 2, kernel_size=3, stride=1),
                )

            def forward(self, x, ctx):
                h = self.h_s(x)
                h = self.fusion(torch.cat([h, ctx], dim=1))
                return h

        h_a = HyperEncoderWithCtx(N, M, D_VQGAN)
        h_s = HyperDecoderWithCtx(N, M, D_VQGAN)

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodecWithCtx(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodecWithCtx(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def forward(self, x):
        with torch.inference_mode():
            h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
            # _h = self.vq_model.decoder(self.vq_model.post_quant_conv(h_vqgan_ctx))
            # x_hat = (_h + 1) / 2

        vqgan_ctx = h_vqgan_ctx.clone()
        y = self.g_a(x)
        y_out = self.latent_codec(y, vqgan_ctx)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    @classmethod
    def from_state_dict(cls, state_dict, strict=False):
        """Return a new model instance from `state_dict`."""
        net = cls()
        net.load_state_dict(state_dict, strict=strict)
        return net

    def compress(self, x):
        h_vqgan = self.vq_model.quant_conv(self.vq_model.encoder(2 * x - 1))
        vqgan_ctx, _, (_, _, idxs) = self.vq_model.quantize(h_vqgan)
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y, vqgan_ctx)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
        }


@register_model("VqganDinoElicS8")
class VqganDinoElicS8(SimpleVAECompressionModel):
    """ELIC 2022; uneven channel groups with checkerboard spatial context.

    Context model from [He2022].
    Based on modified attention model architecture from [Cheng2020].

    [He2022]: `"ELIC: Efficient Learned Image Compression with
    Unevenly Grouped Space-Channel Contextual Adaptive Coding"
    <https://arxiv.org/abs/2203.10886>`_, by Dailan He, Ziming Yang,
    Weikun Peng, Rui Ma, Hongwei Qin, and Yan Wang, CVPR 2022.

    [Cheng2020]: `"Learned Image Compression with Discretized Gaussian
    Mixture Likelihoods and Attention Modules"
    <https://arxiv.org/abs/2001.01568>`_, by Zhengxue Cheng, Heming Sun,
    Masaru Takeuchi, and Jiro Katto, CVPR 2020.

    Args:
        N (int): Number of main network channels
        M (int): Number of latent space channels
        groups (list[int]): Number of channels in each channel group
    """

    def __init__(self, N=192, M=320, D_DINO=120, D_VQGAN=256, groups=None, **kwargs):
        super().__init__(**kwargs)

        if groups is None:
            groups = [16, 16, 32, 64, M - 128]

        self.groups = list(groups)
        assert sum(self.groups) == M

        self.g_a1 = nn.Sequential(
            conv(3, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            AttentionBlock(N),
            conv(N, N, kernel_size=5, stride=2),
        )
        self.g_a2 = nn.Sequential(
            # insert
            conv(N + D_DINO + D_VQGAN, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            conv(N, M, kernel_size=5, stride=2),
            AttentionBlock(M),
        )

        self.vqgan_up = deconv(D_VQGAN, D_VQGAN, kernel_size=5, stride=2)
        self.dino_up = nn.Sequential(
            deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(D_DINO, D_DINO, kernel_size=5, stride=2),
        )

        # self.f_a = nn.Sequential(
        #     DepthConvBlock(M+D_DINO+D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        #     # conv(M, M, kernel_size=1, stride=1)
        # )

        # self.f_s = nn.Sequential(
        #     DepthConvBlock(M + D_DINO + D_VQGAN, M),
        #     DepthConvBlock(M, M),
        #     DepthConvBlock(M, M),
        # )

        self.g_s1 = nn.Sequential(
            AttentionBlock(M),
            deconv(M, N, kernel_size=5, stride=2),
        )
        self.g_s2 = nn.Sequential(
            # insert
            conv(N + D_DINO + D_VQGAN, N, kernel_size=3, stride=1),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            AttentionBlock(N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, N, kernel_size=5, stride=2),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            ResidualBottleneckBlock(N, N),
            deconv(N, 3, kernel_size=5, stride=2),
        )

        h_a = nn.Sequential(
            conv(M, N, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(N, N, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(N, N, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N, N * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(N * 3 // 2, N * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": sequential_channel_ramp(
                sum(self.groups[:k]),
                self.groups[k] * 2,
                min_ch=N,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(1, len(self.groups))
        }

        # In [He2022], this is labeled "g_sp^(k)".
        spatial_context = [
            CheckerboardMaskedConv2d(
                self.groups[k],
                self.groups[k] * 2,
                kernel_size=5,
                stride=1,
                padding=2,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled "Param Aggregation".
        param_aggregation = [
            sequential_channel_ramp(
                # Input: spatial context, channel context, and hyper params.
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + N * 2,
                self.groups[k] * 2,
                min_ch=N * 2,
                num_layers=3,
                make_layer=nn.Conv2d,
                make_act=lambda: nn.ReLU(inplace=True),
                kernel_size=1,
                stride=1,
                padding=0,
            )
            for k in range(len(self.groups))
        ]

        # In [He2022], this is labeled the space-channel context model (SCCTX).
        # The side params and channel context params are computed externally.
        scctx_latent_codec = {
            f"y{k}": CheckerboardLatentCodec(
                latent_codec={
                    "y": GaussianConditionalLatentCodec(quantizer="ste"),
                },
                context_prediction=spatial_context[k],
                entropy_parameters=param_aggregation[k],
            )
            for k in range(len(self.groups))
        }

        # [He2022] uses a "hyperprior" architecture, which reconstructs y using z.
        self.latent_codec = HyperpriorLatentCodec(
            latent_codec={
                # Channel groups with space-channel context model (SCCTX):
                "y": ChannelGroupsLatentCodec(
                    groups=self.groups,
                    channel_context=channel_context,
                    latent_codec=scctx_latent_codec,
                ),
                # Side information branch containing z:
                "hyper": HyperLatentCodec(
                    entropy_bottleneck=EntropyBottleneck(N),
                    h_a=h_a,
                    h_s=h_s,
                    quantizer="ste",
                ),
            },
        )

    def forward(self, x):
        with torch.inference_mode():
            dino_ctx, vqgan_ctx = self.vqgan_dino_ctx.ctx_for_elic(x)
        dino_ctx = dino_ctx.clone()
        vqgan_ctx = vqgan_ctx.clone()
        dino_ctx = self.dino_up(dino_ctx)
        vqgan_ctx = self.vqgan_up(vqgan_ctx)

        y = self.g_a1(x)
        y = torch.cat([y, dino_ctx, vqgan_ctx], dim=1)
        y = self.g_a2(y)
        y_out = self.latent_codec(y)
        y_hat = y_out["y_hat"]
        y_hat = self.g_s1(y_hat)
        y_hat = torch.cat([y_hat, dino_ctx, vqgan_ctx], dim=1)
        x_hat = self.g_s2(y_hat)
        return {
            "x_hat": x_hat,
            "likelihoods": y_out["likelihoods"],
        }

    def compress(self, x):
        y = self.g_a(x)
        outputs = self.latent_codec.compress(y)
        return outputs

    def decompress(self, *args, **kwargs):
        y_out = self.latent_codec.decompress(*args, **kwargs)
        y_hat = y_out["y_hat"]
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return {
            "x_hat": x_hat,
        }

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net
