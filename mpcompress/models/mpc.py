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
import numpy as np
from torch import Tensor

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from mpcompress.latent_codecs.ctx_hyper import (
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

from compressai.models.base import SimpleVAECompressionModel, CompressionModel
from compressai.models.utils import conv, deconv
from compressai.models.sensetime import ResidualBottleneckBlock
from einops import rearrange
import timm
import torchvision.transforms as transforms

# import timm.models.vision_transformer.Block as Block
from compressai.models.sensetime import Elic2022Official
from mpcompress.layers.vit import Block
from mpcompress.layers.layers import DepthConvBlock, SubpelConv2x
import torch.nn.functional as F

from mpcompress.backbone.vqgan.vq_model import VQModel
from mpcompress.utils.coder import encode_uniform_to_bits, decode_uniform_from_bits
from mpcompress.utils.utils import extract_shapes


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
    x = x[:, :, crop_top : crop_top + img_h, crop_left : crop_left + img_w]
    return x


class Dinov2TimmBackbone(nn.Module):
    def __init__(
        self,
        model_size="small",
        img_size=256,
        patch_size=16,
        dynamic_size=False,
        n_last_blocks=4,
        autocast_ctx=torch.float,
    ):
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
            dynamic_img_size=self.dynamic_size,
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


@register_model("MPC2")
class MPC2(Elic2022Official):
    def __init__(
        self,
        N=192,
        M=256,
        D_DINO=384,
        groups=None,
        dino_config=None,
        **kwargs,
    ):
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        super().__init__(N=N, M=M, groups=groups)

        self.N = N
        self.M = M
        self.D_DINO = D_DINO

        self.slot = -4
        self.patch_size = dino_config.get("patch_size", 16)

        self.dino_input_transform = transforms.Compose(
            [
                # transforms.CenterCrop(img_size),
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(**dino_config)
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
        raise NotImplementedError
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
            x = center_padding(x, patch_size=128)
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
            B, C, H, W = x.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            x_dino = self.dino_input_transform(x)
            feat = self.dino.feature_startpart(x_dino, slot=self.slot)

            h = self.pre_vit_blocks(feat)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
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
            x = center_padding(x, patch_size=128)
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


@register_model("MPC2_VQGAN")
class MPC2_VQGAN(MPC2):
    def __init__(self, D_DINO, D_VQGAN, vqgan_config, **kwargs):
        super().__init__(**kwargs)

        M = self.M

        self.vqgan = VQModel(**vqgan_config)
        self.vqgan.to("cuda")

        self.dec1 = nn.Sequential(
            deconv(M, M, kernel_size=5, stride=2),  # x32 -> x16
            nn.ReLU(inplace=True),
            conv(M, D_VQGAN, kernel_size=3, stride=1),  # x16 -> x16
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),  # x16 -> x16
        )

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            # h_vqgan_ctx, _, _ = self.vqgan.quantize(h_vqgan)

            x_norm = self.dino_input_transform(x)
            h_dino = self.dino.feature_startpart(x_norm, slot=self.slot)
            # o_dino = self.dino.feature_output(
            #     h_dino,
            #     slot=self.slot,
            #     n=1,
            #     norm=True,
            # )[0]

            # feat: (B, 257, 384)
            # h_dino = h_dino.clone()
            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            token_H = x.shape[2] // self.patch_size
            token_W = x.shape[3] // self.patch_size
            h = rearrange(h, "B (H W) C -> B C H W", H=token_H, W=token_W)
            y = self.f_a(h)
            y_out = self.latent_codec(y)
            y_hat = y_out["y_hat"]

            # h_hat = self.f_s(y_hat)
            # _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
            # _h_hat = torch.cat(
            #     [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_hat], dim=1
            # )
            # h_dino_hat = self.post_vit_blocks(_h_hat)

            # o_dino_hat = self.dino.feature_output(
            #     h_dino_hat,
            #     slot=self.slot,
            #     n=1,
            #     norm=True,
            # )[0]

        h_vqgan_hat = self.dec1(y_hat.clone())

        if not self.training:
            x_hat = self.vqgan.decoder(self.vqgan.post_quant_conv(h_vqgan_hat))
            x_hat = (x_hat + 1) / 2
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan.clone(),
            "h_vqgan_hat": h_vqgan_hat,
            # "h_dino_hat": o_dino_hat,
            # "h_dino": o_dino.clone(),
            "likelihoods": y_out["likelihoods"],
            "x_hat": x_hat,
        }


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
        self.dino = Dinov2TimmBackbone(
            model_size="small",
            img_size=256,
            patch_size=patch_size,
            dynamic_size=True,
            n_last_blocks=4,
            autocast_ctx=torch.float,
        )
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.mp12_enc = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_DINO, kernel_size=3, stride=1),
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
            conv(D_DINO + D_VQGAN, D_DINO, kernel_size=3, stride=1),
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
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("dino.") and not k.startswith("vqgan.")
        }
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


@register_model("MPC12_VQGAN")
class MPC12_VQGAN(CompressionModel):
    def __init__(
        self,
        N=192,
        M=256,
        D=256,
        D_DINO=384,
        D_VQGAN=256,
        groups=None,
        dino_config={},
        vqgan_config={},
        **kwargs,
    ):
        super().__init__()
        if groups is None:
            groups = [16, 16, 32, 64, M - 128]
        elif isinstance(groups, int):
            groups = [groups] * (M // groups)
        self.groups = groups

        self.slot = -4
        self.patch_size = dino_config.get("patch_size", 16)

        # self.g_a = None
        # self.g_s = None

        self.vqgan = VQModel(**vqgan_config)
        self.dino_input_transform = transforms.Compose(
            [
                transforms.Normalize(
                    mean=[0.4850, 0.4560, 0.4060], std=[0.2290, 0.2240, 0.2250]
                ),
            ]
        )
        self.dino = Dinov2TimmBackbone(**dino_config)
        self.post_reg_tokens = nn.Parameter(torch.zeros(1, D_DINO), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=D_DINO, num_heads=D_DINO // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.mp12_enc = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_DINO, kernel_size=3, stride=1),
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

        self.dec1 = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

        self.dec2 = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_DINO, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_DINO, D_DINO, kernel_size=3, stride=1),
        )

        self.vqgan_down = conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=2)

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
        state_dict = {
            k: v
            for k, v in state_dict.items()
            if not k.startswith("dino.") and not k.startswith("vqgan.")
        }
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
        h_vqgan_ctx_down = self.vqgan_down(h_vqgan_ctx)

        h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
        token_H = x.shape[2] // self.patch_size
        token_W = x.shape[3] // self.patch_size
        h = rearrange(h, "B (H W) C -> B C H W", H=token_H, W=token_W)
        h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
        y = self.f_a(h)
        y_out = self.latent_codec(y, h_vqgan_ctx_down)
        # test here
        # outputs = self.latent_codec.compress(y, h_vqgan_ctx_down)
        # print(extract_shapes(outputs))
        # raise
        y_hat = y_out["y_hat"]

        h_hat = self.f_s(y_hat)
        # after decompress h_hat, mainly decode dino, auxillary decode vqgan (using detach)
        h_vqgan_hat = self.dec1(torch.cat([h_hat.detach(), h_vqgan_ctx], dim=1))
        if not self.training:
            with torch.no_grad():
                _h_vqgan_hat = self.vqgan.decoder(
                    self.vqgan.post_quant_conv(h_vqgan_hat)
                )
                x_hat = (_h_vqgan_hat + 1) / 2
        else:
            x_hat = None

        _h_dion_hat = self.dec2(torch.cat([h_hat, h_vqgan_ctx], dim=1))
        _h_dion_hat = rearrange(_h_dion_hat, "B C H W -> B (H W) C")
        _h_dion_hat = torch.cat(
            [self.post_reg_tokens.expand(x.shape[0], -1, -1), _h_dion_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_dion_hat)

        o_dino_hat = self.dino.feature_output(
            h_dino_hat,
            slot=self.slot,
            n=1,
            norm=True,
        )[0]

        return {
            "h_vqgan": h_vqgan.clone(),
            "h_vqgan_hat": h_vqgan_hat,
            "x_hat": x_hat,
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

    def compress(self, x):
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)

            # compress vqgan
            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            h_vqgan_ctx, _, (_, _, vq_idxs) = self.vqgan.quantize(h_vqgan)

            alphabet_size = self.vqgan.quantize.embedding.weight.size()[0]
            vqgan_string = encode_uniform_to_bits(vq_idxs, alphabet_size)

            x_dino = self.dino_input_transform(x)
            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan_ctx_down = self.vqgan_down(h_vqgan_ctx)

            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
            y = self.f_a(h)
            outputs = self.latent_codec.compress(y, h_vqgan_ctx_down)

        layer_outputs = {
            "vqgan": {
                "strings": [vqgan_string],  # Form compatibility
                "shape": tuple(h_vqgan_ctx.shape[-2:]),
                "alphabet_size": alphabet_size,
            },
            "dino": {
                "strings": outputs["strings"],
                "shape": outputs["shape"],
                # "y_hat": outputs["y_hat"],
                "token_res": (LH, LW),
            }
        }
        # print(extract_shapes(layer_outputs))
        return layer_outputs

    def decompress(self, layer_outputs, return_rec1=False, return_rec2=False, return_lnclf=False, return_mmseg=False):
        results = {}
        vqgan_string = layer_outputs["vqgan"]["strings"][0]
        vqgan_shape = layer_outputs["vqgan"]["shape"]
        batch_size = vqgan_shape[0] * vqgan_shape[1]
        alphabet_size = self.vqgan.quantize.embedding.weight.size()[0]
        vq_idxs = decode_uniform_from_bits(vqgan_string, batch_size, alphabet_size)
        vq_idxs = vq_idxs.long().cuda()

        # vqgan idx to z_q
        h_vqgan_ctx = self.vqgan.quantize.embedding(vq_idxs)
        h_vqgan_ctx = rearrange(h_vqgan_ctx, "(H W) C -> 1 C H W", H=vqgan_shape[0], W=vqgan_shape[1])

        if return_rec1:
            _h_vqgan_hat = self.vqgan.decoder(self.vqgan.post_quant_conv(h_vqgan_ctx))
            x_hat = (_h_vqgan_hat + 1) / 2
            results["rec1"] = x_hat

        h_vqgan_ctx_down = self.vqgan_down(h_vqgan_ctx)

        dino_strings = layer_outputs["dino"]["strings"]
        dino_shape = layer_outputs["dino"]["shape"]
        token_res = layer_outputs["dino"]["token_res"]

        y_out = self.latent_codec.decompress(dino_strings, dino_shape, h_vqgan_ctx_down)
        y_hat = y_out["y_hat"]
        
        h_hat = self.f_s(y_hat)

        if return_rec2:
            h_vqgan_hat = self.dec1(torch.cat([h_hat.detach(), h_vqgan_ctx], dim=1))
            _h_vqgan_hat = self.vqgan.decoder(self.vqgan.post_quant_conv(h_vqgan_hat))
            x_hat = (_h_vqgan_hat + 1) / 2
            results["rec2"] = x_hat

        _h_dion_hat = self.dec2(torch.cat([h_hat, h_vqgan_ctx], dim=1))
        _h_dion_hat = rearrange(_h_dion_hat, "B C H W -> B (H W) C")
        _h_dion_hat = torch.cat(
            [self.post_reg_tokens.expand(h_hat.shape[0], -1, -1), _h_dion_hat], dim=1
        )
        h_dino_hat = self.post_vit_blocks(_h_dion_hat)

        if return_lnclf:
            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=4,
                token_res=None,
                reshape=False,
                return_class_token=True,
                norm=True,
            )
            results["lnclf"] = feat_out

        if return_mmseg:
            feat_out = self.dino.feature_endpart(
                h_dino_hat,
                slot=self.slot,
                n=4,
                token_res=token_res,
                reshape=True,
                return_class_token=False,
                norm=True,
            )
            results["mmseg"] = feat_out

        return results

    def dino_eval_linear(self, x, n=4):  # for lic training
        with torch.inference_mode():
            x = center_padding(x, patch_size=128)

            h_vqgan = self.vqgan.quant_conv(self.vqgan.encoder(2 * x - 1))
            h_vqgan_ctx, _, _ = self.vqgan.quantize(h_vqgan)

            x_dino = self.dino_input_transform(x)
            B, C, H, W = x_dino.shape  # after padding
            LH, LW = H // self.patch_size, W // self.patch_size

            h_dino = self.dino.feature_startpart(x_dino, slot=self.slot)
            h_vqgan_ctx_down = self.vqgan_down(h_vqgan_ctx)

            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
            y = self.f_a(h)
            y_out = self.latent_codec(y, h_vqgan_ctx_down)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            h_hat = self.dec2(torch.cat([h_hat, h_vqgan_ctx], dim=1))
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
            h_vqgan_ctx_down = self.vqgan_down(h_vqgan_ctx)

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
            h_vqgan_ctx_down = self.vqgan_down(h_vqgan_ctx)

            h = self.pre_vit_blocks(h_dino)[:, 1:].contiguous()  # (B, 256, 384)
            h = rearrange(h, "B (H W) C -> B C H W", H=LH, W=LW)
            h = self.mp12_enc(torch.cat([h, h_vqgan_ctx], dim=1))
            y = self.f_a(h)
            y_out = self.latent_codec(y, h_vqgan_ctx_down)
            y_hat = y_out["y_hat"]

            h_hat = self.f_s(y_hat)
            h_hat = self.dec2(torch.cat([h_hat, h_vqgan_ctx], dim=1))
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






