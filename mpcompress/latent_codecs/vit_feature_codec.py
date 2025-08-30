import torch
import torch.nn as nn

from compressai.entropy_models import EntropyBottleneck
from compressai.latent_codecs import (
    ChannelGroupsLatentCodec,
    CheckerboardLatentCodec,
    GaussianConditionalLatentCodec,
    HyperLatentCodec,
    HyperpriorLatentCodec,
)
from mpcompress.latent_codecs.hyperprior import (
    HyperLatentCodecWithCtx,
)
from compressai.layers import (
    CheckerboardMaskedConv2d,
    sequential_channel_ramp,
)
from compressai.registry import register_model

from compressai.models.base import CompressionModel
from compressai.models.utils import conv, deconv
from einops import rearrange

from mpcompress.layers.vit import Block


@register_model("VitUnionLatentCodec")
class VitUnionLatentCodec(CompressionModel):
    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        groups=16,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.post_reg_tokens = nn.Parameter(torch.zeros(1, h_dim), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = nn.Sequential(
            conv(y_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

        h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(sum(self.groups[:k]), z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(z_dim, self.groups[k] * 2, kernel_size=5, stride=1),
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
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + z_dim * 2,
                self.groups[k] * 2,
                min_ch=z_dim * 2,
                num_layers=3,
                interp="linear",
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

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodec(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodec(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )

    def forward(self, h, token_res):
        # h: vit output tensor (B,L,C)
        # can be split into 1d cls token and 2d patch tokens
        B = h.shape[0]
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        y = self.f_a(h)
        hyper_out = self.hyper_lc(y)
        y_out = self.y_lc(y, hyper_out["params"])
        y_hat = y_out["y_hat"]

        _h_hat = self.f_s(y_hat)
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(B, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_hat": h_hat,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, token_res):
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        y = self.f_a(h)
        hyper_out = self.hyper_lc.compress(y)
        y_out = self.y_lc.compress(y, hyper_out["params"])

        return {
            "strings": {"y": y_out["strings"], "z": hyper_out["strings"]},
            "shape": {"y": y_out["shape"], "z": hyper_out["shape"]},
        }

    def decompress(self, strings, shape, **kwargs):
        y_strings_ = strings["y"]
        z_strings_ = strings["z"]
        hyper_out = self.hyper_lc.decompress(z_strings_, shape["z"])
        y_out = self.y_lc.decompress(y_strings_, shape["y"], hyper_out["params"])
        h_hat = self.f_s(y_out["y_hat"])
        _h_hat = rearrange(h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(1, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)
        return {"h_hat": h_hat}


class HyperEncoderWithCtx(nn.Module):
    def __init__(self, z_dim, y_dim, ctx_dim):
        super().__init__()
        self.h_a = nn.Sequential(
            conv(y_dim + ctx_dim, z_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            conv(z_dim, z_dim, kernel_size=5, stride=2),
        )

    def forward(self, x, ctx):
        return self.h_a(torch.cat([x, ctx], dim=1))


class HyperDecoderWithCtx(nn.Module):
    def __init__(self, z_dim, y_dim, ctx_dim):
        super().__init__()
        self.h_s = nn.Sequential(
            deconv(z_dim, z_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim, z_dim * 3 // 2, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(z_dim * 3 // 2, z_dim * 2, kernel_size=3, stride=1),
        )
        self.fusion = nn.Sequential(
            conv(z_dim * 2 + ctx_dim, z_dim * 2, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(z_dim * 2, z_dim * 2, kernel_size=3, stride=1),
        )

    def forward(self, x, ctx):
        h = self.h_s(x)
        h = self.fusion(torch.cat([h, ctx], dim=1))
        return h


@register_model("VitUnionLatentCodecWithCtx")
class VitUnionLatentCodecWithCtx(CompressionModel):
    def __init__(
        self,
        h_dim=384,
        y_dim=256,
        z_dim=192,
        ctx_dim=256,
        groups=16,
        **kwargs,
    ):
        super().__init__()
        if isinstance(groups, list):
            self.groups = groups
        elif isinstance(groups, int):
            self.groups = [groups] * (y_dim // groups)
        assert sum(self.groups) == y_dim, "groups must sum to y_dim"

        self.y_dim = y_dim
        self.z_dim = z_dim

        self.post_reg_tokens = nn.Parameter(torch.zeros(1, h_dim), requires_grad=True)
        self.pre_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )
        self.post_vit_blocks = nn.Sequential(
            *[Block(dim=h_dim, num_heads=h_dim // 64, mlp_ratio=4) for _ in range(2)]
        )

        self.cond_enc = nn.Sequential(
            conv(h_dim + ctx_dim, h_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(h_dim, h_dim, kernel_size=3, stride=1),
        )
        # self.cond_dec1 = nn.Sequential(
        #     deconv(dim, dim, kernel_size=3, stride=1),
        #     nn.ReLU(inplace=True),
        #     deconv(dim, dim + ctx_dim, kernel_size=3, stride=1),
        # )

        self.cond_dec = nn.Sequential(
            conv(h_dim + ctx_dim, h_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(h_dim, h_dim, kernel_size=3, stride=1),
        )
        self.f_ctx_down = conv(ctx_dim, ctx_dim, kernel_size=3, stride=2)

        self.f_a = nn.Sequential(
            conv(h_dim, y_dim, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(y_dim, y_dim, kernel_size=5, stride=2),
        )
        self.f_s = nn.Sequential(
            deconv(y_dim, y_dim, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            deconv(y_dim, h_dim, kernel_size=3, stride=1),
        )

        h_a = HyperEncoderWithCtx(z_dim, y_dim, ctx_dim)
        h_s = HyperDecoderWithCtx(z_dim, y_dim, ctx_dim)

        # In [He2022], this is labeled "g_ch^(k)".
        channel_context = {
            f"y{k}": nn.Sequential(
                conv(sum(self.groups[:k]), z_dim, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(192, 192, kernel_size=5, stride=1),
                nn.ReLU(inplace=True),
                conv(192, self.groups[k] * 2, kernel_size=5, stride=1),
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
                self.groups[k] * 2 + (k > 0) * self.groups[k] * 2 + z_dim * 2,
                self.groups[k] * 2,
                min_ch=z_dim * 2,
                num_layers=3,
                interp="linear",
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

        # Channel groups with space-channel context model (SCCTX):
        self.y_lc = ChannelGroupsLatentCodec(
            groups=self.groups,
            channel_context=channel_context,
            latent_codec=scctx_latent_codec,
        )
        self.hyper_lc = HyperLatentCodecWithCtx(
            entropy_bottleneck=EntropyBottleneck(z_dim),
            h_a=h_a,
            h_s=h_s,
            quantizer="ste",
        )

    def forward(self, h, ctx, token_res):
        # h: vit output tensor (B,L,C)
        # can be split into 1d cls token and 2d patch tokens
        B = h.shape[0]
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h_share = self.cond_enc(torch.cat([h, ctx], dim=1))
        y = self.f_a(h_share)

        ctx_down = self.f_ctx_down(ctx)
        hyper_out = self.hyper_lc(y, ctx_down)
        y_out = self.y_lc(y, hyper_out["params"])
        y_hat = y_out["y_hat"]

        h_hat_share = self.f_s(y_hat)
        _h_hat = self.cond_dec(torch.cat([h_hat_share, ctx], dim=1))
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(B, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)

        return {
            "h_hat": h_hat,
            "h_hat_share": h_hat_share,
            "likelihoods": {
                "y": y_out["likelihoods"]["y"],
                "z": hyper_out["likelihoods"]["z"],
            },
        }

    def compress(self, h, ctx, token_res):
        h = self.pre_vit_blocks(h)[:, 1:].contiguous()
        h = rearrange(h, "B (H W) C -> B C H W", H=token_res[0], W=token_res[1])
        h_share = self.cond_enc(torch.cat([h, ctx], dim=1))
        y = self.f_a(h_share)

        ctx_down = self.f_ctx_down(ctx)

        hyper_out = self.hyper_lc.compress(y, ctx_down)
        y_out = self.y_lc.compress(y, hyper_out["params"])

        return {
            "strings": {"y": y_out["strings"], "z": hyper_out["strings"]},
            "shape": {"y": y_out["shape"], "z": hyper_out["shape"]},
            # "y_hat": y_out["y_hat"],
        }

    def decompress(self, strings, shape, ctx, **kwargs):
        y_strings_ = strings["y"]
        z_strings_ = strings["z"]
        # assert all(len(y_strings) == len(z_strings) for y_strings in y_strings_)
        ctx_down = self.f_ctx_down(ctx)
        hyper_out = self.hyper_lc.decompress(z_strings_, shape["z"], ctx_down)
        y_out = self.y_lc.decompress(y_strings_, shape["y"], hyper_out["params"])
        y_hat = y_out["y_hat"]

        h_hat_share = self.f_s(y_hat)
        _h_hat = self.cond_dec(torch.cat([h_hat_share, ctx], dim=1))
        _h_hat = rearrange(_h_hat, "B C H W -> B (H W) C")
        _h_hat = torch.cat([self.post_reg_tokens.expand(1, -1, -1), _h_hat], dim=1)
        h_hat = self.post_vit_blocks(_h_hat)
        return {"h_hat": h_hat, "h_hat_share": h_hat_share}
