import torch
import torch.nn as nn

from compressai.models.base import CompressionModel
from compressai.models.utils import conv

from mpcompress.backbone.base import Dinov2TimmBackbone, VqganBackbone
from mpcompress.token_codecs.base import UniformTokenCodec
from mpcompress.latent_codecs.vit_feature_codec import (
    VitUnionLatentCodec,
    VitUnionLatentCodecWithCtx,
    VitUnionLatentCodecCtxAsHyper,
    VbrVitUnionLatentCodec,
)
from mpcompress.backbone.base import *
from mpcompress.utils.registery import instantiate_class, register


@register("MPC_I1")
class MPC_I1(CompressionModel):  # VqganTokenUniformCodec
    def __init__(self, vqgan_config, **kwargs):
        super().__init__()
        self.vqgan = VqganBackbone(vqgan_config)
        self.vqgan_codec = UniformTokenCodec(self.vqgan.codebook_size)
        self.patch_size = 16

    def forward(self, x, **kwargs):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_out = self.vqgan_codec(vqgan_enc["tokens"])
        x_hat = self.vqgan.decode(vqgan_enc["z_q"])
        return {"likelihoods": vqgan_out["likelihoods"], "x_hat": x_hat}

    def compress(self, x, **kwargs):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_out = self.vqgan_codec.compress(vqgan_enc["tokens"])
        return vqgan_out

    def decompress(self, strings, shape, **kwargs):
        out = self.vqgan_codec.decompress(strings, shape, **kwargs)
        tokens = out["tokens"]
        z_q = self.vqgan.tokens_to_features(tokens)
        x_hat = self.vqgan.decode(z_q)
        task_feats = {"z_q": z_q, "tokens": tokens, "x_hat": x_hat}
        return task_feats


@register("MPC_I2")
class MPC_I2(CompressionModel):
    def __init__(
        self,
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        if "type" in dino_backbone:
            self.dino = instantiate_class(dino_backbone)
            self.dino_codec = instantiate_class(dino_codec)
        else:
            self.dino = Dinov2TimmBackbone(**dino_backbone)
            self.dino_codec = VitUnionLatentCodec(**dino_codec)
        self.patch_size = self.dino.patch_size

    def forward(self, x, qp=0, **kwargs):  # for lic training
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino)[-1]

        h_dino = h_dino.clone()
        dino_out = self.dino_codec(h_dino, token_res, qp=qp)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def extract_feature(self, x, **kwargs):  # for training
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            return {
                "h_dino": h_dino,
            }

    def offline_forward(self, data, device, qp=0, **kwargs):  # for lic training
        with torch.inference_mode():
            h_dino = data["h_dino"].to(device).float()
            _, _, H, W = data["x_shape"]
            token_res = (H // self.patch_size, W // self.patch_size)

            # x_uint8 = data["x_uint8"].to(device)
            # x = x_uint8 / 255.0
            # h_dino_ref = self.dino.encode(x).float()
            # print(torch.mean(torch.abs(h_dino - h_dino_ref)))
            # assert torch.allclose(h_dino, h_dino_ref), "not consistent"

            o_dino = self.dino.decode_whole(h_dino, token_res)[-1]

        h_dino = h_dino.clone()
        dino_out = self.dino_codec(h_dino, token_res, qp=qp)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def forward_test(self, x, qp=0, tasks=[], **kwargs):
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            coded_unit = self.dino_codec(h_dino, token_res, qp=qp)
            h_dino_hat = coded_unit["h_hat"]

            task_feats = {}
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino_hat)
            if "seg" in tasks:
                task_feats["seg"] = self.dino.decode_seg(h_dino_hat, token_res)

            return coded_unit, task_feats

    def get_feature_numel(self, x):
        h_dino = self.dino.encode(x)
        return h_dino.numel()

    def compress(self, x, qp=0, **kwargs):
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        coded_unit = self.dino_codec.compress(h_dino, token_res, qp=qp)
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        encoded = coded_unit
        token_res = coded_unit["pstate"]["token_res"]
        decoded = self.dino_codec.decompress(**encoded)
        task_feats = {}
        if "cls" in tasks:
            task_feats["cls"] = self.dino.decode_cls(decoded["h_hat"])
        if "seg" in tasks:
            task_feats["seg"] = self.dino.decode_seg(decoded["h_hat"], token_res)
        return task_feats


@register("MPC_I12")
class MPC_I12(CompressionModel):
    def __init__(
        self,
        vqgan_backbone={},
        vqgan_codec={},
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        self.vqgan = VqganBackbone(vqgan_backbone)
        self.vqgan_codec = UniformTokenCodec(**vqgan_codec)
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.dino_codec = VitUnionLatentCodecWithCtx(**dino_codec)
        self.patch_size = self.dino.patch_size

        # additional branch for enhance branch1
        D_DINO = dino_codec["h_dim"]
        D_VQGAN = dino_codec["ctx_dim"]
        self.cond_dec_for_vqgan = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

    def forward(self, x, **kwargs):  # for training
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            # vqgan_out = self.vqgan_codec(vqgan_enc["tokens"]) # just constant likelihoods
            h_vqgan = vqgan_enc["z"]
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino)[-1]

        h_dino = h_dino.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan.clone(),
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino.clone(),
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def extract_feature(self, x, **kwargs):  # for training
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            h_dino = self.dino.encode(x)
            return {
                "tokens": vqgan_enc["tokens"],
                "h_dino": h_dino,
            }

    def offline_forward(self, data, device, **kwargs):  # for training
        with torch.inference_mode():
            h_dino = data["h_dino"].to(device).float()
            tokens = data["tokens"].to(device).long()
            _, _, H, W = data["x_shape"]
            token_res = (H // self.patch_size, W // self.patch_size)

            # x_uint8 = data["x_uint8"].to(device)
            # x = x_uint8 / 255.0
            # h_dino_ref = self.dino.encode(x).to(torch.float16).float()
            # assert torch.allclose(h_dino, h_dino_ref)

            o_dino = self.dino.decode_whole(h_dino, token_res)[-1]

        h_dino = h_dino.clone()
        h_vqgan_ctx = self.vqgan.tokens_to_features(tokens.clone())

        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan_ctx.clone(),
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino.clone(),
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def forward_test(self, x, tasks, **kwargs):
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            vqgan_cu = self.vqgan_codec(
                vqgan_enc["tokens"]
            )  # just constant likelihoods
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            dino_cu = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
            h_dino_hat = dino_cu["h_hat"]

            task_feats = {}
            if "rec1" in tasks:
                task_feats["rec1"] = self.vqgan.decode(h_vqgan_ctx)
            if "rec2" in tasks:
                h_hat_for_vqgan = self.cond_dec_for_vqgan(
                    torch.cat([dino_cu["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                )
                task_feats["rec2"] = self.vqgan.decode(h_hat_for_vqgan)
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino_hat)
            if "seg" in tasks:
                task_feats["seg"] = self.dino.decode_seg(h_dino_hat, token_res)

            coded_data = {
                "type": "frame",
                "data": {
                    "layer1": vqgan_cu,
                    "layer2": dino_cu,
                },
            }
            return coded_data, task_feats

    def compress(self, x, **kwargs):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_cu = self.vqgan_codec.compress(vqgan_enc["tokens"])
        h_vqgan_ctx = vqgan_enc["z_q"]

        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_cu = self.dino_codec.compress(h_dino, h_vqgan_ctx, token_res)

        coded_data = {
            "type": "frame",
            "data": {
                "layer1": vqgan_cu,
                "layer2": dino_cu,
            },
        }
        return coded_data

    def decompress(self, coded_data, tasks=[], **kwargs):
        vqgan_cu = coded_data["data"]["layer1"]
        dino_cu = coded_data["data"]["layer2"]

        token_res = dino_cu["pstate"]["token_res"]
        vqgan_decoded = self.vqgan_codec.decompress(**vqgan_cu)
        h_vqgan_ctx = self.vqgan.tokens_to_features(vqgan_decoded["tokens"])
        dino_decoded = self.dino_codec.decompress(**dino_cu, ctx=h_vqgan_ctx)

        task_feats = {}
        if "rec1" in tasks:
            task_feats["rec1"] = self.vqgan.decode(h_vqgan_ctx)
        if "rec2" in tasks:
            h_hat_for_vqgan = self.cond_dec_for_vqgan(
                torch.cat([dino_decoded["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
            )
            task_feats["rec2"] = self.vqgan.decode(h_hat_for_vqgan)
        if "cls" in tasks:
            task_feats["cls"] = self.dino.decode_cls(dino_decoded["h_hat"])
        if "seg" in tasks:
            task_feats["seg"] = self.dino.decode_seg(dino_decoded["h_hat"], token_res)

        return task_feats


@register("MPC_I12_CtxAsHyper")
class MPC_I12_CtxAsHyper(CompressionModel):
    def __init__(
        self,
        vqgan_backbone={},
        vqgan_codec={},
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        self.vqgan = VqganBackbone(vqgan_backbone)
        self.vqgan_codec = UniformTokenCodec(**vqgan_codec)
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.dino_codec = VitUnionLatentCodecCtxAsHyper(**dino_codec)
        self.patch_size = self.dino.patch_size

        # additional branch for enhance branch1
        D_DINO = dino_codec["h_dim"]
        D_VQGAN = dino_codec["ctx_dim"]
        self.cond_dec_for_vqgan = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

    def forward(self, x, **kwargs):  # for training
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            # vqgan_out = self.vqgan_codec(vqgan_enc["tokens"]) # just constant likelihoods
            h_vqgan = vqgan_enc["z"]
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino)[-1]

        h_dino = h_dino.clone()
        h_vqgan_ctx = h_vqgan_ctx.clone()
        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[-1]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan.clone(),
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino.clone(),
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def forward_test(
        self,
        x,
        tasks,
        **kwargs
    ):
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            vqgan_cu = self.vqgan_codec(
                vqgan_enc["tokens"]
            )  # just constant likelihoods
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            dino_cu = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
            h_dino_hat = dino_cu["h_hat"]

            task_feats = {}
            if "rec1" in tasks:
                task_feats["rec1"] = self.vqgan.decode(h_vqgan_ctx)
            if "rec2" in tasks:
                h_hat_for_vqgan = self.cond_dec_for_vqgan(
                    torch.cat([dino_cu["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                )
                task_feats["rec2"] = self.vqgan.decode(h_hat_for_vqgan)
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino_hat)
            if "seg" in tasks:
                task_feats["seg"] = self.dino.decode_seg(h_dino_hat, token_res)

            coded_data = {
                "type": "frame",
                "data": {
                    "layer1": vqgan_cu,
                    "layer2": dino_cu,
                },
            }
            return coded_data, task_feats

    def compress(self, x, **kwargs):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_cu = self.vqgan_codec.compress(vqgan_enc["tokens"])
        h_vqgan_ctx = vqgan_enc["z_q"]

        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_cu = self.dino_codec.compress(h_dino, h_vqgan_ctx, token_res)

        coded_data = {
            "type": "frame",
            "data": {
                "layer1": vqgan_cu,
                "layer2": dino_cu,
            },
        }
        return coded_data

    def decompress(self, coded_data, tasks=[], **kwargs):
        vqgan_cu = coded_data["data"]["layer1"]
        dino_cu = coded_data["data"]["layer2"]
        token_res = dino_cu["pstate"]["token_res"]
        vqgan_decoded = self.vqgan_codec.decompress(**vqgan_cu)
        h_vqgan_ctx = self.vqgan.tokens_to_features(vqgan_decoded["tokens"])
        dino_decoded = self.dino_codec.decompress(**dino_cu, ctx=h_vqgan_ctx)

        task_feats = {}
        if "rec1" in tasks:
            task_feats["rec1"] = self.vqgan.decode(h_vqgan_ctx)
        if "rec2" in tasks:
            h_hat_for_vqgan = self.cond_dec_for_vqgan(
                torch.cat([dino_decoded["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
            )
            task_feats["rec2"] = self.vqgan.decode(h_hat_for_vqgan)
        if "cls" in tasks:
            task_feats["cls"] = self.dino.decode_cls(dino_decoded["h_hat"])
        if "seg" in tasks:
            task_feats["seg"] = self.dino.decode_seg(dino_decoded["h_hat"], token_res)

        return task_feats
