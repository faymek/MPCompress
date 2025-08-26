import torch
import torch.nn as nn

from compressai.registry import register_model
from compressai.models.base import CompressionModel
from compressai.models.utils import conv

from mpcompress.backbone.base import Dinov2TimmBackbone, VqganBackbone
from mpcompress.token_codecs.base import UniformTokenCodec
from mpcompress.latent_codecs.vit_feature_codec import (
    VitUnionLatentCodec,
    VitUnionLatentCodecWithCtx,
)


@register_model("MPC_I1")
class MPC_I1(CompressionModel):  # VqganTokenUniformCodec
    def __init__(self, vqgan_config, **kwargs):
        super().__init__()
        self.vqgan = VqganBackbone(vqgan_config)
        self.vqgan_codec = UniformTokenCodec(self.vqgan.codebook_size)

    def forward(self, x):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_out = self.vqgan_codec(vqgan_enc["tokens"])
        x_hat = self.vqgan.decode(vqgan_enc["z_q"])
        return {"likelihoods": vqgan_out["likelihoods"], "x_hat": x_hat}

    def compress(self, x):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_out = self.vqgan_codec.compress(vqgan_enc["tokens"])
        return vqgan_out

    def decompress(self, strings, shape, **kwargs):
        out = self.vqgan_codec.decompress(strings, shape, **kwargs)
        tokens = out["tokens"]
        z_q = self.vqgan.tokens_to_features(tokens)
        x_hat = self.vqgan.decode(z_q)
        results = {"z_q": z_q, "tokens": tokens, "x_hat": x_hat}
        return results


@register_model("MPC_I2")
class MPC_I2(CompressionModel):
    def __init__(
        self,
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.dino_codec = VitUnionLatentCodec(**dino_codec)

    def forward(self, x):  # for lic training
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            o_dino = self.dino.decode_whole(h_dino, token_res)[0]

        h_dino = h_dino.clone()
        dino_out = self.dino_codec(h_dino, token_res)
        h_dino_hat = dino_out["h_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat)[0]

        return {
            "h_dino_hat": o_dino_hat,
            "h_dino": o_dino.clone(),
            "likelihoods": dino_out["likelihoods"],
        }

    def forward_test(self, x, return_cls=False, return_seg=False, **kwargs):
        with torch.inference_mode():
            results = {}
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            dino_out = self.dino_codec(h_dino, token_res)
            h_dino_hat = dino_out["h_hat"]

            if return_cls:
                results["cls"] = self.dino.decode_cls(h_dino_hat)
            if return_seg:
                results["seg"] = self.dino.decode_seg(h_dino_hat, token_res)

            results["likelihoods"] = dino_out["likelihoods"]
            return results

    def compress(self, x):
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_out = self.dino_codec.compress(h_dino, token_res)
        dino_out["token_res"] = token_res
        layered_out = {
            "ibranch2": dino_out,
        }
        return layered_out

    def decompress(self, ibranch2, return_cls=False, return_seg=False, **kwargs):
        dino_out = ibranch2
        token_res = dino_out["token_res"]
        dino_out = self.dino_codec.decompress(**dino_out)
        results = {}
        if return_cls:
            results["cls"] = self.dino.decode_cls(dino_out["h_hat"])
        if return_seg:
            results["seg"] = self.dino.decode_seg(dino_out["h_hat"], token_res)
        return results


@register_model("MPC_I12")
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

        # additional branch for enhance branch1
        D_DINO = dino_codec["h_dim"]
        D_VQGAN = dino_codec["ctx_dim"]
        self.cond_dec_for_vqgan = nn.Sequential(
            conv(D_DINO + D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            conv(D_VQGAN, D_VQGAN, kernel_size=3, stride=1),
        )

    def forward(self, x):  # for training
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
            o_dino = self.dino.decode_whole(h_dino)[0]

        h_dino = h_dino.clone()
        dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
        h_dino_hat = dino_out["h_dino_hat"]
        o_dino_hat = self.dino.decode_whole(h_dino_hat, token_res)[0]

        h_hat_for_vqgan = self.cond_dec_for_vqgan(
            torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
        )

        if not self.training:
            with torch.no_grad():
                x_hat = self.vqgan.decode(h_hat_for_vqgan)
        else:
            x_hat = None

        return {
            "h_vqgan": h_vqgan,
            "h_vqgan_hat": h_hat_for_vqgan,
            "h_dino": o_dino,
            "h_dino_hat": o_dino_hat,
            "likelihoods": dino_out["likelihoods"],
            "x_hat": x_hat,
        }

    def forward_test(
        self,
        x,
        return_rec1=False,
        return_rec2=False,
        return_cls=False,
        return_seg=False,
    ):
        with torch.inference_mode():
            vqgan_enc = self.vqgan.encode(x)
            vqgan_out = self.vqgan_codec(
                vqgan_enc["tokens"]
            )  # just constant likelihoods
            h_vqgan_ctx = vqgan_enc["z_q"]

            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            dino_out = self.dino_codec(h_dino, h_vqgan_ctx, token_res)
            h_dino_hat = dino_out["h_hat"]

            results = {}
            if return_rec1:
                results["rec1"] = self.vqgan.decode(h_vqgan_ctx)
            if return_rec2:
                h_hat_for_vqgan = self.cond_dec_for_vqgan(
                    torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
                )
                results["rec2"] = self.vqgan.decode(h_hat_for_vqgan)
            if return_cls:
                results["cls"] = self.dino.decode_cls(h_dino_hat)
            if return_seg:
                results["seg"] = self.dino.decode_seg(h_dino_hat, token_res)

            results["likelihoods"] = {
                "y": dino_out["likelihoods"]["y"],
                "z": dino_out["likelihoods"]["z"],
                "z_q": vqgan_out["likelihoods"]["y"],
            }
            return results

    def compress(self, x):
        vqgan_enc = self.vqgan.encode(x)
        vqgan_out = self.vqgan_codec.compress(vqgan_enc["tokens"])
        h_vqgan_ctx = vqgan_enc["z_q"]

        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_out = self.dino_codec.compress(h_dino, h_vqgan_ctx, token_res)
        dino_out["token_res"] = token_res
        layered_out = {
            "ibranch1": vqgan_out,
            "ibranch2": dino_out,
        }
        return layered_out

    def decompress(
        self,
        ibranch1,
        ibranch2,
        return_rec1=False,
        return_rec2=False,
        return_cls=False,
        return_seg=False,
        **kwargs,
    ):
        results = {}
        vqgan_out = ibranch1
        dino_out = ibranch2
        token_res = dino_out["token_res"]
        vqgan_out = self.vqgan_codec.decompress(**vqgan_out)
        h_vqgan_ctx = self.vqgan.tokens_to_features(vqgan_out["tokens"])
        dino_out = self.dino_codec.decompress(**dino_out, ctx=h_vqgan_ctx)

        if return_rec1:
            results["rec1"] = self.vqgan.decode(h_vqgan_ctx)
        if return_rec2:
            h_hat_for_vqgan = self.cond_dec_for_vqgan(
                torch.cat([dino_out["h_hat_share"].detach(), h_vqgan_ctx], dim=1)
            )
            results["rec2"] = self.vqgan.decode(h_hat_for_vqgan)
        if return_cls:
            results["cls"] = self.dino.decode_cls(dino_out["h_hat"])
        if return_seg:
            results["seg"] = self.dino.decode_seg(dino_out["h_hat"], token_res)

        return results
