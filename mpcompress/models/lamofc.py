import torch

from compressai.registry import register_model
from compressai.models.base import CompressionModel

from mpcompress.backbone.base import Dinov2TimmBackbone, Dinov2OrgBackbone
from mpcompress.latent_codecs.vtm import VtmFeatureCodec
from mpcompress.utils.debug import extract_shapes


@register_model("Dinov2TimmPatchVtmCodec")
class Dinov2TimmPatchVtmCodec(CompressionModel):
    def __init__(
        self,
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.dino_codec = VtmFeatureCodec(**dino_codec)

        self.patch_size = self.dino.patch_size
        self.img_size = self.dino.img_size
        self.dynamic_size = self.dino.dynamic_size

    def forward(self, x):  # for training
        raise NotImplementedError("VTM does not need training.")

    def forward_test(self, x, quality, return_cls=False, return_seg=False, **kwargs):
        with torch.inference_mode():
            results = {}
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            h_dino = self.dino.decode_seg(h_dino, token_res)

            dino_out = self.dino_codec.forward_test(
                h_dino[0].cpu().numpy(), quality=quality
            )

            if return_cls:
                raise NotImplementedError("cls decoding is not supported")
            if return_seg:
                results["seg"] = [torch.from_numpy(dino_out["h_hat"]).to(x.device)]

            results["bits"] = dino_out["bits"]
            return results

    def get_feature_numel(self, x):
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        h_dino = self.dino.decode_seg(h_dino, token_res)[0]
        return h_dino.numel()

    def compress(self, x, quality):
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        h_dino = self.dino.decode_seg(h_dino, token_res)
        dino_out = self.dino_codec.compress(h_dino[0].cpu().numpy(), quality=quality)
        layered_out = {
            "ibranch2": dino_out,
        }
        return layered_out

    def decompress(self, ibranch2, return_cls=False, return_seg=False, **kwargs):
        dino_out = ibranch2
        dino_out = self.dino_codec.decompress(**dino_out)
        results = {}
        if return_cls:
            raise NotImplementedError("cls decoding is not supported")
        if return_seg:
            results["seg"] = [torch.from_numpy(dino_out["h_hat"]).cuda()]
        return results


@register_model("Dinov2OrgSlidePatchVtmCodec")
class Dinov2OrgSlidePatchVtmCodec(CompressionModel):
    def __init__(
        self,
        slide_size=[518, 518],
        slide_stride=[259, 259],
        dino_backbone={},
        dino_codec={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)
        self.dino_codec = VtmFeatureCodec(**dino_codec)

        self.patch_size = self.dino.patch_size
        self.img_size = self.dino.img_size
        self.dynamic_size = self.dino.dynamic_size
        self.slide_size = slide_size
        self.slide_stride = slide_stride

    def forward(self, x):  # for training
        raise NotImplementedError("VTM does not need training.")

    def forward_test(self, x, quality, return_cls=False, return_seg=False, **kwargs):
        # h_dino_list: [ [(B,L,C), ...], ..., [(B,L,C), ...] ]
        # stacked_feature: (N_crop, N_layer, H*W+1, C)
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        stacked_feature = stacked_feature.cpu().numpy()

        dino_out = self.dino_codec.compress(stacked_feature, quality=quality)
        dino_out = self.dino_codec.decompress(**dino_out)

        stacked_feature = torch.from_numpy(dino_out["h_hat"]).cuda()
        feature_list = [
            [
                stacked_feature[i, j].unsqueeze(0)
                for j in range(stacked_feature.shape[1])
            ]
            for i in range(stacked_feature.shape[0])
        ]

        results = {}
        if return_cls:
            raise NotImplementedError("cls decoding is not supported")
        if return_seg:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            results["seg"] = self.dino.slide_decode_seg(feature_list, slide_res)
            print(extract_shapes(results["seg"]))

            results["bits"] = {"vtm": 0}
        return results

    def get_feature_numel(self, x):
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        return stacked_feature.numel()

    def compress(self, x, quality):
        # h_dino_list: [ [(B,L,C), ...], ..., [(B,L,C), ...] ]
        # stacked_feature: (N_crop, N_layer, H*W+1, C)
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        stacked_feature = stacked_feature.cpu().numpy()
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        dino_out = self.dino_codec.compress(stacked_feature, quality=quality)
        dino_out["token_res"] = token_res
        layered_out = {
            "ibranch2": dino_out,
        }
        return layered_out

    def decompress(self, ibranch2, return_cls=False, return_seg=False, **kwargs):
        dino_out = ibranch2
        token_res = dino_out["token_res"]
        dino_out = self.dino_codec.decompress(**dino_out)
        stacked_feature = torch.from_numpy(dino_out["h_hat"]).cuda()
        feature_list = [
            [
                stacked_feature[i, j].unsqueeze(0)
                for j in range(stacked_feature.shape[1])
            ]
            for i in range(stacked_feature.shape[0])
        ]

        results = {}
        if return_cls:
            raise NotImplementedError("cls decoding is not supported")
        if return_seg:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            results["seg"] = self.dino.slide_decode_seg(feature_list, slide_res)
        return results
