import torch

from compressai.registry import register_model
from compressai.models.base import CompressionModel

from mpcompress.backbone.base import Dinov2TimmBackbone, Dinov2OrgBackbone
from mpcompress.latent_codecs.vtm import VtmFeatureCodec
from mpcompress.utils.debug import extract_shapes


@register_model("Dinov2TimmPatchCodec")
class Dinov2TimmOnlyPatchCodec(CompressionModel):
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

    def forward_test(self, x, qp, tasks, **kwargs):
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            h_dino = self.dino.decode_seg(h_dino, token_res)

            coded_unit, decoded = self.dino_codec.forward_test(
                h_dino[0].cpu().numpy(), qp=qp
            )
            task_feats = {}
            if "cls" in tasks:
                raise NotImplementedError("cls decoding is not supported")
            if "seg" in tasks:
                task_feats["seg"] = [
                    torch.from_numpy(decoded["h_hat"]).to(x.device)
                ]

            return coded_unit, task_feats

    def get_feature_numel(self, x):
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        h_dino = self.dino.decode_seg(h_dino, token_res)[0]
        return h_dino.numel()

    def compress(self, x, qp):
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        h_dino = self.dino.decode_seg(h_dino, token_res)
        encoded = self.dino_codec.compress(h_dino[0].cpu().numpy(), qp=qp)
        coded_unit = {
            "strings": encoded["strings"],
            "pstate": encoded["pstate"],
        }
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        encoded = coded_unit
        decoded = self.dino_codec.decompress(**encoded)
        task_feats = {}
        if "cls" in tasks:
            raise NotImplementedError("cls decoding is not supported")
        if "seg" in tasks:
            task_feats["seg"] = [torch.from_numpy(decoded["h_hat"]).cuda()]
        return task_feats


@register_model("Dinov2OrigSlidePatchCodec")
class Dinov2OrigSlideOnlyPatchCodec(CompressionModel):
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

    def forward_test(self, x, qp, tasks=[], **kwargs):
        # h_dino_list: [ [(B,L,C), ...], ..., [(B,L,C), ...] ]
        # stacked_feature: (N_crop, N_layer, H*W+1, C)
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        stacked_feature = stacked_feature.cpu().numpy()

        coded_unit, decoded = self.dino_codec.forward_test(stacked_feature, qp=qp)
        stacked_feature = torch.from_numpy(decoded["h_hat"]).cuda()
        feature_list = [
            [
                stacked_feature[i, j].unsqueeze(0)
                for j in range(stacked_feature.shape[1])
            ]
            for i in range(stacked_feature.shape[0])
        ]

        task_feats = {}
        if "cls" in tasks:
            raise NotImplementedError("cls decoding is not supported")
        if "seg" in tasks:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            task_feats["seg"] = self.dino.slide_decode_seg(feature_list, slide_res)
        return coded_unit, task_feats

    def get_feature_numel(self, x):
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        return stacked_feature.numel()

    def compress(self, x, qp):
        # h_dino_list: [ [(B,L,C), ...], ..., [(B,L,C), ...] ]
        # stacked_feature: (N_crop, N_layer, H*W+1, C)
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        stacked_feature = stacked_feature.cpu().numpy()

        encoded = self.dino_codec.compress(stacked_feature, qp=qp)
        # Returns values in an adapted (partially compatible) CompressAI format.
        # is called coded_unit in this reference software
        coded_unit = {
            "strings": encoded["strings"],
            "pstate": encoded["pstate"],
        }
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        encoded = coded_unit
        decoded = self.dino_codec.decompress(**encoded)
        stacked_feature = torch.from_numpy(decoded["h_hat"]).cuda()
        feature_list = [
            [
                stacked_feature[i, j].unsqueeze(0)
                for j in range(stacked_feature.shape[1])
            ]
            for i in range(stacked_feature.shape[0])
        ]

        task_feats = {}
        if "cls" in tasks:
            raise NotImplementedError("cls decoding is not supported")
        if "seg" in tasks:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            task_feats["seg"] = self.dino.slide_decode_seg(feature_list, slide_res)
        return task_feats
