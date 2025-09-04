"""
Models with no compression , just for testing
"""

import torch
from compressai.registry import register_model
from compressai.models.base import CompressionModel
from mpcompress.backbone.base import Dinov2TimmBackbone


@register_model("Dinov2TimmNoCompress")
class Dinov2TimmNoCompress(CompressionModel):
    def __init__(
        self,
        dino_backbone={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size

    def forward_test(self, x, return_cls=False, return_seg=False, **kwargs):
        with torch.inference_mode():
            results = {}
            h_dino = self.dino.encode(x)

            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            if return_cls:
                results["cls"] = self.dino.decode_cls(h_dino)
            if return_seg:
                results["seg"] = self.dino.decode_seg(h_dino, token_res)

            results["ibranch2"] = {"bits": {"t": 0}}
            return results

    def get_feature_numel(self, x):
        h_dino = self.dino.encode(x)
        return h_dino.numel()
