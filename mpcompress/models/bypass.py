"""
Models with no compression , just for testing
"""

import torch
from compressai.registry import register_model
from compressai.models.base import CompressionModel
from mpcompress.backbone.base import Dinov2TimmBackbone


@register_model("Dinov2TimmBypass")
class Dinov2TimmBypass(CompressionModel):
    def __init__(
        self,
        dino_backbone={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size

    def forward_test(self, x, tasks=[], **kwargs):
        with torch.inference_mode():
            h_dino = self.dino.encode(x)

            token_res = (
                x.shape[2] // self.dino.patch_size,
                x.shape[3] // self.dino.patch_size,
            )
            task_feats = {}
            if "cls" in tasks:
                task_feats["cls"] = self.dino.decode_cls(h_dino)
            if "seg" in tasks:
                task_feats["seg"] = self.dino.decode_seg(h_dino, token_res)

            coded_unit = {
                "strings": {"bypass": [[b""]]},  # empty bytes
                "pstate": {"token_res": token_res},
            }
            return coded_unit, task_feats

    def get_feature_numel(self, x):
        h_dino = self.dino.encode(x)
        return h_dino.numel()
