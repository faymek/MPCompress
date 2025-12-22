"""
Models with no compression, just for testing purposes.

This module provides bypass models that perform feature extraction and task inference
without any compression. These models are useful for testing and benchmarking the
feature extraction pipeline.
"""

import torch
from compressai.registry import register_model
from compressai.models.base import CompressionModel
from mpcompress.backbone.base import Dinov2TimmBackbone


@register_model("Dinov2TimmBypass")
class Dinov2TimmBypass(CompressionModel):
    """
    A bypass model using DINOv2-Timm backbone for feature extraction without compression.

    This model performs feature extraction using a DINOv2-Timm backbone and supports
    multiple downstream tasks (classification, segmentation) without any compression
    operations. It returns empty byte strings as a placeholder for compressed data.

    Args:
        dino_backbone (dict): Configuration dictionary for the DINOv2-Timm backbone.
            Passed directly to Dinov2TimmBackbone constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino (Dinov2TimmBackbone): The DINOv2-Timm backbone model.
        patch_size (int): Patch size used by the backbone model.
    """

    def __init__(
        self,
        dino_backbone={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2TimmBackbone(**dino_backbone)
        self.patch_size = self.dino.patch_size

    def forward_test(self, x, tasks=[], **kwargs):
        """
        Forward pass for testing/inference without compression.

        Extracts features using the DINOv2 backbone and generates task-specific
        features (classification, segmentation) without performing any compression.
        Returns empty byte strings as a placeholder for compressed data.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            tasks (list of str): List of tasks to perform. Supported tasks:

                - "cls": Classification task
                - "seg": Segmentation task
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing:

                - "strings": Dictionary with "bypass" key containing empty bytes
                - "pstate": Dictionary with "token_res" (token resolution)

            task_feats (dict): Dictionary of task-specific features

                - "cls": Classification features (if "cls" in tasks)
                - "seg": Segmentation features (if "seg" in tasks)

        """
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
        """
        Calculate the total number of elements in the extracted features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the feature tensor.
        """
        h_dino = self.dino.encode(x)
        return h_dino.numel()
