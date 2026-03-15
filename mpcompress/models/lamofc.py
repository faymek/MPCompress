"""
Models using DINOv2 backbones with VTM (Video Test Model) feature codec.

This module provides compression models that combine DINOv2 feature extraction
with VTM-based compression. It includes both patch-based and sliding window
approaches for handling different image sizes.
"""

import torch

from compressai.registry import register_model
from compressai.models.base import CompressionModel

from mpcompress.backbone.base import Dinov2TimmBackbone, Dinov2OrgBackbone
from mpcompress.latent_codecs.vtm import VtmFeatureCodec
from mpcompress.entropy_models.fcvq_model import FCVQ


@register_model("Dinov2TimmPatchCodec")
class Dinov2TimmOnlyPatchCodec(CompressionModel):
    """
    Compression model using DINOv2-Timm backbone with VTM feature codec.

    This model extracts features using a DINOv2-Timm backbone and compresses them
    using VTM (Video Test Model) codec. It supports segmentation tasks but not
    classification tasks.

    Args:
        dino_backbone (dict): Configuration dictionary for the DINOv2-Timm backbone.
            Passed directly to Dinov2TimmBackbone constructor.
        dino_codec (dict): Configuration dictionary for the VTM feature codec.
            Passed directly to VtmFeatureCodec constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino (Dinov2TimmBackbone): The DINOv2-Timm backbone model.
        dino_codec (VtmFeatureCodec): The VTM feature codec for compression.
        patch_size (int): Patch size used by the backbone model.
        img_size (int or tuple): Image size expected by the backbone.
        dynamic_size (bool): Whether the model supports dynamic input sizes.
    """

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

    def forward(self, x):
        """
        Forward pass for training (not implemented).

        VTM codec does not require training, so this method raises an error.

        Args:
            x (torch.Tensor): Input image tensor.

        Raises:
            NotImplementedError: Always raised as VTM does not need training.
        """
        raise NotImplementedError("VTM does not need training.")

    def forward_test(self, x, qp, tasks, **kwargs):
        """
        Forward pass for testing/inference with compression.

        Extracts features using DINOv2 backbone, compresses them with VTM codec,
        and generates task-specific features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter for VTM compression.
            tasks (list of str): List of tasks to perform. Supported tasks:

                - "seg": Segmentation task
                - "cls": Classification task (not supported)
            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information

            task_feats (dict): Dictionary of task-specific features:

                - "seg": Segmentation features (if "seg" in tasks)

        """
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
                task_feats["seg"] = [torch.from_numpy(decoded["h_hat"]).to(x.device)]

            return coded_unit, task_feats

    def get_feature_numel(self, x):
        """
        Calculate the total number of elements in the extracted features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the feature tensor after segmentation decoding.
        """
        h_dino = self.dino.encode(x)
        token_res = (
            x.shape[2] // self.dino.patch_size,
            x.shape[3] // self.dino.patch_size,
        )
        h_dino = self.dino.decode_seg(h_dino, token_res)[0]
        return h_dino.numel()

    def compress(self, x, qp):
        """
        Compress input image to byte strings.

        Extracts features using DINOv2 backbone and compresses them using VTM codec.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter for VTM compression.

        Returns:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information
        """
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
        """
        Decompress byte strings to task-specific features.

        Args:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information

            tasks (list of str): List of tasks to perform. Supported tasks:

                - "seg": Segmentation task
                - "cls": Classification task (not supported)

            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            task_feats (dict): Dictionary of task-specific features:

                - "seg": Segmentation features (if "seg" in tasks)

        """
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
    """
    Compression model using DINOv2-Original backbone with sliding window and VTM codec.

    This model uses a sliding window approach to handle large images by processing
    them in overlapping patches. It extracts features using DINOv2-Original backbone
    and compresses them using VTM codec. Supports segmentation tasks but not
    classification tasks.

    Args:
        slide_size (list of int): Size of each sliding window patch [height, width].
            Defaults to [518, 518].
        slide_stride (list of int): Stride for sliding window [height_stride, width_stride].
            Defaults to [259, 259].
        dino_backbone (dict): Configuration dictionary for the DINOv2-Original backbone.
            Passed directly to Dinov2OrgBackbone constructor.
        dino_codec (dict): Configuration dictionary for the VTM feature codec.
            Passed directly to VtmFeatureCodec constructor.
        **kwargs (dict): Additional keyword arguments (currently unused).

    Attributes:
        dino (Dinov2OrgBackbone): The DINOv2-Original backbone model.
        dino_codec (VtmFeatureCodec): The VTM feature codec for compression.
        patch_size (int): Patch size used by the backbone model.
        img_size (int or tuple): Image size expected by the backbone.
        dynamic_size (bool): Whether the model supports dynamic input sizes.
        slide_size (list of int): Size of each sliding window patch.
        slide_stride (list of int): Stride for sliding window.
    """

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

    def forward(self, x):
        """
        Forward pass for training (not implemented).

        VTM codec does not require training, so this method raises an error.

        Args:
            x (torch.Tensor): Input image tensor.

        Raises:
            NotImplementedError: Always raised as VTM does not need training.
        """
        raise NotImplementedError("VTM does not need training.")

    def forward_test(self, x, qp, tasks=[], **kwargs):
        """
        Forward pass for testing/inference with compression using sliding window.

        Processes input image using sliding window approach, extracts features,
        compresses them with VTM codec, and generates task-specific features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter for VTM compression.
            tasks (list of str): List of tasks to perform. Supported tasks:

                - "seg": Segmentation task
                - "cls": Classification task (not supported)

            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information

            task_feats (dict): Dictionary of task-specific features:

                - "cls": Classification task (not supported)
                - "seg": Segmentation features (if "seg" in tasks)

        Note:

            h_dino_list structure: [ [(B,L,C), ...], ..., [(B,L,C), ...] ]
            stacked_feature shape: (N_crop, N_layer, H*W+1, C)
            where N_crop is the number of sliding window crops.
        """
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
        """
        Calculate the total number of elements in the extracted features.

        Uses sliding window approach to extract features and calculates the total
        number of elements across all crops and layers.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the stacked feature tensor.
        """
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        return stacked_feature.numel()

    def compress(self, x, qp):
        """
        Compress input image to byte strings using sliding window approach.

        Processes input image using sliding window, extracts features, and
        compresses them using VTM codec.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Quantization parameter for VTM compression.

        Returns:
            coded_unit (dict): Dictionary containing compressed data in CompressAI-compatible format:

                - "strings": Compressed byte strings
                - "pstate": Compression state information

        Note:

            h_dino_list structure: [ [(B,L,C), ...], ..., [(B,L,C), ...] ]
            stacked_feature shape: (N_crop, N_layer, H*W+1, C)
            where N_crop is the number of sliding window crops.
        """
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feature = torch.stack(org_feature_list)
        stacked_feature = stacked_feature.cpu().numpy()

        encoded = self.dino_codec.compress(stacked_feature, qp=qp)
        coded_unit = {
            "strings": encoded["strings"],
            "pstate": encoded["pstate"],
        }
        return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        """
        Decompress byte strings to task-specific features using sliding window.

        Args:
            coded_unit (dict): Dictionary containing compressed data:

                - "strings": Compressed byte strings
                - "pstate": Compression state information

            tasks (list of str): List of tasks to perform. Supported tasks:

                - "seg": Segmentation task
                - "cls": Classification task (not supported)

            **kwargs (dict): Additional keyword arguments (currently unused).

        Returns:
            task_feats (dict): Dictionary of task-specific features:

                - "cls": Classification task (not supported)
                - "seg": Segmentation features (if "seg" in tasks)

        """
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


@register_model("Dinov2OrigSlideSegFCVQ")
class Dinov2OrigSlideSegFCVQ(CompressionModel):
    """
    DINOv2-Original backbone with sliding window + FCVQ compression for Segmentation.

    This model combines Dinov2OrgBackbone (slide encode) with FCVQ codec
    for feature compression on segmentation tasks.

    Args:
        slide_size (list of int): Size of each sliding window patch [height, width].
        slide_stride (list of int): Stride for sliding window [height_stride, width_stride].
        dino_backbone (dict): Configuration for Dinov2OrgBackbone.
        fcvq_codec (dict): Configuration for FCVQ codec.

    Attributes:
        dino (Dinov2OrgBackbone): The DINOv2 backbone.
        fcvq (FCVQ): The FCVQ codec.
    """

    def __init__(
        self,
        slide_size=[518, 518],
        slide_stride=[259, 259],
        dino_backbone={},
        fcvq_codec={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)

        self.patch_size = self.dino.patch_size
        self.img_size = self.dino.img_size
        self.dynamic_size = self.dino.dynamic_size
        self.slide_size = slide_size
        self.slide_stride = slide_stride

        self.fcvq = FCVQ(**fcvq_codec)
        if hasattr(self.fcvq, "uncondi_entropy_model"):
            self.fcvq.uncondi_entropy_model.get_ready_for_compression()

    def forward(self, x):
        raise NotImplementedError("This model is for inference only.")

    def forward_test(self, x, qp=None, tasks=[], **kwargs):
        """Forward pass with FCVQ compression for segmentation."""
        # Extract features using slide_encode
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        # [['tensor: (1, 1370, 1536)'], ['tensor: (1, 1370, 1536)']]
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        # (N_crop, N_layer, H*W+1, C)
        stacked_feat = torch.stack(org_feature_list)  # [2, 1, 1370, 1536]
        patch_tokens = stacked_feat[:, :, 1:, :]  # [2, 1, 1369, 1536]
        patch_tokens_hat, mse_loss, strings, encoding_inds = self.fcvq.compress(
            patch_tokens
        )
        cls_token_hat = torch.zeros_like(stacked_feat[:, :, 0:1, :])  # [2, 1, 1, 1536]
        stacked_feat_hat = torch.cat([cls_token_hat, patch_tokens_hat], dim=2)
        h_dino_hat_list = [
            [
                stacked_feat_hat[i, j].unsqueeze(0)
                for j in range(stacked_feat_hat.shape[1])
            ]
            for i in range(stacked_feat_hat.shape[0])
        ]

        task_feats = {}
        if "seg" in tasks:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            task_feats["seg"] = self.dino.slide_decode_seg(h_dino_hat_list, slide_res)

        coded_data = {
            "strings": {"fcvq": [strings]},
            "pstate": {"feat_shape": patch_tokens.shape},
        }
        return coded_data, task_feats

    def get_feature_numel(self, x):
        """
        Calculate the total number of elements in the extracted features.

        Uses sliding window approach to extract features and calculates the total
        number of elements across all crops and layers.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            numel (int): Total number of elements in the stacked feature tensor.
        """
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
        stacked_feat = torch.stack(org_feature_list)  # [2, 1, 1370, 1536]
        patch_tokens = stacked_feat[:, :, 1:, :]  # [2, 1, 1369, 1536]
        return patch_tokens.numel()

    def compress(self, x, qp=None):
        """
        Compress input image to byte strings using sliding window + FCVQ.

        Extracts features via slide_encode, compresses patch tokens with FCVQ,
        and returns coded_unit. The qp parameter is unused (FCVQ has no QP).

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int, optional): Unused; kept for API compatibility with VTM models.

        Returns:
            coded_unit (dict): Dictionary containing:
                - "strings": {"vtm": [strings]} where strings is list of bytes from FCVQ
                - "pstate": {"feat_shape": stacked_feat_hat.shape}
        """
        with torch.inference_mode():
            h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
            org_feature_list = [torch.cat(feature_list) for feature_list in h_dino_list]
            stacked_feat = torch.stack(org_feature_list)
            patch_tokens = stacked_feat[:, :, 1:, :]
            patch_tokens_hat, mse_loss, strings, encoding_inds = self.fcvq.compress(
                patch_tokens
            )
            coded_unit = {
                "strings": {"fcvq": [strings]},
                "pstate": {"feat_shape": tuple(patch_tokens.shape)},
            }
            return coded_unit

    def decompress(self, coded_unit, tasks=[], **kwargs):
        """
        Decompress byte strings to task-specific features using sliding window.

        Args:
            coded_unit (dict): Dictionary containing:
                - "strings": {"fcvq": [strings]} from compress
                - "pstate": {"feat_shape": tuple}
            tasks (list of str): List of tasks. Supported: "seg".
            **kwargs: Additional arguments (unused).

        Returns:
            task_feats (dict): Dictionary with "seg" key if "seg" in tasks.
        """
        strings = coded_unit["strings"]["fcvq"][0]
        shape = coded_unit["pstate"]["feat_shape"]
        # feat_shape is (N_crop, N_layer, H*W, C); FCVQ decompress needs the temp shape
        tmp_shape = (shape[0] * shape[1], shape[2], shape[3])

        patch_tokens_hat = self.fcvq.decompress(strings, tmp_shape)
        patch_tokens_hat = patch_tokens_hat.reshape(
            shape[0], shape[1], shape[2], shape[3]
        )
        device = next(self.parameters()).device
        patch_tokens_hat = patch_tokens_hat.to(device)
        cls_token_hat = torch.zeros(
            shape[0],
            shape[1],
            1,
            shape[3],
            device=device,
            dtype=patch_tokens_hat.dtype,
        )
        stacked_feat_hat = torch.cat([cls_token_hat, patch_tokens_hat], dim=2)
        h_dino_hat_list = [
            [
                stacked_feat_hat[i, j].unsqueeze(0)
                for j in range(stacked_feat_hat.shape[1])
            ]
            for i in range(stacked_feat_hat.shape[0])
        ]

        task_feats = {}
        if "cls" in tasks:
            raise NotImplementedError("cls decoding is not supported")
        if "seg" in tasks:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            task_feats["seg"] = self.dino.slide_decode_seg(h_dino_hat_list, slide_res)
        return task_feats


@register_model("Dinov2OrigClsFCVQ")
class Dinov2OrigClsFCVQ(CompressionModel):
    """
    DINOv2-Original backbone with sliding window + FCVQ compression for Classification.

    This model combines Dinov2OrgBackbone with FCVQ codec
    for feature compression on classification tasks.

    Args:
        dino_backbone (dict): Configuration for Dinov2OrgBackbone.
        fcvq_codec (dict): Configuration for FCVQ codec.

    Attributes:
        dino (Dinov2OrgBackbone): The DINOv2 backbone.
        fcvq (FCVQ): The FCVQ codec.
    """

    def __init__(
        self,
        dino_backbone={},
        fcvq_codec={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)

        self.patch_size = self.dino.patch_size
        self.img_size = self.dino.img_size

        self.fcvq = FCVQ(**fcvq_codec)
        if hasattr(self.fcvq, "uncondi_entropy_model"):
            self.fcvq.uncondi_entropy_model.get_ready_for_compression()

    def forward(self, x):
        raise NotImplementedError("This model is for inference only.")

    def forward_test(self, x, qp=None, tasks=[], **kwargs):
        """Forward pass with FCVQ compression for classification."""
        # Extract features
        h_dino = self.dino.encode(x)  # (B, 1+HW, C)

        task_feats = {}
        if "cls" in tasks:
            h_dino_hat, mse_loss, strings, encoding_inds = self.fcvq.compress(h_dino)
            cls_features = self.dino.decode_cls(h_dino_hat)
            task_feats["cls"] = cls_features

        # Return mock coded_data
        coded_data = {
            "strings": {"fcvq": [strings]},
            "pstate": {"feat_shape": h_dino.shape},
        }
        return coded_data, task_feats

    def compress(self, x, qp=None):
        """
        Compress input image to byte strings using FCVQ for classification.

        Extracts features via encode, compresses with FCVQ, returns coded_unit.
        The qp parameter is unused (FCVQ has no QP).

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int, optional): Unused; kept for API compatibility.

        Returns:
            coded_unit (dict): Dictionary containing:
                - "strings": {"fcvq": [strings]}
                - "pstate": {"feat_shape": tuple}
        """
        with torch.inference_mode():
            h_dino = self.dino.encode(x)
            h_dino_hat, mse_loss, strings, encoding_inds = self.fcvq.compress(h_dino)
            coded_unit = {
                "strings": {"fcvq": [strings]},
                "pstate": {
                    "feat_shape": tuple(h_dino.shape),
                },
            }
            return coded_unit

    def get_feature_numel(self, x):
        """Total number of elements in encoded features (for bpfp)."""
        h_dino = self.dino.encode(x)
        return h_dino.numel()

    def decompress(self, coded_unit, tasks=[], **kwargs):
        """
        Decompress byte strings to task-specific features for classification.

        Args:
            coded_unit (dict): Dictionary containing:
                - "strings": {"fcvq": [strings]} from compress
                - "pstate": {"feat_shape": tuple}
            tasks (list of str): List of tasks. Supported: "cls".
            **kwargs: Additional arguments (unused).

        Returns:
            task_feats (dict): Dictionary with "cls" key if "cls" in tasks.
        """
        strings = coded_unit["strings"]["fcvq"][0]
        feat_shape = coded_unit["pstate"]["feat_shape"]

        h_dino_hat = self.fcvq.decompress(strings, feat_shape)
        device = next(self.parameters()).device
        h_dino_hat = h_dino_hat.to(device)

        task_feats = {}
        if "seg" in tasks:
            raise NotImplementedError("seg decoding is not supported")
        if "cls" in tasks:
            task_feats["cls"] = self.dino.decode_cls(h_dino_hat)
        return task_feats


@register_model("Dinov2OrigClsBypass")
class Dinov2OrigClsBypass(CompressionModel):
    """
    DINOv2-Original backbone for classification, WITHOUT compression (bypass).

    This model extracts features via encode and decodes for classification,
    without any compression step. Useful for testing baseline classification
    accuracy without compression.

    Args:
        dino_backbone (dict): Configuration for Dinov2OrgBackbone.

    Attributes:
        dino (Dinov2OrgBackbone): The DINOv2 backbone.
        patch_size (int): Patch size used by the backbone.
        img_size (int or tuple): Image size expected by the backbone.
    """

    def __init__(
        self,
        dino_backbone={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)

        self.patch_size = self.dino.patch_size
        self.img_size = self.dino.img_size

    def forward(self, x):
        raise NotImplementedError("This model is for inference only.")

    def forward_test(self, x, qp=None, tasks=[], **kwargs):
        """
        Forward pass WITHOUT compression - just extract and decode for classification.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Ignored (no compression).
            tasks (list of str): List of tasks. Supported: "cls".

        Returns:
            coded_data (dict): Mock coded data with zero bits.
            task_feats (dict): Dictionary with "cls" key if "cls" in tasks.
        """
        task_feats = {}
        if "seg" in tasks:
            raise NotImplementedError("seg decoding is not supported")
        if "cls" in tasks:
            task_feats["cls"] = self.dino(x, task="cls")

        coded_data = {"bits": {}}
        return coded_data, task_feats

    def compress(self, x, qp=None):
        raise NotImplementedError("No compression available in this model.")

    def decompress(self, *args, **kwargs):
        raise NotImplementedError("No decompression available in this model.")


@register_model("Dinov2OrigSlideSegBypass")
class Dinov2OrigSlideSegBypass(CompressionModel):
    """
    DINOv2-Original backbone with sliding window, WITHOUT compression (bypass).

    This model is identical to Dinov2OrigSlideOnlyPatchCodec but skips the
    VTM compression step. Useful for testing baseline mIoU without compression.

    Args:
        slide_size (list of int): Size of each sliding window patch [height, width].
        slide_stride (list of int): Stride for sliding window [height_stride, width_stride].
        dino_backbone (dict): Configuration dictionary for the DINOv2-Original backbone.

    Attributes:
        dino (Dinov2OrgBackbone): The DINOv2-Original backbone model.
        patch_size (int): Patch size used by the backbone model.
        img_size (int or tuple): Image size expected by the backbone.
        dynamic_size (bool): Whether the model supports dynamic input sizes.
        slide_size (list of int): Size of each sliding window patch.
        slide_stride (list of int): Stride for sliding window.
    """

    def __init__(
        self,
        slide_size=[518, 518],
        slide_stride=[259, 259],
        dino_backbone={},
        **kwargs,
    ):
        super().__init__()
        self.dino = Dinov2OrgBackbone(**dino_backbone)

        self.patch_size = self.dino.patch_size
        self.img_size = self.dino.img_size
        self.dynamic_size = self.dino.dynamic_size
        self.slide_size = slide_size
        self.slide_stride = slide_stride

    def forward(self, x):
        raise NotImplementedError("This model is for inference only.")

    def forward_test(self, x, qp=None, tasks=[], **kwargs):
        """
        Forward pass WITHOUT compression - just extract and decode features.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
            qp (int): Ignored (no compression).
            tasks (list of str): List of tasks to perform:
                - "seg": Segmentation task
                - "cls": Classification task (not supported)

        Returns:
            coded_data (dict): Mock coded data with zero bits.
            task_feats (dict): Dictionary of task-specific features:
                - "seg": Segmentation features (if "seg" in tasks)
        """
        # Extract features using slide_encode
        h_dino_list = self.dino.slide_encode(x, self.slide_size, self.slide_stride)
        task_feats = {}
        if "cls" in tasks:
            # For cls task, use the full image (not sliding window) to get cls features
            # This is simpler and works well for classification
            task_feats["cls"] = self.dino(x, task="cls")
        if "seg" in tasks:
            slide_res = (
                self.slide_size[0] // self.patch_size,
                self.slide_size[1] // self.patch_size,
            )
            task_feats["seg"] = self.dino.slide_decode_seg(h_dino_list, slide_res)

        # Return mock coded_data with zero bits (for eval script compatibility)
        coded_data = {"bits": {}}
        return coded_data, task_feats

    def compress(self, x, qp):
        """Compress is not supported - returns empty."""
        raise NotImplementedError("No compression available in this model.")

    def decompress(self, *args, **kwargs):
        """Decompress is not supported - returns empty."""
        raise NotImplementedError("No compression available in this model.")
