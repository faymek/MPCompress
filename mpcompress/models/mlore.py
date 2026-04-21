"""
MLoRE Model for MPCompress Framework

This module implements MLoRE (Multi-task Low-Rank Expert) models following
the MPCompress framework design specification for FrameCodec and VideoCodec.

The MLoRE model performs multi-task feature compression with support for
tasks like semantic segmentation, edge detection, surface normals, etc.

Framework specification: https://faymek.github.io/MPCompress/framework/
"""

import math
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from compressai.registry import register_model
from compressai.models.base import CompressionModel

# RFC code has been migrated to mpcompress, no need for external RFC dependency


__all__ = [
    "MLoREFrameCodec",
    "MLoREVideoCodec",
    "MLoREWrapperCodec",
]


INTERPOLATE_MODE = 'bilinear'


def calc_bits_from_likelihoods(likelihoods: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Calculate bits from likelihoods dict."""
    bits_items = {}
    for name, lh in likelihoods.items():
        bits = (torch.log(lh).sum() / (-math.log(2))).item()
        bits_items[name] = bits
    return bits_items


def calc_bits_from_strings(strings: Dict[str, List[List[bytes]]]) -> Dict[str, float]:
    """Calculate bits from compressed strings dict."""
    bits_items = {}
    for name, sub_strings in strings.items():
        bits = sum(len(s[0]) for s in sub_strings) * 8.0
        bits_items[name] = bits
    return bits_items


def convert_compressai_keys(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    转换旧版本compressai的entropy_bottleneck key格式到新版本。

    旧版本: entropy_bottleneck._matrix0 / _bias0 / _factor0
    新版本: entropy_bottleneck.matrices.0 / biases.0 / factors.0
    """
    import re

    new_state_dict: Dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        new_key = key
        if "entropy_bottleneck._matrix" in key:
            new_key = re.sub(r"_matrix(\d+)", r"matrices.\1", key)
        elif "entropy_bottleneck._bias" in key:
            new_key = re.sub(r"_bias(\d+)", r"biases.\1", key)
        elif "entropy_bottleneck._factor" in key:
            new_key = re.sub(r"_factor(\d+)", r"factors.\1", key)
        new_state_dict[new_key] = value
    return new_state_dict


@register_model("MLoREFrameCodec")
class MLoREFrameCodec(CompressionModel):
    """
    MLoRE Frame-level Codec for multi-task feature compression.
    
    This FrameCodec implements the MPCompress framework specification
    for compressing a single Access Unit (frame) with multi-task support.
    
    Architecture follows the Feature DU design:
    - Feature Encoder: ViT backbone front-end extracts intermediate features
    - Compress: Hyperprior-style codec compresses features to bitstream
    - Feature Decoder: ViT backbone back-end + task heads produce outputs
    
    Args:
        p: Configuration dict containing model parameters
        stage: Training stage ('stage0', 'stage1', 'stage2')
        pretrained: Whether to load pretrained ViT weights
        img_size: Input image size (H, W)
        drop_path_rate: Drop path rate for stochastic depth
        device: Device to run on
    """
    
    def __init__(
        self,
        p=None,
        stage='stage1',
        pretrained=True,
        img_size=(512, 512),
        drop_path_rate=0.15,
        device='cuda',
        **kwargs
    ):
        super().__init__()
        
        # Create config if not provided
        if p is None:
            from mpcompress.backbone.mlore import create_mlore_config
            p = create_mlore_config(
                tasks=['semseg', 'edge', 'normals', 'sal', 'human_parts'],
                stage=stage,
                img_size=img_size,
            )
        
        self.p = p
        self.stage = stage
        self.tasks = list(p.TASKS.NAMES)
        self.device = device
        self.img_size = img_size
        
        # Initialize backbone based on stage
        self._init_backbone(p, stage, pretrained, img_size, drop_path_rate)
        
        # Initialize task heads
        self._init_heads(p)
        
        print(f"[MLoREFrameCodec] Initialized with stage={stage}, tasks={self.tasks}")
    
    def _init_backbone(self, p, stage, pretrained, img_size, drop_path_rate):
        """Initialize backbone based on training stage."""
        if stage == 'stage0':
            from mpcompress.backbone.mlore_transformers.MLoRE_baseline_nocompress import MLoRE_vit_base_patch16_384
            self.backbone = MLoRE_vit_base_patch16_384(
                p=p, pretrained=pretrained, drop_path_rate=drop_path_rate, img_size=img_size
            )
        elif stage == 'stage1':
            from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom import MLoRE_vit_base_patch16_384
            self.backbone = MLoRE_vit_base_patch16_384(
                p=p, pretrained=pretrained, drop_path_rate=drop_path_rate, img_size=img_size
            )
        elif stage == 'stage2':
            from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom_mona import MLoRE_vit_base_patch16_384
            self.backbone = MLoRE_vit_base_patch16_384(
                p=p, pretrained=pretrained, drop_path_rate=drop_path_rate, img_size=img_size
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        self.backbone_channels = p.final_embed_dim
        self.patch_size = 16
        self.resolution = [img_size[0] // self.patch_size, img_size[1] // self.patch_size]
    
    def _init_heads(self, p):
        """Initialize task prediction heads."""
        from mpcompress.heads.mlore_heads import create_mlore_heads
        
        self.heads = create_mlore_heads(
            tasks=self.tasks,
            backbone_channels=self.backbone_channels,
            num_output_dict=p.TASKS.NUM_OUTPUT,
            head_type='conv',
        )
    
    def forward(self, x, tasks=None, episode_tasks=None, return_feat=False):
        """
        Forward pass for training.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            tasks: List of tasks to compute (default: all)
            episode_tasks: Task grouping for multi-task routing
            return_feat: If True, return features before heads
            
        Returns:
            Dict with task predictions and auxiliary info (bpp_loss, mse_loss)
        """
        if tasks is None:
            tasks = self.tasks
        if episode_tasks is None:
            episode_tasks = [list(tasks)]
        
        img_size = x.size()[-2:]
        
        # Forward through backbone
        task_features, info = self.backbone(x, episode_tasks=episode_tasks)
        
        if return_feat:
            return task_features, info
        
        # Apply task heads
        out = {}
        for task in tasks:
            if task in task_features and not isinstance(task_features[task], int):
                feat = task_features[task]
                h_out = self.heads[task](feat)
                if task != 'scene':  # Dense task - interpolate to input size
                    out[task] = F.interpolate(h_out, img_size, mode=INTERPOLATE_MODE)
                else:  # Global task
                    out[task] = h_out
        
        # Add auxiliary info
        for key in info:
            if 'loss' in key or 'route' in key:
                out[key] = info[key]
        
        return out
    
    @torch.no_grad()
    def forward_test(
        self, 
        x: torch.Tensor, 
        tasks: List[str] = None,
        return_likelihoods: bool = True,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Forward pass for testing/evaluation.
        
        This method performs inference and returns task predictions along with
        rate estimation via likelihoods.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            tasks: List of tasks to compute (default: all)
            return_likelihoods: Whether to return likelihoods for rate estimation
            **kwargs: Additional arguments
            
        Returns:
            Dict containing:
                - Task predictions (e.g., 'semseg', 'edge', etc.)
                - 'likelihoods': Dict of likelihoods for rate estimation (if return_likelihoods)
                - 'bpp_loss', 'mse_loss': Compression losses
        """
        if tasks is None:
            tasks = self.tasks
        episode_tasks = [list(tasks)]
        
        img_size = x.size()[-2:]
        
        # Forward through backbone with compression info
        task_features, info = self.backbone(x, episode_tasks=episode_tasks)
        
        # Apply task heads
        out = {}
        for task in tasks:
            if task in task_features and not isinstance(task_features[task], int):
                feat = task_features[task]
                h_out = self.heads[task](feat)
                if task != 'scene':
                    out[task] = F.interpolate(h_out, img_size, mode=INTERPOLATE_MODE)
                else:
                    out[task] = h_out
        
        # Add compression info
        if return_likelihoods and 'likelihoods' in info:
            out['likelihoods'] = info['likelihoods']
        
        # Add losses
        for key in info:
            if 'loss' in key:
                out[key] = info[key]
        
        return out
    
    def get_feature_numel(self, x: torch.Tensor) -> int:
        """Get number of elements in feature representation."""
        B, _, H, W = x.shape
        feat_h, feat_w = H // self.patch_size, W // self.patch_size
        return B * feat_h * feat_w * self.backbone_channels
    
    def compress(self, x: torch.Tensor, tasks: List[str] = None, **kwargs) -> Dict[str, Any]:
        """
        Compress a single frame (Access Unit).
        
        Following the MPCompress DataUnitCodec interface, compresses input
        image to coded_unit format.
        
        Args:
            x: Input image tensor (B, 3, H, W)
            tasks: List of tasks for encoding hints
            **kwargs: Additional encoding parameters
            
        Returns:
            coded_unit: Dictionary following framework specification:
                {
                    "strings": {"y": [[bytes]], "z": [[bytes]]},  # Compressed bitstreams
                    "pstate": {
                        "shape": (H, W),
                        "input_shape": (B, C, H, W),
                        "tasks": [...],
                        "resolution": [H, W]
                    }
                }
        """
        if tasks is None:
            tasks = self.tasks
        
        # Ensure stage supports compression
        if self.stage == 'stage0':
            raise RuntimeError("Stage0 model does not support compression. Use stage1 or stage2.")
        
        # Extract features
        if hasattr(self.backbone, 'get_features'):
            features = self.backbone.get_features(x)
        else:
            # Fallback: run forward and get intermediate features
            episode_tasks = [list(tasks)]
            _, info = self.backbone(x, episode_tasks=episode_tasks)
            features = info.get('feat_precompress', None)
            if features is None:
                raise RuntimeError("Could not extract features for compression")
        
        # Get compression module
        if not hasattr(self.backbone, 'compress'):
            raise RuntimeError("Backbone does not have compression module")
        compress_module = self.backbone.compress
        
        # Reshape features for compression (B, N, C) -> (B, C, H, W)
        B = features.shape[0]
        H, W = self.resolution
        feat_2d = features.reshape(B, H, W, -1).permute(0, 3, 1, 2)
        
        # Compress features
        compressed = compress_module.compress(feat_2d)
        
        return {
            "strings": compressed["strings"],
            "pstate": {
                "shape": compressed["shape"],
                "input_shape": list(x.shape),
                "tasks": tasks,
                "resolution": self.resolution,
            }
        }
    
    def decompress(
        self, 
        coded_unit: Dict[str, Any] = None,
        strings: Dict[str, List[List[bytes]]] = None,
        pstate: Dict[str, Any] = None,
        tasks: List[str] = None, 
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress a single frame (Access Unit).
        
        Following the MPCompress DataUnitCodec interface, decodes coded_unit
        to task outputs.
        
        Args:
            coded_unit: Dictionary from compress() (alternative to strings/pstate)
            strings: Compressed bitstreams dict
            pstate: State dict for decoding
            tasks: List of tasks to decode (default: from pstate)
            **kwargs: Additional decoding parameters
            
        Returns:
            task_feats: Dictionary with task outputs:
                {
                    "semseg": tensor,
                    "edge": tensor,
                    ...
                }
        """
        # Support both coded_unit dict and separate strings/pstate
        if coded_unit is not None:
            strings = coded_unit.get("strings", strings)
            pstate = coded_unit.get("pstate", pstate)
        
        if strings is None or pstate is None:
            raise ValueError("Must provide either coded_unit or both strings and pstate")
        
        if tasks is None:
            tasks = pstate.get("tasks", self.tasks)
        
        # Ensure stage supports decompression
        if self.stage == 'stage0':
            raise RuntimeError("Stage0 model does not support decompression. Use stage1 or stage2.")
        
        # Decompress features
        if not hasattr(self.backbone, 'compress'):
            raise RuntimeError("Backbone does not have compression module")
        compress_module = self.backbone.compress
        
        decompressed = compress_module.decompress(strings, pstate["shape"])
        feat_hat = decompressed["x_hat"]
        
        # Reshape back to sequence format
        B, C, H, W = feat_hat.shape
        feat_seq = feat_hat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        # Decode through backbone backend
        # skip_compress=True signals that feat_seq is already post-decompression,
        # so forward_withfeat must not re-run its internal compression block
        # (which would cause double compression — the bug fixed by this change).
        episode_tasks = [list(tasks)]
        if hasattr(self.backbone, 'forward_withfeat'):
            import inspect
            sig = inspect.signature(self.backbone.forward_withfeat)
            kwargs = {}
            if 'skip_compress' in sig.parameters:
                kwargs['skip_compress'] = True
            task_features, info = self.backbone.forward_withfeat(
                feat_seq, episode_tasks, **kwargs
            )
        else:
            raise RuntimeError("Backbone does not support forward_withfeat method")
        
        # Apply task heads
        input_shape = pstate["input_shape"]
        img_size = input_shape[-2:]
        
        out = {}
        for task in tasks:
            if task in task_features and not isinstance(task_features[task], int):
                feat = task_features[task]
                h_out = self.heads[task](feat)
                if task != 'scene':
                    out[task] = F.interpolate(h_out, img_size, mode=INTERPOLATE_MODE)
                else:
                    out[task] = h_out
        
        return out
    
    def get_compression_module(self):
        """Get the feature compression module."""
        if hasattr(self.backbone, 'compress'):
            return self.backbone.compress
        return None
    
    def update(self, scale_table=None, force=False):
        """Update entropy model parameters."""
        if hasattr(self.backbone, 'compress') and hasattr(self.backbone.compress, 'update'):
            return self.backbone.compress.update(scale_table, force)
        return False
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """
        Load model weights from checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
            strict: Whether to strictly enforce state_dict key matching
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        elif 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint
        
        # Handle DDP prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Convert old compressai entropy_bottleneck keys if needed (RFC does this)
        new_state_dict = convert_compressai_keys(new_state_dict)
        
        msg = self.load_state_dict(new_state_dict, strict=strict)
        print(f"[MLoREFrameCodec] Loaded checkpoint from {checkpoint_path}")
        if msg.missing_keys:
            print(f"  Missing keys: {msg.missing_keys[:5]}..." if len(msg.missing_keys) > 5 else f"  Missing keys: {msg.missing_keys}")
        if msg.unexpected_keys:
            print(f"  Unexpected keys: {msg.unexpected_keys[:5]}..." if len(msg.unexpected_keys) > 5 else f"  Unexpected keys: {msg.unexpected_keys}")
        
        return msg
    
    def set_grad_mode(self, mode: str):
        """
        Set gradient mode for different training phases.
        
        Args:
            mode: Gradient mode:
                - 'full_finetune': Train all parameters
                - 'only_compress': Train only compression module
                - 'finetune_mona': Fine-tune Mona adapters and heads
        """
        mode_methods = {
            'full_finetune': 'set_grad_fullfinetune',
            'only_compress': 'set_grad_vit_onlycompress_onlyedge',
            'finetune_mona': 'set_grad_vit_finetune_onlyedge_3rdmona_wdecoder',
        }
        
        if mode in mode_methods and hasattr(self.backbone, mode_methods[mode]):
            getattr(self.backbone, mode_methods[mode])()
            print(f"[MLoREFrameCodec] Set gradient mode: {mode}")
        else:
            print(f"Warning: Gradient mode '{mode}' not found or not applicable")


@register_model("MLoREVideoCodec")
class MLoREVideoCodec(nn.Module):
    """
    MLoRE Video-level Codec for multi-task feature compression.
    
    This VideoCodec implements the MPCompress framework specification
    for compressing entire video sequences with multi-task support.
    
    Supports both frame-wise and layer-wise organization:
    - frame_wise: Each frame compressed independently
    - layer_wise: Features and metadata organized by layer
    
    Framework interface methods:
    - compress_video(video_reader, meta, codec_args) -> coded_data
    - decompress_video(coded_data, codec_args) -> results
    - compress_frame(frame, codec_args) -> coded_unit
    - decompress_frame(coded_unit, codec_args) -> task_feats
    
    Args:
        frame_codec: MLoREFrameCodec instance or config
        p: Configuration dict (if frame_codec not provided)
        stage: Training stage ('stage0', 'stage1', 'stage2')
        pretrained: Whether to load pretrained weights
        img_size: Input image size
        device: Device to run on
    """
    
    def __init__(
        self,
        frame_codec: MLoREFrameCodec = None,
        p=None,
        stage: str = 'stage1',
        pretrained: bool = True,
        img_size: Tuple[int, int] = (512, 512),
        device: str = 'cuda',
        **kwargs
    ):
        super().__init__()
        
        if frame_codec is not None:
            self.frame_codec = frame_codec
        else:
            self.frame_codec = MLoREFrameCodec(
                p=p,
                stage=stage,
                pretrained=pretrained,
                img_size=img_size,
                device=device,
                **kwargs
            )
        
        self.device = device
        self.tasks = self.frame_codec.tasks
        self.stage = self.frame_codec.stage
        self.img_size = self.frame_codec.img_size
    
    def compress_video(
        self, 
        video_reader, 
        meta: Dict[str, Any] = None, 
        codec_args: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Compress an entire video sequence.
        
        Following the MPCompress VideoCodec interface, compresses all frames
        and returns coded_data intermediate representation.
        
        Args:
            video_reader: Video reader object supporting iteration
            meta: Video metadata dict with keys like:
                - seq_name: Sequence name
                - src_width, src_height: Original dimensions
                - frame_num: Total frame count
            codec_args: Encoding arguments:
                - tasks: List of tasks to encode for
                
        Returns:
            coded_data: Dictionary following framework specification:
                {
                    "type": "frame_wise_video",
                    "data": {
                        0: coded_unit_0,
                        1: coded_unit_1,
                        ...
                    },
                    "meta": {...}
                }
        """
        if codec_args is None:
            codec_args = {}
        if meta is None:
            meta = {}
        
        tasks = codec_args.get('tasks', self.tasks)
        
        coded_data = {
            "type": "frame_wise_video",
            "data": {},
            "meta": meta,
        }
        
        total_enc_time = 0.0
        
        # Process each frame
        for idx, frame in enumerate(tqdm(video_reader, desc="Compressing")):
            # Convert frame to tensor if needed
            if not isinstance(frame, torch.Tensor):
                import cv2
                if hasattr(frame, 'rgb'):
                    # Use rgb property if available (MPCompress video reader)
                    frame_rgb = frame.rgb
                elif len(frame.shape) == 3 and frame.shape[2] == 3:
                    # Assume BGR numpy array
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                frame_tensor = torch.from_numpy(frame_rgb).permute(2, 0, 1).float() / 255.0
                frame_tensor = frame_tensor.unsqueeze(0).to(self.device)
            else:
                frame_tensor = frame.to(self.device)
                if frame_tensor.dim() == 3:
                    frame_tensor = frame_tensor.unsqueeze(0)
            
            # Compress frame
            start_time = time.time()
            coded_unit = self.frame_codec.compress(frame_tensor, tasks=tasks)
            total_enc_time += time.time() - start_time
            
            coded_data["data"][idx] = coded_unit
        
        coded_data["meta"]["enc_time"] = total_enc_time
        coded_data["meta"]["frame_count"] = len(coded_data["data"])
        
        return coded_data
    
    def decompress_video(
        self, 
        coded_data: Dict[str, Any], 
        codec_args: Dict[str, Any] = None
    ) -> Dict[int, Dict[str, torch.Tensor]]:
        """
        Decompress an entire video sequence.
        
        Following the MPCompress VideoCodec interface, decodes coded_data
        to task outputs for each frame.
        
        Args:
            coded_data: Dictionary from compress_video()
            codec_args: Decoding arguments:
                - tasks: List of tasks to decode
                
        Returns:
            results: Dictionary with frame index as key:
                {
                    0: {"semseg": tensor, "edge": tensor, ...},
                    1: {...},
                    ...
                }
        """
        if codec_args is None:
            codec_args = {}
        
        tasks = codec_args.get('tasks', self.tasks)
        
        if coded_data["type"] != "frame_wise_video":
            raise ValueError(f"Unsupported coded_data type: {coded_data['type']}")
        
        results = {}
        total_dec_time = 0.0
        
        # Process each frame
        frame_indices = sorted(coded_data["data"].keys())
        for idx in tqdm(frame_indices, desc="Decompressing"):
            coded_unit = coded_data["data"][idx]
            
            start_time = time.time()
            task_feats = self.frame_codec.decompress(coded_unit, tasks=tasks)
            total_dec_time += time.time() - start_time
            
            results[idx] = task_feats
        
        # Add timing info to results
        results["_meta"] = {
            "dec_time": total_dec_time,
            "frame_count": len(frame_indices),
        }
        
        return results
    
    def compress_frame(
        self, 
        frame: torch.Tensor, 
        codec_args: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Compress a single frame (Access Unit).
        
        Args:
            frame: Input frame tensor (B, 3, H, W) or (3, H, W)
            codec_args: Encoding arguments
            
        Returns:
            coded_unit: Compressed frame data
        """
        if codec_args is None:
            codec_args = {}
        tasks = codec_args.get('tasks', self.tasks)
        
        # Ensure proper dimensions
        if frame.dim() == 3:
            frame = frame.unsqueeze(0)
        
        return self.frame_codec.compress(frame, tasks=tasks)
    
    def decompress_frame(
        self, 
        coded_unit: Dict[str, Any], 
        codec_args: Dict[str, Any] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress a single frame (Access Unit).
        
        Args:
            coded_unit: Compressed frame data from compress_frame()
            codec_args: Decoding arguments
            
        Returns:
            task_feats: Task prediction outputs
        """
        if codec_args is None:
            codec_args = {}
        tasks = codec_args.get('tasks', self.tasks)
        return self.frame_codec.decompress(coded_unit, tasks=tasks)
    
    def forward(self, x: torch.Tensor, tasks: List[str] = None, **kwargs):
        """Forward pass through frame codec."""
        return self.frame_codec.forward(x, tasks=tasks, **kwargs)
    
    def forward_test(self, x: torch.Tensor, tasks: List[str] = None, **kwargs):
        """Test forward pass through frame codec."""
        return self.frame_codec.forward_test(x, tasks=tasks, **kwargs)
    
    def update(self, scale_table=None, force: bool = False):
        """Update entropy model parameters."""
        return self.frame_codec.update(scale_table, force)
    
    def load_checkpoint(self, checkpoint_path: str, strict: bool = False):
        """Load model weights from checkpoint."""
        return self.frame_codec.load_checkpoint(checkpoint_path, strict=strict)


class MLoREWrapperCodec(nn.Module):
    """
    Wrapper to use RFC's original MLoREWrapper_coding class.
    
    This provides exact compatibility with pretrained RFC weights
    while conforming to MPCompress interfaces.
    
    Args:
        p: Configuration dict
        checkpoint_path: Path to pretrained weights
    """
    
    def __init__(self, p, checkpoint_path=None):
        super().__init__()
        
        # Import migrated wrapper
        from mpcompress.backbone.MLoRE_wrapper import MLoREWrapper_coding
        from mpcompress.utils.mlore_common_config import get_backbone, get_head
        
        # Create backbone
        backbone, backbone_channels = get_backbone(p)
        
        # Create heads
        heads = nn.ModuleDict({
            task: get_head(p, backbone_channels, task)
            for task in p.TASKS.NAMES
        })
        
        # Create wrapper
        self.model = MLoREWrapper_coding(p, backbone, heads)
        
        self.p = p
        self.tasks = list(p.TASKS.NAMES)
        self.backbone_channels = backbone_channels
        
        # Load checkpoint if provided
        if checkpoint_path is not None:
            self.load_checkpoint(checkpoint_path)
    
    def load_checkpoint(self, checkpoint_path):
        """Load pretrained weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
        
        # Handle DDP prefix
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v

        # Convert old compressai entropy_bottleneck keys if needed (RFC does this)
        new_state_dict = convert_compressai_keys(new_state_dict)
        
        self.model.load_state_dict(new_state_dict, strict=False)
        print(f"[MLoREWrapperCodec] Loaded checkpoint from {checkpoint_path}")
    
    def forward(self, x, tasks=None, episode_tasks=None, **kwargs):
        """Forward pass."""
        if episode_tasks is None:
            if tasks is not None:
                episode_tasks = [list(tasks)]
            else:
                episode_tasks = [self.tasks]
        
        return self.model(x, episode_tasks=episode_tasks, **kwargs)
    
    def compress(self, x, tasks=None, **kwargs):
        """Compress features."""
        feat = self.model.get_features(x)
        # Use backbone compression
        compress_module = self.model.backbone.compress
        
        B, N, C = feat.shape
        H = W = int(N ** 0.5)
        feat_2d = feat.reshape(B, H, W, C).permute(0, 3, 1, 2)
        
        compressed = compress_module.compress(feat_2d)
        
        return {
            "strings": compressed["strings"],
            "pstate": {
                "shape": compressed["shape"],
                "input_shape": x.shape,
                "tasks": tasks or self.tasks,
            }
        }
    
    def decompress(self, coded_unit, tasks=None, **kwargs):
        """Decompress to task outputs."""
        pstate = coded_unit["pstate"]
        if tasks is None:
            tasks = pstate.get("tasks", self.tasks)
        
        compress_module = self.model.backbone.compress
        decompressed = compress_module.decompress(
            coded_unit["strings"],
            pstate["shape"]
        )
        
        feat_hat = decompressed["x_hat"]
        B, C, H, W = feat_hat.shape
        feat_seq = feat_hat.permute(0, 2, 3, 1).reshape(B, H * W, C)
        
        episode_tasks = [list(tasks)]
        return self.model.forward_withfeat(
            torch.zeros(1, 3, *pstate["input_shape"][-2:]),
            feat_seq,
            episode_tasks
        )




