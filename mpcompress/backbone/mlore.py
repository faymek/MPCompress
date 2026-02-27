"""
MLoRE Backbone Wrapper for MPCompress Framework

This module wraps the RFC's MLoRE (Multi-task Low-Rank Expert) backbone 
to provide standard encode()/decode() interfaces compatible with the MPCompress framework.

The MLoRE backbone is a Vision Transformer (ViT) based architecture that supports
multi-task learning with built-in feature compression capabilities.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

# RFC code has been migrated to mpcompress, no need for external RFC dependency


INTERPOLATE_MODE = 'bilinear'


class MLoREBackbone(nn.Module):
    """
    MLoRE Backbone wrapper providing standard encode()/decode() interface.
    
    This backbone supports three training stages:
    - stage0: Full model without compression (baseline)
    - stage1: Train compression module only
    - stage2: Fine-tune with Mona adapters
    
    Args:
        p: Configuration dict/easydict containing model parameters
        stage: Training stage ('stage0', 'stage1', 'stage2')
        pretrained: Whether to load pretrained ViT weights
        img_size: Input image size as (H, W) tuple
        drop_path_rate: Drop path rate for stochastic depth
    """
    
    def __init__(
        self,
        p,
        stage='stage1',
        pretrained=True,
        img_size=(512, 512),
        drop_path_rate=0.15,
    ):
        super().__init__()
        self.p = p
        self.stage = stage
        self.img_size = img_size
        
        # Import and create backbone based on stage
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
            raise ValueError(f"Unknown stage: {stage}. Must be 'stage0', 'stage1', or 'stage2'")
        
        # Store backbone channels
        self.backbone_channels = p.final_embed_dim if hasattr(p, 'final_embed_dim') else 640
        self.patch_size = 16
        self.resolution = [img_size[0] // self.patch_size, img_size[1] // self.patch_size]
    
    def encode(self, x):
        """
        Extract intermediate features from input image.
        
        This corresponds to the front-end encoder that extracts features
        before compression.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W)
            
        Returns:
            h: Intermediate feature tensor
        """
        return self.backbone.get_features(x)
    
    def decode(self, h, tasks=None, episode_tasks=None):
        """
        Decode intermediate features to task-specific outputs.
        
        This corresponds to the back-end decoder that processes
        compressed features for downstream tasks.
        
        Args:
            h: Intermediate feature tensor from encode() or compression codec
            tasks: List of task names to decode for (e.g., ['semseg', 'edge'])
            episode_tasks: Episode task grouping for multi-task routing
            
        Returns:
            task_features: Dict mapping task names to feature tensors
            info: Dict containing auxiliary information (bpp_loss, mse_loss, etc.)
        """
        if episode_tasks is None:
            if tasks is not None:
                episode_tasks = [list(tasks)]
            else:
                episode_tasks = [list(self.p.TASKS.NAMES)]
        
        task_features, info = self.backbone.forward_withfeat(h, episode_tasks)
        return task_features, info
    
    def forward(self, x, tasks=None, episode_tasks=None, return_feat=False):
        """
        Full forward pass: encode + decode.
        
        Args:
            x: Input image tensor of shape (B, 3, H, W)
            tasks: List of task names for decoding
            episode_tasks: Episode task grouping
            return_feat: If True, return intermediate features instead of task outputs
            
        Returns:
            If return_feat:
                h: Intermediate features
            Else:
                task_features: Dict of task outputs
                info: Auxiliary information
        """
        if episode_tasks is None:
            if tasks is not None:
                episode_tasks = [list(tasks)]
            else:
                episode_tasks = [list(self.p.TASKS.NAMES)]
        
        return self.backbone(x, episode_tasks=episode_tasks, return_feat=return_feat)
    
    def get_compression_module(self):
        """
        Get the feature compression module for direct access.
        
        Returns:
            compress: FeatCompression module
        """
        if hasattr(self.backbone, 'compress'):
            return self.backbone.compress
        return None
    
    def set_grad_mode(self, mode):
        """
        Set gradient mode for different training phases.
        
        Available modes depend on the wrapper class in use:
        - 'full_finetune': Train all parameters
        - 'only_compress': Train only compression module
        - 'finetune_mona': Fine-tune Mona adapters and decoder
        
        Args:
            mode: Gradient mode string
        """
        # Delegate to backbone if it has set_grad methods
        mode_methods = {
            'full_finetune': 'set_grad_fullfinetune',
            'only_compress': 'set_grad_vit_onlycompress_onlyedge',
            'finetune_mona': 'set_grad_vit_finetune_onlyedge_3rdmona_wdecoder',
        }
        if mode in mode_methods and hasattr(self.backbone, mode_methods[mode]):
            getattr(self.backbone, mode_methods[mode])()
        else:
            print(f"Warning: Gradient mode '{mode}' not found or not applicable")


class MLoREBackboneLarge(MLoREBackbone):
    """
    Large variant of MLoRE backbone using ViT-Large architecture.
    """
    
    def __init__(
        self,
        p,
        stage='stage1',
        pretrained=True,
        img_size=(512, 512),
        drop_path_rate=0.15,
    ):
        nn.Module.__init__(self)
        self.p = p
        self.stage = stage
        self.img_size = img_size
        
        # Import large model variants
        if stage == 'stage0':
            from mpcompress.backbone.mlore_transformers.MLoRE_baseline_nocompress import MLoRE_vit_large_patch16_384
            self.backbone = MLoRE_vit_large_patch16_384(
                p=p, pretrained=pretrained, drop_path_rate=drop_path_rate, img_size=img_size
            )
        elif stage == 'stage1':
            from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom import MLoRE_vit_large_patch16_384
            self.backbone = MLoRE_vit_large_patch16_384(
                p=p, pretrained=pretrained, drop_path_rate=drop_path_rate, img_size=img_size
            )
        elif stage == 'stage2':
            from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom_mona import MLoRE_vit_large_patch16_384
            self.backbone = MLoRE_vit_large_patch16_384(
                p=p, pretrained=pretrained, drop_path_rate=drop_path_rate, img_size=img_size
            )
        else:
            raise ValueError(f"Unknown stage: {stage}")
        
        self.backbone_channels = p.final_embed_dim if hasattr(p, 'final_embed_dim') else 640
        self.patch_size = 16
        self.resolution = [img_size[0] // self.patch_size, img_size[1] // self.patch_size]


def create_mlore_config(
    tasks=['semseg', 'edge', 'normals', 'sal', 'human_parts'],
    stage='stage1',
    img_size=(512, 512),
    final_embed_dim=640,
    rank_list=None,
    **kwargs
):
    """
    Create a configuration dict for MLoRE backbone.
    
    Args:
        tasks: List of task names
        stage: Training stage
        img_size: Input image size (H, W)
        final_embed_dim: Final embedding dimension
        rank_list: List of ranks for LoRA modules
        **kwargs: Additional configuration parameters
        
    Returns:
        p: EasyDict configuration object
    """
    from easydict import EasyDict as edict
    
    # Default task output dimensions
    task_num_output = {
        'semseg': 21,  # PASCAL Context
        'edge': 1,
        'normals': 3,
        'sal': 2,
        'human_parts': 7,
        'depth': 1,
        'scene': 13,  # NYUD
    }
    
    if rank_list is None:
        if stage == 'stage0':
            rank_list = [16, 24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128]
        else:
            rank_list = [64, 72, 80, 88, 96, 128, 128, 160, 192, 224, 256, 256, 256, 288, 320]
    
    p = edict({
        'stage': stage,
        'final_embed_dim': final_embed_dim,
        'rank_list': rank_list,
        'spe_rank': 64,
        'topk': 9,
        'pre_softmax': False,
        'TASKS': edict({
            'NAMES': tasks,
            'NUM_OUTPUT': {t: task_num_output.get(t, 1) for t in tasks},
        }),
        'TRAIN': edict({
            'SCALE': img_size,
        }),
        'TEST': edict({
            'SCALE': img_size,
        }),
        'ignore_index': 255,
        **kwargs
    })
    
    return p




