"""
MLoRE Common Config Utilities

Migrated from RFC/utils/common_config.py for standalone operation.
Provides functions for creating backbones, heads, and loss functions.
"""

import torch
import torch.nn as nn
from easydict import EasyDict as edict


def get_backbone(p):
    """Return the backbone based on stage."""
    if p['stage'] == 'stage1':
        from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom import MLoRE_vit_base_patch16_384
        backbone = MLoRE_vit_base_patch16_384(p=p, pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = p.final_embed_dim
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)] 
    elif p['stage'] == 'stage2':
        from mpcompress.backbone.mlore_transformers.MLoRE_coding_input_featcom_mona import MLoRE_vit_base_patch16_384
        backbone = MLoRE_vit_base_patch16_384(p=p, pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = p.final_embed_dim
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)] 
    elif p['stage'] == 'stage0':
        from mpcompress.backbone.mlore_transformers.MLoRE_baseline_nocompress import MLoRE_vit_base_patch16_384
        backbone = MLoRE_vit_base_patch16_384(p=p, pretrained=True, drop_path_rate=0.15, img_size=p.TRAIN.SCALE)
        backbone_channels = p.final_embed_dim
        p.backbone_channels = backbone_channels
        p.spatial_dim = [[p.TRAIN.SCALE[0]//16, p.TRAIN.SCALE[1]//16] for _ in range(4)] 
    else:
        raise NotImplementedError(f"Unknown stage: {p['stage']}")

    return backbone, backbone_channels


def get_head(p, backbone_channels, task):
    """Return the decoder head for a specific task."""
    if task == 'scene':
        from mpcompress.backbone.mlore_transformers.heads import MLPHead
        return MLPHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    if p['head'] == 'conv':
        from mpcompress.backbone.mlore_transformers.heads import ConvHead
        return ConvHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    elif p['head'] == 'deconv':
        from mpcompress.backbone.mlore_transformers.heads import DEConvHead
        return DEConvHead(backbone_channels, p.TASKS.NUM_OUTPUT[task])
    else:
        raise NotImplementedError(f"Unknown head type: {p['head']}")


def get_loss(p, task=None):
    """Return loss function for a specific task."""
    if task == 'edge':
        from mpcompress.losses.loss_functions import BalancedBinaryCrossEntropyLoss
        criterion = BalancedBinaryCrossEntropyLoss(pos_weight=p['edge_w'], ignore_index=p.ignore_index)
    elif task == 'semseg' or task == 'human_parts' or task == 'scene':
        from mpcompress.losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(ignore_index=p.ignore_index)
    elif task == 'normals':
        from mpcompress.losses.loss_functions import L1Loss
        criterion = L1Loss(normalize=True, ignore_index=p.ignore_index)
    elif task == 'sal':
        from mpcompress.losses.loss_functions import CrossEntropyLoss
        criterion = CrossEntropyLoss(balanced=True, ignore_index=p.ignore_index) 
    elif task == 'depth':
        from mpcompress.losses.loss_functions import L1Loss
        criterion = L1Loss(ignore_invalid_area=p.ignore_invalid_area_depth, ignore_index=0)
    else:
        criterion = None

    return criterion


def get_criterion(p):
    """Return the multi-task loss criterion."""
    from mpcompress.losses.loss_schemes import MultiTaskLoss
    loss_ft = torch.nn.ModuleDict({task: get_loss(p, task) for task in p.TASKS.NAMES})
    loss_weights = p['loss_kwargs']['loss_weights']
    
    return MultiTaskLoss(p, p.TASKS.NAMES, loss_ft, loss_weights)

