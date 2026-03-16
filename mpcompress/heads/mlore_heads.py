"""
MLoRE Task Heads for MPCompress Framework

This module provides task-specific prediction heads for the MLoRE multi-task
learning framework, supporting tasks like semantic segmentation, edge detection,
surface normal estimation, saliency detection, and depth estimation.
"""

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_

# RFC code has been migrated to mpcompress, no need for external RFC dependency


__all__ = [
    "MLoREConvHead",
    "MLoREDEConvHead",
    "MLoREMLPHead",
    "create_mlore_heads",
]


BatchNorm2d = nn.BatchNorm2d


class MLoREConvHead(nn.Module):
    """
    Convolutional prediction head for dense tasks.
    
    Uses a simple projection + prediction architecture suitable for
    semantic segmentation, edge detection, etc.
    
    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of output classes/channels
    """
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.mt_proj = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            BatchNorm2d(in_channels),
            nn.GELU()
        )
        trunc_normal_(self.mt_proj[0].weight, std=0.02)
        
        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal_(self.linear_pred.bias, mean=0, std=0.02)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
            
        Returns:
            Prediction tensor of shape (B, num_classes, H, W)
        """
        return self.linear_pred(self.mt_proj(x))


class MLoREMLPHead(nn.Module):
    """
    MLP prediction head for global/classification tasks.
    
    Uses global average pooling followed by MLP for tasks like
    scene classification.
    
    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of output classes
    """
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.mt_proj = nn.Sequential(
            nn.Linear(in_channels, in_channels),
            nn.LayerNorm(in_channels),
            nn.GELU()
        )
        trunc_normal_(self.mt_proj[0].weight, std=0.02)
        
        self.linear_pred = nn.Linear(in_channels, num_classes)
        nn.init.normal_(self.linear_pred.bias, mean=0, std=0.02)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
            
        Returns:
            Prediction tensor of shape (B, num_classes)
        """
        # Global average pooling
        x = x.mean(-1).mean(-1)  # (B, C)
        return self.linear_pred(self.mt_proj(x))


class MLoREDEConvHead(nn.Module):
    """
    Deconvolutional prediction head for dense tasks with upsampling.
    
    Uses transposed convolution to upsample features before prediction,
    providing higher resolution outputs.
    
    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of output classes/channels
    """
    
    def __init__(self, in_channels, num_classes):
        super().__init__()
        
        self.mt_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels // 2, 2, stride=2, padding=0),
            BatchNorm2d(in_channels // 2),
            nn.GELU(),
            nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1),
            BatchNorm2d(in_channels // 2),
            nn.GELU()
        )
        
        self.linear_pred = nn.Conv2d(in_channels // 2, num_classes, kernel_size=1)
        trunc_normal_(self.mt_proj[0].weight, std=0.02)
        trunc_normal_(self.mt_proj[3].weight, std=0.02)
        trunc_normal_(self.linear_pred.weight, std=0.02)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
            
        Returns:
            Prediction tensor of shape (B, num_classes, 2*H, 2*W)
        """
        return self.linear_pred(self.mt_proj(x))


class MLoREMultiScaleHead(nn.Module):
    """
    Multi-scale prediction head for improved dense prediction.
    
    Aggregates predictions from multiple scales for better
    boundary handling in tasks like edge detection.
    
    Args:
        in_channels: Number of input channels from backbone
        num_classes: Number of output classes/channels
        num_scales: Number of scales to use
    """
    
    def __init__(self, in_channels, num_classes, num_scales=3):
        super().__init__()
        
        self.num_scales = num_scales
        
        self.scale_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
                BatchNorm2d(in_channels // 2),
                nn.GELU(),
                nn.Conv2d(in_channels // 2, num_classes, 1)
            )
            for _ in range(num_scales)
        ])
        
        # Initialize weights
        for head in self.scale_heads:
            trunc_normal_(head[0].weight, std=0.02)
            trunc_normal_(head[3].weight, std=0.02)
        
        # Fusion layer
        self.fusion = nn.Conv2d(num_classes * num_scales, num_classes, 1)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Feature tensor of shape (B, C, H, W)
            
        Returns:
            Fused prediction tensor of shape (B, num_classes, H, W)
        """
        B, C, H, W = x.shape
        
        outputs = []
        for i, head in enumerate(self.scale_heads):
            scale = 2 ** i
            if scale > 1:
                x_scaled = F.avg_pool2d(x, kernel_size=scale, stride=scale)
                out = head(x_scaled)
                out = F.interpolate(out, size=(H, W), mode='bilinear', align_corners=False)
            else:
                out = head(x)
            outputs.append(out)
        
        fused = torch.cat(outputs, dim=1)
        return self.fusion(fused)


def create_mlore_heads(
    tasks,
    backbone_channels,
    num_output_dict=None,
    head_type='conv',
):
    """
    Factory function to create task heads for MLoRE model.
    
    Args:
        tasks: List of task names
        backbone_channels: Number of channels from backbone
        num_output_dict: Dict mapping task names to output dimensions
        head_type: Type of head ('conv', 'deconv', 'mlp')
        
    Returns:
        nn.ModuleDict mapping task names to head modules
    """
    # Default output dimensions for common tasks
    default_num_output = {
        'semseg': 21,        # PASCAL Context (20 + background)
        'edge': 1,           # Binary edge detection
        'normals': 3,        # Surface normals (xyz)
        'sal': 2,            # Binary saliency
        'human_parts': 7,    # Human body parts
        'depth': 1,          # Depth estimation
        'scene': 13,         # Scene classification (NYUD)
    }
    
    if num_output_dict is None:
        num_output_dict = default_num_output
    
    head_classes = {
        'conv': MLoREConvHead,
        'deconv': MLoREDEConvHead,
        'mlp': MLoREMLPHead,
        'multiscale': MLoREMultiScaleHead,
    }
    
    heads = {}
    for task in tasks:
        num_classes = num_output_dict.get(task, 1)
        
        # Use MLP head for global classification tasks
        if task == 'scene':
            heads[task] = MLoREMLPHead(backbone_channels, num_classes)
        else:
            head_class = head_classes.get(head_type, MLoREConvHead)
            heads[task] = head_class(backbone_channels, num_classes)
    
    return nn.ModuleDict(heads)


# Wrapper class to import original RFC heads
class RFCHeadsWrapper(nn.Module):
    """
    Wrapper to use RFC's original head implementations.
    
    This wrapper imports heads from RFC for exact compatibility
    with pretrained weights.
    
    Args:
        tasks: List of task names
        backbone_channels: Number of backbone output channels
        num_output_dict: Dict mapping task -> num_classes
        head_type: Head type ('conv' or 'deconv')
    """
    
    def __init__(
        self,
        tasks,
        backbone_channels,
        num_output_dict=None,
        head_type='conv',
    ):
        super().__init__()
        
        # Import migrated heads
        from mpcompress.backbone.mlore_transformers.heads import ConvHead, DEConvHead, MLPHead
        
        default_num_output = {
            'semseg': 21,
            'edge': 1,
            'normals': 3,
            'sal': 2,
            'human_parts': 7,
            'depth': 1,
            'scene': 13,
        }
        
        if num_output_dict is None:
            num_output_dict = default_num_output
        
        head_classes = {
            'conv': ConvHead,
            'deconv': DEConvHead,
        }
        
        self.heads = nn.ModuleDict()
        for task in tasks:
            num_classes = num_output_dict.get(task, 1)
            if task == 'scene':
                self.heads[task] = MLPHead(backbone_channels, num_classes)
            else:
                head_class = head_classes.get(head_type, ConvHead)
                self.heads[task] = head_class(backbone_channels, num_classes)
    
    def forward(self, features, tasks=None):
        """
        Forward pass for all or specified tasks.
        
        Args:
            features: Dict mapping task names to feature tensors,
                     or single tensor for all tasks
            tasks: List of tasks to process (default: all)
            
        Returns:
            Dict mapping task names to prediction tensors
        """
        if tasks is None:
            tasks = list(self.heads.keys())
        
        outputs = {}
        for task in tasks:
            if task in self.heads:
                if isinstance(features, dict):
                    feat = features.get(task, features.get(list(features.keys())[0]))
                else:
                    feat = features
                outputs[task] = self.heads[task](feat)
        
        return outputs




