# Yuqi Yang
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on Vision Transformer (ViT) in PyTorch by Ross Wightman

INTERPOLATE_MODE = 'bilinear'
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_

import numpy as np
from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

BatchNorm2d = nn.BatchNorm2d
_logger = logging.getLogger(__name__)

class ConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), BatchNorm2d(in_channels), nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal(self.linear_pred.bias, mean=0, std=0.02)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))
    
class MLPHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Linear(in_channels, in_channels), BatchNorm2d(in_channels), nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Linear(in_channels, num_classes)
        nn.init.normal(self.linear_pred.bias, mean=0, std=0.02)

    def forward(self, x):
        x = x.mean(-1).mean(-1) #Avg. Pooling -> B C 
        return self.linear_pred(self.mt_proj(x))
    
class DEConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(
            nn.ConvTranspose2d(in_channels, in_channels//2, 2, stride=2, padding=0), BatchNorm2d(in_channels//2), nn.GELU(),
            nn.Conv2d(in_channels//2, in_channels//2, 3, padding=1), BatchNorm2d(in_channels//2), nn.GELU()
            )

        self.linear_pred = nn.Conv2d(in_channels//2, num_classes, kernel_size=1)
        trunc_normal_(self.mt_proj[0].weight, std=0.02)
        trunc_normal_(self.mt_proj[3].weight, std=0.02)
        trunc_normal_(self.linear_pred.weight, std=0.02)

    def forward(self, x):
        return self.linear_pred(self.mt_proj(x))