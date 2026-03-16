# Yuqi Yang
# Licensed under the CC BY-NC 4.0 license (https://creativecommons.org/licenses/by-nc/4.0/)
# Based on Vision Transformer (ViT) in PyTorch by Ross Wightman

INTERPOLATE_MODE = 'bilinear'
import math
import logging
from functools import partial
from collections import OrderedDict
from copy import deepcopy
from compressai.entropy_models import EntropyBottleneck, GaussianConditional
from compressai.ans import BufferedRansEncoder, RansDecoder
from compressai.models import CompressionModel
from compressai.layers import (
    AttentionBlock,
    ResidualBlock,
    ResidualBlockUpsample,
    ResidualBlockWithStride,
    conv3x3,
    subpel_conv3x3,
)
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.models.helpers import build_model_with_cfg, named_apply, adapt_input_conv
from timm.models.layers import PatchEmbed, Mlp, DropPath, trunc_normal_, lecun_normal_
from models.transformers.attention import SATransformerBlock
import numpy as np
from einops import rearrange as o_rearrange
def rearrange(*args, **kwargs):
    return o_rearrange(*args, **kwargs).contiguous()

BatchNorm2d = nn.BatchNorm2d
_logger = logging.getLogger(__name__)

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic', 'fixed_input_size': True,
        'mean': IMAGENET_INCEPTION_MEAN, 'std': IMAGENET_INCEPTION_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        # 'num_classes': 1000,
        **kwargs
    }

def sep_prompt(x, prompt_length):
    prompt = x[:, :prompt_length, :]
    x = x[:, prompt_length:, :]
    return prompt, x

default_cfgs = {
    # patch models (weights from official Google JAX impl)
    'vit_tiny_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_tiny_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_small_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_small_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch32_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_224.npz'),
    'vit_base_patch32_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_32-i21k-300ep-lr_0.001-aug_light1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.03-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_base_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_base_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch32_224': _cfg(
        url='',  # no official model weights for this combo, only for in21k
        ),
    'vit_large_patch32_384': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_p32_384-9b920ba8.pth',
        input_size=(3, 384, 384), crop_pct=1.0),
    'vit_large_patch16_224': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'),
    'vit_large_patch16_384': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/'
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_384.npz',
        input_size=(3, 384, 384), crop_pct=1.0),

    # patch models, imagenet21k (weights from official Google JAX impl)
    'vit_tiny_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/Ti_16-i21k-300ep-lr_0.001-aug_none-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_32-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_small_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/S_16-i21k-300ep-lr_0.001-aug_light1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch32_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_32-i21k-300ep-lr_0.001-aug_medium1-wd_0.03-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_base_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0.npz',
        num_classes=21843),
    'vit_large_patch32_224_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-vitjx/jx_vit_large_patch32_224_in21k-9046d2e7.pth',
        num_classes=21843),
    'vit_large_patch16_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/augreg/L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1.npz',
        num_classes=21843),
    'vit_huge_patch14_224_in21k': _cfg(
        url='https://storage.googleapis.com/vit_models/imagenet21k/ViT-H_14.npz',
        hf_hub='timm/vit_huge_patch14_224_in21k',
        num_classes=21843),

    # SAM trained models (https://arxiv.org/abs/2106.01548)
    'vit_base_patch32_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_32.npz'),
    'vit_base_patch16_sam_224': _cfg(
        url='https://storage.googleapis.com/vit_models/sam/ViT-B_16.npz'),

    # deit models (FB weights)
    'deit_tiny_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_small_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    'deit_base_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0),
    'deit_tiny_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_small_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_224': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, classifier=('head', 'head_dist')),
    'deit_base_distilled_patch16_384': _cfg(
        url='https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth',
        mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD, input_size=(3, 384, 384), crop_pct=1.0,
        classifier=('head', 'head_dist')),

    # ViT ImageNet-21K-P pretraining by MILL
    'vit_base_patch16_224_miil_in21k': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm/vit_base_patch16_224_in21k_miil.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear', num_classes=11221,
    ),
    'vit_base_patch16_224_miil': _cfg(
        url='https://miil-public-eu.oss-eu-central-1.aliyuncs.com/model-zoo/ImageNet_21K_P/models/timm'
            '/vit_base_patch16_224_1k_miil_84_4.pth',
        mean=(0, 0, 0), std=(1, 1, 1), crop_pct=0.875, interpolation='bilinear',
    ),
}


class Attention(nn.Module):
    def __init__(self, chan_nheads, resolution, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.dim = dim
        self.resolution = resolution
        pixel_no = int(resolution[0] * resolution[1])
        self.pixel_no = pixel_no

        self.chan_nheads = chan_nheads
        chan_head_dim = self.pixel_no // self.chan_nheads
        self.chan_scale = chan_head_dim ** -0.5

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   

        raw_spa_attn = (q @ k.transpose(-2, -1))
        attn = raw_spa_attn * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        raw_spa_attn = raw_spa_attn, attn

        x = (attn @ v).transpose(1, 2).reshape(B, N, C) # (B, task_no+1+HxW, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        raw_attn = [raw_spa_attn]

        return x, raw_attn

class LoraBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=1, rank=6):
        super().__init__()
        self.W = nn.Conv2d(in_channels, rank, kernel_size=kernel_size, stride=1, padding=kernel_size//2)
        self.M = nn.Conv2d(rank, out_channels, kernel_size=1, stride=1)

    def init_weights(self):
        nn.init.kaiming_uniform_(self.W.weight, a=math.sqrt(5))
        nn.init.zeros_(self.W.bias)
        nn.init.kaiming_uniform_(self.M.weight, a=math.sqrt(5))
        nn.init.zeros_(self.M.bias)
    
    def forward(self, x):
        x = self.W(x)
        x = self.M(x)
        return x


class Block(nn.Module):

    def __init__(self, chan_nheads, resolution, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)#train()和eval()对LayerNorm没有影响
        self.attn = Attention(chan_nheads, resolution, dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):

        x_attn, attn_weight = self.attn(self.norm1(x))
        x = x + self.drop_path(x_attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x, attn_weight

class SpatialAtt(nn.Module):
    def __init__(self, dim, dim_out, im_size, with_feat):
        super().__init__()
        self.conv1 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.act = nn.GELU()
        self.ln = nn.LayerNorm(dim_out)
        self.convsp = nn.Linear(im_size, 1)
        self.ln_sp = nn.LayerNorm(dim)
        self.conv2 = nn.Conv2d(dim, dim_out, kernel_size=1)
        self.conv3 = nn.Conv2d(dim_out, dim_out, kernel_size=1)
        self.with_feat = with_feat
        if with_feat:
            self.feat_linear = nn.Conv2d(dim_out *2 , dim_out *2, kernel_size=1)
    
    def forward(self, x, route_feat=None):
        n, _, h, w = x.shape
        feat = self.conv1(x)
        feat = self.ln(feat.reshape(n, -1, h * w).permute(0, 2, 1)).permute(0, 2, 1).reshape(n, -1, h, w)
        feat = self.act(feat)
        feat = self.conv3(feat)

        feat_sp = self.convsp(x.reshape(n, -1, h * w)).reshape(n, 1, -1)
        feat_sp = self.ln_sp(feat_sp).reshape(n, -1, 1, 1)
        feat_sp = self.act(feat_sp)
        feat_sp = self.conv2(feat_sp)
        
        n, c, h, w = feat.shape
        feat = torch.mean(feat.reshape(n, c, h * w), dim=2).reshape(n, c, 1, 1)
        feat = torch.cat([feat, feat_sp], dim=1)

        return feat



class MOEBlock(nn.Module):
    def __init__(self, p, final_embed_dim, im_size, kernel_size=1, with_feat=False, res=None):
        super().__init__()
        self.num_lora = len(p.rank_list)
        self.p = p
        self.lora_list_1 = nn.ModuleList()
        rank_list = p.rank_list
        for i in range(self.num_lora):#不同的rank设置Lora，从小到大
            self.lora_list_1.append(LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=rank_list[i]))
            self.lora_list_1[i].init_weights()
        #self.conv1 = nn.ModuleDict()
        #self.conv2 = nn.ModuleDict()
        #self.conv3 = nn.ModuleDict()
        self.share_conv = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=3, padding=1)
        self.bn = nn.ModuleDict()
        self.bn_all = nn.ModuleDict()
        self.activate = nn.GELU()
        #for task in self.p.TASKS.NAMES:
            #self.conv1[task] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            #self.conv3[task] = nn.Conv2d(final_embed_dim, final_embed_dim, kernel_size=1)
            #self.conv2[task] = LoraBlock(final_embed_dim, final_embed_dim, kernel_size=kernel_size, rank=p.spe_rank)
        all_comb = []
        for i in range(len(self.p.TASKS.NAMES)):
            combs = itertools.combinations(self.p.TASKS.NAMES, i+1)
            all_comb+=combs
        #task_bn_idx = dict()
        for i in range(len(all_comb)):
            #task_bn_idx[''.join(all_comb[i])] = i
            c = list(all_comb[i])
            c.sort()
            self.bn[''.join(c)] = BatchNorm2d(final_embed_dim)
            self.bn_all[''.join(c)] = BatchNorm2d(final_embed_dim)
        
        self.router_1 = nn.ModuleDict() 
        self.pre_softmax = p.pre_softmax#False
        self.desert_k = len(p.rank_list) - p.topk
        #recording the lora activation information [0] is the times of activation; [1] is the activation score
        self.task_active_lora_num = {task:[np.zeros((self.num_lora,)),np.zeros((self.num_lora,))] for task in self.p.TASKS.NAMES}
        for task in self.p.TASKS.NAMES:
            self.router_1[task] = nn.ModuleList()
            self.router_1[task].append(SpatialAtt(final_embed_dim, final_embed_dim // 4, im_size=im_size, with_feat=with_feat))
            self.router_1[task].append(nn.Conv2d(final_embed_dim // 2, self.num_lora * 2 + 1, kernel_size=1))
        
        #self.final_fusion = PromptFusion(final_embed_dim, res[0], res[1])
    def forward(self, x, task_list, prompts=None, route_feat_in=None):
        #out_ori = self.conv1(x) 直接去掉了第一层task-specific conv
        out = x#out_ori
        n, c, h, w = out.shape
        router_all = None

        for t in task_list:
            route_feat = self.router_1[t][0](out)
            prob_all = self.router_1[t][1](route_feat).unsqueeze(2)
            prob_lora, prob_mix = prob_all[:, :self.num_lora * 2], prob_all[:, self.num_lora * 2:]
            route_1_raw, stdev_1 = prob_lora.chunk(2, dim=1)  # n, 15, 1, 1, 1
        
            if self.training:
                noise = torch.randn_like(route_1_raw) * stdev_1
            else:
                noise = 0
            
            route_1_raw = torch.softmax(route_1_raw + noise, dim=1)
            route_1_indice = torch.topk(route_1_raw, self.desert_k, dim=1, largest=False)[1]
            route_1 = route_1_raw.clone()

            #record the activation information
            self.task_active_lora_num[t][0] += n
            self.task_active_lora_num[t][1] += route_1_raw.sum(0).reshape(-1).detach().cpu().numpy()

            for j in range(n):
                for i in range(self.desert_k):
                    route_1[j, route_1_indice[j, i].reshape(-1)] = 0
                    self.task_active_lora_num[t][0][route_1_indice[j, i].reshape(-1).detach().cpu().numpy()] -= 1.0
                    self.task_active_lora_num[t][1][route_1_indice[j, i].reshape(-1).detach().cpu().numpy()] -= route_1_raw[j,route_1_indice[j, i].reshape(-1)].reshape(-1).detach().cpu().numpy()
            
            if router_all==None:
                router_all = route_1
            else:
                router_all += route_1
        
        #
        desert_twice = False
        if desert_twice:#重新再次desert,同时l1 norm
            route_1_indice = torch.topk(router_all, self.desert_k, dim=1, largest=False)[1]
            for j in range(n):
                for i in range(self.desert_k):
                    router_all[j, route_1_indice[j, i].reshape(-1)] = 0
            router_all /= router_all.sum(dim=1,keepdim=True)
        else:#比较暴力的处理方式
            router_all /= len(task_list)
        

        lora_out_1 = []
        for i in range(self.num_lora):
            lora_out_1.append(self.lora_list_1[i](out).unsqueeze(1)) # n, 1, c, h, w
        lora_out_1 = torch.cat(lora_out_1, dim=1)
        lora_out_1 = torch.sum(lora_out_1 * router_all, dim=1)
        out = self.bn_all[''.join(task_list)](lora_out_1) + self.share_conv(out.detach()) #+ self.conv2[task](out) * prob_mix[:, 0]
        out = self.bn[''.join(task_list)](out)
        out = self.activate(out)

        return out, route_feat, router_all, prompts
        #TODO task prompt integration
        #out, prompts = self.final_fusion(out, prompts)
        #return out, route_feat, router_all, prompts

class PromptFusion(nn.Module):
    def __init__(self, dim, h, w, num_layers=4):
        super().__init__()
        self.h, self.w = h,w
        self.trans = SATransformerBlock(dim, 8, 64)
        self.final_proj = nn.Conv2d(dim, dim, kernel_size=3, padding=1)

    def forward(self, x, prompts):
        x = rearrange(x, 'b c h w -> b (h w) c', h=self.h, w=self.w)
        input_x = torch.cat([prompts,x], dim=1)
        out = self.trans(input_x)
        
        prompts, x = out[:,:-x.shape[1]], out[:,-x.shape[1]:]
        #x = out

        x = rearrange(x, 'b (h w) c -> b c h w', h=self.h, w=self.w)
        x = self.final_proj(x)
        return x, prompts


def ste_round(x):
    return torch.round(x) - x.detach() + x

def conv(in_channels, out_channels, kernel_size=5, stride=1):
    return nn.Conv2d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
    )

SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64
def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))

class FeatCompression(CompressionModel):
    def __init__(self,feat_dims, prompt_dims=64, N=256,M=128):
        '''
        input feature: ViT features with downsampling rate=16
        N: embedding dims
        M: bottleneck dims
        '''
        super().__init__()
        
        self.g_a = nn.Sequential(ResidualBlockWithStride(feat_dims, N, 2), 
                                   ResidualBlockWithStride(N, N, 1),
                                   ResidualBlockWithStride(N, N, 1),
                                   ResidualBlockWithStride(N, M, 1),
                                   )
        self.g_s = nn.Sequential(ResidualBlockUpsample(M, N, 2),
                                  ResidualBlockUpsample(N, N, 1),
                                  ResidualBlockUpsample(N, N, 1),
                                  ResidualBlockUpsample(N, feat_dims, 1),
                                  )
        self.entropy_bottleneck = EntropyBottleneck(192)#?
        self.gaussian_conditional = GaussianConditional(None)
        self.h_a = nn.Sequential(
            ResidualBlockWithStride(M, N, 2),
            ResidualBlockWithStride(N, N, 1),
            ResidualBlockWithStride(N, 192, 2),    
        )
        self.h_scale_s = nn.Sequential(
            ResidualBlockUpsample(192, N, 2),
            ResidualBlockUpsample(N, N, 2),
            conv(N, N//2, stride=1, kernel_size=3),
            nn.GELU(),
            conv(N//2, M, stride=1, kernel_size=3),
            )
        self.h_mean_s = nn.Sequential(
            ResidualBlockUpsample(192, N, 2),
            ResidualBlockUpsample(N, N, 2),
            conv(N, N//2, stride=1, kernel_size=3),
            nn.GELU(),
            conv(N//2, M, stride=1, kernel_size=3),
            )
        self.y_bpp = 0.0
        self.z_bpp = 0.0

    def forward(self, x):
        num_pixels = x.shape[1] * x.shape[2] * x.shape[3]
        y = self.g_a(x)
        y_shape = y.shape[2:]
        z = self.h_a(y)
        _, z_likelihood = self.entropy_bottleneck(z)
        z_offset = self.entropy_bottleneck._get_medians()
        z_tmp = z - z_offset
        z_hat = ste_round(z_tmp) + z_offset
        latent_scales = self.h_scale_s(z_hat)
        latent_means = self.h_mean_s(z_hat)

        _, y_likelihood = self.gaussian_conditional(y, latent_scales, latent_means)
        y_hat = ste_round(y - latent_means) + latent_means
        x_hat = self.g_s(y_hat)

        y_bpp = torch.log(y_likelihood).sum() / (-math.log(2) * num_pixels) #有padding怎么处理
        z_bpp = torch.log(z_likelihood).sum() / (-math.log(2) * num_pixels)
        mse = torch.nn.functional.mse_loss(x_hat, x.detach()) #x需要detach吗？
        #self.y_bpp += y_bpp
        #self.z_bpp += z_bpp
        #print('y_bpp & z_bpp: ', self.y_bpp, self.z_bpp)
        return y_bpp+z_bpp, mse, x_hat

    def update(self,scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        y_shape = y.shape[2:]

        z = self.h_a(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scale = self.h_scale_s(z_hat)
        mean = self.h_mean_s(z_hat)
        
        
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []
        
        index = self.gaussian_conditional.build_indexes(scale)
        y_q = self.gaussian_conditional.quantize(y, "symbols", mean)
        y_hat = y_q + mean

        symbols_list.extend(y_q.reshape(-1).tolist())
        indexes_list.extend(index.reshape(-1).tolist())
        
        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales = self.h_scale_s(z_hat)
        means = self.h_mean_s(z_hat)

        y_shape = [z_hat.shape[2] * 4, z_hat.shape[3] * 4]

        y_string = strings[0][0]
        y_hat_slices = []
        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()

        decoder = RansDecoder()
        decoder.set_stream(y_string)

        index = self.gaussian_conditional.build_indexes(scales)
        rv = decoder.decode_stream(index.reshape(-1).tolist(), cdf, cdf_lengths, offsets)
        rv = torch.Tensor(rv).reshape(1, -1, y_shape[0], y_shape[1])
        y_hat = self.gaussian_conditional.dequantize(rv, means)
        x_hat = self.g_s(y_hat)#.clamp_(0, 1)
        return {"x_hat": x_hat}
    
    def _likelihood(self, inputs, scales, means=None):
        half = float(0.5)
        if means is not None:
            values = inputs - means
        else:
            values = inputs

        scales = torch.max(scales, torch.tensor(0.11))
        values = torch.abs(values)
        upper = self._standardized_cumulative((half - values) / scales)
        lower = self._standardized_cumulative((-half - values) / scales)
        likelihood = upper - lower
        return likelihood
    def _standardized_cumulative(self, inputs):
        half = float(0.5)
        const = float(-(2 ** -0.5))
        # Using the complementary error function maximizes numerical precision.
        return half * torch.erfc(const * inputs)



class MLoRE(nn.Module):
    """ MLoRE built upon ViT
    """

    def __init__(self, p, select_list, img_size=224, patch_size=16, in_chans=3, embed_dim=768, depth=12,
                 num_heads=12, chan_nheads=1, mlp_ratio=4., qkv_bias=True,  
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', **kwargs):
        """
        Args:
            p (dcit): parameters
            select_list: selected layers for hierarchical prompting
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
        """
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim)) # one cls token from pretrained weights on ImageNet
        self.pos_drop = nn.Dropout(p=drop_rate)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.resolution = [int(img_size[0]/patch_size), int(img_size[1]/patch_size)]
        self.blocks = nn.Sequential(*[
            Block(
                chan_nheads,
                self.resolution,
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])
        self.norm = norm_layer(embed_dim)

        self.select_list = select_list
        self.num_layers = 4
        assert len(select_list) == self.num_layers-1 
        task_no = len(p.TASKS.NAMES)
        self.resolution = [int(img_size[0]/patch_size), int(img_size[1]/patch_size)]
        pixel_no = int(self.resolution[0] * self.resolution[1])
        self.pixel_no = pixel_no
        self.p = p
        
        prompt_decomposition = False
        if not prompt_decomposition:
            self.task_prompts = nn.Parameter(torch.zeros(1, task_no, p.final_embed_dim)) # task prompts
        else:
            self.shared_task_prompts = nn.Parameter(torch.zeros(1, 1, 32, p.final_embed_dim)) # task prompts
            self.specific_task_prompts_u = nn.Parameter(torch.zeros(1, task_no, 32))
            self.specific_task_prompts_v = nn.Parameter(torch.zeros(1, task_no, p.final_embed_dim))
            #self.task_prompts = self.shared_task_prompts.repeat() + 

        self.fea_fuse = nn.ModuleList()

        final_embed_dim = p.final_embed_dim
        self.num_lora = 15
        self.MLoRE_1 = nn.ModuleList()
        for i_layer in range(self.num_layers):
            self.fea_fuse.append(nn.Conv2d(embed_dim, final_embed_dim, kernel_size=1, stride=1))
            self.MLoRE_1.append(MOEBlock(p, final_embed_dim, im_size=pixel_no, kernel_size=3,with_feat=False, res=self.resolution))
        self.MLoRE_2 = nn.ModuleList()
        for il in range(self.num_layers):
            self.MLoRE_2.append(MOEBlock(p, final_embed_dim, im_size=pixel_no, kernel_size=3,with_feat=False, res=self.resolution))
            
        #self.task_mask = MaskFusion(final_embed_dim, self.resolution)
        #self.compress = FeatCompression(final_embed_dim)

        self.init_weights(weight_init)

    def init_weights(self, mode=''):
        assert mode in ('jax', 'jax_nlhb', 'nlhb', '')
        head_bias = -math.log(self.num_classes) if 'nlhb' in mode else 0.
        trunc_normal_(self.pos_embed, std=.02)
        if mode.startswith('jax'):
            # leave cls token as zeros to match jax impl
            named_apply(partial(_init_vit_weights, head_bias=head_bias, jax_impl=True), self)
        else:
            self.apply(_init_vit_weights)

    def _init_weights(self, m):
        # this fn left here for compat with downstream users
        _init_vit_weights(m)

    @torch.jit.ignore()
    def load_pretrained(self, checkpoint_path, prefix=''):
        #_load_weights(self, checkpoint_path, prefix)
        if checkpoint_path.endswith('.npz'):
            # 使用文件中已有的 _load_weights 函数处理 Google .npz 格式
            _load_weights(self, checkpoint_path, prefix)
        else:
            # 使用 timm 的标准加载器处理 .pth 格式
            from timm.models._builder import load_checkpoint
            load_checkpoint(self, checkpoint_path, filter_fn=checkpoint_filter_fn)
        pass

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'dist_token'}
    def get_features(self,x):
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed[:, 1:]) 
        features = []
        for idx, blk in enumerate(self.blocks):
            x, attn_weight = blk(x)
            
            x, attn_weight = x.detach(), attn_weight
            if idx + 1 in self.select_list: 
                features.append([x,attn_weight])
        features.append([self.norm(x).detach(),attn_weight])
        return features
    
    def forward_withfeat(self, feat, episode_tasks, eval=True):
        '''
        e.g.,
          episode_tasks: [['segmentation', 'depth'], [the other tasks]]
          task_index: [[0,2], [1,3,4]]
        '''

        # multi-scale backbone feature
        all_tasks = [''.join(tasks) for tasks in episode_tasks]
        out_feat = {tasks: [] for tasks in all_tasks}
        last_feat = {tasks: 0 for tasks in all_tasks}
        final_feat = {task: 0 for task in self.p.TASKS.NAMES}
        out_mask = {tasks: 0 for tasks in all_tasks}
        info = {} # pass information through the pipeline

        x_q_list = {tasks: [] for tasks in all_tasks}
        route_feat_1 = {tasks: None for tasks in all_tasks}
        route_feat_2 = {tasks: None for tasks in all_tasks}
        route_prob_1 = [{tasks: None for tasks in all_tasks} for i in range(self.num_layers)]
        route_prob_2 = [{tasks: None for tasks in all_tasks} for i in range(self.num_layers)]
        prompts_out = {task: self.task_prompts[:,i].repeat(feat[0][0].shape[0],1) for i,task in enumerate(self.p.TASKS.NAMES)}
        for idx, blk in enumerate(self.blocks):
            if idx + 1 in self.select_list:
                # extract task-specific feature at this layer
                il = np.sum(idx >= (np.array(self.select_list) - 1)) - 1 # [0,1,2]
                x, attn_weight = feat[il]
                _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il], prompts_out = self.cal_task_feature(x, attn_weight, il, info, episode_tasks, prompts_out)
                for task_list in episode_tasks:
                    input_prompt = [prompts_out[t] for t in task_list]
                    input_prompt = torch.stack(input_prompt, dim=1)
                    tasks = ''.join(task_list)
                    _cur_task_fea_now, route_feat_2[tasks], route_prob_2[il][tasks], new_prompts = self.MLoRE_2[il](_cur_task_fea[tasks], task_list, input_prompt)
                    for i, t in enumerate(task_list):
                        prompts_out[t] = new_prompts[:,i].clone()
                    x_q_list[tasks].append(x_q[tasks])
                    out_feat[tasks].append(_cur_task_fea_now)
            
        # extract task-specific feature at the last layer
        il=self.num_layers-1
        x, attn_weight = feat[il]
        _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il], prompts_out = self.cal_task_feature(x, attn_weight, il, info, episode_tasks, prompts_out)
        
        for task_list in episode_tasks:
            input_prompt = [prompts_out[t] for t in task_list]
            input_prompt = torch.stack(input_prompt, dim=1)
            tasks = ''.join(task_list)
            _cur_task_fea_now, route_feat_2[tasks], route_prob_2[il][tasks], new_prompts = self.MLoRE_2[il](_cur_task_fea[tasks], task_list, input_prompt)
            for i, t in enumerate(task_list):
                prompts_out[t] = new_prompts[:,i].clone()
            x_q_list[tasks].append(x_q[tasks])
            out_feat[tasks].append(_cur_task_fea_now)

            #TODO layer feature aggregation
            # now is the fixed average fusion (1/4), maybe learnable?
            for il in range(self.num_layers):
                last_feat[tasks] = last_feat[tasks] + (1./self.num_layers) * out_feat[tasks][il]

        
        info['route_1_prob'] = route_prob_1
        info['route_2_prob'] = route_prob_2
        

        ####decoder side
        for tasks in last_feat.keys():
            last_feat[tasks] = F.interpolate(last_feat[tasks], scale_factor=4, mode=INTERPOLATE_MODE)
        for task_list in episode_tasks:
            tasks = ''.join(task_list)
            for t in self.p.TASKS.NAMES:
                if t in task_list:
                    final_feat[t] = last_feat[tasks]

        return final_feat, info

    def forward(self, x, episode_tasks, eval=True):
        '''
        e.g.,
          episode_tasks: [['segmentation', 'depth'], [the other tasks]]
          task_index: [[0,2], [1,3,4]]
        '''
        x = self.patch_embed(x)
        x = self.pos_drop(x + self.pos_embed[:, 1:]) 

        # multi-scale backbone feature
        all_tasks = [''.join(tasks) for tasks in episode_tasks]
        out_feat = {tasks: [] for tasks in all_tasks}
        last_feat = {tasks: 0 for tasks in all_tasks}
        final_feat = {task: 0 for task in self.p.TASKS.NAMES}
        out_mask = {tasks: 0 for tasks in all_tasks}
        info = {} # pass information through the pipeline

        x_q_list = {tasks: [] for tasks in all_tasks}
        route_feat_1 = {tasks: None for tasks in all_tasks}
        route_feat_2 = {tasks: None for tasks in all_tasks}
        route_prob_1 = [{tasks: None for tasks in all_tasks} for i in range(self.num_layers)]
        route_prob_2 = [{tasks: None for tasks in all_tasks} for i in range(self.num_layers)]
        prompts_out = {task: self.task_prompts[:,i].repeat(x.shape[0],1) for i,task in enumerate(self.p.TASKS.NAMES)}
        for idx, blk in enumerate(self.blocks):
            x, attn_weight = blk(x)
            
            x, attn_weight = x.detach(), attn_weight
            if idx + 1 in self.select_list: 
                # extract task-specific feature at this layer
                il = np.sum(idx >= (np.array(self.select_list) - 1)) - 1 # [0,1,2]
                _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il], prompts_out = self.cal_task_feature(x, attn_weight, il, info, episode_tasks, prompts_out)
                for task_list in episode_tasks:
                    input_prompt = [prompts_out[t] for t in task_list]
                    input_prompt = torch.stack(input_prompt, dim=1)
                    tasks = ''.join(task_list)
                    _cur_task_fea_now, route_feat_2[tasks], route_prob_2[il][tasks], new_prompts = self.MLoRE_2[il](_cur_task_fea[tasks], task_list, input_prompt)
                    for i, t in enumerate(task_list):
                        prompts_out[t] = new_prompts[:,i].clone()
                    x_q_list[tasks].append(x_q[tasks])
                    out_feat[tasks].append(_cur_task_fea_now)
            
        x = self.norm(x).detach()
        # extract task-specific feature at the last layer
        il=self.num_layers-1
        _cur_task_fea, x_q, info, route_feat_1, route_prob_1[il], prompts_out = self.cal_task_feature(x, attn_weight, il, info, episode_tasks, prompts_out)
        
        for task_list in episode_tasks:
            input_prompt = [prompts_out[t] for t in task_list]
            input_prompt = torch.stack(input_prompt, dim=1)
            tasks = ''.join(task_list)
            _cur_task_fea_now, route_feat_2[tasks], route_prob_2[il][tasks], new_prompts = self.MLoRE_2[il](_cur_task_fea[tasks], task_list, input_prompt)
            for i, t in enumerate(task_list):
                prompts_out[t] = new_prompts[:,i].clone()
            x_q_list[tasks].append(x_q[tasks])
            out_feat[tasks].append(_cur_task_fea_now)

            #TODO layer feature aggregation
            # now is the fixed average fusion (1/4), maybe learnable?
            for il in range(self.num_layers):
                last_feat[tasks] = last_feat[tasks] + (1./self.num_layers) * out_feat[tasks][il]
            info['feat_precompress'] = last_feat[tasks].detach()
            

        info['route_1_prob'] = route_prob_1
        info['route_2_prob'] = route_prob_2
        

        ####decoder side
        for tasks in last_feat.keys():
            last_feat[tasks] = F.interpolate(last_feat[tasks], scale_factor=4, mode=INTERPOLATE_MODE)
        for task_list in episode_tasks:
            tasks = ''.join(task_list)
            for t in self.p.TASKS.NAMES:
                if t in task_list:
                    final_feat[t] = last_feat[tasks]

        return final_feat, info

    def cal_task_feature(self, x, attn_weight, il, info, episode_tasks, prompts):
        ''' Calculate task feature at this layer
        '''
        combined_fea = rearrange(x, 'b (h w) c -> b c h w', h=self.resolution[0], w=self.resolution[1])

        combined_fea = self.fea_fuse[il](combined_fea)#过了一层conv，对应特定layer的conv

        x_q = {''.join(tasks): 0 for tasks in episode_tasks}
        route_feat_out = {''.join(tasks): 0 for tasks in episode_tasks}
        route_prob_out = {''.join(tasks): 0 for tasks in episode_tasks}
        for tasks in episode_tasks:
            input_prompt = [prompts[t] for t in tasks]
            input_prompt = torch.stack(input_prompt, dim=1)
            x_q[''.join(tasks)], route_feat_out[''.join(tasks)], route_prob_out[''.join(tasks)], new_prompts = self.MLoRE_1[il](combined_fea, tasks, input_prompt)
            for i, t in enumerate(tasks):
                prompts[t] = new_prompts[:,i].clone()

        combined_fea = x_q

        return combined_fea, x_q, info, route_feat_out, route_prob_out, prompts


def _init_vit_weights(module: nn.Module, name: str = '', head_bias: float = 0., jax_impl: bool = False):
    """ ViT weight initialization
    * When called without n, head_bias, jax_impl args it will behave exactly the same
      as my original init for compatibility with prev hparam / downstream use cases (ie DeiT).
    * When called w/ valid n (module name) and jax_impl=True, will (hopefully) match JAX impl
    """
    if isinstance(module, nn.Linear):
        if name.startswith('head'):
            nn.init.zeros_(module.weight)
            nn.init.constant_(module.bias, head_bias)
        elif name.startswith('pre_logits'):
            lecun_normal_(module.weight)
            nn.init.zeros_(module.bias)
        else:
            if jax_impl:
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    if 'mlp' in name:
                        nn.init.normal_(module.bias, std=1e-6)
                    else:
                        nn.init.zeros_(module.bias)
            else:
                trunc_normal_(module.weight, std=.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    elif jax_impl and isinstance(module, nn.Conv2d):
        # NOTE conv was left to pytorch default in my original init
        lecun_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, (nn.LayerNorm, nn.GroupNorm, nn.BatchNorm2d)):
        nn.init.zeros_(module.bias)
        nn.init.ones_(module.weight)


@torch.no_grad()
def _load_weights(model, checkpoint_path: str, prefix: str = ''):
    """ Load weights from .npz checkpoints for official Google Brain Flax implementation
    """
    import numpy as np

    def _n2p(w, t=True):
        if w.ndim == 4 and w.shape[0] == w.shape[1] == w.shape[2] == 1:
            w = w.flatten()
        if t:
            if w.ndim == 4:
                w = w.transpose([3, 2, 0, 1])
            elif w.ndim == 3:
                w = w.transpose([2, 0, 1])
            elif w.ndim == 2:
                w = w.transpose([1, 0])
        return torch.from_numpy(w)

    w = np.load(checkpoint_path)
    if not prefix and 'opt/target/embedding/kernel' in w:
        prefix = 'opt/target/'

    if hasattr(model.patch_embed, 'backbone'):
        # hybrid
        backbone = model.patch_embed.backbone
        stem_only = not hasattr(backbone, 'stem')
        stem = backbone if stem_only else backbone.stem
        stem.conv.weight.copy_(adapt_input_conv(stem.conv.weight.shape[1], _n2p(w[f'{prefix}conv_root/kernel'])))
        stem.norm.weight.copy_(_n2p(w[f'{prefix}gn_root/scale']))
        stem.norm.bias.copy_(_n2p(w[f'{prefix}gn_root/bias']))
        if not stem_only:
            for i, stage in enumerate(backbone.stages):
                for j, block in enumerate(stage.blocks):
                    bp = f'{prefix}block{i + 1}/unit{j + 1}/'
                    for r in range(3):
                        getattr(block, f'conv{r + 1}').weight.copy_(_n2p(w[f'{bp}conv{r + 1}/kernel']))
                        getattr(block, f'norm{r + 1}').weight.copy_(_n2p(w[f'{bp}gn{r + 1}/scale']))
                        getattr(block, f'norm{r + 1}').bias.copy_(_n2p(w[f'{bp}gn{r + 1}/bias']))
                    if block.downsample is not None:
                        block.downsample.conv.weight.copy_(_n2p(w[f'{bp}conv_proj/kernel']))
                        block.downsample.norm.weight.copy_(_n2p(w[f'{bp}gn_proj/scale']))
                        block.downsample.norm.bias.copy_(_n2p(w[f'{bp}gn_proj/bias']))
        embed_conv_w = _n2p(w[f'{prefix}embedding/kernel'])
    else:
        embed_conv_w = adapt_input_conv(
            model.patch_embed.proj.weight.shape[1], _n2p(w[f'{prefix}embedding/kernel']))
    model.patch_embed.proj.weight.copy_(embed_conv_w)
    model.patch_embed.proj.bias.copy_(_n2p(w[f'{prefix}embedding/bias']))
    pos_embed_w = _n2p(w[f'{prefix}Transformer/posembed_input/pos_embedding'], t=False)
    if pos_embed_w.shape != model.pos_embed.shape:
        pos_embed_w = resize_pos_embed(  # resize pos embedding when different size from pretrained weights
            pos_embed_w, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
    model.pos_embed.copy_(pos_embed_w)
    model.norm.weight.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/scale']))
    model.norm.bias.copy_(_n2p(w[f'{prefix}Transformer/encoder_norm/bias']))
    for i, block in enumerate(model.blocks.children()):
        block_prefix = f'{prefix}Transformer/encoderblock_{i}/'
        mha_prefix = block_prefix + 'MultiHeadDotProductAttention_1/'
        block.norm1.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/scale']))
        block.norm1.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_0/bias']))
        block.attn.qkv.weight.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/kernel'], t=False).flatten(1).T for n in ('query', 'key', 'value')]))
        block.attn.qkv.bias.copy_(torch.cat([
            _n2p(w[f'{mha_prefix}{n}/bias'], t=False).reshape(-1) for n in ('query', 'key', 'value')]))
        block.attn.proj.weight.copy_(_n2p(w[f'{mha_prefix}out/kernel']).flatten(1))
        block.attn.proj.bias.copy_(_n2p(w[f'{mha_prefix}out/bias']))
        for r in range(2):
            getattr(block.mlp, f'fc{r + 1}').weight.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/kernel']))
            getattr(block.mlp, f'fc{r + 1}').bias.copy_(_n2p(w[f'{block_prefix}MlpBlock_3/Dense_{r}/bias']))
        block.norm2.weight.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/scale']))
        block.norm2.bias.copy_(_n2p(w[f'{block_prefix}LayerNorm_2/bias']))


def resize_pos_embed(posemb, posemb_new, num_tokens=1, gs_new=()):
    # Rescale the grid of position embeddings when loading from state_dict. Adapted from
    # https://github.com/google-research/vision_transformer/blob/00883dd691c63a6830751563748663526e811cee/vit_jax/checkpoint.py#L224
    _logger.info('Resized position embedding: %s to %s', posemb.shape, posemb_new.shape)
    ntok_new = posemb_new.shape[1]
    if num_tokens:
        posemb_tok, posemb_grid = posemb[:, :num_tokens], posemb[0, num_tokens:]
        ntok_new -= num_tokens
    else:
        posemb_tok, posemb_grid = posemb[:, :0], posemb[0]
    gs_old = int(math.sqrt(len(posemb_grid)))
    if not len(gs_new):  # backwards compatibility
        gs_new = [int(math.sqrt(ntok_new))] * 2
    assert len(gs_new) >= 2
    _logger.info('Position embedding grid-size from %s to %s', [gs_old, gs_old], gs_new)
    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=gs_new, mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new[0] * gs_new[1], -1)
    posemb = torch.cat([posemb_tok, posemb_grid], dim=1)
    return posemb


def checkpoint_filter_fn(state_dict, model):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    if 'model' in state_dict:
        # For deit models
        state_dict = state_dict['model']
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k and len(v.shape) < 4:
            # For old models that I trained prior to conv based patchification
            O, I, H, W = model.patch_embed.proj.weight.shape
            v = v.reshape(O, -1, H, W)
        elif k == 'pos_embed' and v.shape != model.pos_embed.shape:
            # To resize pos embedding when using model at different size from pretrained weights
            v = resize_pos_embed(
                v, model.pos_embed, getattr(model, 'num_tokens', 1), model.patch_embed.grid_size)
        out_dict[k] = v
    return out_dict


def _create_MLoRE(variant, pretrained=False, default_cfg=None, **kwargs):
    default_cfg = default_cfg or default_cfgs[variant]
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer models.')

    # NOTE this extra code to support handling of repr size for in21k pretrained models
    # default_num_classes = default_cfg['num_classes']
    # num_classes = kwargs.get('num_classes', default_num_classes)
    # repr_size = kwargs.pop('representation_size', None)
    # if repr_size is not None and num_classes != default_num_classes:
    #     # Remove representation layer if fine-tuning. This may not always be the desired action,
    #     # but I feel better than doing nothing by default for fine-tuning. Perhaps a better interface?
    #     _logger.warning("Removing representation layer for fine-tuning.")
    #     repr_size = None
    # print('npz' in default_cfg['file'])
    model = build_model_with_cfg(
        MLoRE, variant, pretrained,
        pretrained_cfg=default_cfg,
        # representation_size=repr_size,
        pretrained_filter_fn=checkpoint_filter_fn,
        # pretrained_custom_load='npz' in default_cfg['url'],
        **kwargs)
    return model


def MLoRE_vit_large_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Large model (ViT-L/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(6,24,6), patch_size=16, embed_dim=1024, depth=24, num_heads=16, **kwargs)
    model = _create_MLoRE('vit_large_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

def MLoRE_vit_base_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(3,12,3), patch_size=16, embed_dim=768, depth=12, num_heads=12,  **kwargs)
    model = _create_MLoRE('vit_base_patch16_384', pretrained=pretrained, **model_kwargs)
    return model

def MLoRE_vit_small_patch16_384(pretrained=False, **kwargs):
    """ Based on ViT-Small model (ViT-S/16) from original paper (https://arxiv.org/abs/2010.11929).
    ImageNet-1k weights fine-tuned from in21k @ 384x384, source https://github.com/google-research/vision_transformer.
    """
    model_kwargs = dict(select_list=range(3,12,3),patch_size=16, embed_dim=384, depth=12, num_heads=6, **kwargs)
    model = _create_MLoRE('vit_small_patch16_384', pretrained=pretrained, **model_kwargs)
    return model


class ConvHead(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        self.mt_proj = nn.Sequential(nn.Conv2d(in_channels, in_channels, 3, padding=1), BatchNorm2d(in_channels), nn.GELU())
        trunc_normal_(self.mt_proj[0].weight, std=0.02)

        self.linear_pred = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        nn.init.normal(self.linear_pred.bias, mean=0, std=0.02)

    def forward(self, x):
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

if __name__ == '__main__':
    img = torch.randn((3,224,224)).cuda()
    
