"""
统一的评估脚本，支持从配置文件中指定任务
"""

import os
import sys
import json
import time
import argparse
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
import numpy as np
from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
import importlib
import shutil
import tqdm
import math
import warnings
from torch.utils.data import DataLoader
from tqdm import tqdm
from torchvision.transforms import (
    Compose,
    CenterCrop,
    ToTensor,
    PILToTensor,
    RandomResizedCrop,
    RandomCrop,
)

from mpcompress.datasets import *
from mpcompress.heads import *
from mpcompress.models import *
from mpcompress.models.mpc import MPC_I2_Separate
from mpcompress.metrics import *
from mpcompress.metrics.iqa_metrics import create_img_metrics, create_dist_metrics
from mpcompress.utils.tensor_ops import tensor2image, center_pad, center_crop
from mpcompress.utils.utils import rename_key_by_rules
from mpcompress.utils.debug import extract_shapes


# Disable Warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)


def get_obj_from_str(string, reload=False):
    if "." in string:
        module, cls = string.rsplit(".", 1)
        if reload:
            module_imp = importlib.import_module(module)
            importlib.reload(module_imp)
        return getattr(importlib.import_module(module, package=None), cls)
    else:
        return getattr(sys.modules[__name__], string)


def instantiate_class(config, **kwargs):
    config = config.copy()
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        return obj(**config, **kwargs)
    else:
        print(config)
        raise KeyError("Expected key `type` to instantiate.")


def instantiate_transforms(config, **kwargs):
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        if "transforms" in config:
            transforms_list = config.pop("transforms")
            transforms = []
            for t in transforms_list:
                transforms.append(instantiate_class(t))
            return obj(transforms=transforms, **config, **kwargs)
        return obj(**config, **kwargs)
    else:
        raise KeyError("Expected key `type` to instantiate.")


@torch.inference_mode()
def eval_model(cfg):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = instantiate_class(cfg.model).to(device)
    model.eval()

    train_transforms = Compose([
        RandomCrop(512, pad_if_needed=True),
        PILToTensor(),
    ])

    # train_transforms = instantiate_transforms(cfg.train_transforms)
    train_dataset = instantiate_class(cfg.train_dataset, transform=train_transforms)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg.misc.batch_size,
        num_workers=cfg.misc.num_workers,
        shuffle=True,
        pin_memory=False,
    )

    # 创建输出目录
    os.makedirs("extract", exist_ok=True)
    
    pbar = tqdm(train_dataloader)
    for i, data in enumerate(pbar):
        if i >= 400800:
            break
        data = data.to(device)
        x = data / 255.0
        res = model.extract_feature(x)
        reduce_pt = {
            "tokens": res["tokens"][0].to(torch.int16),
            "h_dino": res["h_dino"][0].to(torch.float16),
            "x_uint8": data[0],
        }
        torch.save(reduce_pt, f"extract/{i:08d}.pt")


def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="MPC模型评估脚本")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        nargs="+",  # 允许接收一个或多个参数
        help="配置文件路径，可以指定多个文件",
    )
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")
    return parser


def merge_args_to_config(config, args):
    """将命令行参数合并到配置中"""
    args_dict = dict(vars(args))
    args_dict["device"] = "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    config.args = OmegaConf.create(args_dict)
    return config


def main(config, args):
    eval_model(config)


if __name__ == "__main__":
    """主函数"""
    parser = setup_args()
    args = parser.parse_args()

    # 加载配置
    config = OmegaConf.load(args.config[0])
    for config_path in args.config[1:]:
        overlay_config = OmegaConf.load(config_path)
        config = OmegaConf.merge(config, overlay_config)

    # 将命令行参数合并到配置中
    config = merge_args_to_config(config, args)
    main(config, args)


"""
CUDA_VISIBLE_DEVICES=1 python examples/mpc/run_eval_mpc_extract.py --config examples/mpc/train_config/MPC12-v3-large-extract.yaml
"""