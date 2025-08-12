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

from mpcompress.datasets import *
from mpcompress.heads import *
from mpcompress.models.mpc import *
from mpcompress.metrics import *


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


def calc_psnr(a, b):
    """计算PSNR"""
    mse = F.mse_loss(a, b).item()
    if mse < 1e-10:
        return 100.0
    return -10 * np.log10(mse)


def pad_centering(x, p):
    """中心填充"""
    h, w = x.size(2), x.size(3)
    new_h = (h + p - 1) // p * p
    new_w = (w + p - 1) // p * p
    padding_left = (new_w - w) // 2
    padding_right = new_w - w - padding_left
    padding_top = (new_h - h) // 2
    padding_bottom = new_h - h - padding_top
    x_padded = F.pad(
        x,
        (padding_left, padding_right, padding_top, padding_bottom),
        mode="constant",
        value=0,
    )
    return x_padded, (padding_left, padding_right, padding_top, padding_bottom)


def unpad_centering(x, padding):
    """移除中心填充"""
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


def calc_real_bpp_items(out_enc, x):
    """计算实际比特率"""
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    dino_byte_len = [len(s[0]) for s in out_enc["dino"]["strings"]]
    byte_len_list = {
        "y": sum(dino_byte_len[:-1]),
        "z": dino_byte_len[-1],
        "vq": len(out_enc["vqgan"]["strings"][0]),
    }
    bpp_items = {
        f"bpp_{name}": (length * 8.0 / num_pixels)
        for name, length in byte_len_list.items()
    }
    return bpp_items


def calc_esti_bpp_items(out_net, x):
    """计算估计比特率"""
    import math
    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp_items = {
        f"bpp_{name}": (
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        ).item()
        for name, likelihoods in out_net["likelihoods"].items()
    }
    return bpp_items


@torch.inference_mode()
def inference_file(
    model, x, real=False, recon=2, return_lnclf=False, return_mmseg=False
):
    """推理单个文件"""
    if real:  # 实际压缩
        start = time.time()
        out_enc = model.compress(x)
        enc_time = time.time() - start
        start = time.time()
        out_net = model.decompress(
            out_enc,
            return_rec1=(recon == 1),
            return_rec2=(recon == 2),
            return_lnclf=return_lnclf,
            return_mmseg=return_mmseg,
        )
        dec_time = time.time() - start
        bpp_items = calc_real_bpp_items(out_enc, x)
        bpp = sum(bpp_items.values())
        org_result = {
            "enc_time": enc_time,
            "dec_time": dec_time,
            "bpp": bpp,
        }
    else:  # 估计
        start = time.time()
        out_net = model.forward_test(
            x,
            return_rec1=(recon == 1),
            return_rec2=(recon == 2),
            return_lnclf=return_lnclf,
            return_mmseg=return_mmseg,
        )
        elapsed_time = time.time() - start
        bpp_items = calc_esti_bpp_items(out_net, x)
        bpp = sum(bpp_items.values())
        org_result = {
            "enc_time": elapsed_time / 2.0,  # 粗略估计
            "dec_time": elapsed_time / 2.0,
            "bpp": bpp,
        }

    org_result.update(bpp_items)
    return org_result, out_net

@torch.inference_mode()
def eval_model(cfg):
    """评估模型的主函数"""
    device = torch.device(cfg.args.device)
    model = instantiate_class(cfg.model).to(device)
    checkpoint = torch.load(cfg.args.checkpoint, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    model.eval()

    metric_meter = DictAverageMeter()
    records = []

    # 从配置中获取任务信息
    task_name = cfg.args.task
    assert task_name in cfg.eval_tasks, f"任务 {task_name} 不存在"
    task_config = cfg.eval_tasks[task_name]
    
    # 获取数据集、头部和指标配置
    dataset_config = cfg.datasets[task_config.dataset]
    head_config = cfg.heads[task_config.head]
    metric_config = cfg.metrics[task_config.metric]
    
    print(f"任务: {task_name}")
    print(f"描述: {task_config.description}")
    print(f"数据集: {task_config.dataset}")
    print(f"头部: {task_config.head}")
    print(f"指标: {task_config.metric}")

    # 构建数据集
    print("构建数据集...")
    dataset = instantiate_class(dataset_config)
    print(f"✓ 数据集大小: {len(dataset)}")

    # 构建头部模型和指标
    cls_head = None
    cls_metric = None
    seg_head = None
    seg_metric = None
    
    # 根据任务类型构建相应的头部和指标
    if "cls" in task_name:
        print("构建分类头部...")
        cls_head = instantiate_class(head_config).to(device).eval()
        cls_metric = instantiate_class(metric_config)
    elif "seg" in task_name:
        print("构建分割头部...")
        seg_head = instantiate_class(head_config).to(device).eval()
        seg_metric = instantiate_class(metric_config)

    # 创建输出目录
    if cfg.args.output_dir:
        out_sub_dir = f"{cfg.args.output_dir}/{cfg.args.quality}"
        os.makedirs(out_sub_dir, exist_ok=True)

    # 评估循环
    for i, data in enumerate(dataset):
        # 统一数据格式: (img, img_meta)
        if isinstance(data, tuple) and len(data) == 2:
            x, img_meta = data
        else:
            # 兼容旧格式
            x = data
            img_meta = {"img_path": f"image_{i}"}

        # 模型推理
        if hasattr(model, 'update'):
            model.update()
        
        x = ToTensor()(x).to(device)
        x = x.unsqueeze(0) if x.dim() == 3 else x
        x_padded, padding = pad_centering(x, 128)
        
        out_result, out_net = inference_file(
            model,
            x_padded,
            real=cfg.args.real,
            recon=cfg.args.recon,
            return_lnclf=cls_head is not None,
            return_mmseg=seg_head is not None,
        )

        # 计算图像质量指标
        iqa_result = {}
        if cfg.args.recon != 0:
            x_hat = out_net["rec2" if cfg.args.recon == 2 else "rec1"]
            x_hat = x_hat.clamp(0, 1)
            x_hat = unpad_centering(x_hat, padding)
            
            if cfg.args.output_dir:
                stem = os.path.splitext(os.path.basename(img_meta["img_path"]))[0]
                fout = os.path.join(out_sub_dir, f"{stem}.png")
                img = torch2img(x_hat)
                img.save(fout)

            # 计算PSNR
            iqa_result["PSNR"] = calc_psnr(x_hat, x)

        # 更新分类指标
        if "cls" in task_name:
            logits = cls_head.forward(out_net["lnclf"])
            cls_preds = F.softmax(logits, dim=1)
            values, top_indices = torch.topk(cls_preds, k=5, dim=1)
            cls_metric.update(top_indices, [img_meta["target"]])

        # 更新分割指标
        if "seg" in task_name:
            logits = seg_head.predict(out_net["mmseg"], rescale=True)
            logits = unpad_centering(logits, padding)
            seg_preds = F.softmax(logits, dim=1).argmax(dim=1).squeeze(0)
            seg_preds = seg_preds.cpu().numpy() # [H, W]

            seg_label_path = img_meta["seg_label_path"]
            seg_label = np.array(Image.open(seg_label_path)) # [H, W]
            seg_metric.update(seg_preds, seg_label)

        # 记录结果
        file = img_meta["img_path"]
        record = {"file": file, "quality": cfg.args.quality}
        record.update(out_result)
        record.update(iqa_result)
        
        metric_meter.update(record)
            
        if cfg.args.verbose:
            # _rv = {key: round(value, 4) for key, value in per_file_record.items()}
            print(file, record)
            
        # per_file_record = {key: round(value, 8) for key, value in per_file_record.items()}
        records.append(record)

    # 计算平均值 
    avg_metrics = metric_meter.average()

    # 添加分类和分割指标
    if cls_metric:
        cls_results = cls_metric.compute()
        avg_metrics.update(cls_results)
        
    if seg_metric:
        seg_results = seg_metric.compute()
        avg_metrics.update(seg_results)

    avg_metrics = {key: round(value, 6) for key, value in avg_metrics.items()}
    return avg_metrics, records


def torch2img(x: torch.Tensor) -> Image.Image:
    """将tensor转换为PIL图像"""
    return ToPILImage()(x.clamp(0, 1).squeeze()) 

def setup_args():
    """设置命令行参数"""
    parser = argparse.ArgumentParser(description="统一MPC模型评估脚本")
    parser.add_argument("--config", type=str, required=True, help="统一配置文件路径")
    parser.add_argument("--checkpoint", type=str, required=True, help="模型检查点路径")
    parser.add_argument("--task", type=str, required=True, help="评估任务名称")
    parser.add_argument("--quality", type=str, default="12.0", help="质量参数")
    parser.add_argument("--real", action="store_true", help="使用实际压缩")
    parser.add_argument("--recon", type=int, default=2, choices=[0, 1, 2], help="重建层")
    parser.add_argument("--verbose", action="store_true", help="详细输出")
    parser.add_argument("--cuda", action="store_true", help="使用CUDA")
    parser.add_argument("--output_dir", type=str, default="", help="输出目录")
    return parser


def merge_args_to_config(config, args):
    """将命令行参数合并到配置中"""
    # 创建运行时配置
    config.args = OmegaConf.create({
        "checkpoint": args.checkpoint,
        "task": args.task,
        "quality": args.quality,
        "real": args.real,
        "recon": args.recon,
        "verbose": args.verbose,
        "cuda": args.cuda,
        "output_dir": args.output_dir,
        "device": "cuda" if args.cuda and torch.cuda.is_available() else "cpu"
    })    
    return config


def main():
    """主函数"""
    parser = setup_args()
    args = parser.parse_args()
    
    # 加载配置
    config = OmegaConf.load(args.config)
    
    # 将命令行参数合并到配置中
    config = merge_args_to_config(config, args)
    
    # 运行评估（所有实例化都在eval_model内部完成）
    print(f"【{args.task}】开始评估...")
    avg_metrics, records = eval_model(config)
    
    # 输出结果
    print(f"\n【{args.task}】评估结果:")
    print("=" * 50)
    for key, value in avg_metrics.items():
        print(f"{key}: {value}")
    
    # 保存结果
    if config.args.output_dir:
        result_file = os.path.join(config.args.output_dir, f"{args.task}_results_{config.args.quality}.json")
        result = {
            "task": args.task,
            "quality": config.args.quality,
            "description": config.eval_tasks[args.task].description,
            "metrics": avg_metrics,
            "records": records
        }
        with open(result_file, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\n结果已保存到: {result_file}")


if __name__ == "__main__":
    main() 