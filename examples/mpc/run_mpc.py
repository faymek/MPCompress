"""
Evaluate an end-to-end compression model on an image dataset.
"""

import os
import re
import subprocess
import sys
import time
import glob
import math
import json
import argparse
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F

from PIL import Image
from torchvision.transforms import ToPILImage, ToTensor
from pytorch_msssim import ms_ssim

import compressai
import compressai.models
from compressai.registry import MODELS
import pyiqa
import importlib
from omegaconf import OmegaConf
from mpcompress.models.mpc import *
from mpcompress.heads import Dinov2ClassifierHead

# from dinov2.eval.call_linear import run_linear_s_4, run_linear_s_1
# import dinov2.distributed as distributed
import clip
import numpy as np
from mpcompress.utils.utils import extract_shapes


# torch.backends.cudnn.benchmark = True
# distributed.enable(overwrite=True)

for alias, cls in list(MODELS.items()):
    MODELS[cls.__name__] = cls


torch.backends.cudnn.deterministic = True
torch.set_num_threads(1)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def ms_ssim_db(x, y):
    v = -10 * math.log10(1 - ms_ssim(x, y, data_range=1.0))
    return torch.Tensor([v])


def torch2img(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath)
    img = Image.open(filepath).convert("RGB")
    return ToTensor()(img).to(DEVICE)


def create_clip_sim_metric(name="ViT-B/32", device="cuda"):
    model, preprocess = clip.load(name, device=device)

    def clip_sim(img1_obj, img2_obj):
        if isinstance(img1_obj, str):
            img1 = preprocess(Image.open(img1_obj)).unsqueeze(0).to(device)
            img2 = preprocess(Image.open(img2_obj)).unsqueeze(0).to(device)
        elif isinstance(img1_obj, torch.Tensor):
            img1 = preprocess(torch2img(img1_obj)).unsqueeze(0).to(device)
            img2 = preprocess(torch2img(img2_obj)).unsqueeze(0).to(device)
        with torch.no_grad():
            f1 = model.encode_image(img1)
            f2 = model.encode_image(img2)
            cos_sim = torch.nn.functional.cosine_similarity(f1, f2, dim=-1)
        return cos_sim

    return clip_sim


ms_ssim_metric = pyiqa.create_metric("ms_ssim", device=DEVICE)


def calc_ms_ssim(img1_obj, img2_obj):
    if isinstance(img1_obj, str):
        img1 = read_image(img1_obj)
        img2 = read_image(img2_obj)
    elif isinstance(img1_obj, torch.Tensor):
        img1 = img1_obj
        img2 = img2_obj
    # 如果图像尺寸小于160,填充到161
    # 确保两张图片尺寸相同
    assert img1.shape == img2.shape, (
        f"图片尺寸不匹配: img1 {img1.shape} vs img2 {img2.shape}"
    )
    if img1.shape[-1] < 161 or img1.shape[-2] < 161:
        # ms ssim 最小输入 161*161
        pad_h = max(0, 161 - img1.shape[-2])
        pad_w = max(0, 161 - img1.shape[-1])
        img1 = F.pad(img1, (0, pad_w, 0, pad_h))
        img2 = F.pad(img2, (0, pad_w, 0, pad_h))
    return ms_ssim_metric(img1, img2)


def calc_ms_ssim_db(img1_obj, img2_obj):
    return -10 * torch.log10(1 - calc_ms_ssim(img1_obj, img2_obj))


# https://github.com/chaofengc/IQA-PyTorch/blob/main/docs/ModelCard.md
img_metrics = {
    "PSNR": pyiqa.create_metric("psnr", device=DEVICE),
    "MS-SSIM": calc_ms_ssim,
    "MS-SSIM-dB": calc_ms_ssim_db,
    "VIF": pyiqa.create_metric("vif", device=DEVICE),
    "GMSD": pyiqa.create_metric("gmsd", device=DEVICE),
    "LPIPS-Alex": pyiqa.create_metric("lpips", device=DEVICE),
    "LPIPS-VGG": pyiqa.create_metric("lpips-vgg", device=DEVICE),
    "DISTS": pyiqa.create_metric("dists", device=DEVICE),
    "PieAPP": pyiqa.create_metric("pieapp", device=DEVICE),
    "AHIQ": pyiqa.create_metric("ahiq", device=DEVICE),
    "CLIP-SIM": create_clip_sim_metric("ViT-B/32", device=DEVICE),
    "TOPIQ-FR": pyiqa.create_metric("topiq_fr", device=DEVICE),  # higher is better
    "TOPIQ-NR": pyiqa.create_metric("topiq_nr", device=DEVICE),
    "MUSIQ": pyiqa.create_metric("musiq", device=DEVICE),
}
dist_metrics = {
    "FID": pyiqa.create_metric("fid", device=DEVICE),
}


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
    if "type" in config:
        cls = config.pop("type")
        obj = get_obj_from_str(cls)
        return obj(**config, **kwargs)
    else:
        raise KeyError(
            "When instantiating config, expected key `type` to be present: "
            + str(config)
        )


def reglob_collect_images(rootpath):
    file_list = glob.glob(os.path.join(rootpath, "**/*.*"), recursive=True)
    formats = "jpg|jpeg|png|ppm|bmp|pgm|tif|tiff|webp"
    pattern = f"(?i)([^\\s]+(\\.({formats}))$)"
    result = filter(re.compile(pattern).match, file_list)
    return sorted(result)


def calc_psnr(a, b):
    mse = F.mse_loss(a, b).item()
    # if close to zero, then psnr 100
    if mse < 1e-10:
        return 100.0
    return -10 * math.log10(mse)


def pad_centering(x, p):
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
    return F.pad(
        x,
        (-padding[0], -padding[1], -padding[2], -padding[3]),
    )


@torch.no_grad()
def inference_real(model, x, fout="", recon=2, lnclf_head=None):
    x = x.unsqueeze(0)
    x_padded, padding = pad_centering(x, 128)

    start = time.time()
    out_enc = model.compress(x_padded)
    enc_time = time.time() - start

    start = time.time()
    out_dec = model.decompress(
        out_enc, return_rec1=(recon == 1), return_rec2=(recon == 2), return_lnclf=True
    )
    dec_time = time.time() - start

    iqa_result = {}
    if recon != 0:
        x_hat = out_dec["rec2" if recon == 2 else "rec1"]
        x_hat.clamp_(0, 1)
        x_hat = unpad_centering(x_hat, padding)

        if fout:
            img = torch2img(x_hat)
            img.save(fout)

        iqa_result = {key: func(x_hat, x).item() for key, func in img_metrics.items()}

    if lnclf_head:
        predict_lables = lnclf_head.predict(out_dec["lnclf"], topk=5)
        # convert to list and save to txt
        predict_lables = predict_lables.tolist()
        with open(fout + ".lnclf.txt", "w") as f:
            for label in predict_lables:
                f.write(f"{label}\n")

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
    bpp = sum(byte_len_list.values()) * 8.0 / num_pixels

    org_result = {
        "enc_time": enc_time,
        "dec_time": dec_time,
        "bpp": bpp,
    }
    org_result.update(bpp_items)
    org_result.update(iqa_result)
    return org_result


@torch.inference_mode()
def inference_esti(model, x, fout="", recon=2, lnclf_head=None):
    x = x.unsqueeze(0)
    x_padded, padding = pad_centering(x, 128)

    start = time.time()
    out_net = model.forward_test(
        x_padded, return_rec1=(recon == 1), return_rec2=(recon == 2), return_lnclf=True
    )
    elapsed_time = time.time() - start

    iqa_result = {}
    if recon != 0:
        x_hat = out_net["rec2" if recon == 2 else "rec1"]
        x_hat.clamp_(0, 1)
        x_hat = unpad_centering(x_hat, padding)

        if fout:
            img = torch2img(x_hat)
            img.save(fout)

        iqa_result = {key: func(x_hat, x).item() for key, func in img_metrics.items()}

    if lnclf_head:
        predict_lables = lnclf_head.predict(out_net["lnclf"], topk=5)
        # convert to list and save to txt
        predict_lables = predict_lables.tolist()
        with open(fout + ".lnclf.txt", "w") as f:
            for label in predict_lables:
                f.write(f"{label}\n")

    num_pixels = x.size(0) * x.size(2) * x.size(3)
    bpp_items = {
        f"bpp_{name}": (
            torch.log(likelihoods).sum() / (-math.log(2) * num_pixels)
        ).item()
        for name, likelihoods in out_net["likelihoods"].items()
    }
    bpp = sum(bpp_items.values())

    org_result = {
        "enc_time": elapsed_time / 2.0,  # broad estimation
        "dec_time": elapsed_time / 2.0,
        "bpp": bpp,
    }
    org_result.update(bpp_items)
    org_result.update(iqa_result)
    return org_result


def eval_model(model, quality, args, lnclf_head=None):
    device = next(model.parameters()).device
    avg_metrics = defaultdict(float)
    records = []

    filepaths = reglob_collect_images(args.input_dir)
    if len(filepaths) == 0:
        print(f"Error: no images found in {args.input_dir}.", file=sys.stderr)
        raise SystemExit(1)
    if args.output_dir:
        out_sub_dir = f"{args.output_dir}/{quality}"
        os.makedirs(out_sub_dir, exist_ok=True)

    for file in filepaths:
        fout = ""
        if args.output_dir:
            stem = os.path.splitext(os.path.basename(file))[0]
            fout = os.path.join(out_sub_dir, f"{stem}.png")
        x = read_image(file).to(device)
        if args.half:
            model = model.half()
            x = x.half()
        if args.real:
            model.update()
            rv = inference_real(model, x, fout, recon=args.recon, lnclf_head=lnclf_head)
        else:
            rv = inference_esti(model, x, fout, recon=args.recon, lnclf_head=lnclf_head)
        for k, v in rv.items():
            avg_metrics[k] += v

        record = {"file": file, "quality": quality}
        if args.verbose:
            _rv = {key: round(value, 4) for key, value in rv.items()}
            print(file, _rv)
        rv = {key: round(value, 8) for key, value in rv.items()}
        record.update(rv)
        # print(record)
        records.append(record)
    for k, v in avg_metrics.items():
        avg_metrics[k] = v / len(filepaths)
        avg_metrics[k] = round(avg_metrics[k], 6)
    if args.recon != 0:
        iqa_result = {
            key: func(args.input_dir, out_sub_dir) for key, func in dist_metrics.items()
        }
        avg_metrics.update(iqa_result)
    return avg_metrics, records


def setup_args():
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-a",
        "--arch",
        type=str,
        help="model architecture",
        required=False,
    )
    parser.add_argument(
        "-c",
        "--entropy-coder",
        choices=compressai.available_entropy_coders(),
        default=compressai.available_entropy_coders()[0],
        help="entropy coder (default: %(default)s)",
    )
    parser.add_argument("--cuda", action="store_true", help="enable CUDA")
    parser.add_argument(
        "--half",
        action="store_true",
        help="convert model to half floating point (fp16)",
    )
    parser.add_argument(
        "--real",
        action="store_true",
        help="use evaluated entropy estimation (no entropy coding)",
    )
    parser.add_argument("-exp", "--experiment", type=str, help="experiment name")
    parser.add_argument(
        "-p",
        "--checkpoint",
        dest="checkpoints",
        type=str,
        nargs="*",
        help="checkpoint path list",
    )
    parser.add_argument(
        "-q",
        "--quality",
        dest="qualities",
        nargs="*",
        required=True,
        help="quality labels correspoding to the checkpoint path list",
    )
    parser.add_argument("-i", "--input-dir", type=str)
    parser.add_argument("-o", "--output-dir", type=str, default="")
    parser.add_argument("-r", "--result", type=str, help="result file path")
    parser.add_argument("-v", "--verbose", action="store_true", help="verbose mode")
    parser.add_argument("--label", default="", type=str, help="name label")
    parser.add_argument(
        "--recon", type=int, choices=[0, 1, 2], default=2, help="recon which layer"
    )
    parser.add_argument("--lnclf", action="store_true", help="use linear classifier")
    parser.add_argument("--mmseg", action="store_true", help="use mmsegmentation")
    return parser


class Experiment:
    def __init__(self, exp_name, quality):
        self.exp_name = exp_name
        self.quality = quality
        self.config_file = f"exp/{exp_name}/{quality}/config.yaml"
        self.ckpt_file = f"exp/{exp_name}/{quality}/runner.last.pth.tar"


def main(argv):
    parser = setup_args()
    args = parser.parse_args(argv)
    compressai.set_entropy_coder(args.entropy_coder)

    # if args.arch in MODELS:
    #     net_cls = MODELS[args.arch]
    # else:
    #     net_cls = find_and_import_class(args.arch)

    if args.experiment:
        args.checkpoints = [
            f"exp/{args.experiment}/{q}/runner.last.pth.tar" for q in args.qualities
        ]

    assert len(args.checkpoints) == len(args.qualities), (
        "checkpoint and quality labels mismatch"
    )

    all_avg_metrics = defaultdict(list)
    all_records = []
    for ckpt_file, quality in zip(args.checkpoints, args.qualities):
        if args.verbose:
            print(f"\nEvaluating ckpt: {ckpt_file}")
        config_file = f"exp/{args.experiment}/{quality}/config.yaml"
        config_file = os.path.abspath(config_file)
        ckpt_file = os.path.abspath(ckpt_file)
        print(config_file)
        print(ckpt_file)
        if not os.path.exists(config_file):
            print(f"Error: config file {config_file} does not exist.")
            raise SystemExit(1)
        cfg = OmegaConf.load(config_file)
        model: nn.Module = instantiate_class(cfg.model).to(DEVICE)
        checkpoint = torch.load(ckpt_file, map_location="cpu")
        sd_org = checkpoint["state_dict"]
        model.load_state_dict(sd_org, strict=True)
        model.eval()
        # model = net_cls(N=192)
        if args.cuda and torch.cuda.is_available():
            model = model.to("cuda")
        model.eval()

        if args.lnclf:
            base_path = os.path.realpath(
                os.path.join(os.path.dirname(__file__), "../../data")
            )
            head_checkpoint_path = (
                f"{base_path}/models/clf_head/dinov2_vits14_linear4_head.pth"
            )
            lnclf_head = Dinov2ClassifierHead(384, 4, head_checkpoint_path)
            lnclf_head.eval()
            if args.cuda and torch.cuda.is_available():
                lnclf_head = lnclf_head.to("cuda")

        avg_metrics, records = eval_model(model, quality, args, lnclf_head)
        mem = torch.cuda.max_memory_allocated(device=None)  # bytes
        print(f"After eval_model, GPU mem: \t{mem / (2**30)}GB")

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        all_records.extend(records)
        for k, v in avg_metrics.items():
            all_avg_metrics[k].append(v)

    if args.verbose:
        print()

    used_coder = "estimation" if not args.real else args.entropy_coder
    result = {
        "name": f"{args.experiment} {args.label}",
        "description": f"coding: {used_coder}, cuda: {args.cuda}, quality: {args.qualities}",
        "results": all_avg_metrics,
    }
    print(json.dumps(result, indent=2))
    result["records"] = all_records
    if args.result:
        with open(args.result, "w") as f:
            json.dump(result, f, indent=2)




if __name__ == "__main__":
    main(sys.argv[1:])
