"""Metrics factory and lightweight cached accessors (no third-party cache).

This module provides lazy instantiation of image quality assessment (IQA) metrics
to avoid instantiating all metrics at import time. Metrics are created on-demand
and cached using a simple module-level dictionary for reuse.
"""

import os
from collections import defaultdict
from typing import Dict, Callable, Union, Tuple

import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import ToTensor, ToPILImage
import pyiqa
import clip
from PIL import Image
import lpips
import numpy as np
from tqdm import tqdm

from mpcompress.metrics.utils import *

# Global configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Per-metric function cache: name -> metric callable
# This cache stores instantiated metric functions to avoid repeated initialization
_METRIC_FUNC_CACHE: Dict[str, Callable] = {}
_lpips_ours_model = None  # lazy cache
_det_model_cache: Dict[str, torch.nn.Module] = {}

def _tag_metric(func: Callable, scope: str) -> Callable:
    func._metric_scope = scope
    return func

def split_img_metrics(
    metrics: Dict[str, Callable],
) -> Tuple[Dict[str, Callable], Dict[str, Callable]]:
    """Split metrics into per-frame and directory-level groups."""
    frame_metrics: Dict[str, Callable] = {}
    dir_metrics: Dict[str, Callable] = {}
    for name, func in metrics.items():
        if getattr(func, "_metric_scope", "frame") == "dir":
            dir_metrics[name] = func
        else:
            frame_metrics[name] = func
    return frame_metrics, dir_metrics

def tensor2image(x: torch.Tensor) -> Image.Image:
    """Convert a PyTorch tensor to PIL Image.

    Args:
        x (torch.Tensor): Input tensor with values in [0, 1] range.
            Shape: [C, H, W] or [1, C, H, W] or [B, C, H, W].
            If batch dimension exists, only the first image is used.

    Returns:
        Image.Image: PIL Image object in RGB format.
    """
    return ToPILImage()(x.clamp_(0, 1).squeeze())

def read_image(filepath: str) -> torch.Tensor:
    """Read an image from file and convert to PyTorch tensor.

    Args:
        filepath (str): Path to the image file.

    Returns:
        torch.Tensor: Image tensor with shape [1, C, H, W] and values in [0, 1] range.
            The tensor is moved to the configured device (CPU or GPU).

    Raises:
        AssertionError: If the file does not exist.
    """
    assert os.path.isfile(filepath), f"File not found: {filepath}"
    img = Image.open(filepath).convert("RGB")
    return ToTensor()(img).unsqueeze(0).to(DEVICE)

def create_clip_sim_metric(name: str = "ViT-B/32") -> Callable:
    """Create CLIP image similarity metric function.

    Args:
        name (str, optional): CLIP model name. Defaults to "ViT-B/32".

    Returns:
        Callable: A function that computes CLIP feature similarity between two images.
            The function signature is:
            clip_sim(img1_obj: Union[str, torch.Tensor],
                     img2_obj: Union[str, torch.Tensor]) -> torch.Tensor
            Returns cosine similarity between CLIP-encoded image features.
    """
    model, preprocess = clip.load(name, device=DEVICE)

    def clip_sim(
        img1_obj: Union[str, torch.Tensor], img2_obj: Union[str, torch.Tensor]
    ) -> torch.Tensor:
        if isinstance(img1_obj, str):
            img1 = preprocess(Image.open(img1_obj)).unsqueeze(0).to(DEVICE)
            img2 = preprocess(Image.open(img2_obj)).unsqueeze(0).to(DEVICE)
        else:
            img1 = preprocess(tensor2image(img1_obj)).unsqueeze(0).to(DEVICE)
            img2 = preprocess(tensor2image(img2_obj)).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            f1 = model.encode_image(img1)
            f2 = model.encode_image(img2)
            return torch.nn.functional.cosine_similarity(f1, f2, dim=-1)

    return clip_sim

# MS-SSIM related metrics
# Global MS-SSIM metric instance for reuse
ms_ssim_metric = pyiqa.create_metric("ms_ssim", device=DEVICE)

def padded_ms_ssim(
    img1_obj: Union[str, torch.Tensor], img2_obj: Union[str, torch.Tensor]
) -> torch.Tensor:
    """Compute MS-SSIM metric with padding for small images.

    MS-SSIM requires a minimum input size of 161x161 pixels. This function
    automatically pads smaller images to meet this requirement.

    Args:
        img1_obj (Union[str, torch.Tensor]): First image, either a filepath string
            or a tensor with shape [1, C, H, W] and values in [0, 1].
        img2_obj (Union[str, torch.Tensor]): Second image, either a filepath string
            or a tensor with shape [1, C, H, W] and values in [0, 1].
            Must match img1_obj type and shape.

    Returns:
        torch.Tensor: MS-SSIM score between the two images. Higher values indicate
            better similarity. Range: [0, 1].

    Raises:
        AssertionError: If image shapes do not match.
    """
    if isinstance(img1_obj, str):
        img1 = read_image(img1_obj)
        img2 = read_image(img2_obj)
    else:
        img1, img2 = img1_obj, img2_obj

    assert img1.shape == img2.shape, (
        f"Image size mismatch: img1 {img1.shape} vs img2 {img2.shape}"
    )

    # MS-SSIM minimum input size is 161x161 pixels
    if img1.shape[-1] < 161 or img1.shape[-2] < 161:
        pad_h = max(0, 161 - img1.shape[-2])
        pad_w = max(0, 161 - img1.shape[-1])
        img1 = F.pad(img1, (0, pad_w, 0, pad_h))
        img2 = F.pad(img2, (0, pad_w, 0, pad_h))

    return ms_ssim_metric(img1, img2)


def padded_ms_ssim_db(
    img1_obj: Union[str, torch.Tensor], img2_obj: Union[str, torch.Tensor]
) -> torch.Tensor:
    """Compute MS-SSIM metric in decibel (dB) scale.

    Converts MS-SSIM score to dB using the formula: -10 * log10(1 - MS-SSIM).
    Higher values indicate better similarity.

    Args:
        img1_obj (Union[str, torch.Tensor]): First image, either a filepath string
            or a tensor with shape [1, C, H, W] and values in [0, 1].
        img2_obj (Union[str, torch.Tensor]): Second image, either a filepath string
            or a tensor with shape [1, C, H, W] and values in [0, 1].
            Must match img1_obj type and shape.

    Returns:
        torch.Tensor: MS-SSIM score in decibels. Higher values indicate better similarity.
    """
    return -10 * torch.log10(1 - padded_ms_ssim(img1_obj, img2_obj))

def _get_lpips_ours_model(device):
    global _lpips_ours_model
    if _lpips_ours_model is None:
        _lpips_ours_model = lpips.LPIPS(net="alex").to(device)
        _lpips_ours_model.eval()
    return _lpips_ours_model


def _load_image(obj, device):
    if isinstance(obj, str):
        img = Image.open(obj).convert("RGB")
        return ToTensor()(img).unsqueeze(0).to(device)
    else:
        return obj.to(device)


def lpips_ours(img1_obj, img2_obj, device=DEVICE):
    """
    LPIPS-Ours metric (pyiqa-compatible)

    Args:
        img1_obj: Tensor [1,3,H,W] in [0,1] or image path
        img2_obj: Tensor [1,3,H,W] in [0,1] or image path

    Returns:
        torch.Tensor with shape [1], LPIPS distance (lower is better)
    """

    img1 = _load_image(img1_obj, device)
    img2 = _load_image(img2_obj, device)

    assert img1.shape == img2.shape, \
        f"Image size mismatch: {img1.shape} vs {img2.shape}"

    model = _get_lpips_ours_model(device)

    with torch.no_grad():
        score = model(img1, img2)

    return score  # torch.Tensor [1]


def psnr_frames_metric(frame_dir: str, gt_dir: str) -> dict:
    """Compute per-prefix and global PSNR over image folders.

    Assumes paired filenames in frame_dir and gt_dir.
    """
    prefix_dict = defaultdict(list)
    all_scores = []

    for filename in sorted(os.listdir(frame_dir)):
        if not filename.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
            continue
        file1_path = os.path.join(frame_dir, filename)
        file2_path = os.path.join(gt_dir, filename)

        if not os.path.exists(file2_path):
            continue

        img1 = read_image(file1_path)
        img2 = read_image(file2_path)
        if img1.shape != img2.shape:
            continue

        mse = torch.mean((img1 - img2) ** 2)
        mse = torch.clamp(mse, min=1e-10)
        psnr = -10.0 * torch.log10(mse)

        prefix = filename.split("_")[0]
        prefix_dict[prefix].append(psnr.item())
        all_scores.append(psnr.item())

    results: dict = {}
    for prefix, values in prefix_dict.items():
        results[f"prefix_{prefix}"] = float(np.mean(values))

    results["global_avg"] = float(np.mean(all_scores)) if all_scores else 0.0
    results["count"] = len(all_scores)
    return results


def lpips_frames_metric(frame_dir: str, gt_dir: str) -> dict:
    """Compute per-prefix and global LPIPS distance over image folders.

    This mirrors the behavior in `main_test_metric_2.py`:
      - Pair images by filename between `frame_dir` and `gt_dir`
      - Compute LPIPS (alex backbone) per image pair
      - Aggregate by filename prefix (substring before the first "_")
    """
    prefix_dict = defaultdict(list)
    all_scores = []

    model = _get_lpips_ours_model(DEVICE)
    with torch.no_grad():
        for filename in os.listdir(frame_dir):
            file1_path = os.path.join(frame_dir, filename)
            file2_path = os.path.join(gt_dir, filename)
            if not os.path.exists(file2_path):
                continue

            try:
                img1 = Image.open(file1_path).convert("RGB")
                img2 = Image.open(file2_path).convert("RGB")
                if img1.size != img2.size:
                    continue

                img1_tensor = (
                    torch.tensor(np.array(img1))
                    .to(DEVICE)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )
                img2_tensor = (
                    torch.tensor(np.array(img2))
                    .to(DEVICE)
                    .permute(2, 0, 1)
                    .unsqueeze(0)
                    .float()
                    / 255.0
                )

                score = model(img1_tensor, img2_tensor)

                prefix = filename.split("_")[0]
                prefix_dict[prefix].append(score.item())
                all_scores.append(score.item())
            except Exception:
                continue

    results: dict = {}
    for prefix, values in prefix_dict.items():
        results[f"prefix_{prefix}"] = float(np.mean(values))

    results["global_avg"] = float(np.mean(all_scores)) if all_scores else 0.0
    results["count"] = len(all_scores)
    return results


def create_img_metrics(metric_names: Union[str, list] = None) -> Dict[str, Callable]:
    """Create image quality assessment (IQA) metrics dictionary.

    Heavy metrics are instantiated lazily per metric. This function uses a
    per-metric function cache so repeated requests for the same metric name
    reuse the cached callable.

    Supported metrics:
        - PSNR: Peak Signal-to-Noise Ratio
        - MS-SSIM: Multi-Scale Structural Similarity Index
        - MS-SSIM-dB: MS-SSIM in decibel scale
        - VIF: Visual Information Fidelity
        - GMSD: Gradient Magnitude Similarity Deviation
        - LPIPS-Alex: Learned Perceptual Image Patch Similarity (AlexNet backbone)
        - LPIPS-VGG: Learned Perceptual Image Patch Similarity (VGG backbone)
        - DISTS: Deep Image Structure and Texture Similarity
        - PieAPP: Perceptual Image-Error Assessment through Pairwise Preference
        - AHIQ: Attention-based Hybrid Image Quality
        - CLIP-SIM: CLIP-based image similarity
        - TOPIQ-FR: TopIQ Full Reference
        - TOPIQ-NR: TopIQ No Reference
        - MUSIQ: Multi-scale Image Quality Transformer
        - LPIPS-Frames: Folder-level LPIPS aggregated by filename prefix

    Args:
        metric_names (Union[str, list], optional): Name(s) of metrics to create.
            If None, all available metrics are created. If a string, a single metric
            is created. If a list, multiple metrics are created. Defaults to None.

    Returns:
        Dict[str, Callable]: Dictionary mapping metric names to their callable functions.
            Each callable takes two arguments (img1, img2) and returns a similarity score.

    Raises:
        ValueError: If any requested metric name is not supported.
    """
    metric_factories: Dict[str, Callable[[], Callable]] = {
        "PSNR": lambda: pyiqa.create_metric("psnr", device=DEVICE),
        "MS-SSIM": lambda: padded_ms_ssim,
        "MS-SSIM-dB": lambda: padded_ms_ssim_db,
        "VIF": lambda: pyiqa.create_metric("vif", device=DEVICE),
        "GMSD": lambda: pyiqa.create_metric("gmsd", device=DEVICE),
        "LPIPS-Alex": lambda: pyiqa.create_metric("lpips", device=DEVICE),
        "LPIPS-VGG": lambda: pyiqa.create_metric("lpips-vgg", device=DEVICE),
        "DISTS": lambda: pyiqa.create_metric("dists", device=DEVICE),
        "PieAPP": lambda: pyiqa.create_metric("pieapp", device=DEVICE),
        "AHIQ": lambda: pyiqa.create_metric("ahiq", device=DEVICE),
        "CLIP-SIM": lambda: create_clip_sim_metric("ViT-B/32"),
        "TOPIQ-FR": lambda: pyiqa.create_metric("topiq_fr", device=DEVICE),
        "TOPIQ-NR": lambda: pyiqa.create_metric("topiq_nr", device=DEVICE),
        "MUSIQ": lambda: pyiqa.create_metric("musiq", device=DEVICE),
        "LPIPS-Ours": lambda: lpips_ours,
        "Det-mAP@0.7": lambda: _tag_metric(
            create_detection_map_metric(
                target_classes=(1, 3), score_thr=0.5, iou_threshold=0.7
            ),
            "dir",
        ),
        "Det-Frames@0.7": lambda: _tag_metric(
            create_detection_frame_metrics(
                target_classes=(1, 3), score_thr=0.5, iou_threshold=0.7
            ),
            "dir",
        ),
        "PSNR-Frames": lambda: _tag_metric(psnr_frames_metric, "dir"),
        "LPIPS-Frames": lambda: _tag_metric(lpips_frames_metric, "dir"),
    }

    if metric_names is None:
        names = list(metric_factories.keys())
    elif isinstance(metric_names, str):
        names = [metric_names]
    else:
        names = list(metric_names)

    invalid_metrics = set(names) - set(metric_factories.keys())
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metric names: {invalid_metrics}. Available metrics: {list(metric_factories.keys())}"
        )

    def get_or_create_metric(name: str) -> Callable:
        if name not in _METRIC_FUNC_CACHE:
            _METRIC_FUNC_CACHE[name] = metric_factories[name]()
        return _METRIC_FUNC_CACHE[name]

    return {name: get_or_create_metric(name) for name in names}

def create_dist_metrics(metric_names: Union[str, list] = None) -> Dict[str, Callable]:
    """Create distribution-distance metrics dictionary with per-metric lazy cache.

    Distribution-distance metrics measure the distance between distributions of
    image features rather than pixel-level or perceptual similarity.

    Supported metrics:
        - FID: Fréchet Inception Distance

    Args:
        metric_names (Union[str, list], optional): Name(s) of metrics to create.
            If None, all available metrics are created. If a string, a single metric
            is created. If a list, multiple metrics are created. Defaults to None.

    Returns:
        Dict[str, Callable]: Dictionary mapping metric names to their callable functions.
            Note: Distribution metrics typically require batches of images rather than
            single image pairs.

    Raises:
        ValueError: If any requested metric name is not supported.
    """
    metric_factories: Dict[str, Callable[[], Callable]] = {
        "FID": lambda: pyiqa.create_metric("fid", device=DEVICE),
    }

    if metric_names is None:
        names = list(metric_factories.keys())
    elif isinstance(metric_names, str):
        names = [metric_names]
    else:
        names = list(metric_names)

    invalid_metrics = set(names) - set(metric_factories.keys())
    if invalid_metrics:
        raise ValueError(
            f"Unsupported dist metric names: {invalid_metrics}. Available: {list(metric_factories.keys())}"
        )

    def get_or_create_metric(name: str) -> Callable:
        if name not in _METRIC_FUNC_CACHE:
            _METRIC_FUNC_CACHE[name] = metric_factories[name]()
        return _METRIC_FUNC_CACHE[name]

    return {name: get_or_create_metric(name) for name in names}

def create_detection_frame_metrics(
    target_classes=(1, 3),
    score_thr: float = 0.5,
    iou_threshold: float = 0.7,
) -> Callable:
    """Return detection metrics over frames and classes.

    The callable expects (frame_dir, gt_dir) where gt_dir contains .pt labels.
    """

    def detection_frame_metrics(frame_dir: str, gt_dir: str) -> dict:
        device = DEVICE
        model = _get_det_model(device)

        class_ap_totals = {cid: 0.0 for cid in target_classes}
        class_counts = {cid: 0 for cid in target_classes}
        class_frame_aps = {cid: [] for cid in target_classes}

        frame_map = []
        frame_map_per_class = {cid: [] for cid in target_classes}

        label_path = gt_dir
        gt_list = sorted([f for f in os.listdir(label_path) if f.endswith(".pt")])
        num_classes = 90

        for gt_name in tqdm(gt_list, desc="Processing enhanced frames"):
            frame_path = os.path.join(frame_dir, gt_name.replace(".pt", ".png"))
            if not os.path.isfile(frame_path):
                frame_map.append(0.0)
                continue

            frame_tensor = torchvision.io.read_image(frame_path).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            gt = torch.load(os.path.join(label_path, gt_name))
            gt_boxes = gt["boxes"].numpy()
            gt_labels = gt["labels"].numpy()

            with torch.no_grad():
                pred = model(frame_tensor)

            boxes = pred[0]["boxes"].detach().cpu().numpy()
            labels = pred[0]["labels"].detach().cpu().numpy()
            scores = pred[0]["scores"].detach().cpu().numpy()

            keep = (scores > score_thr) & np.isin(labels, list(target_classes))
            pred_boxes = boxes[keep]
            pred_labels = labels[keep]
            pred_scores = scores[keep]

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                pred_mask = np.isin(pred_labels, list(target_classes))
                gt_mask = np.isin(gt_labels, list(target_classes))

                if np.any(pred_mask) and np.any(gt_mask):
                    mAP, ap_per_class = compute_map(
                        pred_boxes[pred_mask],
                        pred_labels[pred_mask],
                        pred_scores[pred_mask],
                        gt_boxes[gt_mask],
                        gt_labels[gt_mask],
                        num_classes,
                        iou_threshold=iou_threshold,
                    )
                    frame_map.append(float(mAP))
                    for cid in target_classes:
                        if cid in ap_per_class:
                            frame_map_per_class[cid].append(float(ap_per_class[cid]))
                else:
                    frame_map.append(0.0)
                    for cid in target_classes:
                        if np.any(gt_labels == cid):
                            frame_map_per_class[cid].append(0.0)
            else:
                frame_map.append(0.0)
                for cid in target_classes:
                    if len(gt_boxes) > 0 and np.any(gt_labels == cid):
                        frame_map_per_class[cid].append(0.0)

            for cid in target_classes:
                pred_mask = pred_labels == cid
                gt_mask = gt_labels == cid

                if np.any(pred_mask) and np.any(gt_mask):
                    ap = compute_ap_for_class(
                        pred_boxes[pred_mask],
                        pred_scores[pred_mask],
                        gt_boxes[gt_mask],
                    )
                    class_ap_totals[cid] += ap
                    class_counts[cid] += 1
                    class_frame_aps[cid].append(ap)
                elif np.any(gt_mask):
                    class_ap_totals[cid] += 0.0
                    class_counts[cid] += 1
                    class_frame_aps[cid].append(0.0)

        results: dict = {}
        if frame_map:
            results["overall_mAP"] = float(np.mean(frame_map))
            results["overall_mAP_std"] = float(np.std(frame_map))
            results["overall_mAP_max"] = float(np.max(frame_map))
            results["overall_mAP_min"] = float(np.min(frame_map))
            results["overall_mAP_median"] = float(np.median(frame_map))
        else:
            results["overall_mAP"] = 0.0
            results["overall_mAP_std"] = 0.0
            results["overall_mAP_max"] = 0.0
            results["overall_mAP_min"] = 0.0
            results["overall_mAP_median"] = 0.0

        for cid in target_classes:
            if frame_map_per_class[cid]:
                class_maps = frame_map_per_class[cid]
                results[f"class_{cid}_mAP"] = float(np.mean(class_maps))
                results[f"class_{cid}_mAP_std"] = float(np.std(class_maps))
                results[f"class_{cid}_mAP_count"] = len(class_maps)
            else:
                results[f"class_{cid}_mAP"] = 0.0
                results[f"class_{cid}_mAP_std"] = 0.0
                results[f"class_{cid}_mAP_count"] = 0

        total_samples = sum(class_counts.values())
        results["weighted_AP_avg"] = (
            float(sum(class_ap_totals.values()) / total_samples) if total_samples > 0 else 0.0
        )
        results["total_samples"] = total_samples

        for cid in target_classes:
            if class_counts[cid] > 0:
                avg_ap = class_ap_totals[cid] / class_counts[cid]
                class_aps = class_frame_aps[cid]
                results[f"class_{cid}_AP_avg"] = float(avg_ap)
                results[f"class_{cid}_AP_std"] = float(np.std(class_aps)) if len(class_aps) > 1 else 0.0
                results[f"class_{cid}_AP_max"] = float(np.max(class_aps))
                results[f"class_{cid}_AP_min"] = float(np.min(class_aps))
                results[f"class_{cid}_AP_median"] = float(np.median(class_aps))
                results[f"class_{cid}_AP_count"] = class_counts[cid]
            else:
                results[f"class_{cid}_AP_avg"] = 0.0
                results[f"class_{cid}_AP_std"] = 0.0
                results[f"class_{cid}_AP_max"] = 0.0
                results[f"class_{cid}_AP_min"] = 0.0
                results[f"class_{cid}_AP_median"] = 0.0
                results[f"class_{cid}_AP_count"] = 0

        return results

    return detection_frame_metrics


def _get_det_model(device: torch.device):
    key = str(device)
    if key not in _det_model_cache:
        model = load_detection_model(key)  # 你的 load_detection_model 期望的是 'cuda:0'/'cpu'
        model.eval()
        _det_model_cache[key] = model
    return _det_model_cache[key]


def create_detection_map_metric(
    target_classes=(1, 3),
    score_thr: float = 0.5,
    iou_threshold: float = 0.7,
) -> Callable:
    """
    返回一个 pyiqa 风格的 metric callable，但这里输入不是两张图，而是两个目录：
        metric(frame_dir, gt_dir) -> torch.Tensor([overall_mAP])

    说明：
    - frame_dir: 增强后帧（png）目录
    - gt_dir:   GT 标签（pt）目录，文件名与帧名同 stem（xxx.pt <-> xxx.png）
    """

    def detection_map_metric(frame_dir: str, gt_dir: str) -> torch.Tensor:
        device = DEVICE
        model = _get_det_model(device)

        frame_map = []
        frame_map_per_class = {cid: [] for cid in target_classes}

        label_path = gt_dir
        gt_list = sorted([f for f in os.listdir(label_path) if f.endswith(".pt")])

        # COCO 类别总数（你的原注释说 90，这里保持一致）
        num_classes = 90

        for gt_name in tqdm(gt_list, desc="Processing enhanced frames"):
            frame_path = os.path.join(frame_dir, gt_name.replace(".pt", ".png"))
            if not os.path.isfile(frame_path):
                # 缺帧按 0 处理（也可 raise）
                frame_map.append(0.0)
                continue

            frame_tensor = torchvision.io.read_image(frame_path).float() / 255.0
            frame_tensor = frame_tensor.unsqueeze(0).to(device)

            gt = torch.load(os.path.join(label_path, gt_name))
            gt_boxes = gt["boxes"].numpy()
            gt_labels = gt["labels"].numpy()

            with torch.no_grad():
                pred = model(frame_tensor)

            boxes = pred[0]["boxes"].detach().cpu().numpy()
            labels = pred[0]["labels"].detach().cpu().numpy()
            scores = pred[0]["scores"].detach().cpu().numpy()

            # 只保留关注类 & 置信度阈值
            keep = (scores > score_thr) & np.isin(labels, list(target_classes))
            pred_boxes = boxes[keep]
            pred_labels = labels[keep]
            pred_scores = scores[keep]

            if len(pred_boxes) > 0 and len(gt_boxes) > 0:
                pred_mask = np.isin(pred_labels, list(target_classes))
                gt_mask = np.isin(gt_labels, list(target_classes))

                if np.any(pred_mask) and np.any(gt_mask):
                    mAP, ap_per_class = compute_map(
                        pred_boxes[pred_mask],
                        pred_labels[pred_mask],
                        pred_scores[pred_mask],
                        gt_boxes[gt_mask],
                        gt_labels[gt_mask],
                        num_classes,
                        iou_threshold=iou_threshold,
                    )
                    frame_map.append(float(mAP))
                    for cid in target_classes:
                        if cid in ap_per_class:
                            frame_map_per_class[cid].append(float(ap_per_class[cid]))
                else:
                    frame_map.append(0.0)
            else:
                frame_map.append(0.0)

        overall = float(np.mean(frame_map)) if len(frame_map) > 0 else 0.0
        return torch.tensor([overall], device=device, dtype=torch.float32)

    return detection_map_metric
