"""Metrics factory and lightweight cached accessors (no third-party cache).

This module provides lazy instantiation of image quality assessment (IQA) metrics
to avoid instantiating all metrics at import time. Metrics are created on-demand
and cached using a simple module-level dictionary for reuse.
"""

import torch
import torch.nn.functional as F
from torchvision.transforms import ToTensor, ToPILImage
import pyiqa
import clip
from PIL import Image
import os
from typing import Dict, Callable, Union


# Global configuration
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Per-metric function cache: name -> metric callable
# This cache stores instantiated metric functions to avoid repeated initialization
_METRIC_FUNC_CACHE: Dict[str, Callable] = {}


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
