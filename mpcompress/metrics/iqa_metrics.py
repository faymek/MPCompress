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


def tensor2image(x: torch.Tensor) -> Image.Image:
    return ToPILImage()(x.clamp_(0, 1).squeeze())


def read_image(filepath: str) -> torch.Tensor:
    assert os.path.isfile(filepath), f"File not found: {filepath}"
    img = Image.open(filepath).convert("RGB")
    return ToTensor()(img).unsqueeze(0).to(DEVICE)


def create_clip_sim_metric(name: str = "ViT-B/32") -> Callable:
    """Create CLIP image similarity metric function.

    Args:
        name: CLIP model name

    Returns:
        Function that computes CLIP feature similarity between two images
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
ms_ssim_metric = pyiqa.create_metric("ms_ssim", device=DEVICE)


def padded_ms_ssim(
    img1_obj: Union[str, torch.Tensor], img2_obj: Union[str, torch.Tensor]
) -> torch.Tensor:
    """Compute MS-SSIM metric with padding for small images <161 pixels."""
    if isinstance(img1_obj, str):
        img1 = read_image(img1_obj)
        img2 = read_image(img2_obj)
    else:
        img1, img2 = img1_obj, img2_obj

    assert img1.shape == img2.shape, (
        f"Image size mismatch: img1 {img1.shape} vs img2 {img2.shape}"
    )

    # MS-SSIM minimum input size is 161x161
    if img1.shape[-1] < 161 or img1.shape[-2] < 161:
        pad_h = max(0, 161 - img1.shape[-2])
        pad_w = max(0, 161 - img1.shape[-1])
        img1 = F.pad(img1, (0, pad_w, 0, pad_h))
        img2 = F.pad(img2, (0, pad_w, 0, pad_h))

    return ms_ssim_metric(img1, img2)


def padded_ms_ssim_db(
    img1_obj: Union[str, torch.Tensor], img2_obj: Union[str, torch.Tensor]
) -> torch.Tensor:
    return -10 * torch.log10(1 - padded_ms_ssim(img1_obj, img2_obj))


def create_img_metrics(metric_names: Union[str, list] = None) -> Dict[str, Callable]:
    all_metrics = {
        "PSNR": pyiqa.create_metric("psnr", device=DEVICE),
        "MS-SSIM": padded_ms_ssim,
        "MS-SSIM-dB": padded_ms_ssim_db,
        "VIF": pyiqa.create_metric("vif", device=DEVICE),
        "GMSD": pyiqa.create_metric("gmsd", device=DEVICE),
        "LPIPS-Alex": pyiqa.create_metric("lpips", device=DEVICE),
        "LPIPS-VGG": pyiqa.create_metric("lpips-vgg", device=DEVICE),
        "DISTS": pyiqa.create_metric("dists", device=DEVICE),
        "PieAPP": pyiqa.create_metric("pieapp", device=DEVICE),
        "AHIQ": pyiqa.create_metric("ahiq", device=DEVICE),
        "CLIP-SIM": create_clip_sim_metric("ViT-B/32"),
        "TOPIQ-FR": pyiqa.create_metric("topiq_fr", device=DEVICE),  # higher is better
        "TOPIQ-NR": pyiqa.create_metric("topiq_nr", device=DEVICE),
        "MUSIQ": pyiqa.create_metric("musiq", device=DEVICE),
    }

    if metric_names is None:
        return all_metrics

    if isinstance(metric_names, str):
        metric_names = [metric_names]

    # Validate metric names
    invalid_metrics = set(metric_names) - set(all_metrics.keys())
    if invalid_metrics:
        raise ValueError(
            f"Unsupported metric names: {invalid_metrics}. Available metrics: {list(all_metrics.keys())}"
        )

    return {name: all_metrics[name] for name in metric_names}


def create_dist_metrics() -> Dict[str, Callable]:
    return {
        "FID": pyiqa.create_metric("fid", device=DEVICE),
    }
