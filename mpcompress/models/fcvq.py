"""
Dinov2 + FCVQ wrapper for feature compression and classification evaluation.

This module provides a lightweight "codec-style" class (similar in spirit to
lamofc.py's Dinov2OrigSlideOnlyPatchCodec) but built on the FCVQ model
defined in fcvq_model.py.

Typical usage:
    codec = Dinov2FCVQClsCodec(
        compressor_type="FCVQ",
        compressor_kwargs=dict(num_embeddings=8, embedding_dim=64, num_chunks=1, lmbda=1.0),
        dino_kwargs=dict(layers=1, pretrained=True),
        freeze_dino=True,
    )

Training:
    feat_recon, mse_loss, rd_loss, rate, encoding_inds = codec(feat)

Testing bits:
    feat_hat, mse_loss, strings, encoding_inds = codec.compress(feat)
    feat_hat = codec.decompress(strings, feat.shape)
    logits = codec.forward_decode(feat_hat)
"""
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple, Union
import types
import torch
import torch.nn as nn
from mpcompress.latent_codecs.fcvq_model import FCVQ
import math
import itertools
import warnings
from functools import partial
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=FutureWarning)
import mmcv
from mmcv.parallel import collate, scatter
from mmcv.runner import load_checkpoint
from mmseg.apis import init_segmentor
from mmseg.datasets.pipelines import Compose
from mmseg.ops import resize
from mpcompress.backbone.dinov2.hub.classifiers import dinov2_vitg14_lc
from mpcompress.backbone.dinov2.hub.backbones import dinov2_vitg14


class Dinov2FCVQCodec(nn.Module):

    def __init__(
        self,
        fcvq_kwargs: Optional[Dict[str, Any]] = None,
        build_dino: bool = True,
        dino_kwargs: Optional[Dict[str, Any]] = None,
        freeze_dino: bool = True,
    ) -> None:
        super().__init__()

        fcvq_kwargs = fcvq_kwargs or {}
        # if not hasattr(FCVQ, "FCVQ"):
        #     raise ImportError("Can't find FCVQ from fcvq_model.py ")

        self.fcvq: nn.Module = FCVQ(**fcvq_kwargs)

        self.dino = None
        if build_dino:
            
            dino_kwargs = dino_kwargs or {"layers": 1, "pretrained": True}
            self.dino = dinov2_vitg14_lc(**dino_kwargs)

            if freeze_dino:
                self.dino.eval()
                for p in self.dino.parameters():
                    p.requires_grad_(False)

    def forward(
        self, feat: torch.Tensor, **kwargs
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """
        return:
            feat_hat, mse_loss, rd_loss, rate, encoding_inds
        """
        feat_hat, mse_loss, rd_loss, rate, encoding_inds = self.fcvq(feat, **kwargs)
        return feat_hat, mse_loss, rd_loss, rate, encoding_inds

    @torch.no_grad()
    def forward_decode(self, feat: torch.Tensor) -> torch.Tensor:

        if self.dino is None:
            raise RuntimeError("Set build_dino=True")
        return self.dino.forward_decode(feat)

    @torch.no_grad()
    def compress(
        self,
        feat: torch.Tensor,
        get_ready: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[bytes], Any]:
        """
        return:
            feat_hat, mse_loss, strings, encoding_inds
        """
        if get_ready and hasattr(self.fcvq, "uncondi_entropy_model"):
            self.fcvq.uncondi_entropy_model.get_ready_for_compression()

        feat_hat, mse_loss, strings, encoding_inds = self.fcvq.compress(feat)
        coded_unit = {"strings": {"indices": [strings], 
                    },
                    "pstate":{"feat_shape": feat_hat.shape},
                    }
        return coded_unit

    @torch.no_grad()
    def decompress(self, strings: List[bytes], feat_shape: Union[Tuple[int, ...], torch.Size]) -> torch.Tensor:
        task_feats = self.fcvq.decompress(strings, tuple(feat_shape))
        return task_feats

    def state_dict_compressor(self) -> Dict[str, Any]:
        """save fcvq parameters only"""
        return self.fcvq.state_dict()

    def load_state_dict_compressor(self, state_dict: Dict[str, Any], strict: bool = True) -> None:
        """load fcvq only"""
        self.fcvq.load_state_dict(state_dict, strict=strict)


    def seg_eval_from_feature_files(
    self,
    load_vq: bool = False,
    vq_path: Optional[str] = None,
    *,
    list_file: str = "/code/examples/fcvq/val_100.txt",
    img_root: str = "/data/bitahub/VOC2012",
    feat_dir: str = "/data/qiaoxichen/model/dinov2_dataset/seg/test",
    feat_aug_dir: str = "/data/qiaoxichen/model/dinov2_dataset/seg/test",
    head_dataset: str = "voc2012",
    head_type: str = "linear",
    num_classes: int = 21,
    device: str = "cuda",
    ) -> Dict[str, Any]:
        """Segmentation mIoU eval using feature files (original vs FCVQ recon).
        Pipeline aligned with test_bits_seg.py but simplified and device-safe.
        """
        device_t = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # optional: load compressor weights from checkpoint (for standalone testing)
        if load_vq:
            if vq_path is None:
                raise ValueError("load_vq=True but vq_path is None")
            ckpt = torch.load(vq_path, map_location="cpu")
            if isinstance(ckpt, dict) and "vqvae_state_dict" in ckpt:
                self.load_state_dict_compressor(ckpt["vqvae_state_dict"])
            else:
                self.load_state_dict_compressor(ckpt)

        self.eval().to(device_t)
        # -------------------------
        # utils
        # -------------------------
        def unpack_dinov2(pack_feat: np.ndarray, N: int, C: int, H: int, W: int) -> np.ndarray:
            # (N*H, C*W) -> (N, C, H, W)
            return pack_feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)

        def fast_hist(label: np.ndarray, pred: np.ndarray, n: int) -> np.ndarray:
            k = (label >= 0) & (label < n)
            return np.bincount(n * label[k].astype(int) + pred[k].astype(int), minlength=n * n).reshape(n, n)

        def per_class_iu(hist: np.ndarray) -> np.ndarray:
            return np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist) + 1e-10)

        class CenterPadding(torch.nn.Module):
            def __init__(self, multiple: int):
                super().__init__()
                self.multiple = multiple

            def _get_pad(self, size: int):
                new_size = math.ceil(size / self.multiple) * self.multiple
                pad = new_size - size
                left = pad // 2
                right = pad - left
                return left, right

            @torch.inference_mode()
            def forward(self, x):
                pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
                return F.pad(x, pads)

        class LoadImage:
            def __call__(self, results):
                img = results["img"]
                results["filename"] = None
                results["ori_filename"] = None
                results["img"] = img
                results["img_shape"] = img.shape
                results["ori_shape"] = img.shape
                results["pad_shape"] = img.shape
                results["scale_factor"] = 1.0
                results["img_norm_cfg"] = dict(
                    mean=np.zeros(img.shape[2], dtype=np.float32),
                    std=np.ones(img.shape[2], dtype=np.float32),
                    to_rgb=False,
                )
                return results

        # -------------------------
        # mmseg wrapper methods (minimal)
        # -------------------------
        def encode_decode_decode(self_seg, crop_feature_list, img_metas, backbone_model, shape_hw):
            # align to decode head device (critical for BN/LN)
            head_dev = next(self_seg.decode_head.parameters()).device

            # backbone norm may sit elsewhere; move once if needed
            if hasattr(backbone_model, "norm") and hasattr(backbone_model.norm, "weight"):
                if backbone_model.norm.weight.device != head_dev:
                    backbone_model.norm = backbone_model.norm.to(head_dev)

            outputs = [backbone_model.norm(out.to(head_dev)) for out in crop_feature_list]
            outputs = [out[:, 1 + backbone_model.num_register_tokens :] for out in outputs]

            B = outputs[0].shape[0]
            w, h = shape_hw[0], shape_hw[1]
            outputs = [
                out.reshape(
                    B,
                    math.ceil(w / backbone_model.patch_size),
                    math.ceil(h / backbone_model.patch_size),
                    -1,
                ).permute(0, 3, 1, 2).contiguous()
                for out in outputs
            ]

            x = tuple(outputs)
            if self_seg.with_neck:
                x = self_seg.neck(x)

            # ensure head inputs are on head_dev (BN safety)
            x = tuple(t.to(head_dev) for t in x)

            out = self_seg._decode_head_forward_test(x, img_metas)
            out = resize(input=out, size=shape_hw, mode="bilinear", align_corners=self_seg.align_corners)
            return out

        def slide_inference_decode(self_seg, feature_list, img_meta, rescale, backbone_model):
            head_dev = next(self_seg.decode_head.parameters()).device

            h_stride, w_stride = self_seg.test_cfg.stride
            h_crop, w_crop = self_seg.test_cfg.crop_size
            batch_size = feature_list[0][0].shape[0]
            h_img, w_img = img_meta[0]["img_shape"][0], img_meta[0]["img_shape"][1]
            num_classes_local = self_seg.num_classes

            h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
            w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1

            preds = torch.zeros((batch_size, num_classes_local, h_img, w_img), device=head_dev)
            count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=head_dev)

            i = 0
            for h_idx in range(h_grids):
                for w_idx in range(w_grids):
                    y1 = h_idx * h_stride
                    x1 = w_idx * w_stride
                    y2 = min(y1 + h_crop, h_img)
                    x2 = min(x1 + w_crop, w_img)
                    y1 = max(y2 - h_crop, 0)
                    x1 = max(x2 - w_crop, 0)

                    crop_seg_logit = self_seg.encode_decode_decode(
                        feature_list[i], img_meta, backbone_model, self_seg.test_cfg.crop_size
                    )
                    if crop_seg_logit.device != head_dev:
                        crop_seg_logit = crop_seg_logit.to(head_dev)

                    preds += F.pad(
                        crop_seg_logit,
                        (int(x1), int(preds.shape[3] - x2), int(y1), int(preds.shape[2] - y2)),
                    )
                    count_mat[:, :, y1:y2, x1:x2] += 1
                    i += 1

            preds = preds / count_mat

            if rescale:
                resize_shape = img_meta[0]["img_shape"][:2]
                preds = preds[:, :, : resize_shape[0], : resize_shape[1]]
                preds = resize(
                    preds,
                    size=img_meta[0]["ori_shape"][:2],
                    mode="bilinear",
                    align_corners=self_seg.align_corners,
                    warning=False,
                )
            return preds

        def simple_test_decode(self_seg, feature_list, img_meta, backbone_model, rescale=True):
            seg_logit = self_seg.slide_inference_decode(feature_list, img_meta, rescale, backbone_model=backbone_model)
            output = F.softmax(seg_logit, dim=1)

            if img_meta[0]["flip"]:
                fd = img_meta[0]["flip_direction"]
                if fd == "horizontal":
                    output = output.flip(dims=(3,))
                elif fd == "vertical":
                    output = output.flip(dims=(2,))

            seg_pred = output.argmax(dim=1).cpu().numpy()
            return [seg_pred[0]]

        def create_segmenter(cfg, backbone_model):
            # IMPORTANT: this import registers DinoVisionTransformer into mmseg registry (same as test_bits_seg.py)
            import mpcompress.backbone.dinov2.eval.segmentation.models  # noqa: F401

            cfg.model.backbone.out_indices = [39]
            cfg.model.decode_head.in_index = [0]

            segm_local = init_segmentor(cfg)
            segm_local.backbone.forward = partial(
                backbone_model.get_intermediate_layers,
                n=cfg.model.backbone.out_indices,
                reshape=True,
            )
            if hasattr(backbone_model, "patch_size"):
                segm_local.backbone.register_forward_pre_hook(
                    lambda _, x: CenterPadding(backbone_model.patch_size)(x[0])
                )
            segm_local.init_weights()
            return segm_local

        # -------------------------
        # device
        # -------------------------
        device_t = device if (device == "cuda" and torch.cuda.is_available()) else "cpu"

        # -------------------------
        # load compressor weights
        # -------------------------
        if load_vq:
            if vq_path is None:
                raise ValueError("load_vq=True but vq_path is None")
            ckpt = torch.load(vq_path, map_location="cpu")
            if isinstance(ckpt, dict) and "vqvae_state_dict" in ckpt:
                self.load_state_dict_compressor(ckpt["vqvae_state_dict"])
            else:
                self.load_state_dict_compressor(ckpt)

        self.eval().to(device_t)

        # -------------------------
        # build backbone + segmenter
        # -------------------------
        backbone_model = dinov2_vitg14(pretrained=True).to(device_t).eval()

        cfg = mmcv.Config.fromfile(f"/code/examples/fcvq/cfg/dinov2_vitg14_{head_dataset}_{head_type}_config.py")
        segm = create_segmenter(cfg, backbone_model=backbone_model)

        # load seg head weights (CRITICAL: map to device_t, not cpu)
        DINOV2_BASE_URL = "https://dl.fbaipublicfiles.com/dinov2"
        backbone_name = "dinov2_vitg14"
        head_checkpoint_url = f"{DINOV2_BASE_URL}/{backbone_name}/{backbone_name}_{head_dataset}_{head_type}_head.pth"
        load_checkpoint(segm, head_checkpoint_url, map_location=device_t)

        # after load, enforce device (safety)
        segm = segm.to(device_t).eval()
        if hasattr(segm, "decode_head"):
            segm.decode_head = segm.decode_head.to(device_t)

        # bind methods
        segm.slide_inference_decode = types.MethodType(slide_inference_decode, segm)
        segm.encode_decode_decode = types.MethodType(encode_decode_decode, segm)
        segm.simple_test_decode = types.MethodType(simple_test_decode, segm)

        # pipeline + list
        test_pipeline = Compose([LoadImage()] + cfg.data.test.pipeline[1:])
        with open(list_file, "r") as f:
            image_list = "".join(f.readlines()).strip("\n").splitlines()

        # -------------------------
        # eval loop
        # -------------------------
        hist = np.zeros((num_classes, num_classes), dtype=np.float64)
        hist_recon = np.zeros((num_classes, num_classes), dtype=np.float64)
        mse_list, mse_list_recon = [], []
        bits_total = 0

        with torch.no_grad():
            for image_name in tqdm(image_list):
                # load image + label
                image = Image.open(f"{img_root}/JPEGImages/{image_name}.jpg")
                label = Image.open(f"{img_root}/SegmentationClass/{image_name}.png")
                array_label = np.array(label)

                img = np.array(image)[:, :, ::-1]  # BGR
                data = test_pipeline(dict(img=img))
                data = collate([data], samples_per_gpu=1)

                if device_t == "cuda":
                    data = scatter(data, [device_t])[0]
                else:
                    data["img_metas"] = [i.data[0] for i in data["img_metas"]]

                img_metas = data["img_metas"][0]

                # load features
                org_feat = np.load(f"{feat_dir}/{image_name}.npy")
                aug_np = np.load(f"{feat_aug_dir}/{image_name}.npy")
                aug_np = np.clip(aug_np, -5, 5)

                # (packed -> [2,1,1370,1536])
                aug_np_unpack = unpack_dinov2(aug_np, 2, 1, 1370, 1536)
                aug_t = torch.from_numpy(aug_np_unpack).to(device_t)

                # ---- ori seg ----
                aug_list = [
                    [aug_t[vi][cj].unsqueeze(0) for cj in range(aug_t.shape[1])]
                    for vi in range(aug_t.shape[0])
                ]
                pred = segm.simple_test_decode(aug_list, img_metas, backbone_model, rescale=True)
                hist += fast_hist(array_label, pred[0], num_classes)
                mse_list.append(float(np.square(org_feat - aug_np).mean()))

                # ---- compress/decompress ----
                aug_sq = aug_t.squeeze(1)  # [2,1370,1536]
                coded = self.compress(aug_sq)
                strings = coded["strings"]["indices"][0]
                feat_shape = coded["pstate"]["feat_shape"]

                bits_total += sum(len(s) for s in strings) * 8

                feat_hat = self.decompress(strings, feat_shape)  # [2,1370,1536]
                recon = feat_hat.unsqueeze(1)  # [2,1,1370,1536]

                aug_list_recon = [
                    [recon[vi][cj].unsqueeze(0) for cj in range(recon.shape[1])]
                    for vi in range(recon.shape[0])
                ]

                pred_recon = segm.simple_test_decode(aug_list_recon, img_metas, backbone_model, rescale=True)
                hist_recon += fast_hist(array_label, pred_recon[0], num_classes)

                recon_np = recon.detach().cpu().numpy()
                mse_list_recon.append(float(np.square(org_feat - recon_np).mean()))

        all_miou = float(np.nanmean(per_class_iu(hist)))
        all_miou_recon = float(np.nanmean(per_class_iu(hist_recon)))
        bpp_average = float(bits_total / (len(image_list) * 2740 * 1536))

        return dict(
            miou_ori=all_miou,
            miou_recon=all_miou_recon,
            mse_ori=float(np.mean(mse_list)) if mse_list else float("nan"),
            mse_recon=float(np.mean(mse_list_recon)) if mse_list_recon else float("nan"),
            bpp_avg=bpp_average,
        )
