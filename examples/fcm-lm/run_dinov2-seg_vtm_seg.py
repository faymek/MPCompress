import os
import math
import types
import itertools
import logging
from functools import partial
from PIL import Image

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

# mmcv/mmengine相关
from mmcv.image import imread
from mmcv.transforms import Compose
from mmseg.models.utils import resize
from mmengine.runner import load_checkpoint
from mmengine.logging import MMLogger
from mmengine.dataset import default_collate
from mmengine.config import Config
from mmengine.runner import Runner

# 项目本地相关
# import dinov2.eval.segmentation.models
from mpcompress.backbone.dinov2.hub.backbones import dinov2_vitg14

from mmengine.model import BaseModule
from mmseg.registry import MODELS
from mmseg.models.builder import HEADS
from mmseg.apis import init_model
from mmseg.models.decode_heads.decode_head import BaseDecodeHead
from einops import rearrange
from typing import Tuple, List
from torch import Tensor
from mmseg.utils import ConfigType
from omegaconf import OmegaConf
from mmcv.transforms import ToTensor
from mmseg.datasets import PascalVOCDataset


def extract_shapes(nested_structure):
    if isinstance(nested_structure, torch.Tensor):
        return tuple(nested_structure.shape)  # 返回 Tensor 的形状
    elif isinstance(nested_structure, list):
        return [extract_shapes(item) for item in nested_structure]  # 递归处理列表
    elif isinstance(nested_structure, tuple):
        return [extract_shapes(item) for item in nested_structure]  # 递归处理列表
    else:
        return nested_structure  # 其他类型直接返回

# 日志与警告设置
logger = MMLogger.get_instance("mmcv")
logger.setLevel("WARNING")    # Disable mmcv info print

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs

try:
    from mmseg.apis import init_segmentor
    MODELS = None
except ImportError:
    from mmseg.registry import MODELS
    init_segmentor = None


class CenterPadding(torch.nn.Module):
    def __init__(self, multiple):
        super().__init__()
        self.multiple = multiple

    def _get_pad(self, size):
        new_size = math.ceil(size / self.multiple) * self.multiple
        pad_size = new_size - size
        pad_size_left = pad_size // 2
        pad_size_right = pad_size - pad_size_left
        return pad_size_left, pad_size_right

    @torch.inference_mode()
    def forward(self, x):
        pads = list(itertools.chain.from_iterable(self._get_pad(m) for m in x.shape[:1:-1]))
        output = F.pad(x, pads)
        return output


@MODELS.register_module()
class DinoVisionBackbone(BaseModule):
    def __init__(self, model_size="large", img_size=512, patch_size=14, out_indices=[39], final_norm=False, checkpoint=None, **kwargs):
        super().__init__()
        assert model_size in ["small", "base", "large", "giant"]
        if model_size == "large":
            self.model = dinov2_vitg14(pretrained=True, weights=checkpoint, **kwargs)
        else:
            raise ValueError(f"Invalid model size: {model_size}")
        self.pad = CenterPadding(patch_size)
        self.model_size = model_size
        self.img_size = img_size
        self.patch_size = patch_size
        self.out_indices = out_indices
        self.final_norm = final_norm

    def forward(self, x):
        x = self.pad(x)
        # x: [1, 3, 518, 518] -> [1, 1370, 1536]
        outputs = self.feature_start_part(x)
        # some compression here
        token_res = (x.shape[2]//self.patch_size, x.shape[3]//self.patch_size)
        outputs = self.feature_end_part(outputs, token_res=token_res)
        return outputs
    

    def feature_start_part(
        self,
        x,
        n=1,  # Layers or n last layers to take
        reshape=False,
        return_class_token=False,
        norm=False, # do not perform norm in backbone, moved it to head
    ):
        if self.model.chunked_blocks:
            outputs = self.model._get_intermediate_layers_chunked(x, n)
        else:
            outputs = self.model._get_intermediate_layers_not_chunked(x, n)

        return outputs
    
    def feature_end_part(
        self,
        outputs,
        n=1,  # Layers or n last layers to take
        token_res=None,
        reshape=False,
        return_class_token=False,
        norm=False, # do not perform norm in backbone, moved it to head
    ):
        outputs = [self.model.norm(out) for out in outputs]

        # Remove class tokens and retain patch tokens
        # outputs: [1, 1370, 1536] -> [1, 1369, 1536]
        outputs = [out[:, 1 + self.model.num_register_tokens:] for out in outputs]

        # Reshape and reorder dimensions for compatibility, moved it from backbone to head
        # outputs: [1, 1369, 1536] -> [1, 1536, 37, 37]
        w, h = token_res
        outputs = [
            rearrange(out, 'b (h w) c -> b c h w', h=h, w=w)
            for out in outputs
        ]

        return outputs
    


@HEADS.register_module()
class PretrainedBNHead(BaseDecodeHead):
    """Just a batchnorm."""

    def __init__(self, resize_factors=None, patch_size=16, checkpoint=None, **kwargs):
        super().__init__(**kwargs)
        assert self.in_channels == self.channels
        self.bn = nn.SyncBatchNorm(self.in_channels)
        self.resize_factors = resize_factors
        self.patch_size = patch_size
        if checkpoint is not None:
            state_dict = torch.load(checkpoint)["state_dict"]
            state_dict = {
                k.replace("decode_head.", ""): v for k, v in state_dict.items()
            }
            self.load_state_dict(
                state_dict,
                strict=True
            )

    def _forward_feature(self, inputs):
        """Forward function for feature maps before classifying each pixel with
        ``self.cls_seg`` fc.

        Args:
            inputs (list[Tensor]): List of multi-level img features.

        Returns:
            feats (Tensor): A tensor of shape (batch_size, self.channels,
                H, W) which is feature map for last layer of decoder head.
        """
        # print("inputs", [i.shape for i in inputs])
        x = self._transform_inputs(inputs)
        feats = self.bn(x)
        return feats

    def _transform_inputs(self, inputs):
        """Transform inputs for decoder.
        Args:
            inputs (list[Tensor]): List of multi-level img features.
        Returns:
            Tensor: The transformed inputs
        """

        if self.input_transform == "resize_concat":
            # accept lists (for cls token)
            input_list = []
            for x in inputs:
                if isinstance(x, list):
                    input_list.extend(x)
                else:
                    input_list.append(x)
            inputs = input_list
            # an image descriptor can be a local descriptor with resolution 1x1
            for i, x in enumerate(inputs):
                if len(x.shape) == 2:
                    inputs[i] = x[:, :, None, None]
            # select indices
            inputs = [inputs[i] for i in self.in_index]
            # Resizing shenanigans
            # print("before", *(x.shape for x in inputs))
            if self.resize_factors is not None:
                assert len(self.resize_factors) == len(inputs), (
                    len(self.resize_factors),
                    len(inputs),
                )
                inputs = [
                    resize(
                        input=x, scale_factor=f, mode="bilinear" if f >= 1 else "area"
                    )
                    for x, f in zip(inputs, self.resize_factors)
                ]
                # print("after", *(x.shape for x in inputs))
            upsampled_inputs = [
                resize(
                    input=x,
                    size=inputs[0].shape[2:],
                    mode="bilinear",
                    align_corners=self.align_corners,
                )
                for x in inputs
            ]
            # actually no size is changed
            inputs = torch.cat(upsampled_inputs, dim=1)
        elif self.input_transform == "multiple_select":
            inputs = [inputs[i] for i in self.in_index]
        else:
            inputs = inputs[self.in_index]

        return inputs

    def forward(self, inputs):
        """Forward function."""
        output = self._forward_feature(inputs)
        output = self.cls_seg(output)
        return output
    
    def predict(self, inputs: Tuple[Tensor], batch_img_metas: List[dict],
                test_cfg: ConfigType) -> Tensor:
        """Forward function for prediction.

        Args:
            inputs (Tuple[Tensor]): List of multi-level img features.
            batch_img_metas (dict): List Image info where each dict may also
                contain: 'img_shape', 'scale_factor', 'flip', 'img_path',
                'ori_shape', and 'pad_shape'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:PackSegInputs`.
            test_cfg (dict): The testing config.


        Returns:
            Tensor: Outputs segmentation logits map.
        """
        # print("PretrainedBNHead.predict inputs", extract_shapes(inputs))
        # print("PretrainedBNHead.predict inputs[0].shape", inputs[0].shape)
        seg_logits = self.forward(inputs)
        _, _, tok_h, tok_w = seg_logits.shape
        # img_h, img_w = batch_img_metas[0].img_shape
        seg_logits = resize(
            input=seg_logits,
            size=(tok_h * self.patch_size, tok_w * self.patch_size),
            mode='bilinear',
            align_corners=self.align_corners)
        # border crop
        # seg_logits = seg_logits[:, :, :img_h, :img_w]

        # center crop, slighly drop mIoU
        # crop_h = seg_logits.shape[2] - img_h
        # crop_w = seg_logits.shape[3] - img_w
        # crop_left = crop_w // 2
        # crop_right = crop_w - crop_left
        # crop_top = crop_h // 2  
        # crop_bottom = crop_h - crop_top
        # # print("tail crop:    ", crop_left, crop_right, crop_top, crop_bottom)
        # seg_logits = seg_logits[:, :, crop_top:crop_top+img_h, crop_left:crop_left+img_w]
        
        # seg_logits = resize(
        #     input=seg_logits,
        #     size=batch_img_metas[0]['img_shape'],
        #     mode='bilinear',
        #     align_corners=self.align_corners)
        return seg_logits

def fast_hist(label, prediction, n):
    k = (label >= 0) & (label < n)
    return np.bincount(n * label[k].astype(int) + prediction[k].astype(int), minlength=n * n).reshape(n, n)


def per_class_miou(hist):
    iou = np.diag(hist) / (hist.sum(1) + hist.sum(0) - np.diag(hist))
    return iou


def head_decode(self, outputs, img_metas, shape):
    """Decode features and generate segmentation output.

    Args:
        crop_feature_list (List[torch.Tensor]): List of cropped feature maps.
        img_metas (List[dict]): Metadata for the image(s).
        backbone_model (torch.nn.Module): Backbone model for feature normalization.
        shape (Tuple[int, int]): Shape of the original image.

    Returns:
        torch.Tensor: Resized segmentation output.
    """
    # Normalize feature maps using the backbone model's normalization, moved it from backbone to head


    # Process through neck if applicable
    x = tuple(outputs)
    if self.with_neck:
        x = self.neck(x)

    # Forward through decode head and resize output
    out = self.decode_head.predict(x, img_metas, self.test_cfg)
    # out = resize(input=seg_logits, size=shape, mode='bilinear', align_corners=self.align_corners)

    return out


def slide_inference_encode(self, img, img_meta, rescale=True):

    """Extract features using sliding-window inference.

    Args:
        img (torch.Tensor): Input image tensor.
        img_meta (List[dict]): Metadata for the image(s).
        rescale (bool, optional): Whether to rescale features to the original size. Defaults to True.

    Returns:
        List[torch.Tensor]: List of extracted features for sliding-window crops.
    """
    h_stride, w_stride = self.test_cfg.stride
    h_crop, w_crop = self.test_cfg.crop_size
    batch_size, _, h_img, w_img = img.size()

    feature_list = []
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Crop and extract features
            crop_img = img[:, :, y1:y2, x1:x2]
            crop_feature_list = self.backbone.feature_start_part(crop_img)
            feature_list.append(crop_feature_list)

    return feature_list


def slide_inference_decode(self, feature_list, img_meta, rescale=True):
    """Decode features using sliding-window inference.

    Args:
        feature_list (List[List[torch.Tensor]]): List of extracted feature maps for each crop.
        img_meta (List[dict]): Metadata for the image(s).
        rescale (bool, optional): Whether to rescale the output to the original size. Defaults to True.
        backbone_model (torch.nn.Module, optional): Backbone model for decoding. Defaults to None.

    Returns:
        torch.Tensor: Decoded segmentation predictions.
    """
    device = next(self.backbone.parameters()).device
    h_stride, w_stride = self.test_cfg.stride
    h_crop, w_crop = self.test_cfg.crop_size
    batch_size = feature_list[0][0].shape[0]
    img_shape = img_meta[0].img_shape
    h_img, w_img = img_shape[0], img_shape[1]
    num_classes = self.num_classes

    # Initialize predictions and counting matrix
    preds = torch.zeros((batch_size, num_classes, h_img, w_img), device=device)
    count_mat = torch.zeros((batch_size, 1, h_img, w_img), device=device)

    i = 0
    for h_idx in range(0, max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1):
        for w_idx in range(0, max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1):
            y1 = h_idx * h_stride
            x1 = w_idx * w_stride
            y2 = min(y1 + h_crop, h_img)
            x2 = min(x1 + w_crop, w_img)
            y1 = max(y2 - h_crop, 0)
            x1 = max(x2 - w_crop, 0)

            # Decode crop and update predictions
            token_res = (h_crop//self.backbone.patch_size, w_crop//self.backbone.patch_size)
            crop_feature_list = self.backbone.feature_end_part(feature_list[i], token_res=token_res)
            crop_seg_logit = self.head_decode(crop_feature_list, img_meta, (h_crop, w_crop))
            preds += F.pad(crop_seg_logit, (x1, preds.shape[3] - x2, y1, preds.shape[2] - y2))
            count_mat[:, :, y1:y2, x1:x2] += 1
            i += 1

    assert (count_mat == 0).sum() == 0, "Zero count in count matrix detected"

    preds = preds / count_mat

    if rescale:
        # Resize predictions to original image size
        resize_shape = img_meta[0].img_shape[:2]
        preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
        preds = resize(
            preds,
            size=img_meta[0].ori_shape[:2],  # ori_shape is the original shape of the image 
            mode='bilinear',
            align_corners=self.align_corners,
            warning=False
        )

    return preds


def simple_test_encode(self, img, img_meta, rescale=True):
    """Perform simple encoding during testing for a single image.

    Args:
        img (torch.Tensor): Input image tensor.
        img_meta (List[dict]): Metadata for the image, including original shape.
        rescale (bool, optional): Whether to rescale the features to the original image size. Defaults to True.

    Returns:
        List[torch.Tensor]: Extracted features from the input image.
    """
    # Ensure the test mode is valid
    assert self.test_cfg.mode in ['slide', 'whole'], "Invalid test mode"

    # Check that all images have the same original shape
    # ori_shape = img_meta['ori_shape']
    # assert all(meta['ori_shape'] == ori_shape for meta in img_meta), "Mismatch in original shapes across metadata"

    # Extract features based on the test mode
    if self.test_cfg.mode == 'slide':
        feature_list = self.slide_inference_encode(img, img_meta, rescale)
    else:
        raise ValueError("Only 'slide' mode is currently supported")

    return feature_list


def simple_test_decode(self, feature_list, img_meta, rescale=True):
    """Perform simple decoding during testing for a single image.

    Args:
        feature_list (List[torch.Tensor]): Extracted features from the model's backbone.
        img_meta (dict): Metadata for the image, including flip information.
        backbone_model (torch.nn.Module): Backbone model used in the segmentation process.
        rescale (bool, optional): Whether to rescale the output to the original image size. Defaults to True.

    Returns:
        List[np.ndarray]: Decoded segmentation prediction for the input image.
    """
    # Perform sliding window inference for decoding
    seg_logit = self.slide_inference_decode(feature_list, img_meta, rescale)

    # Apply softmax to generate probabilities
    output = F.softmax(seg_logit, dim=1)

    # Handle image flipping if applicable
    # flip = img_meta[0].flip
    # if flip:
    #     flip_direction = img_meta[0].flip_direction
    #     assert flip_direction in ['horizontal', 'vertical'], "Invalid flip direction"
    #     if flip_direction == 'horizontal':
    #         output = output.flip(dims=(3,))
    #     elif flip_direction == 'vertical':
    #         output = output.flip(dims=(2,))

    # Generate segmentation prediction
    seg_logit = output
    seg_pred = seg_logit.argmax(dim=1)

    # Handle ONNX export compatibility
    if torch.onnx.is_in_onnx_export():
        # Ensure the inference backend receives 4D output
        seg_pred = seg_pred.unsqueeze(0)
        return seg_pred

    # Convert predictions to numpy and unravel batch dimension
    seg_pred = seg_pred.cpu().numpy()
    seg_pred = list(seg_pred)

    return seg_pred



def extract_features(runner: Runner, org_feature_path: str):
    """Extract and save features for given images.

    Args:
        model (torch.nn.Module): Segmentation model.
        source_img_path (str): Path to source images.
        org_feature_path (str): Path to save extracted features.
        image_list (List[str]): List of image names to process.0
    """
    device = next(runner.model.parameters()).device


    for data in runner.test_dataloader:
        # img = data["inputs"]
        # img_meta = data["data_samples"]
        data = runner.model.data_preprocessor(data)
        img = data["inputs"]
        img_meta = data["data_samples"]

        with torch.no_grad():
            org_feature_list = runner.model.simple_test_encode(img, img_meta, rescale=True)
            org_feature_list = [torch.cat(feature_list) for feature_list in org_feature_list]
            org_feature_list = torch.stack(org_feature_list)
            # Save features
            image_name = img_meta[0].img_path.split('/')[-1].split('.')[0]
            np.save(f'{org_feature_path}/{image_name}.npy', org_feature_list.cpu().detach().numpy())


def seg_evaluate(runner: Runner, source_img_path: str, org_feature_path: str, rec_feature_path: str):
    """Evaluate segmentation performance.

    Args:
        model (torch.nn.Module): Segmentation model.
        source_img_path (str): Path to source images.
        org_feature_path (str): Path to original features.
        rec_feature_path (str): Path to reconstructed features.
        image_list (List[str]): List of image names to evaluate.
        backbone_model (torch.nn.Module): Backbone model.

    Returns:
        Dict[str, float]: Evaluation metrics.
    """
    device = next(runner.model.parameters()).device

    hist = np.zeros((runner.model.decode_head.num_classes, runner.model.decode_head.num_classes))   # 20 classes + 1 background
    mse_list = []

    for data in runner.test_dataloader:
        # img = data["inputs"]
        # img_meta = data["data_samples"]
        data = runner.model.data_preprocessor(data)
        img_meta = data["data_samples"]

        with torch.no_grad():
            image_name = img_meta[0].img_path.split('/')[-1].split('.')[0]
            # img = Image.open(f'{source_img_path}/JPEGImages/{image_name}.jpg')
            label = Image.open(f'{source_img_path}/SegmentationClass/{image_name}.png')

            # Load features and move to the specified device
            rec_feature_numpy = np.load(f'{rec_feature_path}/{image_name}.npy')
            rec_features_tensor = torch.from_numpy(rec_feature_numpy).to(device)
            
            # Convert features to the required format for the model
            rec_feature_list = [
                [rec_features_tensor[i, j].unsqueeze(0) for j in range(rec_features_tensor.shape[1])]
                for i in range(rec_features_tensor.shape[0])
            ]
            
            # Perform segmentation prediction
            pred = runner.model.simple_test_decode(
                rec_feature_list, 
                img_meta, 
                rescale=True
            )

        array_label = np.array(label)
        hist += fast_hist(array_label, pred[0], runner.model.decode_head.num_classes)

        # Calculate MSE
        org_feature = np.load(f'{org_feature_path}/{image_name}.npy')
        mse = (np.square(org_feature - rec_feature_numpy)).mean()
        mse_list.append(mse)

    # Calculate metrics
    all_iou = per_class_miou(hist)
    all_miou = np.nanmean(all_iou)

    return all_iou, all_miou, mse_list


def seg_pipeline(config_path: str, backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_split_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the depth estimation pipeline."""
    # Load configuration
    cfg = Config.fromfile(config_path)
    
    # Setup source image list
    with open(source_split_name) as f:
        image_list = f.readlines()
        image_list = ''.join(image_list).strip('\n').splitlines()

    # Setup models
    # backbone_model = setup_backbone(backbone_checkpoint_path)
    # model = build_segmentation_model(cfg, backbone_checkpoint_path, head_checkpoint_path)
    runner = Runner.from_cfg(cfg)
    # runner.test()
    runner.model.simple_test_encode = types.MethodType(simple_test_encode, runner.model)
    runner.model.simple_test_decode = types.MethodType(simple_test_decode, runner.model)
    runner.model.slide_inference_encode = types.MethodType(slide_inference_encode, runner.model)
    runner.model.slide_inference_decode = types.MethodType(slide_inference_decode, runner.model)
    runner.model.head_decode = types.MethodType(head_decode, runner.model)
    # print(runner)
    
    # Extract features
    os.makedirs(org_feature_path, exist_ok=True)
    extract_features(runner, org_feature_path)
    
    # Evaluate and print results
    all_iou, all_miou, mse_list = seg_evaluate(runner, source_img_path, org_feature_path, rec_feature_path)
    
    print(f"IoU: ", end=" ")
    for iou in all_iou: print(f"{iou*100:.4f}", end=" ") 
    print(f"\nmIoU: {all_miou*100:.4f}")
    print(f"Feature MSE: {np.mean(mse_list):.8f}")

def vtm_baseline_evaluation():
    # Set up paths
    config_path = 'cfg/dinov2_vitg14_voc2012_linear_config.py'
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_voc2012_linear_head.pth'
    
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/source/VOC2012'
    source_split_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/source/val_20.txt'
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/feature_test'
    vtm_root_path = f'/home/gaocs/projects/FCM-LM/Data/dinov2/seg/vtm_baseline'; print('vtm_root_path: ', vtm_root_path)
    
    # Load configuration
    cfg = Config.fromfile(config_path)
    
    # Setup source image list
    with open(source_split_name) as f:
        image_list = f.readlines()
        image_list = ''.join(image_list).strip('\n').splitlines()

    # Setup models
    backbone_model = setup_backbone(backbone_checkpoint_path)
    model = build_segmentation_model(cfg, backbone_model, head_checkpoint_path)
    
    # Evaluate and print results
    max_v = 103.2168; min_v = -530.9767; trun_high = 20; trun_low = -20

    trun_flag = True; samples = 0; bit_depth = 10; quant_type = 'uniform'
    if trun_flag == False: trun_high = max_v; trun_low = min_v

    QPs = [22]
    for QP in QPs:
        print(trun_low, trun_high, samples, bit_depth, quant_type, QP)
        rec_feature_path = f"{vtm_root_path}/postprocessed/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/QP{QP}"
        # rec_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/feature_test'

        all_iou, all_miou, mse_list = seg_evaluate(model, source_img_path, org_feature_path, rec_feature_path, image_list, backbone_model)
        # print(f"IoU: ", end=" ")
        # for iou in all_iou: print(f"{iou*100:.4f}", end=" ") 
        print(f"\nmIoU: {all_miou*100:.4f}")
        print(f"Feature MSE: {np.mean(mse_list):.8f}")

def hyperprior_baseline_evaluation():
    # Set up paths
    config_path = 'cfg/dinov2_vitg14_voc2012_linear_config.py'
    backbone_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = '/home/gaocs/models/dinov2/dinov2_vitg14_voc2012_linear_head.pth'
    
    source_img_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/source/VOC2012'
    source_split_name = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/source/val_20.txt'
    org_feature_path = '/home/gaocs/projects/FCM-LM/Data/dinov2/seg/feature_test'
    root_path = f'/home/gaocs/projects/FCM-LM/Data/dinov2/seg/hyperprior'; print('root_path: ', root_path)
    
    # Load configuration
    cfg = Config.fromfile(config_path)
    
    # Setup source image list
    with open(source_split_name) as f:
        image_list = f.readlines()
        image_list = ''.join(image_list).strip('\n').splitlines()

    # Setup models
    backbone_model = setup_backbone(backbone_checkpoint_path)
    model = build_segmentation_model(cfg, backbone_model, head_checkpoint_path)
    
    # Evaluate and print results
    max_v = 103.2168; min_v = -530.9767; trun_high = 5; trun_low = -5
    lambda_value_all = [0.0005, 0.001, 0.003, 0.007, 0.015]
    epochs = 800; learning_rate = "1e-4"; batch_size = 128; patch_size = "256 256"   # height first, width later

    trun_flag = True
    samples = 0; bit_depth = 1; quant_type = 'uniform'

    if trun_flag == False: trun_high = max_v; trun_low = min_v

    for lambda_value in lambda_value_all:
        print(trun_low, trun_high, samples, bit_depth, quant_type, lambda_value)
        rec_feature_path = f"{root_path}/decoded/trunl{trun_low}_trunh{trun_high}_{quant_type}{samples}_bitdepth{bit_depth}/lambda{lambda_value}_epoch{epochs}_lr{learning_rate}_bs{batch_size}_patch{patch_size.replace(' ', '-')}"

        all_iou, all_miou, mse_list = seg_evaluate(model, source_img_path, org_feature_path, rec_feature_path, image_list, backbone_model)
        # print(f"IoU: ", end=" ")
        # for iou in all_iou: print(f"{iou*100:.4f}", end=" ") 
        print(f"\nmIoU: {all_miou*100:.4f}")
        print(f"Feature MSE: {np.mean(mse_list):.8f}")

# # run below to evaluate the reconstructed features
# if __name__ == "__main__":
#     # vtm_baseline_evaluation()
#     hyperprior_baseline_evaluation()

# run below to extract original features as the dataset. 
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    base_path = os.path.realpath(os.path.join(os.path.dirname(__file__), '../../data'))
    config_path = os.path.join(os.path.dirname(__file__), 'conf-mmcv/dinov2_vitg14_voc2012_linear_config.py')
    backbone_checkpoint_path = f"{base_path}/models/backbone/dinov2_vitg14_pretrain.pth"
    head_checkpoint_path = f"{base_path}/models/clf_head/dinov2_vitg14_linear_head.pth"
    
    source_img_path = f"{base_path}/dataset/VOC2012"
    source_split_name = f"{base_path}/dataset/VOC2012/VOC2012_sel20.txt"
    org_feature_path = f"{base_path}/test-fc/VOC2012_sel20--dinov2_seg/feat"
    rec_feature_path = f"{base_path}/test-fc/VOC2012_sel20--dinov2_seg/feat"

    # rec_feature_path = f"/home/faymek/MPCompress/data/dataset/VOC2012_sel20/feat_provide"
    # rec_feature_path = f"{base_path}/test-fc/VOC2012_sel20--dinov2_seg/vtm_trunl-20_trunh20_uniform0_bitdepth10/postprocessed/QP42"

    seg_pipeline(config_path, backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_split_name, org_feature_path, rec_feature_path)