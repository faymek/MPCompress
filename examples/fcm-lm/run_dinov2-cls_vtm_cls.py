import os
import json

# import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torchvision import transforms, datasets
from mpcompress.backbone.dinov2.hub.classifiers import dinov2_vitg14_lc
from pathlib import Path
from PIL import Image
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from fc_vtm import run_vtm_compression, get_vtm_fc_config
from mpcompress.datasets import SmallImageNetDataset
import warnings
from omegaconf import OmegaConf

warnings.filterwarnings("ignore", category=UserWarning)  # Disable xFormers UserWarning
os.environ["USE_XFORMERS"] = (
    "0"  # Disable xFormers to obtain/extract consistent features in multiple runs
)


def get_label_from_file(filename, file_path):
    with open(file_path, "r") as f:
        file_lines = f.readlines()

    for line in file_lines:
        parts = line.strip().split()
        if parts[0] == filename:
            return int(parts[-1])  # Return the last element (the number)

    return None  # Return None if the file name is not found


def build_dataset(data_root: str, split: str = "val", batch_size: int = 1):
    # Define data transformations
    data_transform = {
        "train": transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    }

    # Load test dataset and DataLoader
    dataset = SmallImageNetDataset(
        data_root, split=split, transform=data_transform[split]
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataset, dataloader


def extract_features(
    model: torch.nn.Module,
    data_loader: torch.utils.data.DataLoader,
    org_feature_path: str,
):
    """Extract features from backbone"""
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(org_feature_path, exist_ok=True)

    for batch_x, batch_y, img_name in tqdm(data_loader):
        batch_x = batch_x.to(device)
        feature_list = model.forward_backbone(batch_x)
        feature = feature_list[0].unsqueeze(0)

        feat_name = (
            img_name[0].split("/")[-1].split(".")[0]
        )  # get the img name without '.JPEG'
        np.save(f"{org_feature_path}/{feat_name}.npy", feature.cpu().detach().numpy())


def evaluate_cls(
    model: torch.nn.Module,
    org_feature_path: str,
    rec_feature_path: str,
    source_label_name: str,
):
    """Evaluate image classification accuracy and feature reconstruction error.

    Args:
        model (torch.nn.Module): Classification model for evaluation.
        org_feature_path (str): Path to the original features.
        rec_feature_path (str): Path to the reconstructed features.
        source_label_name (str): Path to the source label file.

    Returns:
        Accuracy, feature MSE
    """
    model.eval()
    device = next(model.parameters()).device

    eval_acc = 0.0
    eval_mse = 0.0

    # Retrieve reconstructed feature filenames
    rec_feat_names = [f for f in os.listdir(rec_feature_path) if f.endswith(".npy")]

    for idx, rec_feat_name in enumerate(rec_feat_names):
        # Load reconstructed features
        rec_features_numpy = np.load(f"{rec_feature_path}/{rec_feat_name}")
        rec_features_tensor = torch.from_numpy(rec_features_numpy).to(device)

        with torch.no_grad():
            # Decode features and make predictions
            pred = torch.argmax(model.forward_head(rec_features_tensor), dim=1)

            # Compute accuracy using labels
            label = get_label_from_file(rec_feat_name.split(".")[0], source_label_name)
            label_tensor = torch.tensor(label).to(device)
            num_correct = (pred == label_tensor).sum().item()
            eval_acc += num_correct

            # Compute MSE between original and reconstructed features
            org_feat = np.load(f"{org_feature_path}/{rec_feat_name}")
            mse = np.mean(np.square(org_feat - rec_features_numpy))
            eval_mse += mse

    # Calculate and print metrics
    num_samples = len(rec_feat_names)

    return eval_acc * 100 / num_samples, eval_mse / num_samples


def cls_pipeline(
    backbone_checkpoint_path: str,
    head_checkpoint_path: str,
    data_root: str,
    source_label_name: str,
    org_feature_path: str,
    rec_feature_path: str,
):
    """Main function to run the evaluation."""

    batch_size = 1
    test_dataset, test_dataloader = build_dataset(data_root, "val", batch_size)

    # labels = get_label_from_dataset(test_dataset)   # comment this, only used in the first time to generate source label file

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(
        layers=1,
        pretrained=True,
        weights=[backbone_checkpoint_path, head_checkpoint_path],
    )
    model.to(device)

    # Extract features
    os.makedirs(org_feature_path, exist_ok=True)
    extract_features(model, test_dataloader, org_feature_path)

    # Evaluate and print results
    acc, feat_mse = evaluate_cls(
        model, org_feature_path, rec_feature_path, source_label_name
    )
    print(f"Classification Accuracy: {acc:.2f}%")
    print(f"Feature MSE: {feat_mse:.8f}")


# run below to extract original features as the dataset.
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    base_path = "/home/fz2001/Ant/MPCompress/data"
    backbone_checkpoint_path = f"{base_path}/models/backbone/dinov2_vitg14_pretrain.pth"
    head_checkpoint_path = f"{base_path}/models/clf_head/dinov2_vitg14_linear_head.pth"
    source_data_root = f"{base_path}/dataset/ImageNet_val_sel100"
    source_label_name = f"{base_path}/dataset/ImageNet_val_sel100/labels.txt"
    org_feature_path = f"{base_path}/test-fc/ImageNet--dinov2_cls/feat"
    rec_feature_path = f"{base_path}/test-fc/ImageNet--dinov2_cls/vtm_trunl-20_trunh20_uniform0_bitdepth10/postprocessed/QP42"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(
        layers=1,
        pretrained=True,
        weights=[backbone_checkpoint_path, head_checkpoint_path],
    )
    model.to(device)
    test_dataset, test_dataloader = build_dataset(source_data_root, "val", batch_size=1)

    # Extract features
    print(f"\nExtracting features from {source_data_root} to {org_feature_path}")
    extract_features(model, test_dataloader, org_feature_path)

    QPs = [42]
    for QP in QPs:
        # 读取YAML配置
        cfg = get_vtm_fc_config("dinov2_cls")
        test_root = f"{base_path}/test-fc/ImageNet--dinov2_cls/vtm_{cfg.config_str}"
        print(
            f"\nRunning VTM compression for QP{QP} from {org_feature_path} to {test_root}"
        )
        run_vtm_compression(org_feature_path, test_root, cfg, QP)

    # Evaluate and print results
    for QP in QPs:
        rec_feature_path = f"{test_root}/postprocessed/QP{QP}"
        print(f"\nEvaluating VTM compression for QP{QP} from {rec_feature_path}")
        acc, feat_mse = evaluate_cls(
            model, org_feature_path, rec_feature_path, source_label_name
        )
        print(f"QP{QP} Classification Accuracy: {acc:.4f}")
        print(f"Feature MSE: {feat_mse:.8f}")
