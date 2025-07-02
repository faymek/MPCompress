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
from mpcompress.eval.coding.fc_vtm import run_vtm_compression, get_vtm_fc_config

from mpcompress.eval.coding.fc_hyper import get_hyper_fc_config
from mpcompress.eval.coding.fc_hyper import hyperprior_train_pipeline, hyperprior_evaluate_pipeline

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs


class SmallImageNetDataset(Dataset):
    """自定义数据集类,用于处理小型ImageNet数据集,包含train/val/test三个子文件夹和labels.txt标签文件"""

    def __init__(self, root, transform=None, split="train"):
        """
        参数:
            root (str): 数据集根目录,包含train/val/test子文件夹和labels.txt
            transform (callable, optional): 数据预处理转换
            split (str): 使用的数据子集,可选'train'/'val'/'test'
        """
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.rglob("*") if f.is_file())
        self.transform = transform

        # 读取标签文件
        self.labels = {}
        label_file = Path(root) / 'labels.txt'
        with open(label_file, 'r') as f:
            for line in f:
                img_name, label = line.strip().split()
                self.labels[img_name] = int(label)

    def __getitem__(self, index):

        img_path = self.samples[index]
        img = Image.open(img_path).convert("RGB")
        
        if self.transform:
            img = self.transform(img)

        img_name = img_path.stem
        target = self.labels[img_name]
            
        return img, target, img_name

    def __len__(self):
        return len(self.samples)


def get_label_from_file(filename, file_path):
    with open(file_path, 'r') as f:
        file_lines = f.readlines()
    
    for line in file_lines:
        parts = line.strip().split()
        if parts[0] == filename:
            return int(parts[-1])  # Return the last element (the number)
    
    return None  # Return None if the file name is not found


def build_dataset(data_root: str, split: str ='val', batch_size: int = 1):
    # Define data transformations
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Load test dataset and DataLoader
    dataset = SmallImageNetDataset(data_root, split=split, transform=data_transform[split])

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return dataset, dataloader


def extract_features(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, org_feature_path: str):
    """Extract features from backbone"""
    model.eval()
    device = next(model.parameters()).device
    os.makedirs(org_feature_path, exist_ok=True)

    for batch_x, batch_y, img_name in tqdm(data_loader):
        batch_x = batch_x.to(device)
        feature_list = model.forward_backbone(batch_x)
        feature = feature_list[0].unsqueeze(0)

        feat_name = img_name[0].split('/')[-1].split('.')[0]    # get the img name without '.JPEG'
        np.save(f'{org_feature_path}/{feat_name}.npy', feature.cpu().detach().numpy())


def evaluate_cls(model: torch.nn.Module, org_feature_path: str, rec_feature_path: str, source_label_name: str):
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
    rec_feat_names = [f for f in os.listdir(rec_feature_path) if f.endswith('.npy')]

    for idx, rec_feat_name in enumerate(rec_feat_names):
        # Load reconstructed features
        rec_features_numpy = np.load(f"{rec_feature_path}/{rec_feat_name}")
        rec_features_tensor = torch.from_numpy(rec_features_numpy).to(device)

        with torch.no_grad():
            # Decode features and make predictions
            pred = torch.argmax(model.forward_head(rec_features_tensor), dim=1)

            # Compute accuracy using labels
            label = get_label_from_file(rec_feat_name.split('.')[0], source_label_name)
            label_tensor = torch.tensor(label).to(device)
            num_correct = (pred == label_tensor).sum().item()
            eval_acc += num_correct

            # Compute MSE between original and reconstructed features
            org_feat = np.load(f"{org_feature_path}/{rec_feat_name}")
            mse = np.mean(np.square(org_feat - rec_features_numpy))
            eval_mse += mse

    # Calculate and print metrics
    num_samples = len(rec_feat_names)

    return eval_acc*100 / num_samples, eval_mse / num_samples


def cls_pipeline(backbone_checkpoint_path: str, head_checkpoint_path: str, data_root: str, source_label_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the evaluation."""

    batch_size = 1
    test_dataset, test_dataloader = build_dataset(data_root, 'val', batch_size)
    
    # labels = get_label_from_dataset(test_dataset)   # comment this, only used in the first time to generate source label file

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Extract features
    os.makedirs(org_feature_path, exist_ok=True)
    extract_features(model, test_dataloader, org_feature_path)

    # Evaluate and print results
    acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
    print(f"Classification Accuracy: {acc:.2f}%")
    print(f"Feature MSE: {feat_mse:.8f}")


# # run below to evaluate the reconstructed features
# if __name__ == "__main__":
#     # vtm_baseline_evaluation()
#     hyperprior_baseline_evaluation()

# run below to extract original features as the dataset. 
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    base_path = '/home/faymek/MPCompress/data'
    backbone_checkpoint_path = f'{base_path}/models/backbone/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = f'{base_path}/models/clf_head/dinov2_vitg14_linear_head.pth'
    
    source_train_data_root = f'{base_path}/dataset/ImageNet_train_sel100'
    source_train_label_name = f'{base_path}/dataset/ImageNet_train_sel100/labels.txt'
    source_test_data_root = f'{base_path}/dataset/ImageNet_val_sel100'
    source_test_label_name = f'{base_path}/dataset/ImageNet_val_sel100/labels.txt'

    org_train_feature_path = f'{base_path}/train-fc/ImageNet--dinov2_cls/feat'
    org_test_feature_path = f'{base_path}/test-fc/ImageNet--dinov2_cls/feat'

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    cfg = get_hyper_fc_config("dinov2_cls")    

    # Extract features
    train_dataset, train_dataloader = build_dataset(source_train_data_root, 'train', batch_size=cfg.batch_size)
    # train_dataset, train_dataloader = build_dataset(source_train_data_root, 'train', batch_size=1)
    print(f"\nExtracting features from {source_train_data_root} to {org_train_feature_path}")
    # extract_features(model, train_dataloader, org_train_feature_path)

    test_dataset, test_dataloader = build_dataset(source_test_data_root, 'val', batch_size=1)
    print(f"\nExtracting features from {source_test_data_root} to {org_test_feature_path}")
    # extract_features(model, test_dataloader, org_test_feature_path)
    
    prefix = 'ImageNet--dinov2_cls'
    lambda_value_all = cfg.lambda_value_all

    # Train and print results
    for lambda_value in lambda_value_all:  
        train_root = f'{base_path}/train-fc/ImageNet--dinov2_cls'
        print(f"\nTraining hyperprior compression for lambda{lambda_value} from {org_train_feature_path} to {train_root}")
        hyperprior_train_pipeline(base_path, prefix, cfg, lambda_value)
        print(f"\nTraining hyperprior compression for lambda{lambda_value} Finished!")

    # Evaluate and print results
    for lambda_value in lambda_value_all:
        test_root = f'{base_path}/test-fc/ImageNet--dinov2_cls/'
        print(f"\nRunning hyperprior compression for lambda{lambda_value} from {org_test_feature_path} to {test_root}")
        hyperprior_evaluate_pipeline(base_path, prefix, cfg, lambda_value)

    for lambda_value in lambda_value_all:
        patch_size_str = cfg.patch_size.replace(' ', '-')
        rec_test_feature_path = f'{base_path}/test-fc/{prefix}/hyperprior/decoded/trunl{cfg.trun_low}_trunh{cfg.trun_high}_{cfg.quant_type}{cfg.samples}_bitdepth{cfg.bit_depth}/lambda{lambda_value}_epoch{cfg.epochs}_lr{cfg.learning_rate}_bs{cfg.batch_size}_patch{patch_size_str}'
        print(f"\nEvaluating VTM compression for lambda{lambda_value} from {rec_test_feature_path}")
        acc, feat_mse = evaluate_cls(model, org_test_feature_path, rec_test_feature_path, source_test_label_name)
        print(f"lambda{lambda_value} Classification Accuracy: {acc:.4f}")
        print(f"Feature MSE: {feat_mse:.8f}")
    

