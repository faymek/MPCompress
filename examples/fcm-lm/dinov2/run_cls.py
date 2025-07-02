import os
import json
# import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torchvision import transforms, datasets
from dinov2.hub.classifiers import dinov2_vitg14_lc

import warnings
warnings.filterwarnings("ignore", category=UserWarning) # Disable xFormers UserWarning
os.environ['USE_XFORMERS'] = '0'    # Disable xFormers to obtain/extract consistent features in multiple runs


class DataFolder(datasets.ImageFolder):
    """Custom dataset class that includes the file path in the returned sample."""

    def __init__(self, root: str, transform=None, **kwargs):
        super().__init__(root, transform, **kwargs)

    def __getitem__(self, index: int):
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
        return sample, target, path


def get_label_from_file(filename, file_path):
    with open(file_path, 'r') as f:
        file_lines = f.readlines()
    
    for line in file_lines:
        parts = line.strip().split()
        if parts[0] == filename:
            return int(parts[-1])  # Return the last element (the number)
    
    return None  # Return None if the file name is not found


def build_dataset(source_img_path: str, batch_size: int, transform: str ='test'):
    # Define data transformations
    data_transform = {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]),
    }

    # Load test dataset and DataLoader
    dataset = DataFolder(source_img_path, transform=data_transform["test"])

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


def main(backbone_checkpoint_path: str, head_checkpoint_path: str, source_img_path: str, source_label_name: str, org_feature_path: str, rec_feature_path: str):
    """Main function to run the evaluation."""

    # batch_size = 1
    # test_dataset, test_dataloader = build_dataset(source_img_path, batch_size, 'test')

    # Initialize the model
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = dinov2_vitg14_lc(layers=1, pretrained=True, weights=[backbone_checkpoint_path, head_checkpoint_path])
    model.to(device)

    # Extract features
    # extract_features(model, test_dataloader, org_feature_path)

    # Evaluate and print results
    acc, feat_mse = evaluate_cls(model, org_feature_path, rec_feature_path, source_label_name)
    print(f"Classification Accuracy: {acc:.4f}")
    print(f"Feature MSE: {feat_mse:.8f}")


if __name__ == "__main__":
    base_path = '/home/faymek/MPCompress/data'
    backbone_checkpoint_path = f'{base_path}/models/backbone/dinov2_vitg14_pretrain.pth'
    head_checkpoint_path = f'{base_path}/models/head/dinov2_vitg14_linear_head.pth'
    source_img_path = f'{base_path}/dataset/ImageNet_val_sel100/img'
    source_label_name = f'{base_path}/dataset/ImageNet_val_sel100/labels.txt'
    org_feature_path = f'{base_path}/dataset/ImageNet_val_sel100/feat'
    rec_feature_path = f'{base_path}/test-fc/ImageNet--dinov2_cls--vtm_trunl-20_trunh20_uniform0_bitdepth10/postprocessed/QP42'
    main(backbone_checkpoint_path, head_checkpoint_path, source_img_path, source_label_name, org_feature_path, rec_feature_path)
