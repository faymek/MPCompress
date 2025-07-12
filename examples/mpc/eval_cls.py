import os
import sys
import warnings

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
from mpcompress.heads import Dinov2ClassifierHead

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

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


def evaluate_cls_only_head(
    head: torch.nn.Module,
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
    head.eval()
    device = next(head.parameters()).device

    eval_acc = 0.0
    eval_mse = 0.0

    # Retrieve reconstructed feature filenames
    rec_feat_names = [f for f in os.listdir(rec_feature_path) if f.endswith(".pt")]

    for idx, rec_feat_name in enumerate(rec_feat_names):
        # Load reconstructed features
        rec_features_tensor = torch.load(f"{rec_feature_path}/{rec_feat_name}")
        rec_features_tensor = [
            [x.to(device) for x in lst] for lst in rec_features_tensor
        ]

        with torch.no_grad():
            # Decode features and make predictions
            # pred = torch.argmax(head.forward(rec_features_tensor), dim=1)
            pred = head.predict(rec_features_tensor, topk=1)

            # Compute accuracy using labels
            # print(rec_feat_name.split(".")[0])
            label = get_label_from_file(rec_feat_name.split(".")[0], source_label_name)
            label_tensor = torch.tensor(label).to(device)
            num_correct = (pred == label_tensor).sum().item()
            eval_acc += num_correct

            # Compute MSE between original and reconstructed features
            # org_feat = np.load(f"{org_feature_path}/{rec_feat_name}")
            # mse = np.mean(np.square(org_feat - rec_features_numpy))
            # eval_mse += mse

    # Calculate and print metrics
    num_samples = len(rec_feat_names)

    return eval_acc * 100 / num_samples, eval_mse / num_samples


# run below to extract original features as the dataset.
# You can skip feature extraction if you have download the test dataset from https://drive.google.com/drive/folders/1RZFGlBd6wZr4emuGO4_YJWfKPtAwcMXQ
if __name__ == "__main__":
    base_path = os.path.realpath(os.path.join(os.path.dirname(__file__), "../../data"))
    backbone_checkpoint_path = f"{base_path}/models/backbone/dinov2_vits14_pretrain.pth"
    # head_checkpoint_path = f"{base_path}/models/clf_head/dinov2_vitg14_linear_head.pth"
    head_checkpoint_path = f"{base_path}/models/clf_head/dinov2_vits14_linear4_head.pth"
    source_data_root = f"{base_path}/dataset/ImageNet_val_sel100"
    source_label_name = f"{base_path}/dataset/ImageNet_val_sel100/labels.txt"
    org_feature_path = f"{base_path}/test-fc/ImageNet--dinov2_cls/feat"
    # rec_feature_path = f'{base_path}/test-fc/ImageNet--dinov2_cls/vtm_trunl-20_trunh20_uniform0_bitdepth10/postprocessed/QP42'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    head = Dinov2ClassifierHead(384, 4, head_checkpoint_path)

    # Evaluate and print results
    for QP in [6.0, 12.0, 24.0, 48.0]:
        rec_feature_path = f"/home/faymek/MPCompress/eval_in100/Q{QP}"
        print(f"\nEvaluating VTM compression for QP{QP} from {rec_feature_path}")
        acc, feat_mse = evaluate_cls_only_head(
            head, rec_feature_path, source_label_name
        )
        print(f"QP{QP} Classification Accuracy: {acc:.4f}")
        print(f"Feature MSE: {feat_mse:.8f}")
