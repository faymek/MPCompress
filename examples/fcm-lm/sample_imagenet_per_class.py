"""
场景:在一个典型ImageNet风格目录中,每个类别是一个子文件夹 (如 `n01558993/`)，
脚本会在每个类别下随机抽取固定数量的 JPEG 图片：
DATA_ROOT/
├── n01443537/
│   ├── img1.jpeg
│   └── ...
├── n01558993/
│   ├── imgX.jpeg
│   └── ...
└── ... (1000 类)
抽样后的图片将：
1. 通过DINOv2 ViT-G/14 backbone 提取 `[CLS]` 特征，并保存为 `.npy`。
2. 通过其线性分类头得到预测类别索引。
3. 记录到 `predictions.txt`（两列：`pred_index<TAB>filename`）。

## 配置
修改 `CONFIG` 即可，无需命令行参数。

python
CONFIG = {
    'DATA_ROOT': '/path/to/ImageNet/train',   # 含 1000 个类别子文件夹
    'BACKBONE_PATH': '/path/to/dinov2_vitg14_pretrain.pth',
    'HEAD_PATH': '/path/to/dinov2_vitg14_linear_head.pth',
    'OUTPUT_DIR': './outputs',                # 结果根目录
    'FEATURE_DIR': None,                      # None→ OUTPUT_DIR/features
    'SAMPLES_PER_CLASS': 5,                   # 每类抽几张，<=0 表示该类全部
    'SEED': 42,
}
"""
import os
import random
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from mpcompress.backbone.dinov2.hub.classifiers import dinov2_vitg14_lc

# =============================================================================
# CONFIG
# =============================================================================
CONFIG = {
    'DATA_ROOT': '/home/faymek/Datasets/IN1K/val',
    'BACKBONE_PATH': '/home/liuzk/projects/MPCompress/data/models/backbone/dinov2_vitg14_pretrain.pth',
    'HEAD_PATH': '/home/liuzk/projects/MPCompress/data/models/clf_head/dinov2_vitg14_linear_head.pth',
    'OUTPUT_DIR': '/home/liuzk/projects/MPCompress/data/self_extract',
    'FEATURE_DIR': None,
    'SAMPLES_PER_CLASS': 10,   # <=0 表示整类全选
    'SEED': 42,
}

# =============================================================================
# 固定随机种子
# =============================================================================
random.seed(CONFIG['SEED'])
np.random.seed(CONFIG['SEED'])
torch.manual_seed(CONFIG['SEED'])

# 禁用 xFormers, 与官方示例保持一致
os.environ['USE_XFORMERS'] = '0'
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# =============================================================================
# 自定义数据集（复用 ImageFolder 采样后的 indices）
# =============================================================================
class ImageFolderSubset(Dataset):
    """基于 torchvision ImageFolder 的子集，仅保留 selected_indices。"""
    def __init__(self, img_folder: datasets.ImageFolder, indices: List[int]):
        self.base = img_folder
        self.indices = indices
        self.transform = img_folder.transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        true_idx = self.indices[idx]
        path, target = self.base.samples[true_idx]
        img = Image.open(path).convert('RGB')
        img = self.transform(img)
        return img, Path(path).name, target

# =============================================================================
# 主流程
# =============================================================================

def main():
    cfg = CONFIG

    data_root = Path(cfg['DATA_ROOT']).expanduser()
    if not data_root.is_dir():
        raise RuntimeError(f'DATA_ROOT 不存在: {data_root}')

    # 输出目录
    out_root = Path(cfg['OUTPUT_DIR']).expanduser()
    out_root.mkdir(parents=True, exist_ok=True)
    feat_root = Path(cfg['FEATURE_DIR']).expanduser() if cfg['FEATURE_DIR'] else out_root / 'features'
    feat_root.mkdir(parents=True, exist_ok=True)

    # ---------------------- 1. 构建 ImageFolder ----------------------
    tfm = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    full_ds = datasets.ImageFolder(data_root, transform=tfm)  # 会自动读取子文件夹 label

    # ---------------------- 2. 为每个类别随机抽样 ----------------------
    # 建立类别 → indices 列表映射
    class_to_indices = {i: [] for i in range(len(full_ds.classes))}
    for idx, (_, target) in enumerate(full_ds.samples):
        class_to_indices[target].append(idx)

    selected_indices: List[int] = []
    for cls, idx_list in class_to_indices.items():
        if cfg['SAMPLES_PER_CLASS'] > 0 and len(idx_list) > cfg['SAMPLES_PER_CLASS']:
            chosen = random.sample(idx_list, cfg['SAMPLES_PER_CLASS'])
        else:
            chosen = idx_list  # 全部
        selected_indices.extend(chosen)

    print(f"共选取 {len(selected_indices)} 张图像进行处理 (类别×样本: {len(full_ds.classes)}×{cfg['SAMPLES_PER_CLASS'] or 'ALL'})")

    # ---------------------- 3. DataLoader ----------------------
    subset_ds = ImageFolderSubset(full_ds, selected_indices)
    loader = DataLoader(subset_ds, batch_size=1, shuffle=False)

    # ---------------------- 4. 模型 ----------------------
    model = dinov2_vitg14_lc(
        layers=1,
        pretrained=True,
        weights=[cfg['BACKBONE_PATH'], cfg['HEAD_PATH']],
    ).to(device)
    model.eval()

    # ---------------------- 5. 抽特征 + 分类 ----------------------
    results = []  # (pred_idx, filename, gt_idx)

    with torch.no_grad():
        for img, fname, gt_idx in tqdm(loader, desc='Extracting'):
            img = img.to(device)
            feature_list = model.forward_backbone(img)
            feature = feature_list[0].unsqueeze(0)  # (1, 1, C)
            logits = model.forward_head(feature)
            pred_idx = torch.argmax(logits, dim=1).item()

            # 保存特征
            stem = Path(fname[0]).stem  # 去掉扩展名
            # 加上类别 id 防止不同类别同名冲突
            np.save(feat_root / f'{gt_idx.item()}_{stem}.npy', feature.cpu().numpy())

            results.append((pred_idx, fname[0], gt_idx.item()))

    # ---------------------- 6. 写出 txt ----------------------
    txt_path = out_root / 'predictions.txt'
    with open(txt_path, 'w', encoding='utf-8') as fp:
        for pred_idx, fname, _ in results:
            fp.write(f'{pred_idx}\t{fname}\n')

    # 可选：计算简单 top‑1 accuracy
    correct = sum(int(pred == gt) for pred, _, gt in results)
    acc = correct / len(results) * 100
    print(f'Top‑1 Accuracy on sampled subset: {acc:.2f}%')

    print('\n=== 运行完毕 ===')
    print(f'特征目录 : {feat_root}')
    print(f'预测结果 : {txt_path}\n')


if __name__ == '__main__':
    main()
