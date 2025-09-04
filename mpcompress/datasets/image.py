from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ImageFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.rglob("*") if f.is_file())

        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img


class ClassificationDataset(Dataset):
    """统一的图像文件夹数据集，支持分类和分割任务"""

    def __init__(
        self, root, transform=None, split="", file_list=None, labels_file=None, **kwargs
    ):
        """
        参数:
            root (str): 数据集根目录
            transform (callable, optional): 数据预处理转换
            split (str): 数据子集名称
            file_list (str, optional): 文件列表路径，相对于root
            labels_file (str, optional): 标签文件路径，相对于root
            seg_map_path (str, optional): 分割标签路径，相对于root
        """
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.file_list = file_list
        self.labels_file = labels_file

        # 确定数据目录
        if self.split:
            self.data_dir = self.root / self.split
        else:
            self.data_dir = self.root

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f'Missing directory "{self.data_dir}"')

        # 加载文件列表
        if self.file_list and (self.root / self.file_list).exists():
            with open(self.root / self.file_list, "r") as f:
                self.samples = [line.strip() for line in f.readlines()]
            # 确保文件路径是相对于data_dir的
            self.samples = [str(self.data_dir / sample) for sample in self.samples]
            self.samples = sorted(self.samples)
        else:
            # 如果没有指定file_list，则扫描目录
            self.samples = sorted(
                f
                for f in self.data_dir.rglob("*")
                if f.is_file()
                and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            )
            self.samples = [str(f) for f in self.samples]

        # 加载标签信息
        if not self.labels_file:
            raise ValueError("labels_file is required for classification dataset")
        if not (self.root / self.labels_file).exists():
            raise FileNotFoundError(f"labels_file {self.labels_file} not found")
        self.labels_dict = {}
        with open(self.root / self.labels_file, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    img_name = parts[0]
                    label = int(parts[1])
                    self.labels_dict[img_name] = label

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """返回 (img, img_meta)"""
        img_path = self.samples[index]
        img_name = Path(img_path).stem

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        img_meta = {
            "img_path": img_path,
            "img_name": img_name,
            "ori_size": img.size if hasattr(img, "size") else img.shape[-2:],
            "cls_label": self.labels_dict.get(img_name, None),
        }

        return img, img_meta


class SegmentationDataset(Dataset):
    def __init__(
        self,
        root,
        transform=None,
        img_path="JPEGImages",
        seg_map_path="SegmentationClass",
        file_list=None,
        reduce_zero_label=False,
        **kwargs,
    ):
        """
        参数:
            root (str): 数据集根目录
            transform (callable, optional): 数据预处理转换
            img_path (str): 图像目录名
            seg_map_path (str): 分割标签目录名
            file_list (str, optional): 文件列表路径
        """
        super().__init__()

        self.root = Path(root)
        self.transform = transform
        self.img_path = img_path
        self.seg_map_path = seg_map_path
        self.file_list = file_list
        self.reduce_zero_label = reduce_zero_label

        img_dir = self.root / self.img_path
        if not img_dir.is_dir():
            raise RuntimeError(f'Missing directory "{img_dir}"')

        # 加载文件列表
        if self.file_list and (self.root / self.file_list).exists():
            with open(self.root / self.file_list, "r") as f:
                self.samples = [line.strip() for line in f.readlines()]
            self.samples = [
                str(img_dir / f"{img_name}.jpg") for img_name in self.samples
            ]
        else:
            # 如果没有指定file_list，则扫描目录
            self.samples = sorted(f for f in img_dir.rglob("*.jpg"))
            self.samples = [str(f) for f in self.samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        """返回 (img, img_meta)"""
        img_path = self.samples[index]
        img_name = Path(img_path).stem

        # 加载图像
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        
        seg_label_path = str(self.root / self.seg_map_path / f"{img_name}.png")
        seg_label = np.array(Image.open(seg_label_path))
        if self.reduce_zero_label:
            seg_label = seg_label - 1  # for uint8, 2->1, 1->0, 0->255
        seg_label = seg_label.astype(np.int64)

        img_meta = {
            "img_path": img_path,
            "img_name": img_name,
            "ori_size": img.size if hasattr(img, "size") else img.shape[-2:],
            "seg_label_path": seg_label_path,
            "seg_label": seg_label,
        }

        return img, img_meta


class VOC2012Dataset(SegmentationDataset):
    pass


class ADE20KDataset(SegmentationDataset):
    pass
