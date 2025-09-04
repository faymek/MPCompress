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


class PascalVOCDataset(SegmentationDataset):
    """Pascal VOC dataset.
    From https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/voc.py
    """

    METAINFO = dict(
        classes=('background', 'aeroplane', 'bicycle', 'bird', 'boat',
                 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
                 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep',
                 'sofa', 'train', 'tvmonitor'),
        palette=[[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                 [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                 [64, 0, 0], [192, 0, 0], [64, 128, 0], [192, 128, 0],
                 [64, 0, 128], [192, 0, 128], [64, 128, 128], [192, 128, 128],
                 [0, 64, 0], [128, 64, 0], [0, 192, 0], [128, 192, 0],
                 [0, 64, 128]])

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


class ADE20KDataset(SegmentationDataset):
    """ADE20K dataset.
    From https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/ade.py

    In segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in 150 categories. ``reduce_zero_label`` is fixed to True.
    """

    METAINFO = dict(
        classes=('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road',
                 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk',
                 'person', 'earth', 'door', 'table', 'mountain', 'plant',
                 'curtain', 'chair', 'car', 'water', 'painting', 'sofa',
                 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair',
                 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp',
                 'bathtub', 'railing', 'cushion', 'base', 'box', 'column',
                 'signboard', 'chest of drawers', 'counter', 'sand', 'sink',
                 'skyscraper', 'fireplace', 'refrigerator', 'grandstand',
                 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow',
                 'screen door', 'stairway', 'river', 'bridge', 'bookcase',
                 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill',
                 'bench', 'countertop', 'stove', 'palm', 'kitchen island',
                 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine',
                 'hovel', 'bus', 'towel', 'light', 'truck', 'tower',
                 'chandelier', 'awning', 'streetlight', 'booth',
                 'television receiver', 'airplane', 'dirt track', 'apparel',
                 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle',
                 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain',
                 'conveyer belt', 'canopy', 'washer', 'plaything',
                 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall',
                 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food',
                 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal',
                 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket',
                 'sculpture', 'hood', 'sconce', 'vase', 'traffic light',
                 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate',
                 'monitor', 'bulletin board', 'shower', 'radiator', 'glass',
                 'clock', 'flag'),
        palette=[[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
                 [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
                 [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
                 [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
                 [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
                 [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
                 [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
                 [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
                 [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
                 [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
                 [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
                 [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
                 [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
                 [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
                 [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
                 [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
                 [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
                 [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
                 [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
                 [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
                 [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
                 [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
                 [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
                 [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
                 [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
                 [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
                 [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
                 [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
                 [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
                 [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
                 [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0],
                 [255, 0, 235], [245, 0, 255], [255, 0, 122], [255, 245, 0],
                 [10, 190, 212], [214, 255, 0], [0, 204, 255], [20, 0, 255],
                 [255, 255, 0], [0, 153, 255], [0, 41, 255], [0, 255, 204],
                 [41, 0, 255], [41, 255, 0], [173, 0, 255], [0, 245, 255],
                 [71, 0, 255], [122, 0, 255], [0, 255, 184], [0, 92, 255],
                 [184, 255, 0], [0, 133, 255], [255, 214, 0], [25, 194, 194],
                 [102, 255, 0], [92, 0, 255]])
    def __init__(self, reduce_zero_label=True, **kwargs):
        super().__init__(reduce_zero_label=reduce_zero_label, **kwargs)
