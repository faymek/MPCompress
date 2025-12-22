from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class ImageFolder(Dataset):
    """Load an image folder database.

    Training and testing image samples are respectively stored in separate
    directories:

        rootdir/
            train/
                img000.png
                img001.png
            test/
                img000.png
                img001.png

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function or transform that takes in a
            PIL image and returns a transformed version. Defaults to None.
        split (str): Split mode ('train' or 'val'). Defaults to "train".
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.rglob("*") if f.is_file())

        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of image files in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """Get an image sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            img (PIL.Image.Image or torch.Tensor): The image. If transform is
                provided, returns the transformed version (typically a torch.Tensor).
                Otherwise, returns a PIL Image in RGB format.
        """
        img = Image.open(self.samples[index]).convert("RGB")
        if self.transform:
            return self.transform(img)
        return img


class ClassificationDataset(Dataset):
    """Unified image folder dataset for classification tasks.

    This dataset loads images from a directory structure and associates them
    with classification labels from a labels file. It supports loading images
    from a file list or by scanning the directory.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): Data preprocessing transform function.
            Defaults to None.
        split (str): Subset name (e.g., 'train', 'val'). If empty, uses root
            directory directly. Defaults to "".
        file_list (str, optional): Path to file list relative to root. Each line
            should contain a relative path to an image file. If None, scans the
            directory for image files. Defaults to None.
        labels_file (str, required): Path to labels file relative to root. Each
            line should contain "image_name label" where label is an integer.
            Defaults to None.
        **kwargs (dict): Additional keyword arguments (unused).
    """

    def __init__(
        self, root, transform=None, split="", file_list=None, labels_file=None, **kwargs
    ):
        self.root = Path(root)
        self.transform = transform
        self.split = split
        self.file_list = file_list
        self.labels_file = labels_file

        # Determine data directory
        if self.split:
            self.data_dir = self.root / self.split
        else:
            self.data_dir = self.root

        if not self.data_dir.is_dir():
            raise FileNotFoundError(f'Missing directory "{self.data_dir}"')

        # Load file list
        if self.file_list and (self.root / self.file_list).exists():
            with open(self.root / self.file_list, "r") as f:
                self.samples = [line.strip() for line in f.readlines()]
            # Ensure file paths are relative to data_dir
            self.samples = [str(self.data_dir / sample) for sample in self.samples]
            self.samples = sorted(self.samples)
        else:
            # If file_list is not specified, scan the directory
            self.samples = sorted(
                f
                for f in self.data_dir.rglob("*")
                if f.is_file()
                and f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]
            )
            self.samples = [str(f) for f in self.samples]

        # Load label information
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
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of image files in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """Get an image sample and its metadata from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple (PIL.Image.Image or torch.Tensor, dict): A tuple containing:
                - img (PIL.Image.Image or torch.Tensor): The image. If transform
                    is provided, returns the transformed version (typically a
                    torch.Tensor). Otherwise, returns a PIL Image in RGB format.
                - img_meta (dict): Metadata dictionary with keys:
                    - "img_path" (str): Full path to the image file.
                    - "img_name" (str): Image filename without extension.
                    - "ori_size" (tuple): Original image size (width, height) or
                        (height, width) for tensors.
                    - "cls_label" (int or None): Classification label for the image.
        """
        img_path = self.samples[index]
        img_name = Path(img_path).stem

        # Load image
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
    """Dataset for image segmentation tasks.

    This dataset loads images and their corresponding segmentation masks from
    separate directories. It supports loading images from a file list or by
    scanning the directory.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): Data preprocessing transform function.
            Defaults to None.
        img_path (str): Name of the image subdirectory within root.
            Defaults to "JPEGImages".
        seg_map_path (str): Name of the segmentation mask subdirectory within root.
            Defaults to "SegmentationClass".
        file_list (str, optional): Path to file list relative to root. Each line
            should contain an image name (without extension). If None, scans the
            directory for .jpg files. Defaults to None.
        reduce_zero_label (bool): Whether to reduce zero label. If True, subtracts
            1 from all labels (2->1, 1->0, 0->255 for uint8). Defaults to False.
        **kwargs (dict): Additional keyword arguments (unused).
    """

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

        # Load file list
        if self.file_list and (self.root / self.file_list).exists():
            with open(self.root / self.file_list, "r") as f:
                self.samples = [line.strip() for line in f.readlines()]
            self.samples = [
                str(img_dir / f"{img_name}.jpg") for img_name in self.samples
            ]
        else:
            # If file_list is not specified, scan the directory
            self.samples = sorted(f for f in img_dir.rglob("*.jpg"))
            self.samples = [str(f) for f in self.samples]

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of image files in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, index):
        """Get an image sample and its segmentation mask from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            tuple (PIL.Image.Image or torch.Tensor, dict): A tuple containing:
                - img (PIL.Image.Image or torch.Tensor): The image. If transform
                    is provided, returns the transformed version (typically a
                    torch.Tensor). Otherwise, returns a PIL Image in RGB format.
                - img_meta (dict): Metadata dictionary with keys:
                    - "img_path" (str): Full path to the image file.
                    - "img_name" (str): Image filename without extension.
                    - "ori_size" (tuple): Original image size (width, height) or
                        (height, width) for tensors.
                    - "seg_label_path" (str): Full path to the segmentation mask file.
                    - "seg_label" (numpy.ndarray): Segmentation mask as int64 array.
        """
        img_path = self.samples[index]
        img_name = Path(img_path).stem

        # Load image
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        seg_label_path = str(self.root / self.seg_map_path / f"{img_name}.png")
        seg_label = np.array(Image.open(seg_label_path))
        if self.reduce_zero_label:
            # For uint8: 2->1, 1->0, 0->255
            seg_label = seg_label - 1
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
    """Pascal VOC dataset for semantic segmentation.

    This dataset implements the Pascal VOC 2012 dataset format with 21 classes
    (including background). The dataset structure follows the standard VOC format
    with images in JPEGImages and segmentation masks in SegmentationClass.

    Reference:
        https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/voc.py

    Attributes:
        METAINFO (dict): Dataset metadata containing:
            - classes (tuple): Tuple of 21 class names including 'background'.
            - palette (list): List of RGB color values for visualization.
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
        """Initialize Pascal VOC dataset.

        Args:
            **kwargs: Arguments passed to :class:`SegmentationDataset`. See
                :meth:`SegmentationDataset.__init__` for details.
        """
        super().__init__(**kwargs)


class ADE20KDataset(SegmentationDataset):
    """ADE20K dataset for semantic segmentation.

    This dataset implements the ADE20K dataset format with 150 semantic classes.
    In the segmentation map annotation for ADE20K, 0 stands for background, which
    is not included in the 150 categories. Therefore, ``reduce_zero_label`` is
    fixed to True by default.

    Reference:
        https://github.com/open-mmlab/mmsegmentation/blob/main/mmseg/datasets/ade.py

    Attributes:
        METAINFO (dict): Dataset metadata containing:
            - classes (tuple): Tuple of 150 class names.
            - palette (list): List of RGB color values for visualization.
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
        """Initialize ADE20K dataset.

        Args:
            reduce_zero_label (bool): Whether to reduce zero label. Fixed to True
                for ADE20K since 0 represents background not in 150 categories.
                Defaults to True.
            **kwargs: Additional arguments passed to :class:`SegmentationDataset`.
                See :meth:`SegmentationDataset.__init__` for details.
        """
        super().__init__(reduce_zero_label=reduce_zero_label, **kwargs)
