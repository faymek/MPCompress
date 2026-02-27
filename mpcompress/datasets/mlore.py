"""
MLoRE Dataset Adapters for MPCompress Framework

This module provides dataset wrappers for PASCAL-Context and NYUD datasets,
adapted to work with the MPCompress framework while maintaining compatibility
with RFC's original data loading.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
import torch.utils.data as data
from PIL import Image

# RFC code has been migrated to mpcompress, no need for external RFC dependency


__all__ = [
    "MLoREImageDataset",
    "PASCALContextDataset",
    "NYUDDataset",
    "get_mlore_transforms",
    "get_mlore_dataset",
    "collate_mlore",
]


class MLoREImageDataset(data.Dataset):
    """
    Base class for MLoRE multi-task image datasets.
    
    Provides a unified interface returning (img, img_meta) format
    as specified by the MPCompress framework.
    
    Args:
        root: Dataset root directory
        split: Data split ('train' or 'val')
        tasks: List of tasks to load
        transform: Optional transform pipeline
        img_size: Target image size (H, W)
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        tasks: List[str] = None,
        transform=None,
        img_size: Tuple[int, int] = (512, 512),
    ):
        self.root = root
        self.split = split
        self.tasks = tasks or ['semseg', 'edge']
        self.transform = transform
        self.img_size = img_size
        
        # To be implemented by subclasses
        self.images = []
        self.im_ids = []
        self.labels = {}  # Dict mapping task -> list of label paths
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        # Load image
        img = self._load_image(index)
        
        # Create img_meta following MPCompress format
        img_meta = {
            "img_path": self.images[index],
            "img_name": self.im_ids[index],
            "ori_size": img.shape[:2] if isinstance(img, np.ndarray) else img.size[::-1],
        }
        
        # Create sample dict
        sample = {"image": img, "meta": img_meta}
        
        # Load labels for each task
        for task in self.tasks:
            if task in self.labels and len(self.labels[task]) > index:
                label = self._load_label(index, task)
                if label is not None:
                    sample[task] = label
                    img_meta[f"{task}_label_path"] = self.labels[task][index]
        
        # Apply transforms if provided
        if self.transform is not None:
            sample = self.transform(sample)
        
        return sample
    
    def _load_image(self, index):
        """Load image at given index."""
        img = np.array(Image.open(self.images[index]).convert('RGB')).astype(np.float32)
        return img
    
    def _load_label(self, index, task):
        """Load label for given task at given index."""
        raise NotImplementedError


class PASCALContextDataset(MLoREImageDataset):
    """
    PASCAL-Context dataset adapter for MPCompress.
    
    Supports tasks:
    - semseg: Semantic segmentation (21 classes)
    - edge: Edge detection (binary)
    - human_parts: Human part segmentation (7 classes)
    - normals: Surface normal estimation (3 channels)
    - sal: Saliency detection (binary)
    
    Args:
        root: Dataset root directory
        split: Data split ('train' or 'val')
        tasks: List of tasks to load
        transform: Optional transform pipeline
        download: Whether to download if not exists
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        tasks: List[str] = None,
        transform=None,
        download: bool = False,
        **kwargs
    ):
        super().__init__(root, split, tasks, transform, **kwargs)
        
        # Import original dataset class
        from mpcompress.datasets.rfcdata.pascal_context import PASCALContext
        
        # Map tasks to dataset flags
        task_flags = {
            'semseg': 'do_semseg',
            'edge': 'do_edge',
            'human_parts': 'do_human_parts',
            'normals': 'do_normals',
            'sal': 'do_sal',
        }
        
        flags = {task_flags[t]: True for t in self.tasks if t in task_flags}
        
        # Create underlying dataset
        self._dataset = PASCALContext(
            root=root,
            download=download,
            split=[split] if isinstance(split, str) else split,
            transform=None,  # We handle transforms ourselves
            retname=True,
            **flags
        )
        
        self.images = self._dataset.images
        self.im_ids = self._dataset.im_ids
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        # Get sample from underlying dataset
        sample = self._dataset[index]
        
        # Convert to MPCompress format
        img_meta = {
            "img_path": self.images[index],
            "img_name": sample['meta']['img_name'],
            # RFC/evaluation链路使用img_size/img_name字段
            "img_size": sample['meta']['img_size'],
            # 保留旧字段，避免破坏现有调用
            "ori_size": sample['meta']['img_size'],
        }
        
        result = {
            "image": sample["image"],
            "meta": img_meta,
        }
        
        # Copy task labels
        for task in self.tasks:
            if task in sample:
                result[task] = sample[task]
        
        # Apply transforms
        if self.transform is not None:
            result = self.transform(result)
        
        return result


class NYUDDataset(MLoREImageDataset):
    """
    NYUD (NYU Depth V2) dataset adapter for MPCompress.
    
    Supports tasks:
    - semseg: Semantic segmentation (13 classes)
    - edge: Edge detection (binary)
    - normals: Surface normal estimation (3 channels)
    - depth: Depth estimation (1 channel)
    - scene: Scene classification (13 classes)
    
    Args:
        root: Dataset root directory
        split: Data split ('train' or 'val')
        tasks: List of tasks to load
        transform: Optional transform pipeline
        download: Whether to download if not exists
    """
    
    def __init__(
        self,
        root: str,
        split: str = 'val',
        tasks: List[str] = None,
        transform=None,
        download: bool = False,
        **kwargs
    ):
        super().__init__(root, split, tasks, transform, **kwargs)
        
        # Import original dataset class
        from rfcdata.nyud import NYUD_MT
        
        # Map tasks to dataset flags
        task_flags = {
            'semseg': 'do_semseg',
            'edge': 'do_edge',
            'normals': 'do_normals',
            'depth': 'do_depth',
            'scene': 'do_scene',
        }
        
        flags = {task_flags[t]: True for t in self.tasks if t in task_flags}
        
        # Create underlying dataset
        self._dataset = NYUD_MT(
            root=root,
            download=download,
            split=split,
            transform=None,
            **flags
        )
        
        self.images = self._dataset.images
        self.im_ids = self._dataset.im_ids if hasattr(self._dataset, 'im_ids') else [
            Path(p).stem for p in self.images
        ]
    
    def __len__(self):
        return len(self._dataset)
    
    def __getitem__(self, index):
        # Get sample from underlying dataset
        sample = self._dataset[index]
        
        # Convert to MPCompress format
        img_meta = {
            "img_path": self.images[index],
            "img_name": self.im_ids[index],
            # RFC/evaluation链路使用img_size/img_name字段
            "img_size": sample['meta']['img_size'] if 'meta' in sample else sample['image'].shape[:2],
            # 保留旧字段，避免破坏现有调用
            "ori_size": sample['meta']['img_size'] if 'meta' in sample else sample['image'].shape[:2],
        }
        
        result = {
            "image": sample["image"],
            "meta": img_meta,
        }
        
        # Copy task labels
        for task in self.tasks:
            if task in sample:
                result[task] = sample[task]
        
        # Apply transforms
        if self.transform is not None:
            result = self.transform(result)
        
        return result


def get_mlore_transforms(p=None, split='train'):
    """
    Get data transforms for MLoRE datasets.
    
    Args:
        p: Configuration dict (optional)
        split: Data split ('train' or 'val')
        
    Returns:
        Transform pipeline
    """
    import torchvision
    from mpcompress.datasets.rfcdata import transforms
    
    # Note: transforms are handled inline, no external dependency needed
    
    # Default scales
    train_scale = p.TRAIN.SCALE if p else (512, 512)
    test_scale = p.TEST.SCALE if p else (512, 512)
    
    if split == 'train':
        transform = torchvision.transforms.Compose([
            transforms.RandomScaling(scale_factors=[0.5, 2.0], discrete=False),
            transforms.RandomCrop(size=train_scale, cat_max_ratio=0.75),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.PhotoMetricDistortion(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=train_scale),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
    else:
        transform = torchvision.transforms.Compose([
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.PadImage(size=test_scale),
            transforms.AddIgnoreRegions(),
            transforms.ToTensor(),
        ])
    
    return transform


def get_mlore_dataset(
    dataset_name: str,
    root: str,
    split: str = 'val',
    tasks: List[str] = None,
    transform=None,
    p=None,
    **kwargs
):
    """
    Factory function to create MLoRE datasets.
    
    Args:
        dataset_name: Dataset name ('PASCALContext' or 'NYUD')
        root: Dataset root directory
        split: Data split
        tasks: List of tasks
        transform: Optional transform (auto-created if None)
        p: Configuration dict
        **kwargs: Additional dataset arguments
        
    Returns:
        Dataset instance
    """
    if transform is None:
        transform = get_mlore_transforms(p, split)
    
    if dataset_name == 'PASCALContext':
        return PASCALContextDataset(
            root=root,
            split=split,
            tasks=tasks,
            transform=transform,
            **kwargs
        )
    elif dataset_name == 'NYUD':
        return NYUDDataset(
            root=root,
            split=split,
            tasks=tasks,
            transform=transform,
            **kwargs
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def collate_mlore(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Custom collate function for MLoRE datasets.
    
    Handles variable-sized images and task labels by stacking tensors
    and collecting metadata.
    
    Args:
        batch: List of sample dicts from dataset
        
    Returns:
        Collated batch dict
    """
    # Separate image, labels, and meta
    images = []
    metas = []
    task_labels = {}
    
    for sample in batch:
        images.append(sample['image'])
        metas.append(sample['meta'])
        
        for key in sample:
            if key not in ['image', 'meta']:
                if key not in task_labels:
                    task_labels[key] = []
                task_labels[key].append(sample[key])
    
    # Stack images
    if isinstance(images[0], torch.Tensor):
        images = torch.stack(images, dim=0)
    else:
        images = torch.stack([torch.from_numpy(img) for img in images], dim=0)
    
    # Stack task labels
    for key in task_labels:
        labels = task_labels[key]
        if isinstance(labels[0], torch.Tensor):
            task_labels[key] = torch.stack(labels, dim=0)
        else:
            task_labels[key] = torch.stack([torch.from_numpy(l) for l in labels], dim=0)
    
    # Build result
    result = {
        'image': images,
        'meta': metas,
        **task_labels
    }
    
    return result


# Wrapper to use RFC's original common_config functions
class RFCDatasetWrapper:
    """
    Wrapper to use RFC's original dataset creation functions.
    
    This provides exact compatibility with RFC's data loading pipeline.
    """
    
    @staticmethod
    def get_train_dataset(p, transforms=None):
        """Get training dataset using RFC's function."""
        from mpcompress.utils.mlore_common_config import get_train_dataset
        return get_train_dataset(p, transforms)
    
    @staticmethod
    def get_test_dataset(p, transforms=None):
        """Get test dataset using RFC's function."""
        from mpcompress.utils.mlore_common_config import get_test_dataset
        return get_test_dataset(p, transforms)
    
    @staticmethod
    def get_transformations(p):
        """Get transforms using RFC's function."""
        from mpcompress.utils.mlore_common_config import get_transformations
        return get_transformations(p)
    
    @staticmethod
    def get_train_dataloader(p, dataset, sampler=None):
        """Get training dataloader using RFC's function."""
        from mpcompress.utils.mlore_common_config import get_train_dataloader
        return get_train_dataloader(p, dataset, sampler)
    
    @staticmethod
    def get_test_dataloader(p, dataset):
        """Get test dataloader using RFC's function."""
        from mpcompress.utils.mlore_common_config import get_test_dataloader
        return get_test_dataloader(p, dataset)




