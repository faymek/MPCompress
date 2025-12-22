from pathlib import Path
import os
from torch.utils.data import Dataset

from compressai.registry import register_dataset

import numpy as np
import torch


@register_dataset("FeatureFolder")
class FeatureFolder(Dataset):
    """Load a feature folder database.

    This dataset loads feature files (numpy .npy files) from a directory and
    applies preprocessing including truncation, quantization, packing, and random
    cropping. The features are processed according to the specified model type.

    Args:
        root (str): Root directory containing feature files.
        transform (callable, optional): A function or transform that takes in a
            feature and returns a transformed version. Defaults to None.
        split (str): Split mode ('train' or 'val'). Currently unused, features are
            loaded directly from root directory. Defaults to "train".
        model_type (str): Model type for feature packing. Supported types:
            "llama3", "dinov2", "sd3". Defaults to "sd3".
        task (str): Task type. Defaults to "tti".
        trun_flag (bool): Whether to apply truncation to features. Defaults to False.
        trun_low (float or list[float]): Lower bound(s) for truncation. If list,
            each channel has its own bound. Defaults to -20.
        trun_high (float or list[float]): Upper bound(s) for truncation. If list,
            each channel has its own bound. Defaults to 20.
        quant_type (str): Quantization type. Currently only "uniform" is supported.
            Defaults to "uniform".
        qsamples (int): Number of quantization samples. Defaults to 0.
        bit_depth (int): Bit depth for uniform quantization. Defaults to 1.
        patch_size (tuple[int, int]): Patch size for random cropping in format
            (height, width). Must be a multiple of 64. Defaults to (512, 512).
    """

    def __init__(
        self,
        root,
        transform=None,
        split="train",
        model_type="sd3",
        task="tti",
        trun_flag=False,
        trun_low=-20,
        trun_high=20,
        quant_type="uniform",
        qsamples=0,
        bit_depth=1,
        patch_size=(512, 512),
    ):
        # splitdir = Path(root) / split
        splitdir = Path(root)

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.iterdir() if f.is_file())

        self.transform = transform

        # Feature preprocessing parameters
        self.model_type = model_type
        self.task = task
        self.trun_flag = trun_flag
        self.trun_low = trun_low
        self.trun_high = trun_high
        self.quant_type = quant_type
        self.qsamples = qsamples
        self.bit_depth = bit_depth
        self.patch_size = patch_size

    def __getitem__(self, index):
        """Get a feature sample from the dataset.

        Args:
            index (int): Index of the sample to retrieve.

        Returns:
            feat (numpy.ndarray): Preprocessed feature array of shape (1, H, W) after
                truncation, quantization, packing, and random cropping.
        """
        # Load feature, use float32 for training
        feat = np.load(self.samples[index]).astype(np.float32)
        # Apply preprocessing: truncation, quantization, packing, and cropping
        if self.trun_flag is True:
            feat = FeatureFolder.truncation(feat, self.trun_low, self.trun_high)
        feat = FeatureFolder.uniform_quantization(
            feat, self.trun_low, self.trun_high, self.bit_depth
        )
        feat = FeatureFolder.packing(feat, self.model_type)
        feat = FeatureFolder.random_crop(
            feat, self.patch_size
        )  # (height, width), must be a multiple of 64
        feat = np.expand_dims(feat, axis=0)  # Add channel dimension: (1, H, W)
        return feat

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of feature files in the dataset.
        """
        return len(self.samples)

    @staticmethod
    def truncation(feat, trun_low, trun_high):
        """Truncate feature values to specified range.

        Clips feature values to be within [trun_low, trun_high]. Supports
        per-channel truncation when trun_low and trun_high are lists.

        Args:
            feat (numpy.ndarray): Input feature array of shape (N, C, H, W).
            trun_low (float or list[float]): Lower bound(s) for truncation.
            trun_high (float or list[float]): Upper bound(s) for truncation.

        Returns:
            trun_feat (numpy.ndarray): Truncated feature array of the same shape as input.
        """
        trun_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(trun_low, list):
            for idx in range(len(trun_low)):
                trun_feat[:, idx, :, :] = np.clip(
                    feat[:, idx, :, :], trun_low[idx], trun_high[idx]
                )
        else:
            trun_feat = np.clip(feat, trun_low, trun_high)

        return trun_feat

    @staticmethod
    def uniform_quantization(feat, min_v, max_v, bit_depth):
        """Apply uniform quantization to features.

        Quantizes features to integer values in the range [0, 2^bit_depth - 1]
        using uniform quantization. Supports per-channel quantization when
        min_v and max_v are lists.

        Args:
            feat (numpy.ndarray): Input feature array of shape (N, C, H, W).
            min_v (float or list[float]): Minimum value(s) for quantization range.
            max_v (float or list[float]): Maximum value(s) for quantization range.
            bit_depth (int): Number of bits for quantization (determines quantization levels).

        Returns:
            quant_feat (numpy.ndarray): Quantized feature array of the same shape as input.
        """
        quant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2**bit_depth) - 1) / (max_v[idx] - min_v[idx])
                quant_feat[:, idx, :, :] = (feat[:, idx, :, :] - min_v[idx]) * scale
        else:
            scale = ((2**bit_depth) - 1) / (max_v - min_v)
            quant_feat = (feat - min_v) * scale

        return quant_feat

    @staticmethod
    def uniform_dequantization(feat, min_v, max_v, bit_depth):
        """Apply uniform dequantization to features.

        Converts quantized integer features back to continuous values using
        uniform dequantization. Supports per-channel dequantization when
        min_v and max_v are lists.

        Args:
            feat (numpy.ndarray): Quantized feature array of shape (N, C, H, W).
            min_v (float or list[float]): Minimum value(s) for dequantization range.
            max_v (float or list[float]): Maximum value(s) for dequantization range.
            bit_depth (int): Number of bits used for quantization.

        Returns:
            dequant_feat (numpy.ndarray): Dequantized feature array of the same shape as input.
        """
        feat = feat.astype(np.float32)
        dequant_feat = np.zeros_like(feat).astype(np.float32)
        if isinstance(min_v, list):
            for idx in range(len(min_v)):
                scale = ((2**bit_depth) - 1) / (max_v[idx] - min_v[idx])
                dequant_feat[:, idx, :, :] = feat[:, idx, :, :] / scale + min_v[idx]
        else:
            scale = ((2**bit_depth) - 1) / (max_v - min_v)
            dequant_feat = feat / scale + min_v
        return dequant_feat

    @staticmethod
    def packing(feat, model_type):
        """Pack features according to model type.

        Reshapes features from (N, C, H, W) format to a 2D array format
        specific to the model type. This is used for compatibility with
        different model architectures.

        Args:
            feat (numpy.ndarray): Input feature array of shape (N, C, H, W).
            model_type (str): Model type. Supported types:
                - "llama3": Extracts single channel, returns (H, W)
                - "dinov2": Reshapes to (N*H, C*W)
                - "sd3": Reshapes to (C/4*H, C/4*W)

        Returns:
            feat (numpy.ndarray): Packed feature array with shape depending on model_type.
        """
        N, C, H, W = feat.shape
        if model_type == "llama3":
            feat = feat[0, 0, :, :]
        elif model_type == "dinov2":
            feat = feat.transpose(0, 2, 1, 3).reshape(N * H, C * W)
        elif model_type == "sd3":
            feat = (
                feat.reshape(int(C / 4), int(C / 4), H, W)
                .transpose(0, 2, 1, 3)
                .reshape(int(C / 4 * H), int(C / 4 * W))
            )
        return feat

    @staticmethod
    def unpacking(feat, shape, model_type):
        """Unpack features according to model type.

        Reshapes packed features back to (N, C, H, W) format. This is the
        inverse operation of packing.

        Args:
            feat (numpy.ndarray): Packed feature array.
            shape (tuple[int, int, int, int]): Target shape (N, C, H, W).
            model_type (str): Model type. Supported types:
                - "llama3": Expands to (1, 1, H, W)
                - "dinov2": Reshapes from (N*H, C*W) to (N, C, H, W)
                - "sd3": Reshapes from (C/4*H, C/4*W) to (N, C, H, W)

        Returns:
            feat (numpy.ndarray): Unpacked feature array of shape (N, C, H, W).
        """
        N, C, H, W = shape
        if model_type == "llama3":
            feat = np.expand_dims(feat, axis=0)
            feat = np.expand_dims(feat, axis=0)
        elif model_type == "dinov2":
            feat = feat.reshape(N, H, C, W).transpose(0, 2, 1, 3)
        elif model_type == "sd3":
            feat = (
                feat.reshape(int(C / 4), H, int(C / 4), W)
                .transpose(0, 2, 1, 3)
                .reshape(N, C, H, W)
            )
        return feat

    @staticmethod
    def random_crop(feat, crop_shape):
        """Randomly crop a feature array to specified shape.

        Args:
            feat (numpy.ndarray): Input feature array of shape (H, W).
            crop_shape (tuple[int, int]): Desired crop size in format (height, width).

        Returns:
            feat (numpy.ndarray): Cropped feature array of shape crop_shape.

        Raises:
            ValueError: If crop_shape exceeds the feature dimensions.
        """
        max_row = feat.shape[0] - crop_shape[0]
        max_col = feat.shape[1] - crop_shape[1]

        if max_row < 0 or max_col < 0:
            print(feat.shape[0], crop_shape[0])
            print(feat.shape[1], crop_shape[1])
            raise ValueError("crop_shape exceeds the feature shape")

        start_row = np.random.randint(0, max_row + 1)
        start_col = np.random.randint(0, max_col + 1)

        end_row = start_row + crop_shape[0]
        end_col = start_col + crop_shape[1]

        return feat[start_row:end_row, start_col:end_col]


class FeatureDictPerSampleFolder(Dataset):
    """Dataset for loading feature dictionaries stored as separate .pt files.

    Each sample is stored as a separate PyTorch .pt file containing a dictionary.
    This is useful when each sample has different keys or when features are
    preprocessed and saved individually.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): A function or transform to apply to each
            sample. Defaults to None.
        split (str): Subdirectory name within root (e.g., 'train', 'val').
            Defaults to "train".
    """

    def __init__(self, root, transform=None, split="train"):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        self.samples = sorted(f for f in splitdir.rglob("*.pt") if f.is_file())
        self.transform = transform

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of .pt files in the dataset.
        """
        return len(self.samples)

    def __getitem__(self, idx):
        """Get a feature dictionary sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            data (dict): Feature dictionary loaded from the .pt file. If transform is
                provided, the transformed version is returned.
        """
        file = self.samples[idx]
        with open(file, "rb") as f:
            data = torch.load(f, map_location="cpu", weights_only=True)

        if self.transform:
            data = self.transform(data)

        return data


class FeatureDictPerKeyFolder(Dataset):
    """Dataset for loading feature dictionaries organized by keys.

    Each key in the feature dictionary is stored in a separate folder. This
    organization allows loading only the specified keys, reducing disk I/O usage.
    All keys must have the same number of samples.

    Directory structure:

        root/
            split/
                key1/
                    sample0.pt
                    sample1.pt
                    ...
                key2/
                    sample0.pt
                    sample1.pt
                    ...

    Args:
        root (str): Root directory of the dataset.
        split (str): Subdirectory name within root (e.g., 'train', 'val').
            Defaults to "" (empty string, meaning root itself).
        keys (list[str], optional): List of keys to load. If None, all keys
            found in the directory are loaded. Defaults to None.
        transform (callable, optional): A function or transform to apply to each
            sample dictionary. Defaults to None.
    """

    def __init__(self, root, split="", keys=None, transform=None):
        root = os.path.join(root, split)
        self.root = root
        self.transform = transform
        all_keys = os.listdir(root)
        self.keys = keys if keys is not None else all_keys

        self.samples = {key: [] for key in self.keys}
        for key in self.keys:
            self.samples[key] = [
                os.path.join(root, key, f) for f in os.listdir(os.path.join(root, key))
            ]

        key_lengths = {key: len(self.samples[key]) for key in self.keys}
        assert len(set(key_lengths.values())) == 1, "All keys must have the same length"
        self.length = len(self.samples[self.keys[0]])

        print(f"Dataset loaded successfully, {self.length} samples")
        print(f"Keys included: {self.keys}")

    def __len__(self):
        """Return the number of samples in the dataset.

        Returns:
            length (int): Number of samples (same for all keys).
        """
        return self.length

    def __getitem__(self, idx):
        """Get a feature dictionary sample from the dataset.

        Loads the idx-th sample for each specified key and combines them into
        a single dictionary.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            data (dict): Dictionary containing all specified keys with their corresponding
                values. If transform is provided, the transformed version is returned.
        """
        data = {}
        for key in self.keys:
            with open(self.samples[key][idx], "rb") as f:
                value = torch.load(f, map_location="cpu", weights_only=True)
                data[key] = value

        if self.transform:
            data = self.transform(data)

        return data


def feature_dict_collate_fn(batch):
    """Custom collate function for feature dictionaries.

    Merges multiple feature dictionaries into a single batch. For each key:
    - If the value is a torch.Tensor, stacks all values along dimension 0.
    - If the value is a torch.Size, creates a new Size with batch dimension.
    - Otherwise, takes the first value.

    Args:
        batch (list[dict]): List of feature dictionaries to collate.

    Returns:
        collated (dict): Collated dictionary with the same keys as input, where tensor
            values are stacked and other values are taken from the first sample.
    """
    collated = {}
    first = batch[0]
    for key in first.keys():
        values = [item[key] for item in batch]
        if isinstance(values[0], torch.Tensor):
            collated[key] = torch.stack(values, dim=0)
        elif isinstance(values[0], torch.Size):
            new_size = (len(batch),) + values[0]
            collated[key] = torch.Size(new_size)
        else:
            collated[key] = values[0]
    return collated
