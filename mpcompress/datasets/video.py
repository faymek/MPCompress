from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np
from .video_reader import YUV420VideoReader, PngSequenceVideoReader


class VideoFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

        - rootdir/
            - train/
                - video000.yuv
                - video001.yuv
            - test/
                - video002.yuv
                - video003.yuv

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(
        self, root, transform=None, split="train", src_type="yuv420", sequences=[]
    ):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        assert src_type in ["yuv420", "png"]
        assert len(sequences) > 0
        self.splitdir = splitdir
        self.src_type = src_type
        self.sequences_meta = sequences
        self.sequences_names = list(sequences.keys())
        self.transform = transform

    def __len__(self):
        return len(self.sequences_names)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            reader (VideoReader): Video reader object.
            vid_meta (dict): Video metadata.
        """
        seq_name = self.sequences_names[index]
        vid_meta = self.sequences_meta[seq_name]
        vid_meta["seq_name"] = seq_name
        path = str(self.splitdir / seq_name)
        if self.src_type == "yuv420":
            reader = YUV420VideoReader(
                path, vid_meta["src_width"], vid_meta["src_height"]
            )
        elif self.src_type == "png":
            reader = PngSequenceVideoReader(
                path, vid_meta["src_width"], vid_meta["src_height"]
            )
        return reader, vid_meta


class VideoDetectionDataset(Dataset):
    pass
