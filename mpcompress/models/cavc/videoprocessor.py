import numpy as np
import cv2
from tqdm import tqdm


from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


class MP4VideoFolder(Dataset):
    """Load an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

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

    def __init__(self, root, transform=None, split="train", src_type="yuv420", sequences=[]):
        splitdir = Path(root) / split

        if not splitdir.is_dir():
            raise RuntimeError(f'Missing directory "{splitdir}"')

        assert src_type in ["mp4"]
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
            img: `PIL.Image.Image` or transformed `PIL.Image.Image`.
        """
        seq_name = self.sequences_names[index]
        vid_meta = self.sequences_meta[seq_name]
        path = str(self.splitdir / seq_name) 
        vid_meta["path"] = path
        vid_meta["seq_name"] = seq_name
        reader = VideoReader(path)
        return reader, vid_meta


class VideoDetectionDataset(Dataset):
    pass


class VideoReader:
    def __init__(self, video_path):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
    def __iter__(self):
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0) # 重置到开头
        return self

    def __next__(self):
        ret, frame = self.cap.read()
        if not ret:
            raise StopIteration
        return frame # 返回 BGR frame

    def release(self):
        self.cap.release()


class VideoWriterWrapper:
    """
    对应原 process_video_with_feat 中的写入逻辑
    """
    def __init__(self):
        self.writer = None

    def write_batch(self, image_tensor_list, output_path, fps, size):
        """
        将 Tensor 列表写入视频
        Args:
            image_tensor_list: List of Tensors [1, 3, H, W] (范围 0-1)
            output_path: 输出路径
            fps: 帧率
            size: (width, height)
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.writer = cv2.VideoWriter(output_path, fourcc, fps, size)
        
        for tensor in tqdm(image_tensor_list, desc="Writing Video"):
            # --- 逻辑来自原代码 output_frame 处理部分 ---
            
            # 1. Squeeze & CPU & Numpy
            output_frame = tensor.squeeze().cpu().detach().numpy() # [3,H,W]
            
            # 2. Transpose [3,H,W] -> [H,W,3]
            output_frame = np.transpose(output_frame, (1, 2, 0))
            
            # 3. Clip & Scale & Cast
            output_frame = np.clip(output_frame, 0.0, 1.0)
            output_frame = (output_frame * 255).astype(np.uint8)
            
            # 4. RGB -> BGR (OpenCV 写入需要 BGR)
            output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)
            
            self.writer.write(output_frame)
            
        self.writer.release()