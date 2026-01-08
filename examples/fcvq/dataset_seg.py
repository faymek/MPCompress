import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets

import numpy as np

class Dinov2DatasetTrain(Dataset):
    def __init__(self, train=True):
        if train:
            data_dirs = ['/data/qiaoxichen/model/dinov2_dataset/seg/train']
        else:
            data_dirs = ['/data/qiaoxichen/model/dinov2_dataset/seg/test']

        self.file_list = []
        for d in data_dirs:
            for f in os.listdir(d):
                if f.endswith(".npy"):
                    self.file_list.append(os.path.join(d, f))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):

        file_path = self.file_list[index]
        feat = np.load(file_path)
        feat = feat[0]
        file_name = os.path.basename(file_path)
        return feat, file_name

class Dinov2DatasetTest(datasets.ImageFolder):


    def __init__(
            self,
            root: str,
            transform=None,
            **kwargs
    ):
        super().__init__(
            root,
            transform,
            **kwargs
        )

    def __getitem__(self, index: int):
        
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target, path

