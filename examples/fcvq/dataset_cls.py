import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms, datasets
import numpy as np
class Dinov2DatasetTrain(Dataset):
    def __init__(self, train=True):
        if train:
            data_dir = ['/data/qiaoxichen/model/dinov2_dataset/cls/train']
        else:
            data_dir = ['/data/qiaoxichen/model/dinov2_dataset/cls/test']

        self.data_dir = data_dir
        self.file_list = []
        for dir in data_dir:
            
            self.file_list.extend(os.listdir(dir))

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        
        file_name = self.file_list[index]
        for data_dir in self.data_dir:
            if file_name in os.listdir(data_dir):
                file_path = os.path.join(data_dir, file_name)
                feat = np.load(file_path)
                return feat, file_name

