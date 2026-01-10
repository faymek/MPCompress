import os
from torch.utils.data import Dataset
import numpy as np
from dotenv import load_dotenv

load_dotenv()
PROJECT_HOME = os.getenv("PROJECT_HOME")


class Dinov2DatasetTrain(Dataset):
    def __init__(self, train=True):
        if train:
            data_dir = [f"{PROJECT_HOME}/features/fcvq/cls/train"]
        else:
            data_dir = [f"{PROJECT_HOME}/features/fcvq/cls/test"]

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
