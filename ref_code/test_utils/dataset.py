import os
from torch.utils.data import Dataset
from PIL import Image


class ImageEnhancementDataset(Dataset):
    '''
    用于指标2的后处理增强
    '''
    def __init__(self, input_dir, transform=None):
        self.input_dir = input_dir
        self.transform = transform
        self.input_images = sorted([f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))])

    def __len__(self):
        return len(self.input_images)

    def __getitem__(self, idx):
        input_path = os.path.join(self.input_dir, self.input_images[idx])
        input_image = Image.open(input_path).convert('RGB')
        
        if self.transform:
            input_image = self.transform(input_image)
        
        return input_image, self.input_images[idx]
