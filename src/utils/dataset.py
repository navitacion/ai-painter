import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


# CycleGAN Dataset ---------------------------------------------------------------------------
class CycleGanDataset(Dataset):
    def __init__(self, base_img_paths, style_img_paths, transform, phase='train'):
        self.base_img_paths = base_img_paths
        self.style_img_paths = style_img_paths
        self.transform = transform
        self.phase = phase

    def __len__(self):
        return min([len(self.base_img_paths), len(self.style_img_paths)])

    def __getitem__(self, idx):
        base_img_path = self.base_img_paths[idx]
        style_img_path = self.style_img_paths[idx]
        base_img = Image.open(base_img_path)
        # base_img = cv2.cvtColor(base_img, cv2.COLOR_BGR2RGB)
        style_img = Image.open(style_img_path)
        # style_img = cv2.cvtColor(style_img, cv2.COLOR_BGR2RGB)

        base_img = self.transform(base_img, self.phase)
        style_img = self.transform(style_img, self.phase)

        return base_img, style_img