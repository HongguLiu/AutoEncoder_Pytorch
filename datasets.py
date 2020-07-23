import torch
from torch.utils.data import Dataset
import os
import numpy as np
import cv2
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, txt_path, transform=None):
        fh = open(txt_path, 'r')
        imgs = []
        for line in fh:
            line = line.rstrip()
            imgs.append(line)
        fh.close
        self.imgs = imgs
        self.transform = transform

    def __getitem__(self, index):
        fn = self.imgs[index]
        img = Image.open(fn).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)
        return img
    def __len__(self):
        return len(self.imgs)