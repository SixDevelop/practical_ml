import os
from typing import Callable, Tuple

import torch
from torch.utils.data import Dataset
from torchvision.datasets.mnist import read_image_file, read_label_file


class FashionMNISTDataset(Dataset):
    def __init__(
        self,
        file_dir,
        train: bool=True,
        transform: Callable=None,
        target_transform: Callable=None
    ):
        self.transform = transform
        self.target_transform = target_transform
        self.train = train
        image_file = f"{'train' if self.train else 't10k'}-images-idx3-ubyte"
        self.data = read_image_file(os.path.join(file_dir, image_file))

        label_file = f"{'train' if self.train else 't10k'}-labels-idx1-ubyte"
        self.targets = read_label_file(os.path.join(file_dir, label_file))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label