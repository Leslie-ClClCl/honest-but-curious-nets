import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets.folder import pil_loader
import torchvision.transforms as transforms


class IMDB_WIKI(Dataset):
    r"""
    - [age] is an integer from 0 to 116, indicating the age
    - [gender] is either 1 (male) or 0 (female)
    """
    def __init__(self, image_size=64, root="./", transform=None):
        self.root = root
        meta = pd.read_csv('meta.csv')
        self.meta = meta
        if transform is None:
            transform = transforms.Compose([
                transforms.Resize(size=(image_size, image_size)),
                transforms.ToTensor(),
                transforms.Lambda(lambda t: t.mul(2).sub(1)),  # From [0,1] to [-1,1]
            ])
        self.transform = transform

    def __getitem__(self, index):
        col = self.meta.loc[index]
        age, gender, path = col['age'], col['gender'], col['path']
        image = pil_loader(path)
        image = self.transform(image)
        label = {'age': age, 'gender': gender}
        return image, label

    def __len__(self):
        return len(self.meta)


if __name__ == '__main__':
    dataset = IMDB_WIKI(64)
    data_loader = DataLoader(dataset, len(dataset), shuffle=False)
    images, labels_dict = iter(data_loader).next()
    pass
