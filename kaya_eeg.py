import os
import pandas as pd
import numpy as np
from torch.utils.data import Dataset


class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, data_dir, transform=None, target_transform=None):
        print(annotations_file)
        self.labels = pd.read_csv(annotations_file)
        self.dir = data_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        dpath = os.path.join(self.dir, self.labels.iloc[idx, 0])
        image = np.load(dpath)
        label = self.labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        sample = {"image": image, "label": label}
        return sample

