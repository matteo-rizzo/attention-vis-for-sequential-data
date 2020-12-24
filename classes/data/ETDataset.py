import os
import pickle
from typing import Union

import torch
from torch.utils.data import Dataset

"""Eye Tracking Dataset"""


class ETDataset(Dataset):

    def __init__(self, data_paths: list, labels: list, transform: callable = None):
        """
        @param data_paths: list of paths to sequences
        @param labels: list of corresponding labels
        @param transform: Optional transform to be applied on a sequence
        """
        # self.sequences = [pd.read_csv(path) for path in data_paths]
        self.sequences = [pickle.load(open(os.path.join(path), 'rb')) for path in data_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: Union[int, torch.Tensor]):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.sequences[idx], self.labels[idx]
        sample = (x, y)
        if self.transform:
            sample = (self.transform(x), y)
        return sample
