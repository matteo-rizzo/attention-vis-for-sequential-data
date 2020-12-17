import pandas as pd
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
        self.sequences = [pd.read_csv(path) for path in data_paths]
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx: int):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x, y = self.sequences[idx], self.labels[idx]
        sample = (x, y)
        if self.transform:
            sample = (self.transform(x), y)
        return sample
