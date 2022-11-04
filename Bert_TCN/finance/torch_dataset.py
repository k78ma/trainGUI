import torch
from torch.utils.data import Dataset, DataLoader


class Financial_dataset():

    def __init__(self, x_set, y_set):
        self.x_set = x_set
        self.y_set = y_set

    def __len__(self):
        return self.x_set.shape[0]

    def __getitem__(self, idx):
        return self.x_set[idx, :, :], self.y_set[idx]
