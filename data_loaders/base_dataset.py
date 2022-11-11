'''
This file is just for the purpose of testing the code.
'''

import os
import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pass


class BaseTestDataset(Dataset):
    def __init__(self, data_dir):
        self.data_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pass


def test():
    bs = 256
    import os
    from torch.utils.data import DataLoader
    fnames = ["data.npz"] * 100
    bs = 1
    dataset = BaseDataset(fnames)
    data_loader = DataLoader(dataset, batch_size=bs,
                            shuffle=True, num_workers=8, pin_memory=True)
    X, Y = next(iter(data_loader))
    print(Y.shape, X.shape)

# if __name__ == "__main__":
#     test()
