""" Base dataset loaders """

import os

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    def __init__(self, data_dir):
        self.data_list = os.listdir(data_dir)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pass
