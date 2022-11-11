'''
This file is just for the purpose of testing the code.
'''

import torch.nn as nn
import torch.nn.functional as F


class BaseModel(nn.Module):
    def __init__(self, cfg):
        super(BaseModel, self).__init__()
        pass

    def forward(self, x):
        pass
