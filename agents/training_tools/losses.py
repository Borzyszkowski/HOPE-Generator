""" Implementation of the loss functions """

import torch.nn.functional as F


def nll_loss(output, target):
    """ The negative log likelihood loss """
    return F.nll_loss(output, target)
