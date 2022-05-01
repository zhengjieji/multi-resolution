import os, torch, logging, argparse                                                                              
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class FMSELoss(nn.Module):
    # the higher, the better
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # only use the nonzero pixels for loss calculation
        l = torch.pow((x - y), 2)
        l_nonzero = torch.where(y!=0, l, y) 
        idx_nonzero = torch.nonzero(y)
        return torch.sum(l_nonzero) / idx_nonzero.shape[0]
