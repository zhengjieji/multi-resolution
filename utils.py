import os, torch, logging, argparse, cv2
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm

from dataset import get_dataset

def calc_psnr(x, y):
    # only use the nonzero pixels for loss calculation                                      
    l = torch.pow((x - y), 2)
    l_nonzero = torch.where(y!=0, l, y) 
    idx_nonzero = torch.nonzero(y)
    mean = torch.sum(l_nonzero) / idx_nonzero.shape[0]
    return 10. * torch.log10(1. / mean)

def interpolation(img, batch_size, lq_scale):
    # bicubic interpolation for residual shortcut
    processed = torch.zeros((batch_size, 1, img[0][0].shape[0], img[0][0].shape[1]))
    for i in range(batch_size):
        tmp = torch.zeros((1, int(img[0][0].shape[0] / lq_scale), img[0][0].shape[1]))
        for j in range(img[i][0].shape[0]):
            if j % lq_scale == 0:
                tmp[0][int(j / lq_scale)] = img[i][0][j]
        expanded = torch.tensor(cv2.resize(tmp[0].numpy(), (256, 256), interpolation=cv2.INTER_CUBIC))
        processed[i][0] = expanded
    return processed


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

