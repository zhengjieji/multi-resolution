import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from utils import interpolation


class SRCNN(nn.Module):
    def __init__(self, num_channels=1):
        super(SRCNN, self).__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=9 // 2)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=5, padding=5 // 2)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=5 // 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x


class DRRN(nn.Module):
    def __init__(self):
        super(DRRN, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        inputs = self.input(self.relu(x))
        out = inputs
        for _ in range(5):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.input = nn.Conv2d(in_channels=1, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.conv2 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False)
        self.output = nn.Conv2d(in_channels=128, out_channels=1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, y, args):
        # x is the depth_lq, y is the sidescan
        i = interpolation(x, args.batch_size, args.lq_scale).type(torch.double) 
        residual = i
        inputs = self.input(self.relu(y))
        inputs_x = self.input(self.relu(i))

        # in shortcut should use x
        out = inputs
        for _ in range(1):
            out = self.conv2(self.relu(self.conv1(self.relu(out))))
            out = torch.add(out, inputs_x)

        out = self.output(self.relu(out))
        out = torch.add(out, residual)
        return out

