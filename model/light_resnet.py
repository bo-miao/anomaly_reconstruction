import math
import torch.utils.model_zoo as model_zoo
# from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, Parameter
import torchvision.models as models

from .block import *



class HeadBlock(nn.Module):

    def __init__(self, in_c=3, out_c=32, BatchNorm=nn.BatchNorm2d):
        super(HeadBlock, self).__init__()
        mid = out_c//2
        self.conv1 = nn.Conv2d(in_c, mid, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm1 = BatchNorm(mid)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(mid, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.norm2 = BatchNorm(out_c)
        self.relu2 = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        x = self.relu1(self.norm1(self.conv1(x)))
        x = self.relu2(self.norm2(self.conv2(x)))
        x = self.maxpool(x)
        return x


class Bottleneck(nn.Module):

    def __init__(self, in_c, out_c, stride=1, dilation=1, BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_c, in_c, kernel_size=1, bias=False)
        self.bn1 = BatchNorm(in_c)
        self.conv2 = nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride,
                               dilation=dilation, padding=dilation, bias=False)
        self.bn2 = BatchNorm(in_c)
        self.conv3 = nn.Conv2d(in_c, out_c, kernel_size=1, bias=False)
        self.bn3 = BatchNorm(out_c)
        self.relu = nn.ReLU(inplace=True)
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += residual
        out = self.relu(out)

        return out

