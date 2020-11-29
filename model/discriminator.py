import torch.nn as nn
import torchvision.models as models
from torch.cuda.amp import autocast
import torch.nn.functional as F

from .norm import *


class FocalWithLogitsLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, reduction=True):
        super(FocalWithLogitsLoss, self).__init__()
        self.alpha = alpha  # imbalance
        self.gamma = gamma  # hard
        self.reduction = reduction

    def forward(self, inputs, targets):
        pt = torch.sigmoid(inputs)
        loss = - self.alpha * (1 - pt) ** self.gamma * targets * torch.log(pt) - \
               (1 - self.alpha) * pt ** self.gamma * (1 - targets) * torch.log(1 - pt)
        if self.reduction:
            return torch.mean(loss)
        else:
            return loss


class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=3, reduction=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # imbalance
        self.gamma = gamma  # hard
        self.reduction = reduction

    def forward(self, inputs, targets):
        loss = - self.alpha * (1 - inputs) ** self.gamma * targets * torch.log(inputs) - \
               (1 - self.alpha) * inputs ** self.gamma * (1 - targets) * torch.log(1 - inputs)
        if self.reduction:
            return torch.mean(loss)
        else:
            return loss


class Discriminator(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, f):
        super(Discriminator, self).__init__()
        self.map1 = nn.Linear(input_size, hidden_size)
        self.map2 = nn.Linear(hidden_size, hidden_size)
        self.map3 = nn.Linear(hidden_size, output_size)
        self.f = f

    def forward(self, x):
        x = self.f(self.map1(x))
        x = self.f(self.map2(x))
        return self.f(self.map3(x))


class ResNetD(nn.Module):
    def __init__(self, criterion=None):
        super(ResNetD, self).__init__()
        self.model = models.resnet18(pretrained=False, num_classes=2, norm_layer=norm("group", gn_num_groups=4))
        self.criterion = criterion

    @autocast()
    def forward(self, x, target):
        x = self.model(x)
        loss = self.criterion(x, target)
        return loss, x


class Bottleneck(nn.Module):
    def __init__(self, in_c, out_c, stride=1, dilation=1, BatchNorm=nn.BatchNorm2d):
        super(Bottleneck, self).__init__()
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=in_c, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(in_c),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = torch.nn.Sequential(
            nn.Conv2d(in_c, in_c, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False),
            torch.nn.BatchNorm2d(in_c),
            torch.nn.ReLU(inplace=True)
        )

        self.conv3 = torch.nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=1, bias=False),
            torch.nn.BatchNorm2d(out_c)
        )

        self.need_downsample = stride > 1 or in_c != out_c
        self.downsample = nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False)
        self.relu = torch.nn.ReLU(inplace=True)
        self.activation = torch.nn.Sigmoid()
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.need_downsample:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return self.activation(out)


def _make_layers(block=Bottleneck, num=1, in_c=1, out_c=1, stride=1):
    assert num >= 1, "layer block num must larger than 1"
    layer = []
    for i in range(num-1):
        layer.append(block(in_c, in_c, stride=1))
    layer.append(block(in_c, out_c, stride=stride))
    return nn.Sequential(*layer)


class ShallowNetD(nn.Module):
    def __init__(self, criterion=None, args=None):
        super(ShallowNetD, self).__init__()

        c = [32, 64, 128, 256, 1]
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=c[0], kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(c[0]),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = _make_layers(block=Bottleneck, num=2, in_c=c[0], out_c=c[1], stride=2)
        self.conv3 = _make_layers(block=Bottleneck, num=3, in_c=c[1], out_c=c[2], stride=2)
        self.conv4 = _make_layers(block=Bottleneck, num=4, in_c=c[2], out_c=c[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(c[3], c[-1])
        self.sigmoid = nn.Sigmoid()
        self.criterion = criterion
        nn.init.normal_(self.fc.weight, std=0.01)

    @autocast()
    def forward(self, x, target=None, train=True):
        b = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).view(b, -1)
        x = self.fc(x).view(-1)

        if train:
            loss = self.criterion(x, target)
            return loss, self.sigmoid(x)
        else:
            return self.sigmoid(x)


class ShallowNetD_Logit(nn.Module):
    def __init__(self, criterion=None, args=None):
        super(ShallowNetD_Logit, self).__init__()

        c = [32, 64, 128, 256, 1]
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=c[0], kernel_size=5, stride=2, padding=2),
            torch.nn.BatchNorm2d(c[0]),
            torch.nn.ReLU(inplace=True)
        )

        self.conv2 = _make_layers(block=Bottleneck, num=2, in_c=c[0], out_c=c[1], stride=2)
        self.conv3 = _make_layers(block=Bottleneck, num=3, in_c=c[1], out_c=c[2], stride=2)
        self.conv4 = _make_layers(block=Bottleneck, num=4, in_c=c[2], out_c=c[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d(1)

    @autocast()
    def forward(self, x):
        b = x.size()[0]
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.pool(x).view(b, -1)
        return x
