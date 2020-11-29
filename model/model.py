import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.cuda.amp import autocast

print("PyTorch Version: ",torch.__version__)
print("Torchvision Version: ",torchvision.__version__)

__all__ = ['ResNet50', 'ResNet101','ResNet152']

def Conv1(in_planes, places, stride=2):
    return nn.Sequential(
        nn.Conv2d(in_channels=in_planes,out_channels=places,kernel_size=7,stride=stride,padding=3, bias=False),
        nn.BatchNorm2d(places),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
    )

class Bottleneck(nn.Module):
    def __init__(self,in_places,places, stride=1,downsampling=False, expansion = 4):
        super(Bottleneck,self).__init__()
        self.expansion = expansion
        self.downsampling = downsampling

        self.bottleneck = nn.Sequential(
            nn.Conv2d(in_channels=in_places,out_channels=places,kernel_size=1,stride=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(places),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=places, out_channels=places*self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(places*self.expansion),
        )

        if self.downsampling:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels=in_places, out_channels=places*self.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(places*self.expansion)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        residual = x
        out = self.bottleneck(x)

        if self.downsampling:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        return out


# Arson Explosion Fall Fight Normal
class ResNet(nn.Module):
    def __init__(self,blocks, num_classes=5, expansion = 4):
        super(ResNet,self).__init__()
        self.expansion = expansion

        c = [3, 64, 128, 256, 512, 1024]
        self.conv1 = Conv1(in_planes=c[0], places=c[1])
        self.conv2 = nn.Sequential(
                nn.Conv2d(in_channels=c[1]*3, out_channels=c[1], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=c[1], out_channels=c[1], kernel_size=3, stride=1, padding=1, bias=False),
                nn.BatchNorm2d(c[1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(in_channels=c[1], out_channels=c[1], kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(c[1]),
            )
        self.layer1 = self.make_layer(in_places=c[1], places=c[1], block=blocks[0], stride=1)
        self.layer2 = self.make_layer(in_places=c[3],places=c[2], block=blocks[1], stride=2)
        self.layer3 = self.make_layer(in_places=c[4],places=c[3], block=blocks[2], stride=2)
        self.layer4 = self.make_layer(in_places=c[5],places=c[4], block=blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        #self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(2048, num_classes)
        nn.init.normal_(self.fc.weight, std=0.01)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def make_layer(self, in_places, places, block, stride):
        layers = []
        layers.append(Bottleneck(in_places, places,stride, downsampling =True))
        for i in range(1, block):
            layers.append(Bottleneck(places*self.expansion, places))

        return nn.Sequential(*layers)

    @autocast()
    def forward(self, x):
        b, c, h, w = x.shape
        start_ = 0
        o1 = []
        for i in range(c//3):
            o1.append(self.conv1(x[:, start_*3: (start_+1)*3]))
            start_ += 1
        o1 = torch.cat(o1, dim=1)

        x = self.conv2(o1)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

def ResNet50():
    return ResNet([3, 4, 6, 3])

def ResNet101():
    return ResNet([3, 4, 23, 3])

def ResNet152():
    return ResNet([3, 8, 36, 3])


if __name__=='__main__':
    #model = torchvision.models.resnet50()
    model = ResNet50()
    a = torch.rand(2,9,256,256)
    print(model(a).shape)
