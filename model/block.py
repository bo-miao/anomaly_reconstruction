import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, Parameter
# personal pkg
# from .norm import *
from .light_resnet import *


# class Classifier(nn.Module):
#     def __init__(self, in_c, class_num):
#         super(Classifier, self).__init__()
#         c = [256, 512]
#         self.conv1 = Bottleneck(c[0], c[0])
#         self.conv2 = Bottleneck(c[0], c[0])
#         self.pool = torch.nn.AdaptiveAvgPool1d(output_size)
#         self.fc = nn.Linear(in_dim*one_hot_cls_num, 2048)


class DepthwiseConv3d(nn.Module):
    # 3.8s/10K, Conv3d 1.15s/10K
    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        super(DepthwiseConv3d, self).__init__()
        self.depth_conv = nn.Conv3d(in_channels, in_channels, kernel_size,
                                    stride=stride, padding=padding, dilation=dilation, groups=groups,
                                    bias=bias, padding_mode=padding_mode)
        self.point_conv = nn.Conv3d(in_channels, out_channels, (1, 1, 1),
                                    stride=stride, padding=padding, dilation=dilation, groups=1,
                                    bias=bias, padding_mode=padding_mode)

    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        return x


class Basic_Encoder3d(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Basic_Encoder3d, self).__init__()
        out1 = out_channel if out_channel // in_channel <= 2 else out_channel // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=in_channel, out_channels=out1, kernel_size=(3, 3, 3), stride=1, padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(out1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv3d(in_channels=out1, out_channels=out_channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)


class Basic_Decoder3d(nn.Module):
    def __init__(self, in_channel, out_channel, frame):
        super(Basic_Decoder3d, self).__init__()
        out1 = out_channel if in_channel // out_channel <= 2 else in_channel // 2
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=in_channel, out_channels=out1, kernel_size=(frame, 3, 3), stride=1, padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(out1),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv3d(in_channels=out1, out_channels=out_channel, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)


class Connect3d(nn.Module):
    def __init__(self, in_channel, out_channel, in_frame):
        super(Connect3d, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=in_channel, out_channels=out_channel, kernel_size=(in_frame, 1, 1), stride=1, padding=0),
            torch.nn.BatchNorm3d(out_channel),
            torch.nn.ReLU(inplace=False),
        )

    def forward(self, x):
        return self.block(x)


class Residual(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Residual, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=False)
        )
        self.residual = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
        )
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        residual = self.residual(x)
        y = self.block(x)
        return self.relu(residual + y)


class Basic(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Basic, self).__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=False)
        )

    def forward(self, x):
        return self.block(x)


def msra(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


######### Basic ######################################
class Encoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Encoder, self).__init__()

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        self.moduleConv1 = eval(block)(n_channel * (t_length - 1), 64)
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = eval(block)(64, 128)
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = eval(block)(128, 256)
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(256, 512)
        self.moduleBatchNorm = torch.nn.BatchNorm2d(512)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    @autocast()
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Decoder(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, dim=1024, block='Basic'):
        super(Decoder, self).__init__()

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = eval(block)(dim, 512)
        self.moduleUpsample4 = Upsample(512, 256)

        self.moduleDeconv3 = eval(block)(512, 256)
        self.moduleUpsample3 = Upsample(256, 128)

        self.moduleDeconv2 = eval(block)(256, 128)
        self.moduleUpsample2 = Upsample(128, 64)

        self.moduleDeconv1 = Gen(128, n_channel, 64)

    @autocast()
    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output
###################################################


########## 3D #########################
class Encoder_3d(torch.nn.Module):
    def __init__(self, n_channel=3, block='Basic'):
        super(Encoder_3d, self).__init__()

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=intInput, out_channels=intOutput, kernel_size=(2, 3, 3), stride=1, padding=(0, 1, 1)),
                torch.nn.BatchNorm3d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv3d(in_channels=intOutput, out_channels=intOutput, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            )

        self.moduleConv1 = eval(block)(n_channel, 32)
        self.modulePool1 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.moduleConv2 = eval(block)(32, 64)
        self.modulePool2 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.moduleConv3 = eval(block)(64, 128)
        self.modulePool3 = torch.nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))

        self.moduleConv4 = Basic_(128, 256)
        self.moduleBatchNorm = torch.nn.BatchNorm3d(256)
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    @autocast()
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3
        # 256, 1;  128, 2; 64, 4; 32, 6; in, 8;


class Decoder_3d(torch.nn.Module):
    def __init__(self, n_channel=3, dim=256, block='Basic'):
        super(Decoder_3d, self).__init__()

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv3d(in_channels=intInput, out_channels=nc, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                torch.nn.BatchNorm3d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv3d(in_channels=nc, out_channels=nc, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
                torch.nn.BatchNorm3d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv3d(in_channels=nc, out_channels=intOutput, kernel_size=(1, 3, 3), stride=1, padding=(0, 1, 1)),
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose3d(in_channels=nc, out_channels=intOutput, kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1),
                                         output_padding=(0, 1, 1)),
                torch.nn.BatchNorm3d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        self.moduleConv = eval(block)(dim, 256, 1)
        self.moduleUpsample4 = Upsample(256, 128)

        self.line3 = Connect3d(128, 128, 2)
        self.moduleDeconv3 = eval(block)(256, 128, 1)
        self.moduleUpsample3 = Upsample(128, 64)

        self.line2 = Connect3d(64, 64, 4)
        self.moduleDeconv2 = eval(block)(128, 64, 1)
        self.moduleUpsample2 = Upsample(64, 32)

        self.line1 = Connect3d(32, 32, 6)
        self.moduleDeconv1 = Gen(64, n_channel, 32)

    @autocast()
    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)

        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((self.line3(skip3), tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((self.line2(skip2), tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((self.line1(skip1), tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output

##############################################


###############Light ################################
class Encoder_Light4(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Encoder_Light4, self).__init__()

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        c = [32, 64, 128, 256]

        self.moduleConv1 = eval(block)(n_channel * (t_length - 1), c[0])
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = eval(block)(c[0], c[1])
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = eval(block)(c[1], c[2])
        self.modulePool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv4 = Basic_(c[2], c[3])
        self.moduleBatchNorm = torch.nn.BatchNorm2d(c[3])
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    @autocast()
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)
        tensorPool3 = self.modulePool3(tensorConv3)

        tensorConv4 = self.moduleConv4(tensorPool3)

        return tensorConv4, tensorConv1, tensorConv2, tensorConv3


class Decoder_Light4(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Decoder_Light4, self).__init__()

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        c = [256, 128, 64, 32]

        self.moduleConv = eval(block)(c[0], c[0])
        self.moduleUpsample4 = Upsample(c[0], c[1])

        self.moduleDeconv3 = eval(block)(c[0], c[1])
        self.moduleUpsample3 = Upsample(c[1], c[2])

        self.moduleDeconv2 = eval(block)(c[1], c[2])
        self.moduleUpsample2 = Upsample(c[2], c[3])

        self.moduleDeconv1 = Gen(c[2], n_channel, c[3])

    @autocast()
    def forward(self, x, skip1, skip2, skip3):
        tensorConv = self.moduleConv(x)
        tensorUpsample4 = self.moduleUpsample4(tensorConv)
        cat4 = torch.cat((skip3, tensorUpsample4), dim=1)

        tensorDeconv3 = self.moduleDeconv3(cat4)
        tensorUpsample3 = self.moduleUpsample3(tensorDeconv3)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output


class Encoder_Light3(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Encoder_Light3, self).__init__()

        def Basic_(intInput, intOutput):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=intOutput, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
            )

        c = [32, 64, 128]
        out_channel = c[0]
        self.moduleConv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=n_channel * (t_length - 1), out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=out_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channel),
            torch.nn.ReLU(inplace=False)
        )
        self.modulePool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv2 = eval(block)(c[0], c[1])
        self.modulePool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

        self.moduleConv3 = Basic_(c[1], c[2])
        self.moduleBatchNorm = torch.nn.BatchNorm2d(c[2])
        self.moduleReLU = torch.nn.ReLU(inplace=False)

    @autocast()
    def forward(self, x):
        tensorConv1 = self.moduleConv1(x)
        tensorPool1 = self.modulePool1(tensorConv1)

        tensorConv2 = self.moduleConv2(tensorPool1)
        tensorPool2 = self.modulePool2(tensorConv2)

        tensorConv3 = self.moduleConv3(tensorPool2)

        return tensorConv3, tensorConv1, tensorConv2  # 128, 32, 64


class Decoder_Light3(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Decoder_Light3, self).__init__()

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        c = [128, 64, 32]
        self.moduleConv = eval(block)(c[0], c[0])
        self.moduleUpsample3 = Upsample(c[0], c[1])

        self.moduleDeconv2 = eval(block)(c[0], c[1])
        self.moduleUpsample2 = Upsample(c[1], c[2])

        self.moduleDeconv1 = Gen(c[1], n_channel, c[2])

    @autocast()
    def forward(self, x, skip1, skip2):
        tensorConv = self.moduleConv(x)

        tensorUpsample3 = self.moduleUpsample3(tensorConv)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output


class Encoder_Free(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Encoder_Free, self).__init__()

        c = [16, 32, 16]
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=n_channel * (t_length - 1), out_channels=c[0], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[0]),
            torch.nn.ReLU(inplace=False)
        )
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[0], out_channels=c[1], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[1]),
            torch.nn.ReLU(inplace=False)
        )
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[1], out_channels=c[2], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[2]),
            torch.nn.ReLU(inplace=False)
        )
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    @autocast()
    def forward(self, x):
        feature1 = self.conv1(x)
        feature1_ = self.pool(feature1)

        feature2 = self.conv2(feature1_)
        feature2_ = self.pool(feature2)

        feature3 = self.conv3(feature2_)
        feature3_ = self.pool(feature3)

        return feature3_, feature1, feature2, feature3  # 8,16 64,16 32,32 16,16


class Decoder_Free(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Decoder_Free, self).__init__()

        # def Upsample(nc, intOutput):
        #     return torch.nn.Sequential(
        #         torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
        #                                  output_padding=1),
        #         torch.nn.BatchNorm2d(intOutput),
        #         torch.nn.ReLU(inplace=False)
        #     )

        c = [16, 32, 16, 16, 3]
        # 8,16 64,16 32,32 16,16
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[0], out_channels=c[0], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[0]),
            torch.nn.ReLU(inplace=False)
        )  # 16,32
        self.conv2 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[0]*2, out_channels=c[1], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[1]),
            torch.nn.ReLU(inplace=False)
        )  # 32,64
        self.conv3 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[1]*2, out_channels=c[2], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[2]),
            torch.nn.ReLU(inplace=False)
        )  # 64,32
        self.conv4 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=c[2]*2, out_channels=c[3], kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(c[3]),
            torch.nn.ReLU(inplace=False),
            torch.nn.Conv2d(in_channels=c[3], out_channels=c[4], kernel_size=3, stride=1, padding=1),
            torch.nn.Tanh()
        )  # 64,3

        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)

    @autocast()
    def forward(self, x, skip1, skip2, skip3):
        x = self.conv1(x)
        x = self.upsample(x)

        x = torch.cat((x, skip3), dim=1)
        x = self.conv2(x)
        x = self.upsample(x)

        x = torch.cat((x, skip2), dim=1)
        x = self.conv3(x)
        x = self.upsample(x)

        x = torch.cat((x, skip1), dim=1)
        x = self.conv4(x)

        return x


class Decoder_Norm3(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Decoder_Norm3, self).__init__()

        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=1, padding=1),
                torch.nn.Tanh()
            )

        def Upsample(nc, intOutput):
            return torch.nn.Sequential(
                torch.nn.ConvTranspose2d(in_channels=nc, out_channels=intOutput, kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                torch.nn.BatchNorm2d(intOutput),
                torch.nn.ReLU(inplace=False)
            )

        c = [256, 128, 64]
        self.moduleConv = eval(block)(c[0], c[0])
        self.moduleUpsample3 = Upsample(c[0], c[1])

        self.moduleDeconv2 = eval(block)(c[0], c[1])
        self.moduleUpsample2 = Upsample(c[1], c[2])

        self.moduleDeconv1 = Gen(c[1], n_channel, c[2])

    @autocast()
    def forward(self, x, skip1, skip2):
        tensorConv = self.moduleConv(x)

        tensorUpsample3 = self.moduleUpsample3(tensorConv)
        cat3 = torch.cat((skip2, tensorUpsample3), dim=1)

        tensorDeconv2 = self.moduleDeconv2(cat3)
        tensorUpsample2 = self.moduleUpsample2(tensorDeconv2)
        cat2 = torch.cat((skip1, tensorUpsample2), dim=1)

        output = self.moduleDeconv1(cat2)

        return output
###############################################