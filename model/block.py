import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, Parameter


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


###################################################
class Encoder_Free(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Encoder_Free, self).__init__()

        # in_c = n_channel * (t_length - 1)
        in_c = 3
        c = [16, 32, 16]
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=c[0], kernel_size=3, stride=1, padding=1),
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

        in_c = 16
        c = [16, 32, 16, 16, 3]
        # 8,16 64,16 32,32 16,16
        self.conv1 = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=in_c, out_channels=c[0], kernel_size=3, stride=1, padding=1),
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

        x = torch.cat((x, skip3), dim=1)  # 16->32
        x = self.conv2(x)
        x = self.upsample(x)

        x = torch.cat((x, skip2), dim=1)  # 32->64
        x = self.conv3(x)
        x = self.upsample(x)

        x = torch.cat((x, skip1), dim=1)  # 16->32
        x = self.conv4(x)  # 32->16->3

        return x
###################################################


###################################################
class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)


class Encoder_ResUnet(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, block='Basic'):
        super(Encoder_ResUnet, self).__init__()

        filters = [16, 32, 64, 32]
        channel = 3
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        self.residual_conv_1 = ResidualConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConv(filters[1], filters[2], 2, 1)

        self.bridge = ResidualConv(filters[2], filters[3], 2, 1)
        # HERE TO ADD MEM [3]
        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ResidualConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ResidualConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ResidualConv(filters[1] + filters[0], filters[0], 1, 1)

        self.output_layer = nn.Sequential(
            nn.Conv2d(filters[0], 3, 1, 1),
            nn.Tanh(),
        )

    @autocast()
    def forward(self, x):
        x1 = self.input_layer(x) + self.input_skip(x)
        x2 = self.residual_conv_1(x1)
        x3 = self.residual_conv_2(x2)
        # Bridge         # Decode HERE TO ADD MEMORY 1/8 resolution
        x4 = self.bridge(x3)
        x4 = self.upsample_1(x4)
        x5 = torch.cat([x4, x3], dim=1)

        x6 = self.up_residual_conv1(x5)

        x6 = self.upsample_2(x6)
        x7 = torch.cat([x6, x2], dim=1)

        x8 = self.up_residual_conv2(x7)

        x8 = self.upsample_3(x8)
        x9 = torch.cat([x8, x1], dim=1)

        x10 = self.up_residual_conv3(x9)

        reconstructed_image = self.output_layer(x10)

        return reconstructed_image
################################################################


if __name__ == "__main__":
    m = Encoder_ResUnet()
    a = torch.rand(2,3,256,256)
    print(m(a).shape)
