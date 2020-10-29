import numpy as np
import os, sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.nn import Module, Parameter
import torchvision.models as models

# personal pkg
# from .norm import *
from .block import *
from .discriminator import *


class Unet_3d(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_3d, self).__init__()
        n_channel = args.c
        self.encoder = eval(args.encoder_arch)(n_channel, block='Basic_Encoder3d')
        self.decoder = eval(args.decoder_arch)(n_channel, dim=256, block='Basic_Decoder3d')
        self.activation = torch.nn.Tanh()
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        feature, skip1, skip2, skip3 = self.encoder(x)
        feature = self.decoder(feature, skip1, skip2, skip3)
        reconstructed_image = self.activation(feature)
        reconstructed_image = reconstructed_image.squeeze(2)
        gt = gt.squeeze(2)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet_Light4(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Light4, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet_Light3(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Light3, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):

        feature, skip1, skip2 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet_Free(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet_Free_Classifier(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Classifier, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args
        self.classifier = ShallowNetD(criterion=FocalWithLogitsLoss(), args=args)

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}

        # generate samples
        b, c, h, w = gt.size()
        neg_image = reconstructed_image - gt.detach()
        neg_label = torch.zeros([b]).cuda()
        if train:
            pos_image = reconstructed_image - x[:, :c].detach()
            pos_label = torch.ones([b]).cuda()
            in_image = torch.cat((pos_image, neg_image), dim=0)
            in_label = torch.cat((pos_label, neg_label))
            discriminative_loss, logit = self.classifier(in_image, in_label, train=True)
            loss.update({'discriminative_loss': 0.1*discriminative_loss})
        else:
            in_image, in_label = neg_image, neg_label
            logit = self.classifier(in_image, in_label, train=False)

        return reconstructed_image, loss, logit


class Unet_Free_Dual(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Dual, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class SeparateUnet(nn.Module):
    def __init__(self, criterion=None, args=None):
        super(SeparateUnet, self).__init__()
        n_channel = 3
        self.encoder1 = eval(args.encoder_arch)(2, n_channel)
        self.encoder2 = eval(args.encoder_arch)(2, n_channel)
        self.decoder = eval(args.decoder_arch)(2, n_channel)
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        feature, skip1, skip2 = self.encoder1(x[:, :3])
        feature_, skip1_, skip2_ = self.encoder2(x[:, 3:])
        feature = torch.cat((feature, feature_), dim=1)
        skip1 = torch.cat((skip1, skip1_), dim=1)
        skip2 = torch.cat((skip2, skip2_), dim=1)
        reconstructed_image = self.decoder(feature, skip1, skip2)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel)
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, dim=512)
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


########################## EXPERIMENTS #############################################
class Decoder_optical_flow(torch.nn.Module):
    def __init__(self, t_length=5, n_channel=3, dim=1024, block='Basic'):
        super(Decoder_optical_flow, self).__init__()
        def Gen(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc//4, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc//4),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc//4, out_channels=intOutput, kernel_size=1, stride=1, padding=0),
                torch.nn.Tanh()
            )

        def Gen_OF(intInput, intOutput, nc):
            return torch.nn.Sequential(
                torch.nn.Conv2d(in_channels=intInput, out_channels=nc, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc, out_channels=nc//4, kernel_size=3, stride=1, padding=1),
                torch.nn.BatchNorm2d(nc//4),
                torch.nn.ReLU(inplace=False),
                torch.nn.Conv2d(in_channels=nc//4, out_channels=intOutput, kernel_size=1, stride=1, padding=0),
                # torch.nn.Sigmoid()
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
        self.opticalFlow = Gen_OF(128, n_channel-1, 64)

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
        optical_flow = self.opticalFlow(cat2)

        return output, optical_flow


#### Attention between input frame and
def msra(module: nn.Module) -> None:
    nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    if module.bias is not None:  # pyre-ignore
        nn.init.constant_(module.bias, 0)


class Unet_optical_flow(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_optical_flow, self).__init__()
        n_channel = args.c
        t_length = args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel)
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, dim=512, block='Residual')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, optical_flow=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image, reconstructed_optical_flow = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        optical_flow_loss = self.criterion(reconstructed_optical_flow, optical_flow)
        loss = {'pixel_loss': pixel_loss,
                'optical_flow_loss': self.args.loss_opticalflow * optical_flow_loss}
        return reconstructed_image, loss


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"]="4"
    import time

    d = torch.rand(2, 10, 8, 6, 6).cuda()

    a1 = DepthwiseConv3d(10, 10, (3, 3, 3), groups=10).cuda()
    a2 = nn.Conv3d(10, 10, (3, 3, 3)).cuda()

    t1 = time.time()
    for i in range(10000):
        d1 = a1(d)
    print("1cost ", time.time()-t1)

    t1 = time.time()
    for i in range(10000):
        d2 = a2(d)
    print("2cost ", time.time()-t1)

    print(d1.shape, d2.shape)


    '''
    x = torch.ones(3,12,256,256).cuda()

    y = torch.ones(10,512).cuda()

    z = torch.ones(10,512).cuda()

    g = torch.ones(3,3,256,256).cuda()

    _, _, _, loss = a(x,y,z,g)
    print(loss)
    '''
