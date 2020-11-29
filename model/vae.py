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


class Unet_Free_Seperate(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Seperate, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        b, c, h, w = x.shape

        res = []
        for i in range(c//3):
            x_ = x[:, i*3:(i+1)*3]
            feature, skip1, skip2, skip3 = self.encoder(x_)
            reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
            res.append(reconstructed_image)

        reconstructed_image = torch.cat(res, dim=1)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet_Free_Adversarial(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Adversarial, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        b, c, h, w = x.size()

        res = []
        for i in range(c//3):
            x_ = x[:, i*3:(i+1)*3]
            feature, skip1, skip2, skip3 = self.encoder(x_)
            reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
            res.append(reconstructed_image)

        if train:
            # noise = torch.from_numpy(-1 + 2*np.random.random((c, h, w)), dtype=float).cuda()
            noise = torch.zeros_like(gt[0]).cuda()
            label = label.view(-1)
            gt[label == 1] = noise

        reconstructed_image = torch.cat(res, dim=1)
        pixel_loss = self.criterion(reconstructed_image, gt)
        pixel_loss = torch.abs(pixel_loss)
        pixel_loss = pixel_loss.view(pixel_loss.shape[0], -1).mean(1)

        # anomaly negative loss exp
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class Unet_Free_Adversarial_2decoder(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Adversarial_2decoder, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.decoder2 = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        b, c, h, w = x.size()
        label = label.view(-1)

        if train:
            feature, skip1, skip2, skip3 = self.encoder(x)
            feature_pos = feature[label == 1]
            feature_neg = feature[label == 0]
            skip1_pos = skip1[label == 1]
            skip1_neg = skip1[label == 0]
            skip2_pos = skip2[label == 1]
            skip2_neg = skip2[label == 0]
            skip3_pos = skip3[label == 1]
            skip3_neg = skip3[label == 0]
            gt_pos = gt[label == 1]
            gt_neg = gt[label == 0]

            pixel_loss_neg = 0
            if feature_neg.shape[0] > 0:
                reconstructed_image_neg = self.decoder(feature_neg, skip1_neg, skip2_neg, skip3_neg)
                pixel_loss_neg = self.criterion(reconstructed_image_neg, gt_neg).mean()

            pixel_loss_pos = 0
            if feature_pos.shape[0] > 0:
                reconstructed_image_pos = self.decoder2(feature_pos, skip1_pos, skip2_pos, skip3_pos)
                pixel_loss_pos = self.criterion(reconstructed_image_pos, gt_pos).mean()

            loss = {'pixel_loss': pixel_loss_neg + pixel_loss_pos}
            return gt, loss

        else:
            feature, skip1, skip2, skip3 = self.encoder(x)
            reconstructed_image_neg = self.decoder(feature, skip1, skip2, skip3)
            reconstructed_image_pos = self.decoder2(feature, skip1, skip2, skip3)
            pixel_loss_neg = self.criterion(reconstructed_image_neg, gt)
            pixel_loss_pos = self.criterion(reconstructed_image_pos, gt)
            pixel_loss_neg = pixel_loss_neg.view(pixel_loss_neg.shape[0], -1).mean(1)
            pixel_loss_pos = pixel_loss_pos.view(pixel_loss_pos.shape[0], -1).mean(1)
            loss = {'pixel_loss_neg': pixel_loss_neg, 'pixel_loss_pos': pixel_loss_pos}
            return gt, loss


# mark
class Unet_Free_Adversarial_Classifier(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Adversarial_Classifier, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')
        self.classifier = ShallowNetD_Logit(criterion=FocalLoss(), args=args)
        self.fc = nn.Linear(t_length*256, 1)
        self.sigmoid = nn.Sigmoid()
        nn.init.normal_(self.fc.weight, std=0.01)
        self.criterion = criterion
        self.criterion2 = FocalLoss()
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        b, c, h, w = x.size()
        res = []
        for i in range(c//3):
            x_ = x[:, i*3:(i+1)*3]
            feature, skip1, skip2, skip3 = self.encoder(x_)
            reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
            res.append(reconstructed_image)

        if train:
            # noise = torch.from_numpy(-1 + 2*np.random.random((c, h, w)), dtype=float).cuda()
            noise = torch.zeros_like(gt[0]).cuda()
            label = label.view(-1)
            gt[label == 1] = noise

        reconstructed_image = torch.cat(res, dim=1)
        pixel_loss = self.criterion(reconstructed_image, gt)
        pixel_loss = torch.abs(pixel_loss)
        pixel_loss = pixel_loss.view(pixel_loss.shape[0], -1).mean(1)

        # classifier loss
        input = res
        logits = []
        for i in input:
            logit = self.classifier(i)
            logits.append(logit)

        logits = torch.cat(logits, dim=1)
        logits = self.sigmoid(self.fc(logits))

        if train:
            classifier_loss = self.criterion2(logits, label)
            loss = {'pixel_loss': pixel_loss, 'classifier_loss': classifier_loss}
        else:
            loss = {'pixel_loss': pixel_loss}

        return reconstructed_image, loss, logits


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


class ResUnetAdversarial(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(ResUnetAdversarial, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = Encoder_ResUnet()
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        b, c, h, w = x.size()

        res = []
        for i in range(c//3):
            x_ = x[:, i*3:(i+1)*3]
            reconstructed_image = self.encoder(x_)
            res.append(reconstructed_image)

        if train:
            # noise = torch.from_numpy(-1 + 2*np.random.random((c, h, w)), dtype=float).cuda()
            noise = torch.zeros_like(gt[0]).cuda()
            label = label.view(-1)
            gt[label == 1] = noise

        reconstructed_image = torch.cat(res, dim=1)
        pixel_loss = self.criterion(reconstructed_image, gt)
        pixel_loss = torch.abs(pixel_loss)
        pixel_loss = pixel_loss.view(pixel_loss.shape[0], -1).mean(1)

        # anomaly negative loss exp
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


class ResUnet(nn.Module):
    def __init__(self, criterion=None, args=None, filters=[16, 32, 64, 32]):
        super(ResUnet, self).__init__()

        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        channel = n_channel * t_length
        # channel = 9
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

        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, memory=None,train=True):
        # Encode
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

        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss,
                'memory_loss': memory_loss,
                }
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
