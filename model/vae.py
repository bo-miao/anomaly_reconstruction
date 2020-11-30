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


# UNSUPERVISED
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


# UNSUPERVISED
class Unet(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel) # Encoder
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, dim=512) # Decoder
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


# SUPERVISED
class Unet_Free_Supervised(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Supervised, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic', in_c=n_channel*t_length) # Encoder_Free
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic') # Decoder_Free
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, label=None, train=True):
        if train:
            # noise = torch.from_numpy(-1 + 2*np.random.random((c, h, w)), dtype=float).cuda()
            noise = torch.zeros_like(gt[0]).cuda()
            label = label.view(-1)
            gt[label == 1] = noise

        feature, skip1, skip2, skip3 = self.encoder(x)
        reconstructed_image = self.decoder(feature, skip1, skip2, skip3)
        pixel_loss = self.criterion(reconstructed_image, gt[:,3:6])
        pixel_loss = pixel_loss.view(pixel_loss.shape[0], -1).mean(1)
        loss = {'pixel_loss': pixel_loss}
        return reconstructed_image, loss


# SUPERVISED
class Unet_Free_Adversarial(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Adversarial, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic') # Encoder_Free
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic') # Decoder_Free
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


# SUPERVISED
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


# SUPERVISED (WORSE THAN OTHERS)
class Unet_Free_Adversarial_Classifier(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(Unet_Free_Adversarial_Classifier, self).__init__()
        n_channel = args.c
        t_length = args.t_length // args.interval + 1 if args.interval > 1 else args.t_length
        self.encoder = eval(args.encoder_arch)(t_length, n_channel, block='Basic')  # Encoder_Free
        self.decoder = eval(args.decoder_arch)(t_length, n_channel, block='Basic')  # Decoder_Free
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
