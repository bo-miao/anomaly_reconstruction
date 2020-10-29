import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from .backbone import build_backbone

class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation, BatchNorm=nn.BatchNorm2d):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
                                            stride=1, padding=padding, dilation=dilation, bias=False)
        self.bn = eval(BatchNorm)(planes)
        self.relu = nn.ReLU()

        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)

        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, backbone, output_stride, BatchNorm=nn.BatchNorm2d, droupout=False):
        super(ASPP, self).__init__()
        if backbone == 'mobilenetV2':
            inplanes = 320
        else:
            inplanes = 2048
        if output_stride == 16:
            dilations = [1, 6, 12, 18]
        elif output_stride == 8:
            dilations = [1, 12, 24, 36]
        else:
            raise NotImplementedError

        self.aspp1 = _ASPPModule(inplanes, 256, 1, padding=0, dilation=dilations[0], BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes, 256, 3, padding=dilations[1], dilation=dilations[1], BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes, 256, 3, padding=dilations[2], dilation=dilations[2], BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes, 256, 3, padding=dilations[3], dilation=dilations[3], BatchNorm=BatchNorm)
        self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                             nn.Conv2d(inplanes, 256, 1, stride=1, bias=False),
                                             eval(BatchNorm)(256),
                                             nn.ReLU())

        self.conv = nn.Sequential(nn.Conv2d(1280, 256, 1, bias=False),
                                  eval(BatchNorm)(256),
                                  nn.ReLU())
        self.dropout = nn.Dropout(0.5) if droupout else None
        self._init_weight()

    def forward(self, x):
        # multi-scale pooling
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)
        # conv & dropout
        x = self.conv(x)
        if self.dropout:
            x = self.dropout(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            # elif isinstance(m, SynchronizedBatchNorm2d):
            #     m.weight.data.fill_(1)
            #     m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_aspp(backbone, output_stride, BatchNorm):
    return ASPP(backbone, output_stride, BatchNorm)


class Decoder(nn.Module):
    def __init__(self, num_classes, backbone, BatchNorm=nn.BatchNorm2d):
        super(Decoder, self).__init__()
        if backbone == 'resnet':
            low_level_inplanes = 256
        elif backbone == 'mobilenetV2':
            low_level_inplanes = 24 # channel
        else:
            raise NotImplementedError

        self.conv1 = nn.Sequential(nn.Conv2d(low_level_inplanes, 48, 1, bias=False), eval(BatchNorm)(48), nn.ReLU())

        self.conv2 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       eval(BatchNorm)(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.5),
                                       nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       eval(BatchNorm)(256),
                                       nn.ReLU(),
                                       # nn.Dropout(0.1),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1))  # per pixel update
        self._init_weight()

    def forward(self, x, low_level_feat):
        low_level_feat = self.conv1(low_level_feat)
        x = F.interpolate(x, size=low_level_feat.size()[2:], mode='bilinear', align_corners=True)
        x = torch.cat((x, low_level_feat), dim=1)
        x = self.conv2(x)
        return x

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


def build_decoder(num_classes, backbone, BatchNorm):
    return Decoder(num_classes, backbone, BatchNorm)


class DeepLabModule(nn.Module):
    def __init__(self, backbone='mobilenetV2', output_stride=16, image_channel=3,
                 out_channel=3, BatchNorm = nn.BatchNorm2d, restore_resolution=False):
        super(DeepLabModule, self).__init__()

        self.backbone = build_backbone(backbone, output_stride, BatchNorm, image_channel)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(out_channel, backbone, BatchNorm)
        self.restore_resolution = restore_resolution

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        # restore resolution from 1/4 to 1
        if self.restore_resolution:
            x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x


class deeplab(torch.nn.Module):
    def __init__(self, criterion=None, args=None):
        super(deeplab, self).__init__()
        in_channel, out_channel = 12, 3
        BatchNorm = args.bn
        self.model = DeepLabModule(backbone=args.encoder_arch, output_stride=16, image_channel=in_channel,
                                   out_channel=out_channel, BatchNorm=BatchNorm, restore_resolution=args.restore_resolution)
        self.criterion = criterion
        self.args = args

    @autocast()
    def forward(self, x, gt=None, train=True):
        reconstructed_image = self.model(x)
        if not self.args.restore_resolution:
            _, _, h, w = reconstructed_image.size()
            gt = F.interpolate(gt, size=(h, w), mode='bilinear', align_corners=True)
        pixel_loss = self.criterion(reconstructed_image, gt)
        loss = {'pixel_loss': pixel_loss,
                'compactness_loss': 0,
                'separateness_loss': 0
                }
        return reconstructed_image, loss


if __name__ == "__main__":
    model = DeepLabModule(backbone='mobilenetV2', output_stride=16, image_channel=3)
    model.eval()
    input = torch.rand(1, 3, 256, 256)
    output = model(input)
    print(output.size())


