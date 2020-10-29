from . import mobilenet
from model.backbone import resnet


def build_backbone(backbone, output_stride, BatchNorm, image_channel):
    if backbone == 'resnet':
        return resnet.ResNet101(image_channel, output_stride, BatchNorm)
    elif backbone == 'mobilenetV2':
        return mobilenet.MobileNetV2(image_channel, output_stride, BatchNorm)
    else:
        raise NotImplementedError
