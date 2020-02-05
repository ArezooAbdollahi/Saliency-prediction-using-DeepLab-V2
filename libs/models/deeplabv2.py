#!/usr/bin/env python
# coding: utf-8
#
# Author:   Kazuto Nakashima
# URL:      http://kazuto1011.github.io
# Created:  2017-11-19

from collections import OrderedDict
import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as F

from resnet import _ConvBatchNormReLU, _ResBlock


class _ASPPModule(nn.Module):
    """Atrous Spatial Pyramid Pooling"""

    def __init__(self, in_channels, out_channels, pyramids):
        super(_ASPPModule, self).__init__()
        self.stages = nn.Module()
        for i, (dilation, padding) in enumerate(zip(pyramids, pyramids)):
            self.stages.add_module(
                'c{}'.format(i),
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=3,
                    stride=1,
                    padding=padding,
                    dilation=dilation,
                    bias=True
                )
            )

        for m in self.stages.children():
            nn.init.normal(m.weight, mean=0, std=0.01)
            nn.init.constant(m.bias, 0)



    def forward(self, x):
        h = 0
        for stage in self.stages.children():
            h += stage(x)
        return h



class Interpolate(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self,scale_factor = 2,):
        super(Interpolate, self).__init__()
        self.interp = F.upsample
        self.scale_factor = scale_factor

    def forward(self,x):
        return self.interp(x,scale_factor = self.scale_factor,mode = "bilinear")

class DeepLabV2(nn.Sequential):
    """DeepLab v2"""

    def __init__(self, n_classes, n_blocks, pyramids):
        super(DeepLabV2, self).__init__()
        self.add_module(
            'layer1',
            nn.Sequential(
                OrderedDict([
                    ('conv1', _ConvBatchNormReLU(3, 64, 7, 2, 3, 1)),
                    ('pool', nn.MaxPool2d(3, 2, 1, ceil_mode=True)),
                ])
            )
        )
        self.add_module('layer2', _ResBlock(n_blocks[0], 64, 64, 256, 1, 1))
        self.add_module('layer3', _ResBlock(n_blocks[1], 256, 128, 512, 2, 1))
        self.add_module('layer4', _ResBlock(n_blocks[2], 512, 256, 1024, 1, 2))
        self.add_module('layer5', _ResBlock(n_blocks[3], 1024, 512, 2048, 1, 4))
        # self.add_module('aspp', _ASPPModule(2048, n_classes, pyramids)) Arezoo
        self.add_module('aspp', _ASPPModule(2048, 4, pyramids))



        self.upsample_1 = nn.Sequential(
            nn.Conv2d(4,4,kernel_size = 1,stride = 1,padding = 0),
            nn.BatchNorm2d(4),
            Interpolate(),
            )

        self.weight_init()

        


    def weight_init(self):

        # for n, m in self.layer4.named_modules():
        #   if 'conv2' in n:
        #       print("weight.max = ", m.weight.data.max(), "n = ", n)

        # print('====' * 50)

        for m,k in self.upsample_1.named_modules():
            # import ipdb; ipdb.set_trace()
            if isinstance(k, nn.Conv2d):
                nn.init.kaiming_normal_(k.weight.data)

        # import ipdb; ipdb.set_trace()

    def forward(self, x):
        # import ipdb; ipdb.set_trace()
        y= super(DeepLabV2, self).forward(x)
        y = self.upsample_1(y)
        
        # y = F.sigmoid(self.upsample_1(y))
        
        return y

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


if __name__ == '__main__':
    model = DeepLabV2(n_classes=21, n_blocks=[3, 4, 23, 3], pyramids=[6, 12, 18, 24])
    model.freeze_bn()
    model.eval()
    print list(model.named_children())
    image = torch.autograd.Variable(torch.randn(1, 3, 513, 513))
    print model(image)[0].size()
