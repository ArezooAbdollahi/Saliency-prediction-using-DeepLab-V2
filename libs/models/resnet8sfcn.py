

# ResNet8s atrous convolution.
# Sequential unroll, parallel unroll and parallel unroll with multirange feedback

from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from resnet import _ConvBatchNormReLU, _ResBlock


    


class ResNet8s(nn.Sequential):
    """ResNet8s+ RIGNet Parallel Unroll with OS=8"""

    def __init__(self, n_classes, n_blocks):
        super(ResNet8s, self).__init__()
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
        self.classifier = nn.Sequential(
            OrderedDict([
                ('conv5_4', _ConvBatchNormReLU(2048, 512, 3, 1, 1, 1)),
                ('drop5_4', nn.Dropout2d(p=0.1)),
                ('conv6', nn.Conv2d(512, n_classes, 1, stride=1, padding=0)),
            ])
        )

    def forward(self, x):
        f1 = self.layer1(x)#64 chanel, st=4
        f2 = self.layer2(f1)#256, st 4
        f3 = self.layer3(f2)#512, st=8
        f4 = self.layer4(f3)#1024, st=8
        f5 = self.layer5(f4)#2048, st=8
        yc = self.classifier(f5)#21, st=8
        yc = F.upsample(yc, x.size()[2:],mode='bilinear')
        return yc

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()


