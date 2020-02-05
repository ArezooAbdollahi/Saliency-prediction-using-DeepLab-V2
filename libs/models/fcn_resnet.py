import numpy as np
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models as tvm
import config 


resnet50__caffe_path=config.resnet50__caffe_statedict_path 
resnet101_caffe_path=config.resnet101_caffe_statedict_path 



class FCNResNet(nn.Module):
    #Vanilla FCN ResNet, OS=32
    def __init__(self, _num_classes, _resnet_name='resnet101', _pretrained=True):
        super(FCNResNet, self).__init__()
        print(_resnet_name)
        resnet=tvm.resnet50() if _resnet_name=='resnet50' else tvm.resnet101()
        if _pretrained==True:
            caffemodel=torch.load(resnet50__caffe_path )  if _resnet_name=='resnet50' else  torch.load(resnet101_caffe_path )
            resnet.load_state_dict(caffemodel)
            print('loaded imagenet caffeemodel')
        self.layer0=nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool )
        self.layer1=resnet.layer1
        self.layer2=resnet.layer2
        self.layer3=resnet.layer3
        self.layer4=resnet.layer4
        self.classifier=nn.Sequential(
                #nn.AvgPool2d(kernel_size=7, stride=7, padding=0, ceil_mode=False, count_include_pad=True),
                nn.Conv2d(
                    in_channels=2048,
                    out_channels=_num_classes,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=True
                )
           )
        for m in self.classifier.children():
            nn.init.normal(m.weight, mean=0, std=0.01)
            nn.init.constant(m.bias, 0)
        return

    def forward(self, x):
        f1=self.layer0(x)
        f2=self.layer1(f1)
        f3=self.layer2(f2)
        f4=self.layer3(f3)
        f5=self.layer4(f4)
        f6=self.classifier(f5)
        out = f6
        outup=F.upsample(out, x.size()[2:],mode='bilinear')
        return outup

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
