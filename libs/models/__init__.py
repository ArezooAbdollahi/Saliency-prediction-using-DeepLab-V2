

from libs.models.fcn_resnet import *


from libs.models.resnet import *
from libs.models.resnet8sfcn import *

from libs.models.deeplabv2 import *
#from libs.models.deeplabv2_ifrnet import *


#------------------ResNet50-------------------#
def FCNResNet50(num_classes):
    model= FCNResNet(_num_classes=num_classes, _resnet_name='resnet50')
    return model

#------------------ResNet101_32s-------------------#
def FCNResNet101(num_classes):
    model=  FCNResNet(_num_classes=num_classes, _resnet_name='resnet101')
    return model 


#------------------ResNet101_8s-------------------#

def FCNResNet101_8s(num_classes):
    model=  ResNet8s(n_classes=num_classes, n_blocks=[3,4,23,3])
    init_weights(model)
    return model 


#------------------DeepLabV2-------------------#
def DeepLabV2ResNet101(num_classes):
    model=DeepLabV2(n_classes=num_classes,n_blocks=[3,4,23,3],pyramids=[6,12,18,24])
    return model


def load_resnet101_coco_init_msc_to_nomsc(model, state_dict):
    print('modifying keys for msc to nomsc ... ')
    for k,v in state_dict.items():
        if k[:6] == 'Scale.' or k[:6] == 'scale.':
            kk=k[6:]
            #print(' updating '+str(k)+' as '+str(kk))
            state_dict[kk]=v
            del state_dict[k]
    #key update from msc to no msc done
    #now load it to no msc
    model.load_state_dict(state_dict, strict=False)  # Skip "aspp" layer
    smk=set([n for n,m in model.named_parameters() ])
    sdk=set([ k for k,v in state_dict.items() ])
    ins=list( smk & sdk )
    print('num of keys in model state dict: '  +str(len(smk)))
    print('num of keys in init state dict: '  +str(len(sdk)))
    print('num of keys common in both state dict: '  +str(len(ins)))
    print('coco init loaded ...')
    return model

def init_weights(model):
    print('initializing model parameters ...')
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_normal(m.weight)
            if m.bias is not None:
                nn.init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant(m.weight, 1)
            if m.bias is not None:
                nn.init.constant(m.weight, 1)
    print('initializing model parameters done.')


