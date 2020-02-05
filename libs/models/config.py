import os
import os.path


pretrained_dir='/media/karimr/hddc/semantic_segmentation/pretrained'

vgg16_caffe_path= os.path.join(pretrained_dir,'imagenet_caffemodel', 'vgg16-caffe.pth')
ifrnet_vgg16_726_path= os.path.join(pretrained_dir,'voc11_ifrnet','ifrnet_vgg_voc11_miou7257.pth')
ifrnet_nf_vgg16_voc11_6514_path= os.path.join(pretrained_dir,'voc11_ifrnet','ifrnet_nf_vgg_voc11_miou6514.pth')


resnet50__caffe_statedict_path=os.path.join(pretrained_dir,'imagenet_caffemodel','resnet50_caffe_bgr_0_255.pth')
resnet101_caffe_statedict_path=os.path.join(pretrained_dir,'imagenet_caffemodel','resnet101_caffe_bgr_0_255.pth')




