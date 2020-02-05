
import glob
import os.path as osp
import random
from collections import Counter, defaultdict
from PIL import Image
import cv2
import h5py
import numpy as np
import scipy.io as sio
import scipy.misc as m
import torch
import torchvision
from torch.utils import data
from tqdm import tqdm
import os
from torchvision import transforms, datasets

import torchvision.transforms as standard_transforms
import ipdb
_MEAN = [104.008, 116.669, 122.675]
_STD = [1.0, 1.0, 1.0]

#_MEAN = [0.4914, 0.4822, 0.4465]
#_STD=[0.229, 0.224, 0.225]


class Inpainting(data.Dataset):
    def __init__(self, root = '/home/arezoo/5-DataSet', mode = 'train'):
        self.root = root
        self.mode = mode
        self.mean_std = ([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
        self.images_list = []
        self.labels_list = []
        self.files = []
        self.ignore_label = 255


        # standard_transforms.Normalize(*self.mean_std)

        self.input_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
        # standard_transforms.Normalize(*self.mean_std)
        
    ])
        self.target_transform = standard_transforms.Compose([
        standard_transforms.ToTensor(),
    ])

        # Load all path to images
        if self.mode in ['train','val']:

            with open(os.path.join(root, self.mode + '-MultiScaleModifyCatdeeplab.txt'), 'r') as f:
                # import ipdb; ipdb.set_trace()
                file_list = f.readlines()
                file_list = [x.strip() for x in file_list]
                print("len(file_list)", len(file_list))
        self.files.append(file_list)
        print("len(self.files)",len(self.files))
        self.files=self.files[0]
        #import ipdb; ipdb.set_trace()
        print("len(self.files0)",len(self.files))


    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):
        # import ipdb; ipdb.set_trace()
        image_id = self.files[index]
        image, nameImg, label_128, label_64, label_32, label_16  = self._load_data(image_id)
        image = self.input_transform(image)
        #print(image)
        label_128 = self.target_transform(label_128)
        label_64 = self.target_transform(label_64)
        label_32 = self.target_transform(label_32)
        label_16 = self.target_transform(label_16)
        HeatmapCat=torch.cat((label_128, label_64, label_32, label_16), 0)  #torch.Size([4, 512, 512])

        return image, nameImg, HeatmapCat

    def _load_data(self, image_id):

        # import ipdb; ipdb.set_trace()
        input_id, label_id_128, label_id_64, label_id_32, label_id_16=image_id.split(',')
        
        nameDir= input_id.split('/')[0]
        indexImg= input_id.split('/')[1].split('.')[0]
        nameImg= nameDir+'-'+indexImg
        image = Image.open(os.path.join(self.root,'Saliencyresize-total',input_id)).convert('RGB')

        label_128 = Image.open(os.path.join(self.root,'HeatmapGT',label_id_128)).convert('L')
        label_64 = Image.open(os.path.join(self.root,'HeatmapGT',label_id_64)).convert('L')
        label_32 = Image.open(os.path.join(self.root,'HeatmapGT',label_id_32)).convert('L')
        label_16 = Image.open(os.path.join(self.root,'HeatmapGT',label_id_16)).convert('L')

        return image, nameImg, label_128, label_64, label_32, label_16

    # # import ipdb; ipdb.se_trace()
    # def __init__(self, root, split="train"):
    #     self.root = root
    #     self.split = split
    #     self.mean = np.array(_MEAN)
    #     self.std = np.array(_STD)
    #     self.files = defaultdict(list)
    #     self.images = []
    #     self.labels = []
    #     self.ignore_label = 255
     
    #     print('-----dataset_init-----')
    #     for split in ['train','val']:
    #         file_list = tuple(open( os.path.join( 
    #             root,  split + '-Finetune.txt'), 'r'))
    #         file_list = [id_.rstrip() for id_ in file_list]
    #         self.files[split] = file_list[0:10]

    #     print('self.split: '+self.split+' size: '+str(len(self.files[self.split])))
    #     # print("self.files",self.files)
        
    # def __len__(self):
    #     return len(self.files[self.split])

    # def __getitem__(self, index):
    #         # import ipdb; ipdb.set_trace()

    #         image_id = self.files[self.split][index]
        
    #         image, label, nameofSSIMImage  = self._load_data(image_id) #arezoo
           
    #         image = image.transpose(2, 0, 1) #channel, height, width 
    #         return image.astype(np.float32), label.astype(np.float32), nameofSSIMImage


    # def _normalize(self,  image , mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225) ):        #[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    #     if mean[0] < 1:
    #        image /= 255.0
    #     image-= self.mean
    #     image /= self.std
    #     return image

    # # import ipdb; ipdb.set_trace()
    # def _load_data(self, image_id):
       
    #     gt_id, ssim_id =image_id.split(',')[0], image_id.split(',')[1]
    #     image=cv2.imread( os.path.join("/home/arezoo/5-DataSet/Saliencyresize-total",gt_id), cv2.IMREAD_COLOR).astype(np.float32)
    #     image= self._normalize (image, self.mean, self.std)
    #     ssimd_path=os.path.join("/home/arezoo/5-DataSet/FIXATIONMAPSresize",ssim_id)


       
    #     SSIM_img=cv2.imread( ssimd_path)
    #     SSIM_img=SSIM_img.astype(np.float32)[:,:,0]

    #     return image, SSIM_img, ssim_id #arezoo


   
