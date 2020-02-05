

from __future__ import absolute_import, division, print_function
import random
import os.path 
import os
import cv2
import numpy as np
import csv
from PIL import Image
import json
import datetime

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.autograd import Variable
import torchvision.transforms as standard_transforms
from tensorboardX import SummaryWriter
from torch.backends import cudnn
import torchvision.utils
from libs import models
#from libs.datasets import voc as voc
from libs.utils.loss import CrossEntropyLoss2d
from libs.utils.save_voc import *
from libs.datasets import inpainting as inpainting

import torch.nn.init as init
from weight_initializer import Initializer
from tqdm import tqdm
import ipdb


from utils_saliency.salgan_utils import load_image, postprocess_prediction
from utils_saliency.salgan_utils import normalize_map


from IPython import embed
from evaluation.metrics_functions import AUC_Judd, AUC_Borji, AUC_shuffled, CC, NSS, SIM, EMD

cudnn.benchmark = True
manual_seed=627937
torch.backends.cudnn.deterministic = True
torch.cuda.manual_seed_all(manual_seed)
torch.cuda.manual_seed(manual_seed)
torch.manual_seed(manual_seed)
np.random.seed(manual_seed)
random.seed(manual_seed)


def load_coco_init_msc_to_nomsc(model, state_dict):
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

def get_lr_params(model, key):
	# For Dilated FCN
	print('get_lr_params key: '+str(key))
	if key == '1x':
		for n,p in model.named_parameters():
			if 'layer' in n:
				if p.requires_grad:
					print(n)
					yield p
	# For conv weight in the ASPP module
	if key == '10x':
		for n,p in model.named_parameters():
			if  'layer' not in n and n[-4:]!='bias':
				if p.requires_grad:
					print(n)
					yield p
	# For conv bias in the ASPP module
	if key == '20x':
		for n,p in model.named_parameters():
			if  'layer' not in n and n[-4:]=='bias':
				if p.requires_grad:
					print(n)
					yield p

def poly_lr_scheduler(optimizer, init_lr, iter_no, lr_decay_iter, max_iter, power=0.9,min_lr=1.0e-6):
	if iter_no % lr_decay_iter or iter_no > max_iter:
		return None
	new_lr = init_lr * (1 - float(iter_no) / max_iter)**power
	if new_lr < min_lr :
		new_lr=min_lr
	# optimizer.param_groups[0]['lr'] = new_lr
	# optimizer.param_groups[1]['lr'] = 10 * new_lr
	# optimizer.param_groups[2]['lr'] = 20 * new_lr

class AverageMeter(object):
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0
	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count


def CalculateMetrics(fground_truth, mground_truth, predicted_map):

	# import ipdb; ipdb.set_trace()

	predicted_map = normalize_map(predicted_map)
	predicted_map = postprocess_prediction(predicted_map, (predicted_map.shape[0], predicted_map.shape[1]))
	predicted_map = normalize_map(predicted_map)
	predicted_map *= 255

	fground_truth = cv2.resize(fground_truth, (0,0), fx=0.5, fy=0.5)
	predicted_map = cv2.resize(predicted_map, (0,0), fx=0.5, fy=0.5)
	mground_truth = cv2.resize(mground_truth, (0,0), fx=0.5, fy=0.5)

	fground_truth = fground_truth.astype(np.float32)/255
	predicted_map = predicted_map.astype(np.float32)
	mground_truth = mground_truth.astype(np.float32)

	AUC_judd_answer = AUC_Judd(predicted_map, fground_truth)
	nss_answer = NSS(predicted_map, fground_truth)
	cc_answer = CC(predicted_map, mground_truth)
	sim_answer = SIM(predicted_map, mground_truth)

	return AUC_judd_answer, nss_answer, cc_answer, sim_answer


def calculate_accuracy(output, target):
	return 0                


def lossFunction(output, target):
	loss = nn.MSELoss()
	LossResult = loss(output, target)

	return LossResult
	
#@click.command()
#@click.option('--config', '-c', type=str, required=True)
def main():
	writer = SummaryWriter(comment='Multi-scale-Deeplab-V2-TrainVal, OriginalImg(input)=Cat2000, GT:4 seperated Heatmap, 20 Categories, split:1800,200. 150epoch, lr=1e-5, Adam, loss:MSE')

	CONFIG ={
	  'SAVE_DIR':'snapshot',
	  'GPU_ID':0,
	  'SNAP_PREFIX':'deeplb',
	  'SNAP_PREFIXVal':'deeplbVal',
	  'INIT_MODEL':'./data/models/deeplab_resnet101/coco_init/deeplabv2_resnet101_COCO_init.pth', 
	  'LR':1e-5, ###0.1
	  'WEIGHT_DECAY': 0.00001,


	}
	if not os.path.exists(CONFIG['SAVE_DIR']):
		os.makedirs(CONFIG['SAVE_DIR'])
		print('created save dir...')  
	cuda = torch.cuda.is_available()
	gpu_id= CONFIG['GPU_ID']
	

	# Model
	model=models.DeepLabV2ResNet101(1) #model
	state_dict = torch.load(CONFIG['INIT_MODEL'])
	model.train()
	model.freeze_bn()

	if cuda:
		print("cudaaaa")
		model.cuda(gpu_id)
	# Dataset
	# import ipdb; ipdb.set_trace()
	train_dataset = inpainting.Inpainting(
		root='/home/arezoo/5-DataSet/',
		mode='train' # name of file train.txt
	)

	val_dataset = inpainting.Inpainting(
		root='/home/arezoo/5-DataSet/',
		mode='val' # name of file train.txt
	)

	print('train_dataset len: '+ str( train_dataset.__len__()))
	train_loader = torch.utils.data.DataLoader(
		dataset=train_dataset,
		batch_size=1,
		num_workers=1,
		shuffle=True,
	)

	val_loader = torch.utils.data.DataLoader(
		dataset=val_dataset,
		batch_size=1,
		num_workers=1,
		shuffle=True,
	)

	# Optimizer
	print('optimizer initializing...')
	

	optimizer = torch.optim.Adam(model.parameters(), lr = 0.00001)
	print('optimizer initialized...')
	print('train_dataset len: '+ str( train_dataset.__len__()))
	# Loss definition
	total_epoch=150
	#added for resuming...
	start_epoch=0 
	bestValidationLoss=float("Inf")

	global_train_iter = 0
	global_val_iter = 0

	for epoch in tqdm(range(0,150),desc='epoch: '):

		folderpath=os.path.join('./outputImages', str(epoch))
		if not os.path.exists(folderpath):
			os.makedirs(folderpath)
	#edit ends 
		print("-------------------------********************---------------------")
		print('%s starting epoch: [%3d/%3d] '%(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),epoch,total_epoch)) 
		# import ipdb; ipdb.set_trace()
		model, trainloss= train(CONFIG,model,train_loader,optimizer,epoch, folderpath, global_train_iter, writer)

		modelpath=os.path.join('./models',str("model-")+str(epoch) +".pt")
		torch.save(model.state_dict(), modelpath)


		

		global_train_iter +=1
		if (epoch+1)%5==0:
			folderpath=os.path.join('./outputImagesVal', str(epoch))
			if not os.path.exists(folderpath):
				os.makedirs(folderpath)
			print("epoch validation: " , epoch)

			_, valloss=Val(CONFIG,model,val_loader,epoch, folderpath, global_val_iter, writer)
			if valloss<bestValidationLoss:
				bestValidationLoss=valloss
				model_path_val=os.path.join(CONFIG['SAVE_DIR'], CONFIG['SNAP_PREFIXVal']+'_ckpt_{}_val.pth'.format((epoch+1)))
				torch.save(model,model_path_val)
				global_val_iter += 1

	 
	return          


def Val(CONFIG,model,val_loader,epoch, folderpath, global_val_iter, writer):
	# import ipdb; ipdb.set_trace()
	print('Validation ...')
	cuda = torch.cuda.is_available()
	gpu_id= CONFIG['GPU_ID']
	

	folderpath=os.path.join('./outputImagesVal', str(epoch))

	model.train(False)
	model.freeze_bn()  #
	loss_meter = AverageMeter()
	
	loader_iter = iter(enumerate(val_loader))
	length=val_loader.__len__()
	

	for i, batch in (tqdm(loader_iter,desc = "iter: ",leave = True)):
		with torch.no_grad():
			data, nameImg, target=batch
			nameImg=nameImg[0]

			##############################################################	
			Dirname= nameImg.split('-')[0]
			fground_truth_name= nameImg.split('-')[1]+'.jpg'
			mground_truth_name= nameImg.split('-')[1]+'_SaliencyMap.jpg'
			##############################################################


			data = data.cuda(gpu_id) 
			data = Variable(data, volatile=True)
			output=model(data)
			target = target.cuda(gpu_id)
			target = Variable(target).to('cuda:0')
			output=F.upsample(output,target.size()[2:])

			loss = lossFunction(output, target)
			loss_meter.update(loss)

	
	writer.add_scalar('loss/val_loss',loss_meter.avg,global_val_iter)
	
	
	print("loss_meter.avg")
	print(loss_meter.avg) 
	return model, loss_meter.avg







# @profile
def train(CONFIG,model,train_loader,optimizer,epoch, folderpath, global_train_iter, writer):
	print('training ...')
	cuda = torch.cuda.is_available()
	gpu_id= CONFIG['GPU_ID']
	model.train()
	model.freeze_bn()  #
	loss_meter = AverageMeter()
	poly_lr_scheduler(   #
			optimizer=optimizer,
			init_lr=CONFIG['LR'],
			iter_no=epoch,
			lr_decay_iter=1,
			max_iter=50, 
		)

	loader_iter = iter(enumerate(train_loader))
	length=train_loader.__len__()
	


	for i, batch in (tqdm(loader_iter,desc = "iter: ",leave = True)):
		
		folderpath=os.path.join('./outputImages', str(epoch))

		data, nameImg, target=batch
		nameImg= nameImg[0]

		
		optimizer.zero_grad()
		data = data.cuda(gpu_id) 
		data = Variable(data)
		output = model(data)
		

		target = Variable(target).to('cuda:0')
		output=F.upsample(output,target.size()[2:])

		loss = lossFunction(output, target)
		loss.backward()
		loss_meter.update(loss)
		optimizer.step()

		
	writer.add_scalar('loss/train_loss',loss_meter.avg,global_train_iter)
	
	
	print("loss_meter.avg")
	print(loss_meter.avg) 
	return model, loss_meter.avg


if __name__ == '__main__':
	main()
