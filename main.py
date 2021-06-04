# coding=utf-8
import os
import gzip
import glob
import math
import numpy as np
import nibabel as nib

import matplotlib.pylab as plt
import matplotlib.cm as cm
import matplotlib.image as Image

from data.BraTS2020 import BraTS2020
from data.Trans20 import Trans2020
from data.SA20 import SA20

import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from tqdm import tqdm

import scipy.misc

import warnings
warnings.filterwarnings('ignore')

import models
import loss_function
from utils.utils import *
from loss_function.ImportLoss import DiceLoss

from config import config, modelconfig


def train(**kwargs):
	print("Task -> train...")
	config._parse(kwargs)

	model = getattr(models, config.model)(modelconfig)
	criterion = getattr(loss_function, config.loss_function)()

	if t.cuda.is_available() is False:
		raise Exception("cuda is not working.")
	
	gpu_devices = [i for i in range(config.use_gpu_num)]
	model = nn.DataParallel(model)
	model = model.cuda(device=gpu_devices[0])
	criterion = criterion.cuda()

	ckpt_path = 'ckpt/' + config.model + config.description + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.train()

	optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
	lr_ = config.lr
	if config.lr_decay != 1.0:
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_decay, patience=5, min_lr=config.lr*0.001, verbose=True)
	train_dataset = BraTS2020(config)
	train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

	figure_losses = []
	for epoch in range(config.max_epoch):

		losses = []
		for images, labels in train_dataloader:
			times = images.shape[1]
			for i in range(times):
				_image = images[:, i, ...].cuda()
				_label = labels[:, i, ...].cuda()
				optimizer.zero_grad()
				predict = model(_image)
				loss = criterion(predict, _label)
				_, predict = t.max(predict, dim=1)
				loss.backward()
				optimizer.step()

				losses.append(loss)
			print('loss: {}'.format(sum(losses) / len(losses)))
		losses = sum(losses) / len(losses)
		figure_losses.append(float(losses))

		# if (epoch+1) % 10 == 0:
		# 	# update learn rate
		# 	lr_ = lr_ * config.lr_decay
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] = lr_

		if config.lr_decay != 1.0:
			scheduler.step(losses)

		print('epoch {}/{}: losses: {}, lr: {}'.format(epoch, config.max_epoch, losses, lr_))
		t.save(model.state_dict(), os.path.join(ckpt_path, config.model + '_{}_.pth'.format(epoch)))
	
	print('figure_losses:')
	print(figure_losses)


def test(**kwargs):
	print("TASK -> test...")
	config._parse(kwargs)

	model = getattr(models, config.model)(modelconfig)
	criterion = getattr(loss_function, config.loss_function)()

	if t.cuda.is_available() is False:
		raise Exception("cuda is not working.")
	
	gpu_devices = [i for i in range(config.use_gpu_num)]
	model = nn.DataParallel(model)
	model = model.cuda(device=gpu_devices[0])
	criterion = criterion.cuda()

	ckpt_path = 'ckpt/' + config.model + config.description + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.eval()

	config.max_epoch = 1
	config.batch_size = 1
	for epoch in range(config.max_epoch):
		train_dataset = BraTS2020(config)
		train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

		losses = []
		dices = []
		for name, images, labels, affine, index_min, index_max in train_dataloader:
			times = images.shape[1]
			# image.shape: torch.Size([1, 9, 4, 64, 64, 144])
			predicts = []

			for i in range(times):
				_image = images[:, i, ...].cuda()

				predict = model(_image)
				_, predict = t.max(predict, dim=1)
				predicts.append(predict.int())
			

			out_predict = None
			for j in range(times-1):
				if j == 0:
					out_predict = t.cat((predicts[0], predicts[1]), dim=-1)
				else:
					out_predict = t.cat((out_predict, predicts[j+1]), dim=-1)

			out_predict = out_predict.data.cpu().numpy()[0]
			out_predict = out_val_processing(out_predict)
			pp = np.zeros((240, 240, 155))
			x, y, z = index_min
			pp[x:x+192, y:y+192, z:z+144] = out_predict
			output = nib.Nifti1Image(pp, affine[0])
			path = './predict/' + config.predict_path
			if not os.path.exists(path):
				os.mkdir(path)
			nib.save(output, path + name[0] + '.nii.gz')


def trans_train(**kwargs):
	config._parse(kwargs)
	print('load trans_train')
	model = getattr(models, config.model)(modelconfig)
	criterion = getattr(loss_function, config.loss_function)()
	print('loaded trans_train')

	if t.cuda.is_available() is False:
		print("cuda is not working.")
	
	if t.cuda.is_available():
		gpu_devices = [i for i in range(config.use_gpu_num)]
		model = nn.DataParallel(model)
		model = model.cuda(device=gpu_devices[0])
		criterion = criterion.cuda()

	ckpt_path = 'ckpt/' + config.model + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.train()

	optimizer = optim.Adam(params=model.parameters(), lr=config.lr, betas=(0.9, 0.999))
	ce_loss = nn.CrossEntropyLoss()
	dice_loss = DiceLoss(modelconfig.n_classes)

	dataset = Trans2020(config)

	# use no dataloader
	# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

	num_compute = 0
	max_num_compute = config.max_epoch * len(dataset) * 5

	for epoch in range(config.max_epoch):
		print('epoch: ', epoch)
		for i, (image, label) in enumerate(dataset):
			if config.use_gpu:
				image = image.cuda()
				label = label.cuda()
			for _ in range(5):
				ids = np.random.randint(0, 155, 32) # batch_size = 32
				image_ = image[ids, ...]
				label_ = label[ids, ...]

				outputs = model(image_)
				loss = ce_loss(outputs, label_.long())
				# loss_dice = dice_loss(outputs, label_, softmax=True)


				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				print('epoch: {}, data: {}, Loss -> ce: {}'.format(epoch, i, loss.item()))

				# update learn rate
				lr_ = config.lr * (1.0 - num_compute/max_num_compute) ** 0.9
				for param_group in optimizer.param_groups:
					param_group['lr'] = lr_

				num_compute += 1
		if (epoch+1) % 5 == 0:
			t.save(model.state_dict(), os.path.join(ckpt_path, config.model + '_{}_.pth'.format(epoch)))
	t.save(model.state_dict(), os.path.join(ckpt_path, config.model + '_final_.pth'))


def trans_test(**kwargs):
	config._parse(kwargs)
	print('load trans_test')
	model = getattr(models, config.model)(modelconfig)
	print('loaded trans_test')

	if t.cuda.is_available() is False:
		raise Exception("cuda is not working.")
	
	if config.use_gpu:
		gpu_devices = [i for i in range(config.use_gpu_num)]
		model = nn.DataParallel(model)
		model = model.cuda(device=gpu_devices[0])

	ckpt_path = 'ckpt/' + config.model + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.eval()

	dataset = Trans2020(config)

	# use no dataloader
	# dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

	path = './predict/' + config.predict_path
	if not os.path.exists(path):
		os.mkdir(path)
	for i, (name, image, label, affine) in enumerate(dataset):
		# label is None
		images = []
		for i in range(155):
			image_ = image[i, ...].cuda()
			image_ = image_[np.newaxis, ...]
			output = model(image_)
			_, output = t.max(output, dim=1)
			images.append(output.data.cpu().int()) # [1, 256, 256]
		output = t.cat(images, dim=0)
		output = output.permute(1, 2, 0).numpy()
		tmp = np.zeros((240, 240, 155))
		tmp[24:-24, 24:-24, :] = output
		output = tmp

		output = (output==1)*1.0 + (output==2)*2.0 + (output==3)*4.0
		output = output.astype(np.int16)

		output = nib.Nifti1Image(output, affine)
		nib.save(output, path + name + '.nii.gz')


def trans_train_3d(**kwargs):
	config._parse(kwargs)
	modelconfig.batch_size = config.batch_size
	modelconfig.use_gpu_num = config.use_gpu_num
	print('load trans_train')
	model = getattr(models, config.model)(modelconfig)
	criterion = getattr(loss_function, config.loss_function)()
	print('loaded trans_train')

	if t.cuda.is_available() is False:
		print("cuda is not working.")
	
	if t.cuda.is_available():
		gpu_devices = [i for i in range(config.use_gpu_num)]
		model = nn.DataParallel(model)
		model = model.cuda(device=gpu_devices[0])

	ckpt_path = 'ckpt/' + config.model + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.train()

	optimizer = optim.Adam(params=model.parameters(), lr=config.lr, betas=(0.9, 0.999))
	lr_ = config.lr
	
	dataset = Trans2020(config)
	# use dataloader
	dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=0)

	# num_compute = 0
	# max_num_compute = config.max_epoch * len(dataloader)

	for epoch in range(config.max_epoch):
		for i, (image, label) in enumerate(dataloader):
			# torch.Size([2, 4, 192, 192, 48])
			image_ = image.cuda()
			label_ = label.cuda()
			_loss = []
			# image_ = image[:, :, :, :, :]
			# label_ = label[:, :, :, :]

			outputs = model(image_)
			loss = criterion(outputs, label_.long())
			_loss.append(float(loss))

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			print('epoch: {}/{}, data: {}/125, lr: {}, Loss: {}'.format(epoch+1, config.max_epoch, i, lr_, sum(_loss)/len(_loss)))
		if (epoch+1) % 40 == 0:
			# update learn rate
			lr_ = lr_ * config.lr_decay
			for param_group in optimizer.param_groups:
				param_group['lr'] = lr_
		if (epoch+1) % 5 == 0:
			t.save(model.state_dict(), os.path.join(ckpt_path, config.model + '_{}_.pth'.format(epoch)))
	t.save(model.state_dict(), os.path.join(ckpt_path, config.model + '_final_.pth'))


def trans_test_3d(**kwargs):
	config._parse(kwargs)
	print('load trans_test')
	model = getattr(models, config.model)(modelconfig)
	print('loaded trans_test')

	if t.cuda.is_available() is False:
		raise Exception("cuda is not working.")
	
	if config.use_gpu:
		gpu_devices = [i for i in range(config.use_gpu_num)]
		model = nn.DataParallel(model)
		model = model.cuda(device=gpu_devices[0])

	ckpt_path = 'ckpt/' + config.model + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.eval()

	dataset = Trans2020(config)

	# use no dataloader
	dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=False, num_workers=0)

	path = './predict/' + config.predict_path
	if not os.path.exists(path):
		os.mkdir(path)
	for i, (name, image, label, affine) in enumerate(dataloader):
		# label is None
		images = []
		name = name[0]
		affine = affine[0]
		print('name: {}, image: {}, label: {}, affine: {}'.format(name, image.shape, label.shape, affine.shape))
		
		if config.use_gpu:
				image = image.cuda()
		for kz in range(3):
			zz = kz*48
			image_ = image[:, :, :, :, zz:zz+48]
			output = model(image_)
			_, output = t.max(output, dim=1)
			images.append(output.data.cpu().int())

		output = t.cat(images, dim=-1).numpy()[0]

		tmp = np.zeros((240, 240, 155))
		tmp[24:-24, 24:-24, 5:-6] = output
		output = tmp

		output = (output==1)*1.0 + (output==2)*2.0 + (output==3)*4.0
		output = output.astype(np.int16)

		output = nib.Nifti1Image(output, affine)
		nib.save(output, path + name + '.nii.gz')


def sa_lut_train(**kwargs):
	print("Task -> train...")
	config._parse(kwargs)

	model = getattr(models, config.model)()
	criterion = getattr(loss_function, config.loss_function)()

	if t.cuda.is_available() is False:
		raise Exception("cuda is not working.")
	
	gpu_devices = [i for i in range(config.use_gpu_num)]
	model = nn.DataParallel(model)
	model = model.cuda(device=gpu_devices[0])
	criterion = criterion.cuda()

	ckpt_path = 'ckpt/' + config.model + config.description + '/'
	if not os.path.exists(ckpt_path):
		os.mkdir(ckpt_path)
	if config.load_model:
		model.load_state_dict(t.load(ckpt_path+config.load_model))
	model.train()

	optimizer = optim.Adam(params=model.parameters(), lr=config.lr)
	lr_ = config.lr
	if config.lr_decay != 1.0:
		scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=config.lr_decay, patience=3, min_lr=1e-6, verbose=True)
	train_dataset = SA20(config)
	train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=0, drop_last=True)

	figure_losses = []
	for epoch in range(config.max_epoch):

		losses = []
		for images, labels in train_dataloader:
			# times = images.shape[1]
			# for i in range(times):
			_image = images.cuda()
			_label = labels.cuda()
			optimizer.zero_grad()
			predict = model(_image)
			loss = criterion(predict, _label)
			_, predict = t.max(predict, dim=1)
			loss.backward()
			optimizer.step()

			losses.append(loss)
			print('loss: {}'.format(sum(losses) / len(losses)))
		losses = sum(losses) / len(losses)
		figure_losses.append(float(losses))

		# if (epoch+1) % 10 == 0:
		# 	# update learn rate
		# 	lr_ = lr_ * config.lr_decay
		# 	for param_group in optimizer.param_groups:
		# 		param_group['lr'] = lr_

		if config.lr_decay != 1.0:
			scheduler.step(losses)

		print('epoch {}/{}: losses: {}, lr: {}'.format(epoch, config.max_epoch, losses, lr_))
		t.save(model.state_dict(), os.path.join(ckpt_path, config.model + '_{}_.pth'.format(epoch)))
	
	print('figure_losses:')
	print(figure_losses)


def model_test():
	# For 3d model
	# a = t.tensor(np.random.randn(1, 4, 192, 192, 48)).float()
	# model = models.ResNet_3d(modelconfig)
	# print('input.shape: {}'.format(a.shape))
	# x = model(a)
	# print('output.shape: {}'.format(x.shape))
	# For 2d model
	a = t.tensor(np.random.randn(1, 4, 128, 128, 128)).float()
	model = models.DMFNet(c=4, groups=16, norm='bn', num_classes=4)
	print('input.shape: {}'.format(a.shape))
	x = model(a)
	print('output.shape: {}'.format(x.shape))


def dataset_test(**kwargs):
	# --train_path='/Users/jonty/Downloads/Create/数据集/MICCAI_BraTS2020_TrainingData' --random_width=128
	config._parse(kwargs)
	dataset = SA20(config)
	dataloader = DataLoader(dataset, batch_size=4, shuffle=False)
	# dataset = BraTS2020(config)
	for _ in range(5):
		for image, label in dataloader:
			print('image.shape: {}'.format(image.shape))
			break
	# /Users/jonty/Downloads/Create/数据集/MICCAI_BraTS2020_TrainingData
	# label = (label==1)*1.0 + (label==2)*2.0 + (label==3)*4.0
	# output = np.array(label).astype(np.int16)

	# output = nib.Nifti1Image(output, aff)
	# nib.save(output, '/Users/jonty/Downloads/Create/predict/' + name + '.nii.gz')


def other():
	lr_ = 0.01
	lr_decay = 0.95
	for i in range(32):
		lr_ = lr_ * lr_decay
		print('{}, {}'.format(i, lr_))


def predict_image():
	name = 'BraTS20_Validation_030'
	path_ = '/Users/jonty/Downloads/Create/数据集/MICCAI_BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData'
	file = '/Users/jonty/Downloads/Create/predict/BaseLineModel_lr_schedular_199_val/' + name + '.nii.gz'
	# file = path_ + '/' + name + '/' + name + '.nii.gz'
	image = nib.load(file).get_data()
	img = image[:, :, 88]
	img = img.transpose(1, 0)
	plt.axis('off')
	plt.imshow(img, cmap='inferno')
	plt.savefig('./figure/_BaseLine_' + name + '.png')
	plt.show()




	# path = '/Users/jonty/Library/Mobile Documents/com~apple~CloudDocs/论文/BraTS大脑胶质瘤/BraTS2020/期刊_Pattern_Recognition/Result/Predict'
	# files = glob.glob(path + '/_*')
	# for path in files:
	# 	file = path + '/BraTS20_Validation_006.nii.gz'
	# 	name = path.split('/')[-1]
	# 	print('file: {}'.format(name))

	# 	image = nib.load(file)
	# 	affine = image.affine
	# 	image = image.get_data()

	# 	print(image.shape)
	# 	'''
	# 	034 -> 78
	# 	006 -> 91
	# 	125 -> 45
	# 	030 -> 88
	# 	'''
	# 	img = image[:, :, 91]
	# 	img = img.transpose(1, 0)
	# 	plt.axis('off')
	# 	plt.imshow(img, cmap='inferno')
	# 	plt.savefig('./figure/'+name+'.png')
	# 	plt.show()





if __name__ == '__main__':
	import fire
	fire.Fire()
