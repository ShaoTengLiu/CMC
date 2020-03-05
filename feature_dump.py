from __future__ import print_function

import os
import sys
import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.distributed as dist
import argparse
import socket
from torch.utils.data import distributed
import tensorboard_logger as tb_logger

from torchvision import transforms, datasets
from dataset import RGB2Lab, RGB2YCbCr
from util import adjust_learning_rate, AverageMeter, accuracy

from models.alexnet import MyAlexNetCMC, MyAlexNetCMC_cc
from models.resnet_beta import MyResNetsCMC
### change this to test ll, knn and liblinear
from models.LinearModel_beta import LinearClassifierAlexNet, LinearClassifierResNet
###
import numpy as np
#####
from corruption import create_augmentation
#####
# from spawn import spawn

def parse_option():

	parser = argparse.ArgumentParser('argument for training')

	parser.add_argument('--print_freq', type=int, default=10, help='print frequency')
	parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
	parser.add_argument('--save_freq', type=int, default=5, help='save frequency')
	parser.add_argument('--batch_size', type=int, default=256, help='batch_size')
	parser.add_argument('--num_workers', type=int, default=32, help='num of workers to use')
	parser.add_argument('--epochs', type=int, default=60, help='number of training epochs')

	# optimization
	parser.add_argument('--learning_rate', type=float, default=0.1, help='learning rate')
	parser.add_argument('--lr_decay_epochs', type=str, default='30,40,50', help='where to decay lr, can be a list')
	parser.add_argument('--lr_decay_rate', type=float, default=0.2, help='decay rate for learning rate')
	parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
	parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
	parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for Adam')
	parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for Adam')

	parser.add_argument('--resume', default='', type=str, metavar='PATH',
						help='path to latest checkpoint (default: none)')

	# model definition
	parser.add_argument('--model', type=str, default='alexnet')
	parser.add_argument('--model_path', type=str, default=None, help='the model to test')
	parser.add_argument('--feat_path', type=str, default=None, help='the place to save dumped features')
	parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

	# dataset
	parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet', 'cifar'])

	# add new views
	parser.add_argument('--view', type=str, default='Lab')
	parser.add_argument('--corruption', type=str, default='original')
	parser.add_argument('--level', type=int, default=5, help='The level of corruption')
	parser.add_argument('--test_level', type=int, default=5, help='The level of corruption')
	# path definition
	parser.add_argument('--data_folder', type=str, default=None, help='path to data')
	parser.add_argument('--save_path', type=str, default=None, help='path to save linear classifier')
	parser.add_argument('--tb_path', type=str, default=None, help='path to tensorboard')

	# data crop threshold
	parser.add_argument('--crop_low', type=float, default=0.2, help='low area in crop')

	# log file
	parser.add_argument('--log', type=str, default='time_linear.txt', help='log file')

	# GPU setting
	parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')

	opt = parser.parse_args()

	if opt.dataset == 'imagenet':
		if 'alexnet' not in opt.model:
			opt.crop_low = 0.08

	iterations = opt.lr_decay_epochs.split(',')
	opt.lr_decay_epochs = list([])
	for it in iterations:
		opt.lr_decay_epochs.append(int(it))

	opt.model_name = opt.model_path.split('/')[-2]
	opt.model_name = 'calibrated_{}_bsz_{}_lr_{}_decay_{}'.format(opt.model_name, opt.batch_size, opt.learning_rate,
																  opt.weight_decay)

	# opt.model_name = '{}_view_{}'.format(opt.model_name, opt.view)
	# opt.model_name = '{}_view_{}'.format(opt.model_name, opt.corruption)
	# Corruption is not useful when training linear classifier


	if opt.dataset == 'imagenet100':
		opt.n_label = 100
	if opt.dataset == 'imagenet':
		opt.n_label = 1000
	if opt.dataset == 'cifar':
		opt.n_label = 10

	opt.feat_folder = os.path.join(opt.feat_path, 'tr_'+opt.view+'.npy')
	opt.feat_folder_val = os.path.join(opt.feat_path, 'val_'+opt.view+'_'+opt.corruption+'.npy')
	return opt

def get_train_val_loader(args):
	if args.view == 'Lab' or args.view == 'YCbCr':
		if args.view == 'Lab':
			mean = [(0 + 100) / 2, (-86.183 + 98.233) / 2, (-107.857 + 94.478) / 2]
			std = [(100 - 0) / 2, (86.183 + 98.233) / 2, (107.857 + 94.478) / 2]
			color_transfer = RGB2Lab()
		else:
			mean = [116.151, 121.080, 132.342]
			std = [109.500, 111.855, 111.964]
			color_transfer = RGB2YCbCr()
		normalize = transforms.Normalize(mean=mean, std=std)
		train_dataset = datasets.CIFAR10(
			root=args.data_folder,
			train=True,
			transform=transforms.Compose([
				# transforms.RandomResizedCrop(224, scale=(args.crop_low, 1.0)),
				transforms.RandomCrop(32, padding=4), # maybe not necessary
				transforms.RandomHorizontalFlip(),
				color_transfer,
				transforms.ToTensor(),
				normalize,
			])
		)
		val_dataset = datasets.CIFAR10(
			root=args.data_folder,
			train=False,
			transform=transforms.Compose([
				color_transfer,
				transforms.ToTensor(),
				normalize,
			])
		)
		if args.corruption != 'original':
			teset_raw = np.load(args.data_folder + '/CIFAR-10-C-trainval/val/%s_%s_images.npy' %(args.corruption, str(args.test_level-1)))
			val_dataset.data = teset_raw
	else:
		print('Use RGB images with %s level %s!' %(args.view, str(args.level)))
		NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		normalize_lst = lambda x: list(map(transforms.Normalize(*NORM), x))
		data_augmentation = create_augmentation(args.view, args.level)
		train_transform = transforms.Compose([
			transforms.RandomCrop(32, padding=4), # maybe not necessary
			# transforms.Resize(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			data_augmentation,
			normalize_lst
		])
		train_dataset = datasets.CIFAR10(root=args.data_folder,
			train=True, download=True, transform=train_transform)

		val_transform = transforms.Compose([
			transforms.ToTensor(),
			data_augmentation,
			normalize_lst
		])
		val_dataset = datasets.CIFAR10(root=args.data_folder,
			train=False, download=True, transform=val_transform)

	print('number of train: {}'.format(len(train_dataset)))
	print('number of val: {}'.format(len(val_dataset)))

	train_sampler = None

	train_loader = torch.utils.data.DataLoader(
		train_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True, sampler=train_sampler)

	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True)

	return train_loader, val_loader, train_sampler

def set_model(args):
	if args.model.startswith('alexnet'):
		model = MyAlexNetCMC()
		classifier = LinearClassifierAlexNet(layer=args.layer, n_label=args.n_label, pool_type='max')
	elif args.model.startswith('resnet'):
		model = MyResNetsCMC(name=args.model, view=args.view, level=args.level)
		if args.model.endswith('v1'):
			classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 1)
		elif args.model.endswith('v2'):
			classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 2)
		elif args.model.endswith('v3'):
			classifier = LinearClassifierResNet(args.layer, args.n_label, 'avg', 4)
		elif 'ttt' in args.model:
			classifier = LinearClassifierResNet(10, args.n_label, 'avg', 1)
		else:
			raise NotImplementedError('model not supported {}'.format(args.model))
	else:
		raise NotImplementedError('model not supported {}'.format(args.model))

	# load pre-trained model
	print('==> loading pre-trained model')
	ckpt = torch.load(args.model_path)
	model.load_state_dict(ckpt['model'])
	print("==> loaded checkpoint '{}' (epoch {})".format(args.model_path, ckpt['epoch']))
	print('==> done')

	model = model.cuda()
	classifier = classifier.cuda()

	model.eval()

	criterion = nn.CrossEntropyLoss().cuda(args.gpu)

	return model, classifier, criterion

def get_feature(train_loader, val_loader, model, opt):
	"""
	one epoch training
	"""
	model.eval()

	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	end = time.time()
	dump_feature_tr = None
	tr_label = None
	for idx, (input, target) in enumerate(train_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		# input = input.float()
		if opt.gpu is not None:
			# input = input.cuda(opt.gpu, non_blocking=True)
			input = list(map( lambda x: x.float().cuda(opt.gpu, non_blocking=True), input ))
		target = target.cuda(opt.gpu, non_blocking=True)
		# ===================forward=====================
		with torch.no_grad():
			feat_l, feat_ab = model(input, opt.layer)
			feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)
			# feat = feat_l.detach()
		if idx == 0:
			dump_feature_tr = feat.cpu().detach().numpy()
			tr_label = target.cpu().detach().numpy()
		else:
			dump_feature_tr = np.concatenate((dump_feature_tr,feat.cpu().detach().numpy()),axis=0)
			tr_label = np.concatenate((tr_label,target.cpu().detach().numpy()),axis=0)
	
	dump_feature_val = None
	val_label = None
	for idx, (input, target) in enumerate(val_loader):
		# measure data loading time
		data_time.update(time.time() - end)
		# input = input.float()
		if opt.gpu is not None:
			# input = input.cuda(opt.gpu, non_blocking=True)
			input = list(map( lambda x: x.float().cuda(opt.gpu, non_blocking=True), input ))
		target = target.cuda(opt.gpu, non_blocking=True)
		# ===================forward=====================
		with torch.no_grad():
			feat_l, feat_ab = model(input, opt.layer)
			feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)
			# feat = feat_l.detach()
		if idx == 0:
			dump_feature_val = feat.cpu().detach().numpy()
			val_label = target.cpu().detach().numpy()
		else:
			dump_feature_val = np.concatenate((dump_feature_val,feat.cpu().detach().numpy()),axis=0)
			val_label = np.concatenate((val_label,target.cpu().detach().numpy()),axis=0)

	return dump_feature_tr, tr_label, dump_feature_val, val_label

def main():
	global best_acc1
	best_acc1 = 0

	args = parse_option()

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	# set the data loader
	train_loader, val_loader, train_sampler = get_train_val_loader(args)
	# set the model
	model, classifier, criterion = set_model(args)

	cudnn.benchmark = True

	print("==> dump feature...")
	tr_feat, tr_label, val_feat, val_label = get_feature(train_loader, val_loader, model, args)
	# feat, label = get_feature(train_loader, model, args)
	# np.save(args.feat_folder, tr_feat)
	np.save(args.feat_folder_val, val_feat)
	# np.save('results/feat_from_model/tr_label.npy', tr_label)
	# np.save('results/feat_from_model/val_label.npy', val_label)

if __name__ == '__main__':
	best_acc1 = 0
	main()