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
from models.resnet import MyResNetsCMC
from models.LinearModel import LinearClassifierAlexNet, LinearClassifierResNet

import numpy as np
# from spawn import spawn
from PIL import Image

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
	parser.add_argument('--layer', type=int, default=6, help='which layer to evaluate')

	# dataset
	parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet100', 'imagenet', 'cifar'])

	# add new views
	parser.add_argument('--view', type=str, default='Lab')
	parser.add_argument('--level', type=int, default=5, help='The level of model')
	parser.add_argument('--corruption', type=str, default='original')
	parser.add_argument('--test_level', type=int, default=5, help='The level of corruption')
	# path definition
	parser.add_argument('--data_folder', type=str, default=None, help='path to data')
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
	opt.model_name = '{}_view_{}'.format(opt.model_name, opt.corruption)
	# one may change this view into corruption to make the name more reasonable

	if opt.dataset == 'imagenet100':
		opt.n_label = 100
	if opt.dataset == 'imagenet':
		opt.n_label = 1000
	if opt.dataset == 'cifar':
		opt.n_label = 10

	return opt

def get_val_loader(args):
	common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
							'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
							'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression']
	print('Use %s!' %(args.view))
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

		te_transform = transforms.Compose([
			color_transfer,
			transforms.ToTensor(),
			normalize,
		])
		val_dataset = datasets.CIFAR10(
			root=args.data_folder,
			train=False,
			transform=te_transform
		)
		if args.corruption in common_corruptions:
			print('Test on %s!' %(args.corruption))
			teset_raw = np.load(args.data_folder + '/CIFAR-10-C-trainval/val/%s_%s_images.npy' %(args.corruption, str(args.test_level - 1)))
			teset_raw = color_transfer( Image.fromarray(teset_raw.astype(np.uint8)) )
			val_dataset.data = teset_raw
	else:
		NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
		normalize = transforms.Normalize(*NORM)
		te_transform = transforms.Compose([
			transforms.ToTensor(),
			# normalize,
		])
		val_dataset = datasets.CIFAR10(
			root=args.data_folder,
			train=False,
			transform=te_transform
		)
		if args.corruption in common_corruptions:
			print('Test on %s!' %(args.corruption))
			teset_raw = np.load(args.data_folder + '/CIFAR-10-C-trainval/val/%s_%s_images.npy' %(args.corruption, str(args.test_level - 1)))
			val_dataset.data = teset_raw

	print('number of val: {}'.format(len(val_dataset)))

	val_loader = torch.utils.data.DataLoader(
		val_dataset, batch_size=args.batch_size, shuffle=False,
		num_workers=args.num_workers, pin_memory=True)

	return val_loader

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

	return model, classifier

def validate(val_loader, model, classifier, opt):
	"""
	evaluation
	"""
	batch_time = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to evaluate mode
	model.eval()
	classifier.eval()

	with torch.no_grad():
		end = time.time()
		for idx, (input, target) in enumerate(val_loader):
			input = input.float()
			if opt.gpu is not None:
				input = input.cuda(opt.gpu, non_blocking=True)
			target = target.cuda(opt.gpu, non_blocking=True)
			# compute output
			feat_l, feat_ab = model(input, opt.layer)
			feat = torch.cat((feat_l.detach(), feat_ab.detach()), dim=1)
			output = classifier(feat)
			##### This may be not necessary
			target = target.long()

			# measure accuracy and record loss
			acc1, acc5 = accuracy(output, target, topk=(1, 5))
			top1.update(acc1[0], input.size(0))
			top5.update(acc5[0], input.size(0))

			# measure elapsed time
			batch_time.update(time.time() - end)
			end = time.time()
			break # temp
			# if idx % opt.print_freq == 0:
			# 	print('Test: [{0}/{1}]\t'
			# 		  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
			# 		  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
			# 		  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
			# 		   idx, len(val_loader), batch_time=batch_time,
			# 		   top1=top1, top5=top5))

		print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
			  .format(top1=top1, top5=top5))

	return top1.avg, top5.avg

def main():
	global best_acc1
	best_acc1 = 0

	args = parse_option()

	if args.gpu is not None:
		print("Use GPU: {} for training".format(args.gpu))

	# set the data loader
	val_loader = get_val_loader(args)
	# set the model
	model, classifier = set_model(args)

	cudnn.benchmark = True

	# optionally resume linear classifier
	args.start_epoch = 1
	if args.resume:
		if os.path.isfile(args.resume):
			print("=> loading checkpoint '{}'".format(args.resume))
			checkpoint = torch.load(args.resume, map_location='cpu')
			classifier.load_state_dict(checkpoint['classifier'])
			del checkpoint
			torch.cuda.empty_cache()
		else:
			print("=> no checkpoint found at '{}'".format(args.resume))

	# routine
	for epoch in range(args.start_epoch, args.epochs + 1):
		print("==> testing...")
		test_acc, test_acc5 = validate(val_loader, model, classifier, args)

		# tensorboard logger
		pass

if __name__ == '__main__':
	best_acc1 = 0
	main()