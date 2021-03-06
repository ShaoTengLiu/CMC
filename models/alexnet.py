from __future__ import print_function

import torch
import torch.nn as nn
from corruption import C_list
import numpy as np

class MyAlexNetCMC(nn.Module):
	def __init__(self, feat_dim=128):
		super(MyAlexNetCMC, self).__init__()
		self.encoder = alexnet(feat_dim=feat_dim)
		self.encoder = nn.DataParallel(self.encoder)

	def forward(self, x, layer=8):
		return self.encoder(x, layer)
class alexnet(nn.Module):
	def __init__(self, feat_dim=128):
		super(alexnet, self).__init__()

		self.l_to_ab = alexnet_half(in_channel=1, feat_dim=feat_dim)
		self.ab_to_l = alexnet_half(in_channel=2, feat_dim=feat_dim)

	def forward(self, x, layer=8):
		l, ab = torch.split(x, [1, 2], dim=1)
		feat_l = self.l_to_ab(l, layer)
		feat_ab = self.ab_to_l(ab, layer)
		return feat_l, feat_ab

class MyAlexNetCMC_c(nn.Module):
	def __init__(self, feat_dim=128):
		super(MyAlexNetCMC_c, self).__init__()
		self.encoder = alexnet_c(feat_dim=feat_dim)
		self.encoder = nn.DataParallel(self.encoder)

	def forward(self, x, layer=8):
		return self.encoder(x, layer)
class alexnet_c(nn.Module):
	def __init__(self, feat_dim=128):
		super(alexnet_c, self).__init__()

		self.c_a = alexnet_half(in_channel=3, feat_dim=feat_dim)
		self.c_b = alexnet_half(in_channel=3, feat_dim=feat_dim)

	def forward(self, x_a, x_b, layer=8):
		# l, ab = torch.split(x, [1, 2], dim=1)
		feat_a = self.c_a(x_a, layer)
		feat_b = self.c_b(x_b, layer)
		return feat_a, feat_b

class MyAlexNetCMC_cc(nn.Module):
	def __init__(self, feat_dim=128, corruption='original'):
		super(MyAlexNetCMC_cc, self).__init__()
		self.encoder = alexnet_cc(feat_dim=feat_dim, corruption=corruption)
		self.encoder = nn.DataParallel(self.encoder)

	def forward(self, x, layer=8):
		return self.encoder(x, layer)	
class alexnet_cc(nn.Module):
	def __init__(self, feat_dim=128, corruption='original'):
		super(alexnet_cc, self).__init__()

		self.l_to_ab = alexnet_half(in_channel=3, feat_dim=feat_dim)
		self.ab_to_l = alexnet_half(in_channel=3, feat_dim=feat_dim)
		self.corruption = corruption

	def forward(self, x, layer=8):
		#####
		l = x.float().cuda()
		# ab = torch.from_numpy(C_list()[self.corruption](x)).float().cuda()
		# print(x.size())
		x_np = x.cpu().detach().numpy()
		x_np_tran = []
		for i in range(x_np.shape[0]):
			# x_np[i] = C_list()[self.corruption](x_np[i].transpose(1, 2, 0)).transpose(2, 0, 1)
			x_np_tran.append( C_list()[self.corruption](x_np[i].transpose(1, 2, 0)).transpose(2, 0, 1) )
		x_np_tran = np.array(x_np_tran)
		ab = torch.from_numpy(x_np_tran).float().cuda()
		##### This is strange
		feat_l = self.l_to_ab(l, layer)
		feat_ab = self.ab_to_l(ab, layer)
		return feat_l, feat_ab

class alexnet_half(nn.Module):
	def __init__(self, in_channel=1, feat_dim=128):
		super(alexnet_half, self).__init__()
		self.conv_block_1 = nn.Sequential(
			nn.Conv2d(in_channel, 96//2, 11, 4, 2, bias=False),
			nn.BatchNorm2d(96//2),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(3, 2),
			nn.AdaptiveMaxPool2d(27),
		)
		self.conv_block_2 = nn.Sequential(
			nn.Conv2d(96//2, 256//2, 5, 1, 2, bias=False),
			nn.BatchNorm2d(256//2),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(3, 2),
			nn.AdaptiveMaxPool2d(13),
		)
		self.conv_block_3 = nn.Sequential(
			nn.Conv2d(256//2, 384//2, 3, 1, 1, bias=False),
			nn.BatchNorm2d(384//2),
			nn.ReLU(inplace=True),
		)
		self.conv_block_4 = nn.Sequential(
			nn.Conv2d(384//2, 384//2, 3, 1, 1, bias=False),
			nn.BatchNorm2d(384//2),
			nn.ReLU(inplace=True),
		)
		self.conv_block_5 = nn.Sequential(
			nn.Conv2d(384//2, 256//2, 3, 1, 1, bias=False),
			nn.BatchNorm2d(256//2),
			nn.ReLU(inplace=True),
			# nn.MaxPool2d(3, 2),
			nn.AdaptiveMaxPool2d(6),
		)
		self.fc6 = nn.Sequential(
			nn.Linear(256 * 6 * 6 // 2, 4096 // 2),
			nn.BatchNorm1d(4096 // 2),
			nn.ReLU(inplace=True),
		)
		self.fc7 = nn.Sequential(
			nn.Linear(4096 // 2, 4096 // 2),
			nn.BatchNorm1d(4096 // 2),
			nn.ReLU(inplace=True),
		)
		self.fc8 = nn.Sequential(
			nn.Linear(4096 // 2, feat_dim)
		)
		self.l2norm = Normalize(2)

	def forward(self, x, layer):
		if layer <= 0:
			return x
		x = self.conv_block_1(x)
		if layer == 1:
			return x
		x = self.conv_block_2(x)
		if layer == 2:
			return x
		x = self.conv_block_3(x)
		if layer == 3:
			return x
		x = self.conv_block_4(x)
		if layer == 4:
			return x
		x = self.conv_block_5(x)
		if layer == 5:
			return x
		x = x.view(x.shape[0], -1)
		x = self.fc6(x)
		if layer == 6:
			return x
		x = self.fc7(x)
		if layer == 7:
			return x
		x = self.fc8(x)
		x = self.l2norm(x)
		return x

class Normalize(nn.Module):

	def __init__(self, power=2):
		super(Normalize, self).__init__()
		self.power = power

	def forward(self, x):
		norm = x.pow(self.power).sum(1, keepdim=True).pow(1. / self.power)
		out = x.div(norm)
		return out


if __name__ == '__main__':

	import torch
	model = alexnet().cuda()
	data = torch.rand(10, 3, 224, 224).cuda()
	out = model.compute_feat(data, 5)

	for i in range(10):
		out = model.compute_feat(data, i)
		print(i, out.shape)