import numpy as np
import torch
# path = '/home/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/val/gaussian_noise.npy'
# # path = '/home/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/val/labels.npy'
# gn_1 = np.load(path)[0:10000]
# gn_2 = np.load(path)[40000:50000]
# print(gn_1.shape)
# print(gn_2.shape)
# print(gn_1 == gn_2)
# lst = [[1,2,3],[4,5,6],[7,8,9]]
lst = [1,2,3]
lst = torch.randn(2, 3, 5) 
# lst[1] = np.array([0,0,0])
lst = lst.numpy()
print(lst.shape)
# lst = torch.from_numpy(lst)
lst = lst.transpose(2,0,1)
# for i in range(lst.shape[0]):
#     lst[i] = np.array([0,0,0])
print(lst.shape)