import numpy as np
path = '/home/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/val/gaussian_noise.npy'
# path = '/home/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/val/labels.npy'
gn_1 = np.load(path)[0:10000]
gn_2 = np.load(path)[40000:50000]
print(gn_1.shape)
print(gn_2.shape)
print(gn_1 == gn_2)