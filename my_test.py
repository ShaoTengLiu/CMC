import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets

path = '../data/myCIFAR-10-C/CIFAR-10-C-trainval/val/gaussian_noise_0_images.npy'
des = './results/temp/pic/other_gaussian_noise1_large.png'
label_path = '../data/myCIFAR-10-C/CIFAR-10-C-trainval/val/labels.npy'

teset_raw = np.load( path )
img_np = teset_raw[100]
print(img_np.shape)
img = Image.fromarray(img_np)
img = transforms.Resize(224)(img)
# img.save(des)

label = np.load(label_path)[40000:50000]
print(label[100])