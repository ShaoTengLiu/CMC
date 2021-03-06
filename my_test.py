import numpy as np
import torch
from PIL import Image
from torchvision import transforms, datasets
from corruption import create_augmentation

data_folder = '../data/myCIFAR-10-C/'
# des = './show/pic/origin_n.jpg'

# NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# normalize = transforms.Normalize(*NORM)
# te_transform = transforms.Compose([
#     transforms.ToTensor(),
#     # normalize,
# ])
# add_transform = transforms.Compose([
#     transforms.ToTensor(),
#     normalize,
# ])
# val_dataset = datasets.CIFAR10(
#     root=data_folder,
#     train=False,
#     # transform=te_transform
#     transform=add_transform
# )

# print('number of val: {}'.format(len(val_dataset)))

# sample = val_dataset.data[100]
# print(sample.shape)
# img = Image.fromarray(np.uint8(sample))
# # img.save(des)

# val_loader = torch.utils.data.DataLoader(
#     val_dataset, batch_size=256, shuffle=False,
#     num_workers=9, pin_memory=True)

convert_img = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage()])

# for idx, (input, target) in enumerate(val_loader):
#     input = input.float() # (256, 3, 32, 32)
#     sample = input[100].detach().numpy()
#     img = convert_img( sample.transpose(1, 2, 0) )
#     # img = add_transform(img)
#     # sample = img.detach().numpy()
#     # img = convert_img( sample.transpose(1, 2, 0) )
#     # img.save(des)
#     break

# gn5_path = '/home/dqwang/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/val/gaussian_noise_4_images.npy'
gn1_path = '/home/dqwang/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/val/contrast_3_images.npy'
# des5 = './show/pic/gn_5.jpg'
des1 = './show/pic/contrast_4.jpg'

# gn5 = np.load(gn5_path)[100]
gn1 = np.load(gn1_path)[100]

# img5 = convert_img(gn5)
img1 = convert_img(gn1)

# img5.save(des5)
img1.save(des1)

# NORM = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
# normalize_lst = lambda x: list(map(transforms.Normalize(*NORM), x))
# data_augmentation = create_augmentation('gaussian_noise', 5)
# train_transform = transforms.Compose([
#     transforms.RandomCrop(32, padding=4), # maybe not necessary
#     # transforms.Resize(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.ToTensor(),
#     data_augmentation,
#     normalize_lst
# ])

# train_dataset = datasets.CIFAR10(root=data_folder,
#     train=True, download=True, transform=train_transform)
# train_sampler = None

# # train loader
# train_loader = torch.utils.data.DataLoader(
#     train_dataset, batch_size=128, shuffle=(train_sampler is None),
#     num_workers=9, pin_memory=True, sampler=train_sampler)

# for idx, (input, target) in enumerate(train_loader):
#     input = input.cuda()
#     print(type(input[1]))
#     # input = input.float() # (256, 3, 32, 32)
#     # sample = input[100].detach().numpy()
#     # img = convert_img( sample.transpose(1, 2, 0) )

#     # img = add_transform(img)
#     # sample = img.detach().numpy()
#     # img = convert_img( sample.transpose(1, 2, 0) )
#     # img.save(des)
#     break