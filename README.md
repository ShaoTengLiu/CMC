Expand the implementation of this paper:
- CMC: Contrastive Multiview Coding ([Paper](http://arxiv.org/abs/1906.05849))



## Installation

This repo was tested with Ubuntu 16.04.5 LTS, Python 3.5, PyTorch 0.4.0, and CUDA 9.0. But it should be runnable with recent PyTorch versions >=0.4.0

**Note:** It seems to us that training with Pytorch version >= 1.0 yields slightly worse results. If you find the similar discrepancy and figure out the problem, please report this since we are trying to fix it as well.

**The environment has been equipped in cmc on dqwang@cthulhu1.ist.berkeley.edu**

**One can use "conda activate cmc" to use**

**The location of code is in ~/stliu/dynamic_cmc**



## Data Location

**~/stliu/data/myCIFAR-10-C**: the dataroot

**~/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval**: data created by dequan

**~/stliu/data/myCIFAR-10-C/CIFAR-10-C-trainval/cifar-10-batches-py**: the data datasets.CIFAR10 will use





## Training ResNets(of ttt) with CMC on Cifar

Run **script.sh** to train cmc

**script.sh**:

```
CUDA_VISIBLE_DEVICES=0 python train_CMC.py --model resnet_ttt --batch_size 128 --num_workers 24 \
 --feat_dim 64 \
 --data_folder ../data/myCIFAR-10-C/ \
 --model_path ./results/model/ \
 --tb_path ./results/tb/ \
 --view contrast --level 5
```

Change **view and level** to change the data augmentation of cmc model.

Change view to Lab for CMC baseline (the level is not important when you use this)

--model_path and --tb_path is the location to save model and tensor board

Choose resent_ttt for --model to use the same ResNet as TTT.



## Training Linear Classifier

Run **script_lc.sh** to train the linear classifier

**script_lc.sh**:

```
CUDA_VISIBLE_DEVICES=0 python LinearProbing.py --dataset cifar \
 --num_workers 9 \
 --data_folder ../data/myCIFAR-10-C/ \
 --save_path ./results/temp/model_lc/ \
 --tb_path ./results/temp/tb_lc/ \
 --model_path ./results/neo/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_gaussian_noise_level_5/ckpt_epoch_240.pth \
 --model resnet_ttt --learning_rate 30 --layer 6 \
 --view gaussian_noise --level 5
```

 --model_path is the location of the model (trained from script.sh) you want to use.

 **--learning_rate** is set to 30, as required by the author.

--view and --level should be corresponding to the model you use (--model_path)