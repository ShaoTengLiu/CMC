CUDA_VISIBLE_DEVICES=2 python LinearProbing.py --dataset cifar \
 --data_folder /home/stliu/data/myCIFAR-10-C/ \
 --save_path ./results/model_lc/ \
 --tb_path ./results/tb_lc/ \
 --model_path ./results/model/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_gaussian_noise/ckpt_epoch_240.pth \
 --model alexnet --learning_rate 0.1 --layer 5 \
 --view gaussian_noise --level 5 # comment this line for baseline
#  --corruption gaussian_noise --level 5 # use this to test baseline on corruption dataset
# change model_path and view and corruption