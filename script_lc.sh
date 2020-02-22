CUDA_VISIBLE_DEVICES=0 python LinearProbing.py --dataset cifar \
 --num_workers 0 \
 --data_folder /home/stliu/data/myCIFAR-10-C/ \
 --save_path ./results/model_lc/ \
 --tb_path ./results/tb_lc/ \
 --model_path ./results/model/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_contrast/ckpt_epoch_240.pth \
 --model alexnet --learning_rate 0.1 --layer 5 \
 --view contrast \
 --corruption contrast --level 5 # use this to test baseline on corruption dataset

# change model_path & view & corruption & main

# g_n is gaussian on original