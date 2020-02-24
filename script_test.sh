CUDA_VISIBLE_DEVICES=0 python test_lc.py --dataset cifar \
 --num_workers 1 \
 --data_folder /home/stliu/data/myCIFAR-10-C/ \
 --model alexnet --learning_rate 0.1 --layer 5 \
 --model_path ./results/model/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_contrast/ckpt_epoch_240.pth \
 --resume ./results/model_lc/calibrated_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_contrast_bsz_256_lr_0.1_decay_0_view_contrast/ckpt_epoch_60.pth \
 --view contrast \
 --corruption gaussian_noise --level 5 # use this to test baseline on corruption dataset

# change model_path & view & corruption & main

# g_n is gaussian on original

# common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
# 				'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
# 				'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression','scale']