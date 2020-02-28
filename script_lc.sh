# CUDA_VISIBLE_DEVICES=0 python LinearProbing.py --dataset cifar \
#  --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --save_path ./results/model_lc/ \
#  --tb_path ./results/tb_lc/ \
#  --model_path ./results/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_240.pth \
#  --model resnet_ttt --learning_rate 0.1 --layer 5 \
#  --view Lab # use this to test baseline on corruption dataset

# change model_path & view & corruption & main

# g_n is gaussian on original

CUDA_VISIBLE_DEVICES=3 python LinearProbing.py --dataset cifar \
 --num_workers 9 \
 --data_folder ../data/myCIFAR-10-C/ \
 --save_path ./results/model_lc/ \
 --tb_path ./results/tb_lc/ \
 --model_path ./results/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_pixelate_level_1/ckpt_epoch_240.pth \
 --model resnet_ttt --learning_rate 30 --layer 6 \
 --view pixelate --level 1

# common_corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" \
#             "motion_blur" "zoom_blur" "snow" "frost" "fog" \
#             "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression" "scale")