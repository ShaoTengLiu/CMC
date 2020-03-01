# CUDA_VISIBLE_DEVICES=1 python test_lc.py --dataset cifar \
#  --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model alexnet --learning_rate 0.1 --layer 5 \
#  --model_path ./results/model/memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_Lab/ckpt_epoch_240.pth \
#  --resume ./results/model_lc/calibrated_memory_nce_16384_alexnet_lr_0.03_decay_0.0001_bsz_256_view_Lab_bsz_256_lr_0.1_decay_0_view_original/ckpt_epoch_60.pth \
#  --view Lab \
#  --corruption original --level 5 # use this to test baseline on corruption dataset

# change model_path & view & corruption & main

# g_n is gaussian on original

# common_corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur', 'glass_blur',
# 				'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
# 				'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression','scale']

# CUDA_VISIBLE_DEVICES=0 python test_lc_beta.py --dataset cifar \
#  --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model resnet_ttt --layer 5 \
#  --model_path ./results/beta/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_pixelate_level_5/ckpt_epoch_240.pth \
#  --resume ./results/beta/model_lc/calibrated_memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_pixelate_level_5_bsz_256_lr_30.0_decay_0/resnet_ttt_layer6.pth \
#  --view pixelate --level 5 \
#  --corruption gaussian_noise --test_level 5

common_corruptions=("original", "gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" \
            "motion_blur" "zoom_blur" "snow" "frost" "fog" \
            "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression" "scale")

for corruption in ${common_corruptions[@]}
#也可以写成for element in ${array[*]}
do
CUDA_VISIBLE_DEVICES=7 python test_lc_beta.py --dataset cifar \
 --num_workers 9 \
 --data_folder ../data/myCIFAR-10-C/ \
 --model resnet_ttt --layer 5 \
 --model_path ./results/beta/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_augmentation_level_5/ckpt_epoch_240.pth \
 --resume ./results/beta/model_lc_ll/calibrated_memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_augmentation_level_5_bsz_256_lr_30.0_decay_0/ckpt_epoch_60.pth \
 --view pixelate --level 5 \
 --corruption $corruption --test_level 5
done

# bash script_test.sh > ./results/beta/test_lc/pixelate_ll.txt