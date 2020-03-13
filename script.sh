# CUDA_VISIBLE_DEVICES=1 python train_CMC.py --batch_size 256 --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model_path ./results/temp/model/ \
#  --tb_path ./results/temp/tb/ \
#  --view Lab
CUDA_VISIBLE_DEVICES=7 python train_CMC_beta.py --model resnet_ttt --batch_size 128 --num_workers 1 \
 --feat_dim 64 \
 --data_folder ../data/myCIFAR-10-C/ \
 --model_path ./results/beta/model/ \
 --tb_path ./results/beta/tb/ \
 --view scale --level 1
#  --oracle original
#  --resume ./results/beta/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_contrast_level_5/ckpt_epoch_140.pth \

# common_corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" \
#             "motion_blur" "zoom_blur" "snow" "frost" "f og" \
#             "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression" "scale")

# for corruption in ${common_corruptions[@]}
# #也可以写成for element in ${array[*]}
# do
# CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7,8,9 python train_CMC.py --model resnet_ttt --batch_size 128 --num_workers 24 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model_path ./results/model/ \
#  --tb_path ./results/tb/ \
#  --view $corruption
# done