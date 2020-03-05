# CUDA_VISIBLE_DEVICES=0 python feature_dump.py --dataset cifar \
#  --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model_path ./results/beta/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_240.pth \
#  --feat_path ./results/feat_from_model \
#  --model resnet_ttt --learning_rate 30 \
#  --view Lab --level 5 \
#  --corruption brightness --test_level 5

common_corruptions=("original" "gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" \
            "motion_blur" "zoom_blur" "snow" "frost" "fog" \
            "brightness" "contrast" "elastic_transform" "pixelate" "jpeg_compression" "scale")

for corruption in ${common_corruptions[@]}
#也可以写成for element in ${array[*]}
do
CUDA_VISIBLE_DEVICES=0 python feature_dump.py --dataset cifar \
 --num_workers 9 \
 --data_folder ../data/myCIFAR-10-C/ \
 --model_path ./results/beta/model/memory_nce_16384_resnet_ttt_lr_0.03_decay_0.0001_bsz_128_view_Lab/ckpt_epoch_240.pth \
 --feat_path ./results/feat_from_model \
 --model resnet_ttt --learning_rate 30 \
 --view Lab --level 5 \
 --corruption $corruption --test_level 5
done