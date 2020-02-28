# CUDA_VISIBLE_DEVICES=1 python train_CMC.py --batch_size 256 --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model_path ./results/temp/model/ \
#  --tb_path ./results/temp/tb/ \
#  --view Lab


CUDA_VISIBLE_DEVICES=7 python train_CMC.py --model resnet_ttt --batch_size 128 --num_workers 24 \
 --data_folder ../data/myCIFAR-10-C/ \
 --model_path ./results/neo/model/ \
 --tb_path ./results/neo/tb/ \
 --view snow --level 5


# common_corruptions=("gaussian_noise" "shot_noise" "impulse_noise" "defocus_blur" "glass_blur" \
#             "motion_blur" "zoom_blur" "snow" "frost" "fog" \
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