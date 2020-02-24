# CUDA_VISIBLE_DEVICES=0,1,2,3 python train_CMC.py --batch_size 256 --num_workers 9 \
#  --data_folder ../data/myCIFAR-10-C/ \
#  --model_path ./results/model/ \
#  --tb_path ./results/tb/ \
#  --view Lab
 
 
CUDA_VISIBLE_DEVICES=4,5,6,7 python train_CMC.py --model resnet50v1 --batch_size 128 --num_workers 24 \
 --data_folder ../data/myCIFAR-10-C/ \
 --model_path ./results/temp/model/ \
 --tb_path ./results/temp/tb/