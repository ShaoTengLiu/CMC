CUDA_VISIBLE_DEVICES=2 python train_CMC.py --batch_size 256 --num_workers 9 \
 --data_folder /home/stliu/data/myCIFAR-10-C/\
 --model_path ./results/model/\
 --tb_path ./results/tb/ \
 --view glass_blur # comment for using baseline

 # cmc2:i_n
 # cmcR:contrast

 # 36 - > 18