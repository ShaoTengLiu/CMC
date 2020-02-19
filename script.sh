CUDA_VISIBLE_DEVICES=0,1 python train_CMC.py --batch_size 256 --num_workers 36 \
 --data_folder /home/stliu/data/myCIFAR-10-C/\
 --model_path ./results/model/\
 --tb_path ./results/tb/ \
 --view contrast # comment for using baseline

 # cmc2:i_n
 # cmcR:contrast