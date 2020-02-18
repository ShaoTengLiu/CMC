CUDA_VISIBLE_DEVICES=1 python train_CMC.py --batch_size 256 --num_workers 36 \
 --data_folder /home/stliu/data/myCIFAR-10-C/\
 --model_path ./results/model/\
 --tb_path ./results/tb/\
 --view gaussian_noise