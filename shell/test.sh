#!/bin/bash
# python PATH
cd ..

gpus='0,1,2,3'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

suffix='unguided'${gpus}
port=$(date "+%S")
dist_url='tcp://127.0.0.1:70'${port}
echo ${dist_url}

python net.py \
    --test_batch_size 64 \
    --multiprocessing_distributed \
    --evaluate \
    --resume '/home/miaobo/project/anomaly_demo/ckpt/best_unguided0,1,2,3_Unet_ucf_checkpoint.pth.tar' \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --dataset_type 'ucf' \
    --label 1 \
    --training_folder 'training_simple/frames' \
    --testing_folder 'testing_simple/frames' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --arch 'Unet' \
    --is_amp 1 \
    --optim 'adamW' \
