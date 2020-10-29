#!/bin/bash
# python PATH
cd ..

gpus='0,1,2,3'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

suffix='toy'${gpus}
port=$(date "+%S")
dist_url='tcp://127.0.0.1:79'${port}
echo ${dist_url}

python net.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    -b 256  \
    --test_batch_size 256 \
    --h 128 \
    --w 128 \
    --arch 'Unet' \
    --dataset_type 'ucf' \
    --label 1 \
    --training_folder 'training_toy/frames_low' \
    --testing_folder 'testing_toy/frames_low' \
    --label_folder 'label_low' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 1 \
    --epochs 60 \
    --is_amp 1 \
    --optim 'adamW' \
    # --resume '/home/miaobo/project/anomaly_demo/ckpt/best_guided0,1,2,3_Unet_ucf_checkpoint.pth.tar'
