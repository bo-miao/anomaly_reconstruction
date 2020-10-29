#!/bin/bash
# python PATH
cd ..

gpus='2,3'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

suffix='avenue'${gpus}
port=$(date "+%S")
dist_url='tcp://127.0.0.1:73'${port}
echo ${dist_url}

python net.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --print_freq 1 \
    --t_length 5 \
    -b 128  \
    --test_batch_size 128 \
    --h 360 \
    --w 640 \
    --interval 2 \
    --arch 'Unet_Light' \
    --encoder_arch 'Encoder_Light' \
    --decoder_arch 'Decoder_Light' \
    --dataset_type 'avenue' \
    --label 1 \
    --training_folder 'training/frames' \
    --testing_folder 'testing/frames' \
    --label_folder 'label' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 3 \
    --epochs 60 \
    --is_amp 1 \
    --optim 'adamW' \
    #--resume '/home/miaobo/project/anomaly_demo/ckpt/best_guided0,1,2,3_Unet_ucf_checkpoint.pth.tar'
