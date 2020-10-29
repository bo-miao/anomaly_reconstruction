#!/bin/bash
# python PATH
cd ..

gpus='2,3'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

suffix='ped2'${gpus}
port=$(date "+%S")
dist_url='tcp://127.0.0.1:70'${port}
echo ${dist_url}

python net.py \
    --test_batch_size 64 \
    --multiprocessing_distributed \
    --evaluate \
    --resume '/home/miaobo/project/anomaly_demo/ckpt/best_3ped22,3,4,5_Unet_ped2_checkpoint.pth.tar' \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --dataset_type 'ped2' \
    --t_length 5 \
    --interval 2 \
    --object_detection 0 \
    --h 240 \
    --w 360 \
    --label 1 \
    --training_folder 'training/frames' \
    --testing_folder 'testing/frames' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --arch 'Unet' \
    --encoder_arch 'Encoder' \
    --decoder_arch 'Decoder' \
    --dataset_type 'ped2' \
    --is_amp 1 \
    --optim 'adamW' \
