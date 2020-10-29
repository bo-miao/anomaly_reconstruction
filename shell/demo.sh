#!/bin/bash
# python PATH
cd ..

gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
dist_url='tcp://127.0.0.1:70'${port}
echo ${dist_url}

python net.py \
    --demo 'webcam' \
    --multiprocessing_distributed \
    --resume '/home/miaobo/project/anomaly_demo/ckpt/best_unguided0,1,2,3_Unet_ucf_checkpoint.pth.tar' \
    --dist_url ${dist_url} \
    --gpu 0 \
    --arch 'Unet' \
    --is_amp 1 \
    --optim 'adamW' \
