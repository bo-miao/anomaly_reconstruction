#!/bin/bash
# python PATH

# hyperparameter
echo -n "input the gpu (seperate by comma (,) ): "
#read gpus
gpus='2,3'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

suffix='expnorm'
port=$(date "+%S")
dist_url='tcp://127.0.0.1:70'${port}
echo ${dist_url}


python net_gan.py \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --multiprocessing_distributed \
    -b 20 \
    --arch_g 'convAE_Patch' \
    --arch_d 'ResNetD' \
    --dataset_type 'ped2' \
    --epochs 150 \
    --is_amp 1 \
    --wgan_gp 1 

# one gpu b 12
