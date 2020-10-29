#!/bin/bash
# python PATH
cd ..

gpus='2,3,4,5'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'object'${gpus}
dist_url='tcp://127.0.0.1:72'${port}
echo ${dist_url}

python net.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --print_freq 20 \
    --t_length 3 \
    --interval 1 \
    --visualize 0 \
    --visualize_input 0 \
    --object_detection 1 \
    -b 64  \
    --test_batch_size 64 \
    --workers_test 4 \
    --h 240 \
    --w 432 \
    --arch 'Unet_Free' \
    --encoder_arch 'Encoder_Free' \
    --decoder_arch 'Decoder_Free' \
    --dataset_type 'shanghaitech' \
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
    #--resume '/home/miaobo/project/anomaly_demo/ckpt/best_32shanghai0,1,2,3_Unet_shanghaitech_checkpoint.pth.tar'

# 384 640 HW 480 856
# 240 432/416 can divide 16/32