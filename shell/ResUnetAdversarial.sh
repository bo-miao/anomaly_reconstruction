#!/bin/bash
# python PATH
cd ..

gpus='0,1'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'ResUnetAdversarial'${gpus}
dist_url='tcp://127.0.0.1:72'${port}
echo ${dist_url}

python net.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --object_detection 0 \
    --print_freq 20 \
    --t_length 3 \
    --interval 1 \
    --visualize 0 \
    --visualize_input 0 \
    -b 64  \
    --test_batch_size 64 \
    --workers_test 4 \
    --h 256 \
    --w 256 \
    --discriminator '' \
    --arch 'ResUnetAdversarial' \
    --encoder_arch 'Encoder_Free' \
    --decoder_arch 'Decoder_Free' \
    --dataset_type 'airs_anomaly2' \
    --label 1 \
    --training_folder 'training/frames' \
    --testing_folder 'testing/frames' \
    --label_folder 'label' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 2 \
    --epochs 100 \
    --is_amp 1 \
    --optim 'adamW' \
    #--evaluate \
    #--demo 'video|/data/miaobo/script/video2/Fall_0004.mp4' \
    #--resume '/home/miaobo/project/anomaly_demo2/ckpt/best_24Unet_Free_Adversarial_Classifier6,7_Unet_Free_Adversarial_Classifier__airs_anomaly2_checkpoint.pth.tar' \

