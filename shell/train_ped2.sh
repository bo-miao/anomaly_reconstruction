#!/bin/bash
# python PATH
cd ..

gpus='0,1'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'object'${gpus}
dist_url='tcp://127.0.0.1:73'${port}
echo ${dist_url}

python net.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --t_length 3 \
    --interval 1 \
    -b 32  \
    --test_batch_size 32 \
    --object_detection 1 \
    --visualize 0 \
    --visualize_input 0 \
    --h 240 \
    --w 360 \
    --path_h 64 \
    --path_w 64 \
    --discriminator '' \
    --arch 'Unet_Free_Adversarial' \
    --encoder_arch 'Encoder_Free' \
    --decoder_arch 'Decoder_Free' \
    --dataset_type 'ped2' \
    --label 1 \
    --training_folder 'training/frames' \
    --testing_folder 'testing/frames' \
    --label_folder 'label' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 3 \
    --epochs 30 \
    --is_amp 1 \
    --optim 'adamW' \
    #--resume '/home/miaobo/project/anomaly_demo/ckpt/best_object1,2,3,4_Unet_Light3_ped2_checkpoint.pth.tar'
