#!/bin/bash
# python PATH
cd ..

gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"

port=$(date "+%S")
suffix=${port}'0to1classification_ucf'${gpus}
dist_url='tcp://127.0.0.1:72'${port}
echo ${dist_url}

python net_classification.py \
    --multiprocessing_distributed \
    --suffix ${suffix} \
    --dist_url ${dist_url} \
    --object_detection 0 \
    --print_freq 20 \
    --t_length 3 \
    --interval 1 \
    --visualize 0 \
    --visualize_input 0 \
    -b 256  \
    --test_batch_size 256 \
    --workers_test 4 \
    --h 256 \
    --w 256 \
    --arch 'ResNet50' \
    --dataset_type 'ucf' \
    --label 1 \
    --training_folder 'training_simple/frames' \
    --testing_folder 'testing_simple/frames' \
    --label_folder 'label' \
    --dataset_path  '/data/miaobo' \
    --gpu 0 \
    --eval_per_epoch 2 \
    --epochs 40 \
    --is_amp 1 \
    --optim 'sgd' \
    --lr 0.1 \
    --lr_mode 'step' \
    --resume '/home/miaobo/project/anomaly_demo2/ckpt/best_14classification_ucf2,3,4,5_ResNet50__ucf_checkpoint.pth.tar' \
    --evaluate \
    --demo 'video|/data/miaobo/ucf/video/Explosion013_x264.mp4' \

