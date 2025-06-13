#!/bin/bash

#Set your dataset name and number of domains
# DATASET="office-home"
# NUM_DOMAINS=4
# CUDA_DEVICE=2

# for ((i=0; i<$NUM_DOMAINS; i++))
# do
#   echo "Running domain $i for dataset $DATASET"
#   CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py \
#     --output train_output \
#     --dataset "$DATASET" \
#     --test_envs $i \
#     --batch_size 108
# done


CUDA_VISIBLE_DEVICES=1 python generate_adv_data.py --dataset PACS
CUDA_VISIBLE_DEVICES=1 python generate_adv_data.py --dataset office-home
CUDA_VISIBLE_DEVICES=1 python generate_adv_data.py --dataset VLCS
CUDA_VISIBLE_DEVICES=1 python train.py --output train_output --dataset office-home --test_envs 3
# CUDA_VISIBLE_DEVICES=1 python train.py --output train_output --dataset DomainNet --test_envs 4 --batch_size 64 # hold off
# CUDA_VISIBLE_DEVICES=1 python train.py --output train_output --dataset DomainNet --test_envs 5 --batch_size 64 # hold off