#!/bin/bash

#Set your dataset name and number of domains
DATASET="office-home"
NUM_DOMAINS=4
CUDA_DEVICE=2

for ((i=0; i<$NUM_DOMAINS; i++))
do
  echo "Running domain $i for dataset $DATASET"
  CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python train.py \
    --output train_output \
    --dataset "$DATASET" \
    --test_envs $i \
    --batch_size 108
done


# CUDA_VISIBLE_DEVICES=2 python train.py --output train_output --dataset DomainNet --test_envs 0
# CUDA_VISIBLE_DEVICES=2 python train.py --output train_output --dataset DomainNet --test_envs 1
# CUDA_VISIBLE_DEVICES=2 python train.py --output train_output --dataset DomainNet --test_envs 2
# CUDA_VISIBLE_DEVICES=2 python train.py --output train_output --dataset DomainNet --test_envs 3
# CUDA_VISIBLE_DEVICES=2 python train.py --output train_output --dataset DomainNet --test_envs 4
# CUDA_VISIBLE_DEVICES=2 python train.py --output train_output --dataset DomainNet --test_envs 5

#CUDA_VISIBLE_DEVICES=3 python train.py --output train_output --dataset VLCS --test_envs 0