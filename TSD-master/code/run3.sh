#!/usr/bin/env bash


set -e                # stop on first error
set -u                # treat unset vars as errors

###############################################################################
# User-configurable section
###############################################################################
#DATASET="DomainNet"                      # PACS, office-home, …
GPU=3                             
BATCH=64                           # test-time batch size
#NET="resnet18"                      # must match the checkpoint
###############################################################################

# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset PACS --test_envs $GPU --seed 0
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset PACS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset PACS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset VLCS --test_envs $GPU --seed 0
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset VLCS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset VLCS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset office-home --test_envs $GPU --seed 0
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset office-home --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset office-home --test_envs $GPU --seed 2

# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU

# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU --seed 1

# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU --seed 2

wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/vql3gqoz
wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/5kvfupo7

# for DATASET in PACS VLCS office-home; do
#   for ALG in SHOT-IM; do #TTA3 (TSD BN, PL)?
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 20 40 60 80 100; do
#         echo "▶︎  Rate=$RATE"
#         CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#             --adapt_alg "$ALG" \
#             --dataset  "$DATASET" \
#             --attack_rate $RATE \
#             --test_envs $DOMAIN_IDX \
#             --batch_size $BATCH \
#             --steps 5 
#       done
#     done
#   done
# done

# for DATASET in PACS VLCS office-home; do
#   for ALG in PL; do #TTA3 (TSD BN, PL)?
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 20 40 60 80 100; do
#         echo "▶︎  Rate=$RATE"
#         CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#             --adapt_alg "$ALG" \
#             --dataset  "$DATASET" \
#             --attack_rate $RATE \
#             --test_envs $DOMAIN_IDX \
#             --batch_size $BATCH \
#             --steps 5 
#       done
#     done
#   done
# done
