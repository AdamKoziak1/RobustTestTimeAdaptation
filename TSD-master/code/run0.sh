#!/usr/bin/env bash
set -e                # stop on first error
set -u                # treat unset vars as errors

###############################################################################
# User-configurable section
###############################################################################
#DATASET="PACS"                      # PACS, office-home, …
GPU=0                            
BATCH=64                           # test-time batch size
#NET="resnet18"                      # must match the checkpoint
###############################################################################

export CUDA_VISIBLE_DEVICES=$GPU


# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU --seed 2


# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset PACS --test_envs $GPU --seed 0
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset PACS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset PACS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset VLCS --test_envs $GPU --seed 0
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset VLCS --test_envs $GPU --seed 1
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset VLCS --test_envs $GPU --seed 2
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset office-home --test_envs $GPU --seed 0
# CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset office-home --test_envs $GPU --seed 1
#CUDA_VISIBLE_DEVICES=$GPU python generate_adv_data.py --dataset office-home --test_envs $GPU --seed 2

for DATASET in PACS VLCS office-home; do
  for ALG in TTA3; do #TTA3 (TSD BN, PL)?
    for DOMAIN_IDX in 0 1 2 3; do
      for RATE in 0 20 40 60 80 100; do
        echo "▶︎  Rate=$RATE"
        CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
            --adapt_alg "$ALG" \
            --dataset  "$DATASET" \
            --attack_rate $RATE \
            --test_envs $DOMAIN_IDX \
            --batch_size $BATCH \
            --steps 1 \
            --lambda1 0.0 \
            --lambda2 0.0 \
            --lambda3 10.0 \
            --cr_type "l2" \
            --lr 0.00001
      done
    done
  done
done
wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/xjf2x6c1

# for DATASET in PACS VLCS office-home; do
#   for ALG in Tent; do #TTA3 (TSD BN, PL)?
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 20 40 60 80 100; do
#         echo "▶︎  Rate=$RATE"
#         CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#             --adapt_alg "$ALG" \
#             --dataset  "$DATASET" \
#             --attack_rate $RATE \
#             --test_envs $DOMAIN_IDX \
#             --batch_size $BATCH \
#             --steps 5 \
#             --lr 0.001
#       done
#     done
#   done
# done
wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/glmg6syc
# for DATASET in PACS VLCS office-home; do
#   for ALG in TSD; do #TTA3 (TSD BN, PL)?
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 20 40 60 80 100; do
#         echo "▶︎  Rate=$RATE"
#         CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#             --adapt_alg "$ALG" \
#             --dataset  "$DATASET" \
#             --attack_rate $RATE \
#             --test_envs $DOMAIN_IDX \
#             --batch_size $BATCH \
#             --steps 10 
#       done
#     done
#   done
# done

