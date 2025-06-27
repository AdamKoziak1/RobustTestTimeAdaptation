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

CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU
CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU
CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU

CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU --seed 1
CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU --seed 1
CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU --seed 1

CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset PACS --test_envs $GPU --seed 2
CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset VLCS --test_envs $GPU --seed 2
CUDA_VISIBLE_DEVICES=$GPU python train.py --output train_output --dataset office-home --test_envs $GPU --seed 2

# for DATASET in PACS VLCS office-home; do
#   for ALG in Tent TTA3 TSD PL SHOT-IM; do #TTA3 (TSD BN, PL)?
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0; do
#         for MASK in 0; do
#           echo "▶︎  Rate=$RATE  Mask=$MASK"
#           CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#               --adapt_alg "$ALG" \
#               --dataset  "$DATASET" \
#               --attack_rate $RATE \
#               --mask_id $MASK \
#               --test_envs $DOMAIN_IDX \
#               --batch_size $BATCH \
#               --lambda1 10.1
#         done
#       done
#       # for RATE in 20 40 60 80; do
#       #   for MASK in 0 1 2 3 4; do
#       #     echo "▶︎  Rate=$RATE  Mask=$MASK"
#       #     CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#       #         --adapt_alg "$ALG" \
#       #         --dataset  "$DATASET" \
#       #         --attack_rate $RATE \
#       #         --mask_id $MASK \
#       #         --test_envs $DOMAIN_IDX \
#       #         --batch_size $BATCH \
#       #         --episodic
#       #   done
#       # done
#     done
#   done
# done



## ----------------------------------------- DOMAINNET STUFF -----------------------------------------------------------------------
# export CUDA_VISIBLE_DEVICES=$GPU

# CUDA_VISIBLE_DEVICES=3 python train.py --output train_output --dataset DomainNet --test_envs 4 --batch_size 64
# CUDA_VISIBLE_DEVICES=3 python train.py --output train_output --dataset DomainNet --test_envs 5 --batch_size 64

# CUDA_VISIBLE_DEVICES=3 python generate_adv_data.py --dataset DomainNet
# CUDA_VISIBLE_DEVICES=3 python generate_masks.py --dataset DomainNet


# set -e                # stop on first error
# set -u                # treat unset vars as errors

# ###############################################################################
# # User-configurable section
# ###############################################################################
# DATASET="DomainNet"                      # PACS, office-home, …
# GPU=3                             
# BATCH=64                           # test-time batch size
# #NET="resnet18"                      # must match the checkpoint
# ###############################################################################

# export CUDA_VISIBLE_DEVICES=$GPU
# for ALG in Tent T3A; do #TTA3 (TSD BN, PL)?
#   for DOMAIN_IDX in 0 1 2 3; do
#     for RATE in 0 10 20 30 40 50 60 70 80 90 100; do
#       for MASK in 0 1 2 3 4; do
#         echo "▶︎  Rate=$RATE  Mask=$MASK"
#         CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#             --adapt_alg "$ALG" \
#             --dataset  "$DATASET" \
#             --attack_rate $RATE \
#             --mask_id $MASK \
#             --test_envs $DOMAIN_IDX \
#             --batch_size $BATCH
#       done
#     done
#   done
# done
# ----------------------------------------- DOMAINNET STUFF -----------------------------------------------------------------------


# ----------------------------------------- TO RUN AFTER -----------------------------------------------------------------------
# TO RUN:
# set -e                # stop on first error
# set -u                # treat unset vars as errors

# ###############################################################################
# # User-configurable section
# ###############################################################################
# #DATASET="VLCS"                      # PACS, office-home, …
# GPU=3                            
# BATCH=64                           # test-time batch size
# #NET="resnet18"                      # must match the checkpoint
# ###############################################################################
# for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#   for DATASET in PACS VLCS office-home; do
#     for DOMAIN_IDX in 0 1 2 3; do 
#       for RATE in 0 10 20 30 40 50 60 70 80 90 100; do
#         for MASK in 0 1 2 3 4; do
#           echo "▶︎  Rate=$RATE  Mask=$MASK"
#           CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#               --adapt_alg "$ALG" \
#               --dataset  "$DATASET" \
#               --attack_rate $RATE \
#               --mask_id $MASK \
#               --test_envs $DOMAIN_IDX \
#               --batch_size $BATCH \
#               --lambda1 1.0 \
#               --lambda2 1.0 \
#               --lambda3 0.0 
#         done
#       done
#     done
#   done
# done

# for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#   for DATASET in PACS VLCS office-home; do
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 10 20 30 40 50 60 70 80 90 100; do
#         for MASK in 0 1 2 3 4; do
#           echo "▶︎  Rate=$RATE  Mask=$MASK"
#           CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#               --adapt_alg "$ALG" \
#               --dataset  "$DATASET" \
#               --attack_rate $RATE \
#               --mask_id $MASK \
#               --test_envs $DOMAIN_IDX \
#               --batch_size $BATCH \
#               --lambda1 1.0 \
#               --lambda2 0.0 \
#               --lambda3 1.0 \
#               --cr_type "cosine"
#         done
#       done
#     done
#   done
# done

# for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#   for DATASET in PACS VLCS office-home; do
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 10 20 30 40 50 60 70 80 90 100; do
#         for MASK in 0 1 2 3 4; do
#           echo "▶︎  Rate=$RATE  Mask=$MASK"
#           CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#               --adapt_alg "$ALG" \
#               --dataset  "$DATASET" \
#               --attack_rate $RATE \
#               --mask_id $MASK \
#               --test_envs $DOMAIN_IDX \
#               --batch_size $BATCH \
#               --lambda1 1.0 \
#               --lambda2 0.0 \
#               --lambda3 1.0 \
#               --cr_type "l2"
#         done
#       done
#     done
#   done
# done