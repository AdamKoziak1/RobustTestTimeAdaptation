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
wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/xn1jhseg
wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/28x2bxgo

# # erm     185
# for DATASET in PACS VLCS office-home; do
#   for ALG in ERM; do 
#     for RATE in 0 20 40 60 80 100; do
#     echo "▶︎  Rate=$RATE"
#     CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#         --adapt_alg "$ALG" \
#         --dataset  "$DATASET" \
#         --attack_rate $RATE \
#         --test_envs $GPU \
#         --batch_size $BATCH \
#         --steps 1 \
#         --svd_drop_k 185
#     done
#   done
# done

# # tent    160 185
# for DATASET in PACS VLCS office-home; do
#   for ALG in Tent; do 
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 160 185; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 3 \
#           --lr 0.001 \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

# # PL      130 185
# for DATASET in PACS VLCS office-home; do
#   for ALG in PL; do 
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 130 185; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 3 \
#           --lr 0.00001 \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

# #SHOT-IM 145 185
# for DATASET in PACS VLCS office-home; do
#   for ALG in SHOT-IM; do 
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 145 185; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 1 \
#           --lr 0.00001 \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

# # T3A     150 185
# for DATASET in PACS VLCS office-home; do
#   for ALG in T3A; do 
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 150 185; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 1 \
#           --lr 0.00001 \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

# # TSD     125 190
# for DATASET in PACS VLCS office-home; do
#   for ALG in TSD; do 
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 125 190; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 10 \
#           --lr 0.0001 \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

# # TTA3-MI 150 185
# for DATASET in PACS VLCS office-home; do
#   for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 150 185; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 1 \
#           --lambda1 0.0 \
#           --lambda2 0.0 \
#           --lambda3 0.0 \
#           --lr 0.001 \
#           --update_param "affine" \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

# # TTA3-CR 160 185
# for DATASET in PACS VLCS office-home; do
#   for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#     for RATE in 0 20 40 60 80 100; do
#       for SVD in 160 185; do
#       echo "▶︎  Rate=$RATE"
#       CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#           --adapt_alg "$ALG" \
#           --dataset  "$DATASET" \
#           --attack_rate $RATE \
#           --test_envs $GPU \
#           --batch_size $BATCH \
#           --steps 1 \
#           --lambda1 0.0 \
#           --lambda2 0.0 \
#           --lambda3 20.0 \
#           --lr 0.001 \
#           --cr_start 0 \
#           --update_param "affine" \
#           --svd_drop_k $SVD
#       done
#     done
#   done
# done

