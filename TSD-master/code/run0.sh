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
  for ALG in ERM; do #TTA3 (TSD BN, PL)?
    for DOMAIN_IDX in 0 1 2 3; do
      for RATE in 0 20 40 60 80 100; do
        echo "▶︎  Rate=$RATE"
        CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
            --adapt_alg "$ALG" \
            --dataset  "$DATASET" \
            --attack_rate $RATE \
            --test_envs $DOMAIN_IDX \
            --batch_size $BATCH \
            --steps 10 
      done
    done
  done
done

for DATASET in PACS VLCS office-home; do
  for ALG in TSD; do #TTA3 (TSD BN, PL)?
    for DOMAIN_IDX in 0 1 2 3; do
      for RATE in 0 20 40 60 80 100; do
        echo "▶︎  Rate=$RATE"
        CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
            --adapt_alg "$ALG" \
            --dataset  "$DATASET" \
            --attack_rate $RATE \
            --test_envs $DOMAIN_IDX \
            --batch_size $BATCH \
            --steps 10 
      done
    done
  done
done



## FOR AFTER FIX L2

# for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#   for DATASET in PACS VLCS office-home; do
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 20 40 60 80 100; do
#         for MASK in 0 1 2 3 4; do
#           echo "▶︎  Rate=$RATE  Mask=$MASK"
#           CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
#               --adapt_alg "$ALG" \
#               --dataset  "$DATASET" \
#               --attack_rate $RATE \
#               --mask_id $MASK \
#               --test_envs $DOMAIN_IDX \
#               --batch_size $BATCH \
#               --lambda1 0.0 \
#               --lambda2 1.0 \
#               --lambda3 1.0 \
#               --cr_type l2
#         done
#       done
#     done
#   done
# done

# for ALG in TTA3; do #TTA3 (TSD BN, PL)?
#   for DATASET in PACS VLCS office-home; do
#     for DOMAIN_IDX in 0 1 2 3; do
#       for RATE in 0 20 40 60 80 100; do
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
#               --lambda3 1.0 \
#               --cr_type cosine
#         done
#       done
#     done
#   done
# done


## extra runs

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
#               --lambda1 0.0 \
#               --lambda2 1.0 \
#               --lambda3 0.0 
#         done
#       done
#     done
#   done
# done



# for ALG in Tent; do #TTA3 (TSD BN, PL)?
#   for DOMAIN_IDX in 1 2 3; do
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