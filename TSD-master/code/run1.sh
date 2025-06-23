#!/usr/bin/env bash
set -e                # stop on first error
set -u                # treat unset vars as errors

###############################################################################
# User-configurable section
###############################################################################
#DATASET="office-home"                      # PACS, office-home, …
GPU=1                            
BATCH=64                           # test-time batch size
#NET="resnet18"                      # must match the checkpoint
###############################################################################

export CUDA_VISIBLE_DEVICES=$GPU

for DATASET in PACS VLCS office-home; do
  for ALG in PL; do #TTA3 (TSD BN, PL)?
    for DOMAIN_IDX in 0 1 2 3; do
      for RATE in 0 100; do
        for MASK in 0; do
          echo "▶︎  Rate=$RATE  Mask=$MASK"
          CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
              --adapt_alg "$ALG" \
              --dataset  "$DATASET" \
              --attack_rate $RATE \
              --mask_id $MASK \
              --test_envs $DOMAIN_IDX \
              --batch_size $BATCH 
        done
      done
      for RATE in 20 40 60 80; do
        for MASK in 0 1 2 3 4; do
          echo "▶︎  Rate=$RATE  Mask=$MASK"
          CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
              --adapt_alg "$ALG" \
              --dataset  "$DATASET" \
              --attack_rate $RATE \
              --mask_id $MASK \
              --test_envs $DOMAIN_IDX \
              --batch_size $BATCH 
        done
      done
    done
  done
done


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
#               --lambda2 0.0 \
#               --lambda3 1.0 \
#               --cr_type cosine;
#         done
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
#               --cr_type l2
#         done
#       done
#     done
#   done
# done


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
#               --cr_type cosine
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
#               --lambda3 0.0 
#         done
#       done
#     done
#   done
# done

#HELD OFF WITH EXTRA RATES

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
#               --lambda2 0.0 \
#               --lambda3 1.0 \
#               --cr_type "l2"
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
#               --lambda1 0.0 \
#               --lambda2 1.0 \
#               --lambda3 1.0 \
#               --cr_type cosine
#         done
#       done
#     done
#   done
# done
