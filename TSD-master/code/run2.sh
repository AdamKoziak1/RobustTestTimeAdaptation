#!/usr/bin/env bash
set -e                # stop on first error
set -u                # treat unset vars as errors

###############################################################################
# User-configurable section
###############################################################################
DATASET="VLCS"                      # PACS, office-home, …
GPU=2                            
BATCH=108                           # test-time batch size
#NET="resnet18"                      # must match the checkpoint
###############################################################################

export CUDA_VISIBLE_DEVICES=$GPU

python generate_adv_data.py --dataset "$DATASET"
python adv/generate_masks.py --dataset "$DATASET"

for ALG in Tent T3A; do #TTA3 (TSD BN, PL)?
  for DOMAIN_IDX in 1 2 3; do
    for RATE in 0 10 20 30 40 50 60 70 80 90 100; do
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
