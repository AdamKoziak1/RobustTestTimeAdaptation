#!/usr/bin/env bash
set -e                # stop on first error
set -u                # treat unset vars as errors

###############################################################################
# User-configurable section
###############################################################################
ALG="Tent"                          # one of: Tent, TSD, T3A, BN, PL, …
DATASET="PACS"                      # PACS, office-home, …
DOMAIN_IDX=0                        # 0=art_painting, 1=cartoon, … for PACS
#CHECKPOINT="ABS/PATH/TO/model.pkl"  # pre-trained source model
DATA_ROOT="/home/adam/Downloads/RobustTestTimeAdaptation/datasets/"   # e.g. …/RobustTestTimeAdaptation/datasets/
ADV_ROOT="ABS/PATH/TO/datasets_adv" # root that holds the masks & tensors
GPU=2                               
NET="resnet18"                      # must match the checkpoint
BATCH=108                           # test-time batch size
OUTDIR="ABS/PATH/TO/tta_results"    # where logs & models will be written
###############################################################################

export CUDA_VISIBLE_DEVICES=$GPU

for RATE in 0 10 20 30 40 50 60 70 80 90 100; do
  for MASK in 0 1 2 3 4; do
    echo "▶︎  Rate=$RATE  Mask=$MASK"
    CUDA_VISIBLE_DEVICES=$GPU python unsupervise_adapt.py \
        --adapt_alg "$ALG" \
        --dataset  "$DATASET" \
        --attack_rate $RATE \
        --mask_id $MASK \
        #--net "$NET" \
        #--batch_size $BATCH \
        #--lr 1e-4 \
        #--test_envs $DOMAIN_IDX \
        #--attack linf_eps-8_steps-20 \
        #--data_file "" \
        #--data_dir "$DATA_ROOT/$DATASET" \
        #--attack_data_dir "$ADV_ROOT" \
        #--pretrain_dir "$CHECKPOINT" \
        #--output "$OUTDIR" \
        #--gpu_id "$GPU" \
        #--seed 0 --source_seed 0 \
        #--steps 1  \
        #--episodic  \
        #--filter_K 100
  done
done
