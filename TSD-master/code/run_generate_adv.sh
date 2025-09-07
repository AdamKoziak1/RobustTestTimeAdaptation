python generate_adv_data.py --dataset "PACS" --test_envs 0 --seed 0 --attack l2

# for DATASET in PACS VLCS office-home; do
#     for SEED in 0 1 2; do
#         python generate_adv_data.py --dataset "$DATASET" --test_envs $CUDA_VISIBLE_DEVICES --seed $SEED --attack linf --max_epoch 0
#     done
# done

# for DATASET in PACS VLCS office-home; do
#     for SEED in 0 1 2; do
#         python generate_adv_data.py --dataset "$DATASET" --test_envs $CUDA_VISIBLE_DEVICES --seed $SEED --attack l2 --max_epoch 0 --eps 16 --steps 100 --alpha_adv 16
#     done
# done

for DATASET in PACS VLCS office-home; do
    for SEED in 0 1 2; do
        for TEST_ENV in 0 1 2 3; do
            python generate_adv_data.py --dataset "$DATASET" --test_envs $TEST_ENV --seed $SEED --attack linf --max_epoch 0
            python generate_adv_data.py --dataset "$DATASET" --test_envs $TEST_ENV --seed $SEED --attack l2 --max_epoch 0 --eps 16 --steps 100 --alpha_adv 16
        done
    done
done