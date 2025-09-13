#!/usr/bin/env python3
"""
Batch evaluation of test-time adaptation (TTA) across multiple datasets and
domains.

For every (dataset, test_env) pair it
    1. loads the pre-trained ERM source model (seed 0)
    2. builds a test loader for the held-out domain (clean or attacked)
    3. wraps the model with the requested adaptation method (Tent, TTA3, …)
    4. runs adaptation over the entire test set (single pass)
    5. reports Top-1 accuracy

The logic for argument parsing, loader construction and adaptation wrappers is
reused from *unsupervise_adapt.py* to avoid duplication.

Example
-------
python unsupervised_adapt_dataset.py \
    --adapt_alg TTA3 --steps 10 --lambda3 10 --cr_type l2 --attack clean
"""

import os
import time
import statistics
import torch
from sklearn.metrics import accuracy_score
from unsupervise_adapt import (
    get_args,
    adapt_loader,
    make_adapt_model,
    log_args
)
from utils.util import set_random_seed, load_ckpt, img_param_init
from alg import alg
import wandb

DATASETS     = ["PACS", "VLCS", "office-home"]
TEST_DOMAINS = [0, 1]
SEED         = 0 

def evaluate_domain(args):
    dom_id = args.test_envs[0]

    ckpt_path = os.path.join(
        args.data_file, "train_output",
        args.dataset, f"test_{dom_id}", f"seed_{SEED}", "model.pkl")

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    erm_class  = alg.get_algorithm_class("ERM")
    base_model = erm_class(args).cuda().train()
    base_model = load_ckpt(base_model, ckpt_path)

    test_loader = adapt_loader(args)           
    adapt_model = make_adapt_model(args, base_model)

    preds, gts = [], []
    for xb, yb in test_loader:
        logits = adapt_model(xb.cuda())          
        preds.append(logits.detach().cpu())
        gts.append(yb)

    preds = torch.cat(preds).argmax(1).numpy()
    gts   = torch.cat(gts).numpy()
    acc   = 100.0 * accuracy_score(gts, preds)
    return acc


def main():
    args = get_args()

    args.seed = SEED 
    set_random_seed(args.seed)
    
    run_name = f"{args.adapt_alg}_drop{args.svd_drop_k}_{args.steps}steps_lr{args.lr}"
    wandb.init(name=run_name, config=vars(args))

    start = time.time()
    accs = []
    for dataset in DATASETS:
        for dom in TEST_DOMAINS:
            args.dataset   = dataset
            args.test_envs = [dom]
            args = img_param_init(args)
            args.data_dir = os.path.join(args.data_file, "datasets", dataset)

            acc = evaluate_domain(args)
            print(f"{dataset:12s}  dom {dom}: {acc:6.2f}%")
            accs.append(acc)

    print("\n==================  Summary  ==================")
    mean = round(statistics.mean(accs), 2)
    dur   = time.time() - start

    log_args(args)

    wandb.log({"acc_mean": mean, 
               "time_taken_s": dur})
    wandb.finish()


if __name__ == "__main__":
    main()
