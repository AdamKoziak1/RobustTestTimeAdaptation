#!/usr/bin/env python3
"""
Batch evaluation of test-time adaptation (TTA) across multiple datasets and domains.

python unsupervised_adapt_dataset.py \
    --adapt_alg TTA3 --steps 10 --lam_cr 10 --cr_type l2 --attack clean
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
ATTACK_RATES = [0, 100]

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
    
    #run_name = f"{args.adapt_alg}_steps{args.steps}_lr{args.lr}_input{args.svd_input_rank_ratio}_feat{args.svd_feat_rank_ratio}_svdmode-{args.svd_feat_mode}"
    run_name = f"{args.adapt_alg}"

    wandb.init(name=run_name, config=vars(args), project="tta3_sweeps")

    start = time.time()
    results = {}
    for rate in ATTACK_RATES:
        args.attack_rate = rate
        accs = []
        for dataset in DATASETS:
            for dom in TEST_DOMAINS:
                args.dataset   = dataset
                args.test_envs = [dom]
                args = img_param_init(args)
                args.data_dir = os.path.join(args.data_file, "datasets", dataset)

                acc = evaluate_domain(args)
                print(f"{dataset:12s}  dom {dom} rate {rate}%: {acc:6.2f}%")
                accs.append(acc)
        mean = round(statistics.mean(accs), 2)
        results[rate] = mean

    for rate in ATTACK_RATES:
        mean = results[rate]
        wandb.log({f"acc_{rate}": mean}, commit=False)
    
    dur   = time.time() - start
    log_args(args, dur)

    wandb.finish()


if __name__ == "__main__":
    main()
