#!/usr/bin/env python3
"""
Batch evaluation of test-time adaptation (TTA) across multiple datasets and domains.

python unsupervised_adapt_dataset.py \
    --adapt_alg TTA3 --steps 10 --lam_cr 10 --cr_type l2 --attack clean
"""

import copy
import os
import statistics
import sys
import time
import torch
from sklearn.metrics import accuracy_score
from unsupervise_adapt import (
    get_args,
    adapt_loader,
    make_adapt_model,
    log_args,
    resolve_source_checkpoint,
    wrap_with_input_defense,
)
from utils.adv_attack import build_attack_transform, pgd_attack
from utils.util import set_random_seed, load_ckpt, img_param_init
from alg import alg
import wandb

DATASETS     = ["PACS", "VLCS", "office-home"]
TEST_DOMAINS = [1]
SEED         = 1 
ATTACK_RATES = [0, 100]


def _cli_overrides(tokens):
    overrides = set()
    for token in tokens:
        if not token.startswith("--"):
            continue
        key = token[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        overrides.add(key.replace("-", "_"))
    return overrides

def evaluate_domain(args):
    dom_id = args.test_envs[0]

    ckpt_path = resolve_source_checkpoint(args, dom_id)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    erm_class  = alg.get_algorithm_class("ERM")
    base_model = erm_class(args).cuda().train()
    base_model = load_ckpt(base_model, ckpt_path)

    test_loader = adapt_loader(args)           
    adapt_model = make_adapt_model(args, base_model)

    attack_model = None
    if args.attack != "clean" and args.attack_source in ("on_the_fly", "live"):
        if args.attack_source == "on_the_fly":
            attack_model = copy.deepcopy(base_model)
            attack_model = wrap_with_input_defense(attack_model, args)
            attack_model = attack_model.cuda().eval()
            for param in attack_model.parameters():
                param.requires_grad_(False)
        else:
            attack_model = getattr(adapt_model, "model", None)

    attack_transform = None
    if attack_model is not None and args.attack_fft_rho < 1.0:
        attack_transform = build_attack_transform(
            fft_rho=args.attack_fft_rho,
            fft_alpha=args.attack_fft_alpha,
            device=torch.device("cuda"),
        )
    attack_rng = torch.Generator(device="cuda")
    attack_rng.manual_seed(args.seed)

    preds, gts = [], []
    for xb, yb in test_loader:
        xb = xb.cuda()
        yb = yb.cuda()
        if attack_model is not None and args.attack_rate > 0:
            if args.attack_rate >= 100:
                xb = pgd_attack(
                    attack_model,
                    xb,
                    yb,
                    args.attack_eps / 255.0,
                    args.attack_alpha / 255.0,
                    args.attack_steps,
                    norm=args.attack_norm,
                    input_transform=attack_transform,
                )
            else:
                mask = torch.rand(
                    (xb.size(0),),
                    generator=attack_rng,
                    device=xb.device,
                ) < (args.attack_rate / 100.0)
                if mask.any():
                    adv = pgd_attack(
                        attack_model,
                        xb[mask],
                        yb[mask],
                        args.attack_eps / 255.0,
                        args.attack_alpha / 255.0,
                        args.attack_steps,
                        norm=args.attack_norm,
                        input_transform=attack_transform,
                    )
                    xb = xb.clone()
                    xb[mask] = adv
        logits = adapt_model(xb)          
        preds.append(logits.detach().cpu())
        gts.append(yb.detach().cpu())

    preds = torch.cat(preds).argmax(1).numpy()
    gts   = torch.cat(gts).numpy()
    acc   = 100.0 * accuracy_score(gts, preds)
    return acc


def main():
    args = get_args()
    cli_overrides = _cli_overrides(sys.argv[1:])
    # if "attack_rate" in cli_overrides:
    #     attack_rates = [args.attack_rate]
    # else:
    #     attack_rates = ATTACK_RATES
    attack_rates = ATTACK_RATES
    args.seed = SEED 
    set_random_seed(args.seed)
    
    #run_name = f"{args.adapt_alg}_steps{args.steps}_lr{args.lr}_input{args.svd_input_rank_ratio}_feat{args.svd_feat_rank_ratio}_svdmode-{args.svd_feat_mode}"
    run_name = f"{args.adapt_alg}"

    wandb.init(name=run_name, config=vars(args), project="tta3_sweeps")

    start = time.time()
    results = {}
    for rate in attack_rates:
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

    for rate in attack_rates:
        mean = results[rate]
        wandb.log({f"acc_{rate}": mean}, commit=False)
    
    dur   = time.time() - start
    log_args(args, dur)

    wandb.finish()


if __name__ == "__main__":
    main()
