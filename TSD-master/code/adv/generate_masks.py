# coding=utf-8
"""
Generate random attack indicator vectors for every domain.

Layout it expects
-----------------
datasets_adv/{DATASET}/
    clean/{DOMAIN}/0.pt … n.pt          ← saved by generate_adv_data.py
    {CONFIG_ID}/{DOMAIN}/…              ← adversarial tensors
    masks/{RATE}/                       ← created here
        {DOMAIN}_mask_{K}.npy
            K ∈ {0…4}   (five masks)
            RATE ∈ {50,60,70,80,90,100}
"""

import argparse, os, numpy as np, random
from tqdm import tqdm

def one_mask(num_samples: int, attack_pct: int) -> np.ndarray:
    """Return Boolean indicator of length = num_samples with given % of 1s."""
    num_attacked = round(num_samples * attack_pct / 100)
    idx = random.sample(range(num_samples), num_attacked)
    vec = np.zeros(num_samples, dtype=bool)
    vec[idx] = True
    return vec

def main(args):
    random.seed(args.seed)
    np.random.seed(args.seed)

    ds_root = os.path.join(args.adv_root, args.dataset)
    clean_root = os.path.join(ds_root, "clean")

    # every folder under clean/ is a domain
    domains = sorted(d for d in os.listdir(clean_root)
                     if os.path.isdir(os.path.join(clean_root, d)))

    # count samples per domain once
    sizes = {d: len(os.listdir(os.path.join(clean_root, d))) for d in domains}

    print(f"Found {len(domains)} domains:")
    for d, n in sizes.items():
        print(f"  {d:<20s}: {n} samples")

    for rate in range(0, 101, 10):           # 50,60,…,100
        rate_dir = os.path.join(ds_root, "masks", str(rate))
        os.makedirs(rate_dir, exist_ok=True)

        for dom in domains:
            n = sizes[dom]
            for k in range(args.n_vec):
                vec = one_mask(n, rate)
                out = os.path.join(rate_dir, f"{dom}_mask_{k}.npy")
                np.save(out, vec)

    print("✓ All indicator vectors written.")

if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--adv_root", default="../../datasets_adv",
                    help="Root that contains the <dataset> folder")
    pa.add_argument("--dataset",   default="PACS")
    pa.add_argument("--n_vec",     type=int, default=5,
                    help="how many random vectors per rate")
    pa.add_argument("--seed",      type=int, default=0)
    cfg = pa.parse_args()
    main(cfg)
