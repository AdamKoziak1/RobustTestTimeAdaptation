# coding=utf-8
"""
check_masks.py

Go through all mask files under:
    <adv_root>/<dataset>/masks/<rate>/
and print, for each (domain, mask index), the expected attack percentage
(via <rate> folder name) and the actual percentage in the .npy file.

Usage:
    python check_masks.py --adv_root ../../datasets_adv --dataset PACS
"""

import os
import argparse
import numpy as np

def main(args):
    adv_root = args.adv_root
    dataset  = args.dataset

    masks_root = os.path.join(adv_root, dataset, "masks")
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f"Masks directory not found: {masks_root}")

    print(f"\nChecking masks under: {masks_root}\n")
    header = f"{'Rate (%)':>8s} | {'Domain':>20s} | {'Mask #':>6s} | {'Actual (%)':>10s}"
    print(header)
    print("-" * len(header))

    # Iterate over each rate directory (e.g. “50”, “60”, …)
    for rate_dir in sorted(os.listdir(masks_root), key=lambda x: int(x) if x.isdigit() else x):
        rate_path = os.path.join(masks_root, rate_dir)
        if not os.path.isdir(rate_path) or not rate_dir.isdigit():
            continue

        expected_rate = int(rate_dir)
        # For each mask file: “<domain>_mask_<k>.npy”
        for fname in sorted(os.listdir(rate_path)):
            if not fname.endswith(".npy"):
                continue

            parts = fname.split("_mask_")
            if len(parts) != 2 or not parts[1].endswith(".npy"):
                continue

            domain = parts[0]
            mask_idx_str = parts[1].replace(".npy", "")
            try:
                mask_idx = int(mask_idx_str)
            except ValueError:
                continue

            mask_path = os.path.join(rate_path, fname)
            mask = np.load(mask_path)
            total = mask.size
            attacked = int(mask.sum())
            actual_rate = (attacked / total) * 100.0

            print(f"{expected_rate:8d} | {domain:20s} | {mask_idx:6d} | {actual_rate:10.2f} | {total}")

    print("\nDone.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Verify expected vs. actual percentages in attack masks."
    )
    parser.add_argument(
        "--adv_root",
        type=str,
        default="../../datasets_adv",
        help="Root folder that contains <dataset>/masks/",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default='PACS',
        help="Name of the dataset (e.g. PACS, office-home, etc.)",
    )
    args = parser.parse_args()
    main(args)
