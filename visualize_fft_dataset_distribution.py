#!/usr/bin/env python3
"""Aggregate FFT statistics across a dataset split for clean vs attacked samples."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import numpy as np

from visualize_fft_pipeline import (
    compute_fft_stages,
    config_id,
    load_image,
    log_scale,
    normalize,
    resolve_domain_name,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="PACS", help="Dataset name (e.g. PACS, VLCS)")
    parser.add_argument("--test-env", default=0, type=int, help="Held-out domain index")
    parser.add_argument(
        "--attack",
        default="linf_eps-8.0_steps-20",
        type=str,
        help="Attack configuration (e.g., linf_eps-8.0_steps-20_fft-spatial_k-0.75) or 'clean'.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed identifying the saved tensors")
    parser.add_argument("--data-root", default="datasets", help="Root directory of the original datasets")
    parser.add_argument("--adv-root", default="datasets_adv", help="Root directory containing clean/attacked tensors")
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of spatial frequencies to keep when reconstructing",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=0,
        help="Maximum number of samples to aggregate (0 = use all available)",
    )
    parser.add_argument(
        "--sample-stride",
        type=int,
        default=20,
        help="Process every Nth sample index to thin large datasets",
    )
    parser.add_argument("--dpi", type=int, default=400, help="Figure DPI")
    parser.add_argument("--output-dir", default="fft_snapshots", help="Directory for saved figures")
    return parser.parse_args()


def collect_indices(clean_dir: Path, attacked_dir: Path) -> List[int]:
    def available_ids(folder: Path) -> Sequence[int]:
        ids: List[int] = []
        for file in folder.iterdir():
            if file.suffix == ".pt":
                stem = file.stem
                try:
                    ids.append(int(stem))
                except ValueError:
                    continue
        return ids

    clean_ids = set(available_ids(clean_dir))
    attacked_ids = set(available_ids(attacked_dir))
    common = sorted(clean_ids & attacked_ids)
    if not common:
        raise RuntimeError(
            f"No overlapping samples between\n  {clean_dir}\n  {attacked_dir}"
        )
    return common


def aggregate(
    args: argparse.Namespace,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str]:
    data_root = Path(args.data_root).expanduser()
    adv_root = Path(args.adv_root).expanduser()
    domain_name = resolve_domain_name(data_root, adv_root, args.dataset, args.test_env, args.seed)

    attack_folder = f"resnet18_{args.attack}"
    base_dir = adv_root / f"seed_{args.seed}" / args.dataset

    clean_dir = base_dir / "clean" / domain_name
    attacked_dir = base_dir / attack_folder / domain_name

    if not clean_dir.is_dir():
        raise FileNotFoundError(f"Missing clean directory: {clean_dir}")
    if not attacked_dir.is_dir():
        raise FileNotFoundError(f"Missing attacked directory: {attacked_dir}")

    indices = collect_indices(clean_dir, attacked_dir)
    if args.sample_stride > 1:
        indices = indices[:: args.sample_stride]
    if args.max_samples > 0:
        indices = indices[: args.max_samples]
    if not indices:
        raise RuntimeError("No samples selected after applying stride/max filters")

    freq_clean_sum = None
    freq_attacked_sum = None
    recon_clean_sum = None
    recon_attacked_sum = None

    for idx in indices:
        clean_tensor = load_image(clean_dir / f"{idx}.pt")
        attacked_tensor = load_image(attacked_dir / f"{idx}.pt")

        clean_stages = compute_fft_stages(clean_tensor, args.keep_ratio)
        attacked_stages = compute_fft_stages(attacked_tensor, args.keep_ratio)

        if freq_clean_sum is None:
            shape_freq = clean_stages["shifted_abs"].shape
            shape_recon = clean_stages["reconstruction"].shape
            freq_clean_sum = np.zeros(shape_freq, dtype=np.float64)
            freq_attacked_sum = np.zeros(shape_freq, dtype=np.float64)
            recon_clean_sum = np.zeros(shape_recon, dtype=np.float64)
            recon_attacked_sum = np.zeros(shape_recon, dtype=np.float64)

        freq_clean_sum += clean_stages["shifted_abs"]
        freq_attacked_sum += attacked_stages["shifted_abs"]
        recon_clean_sum += clean_stages["reconstruction"]
        recon_attacked_sum += attacked_stages["reconstruction"]

    count = float(len(indices))
    freq_clean_mean = freq_clean_sum / count
    freq_attacked_mean = freq_attacked_sum / count
    recon_clean_mean = recon_clean_sum / count
    recon_attacked_mean = recon_attacked_sum / count

    return (
        freq_clean_mean,
        freq_attacked_mean,
        recon_clean_mean,
        recon_attacked_mean,
        domain_name,
    )


def visualize_distributions(
    freq_clean: np.ndarray,
    freq_attacked: np.ndarray,
    recon_clean: np.ndarray,
    recon_attacked: np.ndarray,
    args: argparse.Namespace,
    domain_name: str,
) -> None:
    freq_diff = log_scale(np.abs(freq_attacked.mean(axis=0) - freq_clean.mean(axis=0)))
    #freq_diff = np.abs(freq_attacked.mean(axis=0) - freq_clean.mean(axis=0))
    max_abs_freq = float(np.max(np.abs(freq_diff)))
    if max_abs_freq < 1e-8:
        max_abs_freq = 1.0

    recon_diff = recon_attacked - recon_clean
    recon_diff_rgb = recon_diff.transpose(1, 2, 0).astype(np.float32)
    max_abs_recon = float(np.max(np.abs(recon_diff_rgb)))
    if max_abs_recon < 1e-8:
        max_abs_recon = 1.0

    diff_cmap = mcolors.LinearSegmentedColormap.from_list(
        "darkbright_diff",
        [(0.0, "#0b132b"), (0.5, "#00060E"), (1.0, "#00f7ff")],
    )
    diff_norm = mcolors.TwoSlopeNorm(vmin=-max_abs_freq, vcenter=0.0, vmax=max_abs_freq)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    attack_slug = args.attack.replace(".", "p").replace("/", "-")
    meta = f"{args.dataset}_env{args.test_env}_seed{args.seed}_{domain_name}_attack-{attack_slug}"

    # Frequency difference figure -----------------------------------------
    fig_freq, ax_freq = plt.subplots(figsize=(4.2, 3.4))
    im0 = ax_freq.imshow(freq_diff, cmap=diff_cmap, norm=diff_norm)
    ax_freq.set_title("log|F| difference (adv - clean)")
    ax_freq.set_axis_off()
    cbar0 = fig_freq.colorbar(im0, ax=ax_freq, fraction=0.035, pad=0.02)
    cbar0.ax.tick_params(labelsize=8)
    fig_freq.suptitle(f"{args.dataset} env {args.test_env} – {args.attack}", fontsize=12)
    fig_freq.tight_layout(rect=[0, 0, 1, 0.97])
    freq_path = output_dir / f"dataset_fft_diff_freq_{meta}.png"
    fig_freq.savefig(freq_path, dpi=args.dpi)
    plt.close(fig_freq)

    # Reconstruction difference figure ------------------------------------
    recon_vis = 0.5 + recon_diff_rgb / (2.0 * max_abs_recon)
    fig_recon, ax_recon = plt.subplots(figsize=(4.2, 3.4))
    ax_recon.imshow(np.clip(recon_vis, 0.0, 1.0))
    ax_recon.set_title("Reconstruction difference (adv - clean)")
    ax_recon.set_axis_off()
    fig_recon.suptitle(f"{args.dataset} env {args.test_env} – {args.attack}", fontsize=12)
    fig_recon.tight_layout(rect=[0, 0, 1, 0.97])
    recon_path = output_dir / f"dataset_fft_diff_recon_{meta}.png"
    fig_recon.savefig(recon_path, dpi=args.dpi)
    plt.close(fig_recon)


def main() -> None:
    args = parse_args()
    freq_clean, freq_attacked, recon_clean, recon_attacked, domain = aggregate(args)
    visualize_distributions(
        freq_clean,
        freq_attacked,
        recon_clean,
        recon_attacked,
        args,
        domain,
    )


if __name__ == "__main__":
    main()
