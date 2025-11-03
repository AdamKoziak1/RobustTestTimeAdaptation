#!/usr/bin/env python3
"""Visualize FFT masking pipeline for clean vs attacked images."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image


StageMap = Dict[str, np.ndarray]
StageBundle = Dict[str, StageMap]

COLOR_STAGE_KEYS = {"spatial", "reconstruction"}
FREQUENCY_STAGE_KEYS = {"fft_abs", "shifted_abs", "shifted_masked_abs"}

CHANNEL_COLOR_LUT = np.array(
    [
        [1.0, 0.0, 0.0],  # R
        [0.0, 1.0, 0.0],  # G
        [0.0, 0.0, 1.0],  # B
        [1.0, 1.0, 0.0],  # Y (fallback)
        [0.0, 1.0, 1.0],  # C (fallback)
        [1.0, 0.0, 1.0],  # M (fallback)
    ],
    dtype=np.float32,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    # Direct path inputs ----------------------------------------------------
    parser.add_argument("--clean", help="Path to the clean image/tensor")
    parser.add_argument("--attacked", help="Path to the attacked image/tensor")

    # Dataset metadata ------------------------------------------------------
    parser.add_argument("--dataset", default="PACS", help="Dataset name (e.g. PACS, VLCS)")
    parser.add_argument("--test-env", default=2, type=int, help="Held-out domain index")
    parser.add_argument("--index", default=1099, type=int, help="Sample index (matches saved .pt name)")
    parser.add_argument(
        "--attack",
        default="linf_eps-8.0_steps-20",
        type=str,
        help="Attack configuration (e.g., linf_eps-8.0_steps-20_fft-spatial_k-0.75) or 'clean'.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Seed used when saving attacked tensors")
    parser.add_argument(
        "--data-root",
        default="datasets",
        help="Root directory of the original ImageFolder datasets",
    )
    parser.add_argument(
        "--adv-root",
        default="datasets_adv",
        help="Root directory containing clean/attacked tensors",
    )

    # Visualization ---------------------------------------------------------
    parser.add_argument(
        "--keep-ratio",
        type=float,
        default=0.5,
        help="Fraction of spatial frequencies to keep (match utils.fft.FFTDrop2D)",
    )
    parser.add_argument(
        "--output-dir",
        default="fft_snapshots",
        help="Directory where figures will be stored",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI for saved snapshots",
    )

    return parser.parse_args()


def config_id(net: str, attack: str, eps: float, steps: int) -> str:
    return f"{net}_{attack}_eps-{eps}_steps-{steps}"


def resolve_domain_name(
    data_root: Path, adv_root: Path, dataset: str, test_env: int, seed: int
) -> str:
    dataset_root = data_root / dataset
    candidates: List[str] = []

    if dataset_root.is_dir():
        candidates = sorted(
            [p.name for p in dataset_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        )

    if not candidates:
        clean_root = adv_root / f"seed_{seed}" / dataset / "clean"
        if not clean_root.is_dir():
            raise FileNotFoundError(
                f"Could not resolve domains under {dataset_root} or {clean_root}"
            )
        candidates = sorted(
            [p.name for p in clean_root.iterdir() if p.is_dir() and not p.name.startswith(".")]
        )

    if not candidates:
        raise RuntimeError(f"No domain folders found for dataset '{dataset}'")

    if test_env < 0 or test_env >= len(candidates):
        raise IndexError(
            f"test-env {test_env} out of range; found {len(candidates)} domains: {candidates}"
        )
    return candidates[test_env]


def resolve_input_paths(args: argparse.Namespace) -> Tuple[Path, Path]:
    if args.clean or args.attacked:
        if not (args.clean and args.attacked):
            raise ValueError("Both --clean and --attacked must be provided together")
        return Path(args.clean), Path(args.attacked)

    missing = [
        name
        for name, present in [
            ("--dataset", args.dataset is not None),
            ("--test-env", args.test_env is not None),
            ("--index", args.index is not None),
            ("--attack", args.attack is not None),
        ]
        if not present
    ]
    if missing:
        opts = ", ".join(missing)
        raise ValueError(f"Missing required arguments to resolve dataset tensors: {opts}")

    data_root = Path(args.data_root).expanduser()
    adv_root = Path(args.adv_root).expanduser()
    domain = resolve_domain_name(data_root, adv_root, args.dataset, args.test_env, args.seed)

    attack_folder = f"resnet18_{args.attack}"
    base = adv_root / f"seed_{args.seed}" / args.dataset

    clean_path = base / "clean" / domain / f"{args.index}.pt"
    attacked_path = base / attack_folder / domain / f"{args.index}.pt"

    for desc, path in ("clean", clean_path), ("attacked", attacked_path):
        if not path.is_file():
            raise FileNotFoundError(f"{desc} tensor not found at {path}")

    return clean_path, attacked_path


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    raise TypeError(f"Expected tensor at {path}, found {type(tensor)}")


def load_image(path: Path) -> np.ndarray:
    suffix = path.suffix.lower()
    if suffix == ".pt":
        tensor = _load_tensor(path).float().clamp(0.0, 1.0)
        if tensor.ndim != 3:
            raise ValueError(f"Tensor at {path} should have shape (C,H,W); got {tuple(tensor.shape)}")
        return tensor.numpy()

    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return np.transpose(array, (2, 0, 1))


def center_mask(shape: Tuple[int, int], keep_ratio: float) -> np.ndarray:
    height, width = shape
    keep_h = max(1, int(np.ceil(height * keep_ratio)))
    keep_w = max(1, int(np.ceil(width * keep_ratio)))
    start_h = (height - keep_h) // 2
    start_w = (width - keep_w) // 2
    mask = np.zeros((height, width), dtype=np.float32)
    mask[start_h : start_h + keep_h, start_w : start_w + keep_w] = 1.0
    return mask


def compute_fft_stages(image: np.ndarray, keep_ratio: float) -> StageMap:
    channels, height, width = image.shape
    #freq = np.fft.fft2(image, axes=(-2, -1))
    freq = np.fft.fft2(image, axes=(-2, -1), norm="ortho")
    freq_shifted = np.fft.fftshift(freq, axes=(-2, -1))

    mask = center_mask((height, width), keep_ratio)
    masked_shifted = freq_shifted * mask[None, ...]
    masked = np.fft.ifftshift(masked_shifted, axes=(-2, -1))
    reconstructed = np.fft.ifft2(masked, axes=(-2, -1), norm="ortho").real
    #reconstructed = np.fft.ifft2(masked, axes=(-2, -1)).real

    stages: StageMap = {
        "spatial": image,
        "fft_abs": np.abs(freq),
        "shifted_abs": np.abs(freq_shifted),
        "mask": mask,
        "shifted_masked_abs": np.abs(masked_shifted),
        "reconstruction": reconstructed,
    }
    return stages


def normalize(channel_data: np.ndarray) -> np.ndarray:
    data = channel_data.astype(np.float32)
    min_val = data.min()
    max_val = data.max()
    if max_val > min_val:
        data = (data - min_val) / (max_val - min_val)
    else:
        data = np.zeros_like(data)
    return data


def log_scale(array: np.ndarray) -> np.ndarray:
    return np.log1p(array)


def channel_labels(count: int) -> List[str]:
    default = ["R", "G", "B"]
    if count == 3:
        return default
    return [f"C{idx}" for idx in range(count)]


def replicate_mask(mask: np.ndarray, channels: int) -> np.ndarray:
    return np.repeat(mask[None, ...], channels, axis=0)


def colorize_channel(slice_2d: np.ndarray, channel_idx: int) -> np.ndarray:
    base = normalize(slice_2d)
    lut_idx = channel_idx if channel_idx < len(CHANNEL_COLOR_LUT) else -1
    color = CHANNEL_COLOR_LUT[lut_idx]
    tinted = np.zeros((*base.shape, 3), dtype=np.float32)
    for c in range(3):
        tinted[..., c] = base * color[c]
    return np.clip(tinted, 0.0, 1.0)


def prepare_stage_for_channels(
    stages: StageMap, channels: int, *, apply_log: bool = True
) -> Dict[str, np.ndarray]:
    prepared: Dict[str, np.ndarray] = {}
    for key, value in stages.items():
        if value.ndim == 2:
            prepared[key] = replicate_mask(value, channels)
        else:
            prepared[key] = value
    if apply_log:
        prepared["fft_abs"] = log_scale(log_scale(prepared["fft_abs"]))
        prepared["shifted_abs"] = log_scale(log_scale(prepared["shifted_abs"]))
        prepared["shifted_masked_abs"] = log_scale(log_scale(prepared["shifted_masked_abs"]))
        # prepared["fft_abs"] = prepared["fft_abs"]
        # prepared["shifted_abs"] = prepared["shifted_abs"]
        # prepared["shifted_masked_abs"] = prepared["shifted_masked_abs"]
    return prepared


def plot_channel_comparison(
    bundles: StageBundle,
    output_dir: Path,
    dpi: int,
    ordered: List[str],
    file_prefix: str,
) -> None:
    names = list(bundles.keys())
    reference = bundles[names[0]]
    channels = reference["spatial"].shape[0]
    labels = channel_labels(channels)

    prepared = {
        name: prepare_stage_for_channels(stages, channels, apply_log=(name != "difference"))
        for name, stages in bundles.items()
    }
    n_variants = len(names)
    n_cols = channels * n_variants

    titles = {
        "spatial": "Spatial",
        "fft_abs": "FFT |F|",
        "shifted_abs": "Shifted |F|",
        "shifted_masked_abs": "Masked |F|",
        "reconstruction": "Reconstruction",
    }

    for key in ordered:
        fig, axes = plt.subplots(
            1,
            n_cols,
            figsize=(2.1 * n_cols, 2.2),
            squeeze=False,
        )
        axes = axes.reshape(-1)
        for c_idx in range(channels):
            for v_idx, name in enumerate(names):
                col = c_idx * n_variants + v_idx
                ax = axes[col]
                stage_raw = bundles[name][key]
                if key in COLOR_STAGE_KEYS and stage_raw.ndim == 3 and stage_raw.shape[0] == 3:
                    channel_slice = stage_raw[c_idx]
                    ax.imshow(colorize_channel(channel_slice, c_idx))
                elif key in FREQUENCY_STAGE_KEYS and stage_raw.ndim == 3 and stage_raw.shape[0] == channels:
                    ax.imshow(colorize_channel(prepared[name][key][c_idx], c_idx))
                else:
                    ax.imshow(normalize(prepared[name][key][c_idx]), cmap="magma")
                ax.set_axis_off()
                ax.set_title(f"{labels[c_idx]} {name}", fontsize=9)
        fig.suptitle(
            f"Channel-wise – {titles.get(key, key.replace('_', ' '))}", fontsize=13
        )
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.subplots_adjust(wspace=0.005)
        out_path = output_dir / f"{file_prefix}_channels_{key}.png"
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)


def plot_average_comparison(
    bundles: StageBundle,
    output_dir: Path,
    dpi: int,
    ordered: List[str],
    file_prefix: str,
) -> None:
    names = list(bundles.keys())
    reference = bundles[names[0]]
    channels = reference["spatial"].shape[0]
    prepared = {
        name: prepare_stage_for_channels(stages, channels, apply_log=(name != "difference"))
        for name, stages in bundles.items()
    }

    n_variants = len(names)

    titles = {
        "spatial": "Spatial",
        "fft_abs": "FFT |F|",
        "shifted_abs": "Shifted |F|",
        "shifted_masked_abs": "Masked |F|",
        "reconstruction": "Reconstruction",
    }

    for key in ordered:
        for name in names:
            raw_stage = bundles[name][key]
            prep_stage = prepared[name][key]

            if key in COLOR_STAGE_KEYS and raw_stage.ndim == 3 and raw_stage.shape[0] == 3:
                if name == "difference":
                    diff_rgb = raw_stage.transpose(1, 2, 0).astype(np.float32)
                    max_abs = float(np.max(np.abs(diff_rgb)))
                    if max_abs < 1e-8:
                        max_abs = 1.0
                    img = 0.5 + diff_rgb / (2.0 * max_abs)
                else:
                    img = np.clip(raw_stage.transpose(1, 2, 0), 0.0, 1.0)
                cmap = None
                img = np.clip(img, 0.0, 1.0)
            else:
                data = prep_stage
                if data.ndim == 3:
                    data = data.mean(axis=0)
                if name == "difference" and np.any(data):
                    if key in FREQUENCY_STAGE_KEYS:
                        data = np.log1p(np.maximum(data, 0.0))
                    data = data.astype(np.float32)
                    max_abs = float(np.max(data))
                    if max_abs < 1e-8:
                        max_abs = 1.0
                    img = np.clip(data / max_abs, 0.0, 1.0)
                else:
                    img = normalize(data)
                cmap = "magma"

            out_path = output_dir / f"{file_prefix}_{key}_{name}.png"
            if img.ndim == 2:
                plt.imsave(out_path, img, cmap=cmap)
            else:
                plt.imsave(out_path, img)


def process_pair(
    clean_path: Path,
    attacked_path: Path,
    keep_ratio: float,
    output_dir: Path,
    dpi: int,
    file_prefix: str,
) -> None:
    clean_image = load_image(clean_path)
    attacked_image = load_image(attacked_path)

    if clean_image.shape != attacked_image.shape:
        raise ValueError(
            f"Shape mismatch: clean {clean_image.shape} vs attacked {attacked_image.shape}"
        )

    clean_stages = compute_fft_stages(clean_image, keep_ratio)
    attacked_stages = compute_fft_stages(attacked_image, keep_ratio)

    ordered_keys = [
        "spatial",
        "fft_abs",
        "shifted_abs",
        "shifted_masked_abs",
        "reconstruction",
    ]

    diff_stages: StageMap = {}
    for key in ordered_keys:
        if key in {"fft_abs", "shifted_abs", "shifted_masked_abs"}:
            diff = np.abs(attacked_stages[key] - clean_stages[key])
        else:
            diff = attacked_stages[key] - clean_stages[key]
        diff_stages[key] = diff

    channel_bundles: StageBundle = {
        "clean": clean_stages,
        "attacked": attacked_stages,
    }

    average_bundles: StageBundle = {
        "clean": clean_stages,
        "attacked": attacked_stages,
        "difference": diff_stages,
    }
    #plot_channel_comparison(channel_bundles, output_dir, dpi, ordered_keys, file_prefix)
    plot_average_comparison(average_bundles, output_dir, dpi, ordered_keys, file_prefix)


def main() -> None:
    args = parse_args()
    clean_path, attacked_path = resolve_input_paths(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    attack_slug = args.attack.replace(".", "p").replace("/", "-")
    domain_name = clean_path.parent.name
    idx_label = clean_path.stem
    file_prefix = (
        #f"{args.dataset}_env{args.test_env}_seed{args.seed}_{domain_name}_idx{idx_label}_attack-{attack_slug}"
        f"linf"
    )
    process_pair(clean_path, attacked_path, args.keep_ratio, output_dir, args.dpi, file_prefix)


if __name__ == "__main__":
    main()
