#!/usr/bin/env python3
"""Visualize first-block feature maps under FFT configurations."""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch

from alg import alg
from utils.fft import FFTDrop2D
from utils.util import img_param_init, load_ckpt


@dataclass(frozen=True)
class Variant:
    key: str
    input_fft: bool
    feature_fft: bool

    @property
    def title(self) -> str:
        in_lbl = "in:fft" if self.input_fft else "in:orig"
        feat_lbl = "feat:fft" if self.feature_fft else "feat:orig"
        return f"{in_lbl} | {feat_lbl}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)

    parser.add_argument("--dataset", default="PACS", help="Dataset name (e.g. PACS, VLCS)")
    parser.add_argument("--test-env", type=int, default=2, help="Held-out domain index")
    parser.add_argument("--index", type=int, default=1099, help="Sample index (matches saved .pt name)")
    parser.add_argument("--seed", type=int, default=0, help="Seed used when saving attacked tensors")
    parser.add_argument("--net", default="resnet18", help="Backbone name (matches training config)")
    parser.add_argument("--algorithm", default="ERM", help="Algorithm name for checkpoint loading")
    parser.add_argument("--classifier", default="linear", help="Classifier head type (matches training)")
    parser.add_argument("--num-classes", type=int, default=None, help="Override class count if dataset is custom")

    parser.add_argument("--attacks", nargs="+", default=[
        "clean",
        "linf_eps-8.0_steps-20",
        "l2_eps-112.0_steps-100",
    ], help="Attack identifiers to visualize")

    parser.add_argument("--data-root", default="datasets", help="Root directory containing raw datasets")
    parser.add_argument("--adv-root", default="datasets_adv", help="Root with saved clean/attack tensors")
    parser.add_argument(
        "--train-root",
        default="train_output",
        help="Root directory containing pretrained checkpoints",
    )
    parser.add_argument("--model-path", default=None, help="Optional explicit checkpoint to load")

    parser.add_argument("--fft-input-keep-ratio", type=float, default=0.5, help="Keep ratio for input FFT filter")
    parser.add_argument("--fft-input-mode", choices=["spatial", "channel"], default="spatial")
    parser.add_argument("--fft-input-alpha", type=float, default=1.0, help="Residual mix weight for input FFT")
    parser.add_argument(
        "--fft-input-use-residual",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use residual mixing for input FFT",
    )
    parser.add_argument(
        "--fft-input-learn-alpha",
        type=int,
        default=0,
        choices=[0, 1],
        help="Learn residual alpha for input FFT",
    )
    parser.add_argument(
        "--skip-input-fft",
        action="store_true",
        help="Skip the input FFT toggle (keep original input only)",
    )

    parser.add_argument(
        "--fft-feature-keep-ratio", type=float, default=0.5, help="Keep ratio for feature FFT filter"
    )
    parser.add_argument("--fft-feature-mode", choices=["spatial", "channel"], default="spatial")
    parser.add_argument("--fft-feature-alpha", type=float, default=1.0, help="Residual mix weight for feature FFT")
    parser.add_argument(
        "--fft-feature-use-residual",
        type=int,
        default=1,
        choices=[0, 1],
        help="Use residual mixing for feature FFT",
    )
    parser.add_argument(
        "--fft-feature-learn-alpha",
        type=int,
        default=0,
        choices=[0, 1],
        help="Learn residual alpha for feature FFT",
    )
    parser.add_argument(
        "--skip-feature-fft",
        action="store_true",
        help="Skip the feature FFT toggle (keep original feature map only)",
    )

    parser.add_argument(
        "--channel-index",
        type=int,
        default=-1,
        help="Channel to visualize (−1 uses mean over channels)",
    )
    parser.add_argument("--cmap", default="magma", help="Matplotlib colormap for heatmaps")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--output-dir", default="feature_map_snapshots", help="Directory for saved plots")

    parser.add_argument("--log-wandb", action="store_true", help="Log figures to Weights & Biases")
    parser.add_argument("--wandb-project", default="fft_feature_maps", help="W&B project name")
    parser.add_argument("--wandb-entity", default=None, help="Optional W&B entity")
    parser.add_argument("--wandb-run-name", default=None, help="Optional W&B run name")

    parser.add_argument("--device", default="cuda", help="Computation device (auto fallback if unavailable)")

    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


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


def resolve_sample_paths(args: argparse.Namespace) -> Tuple[Dict[str, Path], str]:
    data_root = Path(args.data_root).expanduser()
    adv_root = Path(args.adv_root).expanduser()
    domain = resolve_domain_name(data_root, adv_root, args.dataset, args.test_env, args.seed)

    base = adv_root / f"seed_{args.seed}" / args.dataset
    clean_path = base / "clean" / domain / f"{args.index}.pt"

    paths: Dict[str, Path] = {}
    for attack in args.attacks:
        if attack == "clean":
            candidate = clean_path
        else:
            attack_dir = base / f"{args.net}_{attack}"
            candidate = attack_dir / domain / f"{args.index}.pt"
        if not candidate.is_file():
            raise FileNotFoundError(f"Sample for attack '{attack}' not found at {candidate}")
        paths[attack] = candidate
    return paths, domain


def _load_tensor(path: Path) -> torch.Tensor:
    try:
        tensor = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        tensor = torch.load(path, map_location="cpu")
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu()
    raise TypeError(f"Expected tensor at {path}, found {type(tensor)}")


def load_samples(paths: Mapping[str, Path]) -> Dict[str, torch.Tensor]:
    samples: Dict[str, torch.Tensor] = {}
    for name, path in paths.items():
        tensor = _load_tensor(path).float().clamp(0.0, 1.0)
        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(0)
        if tensor.ndim != 4 or tensor.shape[0] != 1:
            raise ValueError(f"Tensor at {path} should have shape (1,C,H,W); got {tuple(tensor.shape)}")
        samples[name] = tensor
    return samples


def build_algorithm(args: argparse.Namespace) -> torch.nn.Module:
    alg_args = argparse.Namespace(**vars(args))
    alg_args.fft_feat_max_layer = 0
    alg_args.fft_feat_keep_ratio = 1.0
    alg_args.fft_feat_mode = "spatial"
    alg_args.fft_feat_use_residual = False
    alg_args.fft_feat_alpha = 1.0
    alg_args.fft_feat_learn_alpha = False
    alg_args.svd_feat_max_layer = 0
    alg_args.svd_feat_rank_ratio = 1.0
    alg_args.svd_feat_mode = "spatial"
    alg_args.nuc_top = 0
    alg_args.nuc_kernel = 3
    alg_args.nuc_after_stem = False

    img_param_init(alg_args)
    if args.num_classes is not None:
        alg_args.num_classes = args.num_classes
    if not hasattr(alg_args, "num_classes") or alg_args.num_classes is None:
        raise ValueError(
            "Could not infer num_classes; provide --num-classes for custom datasets"
        )

    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(alg_args)
    algorithm.eval()

    model_path = args.model_path
    if model_path is None:
        model_path = Path(args.train_root) / args.dataset / f"test_{args.test_env}" / f"seed_{args.seed}" / "model.pkl"

    if Path(model_path).is_file():
        load_ckpt(algorithm, model_path)
    else:
        print(f"[warn] Checkpoint not found at {model_path}; using initial weights")
    return algorithm


def make_fft_module(
    keep_ratio: float,
    mode: str,
    use_residual: bool,
    alpha: float,
    learn_alpha: bool,
    device: torch.device,
) -> Optional[FFTDrop2D]:
    if keep_ratio >= 1.0:
        return None
    module = FFTDrop2D(
        keep_ratio=keep_ratio,
        mode=mode,
        backprop_mode="exact",
        use_residual=use_residual,
        alpha=alpha,
        learn_alpha=learn_alpha,
    )
    module.to(device)
    module.eval()
    return module


def standardize_input(x: torch.Tensor, module: torch.nn.Module) -> torch.Tensor:
    mean = getattr(module, "_in_mean", None)
    std = getattr(module, "_in_std", None)
    if mean is None or std is None:
        return x
    return (x - mean.to(x.device)) / std.to(x.device)


def extract_first_block(
    featurizer: torch.nn.Module,
    x: torch.Tensor,
    input_fft: Optional[FFTDrop2D],
    feature_fft: Optional[FFTDrop2D],
) -> torch.Tensor:
    with torch.no_grad():
        xb = x.clone()
        if input_fft is not None:
            xb = input_fft(xb)
        xb = standardize_input(xb, featurizer)
        out = featurizer.conv1(xb)
        out = featurizer.bn1(out)
        out = featurizer.relu(out)
        if feature_fft is not None:
            out = feature_fft(out)
    return out.detach().cpu()


def reduce_feature_map(feature: torch.Tensor, channel_idx: int) -> np.ndarray:
    feat = feature.squeeze(0)
    if channel_idx < 0:
        reduced = feat.mean(dim=0)
    else:
        if channel_idx >= feat.shape[0]:
            raise IndexError(
                f"channel-index {channel_idx} out of range for feature map with {feat.shape[0]} channels"
            )
        reduced = feat[channel_idx]
    return reduced.numpy()


def normalize_map(array: np.ndarray) -> np.ndarray:
    min_val = float(array.min())
    max_val = float(array.max())
    if np.isclose(max_val, min_val):
        return np.zeros_like(array)
    return (array - min_val) / (max_val - min_val)


def prepare_variants(args: argparse.Namespace) -> List[Variant]:
    variants: List[Variant] = []
    input_switch = [False] if args.skip_input_fft else [False, True]
    feature_switch = [False] if args.skip_feature_fft else [False, True]
    for input_on in input_switch:
        for feature_on in feature_switch:
            key = f"in-{'fft' if input_on else 'orig'}__feat-{'fft' if feature_on else 'orig'}"
            variants.append(Variant(key=key, input_fft=input_on, feature_fft=feature_on))
    return variants


def plot_feature_maps(
    dataset: str,
    domain: str,
    sample_index: int,
    attacks: Iterable[str],
    variants: Iterable[Variant],
    heatmaps: Mapping[str, Mapping[str, np.ndarray]],
    cmap: str,
    dpi: int,
    output_dir: Path,
) -> Path:
    attack_list = list(attacks)
    variant_list = list(variants)
    rows = len(attack_list)
    cols = len(variant_list)

    fig, axes = plt.subplots(rows, cols, figsize=(3.3 * cols, 3.2 * rows), squeeze=False)

    for row_idx, attack in enumerate(attack_list):
        for col_idx, variant in enumerate(variant_list):
            ax = axes[row_idx, col_idx]
            data = heatmaps[attack][variant.key]
            normed = normalize_map(data)
            im = ax.imshow(normed, cmap=cmap, vmin=0.0, vmax=1.0)
            ax.set_axis_off()
            if row_idx == 0:
                ax.set_title(variant.title, fontsize=11)
            if col_idx == 0:
                ax.set_ylabel(attack, rotation=90, fontsize=11)
    fig.suptitle(
        f"{dataset} · {domain} · sample {sample_index}", fontsize=14
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.02, hspace=0.02)

    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{dataset}_{domain}_idx-{sample_index}_feature-maps.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def maybe_log_wandb(
    enabled: bool,
    run_kwargs: Mapping[str, Optional[str]],
    figure_path: Path,
) -> None:
    if not enabled:
        return
    try:
        import wandb
    except ImportError:  # pragma: no cover - optional dependency
        print("[warn] wandb not installed; skipping logging")
        return

    run = wandb.init(
        project=run_kwargs["project"],
        entity=run_kwargs["entity"],
        name=run_kwargs["run_name"],
        config={"figure": figure_path.name},
    )
    assert run is not None
    wandb.log({"feature_maps": wandb.Image(str(figure_path))})
    wandb.finish()


def main() -> None:
    args = parse_args()
    args.fft_input_use_residual = bool(args.fft_input_use_residual)
    args.fft_input_learn_alpha = bool(args.fft_input_learn_alpha)
    args.fft_feature_use_residual = bool(args.fft_feature_use_residual)
    args.fft_feature_learn_alpha = bool(args.fft_feature_learn_alpha)

    device = select_device(args.device)
    if device.type == "cpu" and args.device != "cpu":
        print("[info] CUDA unavailable; using CPU")

    variants = prepare_variants(args)
    if not variants:
        raise RuntimeError("No visualization variants configured")

    sample_paths, domain = resolve_sample_paths(args)
    samples = load_samples(sample_paths)

    algorithm = build_algorithm(args)
    algorithm.to(device)
    algorithm.eval()
    featurizer = algorithm.featurizer
    featurizer.to(device)
    featurizer.eval()

    input_fft_module = make_fft_module(
        args.fft_input_keep_ratio,
        args.fft_input_mode,
        args.fft_input_use_residual,
        args.fft_input_alpha,
        args.fft_input_learn_alpha,
        device,
    )
    feature_fft_module = make_fft_module(
        args.fft_feature_keep_ratio,
        args.fft_feature_mode,
        args.fft_feature_use_residual,
        args.fft_feature_alpha,
        args.fft_feature_learn_alpha,
        device,
    )

    heatmaps: Dict[str, Dict[str, np.ndarray]] = {}
    for attack, tensor in samples.items():
        tensor_device = tensor.to(device)
        heatmaps[attack] = {}
        for variant in variants:
            input_fft = input_fft_module if variant.input_fft else None
            feature_fft = feature_fft_module if variant.feature_fft else None
            feature = extract_first_block(featurizer, tensor_device, input_fft, feature_fft)
            heatmaps[attack][variant.key] = reduce_feature_map(feature, args.channel_index)

    out_path = plot_feature_maps(
        dataset=args.dataset,
        domain=domain,
        sample_index=args.index,
        attacks=args.attacks,
        variants=variants,
        heatmaps=heatmaps,
        cmap=args.cmap,
        dpi=args.dpi,
        output_dir=Path(args.output_dir),
    )

    maybe_log_wandb(
        enabled=args.log_wandb,
        run_kwargs={
            "project": args.wandb_project,
            "entity": args.wandb_entity,
            "run_name": args.wandb_run_name,
        },
        figure_path=out_path,
    )

    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
