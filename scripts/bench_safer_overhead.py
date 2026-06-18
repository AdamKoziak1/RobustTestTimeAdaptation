#!/usr/bin/env python3
"""
Measure per-step wall-clock and FLOPs overhead of the SAFER wrapper on Tent.

Builds Tent / Tent+JPEG / Tent+SAFER(N=2) / Tent+SAFER(N=4) on a shared
ResNet-18 source model for PACS:Art (domain 0), runs each for a few
adaptation steps on real batches, and reports:
  - mean +/- std wall-clock per adaptation step (forward + backward + update)
  - images/sec
  - GFLOPs per step (forward + backward), measured with
    torch.utils.flop_counter.FlopCounterMode
  - cost relative to the unwrapped Tent baseline

Example:
  python scripts/bench_safer_overhead.py \
      --dataset PACS --domain 0 --seed 0 --batch-size 64 \
      --warmup 5 --iters 20 \
      --output-csv sweeps/compute_overhead_measured.csv \
      --output-tex sweeps/compute_overhead_measured_rows.tex
"""

from __future__ import annotations

import argparse
import copy
import csv
import statistics
import sys
from pathlib import Path
from typing import Callable, List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.flop_counter import FlopCounterMode
from torchvision import transforms
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alg.alg import get_algorithm_class  # noqa: E402
from utils.util import img_param_init, load_ckpt, set_random_seed  # noqa: E402
from utils.image_ops import InputDefense  # noqa: E402
from utils.safer_view import SAFERViewModule  # noqa: E402
from adapt_algorithm import (  # noqa: E402
    Tent,
    SAFERPooledPredictor,
    collect_params,
    configure_model,
)


class InputDefenseWrapper(nn.Module):
    """Mirrors unsupervise_adapt.InputDefenseWrapper (duplicated here to avoid
    importing unsupervise_adapt, whose top-level `from peft import ...` is
    incompatible with the transformers version pinned in requirements.txt)."""

    def __init__(self, model: nn.Module, defense: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.defense = defense
        self.defense.requires_grad_(False)
        self.defense.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.defense(x)
        return self.model.predict(x) if hasattr(self.model, "predict") else self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


DOMAIN_LABELS = {
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "office-home": ["Art", "Clipart", "Product", "RealWorld"],
}

VARIANTS: List[Tuple[str, int]] = [
    ("Tent", 1),
    ("Tent+JPEG", 1),
    ("Tent+SAFER (N=1)", 2),
    ("Tent+SAFER (N=2)", 3),
    ("Tent+SAFER (N=3)", 4),
    ("Tent+SAFER (N=4)", 5),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure wall-clock/FLOPs overhead of Tent / Tent+JPEG / Tent+SAFER."
    )
    parser.add_argument("--root", type=Path, default=ROOT, help="Repository root.")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--domain", type=int, default=0, help="Target domain id (0 = PACS:Art).")
    parser.add_argument("--seed", type=int, default=0, help="Source-model seed / data shuffling seed.")
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--jpeg-quality", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=5, help="Untimed warmup adaptation steps.")
    parser.add_argument("--iters", type=int, default=20, help="Timed adaptation steps.")
    parser.add_argument("--cpu", action="store_true", help="Force CPU.")
    parser.add_argument("--output-csv", type=Path, default=Path("sweeps/compute_overhead_measured.csv"))
    parser.add_argument("--output-tex", type=Path, default=Path("sweeps/compute_overhead_measured_rows.tex"))
    return parser.parse_args()


def _device(args: argparse.Namespace) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _load_batch(args: argparse.Namespace, device: torch.device) -> torch.Tensor:
    domain_name = DOMAIN_LABELS[args.dataset][args.domain]
    data_dir = args.root / "datasets" / args.dataset / domain_name
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Dataset domain dir not found: {data_dir}")
    tfm = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = ImageFolder(root=str(data_dir), transform=tfm)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        generator=torch.Generator().manual_seed(args.seed),
    )
    images, _labels = next(iter(loader))
    return images.to(device)


def _build_source_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    class _Args:
        pass

    a = _Args()
    a.net = args.net
    a.classifier = "linear"
    a.dataset = args.dataset
    a.data_dir = str(args.root / "datasets" / args.dataset)
    img_param_init(a)

    erm_cls = get_algorithm_class("ERM")
    model = erm_cls(a)
    ckpt_path = (
        args.root
        / "train_output"
        / args.dataset
        / f"test_{args.domain}"
        / f"seed_{args.seed}"
        / "model.pkl"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {ckpt_path}")
    model = load_ckpt(model, str(ckpt_path))
    return model.to(device)


def _make_view_module(model: nn.Module, num_views: int) -> SAFERViewModule:
    return SAFERViewModule(
        num_aug_views=num_views,
        include_original=True,
        aug_prob=1.0,
        aug_max_ops=3,
        augmentations=None,
        require_freq_or_blur=True,
        aug_seed=123,
        feature_normalize=False,
        view_weighting=True,
        primary_view_pool="cc_drop",
        js_weight=0.0,
        cc_weight=0.0,
        js_mode="pooled",
        js_view_pool="cc_drop",
        cc_mode="pairwise",
        cc_view_pool="cc_drop",
        cc_impl="fast",
        offdiag_weight=1.0,
        adaptive_alpha_mode="none",
        mean=None,
        std=None,
        input_is_normalized=None,
        stat_modules=(model.featurizer, model),
    )


def _build_adapt_model(
    label: str,
    base_model: nn.Module,
    args: argparse.Namespace,
    device: torch.device,
) -> nn.Module:
    """Build a freshly-wrapped Tent adaptation model for one overhead variant."""
    model = copy.deepcopy(base_model).to(device)
    model = configure_model(model)
    params, _ = collect_params(model)
    optimizer = torch.optim.Adam(params, lr=args.lr)

    if label == "Tent":
        wrapped: nn.Module = model
    elif label == "Tent+JPEG":
        defense = InputDefense(jpeg_quality=args.jpeg_quality).to(device)
        wrapped = InputDefenseWrapper(model, defense)
    elif label.startswith("Tent+SAFER"):
        num_views = int(label.split("N=")[1].rstrip(")"))
        view_module = _make_view_module(model, num_views).to(device)
        wrapped = SAFERPooledPredictor(model, view_module, log_metrics=False)
    else:
        raise ValueError(f"Unknown variant label: {label}")

    return Tent(wrapped, optimizer, steps=1, episodic=False).to(device)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def _time_steps(
    adapt_model: nn.Module,
    x: torch.Tensor,
    warmup: int,
    iters: int,
    device: torch.device,
) -> List[float]:
    import time

    for _ in range(warmup):
        adapt_model(x)
    _sync(device)

    times: List[float] = []
    for _ in range(iters):
        _sync(device)
        start = time.perf_counter()
        adapt_model(x)
        _sync(device)
        times.append(time.perf_counter() - start)
    return times


def _measure_flops(adapt_model: nn.Module, x: torch.Tensor) -> float:
    with FlopCounterMode(display=False) as counter:
        adapt_model(x)
    return float(counter.get_total_flops())


def main() -> int:
    args = _parse_args()
    device = _device(args)
    set_random_seed(args.seed)

    print(f"[info] device={device} dataset={args.dataset} domain={args.domain} "
          f"({DOMAIN_LABELS[args.dataset][args.domain]}) batch_size={args.batch_size}")

    x = _load_batch(args, device)
    base_model = _build_source_model(args, device)

    rows = []
    baseline_time = None
    for label, views in VARIANTS:
        print(f"[info] benchmarking {label} ...")
        adapt_model = _build_adapt_model(label, base_model, args, device)

        times = _time_steps(adapt_model, x, args.warmup, args.iters, device)
        mean_s = statistics.mean(times)
        std_s = statistics.stdev(times) if len(times) > 1 else 0.0
        images_per_sec = args.batch_size / mean_s

        # Re-build a fresh copy for FLOP counting so timing isn't disturbed by
        # the extra dispatcher-mode overhead.
        flop_model = _build_adapt_model(label, base_model, args, device)
        for _ in range(2):
            flop_model(x)  # warm up cuDNN autotuning before counting
        gflops = _measure_flops(flop_model, x) / 1e9

        if baseline_time is None:
            baseline_time = mean_s
        rel_cost = mean_s / baseline_time

        rows.append({
            "method": label,
            "views": views,
            "batch_size": args.batch_size,
            "step_time_ms_mean": mean_s * 1e3,
            "step_time_ms_std": std_s * 1e3,
            "images_per_sec": images_per_sec,
            "gflops_per_step": gflops,
            "rel_step_cost": rel_cost,
        })
        print(
            f"    step_time = {mean_s * 1e3:7.2f} +/- {std_s * 1e3:5.2f} ms   "
            f"({images_per_sec:7.1f} img/s)   "
            f"GFLOPs/step = {gflops:8.2f}   "
            f"rel.cost = {rel_cost:5.2f}x"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[ok] wrote csv: {args.output_csv}")

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    with args.output_tex.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(
                f"\\texttt{{{row['method']}}} & {row['views']} & "
                f"{row['step_time_ms_mean']:.1f} $\\pm$ {row['step_time_ms_std']:.1f} & "
                f"{row['images_per_sec']:.0f} & "
                f"{row['gflops_per_step']:.1f} & "
                f"${row['rel_step_cost']:.2f}\\times$ \\\\\n"
            )
    print(f"[ok] wrote tex rows: {args.output_tex}")
    print(
        "[note] FlopCounterMode only sees ATen tensor ops, so the JPEG codec "
        "(PIL/numpy-backed, jpeg_backprop='exact') is invisible to it - the "
        "Tent+JPEG GFLOPs figure equals the bare-Tent figure even though its "
        "measured wall-clock is much higher. Treat the wall-clock/img-s columns "
        "as authoritative for that row; GFLOPs there reflects only the backbone."
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
