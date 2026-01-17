#!/usr/bin/env python3
# coding=utf-8
"""
Analyze clean vs attacked prediction distributions per image.

Example:
  python analyze_label_distributions.py \
    --data_root /home/adam/Downloads/RobustTestTimeAdaptation/datasets \
    --adv_root /home/adam/Downloads/RobustTestTimeAdaptation/datasets_adv \
    --dataset PACS --domain 0 --seed 1 \
    --config resnet18_linf_eps-8.0_steps-20 \
    --max_images 50 --topk 10
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from alg.alg import get_algorithm_class
from utils.util import img_param_init


def parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", default="datasets", help="Root of ImageFolder datasets.")
    pa.add_argument("--adv_root", default="datasets_adv", help="Root of saved clean/adv tensors.")
    pa.add_argument("--dataset", default="PACS", choices=["PACS", "VLCS", "office-home"])
    pa.add_argument("--domain", type=int, default=0, help="Domain ID (index in dataset list).")
    pa.add_argument("--seed", type=int, default=0)
    pa.add_argument(
        "--config",
        default="resnet18_linf_eps-8.0_steps-20",
        help="Attack configuration folder under datasets_adv.",
    )
    pa.add_argument("--model", default=None, help="Path to model_<domain>_best.pt.")
    pa.add_argument("--batch_size", type=int, default=128)
    pa.add_argument("--cpu", action="store_true")
    pa.add_argument("--no_label_lookup", action="store_true", help="Skip ImageFolder label resolution.")
    pa.add_argument("--max_images", type=int, default=50, help="Max images to plot (-1 for all).")
    pa.add_argument("--topk", type=int, default=10, help="Top-k classes to plot (<=0 plots all).")
    pa.add_argument("--sample_seed", type=int, default=0, help="Seed for sampling plot indices.")
    pa.add_argument("--indices", type=int, nargs="*", default=None, help="Explicit indices to plot.")
    pa.add_argument("--out_dir", default="label_dist_plots", help="Output directory.")
    return pa.parse_args()


def _try_load_tensor(path: str) -> torch.Tensor:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _infer_net_from_config(cfg: str) -> str:
    return cfg.split("_")[0]


def resolve_domain_name(args: argparse.Namespace) -> str:
    base = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Data root missing: {base}")
    subdirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if args.domain < 0 or args.domain >= len(subdirs):
        raise IndexError(f"Domain id {args.domain} out of range [0,{len(subdirs)-1}] from {base}")
    return subdirs[args.domain]


def collect_indices(clean_dir: str, adv_dir: str) -> List[int]:
    if not os.path.isdir(clean_dir):
        raise FileNotFoundError(f"Not found: {clean_dir}")
    if not os.path.isdir(adv_dir):
        raise FileNotFoundError(f"Not found: {adv_dir}")

    def grab(dirpath: str) -> set[int]:
        ids = []
        for fn in os.listdir(dirpath):
            if fn.endswith(".pt"):
                stem = fn[:-3]
                try:
                    ids.append(int(stem))
                except ValueError:
                    continue
        return set(ids)

    clean_ids = grab(clean_dir)
    adv_ids = grab(adv_dir)
    common = sorted(clean_ids & adv_ids)
    if not common:
        raise RuntimeError(f"No overlapping indices between:\n  {clean_dir}\n  {adv_dir}")
    return common


def maybe_build_label_map(
    data_root: str, dataset: str, domain_name: str, skip: bool = False
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, str]]]:
    if skip:
        return None, None
    root_imgs = os.path.join(data_root, dataset, domain_name)
    if not os.path.isdir(root_imgs):
        print(f"[warn] Original dataset path not found for labels: {root_imgs}", file=sys.stderr)
        return None, None
    ds = ImageFolder(root_imgs)
    idx_to_class = {v: k for k, v in ds.class_to_idx.items()}
    labels = np.array([lab for _, lab in ds.imgs], dtype=np.int64)
    return labels, idx_to_class


def default_model_path(adv_root: str, seed: int, dataset: str, domain_name: str) -> str:
    return os.path.join(adv_root, f"seed_{seed}", dataset, "clean", f"model_{domain_name}_best.pt")


def build_model(args: argparse.Namespace, domain_name: str, device: torch.device) -> torch.nn.Module:
    ckpt_path = args.model or default_model_path(args.adv_root, args.seed, args.dataset, domain_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    sd = _try_load_tensor(ckpt_path)
    if not isinstance(sd, dict):
        raise RuntimeError("Unexpected checkpoint format (expected state_dict).")

    if "classifier.fc.weight" in sd:
        num_classes = sd["classifier.fc.weight"].shape[0]
    elif "classifier.weight" in sd:
        num_classes = sd["classifier.weight"].shape[0]
    else:
        fc_keys = [k for k in sd.keys() if k.endswith("fc.weight")]
        if not fc_keys:
            raise RuntimeError("Could not infer num_classes from checkpoint.")
        num_classes = sd[fc_keys[0]].shape[0]

    net = _infer_net_from_config(args.config)

    class A:
        pass

    a = A()
    a.net = net
    a.classifier = "linear"
    a.num_classes = num_classes
    a.dataset = args.dataset
    a.data_dir = os.path.join(args.data_root, args.dataset)
    img_param_init(a)

    erm = get_algorithm_class("ERM")(a).to(device)
    erm.load_state_dict(sd, strict=True)
    erm.eval()
    return erm


class PairedTensorDataset(Dataset):
    def __init__(
        self,
        clean_dir: str,
        adv_dir: str,
        indices: List[int],
        labels: Optional[np.ndarray],
    ) -> None:
        self.clean_dir = clean_dir
        self.adv_dir = adv_dir
        self.indices = indices
        self.labels = labels

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        index = self.indices[idx]
        clean = _try_load_tensor(os.path.join(self.clean_dir, f"{index}.pt"))
        adv = _try_load_tensor(os.path.join(self.adv_dir, f"{index}.pt"))
        label = -1 if self.labels is None else int(self.labels[index])
        return clean, adv, index, label


def _entropy(probs: torch.Tensor) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = p.clamp_min(1e-12)
    q = q.clamp_min(1e-12)
    return (p * (p.log() - q.log())).sum(dim=1)


def _js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _summarize(rows: List[Dict[str, float]], keys: List[str]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not rows:
        return summary
    for key in keys:
        vals = np.array([row[key] for row in rows if row[key] is not None], dtype=np.float64)
        if vals.size == 0:
            continue
        summary[key] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "median": float(np.median(vals)),
        }
    return summary


def main() -> None:
    args = parse_args()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

    domain_name = resolve_domain_name(args)
    clean_dir = os.path.join(args.adv_root, f"seed_{args.seed}", args.dataset, "clean", domain_name)
    adv_dir = os.path.join(args.adv_root, f"seed_{args.seed}", args.dataset, args.config, domain_name)

    indices = collect_indices(clean_dir, adv_dir)
    labels, idx_to_class = maybe_build_label_map(
        args.data_root, args.dataset, domain_name, args.no_label_lookup
    )

    if args.indices:
        requested = [i for i in args.indices if i in indices]
        missing = set(args.indices) - set(requested)
        if missing:
            print(f"[warn] ignoring missing indices: {sorted(missing)}", file=sys.stderr)
        plot_indices = requested
    else:
        if args.max_images < 0 or args.max_images >= len(indices):
            plot_indices = list(indices)
        else:
            rng = np.random.default_rng(args.sample_seed)
            plot_indices = list(rng.choice(indices, size=args.max_images, replace=False))

    plot_set = set(plot_indices)
    model = build_model(args, domain_name, device)

    dataset = PairedTensorDataset(clean_dir, adv_dir, indices, labels)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

    rows: List[Dict[str, float]] = []
    cache: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    with torch.no_grad():
        for clean, adv, idxs, lbls in loader:
            clean = clean.to(device)
            adv = adv.to(device)
            logits_clean = model.predict(clean) if hasattr(model, "predict") else model(clean)
            logits_adv = model.predict(adv) if hasattr(model, "predict") else model(adv)
            probs_clean = torch.softmax(logits_clean, dim=1)
            probs_adv = torch.softmax(logits_adv, dim=1)

            ent_clean = _entropy(probs_clean)
            ent_adv = _entropy(probs_adv)
            conf_clean, pred_clean = probs_clean.max(dim=1)
            conf_adv, pred_adv = probs_adv.max(dim=1)

            top2_clean = torch.topk(probs_clean, k=min(2, probs_clean.size(1)), dim=1).values
            top2_adv = torch.topk(probs_adv, k=min(2, probs_adv.size(1)), dim=1).values
            margin_clean = top2_clean[:, 0] - (top2_clean[:, 1] if top2_clean.size(1) > 1 else 0.0)
            margin_adv = top2_adv[:, 0] - (top2_adv[:, 1] if top2_adv.size(1) > 1 else 0.0)

            kl_clean_adv = _kl_divergence(probs_clean, probs_adv)
            kl_adv_clean = _kl_divergence(probs_adv, probs_clean)
            js_div = _js_divergence(probs_clean, probs_adv)
            l1_dist = (probs_clean - probs_adv).abs().sum(dim=1)
            l2_dist = torch.norm(probs_clean - probs_adv, dim=1)

            for i, idx in enumerate(idxs.tolist()):
                label_val = int(lbls[i])
                label = None if label_val < 0 else label_val
                row = {
                    "index": idx,
                    "label": None if label is None else int(label),
                    "pred_clean": int(pred_clean[i]),
                    "pred_adv": int(pred_adv[i]),
                    "pred_changed": int(pred_clean[i] != pred_adv[i]),
                    "entropy_clean": float(ent_clean[i].item()),
                    "entropy_adv": float(ent_adv[i].item()),
                    "conf_clean": float(conf_clean[i].item()),
                    "conf_adv": float(conf_adv[i].item()),
                    "margin_clean": float(margin_clean[i].item()),
                    "margin_adv": float(margin_adv[i].item()),
                    "kl_clean_adv": float(kl_clean_adv[i].item()),
                    "kl_adv_clean": float(kl_adv_clean[i].item()),
                    "js_div": float(js_div[i].item()),
                    "l1_dist": float(l1_dist[i].item()),
                    "l2_dist": float(l2_dist[i].item()),
                }
                if label is not None:
                    row["correct_clean"] = int(pred_clean[i] == label)
                    row["correct_adv"] = int(pred_adv[i] == label)
                rows.append(row)

                if idx in plot_set:
                    cache[idx] = (probs_clean[i].cpu(), probs_adv[i].cpu())

    os.makedirs(args.out_dir, exist_ok=True)
    plots_dir = os.path.join(args.out_dir, "per_image")
    os.makedirs(plots_dir, exist_ok=True)

    metrics_by_idx = {row["index"]: row for row in rows}
    for idx in sorted(plot_set):
        if idx not in cache:
            continue
        pc, pa = cache[idx]
        combined = 0.5 * (pc + pa)
        if args.topk is None or args.topk <= 0 or args.topk >= combined.numel():
            top_idx = torch.arange(combined.numel())
        else:
            top_idx = torch.topk(combined, k=args.topk).indices
        top_idx = top_idx.sort().values

        class_labels = []
        for j in top_idx.tolist():
            if idx_to_class is None:
                class_labels.append(str(j))
            else:
                class_labels.append(idx_to_class.get(j, str(j)))

        x = np.arange(len(top_idx))
        width = 0.4
        fig, ax = plt.subplots(figsize=(max(6, len(top_idx) * 0.35), 4.5))
        ax.bar(x - width / 2, pc[top_idx].numpy(), width, label="clean", color="#4c78a8")
        ax.bar(x + width / 2, pa[top_idx].numpy(), width, label="attacked", color="#f58518")
        ax.set_xticks(x)
        ax.set_xticklabels(class_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Probability")
        ax.legend(loc="upper right")

        stats = metrics_by_idx.get(idx, {})
        title = f"idx {idx} | clean {stats.get('pred_clean')} adv {stats.get('pred_adv')}"
        if "entropy_clean" in stats:
            title += f" | Hc {stats['entropy_clean']:.2f} Ha {stats['entropy_adv']:.2f}"
        if stats.get("label") is not None:
            title += f" | gt {stats['label']}"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(os.path.join(plots_dir, f"idx_{idx}.png"), dpi=150)
        plt.close(fig)

    csv_path = os.path.join(args.out_dir, "label_stats.csv")
    with open(csv_path, "w", newline="") as f:
        fieldnames = list(rows[0].keys()) if rows else []
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    numeric_keys = [
        "entropy_clean",
        "entropy_adv",
        "conf_clean",
        "conf_adv",
        "margin_clean",
        "margin_adv",
        "kl_clean_adv",
        "kl_adv_clean",
        "js_div",
        "l1_dist",
        "l2_dist",
    ]
    summary = {"overall": _summarize(rows, numeric_keys)}
    changed = [row for row in rows if row["pred_changed"] == 1]
    unchanged = [row for row in rows if row["pred_changed"] == 0]
    summary["pred_changed"] = _summarize(changed, numeric_keys)
    summary["pred_unchanged"] = _summarize(unchanged, numeric_keys)

    if rows and rows[0].get("label") is not None:
        summary["accuracy_clean"] = float(np.mean([r["correct_clean"] for r in rows]))
        summary["accuracy_adv"] = float(np.mean([r["correct_adv"] for r in rows]))

    summary_path = os.path.join(args.out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[done] wrote {len(rows)} rows to {csv_path}")
    print(f"[done] plots in {plots_dir}")
    print(f"[done] summary at {summary_path}")


if __name__ == "__main__":
    main()
