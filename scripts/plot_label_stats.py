#!/usr/bin/env python3
# coding=utf-8
"""
Create aggregate plots and a compact report from label_stats.csv produced by
analyze_label_distributions.py.
"""

import argparse
import csv
import json
import os
from collections import Counter
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    pa = argparse.ArgumentParser()
    pa.add_argument("--input_csv", default="label_dist_plots/label_stats.csv")
    pa.add_argument("--out_dir", default="label_dist_plots/aggregate")
    pa.add_argument("--bins", type=int, default=50)
    pa.add_argument("--top_n", type=int, default=12)
    return pa.parse_args()


def _to_num(value: str) -> float:
    if value is None:
        return np.nan
    txt = str(value).strip()
    if txt == "" or txt.lower() == "none":
        return np.nan
    try:
        return float(txt)
    except ValueError:
        return np.nan


def load_rows(path: str) -> List[Dict[str, float]]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing input csv: {path}")
    rows = []
    with open(path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for raw in reader:
            row = {k: _to_num(v) for k, v in raw.items()}
            rows.append(row)
    if not rows:
        raise RuntimeError(f"No rows found in {path}")
    return rows


def as_array(rows: List[Dict[str, float]], key: str) -> np.ndarray:
    arr = np.array([r.get(key, np.nan) for r in rows], dtype=np.float64)
    return arr[np.isfinite(arr)]


def _safe_mean(arr: np.ndarray) -> float:
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def _plot_hist_metrics(rows: List[Dict[str, float]], out_dir: str, bins: int) -> None:
    pairs = [
        ("entropy_clean", "entropy_adv", "Entropy"),
        ("conf_clean", "conf_adv", "Max probability"),
        ("margin_clean", "margin_adv", "Top-1 minus Top-2"),
    ]
    singles = [
        ("js_div", "JS divergence"),
        ("l1_dist", "L1 distance"),
        ("l2_dist", "L2 distance"),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.ravel()

    for i, (k1, k2, title) in enumerate(pairs):
        a = as_array(rows, k1)
        b = as_array(rows, k2)
        axes[i].hist(a, bins=bins, alpha=0.6, label="clean", color="#4c78a8")
        axes[i].hist(b, bins=bins, alpha=0.6, label="attacked", color="#f58518")
        axes[i].set_title(title)
        axes[i].legend()

    for j, (k, title) in enumerate(singles, start=3):
        a = as_array(rows, k)
        axes[j].hist(a, bins=bins, color="#54a24b", alpha=0.85)
        axes[j].set_title(title)

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "metrics_histograms.png"), dpi=160)
    plt.close(fig)


def _plot_shift_scatter(rows: List[Dict[str, float]], out_dir: str) -> None:
    conf_clean = as_array(rows, "conf_clean")
    conf_adv = as_array(rows, "conf_adv")
    ent_clean = as_array(rows, "entropy_clean")
    ent_adv = as_array(rows, "entropy_adv")

    pred_changed = np.array([int(r.get("pred_changed", 0)) for r in rows], dtype=np.int64)
    colors = np.where(pred_changed == 1, "#e45756", "#4c78a8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    n = min(len(rows), conf_clean.size, conf_adv.size, colors.size)
    axes[0].scatter(conf_clean[:n], conf_adv[:n], s=8, alpha=0.35, c=colors[:n], linewidths=0)
    axes[0].plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    axes[0].set_xlabel("Clean max probability")
    axes[0].set_ylabel("Attacked max probability")
    axes[0].set_title("Confidence shift per sample")

    n2 = min(len(rows), ent_clean.size, ent_adv.size, colors.size)
    axes[1].scatter(ent_clean[:n2], ent_adv[:n2], s=8, alpha=0.35, c=colors[:n2], linewidths=0)
    xmax = float(max(ent_clean.max() if ent_clean.size else 1.0, ent_adv.max() if ent_adv.size else 1.0))
    axes[1].plot([0, xmax], [0, xmax], linestyle="--", color="black", linewidth=1)
    axes[1].set_xlabel("Clean entropy")
    axes[1].set_ylabel("Attacked entropy")
    axes[1].set_title("Entropy shift per sample")

    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "shift_scatter.png"), dpi=160)
    plt.close(fig)


def _plot_classwise(rows: List[Dict[str, float]], out_dir: str, top_n: int) -> Dict[str, object]:
    label_vals = np.array([r.get("label", np.nan) for r in rows], dtype=np.float64)
    finite_mask = np.isfinite(label_vals)
    out: Dict[str, object] = {}

    pred_adv = np.array([int(r.get("pred_adv", np.nan)) for r in rows], dtype=np.int64)
    pred_clean = np.array([int(r.get("pred_clean", np.nan)) for r in rows], dtype=np.int64)
    pred_changed = np.array([int(r.get("pred_changed", 0)) for r in rows], dtype=np.int64)

    adv_counts = Counter(pred_adv.tolist())
    top_adv = adv_counts.most_common(top_n)
    if top_adv:
        cls = [k for k, _ in top_adv]
        vals = [v for _, v in top_adv]
        fig, ax = plt.subplots(figsize=(max(8, 0.6 * len(cls)), 4.8))
        ax.bar(np.arange(len(cls)), vals, color="#f58518")
        ax.set_xticks(np.arange(len(cls)))
        ax.set_xticklabels([str(c) for c in cls], rotation=45, ha="right")
        ax.set_ylabel("Count")
        ax.set_title(f"Top attacked predictions (top {len(cls)})")
        fig.tight_layout()
        fig.savefig(os.path.join(out_dir, "attacked_prediction_counts.png"), dpi=160)
        plt.close(fig)

    flip_pairs = Counter(zip(pred_clean.tolist(), pred_adv.tolist()))
    flip_pairs = Counter({k: v for k, v in flip_pairs.items() if k[0] != k[1]})
    out["top_clean_to_adv_flips"] = [
        {"clean_pred": int(c), "adv_pred": int(a), "count": int(n)}
        for (c, a), n in flip_pairs.most_common(top_n)
    ]

    out["pred_changed_rate"] = float(np.mean(pred_changed))

    if not finite_mask.any():
        return out

    label = label_vals[finite_mask].astype(np.int64)
    clean_ok = np.array([int(r.get("correct_clean", np.nan)) for r in rows], dtype=np.float64)[finite_mask]
    adv_ok = np.array([int(r.get("correct_adv", np.nan)) for r in rows], dtype=np.float64)[finite_mask]

    classes = np.unique(label)
    acc_clean = []
    acc_adv = []
    counts = []
    for c in classes:
        m = label == c
        counts.append(int(np.sum(m)))
        acc_clean.append(float(np.mean(clean_ok[m])))
        acc_adv.append(float(np.mean(adv_ok[m])))

    x = np.arange(len(classes))
    width = 0.42
    fig, ax = plt.subplots(figsize=(max(9, 0.55 * len(classes)), 4.8))
    ax.bar(x - width / 2, acc_clean, width, label="clean", color="#4c78a8")
    ax.bar(x + width / 2, acc_adv, width, label="attacked", color="#e45756")
    ax.set_xticks(x)
    ax.set_xticklabels([str(c) for c in classes], rotation=45, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("Accuracy")
    ax.set_title("Class-wise accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "class_accuracy.png"), dpi=160)
    plt.close(fig)

    pred_adv_l = np.array([int(r.get("pred_adv", np.nan)) for r in rows], dtype=np.float64)[finite_mask].astype(np.int64)
    max_class = int(max(np.max(classes), np.max(pred_adv_l)))
    mat = np.zeros((max_class + 1, max_class + 1), dtype=np.float64)
    for t, p in zip(label, pred_adv_l):
        mat[t, p] += 1.0
    row_sum = mat.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    mat_norm = mat / row_sum

    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    im = ax.imshow(mat_norm, aspect="auto", interpolation="nearest", cmap="magma")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("Attacked prediction")
    ax.set_ylabel("True class")
    ax.set_title("True class -> attacked prediction (row-normalized)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "true_to_attacked_heatmap.png"), dpi=170)
    plt.close(fig)

    class_rows = []
    for c, n, ac, aa in zip(classes.tolist(), counts, acc_clean, acc_adv):
        class_rows.append(
            {
                "class": int(c),
                "count": int(n),
                "acc_clean": float(ac),
                "acc_adv": float(aa),
                "acc_drop": float(ac - aa),
            }
        )
    class_rows = sorted(class_rows, key=lambda d: d["acc_drop"], reverse=True)
    out["worst_classes_by_acc_drop"] = class_rows[:top_n]
    return out


def _plot_delta_hist(rows: List[Dict[str, float]], out_dir: str, bins: int) -> None:
    dc = as_array(rows, "conf_adv") - as_array(rows, "conf_clean")
    de = as_array(rows, "entropy_adv") - as_array(rows, "entropy_clean")
    dm = as_array(rows, "margin_adv") - as_array(rows, "margin_clean")

    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.2))
    axes[0].hist(dc, bins=bins, color="#f58518", alpha=0.85)
    axes[0].set_title("Delta max probability (adv-clean)")
    axes[1].hist(de, bins=bins, color="#4c78a8", alpha=0.85)
    axes[1].set_title("Delta entropy (adv-clean)")
    axes[2].hist(dm, bins=bins, color="#54a24b", alpha=0.85)
    axes[2].set_title("Delta margin (adv-clean)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "delta_histograms.png"), dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    rows = load_rows(args.input_csv)

    _plot_hist_metrics(rows, args.out_dir, args.bins)
    _plot_shift_scatter(rows, args.out_dir)
    _plot_delta_hist(rows, args.out_dir, args.bins)
    report = _plot_classwise(rows, args.out_dir, args.top_n)

    summary = {
        "num_rows": int(len(rows)),
        "pred_changed_rate": float(np.mean(as_array(rows, "pred_changed"))),
        "mean_conf_clean": _safe_mean(as_array(rows, "conf_clean")),
        "mean_conf_adv": _safe_mean(as_array(rows, "conf_adv")),
        "mean_entropy_clean": _safe_mean(as_array(rows, "entropy_clean")),
        "mean_entropy_adv": _safe_mean(as_array(rows, "entropy_adv")),
        "mean_js_div": _safe_mean(as_array(rows, "js_div")),
    }
    summary.update(report)

    out_json = os.path.join(args.out_dir, "aggregate_report.json")
    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"[ok] wrote plots + report to {args.out_dir}")


if __name__ == "__main__":
    main()
