#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def to_method_label(record: wt.RunRecord) -> str:
    adapt_alg = wt.get_value(record, "adapt_alg")
    if not isinstance(adapt_alg, str):
        adapt_alg = "Method"
    wrap_raw = wt.get_value(record, "s_wrap_alg")
    wrap = bool(wt.to_int(wrap_raw) or 0)
    alpha_mode = wt.get_value(record, "s_alpha_mode")
    alpha_text = str(alpha_mode).lower() if alpha_mode is not None else "none"

    if not wrap:
        return adapt_alg
    if alpha_text == "sigmoid":
        return f"{adapt_alg} + SAFER-A"
    return f"{adapt_alg} + SAFER"


def load_records(
    sweep_ids: Sequence[str],
    entity: str,
    project: str,
    wandb_timeout: Optional[int],
    verbose: bool,
) -> List[wt.RunRecord]:
    records: List[wt.RunRecord] = []
    for sweep_id in sweep_ids:
        runs = wt.load_runs_from_wandb_api(
            sweep_id,
            entity,
            project,
            api_timeout=wandb_timeout,
        )
        records.extend(runs)
        if verbose:
            path = sweep_id if "/" in sweep_id else f"{entity}/{project}/{sweep_id}"
            print(f"[info] Loaded {len(runs)} runs from {path}", file=sys.stderr)
    return records


def collect_curve_points(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Tuple[Dict[str, Dict[int, float]], Dict[str, Dict[int, float]]]:
    grouped: Dict[Tuple[str, int], List[wt.RunRecord]] = {}
    for record in records:
        if mean_key not in record.summary:
            continue
        dataset_name = wt.get_value(record, "dataset")
        if dataset_name != dataset:
            continue

        dom_val = wt.get_value(record, "test_envs", "test_env")
        if isinstance(dom_val, list):
            dom_val = dom_val[0] if dom_val else None
        dom = wt.to_int(dom_val)
        if dom != domain_id:
            continue

        rate = wt.to_int(wt.get_value(record, "attack_rate"))
        if rate is None:
            continue
        label = to_method_label(record)
        grouped.setdefault((label, rate), []).append(record)

    curve_mean: Dict[str, Dict[int, float]] = {}
    curve_std: Dict[str, Dict[int, float]] = {}
    for (label, rate), runs in grouped.items():
        chosen = wt.select_run(runs, select_mode, select_metric)
        mean_raw = chosen.summary.get(mean_key)
        std_raw = chosen.summary.get(std_key)
        try:
            mean_val = float(mean_raw)
        except (TypeError, ValueError):
            continue
        try:
            std_val = float(std_raw)
        except (TypeError, ValueError):
            std_val = 0.0
        curve_mean.setdefault(label, {})[rate] = mean_val
        curve_std.setdefault(label, {})[rate] = std_val

    if verbose:
        for label, rates in sorted(curve_mean.items()):
            keys = ",".join(str(k) for k in sorted(rates.keys()))
            print(f"[info] {label}: rates [{keys}]", file=sys.stderr)
    return curve_mean, curve_std


def sort_labels(labels: Sequence[str]) -> List[str]:
    preferred = ["Tent", "Tent + SAFER", "Tent + SAFER-A"]
    order: Dict[str, int] = {name: idx for idx, name in enumerate(preferred)}
    return sorted(labels, key=lambda x: (order.get(x, 999), x))


def write_points_csv(
    path: Path,
    curve_mean: Mapping[str, Mapping[int, float]],
    curve_std: Mapping[str, Mapping[int, float]],
) -> None:
    rows: List[Tuple[str, int, float, float]] = []
    for label, rate_map in curve_mean.items():
        for rate, value in rate_map.items():
            std = curve_std.get(label, {}).get(rate, 0.0)
            rows.append((label, rate, value, std))
    rows.sort(key=lambda x: (x[0], x[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "attack_rate", "acc_mean", "acc_std"])
        for row in rows:
            writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot attack-rate robustness curves from W&B sweep runs."
    )
    parser.add_argument("--sweep-ids", required=True, help="Comma-separated sweep IDs/paths.")
    parser.add_argument("--entity", default="bigslav", help="W&B entity for bare sweep IDs.")
    parser.add_argument("--project", default="safer", help="W&B project for bare sweep IDs.")
    parser.add_argument("--dataset", default="PACS", help="Dataset name.")
    parser.add_argument("--domain-id", type=int, default=0, help="Single domain id.")
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated method labels to include (e.g., 'Tent,Tent + SAFER').",
    )
    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for mean accuracy.")
    parser.add_argument("--std-key", default="acc_std", help="Summary key for std accuracy.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument(
        "--title",
        default=None,
        help="Optional plot title (default is auto-generated).",
    )
    parser.add_argument("--ymin", type=float, default=0.0, help="Y-axis minimum.")
    parser.add_argument("--ymax", type=float, default=100.0, help="Y-axis maximum.")
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--output", type=Path, default=Path("sweeps/attack_rate_curve.png"))
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("sweeps/attack_rate_curve_points.csv"),
        help="CSV export for plotted points.",
    )
    parser.add_argument("--wandb-timeout", type=int, default=60, help="W&B API timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")
    args = parser.parse_args()

    sweep_ids = parse_csv_strings(args.sweep_ids)
    if not sweep_ids:
        raise ValueError("No sweep IDs provided.")
    method_filter = set(parse_csv_strings(args.methods)) if args.methods else set()

    records = load_records(
        sweep_ids=sweep_ids,
        entity=args.entity,
        project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )
    select_metric = args.select_metric or args.mean_key
    curve_mean, curve_std = collect_curve_points(
        records=records,
        dataset=args.dataset,
        domain_id=args.domain_id,
        mean_key=args.mean_key,
        std_key=args.std_key,
        select_mode=args.select,
        select_metric=select_metric,
        verbose=args.verbose,
    )

    labels = sort_labels(list(curve_mean.keys()))
    if method_filter:
        labels = [label for label in labels if label in method_filter]
    if not labels:
        raise ValueError("No matching methods found for the selected dataset/domain.")

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    markers = ["o", "s", "^", "D", "P", "X"]

    for idx, label in enumerate(labels):
        points = sorted(curve_mean[label].items(), key=lambda x: x[0])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        yerr = [curve_std.get(label, {}).get(rate, 0.0) for rate in xs]
        marker = markers[idx % len(markers)]
        ax.errorbar(
            xs,
            ys,
            yerr=yerr,
            marker=marker,
            linewidth=1.8,
            markersize=5.2,
            capsize=3.0,
            label=label,
        )

    ax.set_xlabel("Attack rate (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    title = args.title or f"{args.dataset} domain {args.domain_id}: robustness curve"
    ax.set_title(title)
    ax.legend(frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)

    write_points_csv(args.csv_output, curve_mean=curve_mean, curve_std=curve_std)
    print(f"[ok] wrote plot: {args.output}")
    print(f"[ok] wrote points: {args.csv_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
