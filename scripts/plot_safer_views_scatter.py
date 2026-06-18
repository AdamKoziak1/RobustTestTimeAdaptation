#!/usr/bin/env python3
"""Plot accuracy vs. wall-clock cost for SAFER with N=1..4 augmented views.

E8: pairs with the compute table (E3 / tab:compute) to justify a cheap N=2
operating point. Joins:
  - measured per-step wall-clock from scripts/bench_safer_overhead.py
    (sweeps/compute_overhead_measured.csv, rows "Tent+SAFER (N=k)")
  - accuracy from the existing views-1-to-4 ablation sweep
    (sweeps/ablation_views_1to4_tent_pacs_dom0.yaml, cached id mxif4s7w),
    grouped by attack_rate

into one scatter plot: x = step time (ms), y = accuracy (%), one marker per
(N, attack_rate), connected by attack-rate-colored lines across N=1..4.

Example:
  python scripts/plot_safer_views_scatter.py \
      --sweep-ids mxif4s7w \
      --overhead-csv sweeps/compute_overhead_measured.csv \
      --dataset PACS --domain-id 0 --attack-rates 0,50,100 \
      --output sweeps/views_acc_vs_wallclock_dom0.png \
      --csv-output sweeps/views_acc_vs_wallclock_dom0.csv
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402

OVERHEAD_LABEL_RE = re.compile(r"SAFER\s*\(N=(\d+)\)")


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


OVERHEAD_METRICS = {
    "step_time": {
        "csv_col": "step_time_ms_mean",
        "annotate_fmt": "{:.0f} ms/step",
        "axis_label": "Measured step time (ms)",
    },
    "gflops": {
        "csv_col": "gflops_per_step",
        "annotate_fmt": "{:.0f} GFLOPs/step",
        "axis_label": "Compute per step (GFLOPs)",
    },
}


def load_overhead_metric(path: Path, metric: str) -> Dict[int, float]:
    """Return {num_views N: metric value} parsed from the overhead CSV.

    `metric` selects which measured column to use (see OVERHEAD_METRICS):
    wall-clock step time (ms) or analytic compute (GFLOPs/step).
    """
    csv_col = OVERHEAD_METRICS[metric]["csv_col"]
    values: Dict[int, float] = {}
    with path.open("r", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            match = OVERHEAD_LABEL_RE.search(row.get("method", ""))
            if not match:
                continue
            n = int(match.group(1))
            values[n] = float(row[csv_col])
    if not values:
        raise ValueError(
            f"No 'Tent+SAFER (N=k)' rows found in {path}. "
            "Run scripts/bench_safer_overhead.py first (it now covers N=1..4)."
        )
    return values


def load_records(
    sweep_ids: Sequence[str],
    entity: str,
    project: str,
    wandb_timeout: Optional[int],
    verbose: bool,
) -> List[wt.RunRecord]:
    records: List[wt.RunRecord] = []
    for sweep_id in sweep_ids:
        runs = wt.load_runs_from_wandb_api(sweep_id, entity, project, api_timeout=wandb_timeout)
        records.extend(runs)
        if verbose:
            path = sweep_id if "/" in sweep_id else f"{entity}/{project}/{sweep_id}"
            print(f"[info] Loaded {len(runs)} runs from {path}", file=sys.stderr)
    return records


def collect_accuracy(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    attack_rates: Sequence[int],
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Dict[int, Dict[int, Tuple[float, float]]]:
    """Return {attack_rate: {num_views N: (acc_mean, acc_std)}}."""
    grouped: Dict[Tuple[int, int], List[wt.RunRecord]] = {}
    for record in records:
        if mean_key not in record.summary:
            continue
        if wt.get_value(record, "dataset") != dataset:
            continue
        dom_val = wt.get_value(record, "test_envs", "test_env")
        if isinstance(dom_val, list):
            dom_val = dom_val[0] if dom_val else None
        if wt.to_int(dom_val) != domain_id:
            continue
        rate = wt.to_int(wt.get_value(record, "attack_rate"))
        if rate is None or rate not in attack_rates:
            continue
        n_views = wt.to_int(wt.get_value(record, "s_num_views"))
        if n_views is None:
            continue
        grouped.setdefault((rate, n_views), []).append(record)

    out: Dict[int, Dict[int, Tuple[float, float]]] = {rate: {} for rate in attack_rates}
    for (rate, n_views), runs in grouped.items():
        chosen = wt.select_run(runs, select_mode, select_metric)
        try:
            mean_val = float(chosen.summary.get(mean_key))
        except (TypeError, ValueError):
            continue
        try:
            std_val = float(chosen.summary.get(std_key))
        except (TypeError, ValueError):
            std_val = 0.0
        out[rate][n_views] = (mean_val, std_val)

    if verbose:
        for rate in attack_rates:
            keys = ",".join(str(k) for k in sorted(out[rate].keys()))
            print(f"[info] rate={rate}%: views [{keys}]", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Scatter accuracy vs. measured wall-clock for SAFER N=1..4."
    )
    parser.add_argument("--sweep-ids", required=True, help="Comma-separated W&B sweep IDs (views_1to4).")
    parser.add_argument("--entity", default="bigslav")
    parser.add_argument("--project", default="safer")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--attack-rates", default="0,50,100")
    parser.add_argument(
        "--overhead-csv",
        type=Path,
        default=Path("sweeps/compute_overhead_measured.csv"),
        help="CSV produced by scripts/bench_safer_overhead.py.",
    )
    parser.add_argument("--mean-key", default="acc_mean")
    parser.add_argument("--std-key", default="acc_std")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument("--xlabel", default="Number of augmented views $N$")
    parser.add_argument("--ylabel", default="Accuracy (%)")
    parser.add_argument(
        "--overhead-metric", choices=["step_time", "gflops"], default="step_time",
        help="Which measured compute-overhead metric to display alongside accuracy: "
             "wall-clock step time in ms (step_time) or analytic compute in "
             "GFLOPs/step (gflops). Both come from --overhead-csv.",
    )
    parser.add_argument(
        "--step-time-style", choices=["annotate", "twinx", "none"], default="annotate",
        help="How to show the overhead metric: inline text labels (annotate), "
             "a separate line on a parallel right-hand axis (twinx), or omit it (none).",
    )
    parser.add_argument("--step-time-ylabel", default=None, help="Override the twinx axis label (default depends on --overhead-metric).")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("sweeps/views_acc_vs_wallclock_dom0.png"))
    parser.add_argument("--csv-output", type=Path, default=Path("sweeps/views_acc_vs_wallclock_dom0.csv"))
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sweep_ids = parse_csv_strings(args.sweep_ids)
    if not sweep_ids:
        raise ValueError("No sweep IDs provided.")
    attack_rates = parse_csv_ints(args.attack_rates)
    if not attack_rates:
        raise ValueError("No attack rates provided.")

    overhead = load_overhead_metric(args.overhead_csv, args.overhead_metric)
    metric_spec = OVERHEAD_METRICS[args.overhead_metric]
    records = load_records(
        sweep_ids=sweep_ids,
        entity=args.entity,
        project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )
    select_metric = args.select_metric or args.mean_key
    accuracy = collect_accuracy(
        records=records,
        dataset=args.dataset,
        domain_id=args.domain_id,
        attack_rates=attack_rates,
        mean_key=args.mean_key,
        std_key=args.std_key,
        select_mode=args.select,
        select_metric=select_metric,
        verbose=args.verbose,
    )

    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    markers = ["o", "s", "^", "D", "P", "X"]
    csv_rows: List[List[object]] = []

    any_line = False
    all_n_views: List[int] = []
    for idx, rate in enumerate(attack_rates):
        acc_by_n = accuracy.get(rate, {})
        points = []
        for n_views in sorted(acc_by_n):
            if n_views not in overhead:
                continue
            mean_acc, std_acc = acc_by_n[n_views]
            overhead_val = overhead[n_views]
            points.append((n_views, overhead_val, mean_acc, std_acc))
            csv_rows.append([rate, n_views, overhead_val, mean_acc, std_acc])
        if not points:
            continue
        any_line = True
        xs = [p[0] for p in points]
        ys = [p[2] for p in points]
        yerr = [p[3] for p in points]
        all_n_views.extend(xs)
        marker = markers[idx % len(markers)]
        line, = ax.plot(
            xs, ys,
            marker=marker, linewidth=1.6, markersize=6.5,
            label=f"attack rate {rate}%",
        )
        lo = [y - e for y, e in zip(ys, yerr)]
        hi = [y + e for y, e in zip(ys, yerr)]
        ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15, linewidth=0)

    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    handles, labels = ax.get_legend_handles_labels()
    n_ticks = sorted(set(all_n_views))
    if n_ticks:
        ax.set_xticks(n_ticks)

    if args.step_time_style == "annotate":
        for n_views in n_ticks:
            val = overhead.get(n_views)
            if val is not None:
                ax.annotate(
                    metric_spec["annotate_fmt"].format(val), xy=(n_views, 0.02), xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7.5, color="0.4",
                )
    elif args.step_time_style == "twinx" and n_ticks:
        ax2 = ax.twinx()
        overhead_color = "0.35"
        overhead_xs = n_ticks
        overhead_ys = [overhead[n] for n in overhead_xs]
        line2 = ax2.plot(
            overhead_xs, overhead_ys, marker="x", linestyle="--", linewidth=1.4, markersize=7,
            color=overhead_color, label="Measured step time" if args.overhead_metric == "step_time" else "Compute (GFLOPs/step)",
        )
        ax2.set_ylabel(args.step_time_ylabel or metric_spec["axis_label"], color=overhead_color)
        ax2.tick_params(axis="y", colors=overhead_color)
        ax2.spines["right"].set_color(overhead_color)
        handles2, labels2 = ax2.get_legend_handles_labels()
        handles, labels = handles + handles2, labels + labels2

    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_title(
        args.title
        or f"{wt.domain_label(args.dataset, args.domain_id)}: SAFER sensitivity to number of views (N=1..4)"
    )
    if handles:
        ax.legend(handles, labels, frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"[ok] wrote plot: {args.output}")

    csv_rows.sort(key=lambda row: (row[0], row[1]))
    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        overhead_col = "step_time_ms" if args.overhead_metric == "step_time" else "gflops_per_step"
        writer.writerow(["attack_rate", "num_views", overhead_col, "acc_mean", "acc_std"])
        writer.writerows(csv_rows)
    print(f"[ok] wrote csv: {args.csv_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
