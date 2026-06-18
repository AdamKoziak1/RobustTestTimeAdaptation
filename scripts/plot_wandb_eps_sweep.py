#!/usr/bin/env python3
"""Plot accuracy vs Linf perturbation budget (epsilon) from W&B sweep runs.

Companion to plot_wandb_attack_rate_curve.py: instead of grouping curves by
method and plotting accuracy vs attack_rate, this groups by method and plots
accuracy vs the epsilon parsed out of the run's `attack` config string
(e.g. "linf_eps-8.0_steps-20" -> 8.0). Used for fig:supp-eps-sweep.

Example:
  python scripts/plot_wandb_eps_sweep.py \
      --sweep-ids <sweep_id> \
      --dataset PACS --domain-id 0 \
      --output sweeps/eps_sweep_dom0.png \
      --csv-output sweeps/eps_sweep_dom0.csv
"""

from __future__ import annotations

import argparse
import csv
import re
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

EPS_RE = re.compile(r"eps-([0-9]+(?:\.[0-9]+)?)")


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
    primary_pool = wt.get_value(record, "s_primary_view_pool")
    pool_text = str(primary_pool).lower() if primary_pool is not None else ""

    if not wrap:
        return adapt_alg
    if adapt_alg == "ERM" and pool_text == "mean":
        return "Static TTE (mean)"
    if alpha_text == "sigmoid":
        return f"{adapt_alg} + SAFER-A"
    if pool_text == "mean":
        return f"{adapt_alg} + SAFER (mean)"
    return f"{adapt_alg} + SAFER"


def parse_eps(record: wt.RunRecord) -> Optional[float]:
    attack = wt.get_value(record, "attack")
    if not isinstance(attack, str):
        return None
    match = EPS_RE.search(attack)
    if not match:
        return None
    return float(match.group(1))


def sort_labels(labels: Sequence[str]) -> List[str]:
    preferred = ["Static TTE (mean)", "Tent", "Tent + SAFER", "Tent + SAFER (mean)", "Tent + SAFER-A"]
    order: Dict[str, int] = {name: idx for idx, name in enumerate(preferred)}
    return sorted(labels, key=lambda x: (order.get(x, 999), x))


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


def collect_curve_points(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    attack_rate: int,
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Tuple[Dict[str, Dict[float, float]], Dict[str, Dict[float, float]]]:
    grouped: Dict[Tuple[str, float], List[wt.RunRecord]] = {}
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
        if rate != attack_rate:
            continue

        eps = parse_eps(record)
        if eps is None:
            continue

        label = to_method_label(record)
        grouped.setdefault((label, eps), []).append(record)

    curve_mean: Dict[str, Dict[float, float]] = {}
    curve_std: Dict[str, Dict[float, float]] = {}
    for (label, eps), runs in grouped.items():
        chosen = wt.select_run(runs, select_mode, select_metric)
        try:
            mean_val = float(chosen.summary.get(mean_key))
        except (TypeError, ValueError):
            continue
        try:
            std_val = float(chosen.summary.get(std_key))
        except (TypeError, ValueError):
            std_val = 0.0
        curve_mean.setdefault(label, {})[eps] = mean_val
        curve_std.setdefault(label, {})[eps] = std_val

    if verbose:
        for label, eps_map in sorted(curve_mean.items()):
            keys = ",".join(str(k) for k in sorted(eps_map.keys()))
            print(f"[info] {label}: eps [{keys}]", file=sys.stderr)
    return curve_mean, curve_std


def write_points_csv(
    path: Path,
    curve_mean: Mapping[str, Mapping[float, float]],
    curve_std: Mapping[str, Mapping[float, float]],
) -> None:
    rows: List[Tuple[str, float, float, float]] = []
    for label, eps_map in curve_mean.items():
        for eps, value in eps_map.items():
            rows.append((label, eps, value, curve_std.get(label, {}).get(eps, 0.0)))
    rows.sort(key=lambda x: (x[0], x[1]))
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["method", "eps_pixel_scale", "acc_mean", "acc_std"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs Linf perturbation budget (epsilon) from W&B sweep runs."
    )
    parser.add_argument("--sweep-ids", required=True, help="Comma-separated sweep IDs/paths.")
    parser.add_argument("--entity", default="bigslav")
    parser.add_argument("--project", default="safer")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--attack-rate", type=int, default=100, help="Fixed attack rate to filter on.")
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated method labels to include (e.g., 'Tent,Tent + SAFER').",
    )
    parser.add_argument("--mean-key", default="acc_mean")
    parser.add_argument("--std-key", default="acc_std")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument("--title", default=None)
    parser.add_argument("--ymin", type=float, default=0.0)
    parser.add_argument("--ymax", type=float, default=100.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("sweeps/eps_sweep_dom0.png"))
    parser.add_argument("--csv-output", type=Path, default=Path("sweeps/eps_sweep_dom0.csv"))
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
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
        attack_rate=args.attack_rate,
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
        raise ValueError("No matching methods found for the selected dataset/domain/attack_rate.")

    all_eps: set = set()
    for label in labels:
        all_eps.update(curve_mean[label].keys())

    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    markers = ["o", "s", "^", "D", "P", "X"]

    for idx, label in enumerate(labels):
        points = sorted(curve_mean[label].items(), key=lambda x: x[0])
        if not points:
            continue
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        yerr = [curve_std.get(label, {}).get(eps, 0.0) for eps in xs]
        line, = ax.plot(
            xs,
            ys,
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=5.2,
            label=label,
        )
        lo = [y - e for y, e in zip(ys, yerr)]
        hi = [y + e for y, e in zip(ys, yerr)]
        ax.fill_between(xs, lo, hi, color=line.get_color(), alpha=0.15, linewidth=0)

    ax.set_xlabel(r"Perturbation budget $\varepsilon$ (/255, $\ell_\infty$)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xticks(sorted(all_eps))
    ax.set_ylim(args.ymin, args.ymax)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    title = args.title or (
        f"{wt.domain_label(args.dataset, args.domain_id)}: Accuracy vs perturbation budget\n"
        f"(attack rate {args.attack_rate}%)"
    )
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
