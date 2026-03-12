#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def coerce_plot_value(value: Any) -> Tuple[str, Any]:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, bool):
        return str(int(value)), int(value)
    if isinstance(value, int):
        return str(value), value
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            iv = int(round(value))
            return str(iv), iv
        return f"{value:g}", value
    if value is None:
        return "None", "None"
    text = str(value)
    return text, text


def load_records(
    sweep_ids: Sequence[str],
    entity: str,
    project: str,
    wandb_timeout: Optional[int],
    verbose: bool,
) -> List[wt.RunRecord]:
    out: List[wt.RunRecord] = []
    for sweep_id in sweep_ids:
        runs = wt.load_runs_from_wandb_api(
            sweep_id,
            entity,
            project,
            api_timeout=wandb_timeout,
        )
        out.extend(runs)
        if verbose:
            path = sweep_id if "/" in sweep_id else f"{entity}/{project}/{sweep_id}"
            print(f"[info] Loaded {len(runs)} runs from {path}", file=sys.stderr)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot metric vs a sweep parameter from W&B runs.")
    parser.add_argument("--sweep-ids", required=True, help="Comma-separated sweep IDs/paths.")
    parser.add_argument("--entity", default="bigslav", help="W&B entity for bare IDs.")
    parser.add_argument("--project", default="safer", help="W&B project for bare IDs.")
    parser.add_argument("--dataset", default="PACS", help="Dataset filter.")
    parser.add_argument("--domain-id", type=int, default=0, help="Domain filter.")
    parser.add_argument("--x-key", required=True, help="Config key for x-axis (e.g., s_num_views).")
    parser.add_argument("--attack-rates", default="0,50,100", help="Comma-separated attack rates for lines.")
    parser.add_argument("--filter", action="append", default=[], help="Extra filter key=value (repeatable).")
    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for y-axis.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument("--title", default=None, help="Optional title.")
    parser.add_argument("--xlabel", default=None, help="Optional x-axis label.")
    parser.add_argument("--ylabel", default="Accuracy (%)", help="Y-axis label.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, required=True, help="Output plot path.")
    parser.add_argument("--csv-output", type=Path, default=None, help="Optional CSV export.")
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sweep_ids = parse_csv_strings(args.sweep_ids)
    if not sweep_ids:
        raise ValueError("No sweep IDs provided.")
    attack_rates = parse_csv_ints(args.attack_rates)
    if not attack_rates:
        raise ValueError("No attack rates provided.")

    filters = wt.parse_filter_args(args.filter)
    filters.setdefault("dataset", [args.dataset])
    filters.setdefault("test_envs", [args.domain_id])

    records = load_records(
        sweep_ids=sweep_ids,
        entity=args.entity,
        project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )
    select_metric = args.select_metric or args.mean_key

    grouped: Dict[Tuple[str, Any, int], List[wt.RunRecord]] = {}
    for record in records:
        if args.mean_key not in record.summary:
            continue
        if not wt.record_matches_filters(record, filters):
            continue
        rate = wt.to_int(wt.get_value(record, "attack_rate"))
        if rate is None or rate not in attack_rates:
            continue
        x_raw = wt.get_value(record, args.x_key)
        x_label, x_sort = coerce_plot_value(x_raw)
        grouped.setdefault((x_label, x_sort, rate), []).append(record)

    points_by_rate: Dict[int, List[Tuple[str, Any, float]]] = {rate: [] for rate in attack_rates}
    for (x_label, x_sort, rate), runs in grouped.items():
        chosen = wt.select_run(runs, args.select, select_metric)
        mean_val = chosen.summary.get(args.mean_key)
        try:
            y = float(mean_val)
        except (TypeError, ValueError):
            continue
        points_by_rate[rate].append((x_label, x_sort, y))

    for rate in points_by_rate:
        points_by_rate[rate].sort(key=lambda x: (isinstance(x[1], str), x[1]))

    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    markers = ["o", "s", "^", "D", "P"]
    for idx, rate in enumerate(attack_rates):
        pts = points_by_rate.get(rate, [])
        if not pts:
            continue
        xs_lbl = [p[0] for p in pts]
        ys = [p[2] for p in pts]
        xs_idx = list(range(len(xs_lbl)))
        ax.plot(
            xs_idx,
            ys,
            marker=markers[idx % len(markers)],
            linewidth=1.8,
            markersize=5.0,
            label=f"{rate}%",
        )
        ax.set_xticks(xs_idx)
        ax.set_xticklabels(xs_lbl)

    ax.set_xlabel(args.xlabel or args.x_key)
    ax.set_ylabel(args.ylabel)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_title(args.title or f"{args.dataset} domain {args.domain_id}: {args.x_key} sensitivity")
    ax.legend(title="Attack rate", frameon=False)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"[ok] wrote plot: {args.output}")

    if args.csv_output:
        rows: List[List[object]] = []
        for rate in attack_rates:
            for x_label, _, y in points_by_rate.get(rate, []):
                rows.append([x_label, rate, y])
        args.csv_output.parent.mkdir(parents=True, exist_ok=True)
        with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([args.x_key, "attack_rate", args.mean_key])
            writer.writerows(rows)
        print(f"[ok] wrote csv: {args.csv_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
