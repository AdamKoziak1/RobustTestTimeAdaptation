#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def rolling_mean(values: Sequence[float], window: int) -> List[float]:
    if window <= 1:
        return list(values)
    out: List[float] = []
    q: deque[float] = deque()
    running = 0.0
    for v in values:
        q.append(v)
        running += v
        if len(q) > window:
            running -= q.popleft()
        out.append(running / len(q))
    return out


def normalize_run_path(run_ref: str, entity: str, project: str) -> str:
    if run_ref.count("/") == 2:
        return run_ref
    if run_ref.count("/") == 1:
        return run_ref
    return f"{entity}/{project}/{run_ref}"


def fetch_batch_acc_history(api, run_path: str, metric: str) -> Tuple[List[int], List[float]]:
    steps: List[int] = []
    values: List[float] = []
    run = api.run(run_path)
    for idx, row in enumerate(run.scan_history(keys=[metric])):
        if metric not in row:
            continue
        try:
            val = float(row[metric])
        except (TypeError, ValueError):
            continue
        step_val = row.get("_step")
        if isinstance(step_val, int):
            step = step_val
        else:
            step = idx
        steps.append(step)
        values.append(val)
    return steps, values


def write_csv(path: Path, rows: Iterable[Tuple[str, int, float, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["label", "step", "batch_acc", "rolling_batch_acc"])
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot W&B batch_acc stability curves for one or more runs.")
    parser.add_argument("--run-ids", required=True, help="Comma-separated run IDs/paths.")
    parser.add_argument("--labels", default="", help="Optional comma-separated labels matching --run-ids.")
    parser.add_argument("--entity", default="your_wandb_entity", help="W&B entity for bare run IDs.")
    parser.add_argument("--project", default="your_wandb_project", help="W&B project for bare run IDs.")
    parser.add_argument("--metric", default="batch_acc", help="History metric to plot.")
    parser.add_argument("--window", type=int, default=25, help="Rolling window size.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("sweeps/batch_acc_stability.png"))
    parser.add_argument("--csv-output", type=Path, default=Path("sweeps/batch_acc_stability.csv"))
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    run_ids = parse_csv_strings(args.run_ids)
    if not run_ids:
        raise ValueError("No run IDs provided.")
    labels = parse_csv_strings(args.labels) if args.labels else run_ids
    if len(labels) != len(run_ids):
        raise ValueError("--labels must match --run-ids length.")

    import wandb  # type: ignore

    api = wandb.Api(timeout=args.wandb_timeout) if args.wandb_timeout > 0 else wandb.Api()

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    csv_rows: List[Tuple[str, int, float, float]] = []
    for run_ref, label in zip(run_ids, labels):
        run_path = normalize_run_path(run_ref, args.entity, args.project)
        steps, vals = fetch_batch_acc_history(api, run_path, args.metric)
        if not vals:
            if args.verbose:
                print(f"[warn] No {args.metric} history for {run_path}", file=sys.stderr)
            continue
        smooth = rolling_mean(vals, args.window)
        ax.plot(steps, smooth, linewidth=1.8, label=label)
        csv_rows.extend((label, s, v, m) for s, v, m in zip(steps, vals, smooth))
        if args.verbose:
            print(f"[info] {label}: {len(vals)} points", file=sys.stderr)

    ax.set_xlabel("Step")
    ax.set_ylabel(args.metric)
    ax.set_title(f"Stability over time ({args.metric}, rolling window={args.window})")
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.legend(frameon=False)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    write_csv(args.csv_output, csv_rows)
    print(f"[ok] wrote plot: {args.output}")
    print(f"[ok] wrote csv: {args.csv_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
