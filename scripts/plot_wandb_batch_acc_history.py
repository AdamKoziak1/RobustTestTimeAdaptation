#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from collections import deque
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METRIC_DISPLAY = {
    "batch_acc": "Batch Accuracy",
}

METRIC_YLABEL = {
    "batch_acc": "Accuracy (%)",
}


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


def parse_legend_loc(text: str):
    parts = [p.strip() for p in text.split(",")]
    if len(parts) == 2:
        try:
            return (float(parts[0]), float(parts[1]))
        except ValueError:
            pass
    return text


def load_history_from_csv(path: Path) -> Dict[Tuple[str, str], Tuple[List[int], List[float], List[float]]]:
    history: Dict[Tuple[str, str], Tuple[List[int], List[float], List[float]]] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            key = (row["label"], row["metric"])
            steps, vals, smooth = history.setdefault(key, ([], [], []))
            steps.append(int(row["step"]))
            vals.append(float(row["value"]))
            smooth.append(float(row["rolling_value"]))
    return history


def normalize_run_path(run_ref: str, entity: str, project: str) -> str:
    if run_ref.count("/") == 2:
        return run_ref
    if run_ref.count("/") == 1:
        return run_ref
    return f"{entity}/{project}/{run_ref}"


def fetch_metric_history(api, run_path: str, metric: str) -> Tuple[List[int], List[float]]:
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
        writer.writerow(["label", "metric", "step", "value", "rolling_value"])
        writer.writerows(rows)


def metric_display_name(metric: str) -> str:
    return METRIC_DISPLAY.get(metric, metric)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot W&B history stability curves for one or more runs.")
    parser.add_argument("--run-ids", default="", help="Comma-separated run IDs/paths.")
    parser.add_argument("--labels", default="", help="Optional comma-separated labels matching --run-ids.")
    parser.add_argument("--entity", default="bigslav", help="W&B entity for bare run IDs.")
    parser.add_argument("--project", default="safer", help="W&B project for bare run IDs.")
    parser.add_argument(
        "--metrics",
        default="batch_acc",
        help="Comma-separated history metrics to plot (e.g., batch_acc,loss).",
    )
    parser.add_argument("--window", type=int, default=25, help="Rolling window size.")
    parser.add_argument("--legend-loc", default="best", help="Matplotlib legend location.")
    parser.add_argument("--legend-strip", default="", help="Substring to remove from legend labels (display only).")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--fig-width", type=float, default=6.8, help="Figure width in inches.")
    parser.add_argument(
        "--fig-height-per-metric", type=float, default=3.2, help="Figure height per metric row, in inches."
    )
    parser.add_argument("--no-title", action="store_true", help="Omit subplot titles.")
    parser.add_argument("--output", type=Path, default=Path("sweeps/batch_acc_stability.png"))
    parser.add_argument("--csv-output", type=Path, default=Path("sweeps/batch_acc_stability.csv"))
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument(
        "--from-csv", type=Path, default=None, help="Re-render from a previously exported history CSV instead of querying W&B."
    )
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    metrics = parse_csv_strings(args.metrics)
    if not metrics:
        raise ValueError("No metrics provided.")

    history_from_csv: Optional[Dict[Tuple[str, str], Tuple[List[int], List[float], List[float]]]] = None
    api = None
    if args.from_csv:
        history_from_csv = load_history_from_csv(args.from_csv)
        if args.labels:
            labels = parse_csv_strings(args.labels)
        else:
            seen: List[str] = []
            for (label, metric) in history_from_csv.keys():
                if metric == metrics[0] and label not in seen:
                    seen.append(label)
            labels = seen
        run_ids = labels
    else:
        run_ids = parse_csv_strings(args.run_ids)
        if not run_ids:
            raise ValueError("No run IDs provided.")
        labels = parse_csv_strings(args.labels) if args.labels else run_ids
        if len(labels) != len(run_ids):
            raise ValueError("--labels must match --run-ids length.")

        import wandb  # type: ignore

        api = wandb.Api(timeout=args.wandb_timeout) if args.wandb_timeout > 0 else wandb.Api()

    fig, axes = plt.subplots(
        len(metrics), 1, figsize=(args.fig_width, args.fig_height_per_metric * len(metrics)), squeeze=False
    )
    csv_rows: List[Tuple[str, str, int, float, float]] = []

    for metric_idx, metric in enumerate(metrics):
        ax = axes[metric_idx][0]
        any_line = False
        for run_ref, label in zip(run_ids, labels):
            if history_from_csv is not None:
                steps, vals, smooth = history_from_csv.get((label, metric), ([], [], []))
                if not vals:
                    if args.verbose:
                        print(f"[warn] No {metric} history for {label}", file=sys.stderr)
                    continue
            else:
                run_path = normalize_run_path(run_ref, args.entity, args.project)
                steps, vals = fetch_metric_history(api, run_path, metric)
                if not vals:
                    if args.verbose:
                        print(f"[warn] No {metric} history for {run_path}", file=sys.stderr)
                    continue
                smooth = rolling_mean(vals, args.window)
            display_label = label.replace(args.legend_strip, "") if args.legend_strip else label
            ax.plot(steps, smooth, linewidth=1.8, label=display_label)
            csv_rows.extend((label, metric, s, v, m) for s, v, m in zip(steps, vals, smooth))
            any_line = True
            if args.verbose:
                print(f"[info] {label} {metric}: {len(vals)} points", file=sys.stderr)

        metric_name = metric_display_name(metric)
        ax.set_xlabel("Step")
        ax.set_ylabel(METRIC_YLABEL.get(metric, metric_name))
        if not args.no_title:
            ax.set_title(f"{metric_name} (rolling window={args.window})")
        ax.grid(True, linewidth=0.4, alpha=0.35)
        if any_line:
            ax.legend(frameon=False, loc=parse_legend_loc(args.legend_loc))

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
