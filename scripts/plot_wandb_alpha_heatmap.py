#!/usr/bin/env python3
"""Plot a (tau, kappa) accuracy heatmap for SAFER-A from W&B sweep runs.

E7: consolidates the four 1-D SAFER-A sensitivity panels (sweep 161jsdp8 /
plot_wandb_param_curve.py over s_alpha_conf_threshold, s_alpha_sigmoid_slope,
s_alpha_attack_value, s_alpha_clean_value) into a single 2-D heatmap of
accuracy over (tau = s_alpha_conf_threshold, kappa = s_alpha_sigmoid_slope),
one panel per attack rate.

Example:
  python scripts/plot_wandb_alpha_heatmap.py \
      --sweep-ids <tau_kappa_grid_sweep_id> \
      --dataset PACS --domain-id 0 --attack-rates 0,100 \
      --output sweeps/alpha_tau_kappa_heatmap_dom0.png \
      --csv-output sweeps/alpha_tau_kappa_heatmap_dom0.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
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


def load_records(
    sweep_ids: Sequence[str],
    entity: str,
    project: str,
    wandb_timeout: Optional[int],
    verbose: bool,
    extra_run_ids: Sequence[str] = (),
) -> List[wt.RunRecord]:
    records: List[wt.RunRecord] = []
    for sweep_id in sweep_ids:
        runs = wt.load_runs_from_wandb_api(sweep_id, entity, project, api_timeout=wandb_timeout)
        records.extend(runs)
        if verbose:
            path = sweep_id if "/" in sweep_id else f"{entity}/{project}/{sweep_id}"
            print(f"[info] Loaded {len(runs)} runs from {path}", file=sys.stderr)
    if extra_run_ids:
        import wandb  # type: ignore

        api = wandb.Api(timeout=wandb_timeout) if wandb_timeout and wandb_timeout > 0 else wandb.Api()
        for run_id in extra_run_ids:
            path = run_id if "/" in run_id else f"{entity}/{project}/{run_id}"
            run = api.run(path)
            config = {k: v for k, v in run.config.items() if not k.startswith("_")}
            records.append(wt.RunRecord(path=run.id, config=config, summary=dict(run.summary), mtime=0.0))
            if verbose:
                print(f"[info] Loaded standalone run {path}", file=sys.stderr)
    return records


def collect_grid(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    attack_rates: Sequence[int],
    mean_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Dict[int, Dict[Tuple[float, float], float]]:
    grouped: Dict[Tuple[int, float, float], List[wt.RunRecord]] = {}
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
        tau = wt.get_value(record, "s_alpha_conf_threshold")
        kappa = wt.get_value(record, "s_alpha_sigmoid_slope")
        if tau is None or kappa is None:
            continue
        grouped.setdefault((rate, float(tau), float(kappa)), []).append(record)

    grid: Dict[int, Dict[Tuple[float, float], float]] = {rate: {} for rate in attack_rates}
    for (rate, tau, kappa), runs in grouped.items():
        chosen = wt.select_run(runs, select_mode, select_metric)
        try:
            value = float(chosen.summary.get(mean_key))
        except (TypeError, ValueError):
            continue
        grid[rate][(tau, kappa)] = value

    if verbose:
        for rate in attack_rates:
            print(f"[info] rate={rate}%: {len(grid[rate])} (tau,kappa) cells", file=sys.stderr)
    return grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot a (tau, kappa) SAFER-A accuracy heatmap.")
    parser.add_argument("--sweep-ids", required=True, help="Comma-separated sweep IDs/paths.")
    parser.add_argument("--extra-run-ids", default="", help="Comma-separated standalone run IDs/paths to merge in (e.g. reruns of crashed sweep combos).")
    parser.add_argument("--entity", default="bigslav")
    parser.add_argument("--project", default="safer")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--attack-rates", default="0,100", help="Comma-separated attack rates, one panel each.")
    parser.add_argument("--mean-key", default="acc_mean")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument("--cmap", default="viridis")
    parser.add_argument("--annotate", action="store_true", help="Write accuracy values into cells.")
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("sweeps/alpha_tau_kappa_heatmap_dom0.png"))
    parser.add_argument("--csv-output", type=Path, default=Path("sweeps/alpha_tau_kappa_heatmap_dom0.csv"))
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    sweep_ids = parse_csv_strings(args.sweep_ids)
    if not sweep_ids:
        raise ValueError("No sweep IDs provided.")
    attack_rates = parse_csv_ints(args.attack_rates)
    if not attack_rates:
        raise ValueError("No attack rates provided.")

    records = load_records(
        sweep_ids=sweep_ids,
        entity=args.entity,
        project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
        extra_run_ids=parse_csv_strings(args.extra_run_ids),
    )
    select_metric = args.select_metric or args.mean_key
    grid = collect_grid(
        records=records,
        dataset=args.dataset,
        domain_id=args.domain_id,
        attack_rates=attack_rates,
        mean_key=args.mean_key,
        select_mode=args.select,
        select_metric=select_metric,
        verbose=args.verbose,
    )

    all_taus = sorted({tau for rate in attack_rates for (tau, _kappa) in grid[rate].keys()})
    all_kappas = sorted({kappa for rate in attack_rates for (_tau, kappa) in grid[rate].keys()})
    if not all_taus or not all_kappas:
        raise ValueError("No matching (tau, kappa) cells found for the selected dataset/domain/rates.")

    n_panels = len(attack_rates)
    fig, axes = plt.subplots(1, n_panels, figsize=(5.4 * n_panels, 4.6), squeeze=False)
    axes = axes[0]

    csv_rows: List[List[object]] = []
    im = None
    for panel_idx, rate in enumerate(attack_rates):
        ax = axes[panel_idx]
        matrix = np.full((len(all_kappas), len(all_taus)), np.nan)
        for (tau, kappa), value in grid[rate].items():
            r = all_kappas.index(kappa)
            c = all_taus.index(tau)
            matrix[r, c] = value
            csv_rows.append([rate, tau, kappa, value])

        im = ax.imshow(matrix, origin="lower", aspect="equal", cmap=args.cmap, vmin=0.0, vmax=100.0)
        ax.set_xticks(range(len(all_taus)))
        ax.set_xticklabels([f"{t:g}" for t in all_taus], rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(all_kappas)))
        ax.set_yticklabels([f"{k:g}" for k in all_kappas], fontsize=8)
        ax.set_xlabel(r"Confidence threshold $\tau$")
        if panel_idx == 0:
            ax.set_ylabel(r"Sigmoid slope $\kappa$")
        ax.set_title(f"Attack rate {rate}%")

        if args.annotate:
            for r in range(len(all_kappas)):
                for c in range(len(all_taus)):
                    val = matrix[r, c]
                    if not np.isnan(val):
                        ax.text(c, r, f"{val:.0f}", ha="center", va="center", fontsize=6.5, color="white")

    if im is not None:
        fig.colorbar(im, ax=axes.tolist(), shrink=0.85, label="Accuracy (%)")
    fig.suptitle(
        f"{wt.domain_label(args.dataset, args.domain_id)}: SAFER-A accuracy over "
        r"($\tau$, $\kappa$)",
        fontsize=11,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[ok] wrote plot: {args.output}")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    csv_rows.sort(key=lambda row: (row[0], row[1], row[2]))
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["attack_rate", "tau", "kappa", args.mean_key])
        writer.writerows(csv_rows)
    print(f"[ok] wrote csv: {args.csv_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
