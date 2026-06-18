#!/usr/bin/env python3
"""Plot accuracy vs. EOT strength K for the defense-aware (E10) SAFER attack.

Joins two sweeps:
  - sweeps/eot_safer_defense_aware_attack_pacs_dom0.yaml (Tent+SAFER,
    s_wrap_alg=1, attack_eot_views in {1,4,8}): one curve, accuracy vs K.
  - sweeps/eot_safer_defense_aware_attack_baseline_tent_pacs_dom0.yaml
    (bare Tent, s_wrap_alg=0, attack_eot_views=1 only): a single reference
    point, drawn as a horizontal dashed line (EOT averaging is a no-op for a
    deterministic model, so one K is enough).

Both runs share attack_source=live, attack_preset=linf8, attack_steps=10,
attack_rate=100 - the only thing that differs is whether SAFER is in the loop
and how many stochastic realizations (K) the adversary averages over per PGD
step before taking its ascent step (Expectation over Transformation).

Example:
  python scripts/plot_wandb_eot_safer_attack.py \
      --safer-sweep-ids <eot_safer_sweep_id> \
      --baseline-sweep-ids <eot_safer_baseline_sweep_id> \
      --dataset PACS --domain-id 0 \
      --output sweeps/eot_safer_attack_dom0.png \
      --csv-output sweeps/eot_safer_attack_dom0.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


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


def _matches_common_filters(record: wt.RunRecord, dataset: str, domain_id: int, mean_key: str) -> bool:
    if mean_key not in record.summary:
        return False
    if wt.get_value(record, "dataset") != dataset:
        return False
    dom_val = wt.get_value(record, "test_envs", "test_env")
    if isinstance(dom_val, list):
        dom_val = dom_val[0] if dom_val else None
    return wt.to_int(dom_val) == domain_id


def collect_safer_curve(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Dict[int, Tuple[float, float]]:
    """Return {K (attack_eot_views): (acc_mean, acc_std)} for s_wrap_alg=1 runs."""
    grouped: Dict[int, List[wt.RunRecord]] = {}
    for record in records:
        if not _matches_common_filters(record, dataset, domain_id, mean_key):
            continue
        if wt.to_int(wt.get_value(record, "s_wrap_alg")) != 1:
            continue
        k = wt.to_int(wt.get_value(record, "attack_eot_views"))
        if k is None:
            continue
        grouped.setdefault(k, []).append(record)

    out: Dict[int, Tuple[float, float]] = {}
    for k, runs in grouped.items():
        chosen = wt.select_run(runs, select_mode, select_metric)
        try:
            mean_val = float(chosen.summary.get(mean_key))
        except (TypeError, ValueError):
            continue
        try:
            std_val = float(chosen.summary.get(std_key))
        except (TypeError, ValueError):
            std_val = 0.0
        out[k] = (mean_val, std_val)

    if verbose:
        keys = ",".join(str(k) for k in sorted(out))
        print(f"[info] Tent+SAFER: K values [{keys}]", file=sys.stderr)
    return out


def collect_baseline_point(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Optional[Tuple[float, float]]:
    """Return (acc_mean, acc_std) for the single bare-Tent (s_wrap_alg=0) reference run."""
    matches: List[wt.RunRecord] = []
    for record in records:
        if not _matches_common_filters(record, dataset, domain_id, mean_key):
            continue
        if wt.to_int(wt.get_value(record, "s_wrap_alg")) != 0:
            continue
        matches.append(record)
    if not matches:
        if verbose:
            print("[warn] no bare-Tent (s_wrap_alg=0) baseline run found", file=sys.stderr)
        return None
    chosen = wt.select_run(matches, select_mode, select_metric)
    try:
        mean_val = float(chosen.summary.get(mean_key))
    except (TypeError, ValueError):
        return None
    try:
        std_val = float(chosen.summary.get(std_key))
    except (TypeError, ValueError):
        std_val = 0.0
    if verbose:
        print(f"[info] Tent baseline: acc = {mean_val:.2f} +/- {std_val:.2f}", file=sys.stderr)
    return mean_val, std_val


def _variant_alpha_mode(record: wt.RunRecord) -> str:
    alpha_mode = wt.get_value(record, "s_alpha_mode")
    return str(alpha_mode).lower() if alpha_mode is not None else "none"


def _variant_pool(record: wt.RunRecord) -> str:
    pool = wt.get_value(record, "s_primary_view_pool")
    return str(pool).lower() if pool is not None else ""


def is_safer_base_variant(record: wt.RunRecord) -> bool:
    """Reliability-guided (cc_drop) SAFER, no adaptive mixing -- the base curve."""
    return _variant_alpha_mode(record) != "sigmoid" and _variant_pool(record) != "mean"


def is_safer_a_variant(record: wt.RunRecord) -> bool:
    """SAFER-A (adaptive-mixing, s_alpha_mode=sigmoid) variant."""
    return _variant_alpha_mode(record) == "sigmoid"


def is_safer_mean_variant(record: wt.RunRecord) -> bool:
    """SAFER with uniform mean view-pooling (no reliability weighting)."""
    return _variant_alpha_mode(record) != "sigmoid" and _variant_pool(record) == "mean"


def collect_clean_point(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
    label: str = "Tent+SAFER",
    variant_filter: Optional[Callable[[wt.RunRecord], bool]] = None,
) -> Optional[Tuple[float, float]]:
    """Return (acc_mean, acc_std) for the unattacked (attack_rate=0) reference
    of a given SAFER variant -- a natural K=0 anchor ("no adaptive attack at
    all"). Clean accuracy doesn't depend on the attack config, so this can be
    sourced from any sweep that ran the same model/data/seeds/variant at
    attack_rate=0 (e.g. the main robustness-curve sweep). `variant_filter`
    disambiguates which SAFER variant (base / SAFER-A / mean-pooling) a clean
    run belongs to, since a single shared sweep may contain several."""
    matches: List[wt.RunRecord] = []
    for record in records:
        if not _matches_common_filters(record, dataset, domain_id, mean_key):
            continue
        if wt.to_int(wt.get_value(record, "s_wrap_alg")) != 1:
            continue
        if wt.to_int(wt.get_value(record, "attack_rate")) != 0:
            continue
        if variant_filter is not None and not variant_filter(record):
            continue
        matches.append(record)
    if not matches:
        if verbose:
            print(f"[warn] no clean (attack_rate=0, s_wrap_alg=1) reference run found for '{label}'", file=sys.stderr)
        return None
    chosen = wt.select_run(matches, select_mode, select_metric)
    try:
        mean_val = float(chosen.summary.get(mean_key))
    except (TypeError, ValueError):
        return None
    try:
        std_val = float(chosen.summary.get(std_key))
    except (TypeError, ValueError):
        std_val = 0.0
    if verbose:
        print(f"[info] {label} clean (K=0): acc = {mean_val:.2f} +/- {std_val:.2f}", file=sys.stderr)
    return mean_val, std_val


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Plot accuracy vs. EOT strength K for the defense-aware SAFER attack (E10)."
    )
    parser.add_argument("--safer-sweep-ids", required=True, help="Comma-separated W&B sweep IDs (Tent+SAFER, K sweep).")
    parser.add_argument(
        "--safer-a-sweep-ids", default=None,
        help="Optional comma-separated W&B sweep IDs (Tent+SAFER-A variant, "
             "s_alpha_mode=sigmoid, K sweep) -- plotted as an extra curve "
             "showing the clean-retention tradeoff under EOT attack.",
    )
    parser.add_argument(
        "--safer-mean-sweep-ids", default=None,
        help="Optional comma-separated W&B sweep IDs (Tent+SAFER with mean/"
             "uniform view pooling instead of reliability-guided cc_drop, "
             "K sweep) -- plotted as an extra curve showing the robustness "
             "loss from removing the reliability weighting under EOT attack.",
    )
    parser.add_argument("--baseline-sweep-ids", required=True, help="Comma-separated W&B sweep IDs (bare-Tent reference).")
    parser.add_argument(
        "--clean-sweep-ids",
        default=None,
        help="Optional comma-separated W&B sweep IDs providing an unattacked "
             "(attack_rate=0) Tent+SAFER reference, plotted as a K=0 anchor "
             "point on the base Tent+SAFER curve (e.g. the main "
             "robustness-curve sweep).",
    )
    parser.add_argument(
        "--safer-a-clean-sweep-ids",
        default=None,
        help="Optional comma-separated W&B sweep IDs providing an unattacked "
             "(attack_rate=0, s_alpha_mode=sigmoid) Tent+SAFER-A reference, "
             "plotted as a K=0 anchor point on the SAFER-A curve.",
    )
    parser.add_argument(
        "--safer-mean-clean-sweep-ids",
        default=None,
        help="Optional comma-separated W&B sweep IDs providing an unattacked "
             "(attack_rate=0, s_primary_view_pool=mean) Tent+SAFER (mean) "
             "reference, plotted as a K=0 anchor point on the SAFER (mean) curve.",
    )
    parser.add_argument("--entity", default="bigslav")
    parser.add_argument("--project", default="safer")
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--domain-id", type=int, default=0)
    parser.add_argument("--mean-key", default="acc_mean")
    parser.add_argument("--std-key", default="acc_std")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument("--xlabel", default="EOT samples per PGD step, $K$")
    parser.add_argument("--ylabel", default="Accuracy (%)")
    parser.add_argument("--title", default=None)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument("--output", type=Path, default=Path("sweeps/eot_safer_attack_dom0.png"))
    parser.add_argument("--csv-output", type=Path, default=Path("sweeps/eot_safer_attack_dom0.csv"))
    parser.add_argument("--wandb-timeout", type=int, default=60)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    baseline_sweep_ids = parse_csv_strings(args.baseline_sweep_ids)
    if not baseline_sweep_ids:
        raise ValueError("--baseline-sweep-ids is required.")

    # (cli arg value, curve label, color, clean-sweep-ids arg, variant filter for
    # disambiguating clean (attack_rate=0) reference runs) -- order controls
    # draw/legend order. The base Tent+SAFER curve is required; SAFER-A and
    # mean-pooling are optional extra curves layered on top to compare
    # clean-retention and pooling-strategy variants under the same EOT attack.
    # Each variant gets its own optional K=0 clean anchor point, since their
    # clean accuracies differ (SAFER-A retains more clean accuracy by design;
    # mean-pooling has no reliability weighting).
    variant_specs = [
        (args.safer_sweep_ids, "Tent + SAFER", "C0", args.clean_sweep_ids, is_safer_base_variant),
        (args.safer_a_sweep_ids, "Tent + SAFER-A", "C2", args.safer_a_clean_sweep_ids, is_safer_a_variant),
        (args.safer_mean_sweep_ids, "Tent + SAFER (mean)", "C3", args.safer_mean_clean_sweep_ids, is_safer_mean_variant),
    ]

    select_metric = args.select_metric or args.mean_key

    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    csv_rows: List[List[object]] = [["method", "attack_eot_views", "acc_mean", "acc_std"]]
    all_ks: List[int] = []

    for idx, (sweep_ids_str, label, color, clean_ids_str, variant_filter) in enumerate(variant_specs):
        sweep_ids = parse_csv_strings(sweep_ids_str) if sweep_ids_str else []
        if not sweep_ids:
            if idx == 0:
                raise ValueError("--safer-sweep-ids is required.")
            continue
        records = load_records(sweep_ids, args.entity, args.project, args.wandb_timeout, args.verbose)
        curve = collect_safer_curve(
            records=records,
            dataset=args.dataset,
            domain_id=args.domain_id,
            mean_key=args.mean_key,
            std_key=args.std_key,
            select_mode=args.select,
            select_metric=select_metric,
            verbose=args.verbose,
        )
        if not curve:
            if idx == 0:
                raise ValueError("No matching Tent+SAFER (s_wrap_alg=1) runs found across the given sweep IDs.")
            if args.verbose:
                print(f"[warn] no runs found for '{label}' across {sweep_ids}", file=sys.stderr)
            continue

        clean_point: Optional[Tuple[float, float]] = None
        clean_ids = parse_csv_strings(clean_ids_str) if clean_ids_str else []
        if clean_ids:
            clean_records = load_records(clean_ids, args.entity, args.project, args.wandb_timeout, args.verbose)
            clean_point = collect_clean_point(
                records=clean_records,
                dataset=args.dataset,
                domain_id=args.domain_id,
                mean_key=args.mean_key,
                std_key=args.std_key,
                select_mode=args.select,
                select_metric=select_metric,
                verbose=args.verbose,
                label=label,
                variant_filter=variant_filter,
            )

        ks = sorted(curve)
        means = [curve[k][0] for k in ks]
        stds = [curve[k][1] for k in ks]
        if clean_point is not None:
            ks = [0] + ks
            means = [clean_point[0]] + means
            stds = [clean_point[1]] + stds

        all_ks.extend(ks)
        ax.plot(ks, means, marker="o", linewidth=1.8, markersize=6.5, color=color, label=label)
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        ax.fill_between(ks, lo, hi, color=color, alpha=0.15, linewidth=0)
        for k, m, s in zip(ks, means, stds):
            csv_rows.append([label, k, m, s])

    baseline_records = load_records(baseline_sweep_ids, args.entity, args.project, args.wandb_timeout, args.verbose)
    baseline_point = collect_baseline_point(
        records=baseline_records,
        dataset=args.dataset,
        domain_id=args.domain_id,
        mean_key=args.mean_key,
        std_key=args.std_key,
        select_mode=args.select,
        select_metric=select_metric,
        verbose=args.verbose,
    )
    if baseline_point is not None:
        b_mean, b_std = baseline_point
        ax.axhline(b_mean, color="C1", linestyle="--", linewidth=1.6, label="Tent")
        if b_std > 0:
            ax.axhspan(b_mean - b_std, b_mean + b_std, color="C1", alpha=0.12, linewidth=0)
        csv_rows.append(["Tent", "", baseline_point[0], baseline_point[1]])

    ks_sorted = sorted(set(all_ks))
    ax.set_xlabel(args.xlabel)
    ax.set_ylabel(args.ylabel)
    if ks_sorted:
        ax.set_xticks(ks_sorted)
    ax.grid(True, linewidth=0.4, alpha=0.35)
    ax.set_title(
        args.title
        or f"{wt.domain_label(args.dataset, args.domain_id)}: Defense-aware (EOT) attack on SAFER"
    )
    ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=args.dpi)
    plt.close(fig)
    print(f"[ok] wrote plot: {args.output}")

    args.csv_output.parent.mkdir(parents=True, exist_ok=True)
    with args.csv_output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerows(csv_rows)
    print(f"[ok] wrote csv: {args.csv_output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
