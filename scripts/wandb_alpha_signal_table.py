#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


DEFAULT_SIGNALS = [
    "feat_disagreement",
    "prob_disagreement",
    "entropy_gap",
    "margin_gap",
    "orig_entropy",
    "orig_conf",
]


@dataclass(frozen=True)
class RowSpec:
    label: str
    filters: Mapping[str, List[object]]


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def parse_csv_ints(text: str) -> List[int]:
    return [int(chunk.strip()) for chunk in text.split(",") if chunk.strip()]


def format_value(value: Optional[float], precision: int, signed: bool = False) -> str:
    if value is None:
        return "--"
    if signed:
        return f"{value:+.{precision}f}"
    return f"{value:.{precision}f}"


def load_records_for_sweeps(
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


def compute_domain_avg_by_rate(
    records: Sequence[wt.RunRecord],
    row: RowSpec,
    dataset: str,
    domain_ids: Sequence[int],
    attack_rates: Sequence[int],
    mean_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Dict[int, Optional[float]]:
    filters: Dict[str, List[object]] = dict(row.filters)
    filters["dataset"] = [dataset]

    matched = [
        record
        for record in records
        if mean_key in record.summary and wt.record_matches_filters(record, filters)
    ]
    if verbose:
        print(f"[info] {row.label}: {len(matched)} matched runs", file=sys.stderr)

    grouped: Dict[tuple[int, int], List[wt.RunRecord]] = {}
    for record in matched:
        dom_val = wt.get_value(record, "test_envs", "test_env")
        if isinstance(dom_val, list):
            dom_val = dom_val[0] if dom_val else None
        dom_id = wt.to_int(dom_val)
        rate = wt.to_int(wt.get_value(record, "attack_rate"))
        if dom_id is None or rate is None:
            continue
        grouped.setdefault((dom_id, rate), []).append(record)

    out: Dict[int, Optional[float]] = {}
    for rate in attack_rates:
        domain_means: List[float] = []
        for dom_id in domain_ids:
            runs = grouped.get((dom_id, rate), [])
            if not runs:
                continue
            chosen = wt.select_run(runs, select_mode, select_metric)
            mean_val = chosen.summary.get(mean_key)
            try:
                domain_means.append(float(mean_val))
            except (TypeError, ValueError):
                continue
        if not domain_means:
            out[rate] = None
            if verbose:
                print(
                    f"[warn] Missing all domains for {row.label} @ rate={rate}",
                    file=sys.stderr,
                )
            continue
        out[rate] = sum(domain_means) / len(domain_means)
        if verbose and len(domain_means) < len(domain_ids):
            print(
                f"[warn] Partial domains for {row.label} @ rate={rate}: "
                f"{len(domain_means)}/{len(domain_ids)}",
                file=sys.stderr,
            )
    return out


def render_table(
    dataset: str,
    domain_ids: Sequence[int],
    attack_rates: Sequence[int],
    delta_rate: int,
    clean_rate: int,
    precision: int,
    row_values: Mapping[str, Mapping[int, Optional[float]]],
    placement: str,
    table_label: str,
) -> str:
    baseline_label = "Tent (no SAFER wrapper)"
    baseline_delta = row_values.get(baseline_label, {}).get(delta_rate)
    baseline_clean = row_values.get(baseline_label, {}).get(clean_rate)
    col_spec = "l" + "c" * (len(attack_rates) + 2)
    delta_header = f"$\\Delta_{{{delta_rate}}}$ vs Tent"
    domain_text = (
        "all PACS domains"
        if len(domain_ids) > 1
        else f"PACS domain {domain_ids[0]}"
    )

    lines = [
        f"\\begin{{table}}[{placement}]",
        "\\centering",
        (
            "\\caption{Alpha-signal ablation for SAFER-A on \\texttt{Tent}, "
            f"averaged over {domain_text}.}}"
        ),
        f"\\label{{{table_label}}}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        "Method & "
        + " & ".join([f"{rate}\\%" for rate in attack_rates])
        + f" & {delta_header} & Clean drop \\\\",
        "\\hline",
    ]

    ordered_labels = list(row_values.keys())
    for label in ordered_labels:
        values = row_values.get(label, {})
        cells = [format_value(values.get(rate), precision) for rate in attack_rates]
        delta_cell = "--"
        clean_drop_cell = "--"
        if label != baseline_label:
            row_delta = values.get(delta_rate)
            row_clean = values.get(clean_rate)
            if baseline_delta is not None and row_delta is not None:
                delta_cell = format_value(row_delta - baseline_delta, precision, signed=True)
            if baseline_clean is not None and row_clean is not None:
                clean_drop_cell = format_value(baseline_clean - row_clean, precision, signed=True)
        lines.append(f"{label} & " + " & ".join(cells + [delta_cell, clean_drop_cell]) + " \\\\")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Build a self-contained LaTeX table for alpha-signal ablations from W&B sweep IDs. "
            "No cache YAML is required."
        )
    )
    parser.add_argument(
        "--signal-sweep-ids",
        required=True,
        help="Comma-separated sweep IDs/paths containing alpha-signal runs.",
    )
    parser.add_argument(
        "--baseline-sweep-ids",
        default="",
        help=(
            "Optional comma-separated sweep IDs/paths for Tent baseline (s_wrap_alg=0). "
            "If omitted, baseline is searched in signal sweeps."
        ),
    )
    parser.add_argument("--entity", default="your_wandb_entity", help="W&B entity for bare sweep IDs.")
    parser.add_argument("--project", default="your_wandb_project", help="W&B project for bare sweep IDs.")
    parser.add_argument("--dataset", default="PACS", help="Dataset name.")
    parser.add_argument("--domain-ids", default="0", help="Comma-separated domain ids.")
    parser.add_argument("--attack-rates", default="0,50,100", help="Comma-separated attack rates.")
    parser.add_argument("--signals", default=",".join(DEFAULT_SIGNALS), help="Comma-separated alpha signals.")
    parser.add_argument("--adapt-alg", default="Tent", help="Base algorithm to filter.")
    parser.add_argument("--primary-pool", default="cc_drop", help="Pooling filter for SAFER rows.")
    parser.add_argument("--alpha-mode", default="sigmoid", help="Alpha mode for SAFER rows.")
    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for mean accuracy.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument("--delta-rate", type=int, default=100, help="Rate used for Delta column.")
    parser.add_argument("--clean-rate", type=int, default=0, help="Rate used for Clean drop.")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision.")
    parser.add_argument("--placement", default="t", help="LaTeX placement.")
    parser.add_argument("--label", default="tab:abl-alpha-signal", help="LaTeX table label.")
    parser.add_argument("--output", type=Path, help="Optional output .tex file (stdout if omitted).")
    parser.add_argument("--wandb-timeout", type=int, default=60, help="W&B API timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")
    args = parser.parse_args()

    signal_sweep_ids = parse_csv_strings(args.signal_sweep_ids)
    baseline_sweep_ids = parse_csv_strings(args.baseline_sweep_ids)
    if not signal_sweep_ids:
        raise ValueError("No signal sweep IDs provided.")
    domain_ids = parse_csv_ints(args.domain_ids)
    attack_rates = parse_csv_ints(args.attack_rates)
    signals = parse_csv_strings(args.signals)
    if not domain_ids:
        raise ValueError("No domain IDs provided.")
    if not attack_rates:
        raise ValueError("No attack rates provided.")
    if not signals:
        raise ValueError("No signals provided.")

    signal_records = load_records_for_sweeps(
        sweep_ids=signal_sweep_ids,
        entity=args.entity,
        project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )
    baseline_records = signal_records
    if baseline_sweep_ids:
        baseline_records = load_records_for_sweeps(
            sweep_ids=baseline_sweep_ids,
            entity=args.entity,
            project=args.project,
            wandb_timeout=args.wandb_timeout,
            verbose=args.verbose,
        )

    select_metric = args.select_metric or args.mean_key
    rows: List[RowSpec] = [
        RowSpec(
            label="Tent (no SAFER wrapper)",
            filters={"adapt_alg": [args.adapt_alg], "s_wrap_alg": [0]},
        )
    ]
    for signal in signals:
        rows.append(
            RowSpec(
                label=f"Tent + SAFER-A ({signal})",
                filters={
                    "adapt_alg": [args.adapt_alg],
                    "s_wrap_alg": [1],
                    "s_primary_view_pool": [args.primary_pool],
                    "s_alpha_mode": [args.alpha_mode],
                    "s_alpha_signal": [signal],
                },
            )
        )

    row_values: Dict[str, Dict[int, Optional[float]]] = {}
    for row in rows:
        records = baseline_records if row.label == "Tent (no SAFER wrapper)" else signal_records
        row_values[row.label] = compute_domain_avg_by_rate(
            records=records,
            row=row,
            dataset=args.dataset,
            domain_ids=domain_ids,
            attack_rates=attack_rates,
            mean_key=args.mean_key,
            select_mode=args.select,
            select_metric=select_metric,
            verbose=args.verbose,
        )

    output_text = (
        render_table(
            dataset=args.dataset,
            domain_ids=domain_ids,
            attack_rates=attack_rates,
            delta_rate=args.delta_rate,
            clean_rate=args.clean_rate,
            precision=args.precision,
            row_values=row_values,
            placement=args.placement,
            table_label=args.label,
        ).rstrip()
        + "\n"
    )

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(output_text, encoding="utf-8")
    else:
        sys.stdout.write(output_text)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
