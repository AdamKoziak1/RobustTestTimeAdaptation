#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def _to_float(value: object) -> Optional[float]:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def dataset_display_name(dataset: str) -> str:
    key = dataset.strip().lower()
    if key == "pacs":
        return "PACS"
    if key == "vlcs":
        return "VLCS"
    if key in {"office-home", "officehome"}:
        return "OfficeHome"
    return dataset


def resolve_domain_name(dataset: str, domain_id: int) -> str:
    labels = wt.DATASET_DOMAIN_LABELS.get(dataset)
    if labels and 0 <= domain_id < len(labels):
        return labels[domain_id]
    return f"Domain {domain_id}"


def method_label(record: wt.RunRecord) -> str:
    adapt_alg = wt.get_value(record, "adapt_alg")
    if not isinstance(adapt_alg, str):
        adapt_alg = "Method"
    wrap_raw = wt.get_value(record, "s_wrap_alg")
    wrap = bool(wt.to_int(wrap_raw) or 0)

    jpeg_q = wt.to_int(wt.get_value(record, "jpeg_input_quality"))
    fft_in = wt.get_value(record, "fft_input_keep_ratio")
    fft_feat = wt.get_value(record, "fft_feat_keep_ratio")
    gauss_in = wt.get_value(record, "gauss_input_sigma")
    gauss_feat = wt.get_value(record, "gauss_feat_sigma")

    fft_in_f = _to_float(fft_in)
    fft_feat_f = _to_float(fft_feat)
    gauss_in_f = _to_float(gauss_in)
    gauss_feat_f = _to_float(gauss_feat)

    if not wrap:
        if jpeg_q is not None and jpeg_q < 100:
            return f"{adapt_alg} + JPEG"
        if (fft_in_f is not None and fft_in_f < 1.0) or (fft_feat_f is not None and fft_feat_f < 1.0):
            return f"{adapt_alg} + FFT"
        if (gauss_in_f is not None and gauss_in_f > 0.0) or (gauss_feat_f is not None and gauss_feat_f > 0.0):
            return f"{adapt_alg} + Blur"
        return adapt_alg
    alpha_mode = str(wt.get_value(record, "s_alpha_mode") or "none").lower()
    if alpha_mode == "sigmoid":
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


def pick_runs(
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_id: int,
    attack_rate: int,
    extra_filters: Mapping[str, List[object]],
    mean_key: str,
    select_mode: str,
    select_metric: str,
    verbose: bool,
) -> Dict[str, wt.RunRecord]:
    grouped: Dict[str, List[wt.RunRecord]] = {}
    for record in records:
        if mean_key not in record.summary:
            continue
        if extra_filters and not wt.record_matches_filters(record, extra_filters):
            continue
        ds_name = wt.get_value(record, "dataset")
        if ds_name != dataset:
            continue
        dom_val = wt.get_value(record, "test_envs", "test_env")
        if isinstance(dom_val, list):
            dom_val = dom_val[0] if dom_val else None
        dom = wt.to_int(dom_val)
        if dom != domain_id:
            continue
        rate = wt.to_int(wt.get_value(record, "attack_rate"))
        if rate != attack_rate:
            continue
        grouped.setdefault(method_label(record), []).append(record)

    chosen: Dict[str, wt.RunRecord] = {}
    for label, runs in grouped.items():
        chosen[label] = wt.select_run(runs, select_mode, select_metric)
        if verbose:
            print(f"[info] {label}: {len(runs)} runs matched", file=sys.stderr)
    return chosen


def fmt(value: Optional[float], precision: int = 2) -> str:
    if value is None:
        return "--"
    return f"{value:.{precision}f}"


def fmt_ratio(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}x"


def render_latex(
    rows: Sequence[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]],
    dataset: str,
    domain_id: int,
    attack_rate: int,
    placement: str,
    label: str,
) -> str:
    ds = dataset_display_name(dataset)
    dom_name = resolve_domain_name(dataset, domain_id)
    lines = [
        f"\\begin{{table}}[{placement}]",
        "\\centering",
        (
            "\\caption{Compute overhead at evaluation time on "
            f"the {dom_name} domain of {ds} under {attack_rate}\\% attack rate.}}"
        ),
        f"\\label{{{label}}}",
        "\\begin{tabular}{lrrrrrr}",
        "\\hline",
        "Method & Acc (\\%) & Time (s) & Time $\\times$ & VRAM (MB) & VRAM $\\times$ & Params (M) \\\\",
        "\\hline",
    ]
    for row in rows:
        label_text, acc, time_s, time_x, vram_mb, vram_x, params_m = row
        lines.append(
            f"{label_text} & {fmt(acc)} & {fmt(time_s)} & {fmt_ratio(time_x)} & "
            f"{fmt(vram_mb)} & {fmt_ratio(vram_x)} & {fmt(params_m, precision=3)} \\\\"
        )
    lines.extend(["\\hline", "\\end{tabular}", "\\end{table}"])
    return "\n".join(lines)


def write_csv(
    path: Path,
    rows: Sequence[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["method", "acc_mean", "time_taken_s", "time_vs_baseline", "max_vram_overhead_mb", "vram_vs_baseline", "sum_params_m"]
        )
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate a compact compute-overhead table from W&B sweep runs."
    )
    parser.add_argument("--sweep-ids", required=True, help="Comma-separated sweep IDs/paths.")
    parser.add_argument("--entity", default="bigslav", help="W&B entity for bare sweep IDs.")
    parser.add_argument("--project", default="safer", help="W&B project for bare sweep IDs.")
    parser.add_argument("--dataset", default="PACS", help="Dataset name.")
    parser.add_argument("--domain-id", type=int, default=0, help="Single domain id.")
    parser.add_argument("--attack-rate", type=int, default=100, help="Attack rate to compare.")
    parser.add_argument("--baseline-method", default="Tent", help="Method used for ratio columns.")
    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for mean accuracy.")
    parser.add_argument("--std-key", default="acc_std", help="Summary key for std (unused in table).")
    parser.add_argument("--time-key", default="time_taken_s", help="Summary key for runtime.")
    parser.add_argument("--vram-key", default="max_vram_overhead_mb", help="Summary key for VRAM.")
    parser.add_argument("--param-key", default="sum_params", help="Summary key for parameter count.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric for run selection (default: mean key).")
    parser.add_argument(
        "--methods",
        default="",
        help="Optional comma-separated method labels to include in output order.",
    )
    parser.add_argument("--filter", action="append", default=[], help="Extra filter key=value (repeatable).")
    parser.add_argument("--placement", default="t", help="LaTeX placement.")
    parser.add_argument("--label", default="tab:compute-overhead", help="LaTeX label.")
    parser.add_argument("--output", type=Path, help="Optional output .tex file (stdout if omitted).")
    parser.add_argument(
        "--csv-output",
        type=Path,
        default=Path("sweeps/compute_overhead.csv"),
        help="CSV export path.",
    )
    parser.add_argument("--wandb-timeout", type=int, default=60, help="W&B API timeout in seconds.")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")
    args = parser.parse_args()

    sweep_ids = parse_csv_strings(args.sweep_ids)
    if not sweep_ids:
        raise ValueError("No sweep IDs provided.")
    select_metric = args.select_metric or args.mean_key
    method_order = parse_csv_strings(args.methods)
    extra_filters = wt.parse_filter_args(args.filter)

    records = load_records(
        sweep_ids=sweep_ids,
        entity=args.entity,
        project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )
    selected = pick_runs(
        records=records,
        dataset=args.dataset,
        domain_id=args.domain_id,
        attack_rate=args.attack_rate,
        extra_filters=extra_filters,
        mean_key=args.mean_key,
        select_mode=args.select,
        select_metric=select_metric,
        verbose=args.verbose,
    )
    if not selected:
        raise ValueError("No runs matched dataset/domain/attack-rate filters.")

    labels = list(selected.keys())
    if method_order:
        keep = [label for label in method_order if label in selected]
        extras = [label for label in labels if label not in keep]
        labels = keep + sorted(extras)
    else:
        labels = sorted(labels, key=lambda x: (0 if x == args.baseline_method else 1, x))

    baseline = selected.get(args.baseline_method)
    baseline_time = _to_float(baseline.summary.get(args.time_key)) if baseline else None
    baseline_vram = _to_float(baseline.summary.get(args.vram_key)) if baseline else None

    rows: List[Tuple[str, Optional[float], Optional[float], Optional[float], Optional[float], Optional[float], Optional[float]]] = []
    for label in labels:
        run = selected[label]
        acc = _to_float(run.summary.get(args.mean_key))
        time_s = _to_float(run.summary.get(args.time_key))
        vram_mb = _to_float(run.summary.get(args.vram_key))
        params = _to_float(run.summary.get(args.param_key))
        params_m = (params / 1_000_000.0) if params is not None else None

        time_x = None
        if time_s is not None and baseline_time and baseline_time > 0:
            time_x = time_s / baseline_time
        vram_x = None
        if vram_mb is not None and baseline_vram and baseline_vram > 0:
            vram_x = vram_mb / baseline_vram
        rows.append((label, acc, time_s, time_x, vram_mb, vram_x, params_m))

    latex = render_latex(
        rows=rows,
        dataset=args.dataset,
        domain_id=args.domain_id,
        attack_rate=args.attack_rate,
        placement=args.placement,
        label=args.label,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(latex + "\n", encoding="utf-8")
    else:
        sys.stdout.write(latex + "\n")

    write_csv(args.csv_output, rows)
    print(f"[ok] wrote csv: {args.csv_output}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
