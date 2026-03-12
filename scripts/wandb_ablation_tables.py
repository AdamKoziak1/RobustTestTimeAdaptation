#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


DEFAULT_CACHE_PATH = Path("sweeps/wandb_ablation_table_cache.yaml")
DEFAULT_DATASET = "PACS"
DEFAULT_ATTACK_RATES = [0, 50, 100]
DEFAULT_DOMAIN_IDS = [0, 1, 2, 3]
DEFAULT_DELTA_RATE = 100
DEFAULT_CLEAN_RATE = 0


@dataclass(frozen=True)
class RowSpec:
    label: str
    sweep_alias: str
    filters: Mapping[str, List[Any]]


@dataclass
class SweepRef:
    ref: str
    entity: Optional[str] = None
    project: Optional[str] = None


@dataclass
class CacheConfig:
    entity: str
    project: str
    dataset: str
    attack_rates: List[int]
    domain_ids: List[int]
    sweeps: Dict[str, SweepRef]


POOLING_ROWS: Sequence[RowSpec] = (
    RowSpec(
        "Tent (no SAFER wrapper)",
        "tent_baseline",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [0]},
    ),
    RowSpec(
        "Tent + SAFER (mean)",
        "tent_pooling",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"], "s_primary_view_pool": ["mean"]},
    ),
    RowSpec(
        "Tent + SAFER (entropy)",
        "tent_pooling",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"], "s_primary_view_pool": ["entropy"]},
    ),
    RowSpec(
        "Tent + SAFER (cc)",
        "tent_pooling",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"], "s_primary_view_pool": ["cc"]},
    ),
    RowSpec(
        "Tent + SAFER (cc_drop)",
        "tent_pooling",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"], "s_primary_view_pool": ["cc_drop"]},
    ),
)


SIMPLE_ROWS: Sequence[RowSpec] = (
    RowSpec(
        "Tent",
        "tent_baseline",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [0]},
    ),
    RowSpec(
        "Tent + JPEG",
        "tent_jpeg",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [0]},
    ),
    RowSpec(
        "Tent + FFT",
        "tent_fft",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [0]},
    ),
    RowSpec(
        "Tent + Blur",
        "tent_blur",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [0]},
    ),
    RowSpec(
        "Tent + SAFER (uniform / mean)",
        "tent_pooling",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"], "s_primary_view_pool": ["mean"]},
    ),
    RowSpec(
        "Tent + SAFER (cc_drop)",
        "tent_pooling",
        {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"], "s_primary_view_pool": ["cc_drop"]},
    ),
)


CACHE_TEMPLATE = """# Sweep-ID cache used by scripts/wandb_ablation_tables.py
# Fill each alias with a W&B sweep id (abcd1234) or full sweep path (entity/project/abcd1234).
entity: bigslav
project: safer

dataset: PACS
attack_rates: [0, 50, 100]
domain_ids: [0, 1, 2, 3]

sweeps:
  tent_baseline: ""
  tent_pooling: ""
  tent_jpeg: ""
  tent_fft: ""
  tent_blur: ""
"""


def required_aliases() -> List[str]:
    aliases: List[str] = []
    for row in list(POOLING_ROWS) + list(SIMPLE_ROWS):
        if row.sweep_alias not in aliases:
            aliases.append(row.sweep_alias)
    return aliases


def load_yaml_mapping(path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to parse the sweep cache file.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if data is None:
        return {}
    if not isinstance(data, Mapping):
        raise ValueError(f"Expected a mapping in {path}, got {type(data).__name__}.")
    return data


def parse_sweep_ref(alias: str, raw: Any) -> SweepRef:
    if isinstance(raw, str):
        return SweepRef(ref=raw.strip())
    if not isinstance(raw, Mapping):
        raise ValueError(f"Sweep entry for '{alias}' must be a string or mapping.")
    ref_raw = raw.get("sweep") or raw.get("sweep_id") or raw.get("sweep_path") or raw.get("path")
    ref = str(ref_raw).strip() if ref_raw is not None else ""
    ent_raw = raw.get("entity")
    proj_raw = raw.get("project")
    entity = str(ent_raw).strip() if ent_raw is not None else None
    project = str(proj_raw).strip() if proj_raw is not None else None
    return SweepRef(ref=ref, entity=entity, project=project)


def parse_list_ints(raw: Any, field_name: str) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return wt.parse_csv_ints(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        out: List[int] = []
        for value in raw:
            out.append(int(value))
        return out
    raise ValueError(f"Field '{field_name}' must be a list of ints or CSV string.")


def parse_dataset(raw: Any) -> str:
    if raw is None:
        return DEFAULT_DATASET
    if isinstance(raw, str):
        value = raw.strip()
        return value if value else DEFAULT_DATASET
    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes, bytearray)):
        for value in raw:
            text = str(value).strip()
            if text:
                return text
    return DEFAULT_DATASET


def load_cache(
    path: Path,
    entity_override: Optional[str],
    project_override: Optional[str],
    dataset_override: Optional[str],
) -> CacheConfig:
    if not path.exists():
        raise ValueError(
            f"Cache file not found: {path}. "
            f"Create it with --init-cache or provide an existing file via --cache-file."
        )
    data = load_yaml_mapping(path)
    entity = entity_override or str(data.get("entity") or "bigslav")
    project = project_override or str(data.get("project") or "safer")
    dataset = dataset_override or parse_dataset(data.get("dataset", data.get("datasets")))
    attack_rates = parse_list_ints(data.get("attack_rates"), "attack_rates") or list(DEFAULT_ATTACK_RATES)
    domain_ids = parse_list_ints(data.get("domain_ids"), "domain_ids") or list(DEFAULT_DOMAIN_IDS)

    sweeps_raw = data.get("sweeps", {})
    if not isinstance(sweeps_raw, Mapping):
        raise ValueError("The cache file field 'sweeps' must be a mapping.")
    sweeps: Dict[str, SweepRef] = {}
    for alias, raw_spec in sweeps_raw.items():
        sweeps[str(alias)] = parse_sweep_ref(str(alias), raw_spec)

    return CacheConfig(
        entity=entity,
        project=project,
        dataset=dataset,
        attack_rates=attack_rates,
        domain_ids=domain_ids,
        sweeps=sweeps,
    )


def write_cache_template(path: Path, force: bool) -> None:
    if path.exists() and not force:
        raise ValueError(
            f"Refusing to overwrite existing cache file: {path}. "
            "Use --force-init-cache to overwrite."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(CACHE_TEMPLATE, encoding="utf-8")


def resolve_sweep_path(ref: str, entity: str, project: str) -> str:
    if "/" in ref:
        return ref
    return f"{entity}/{project}/{ref}"


def fetch_records(
    sweeps: Mapping[str, SweepRef],
    aliases: Sequence[str],
    default_entity: str,
    default_project: str,
    wandb_timeout: Optional[int],
    verbose: bool,
) -> Dict[str, List[wt.RunRecord]]:
    records_by_path: Dict[str, List[wt.RunRecord]] = {}
    records_by_alias: Dict[str, List[wt.RunRecord]] = {}
    for alias in aliases:
        if alias not in sweeps or not sweeps[alias].ref:
            if verbose:
                print(f"[warn] No sweep configured for alias '{alias}'; rows will be empty.", file=sys.stderr)
            records_by_alias[alias] = []
            continue
        spec = sweeps[alias]
        entity = spec.entity or default_entity
        project = spec.project or default_project
        ref = spec.ref.strip()
        sweep_path = resolve_sweep_path(ref, entity, project)
        if sweep_path not in records_by_path:
            records_by_path[sweep_path] = wt.load_runs_from_wandb_api(
                ref,
                entity,
                project,
                api_timeout=wandb_timeout,
            )
            if verbose:
                print(
                    f"[info] Loaded {len(records_by_path[sweep_path])} runs from {sweep_path}",
                    file=sys.stderr,
                )
        records_by_alias[alias] = records_by_path[sweep_path]
    return records_by_alias


def format_value(value: Optional[float], precision: int, signed: bool = False) -> str:
    if value is None:
        return "--"
    if signed:
        return f"{value:+.{precision}f}"
    return f"{value:.{precision}f}"


def dataset_display_name(dataset: str) -> str:
    key = dataset.strip().lower()
    if key == "pacs":
        return "PACS"
    if key == "vlcs":
        return "VLCS"
    if key in {"office-home", "officehome"}:
        return "OfficeHome"
    return dataset


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
    filters = dict(row.filters)
    filters["dataset"] = [dataset]
    matched = [
        record
        for record in records
        if mean_key in record.summary and wt.record_matches_filters(record, filters)
    ]

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
                    f"[warn] Missing all domains for row '{row.label}' at attack_rate={rate}",
                    file=sys.stderr,
                )
            continue

        if verbose and len(domain_means) < len(domain_ids):
            print(
                f"[warn] Partial domains for row '{row.label}' at attack_rate={rate}: "
                f"{len(domain_means)}/{len(domain_ids)}",
                file=sys.stderr,
            )
        out[rate] = sum(domain_means) / len(domain_means)
    return out


def render_pooling_table(
    dataset: str,
    attack_rates: Sequence[int],
    delta_rate: int,
    precision: int,
    row_values: Mapping[str, Mapping[int, Optional[float]]],
    placement: str,
) -> str:
    baseline_label = POOLING_ROWS[0].label
    baseline_delta = row_values.get(baseline_label, {}).get(delta_rate)
    col_spec = "l" + "c" * (len(attack_rates) + 1)
    delta_header = f"$\\Delta_{{{delta_rate}}}$ vs Tent"

    lines = [
        f"\\begin{{table}}[{placement}]",
        "\\centering",
        (
            "\\caption{Pooling ablation on a single base method (\\texttt{Tent}), "
            f"averaged over {dataset_display_name(dataset)} domains.}}"
        ),
        "\\label{tab:abl-pooling-tent}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        "Method & " + " & ".join([f"{rate}\\%" for rate in attack_rates]) + f" & {delta_header} \\\\",
        "\\hline",
    ]

    for row in POOLING_ROWS:
        values = row_values.get(row.label, {})
        cells = [format_value(values.get(rate), precision) for rate in attack_rates]
        delta_cell = "--"
        if row.label != baseline_label and baseline_delta is not None:
            row_delta = values.get(delta_rate)
            if row_delta is not None:
                delta_cell = format_value(row_delta - baseline_delta, precision, signed=True)
        lines.append(f"{row.label} & " + " & ".join(cells + [delta_cell]) + " \\\\")

    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines)


def render_simple_table(
    dataset: str,
    attack_rates: Sequence[int],
    delta_rate: int,
    clean_rate: int,
    precision: int,
    row_values: Mapping[str, Mapping[int, Optional[float]]],
    placement: str,
) -> str:
    baseline_label = SIMPLE_ROWS[0].label
    baseline_delta = row_values.get(baseline_label, {}).get(delta_rate)
    baseline_clean = row_values.get(baseline_label, {}).get(clean_rate)
    col_spec = "l" + "c" * (len(attack_rates) + 2)
    delta_header = f"$\\Delta_{{{delta_rate}}}$ vs Tent"

    lines = [
        f"\\begin{{table}}[{placement}]",
        "\\centering",
        (
            "\\caption{Simple transform defenses vs full SAFER on \\texttt{Tent} "
            f"({dataset_display_name(dataset)} average).}}"
        ),
        "\\label{tab:abl-simple-vs-safer}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
        "Method & "
        + " & ".join([f"{rate}\\%" for rate in attack_rates])
        + f" & {delta_header} & Clean drop \\\\",
        "\\hline",
    ]

    for row in SIMPLE_ROWS:
        values = row_values.get(row.label, {})
        cells = [format_value(values.get(rate), precision) for rate in attack_rates]
        delta_cell = "--"
        clean_drop_cell = "--"
        if row.label != baseline_label:
            row_delta = values.get(delta_rate)
            row_clean = values.get(clean_rate)
            if baseline_delta is not None and row_delta is not None:
                delta_cell = format_value(row_delta - baseline_delta, precision, signed=True)
            if baseline_clean is not None and row_clean is not None:
                clean_drop_cell = format_value(baseline_clean - row_clean, precision, signed=True)
        lines.append(f"{row.label} & " + " & ".join(cells + [delta_cell, clean_drop_cell]) + " \\\\")

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
            "Generate ablation LaTeX tables from W&B sweep IDs. "
            "This script outputs tab:abl-pooling-tent and tab:abl-simple-vs-safer."
        )
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Sweep-ID cache file (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument("--init-cache", action="store_true", help="Write a template cache file and exit.")
    parser.add_argument(
        "--force-init-cache",
        action="store_true",
        help="Allow --init-cache to overwrite an existing file.",
    )
    parser.add_argument("--entity", help="Default W&B entity (overrides cache file).")
    parser.add_argument("--project", help="Default W&B project (overrides cache file).")
    parser.add_argument("--dataset", help="Dataset name (default: cache file).")
    parser.add_argument("--attack-rates", help="Comma-separated attack rates (default: cache file).")
    parser.add_argument("--domain-ids", help="Comma-separated domain ids (default: cache file).")
    parser.add_argument("--delta-rate", type=int, default=DEFAULT_DELTA_RATE, help="Rate used for Delta columns.")
    parser.add_argument("--clean-rate", type=int, default=DEFAULT_CLEAN_RATE, help="Rate used for Clean drop.")
    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for mean accuracy.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric used for run selection (default: mean key).")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision.")
    parser.add_argument("--placement", default="t", help="LaTeX table placement.")
    parser.add_argument("--output", type=Path, help="Optional output .tex file (stdout if omitted).")
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds (default: 60).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")
    args = parser.parse_args()

    if args.init_cache:
        write_cache_template(args.cache_file, force=args.force_init_cache)
        print(f"Wrote sweep cache template to {args.cache_file}")
        return 0

    cache_cfg = load_cache(args.cache_file, args.entity, args.project, args.dataset)
    dataset = args.dataset or cache_cfg.dataset
    attack_rates = wt.parse_csv_ints(args.attack_rates) if args.attack_rates else cache_cfg.attack_rates
    domain_ids = wt.parse_csv_ints(args.domain_ids) if args.domain_ids else cache_cfg.domain_ids
    if not attack_rates:
        raise ValueError("No attack rates configured.")
    if not domain_ids:
        raise ValueError("No domain ids configured.")

    aliases = required_aliases()
    records_by_alias = fetch_records(
        sweeps=cache_cfg.sweeps,
        aliases=aliases,
        default_entity=cache_cfg.entity,
        default_project=cache_cfg.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )

    select_metric = args.select_metric or args.mean_key

    pooling_values: Dict[str, Dict[int, Optional[float]]] = {}
    for row in POOLING_ROWS:
        pooling_values[row.label] = compute_domain_avg_by_rate(
            records=records_by_alias[row.sweep_alias],
            row=row,
            dataset=dataset,
            domain_ids=domain_ids,
            attack_rates=attack_rates,
            mean_key=args.mean_key,
            select_mode=args.select,
            select_metric=select_metric,
            verbose=args.verbose,
        )

    simple_values: Dict[str, Dict[int, Optional[float]]] = {}
    for row in SIMPLE_ROWS:
        simple_values[row.label] = compute_domain_avg_by_rate(
            records=records_by_alias[row.sweep_alias],
            row=row,
            dataset=dataset,
            domain_ids=domain_ids,
            attack_rates=attack_rates,
            mean_key=args.mean_key,
            select_mode=args.select,
            select_metric=select_metric,
            verbose=args.verbose,
        )

    tables = [
        render_pooling_table(
            dataset=dataset,
            attack_rates=attack_rates,
            delta_rate=args.delta_rate,
            precision=args.precision,
            row_values=pooling_values,
            placement=args.placement,
        ),
        render_simple_table(
            dataset=dataset,
            attack_rates=attack_rates,
            delta_rate=args.delta_rate,
            clean_rate=args.clean_rate,
            precision=args.precision,
            row_values=simple_values,
            placement=args.placement,
        ),
    ]

    output_text = "\n\n".join(tables).rstrip() + "\n"
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
