#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
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


DEFAULT_CACHE_PATH = Path("sweeps/wandb_paper_table_cache.yaml")
DEFAULT_DATASETS = ["PACS", "VLCS", "office-home"]
DEFAULT_ATTACK_RATES = [0, 50, 100]
DEFAULT_DOMAIN_IDS = [0, 1, 2, 3]


@dataclass(frozen=True)
class MethodRow:
    label: str
    sweep_alias: str
    filters: Mapping[str, List[Any]]


@dataclass(frozen=True)
class MethodGroup:
    title: str
    rows: Sequence[MethodRow]


@dataclass
class SweepRef:
    ref: str
    entity: Optional[str] = None
    project: Optional[str] = None


@dataclass
class CacheConfig:
    entity: str
    project: str
    datasets: List[str]
    attack_rates: List[int]
    domain_ids: List[int]
    sweeps: Dict[str, SweepRef]


METHOD_GROUPS: Sequence[MethodGroup] = (
    MethodGroup(
        title="No adaptation / source anchors",
        rows=(
            MethodRow("ERM", "erm_baseline", {"adapt_alg": ["ERM"]}),
            MethodRow("Robust ERM", "robust_erm_source", {"adapt_alg": ["ERM"], "use_adv_source": [1]}),
        ),
    ),
    MethodGroup(
        title="Standard TTA baselines",
        rows=(
            MethodRow("Tent", "tent_baseline", {"adapt_alg": ["Tent"]}),
            MethodRow("Tent+MedBN", "medbn_tent", {"adapt_alg": ["MedBN"]}),
            MethodRow("PL", "pl_baseline", {"adapt_alg": ["PL"]}),
            MethodRow("EATA", "eata_baseline", {"adapt_alg": ["EATA"]}),
            MethodRow("TSD", "tsd_baseline", {"adapt_alg": ["TSD"]}),
            MethodRow("TeSLA", "tesla_baseline", {"adapt_alg": ["TeSLA"]}),
        ),
    ),
    MethodGroup(
        title="SAFER plug-in variants with sigmoid/alpha",
        rows=(
            MethodRow("Tent + SAFER_S", "tent_safer_sig", {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
            MethodRow("PL + SAFER_S", "pl_safer_sig", {"adapt_alg": ["PL"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
            #MethodRow("EATA + SAFER_S", "eata_safer_sig", {"adapt_alg": ["EATA"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
            MethodRow("TSD + SAFER_S", "tsd_safer_sig", {"adapt_alg": ["TSD"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
        ),
    ),
    MethodGroup(
        title="SAFER plug-in variants (main claim)",
        rows=(
            MethodRow("Tent + SAFER", "tent_safer", {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
            MethodRow("PL + SAFER", "pl_safer", {"adapt_alg": ["PL"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
            #MethodRow("EATA + SAFER", "eata_safer", {"adapt_alg": ["EATA"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
            MethodRow("TSD + SAFER", "tsd_safer", {"adapt_alg": ["TSD"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
        ),
    ),
)


CACHE_TEMPLATE = """# Sweep-ID cache used by scripts/wandb_paper_tables.py
# Fill each alias with a W&B sweep id (abcd1234) or full sweep path (entity/project/abcd1234).
# Aliases can point to the same sweep when the run is selected by filters (e.g., adapt_alg).
entity: bigslav
project: safer

datasets:
  - PACS
  - VLCS
  - office-home

attack_rates: [0, 50, 100]
domain_ids: [0, 1, 2, 3]

sweeps:
  erm_baseline: ""
  robust_erm_source: ""
  tent_baseline: ""
  eata_baseline: ""
  tsd_baseline: ""
  tesla_baseline: ""
  medbn_tent: ""
  tent_safer: ""
  eata_safer: ""
  tsd_safer: ""
"""


def required_aliases() -> List[str]:
    aliases: List[str] = []
    for group in METHOD_GROUPS:
        for row in group.rows:
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


def parse_list_strings(raw: Any, field_name: str) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return wt.parse_csv_strings(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        out: List[str] = []
        for value in raw:
            if value is None:
                continue
            text = str(value).strip()
            if text:
                out.append(text)
        return out
    raise ValueError(f"Field '{field_name}' must be a list of strings or CSV string.")


def parse_list_ints(raw: Any, field_name: str) -> List[int]:
    if raw is None:
        return []
    if isinstance(raw, str):
        return wt.parse_csv_ints(raw)
    if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
        out: List[int] = []
        for value in raw:
            out.append(int(value))
        return out
    raise ValueError(f"Field '{field_name}' must be a list of ints or CSV string.")


def load_cache(
    path: Path,
    entity_override: Optional[str],
    project_override: Optional[str],
) -> CacheConfig:
    if not path.exists():
        raise ValueError(
            f"Cache file not found: {path}. "
            f"Create it with --init-cache or provide an existing file via --cache-file."
        )
    data = load_yaml_mapping(path)
    entity = entity_override or str(data.get("entity") or "bigslav")
    project = project_override or str(data.get("project") or "safer")
    datasets = parse_list_strings(data.get("datasets"), "datasets") or list(DEFAULT_DATASETS)
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
        datasets=datasets,
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
    verbose: bool,
) -> Dict[str, List[wt.RunRecord]]:
    records_by_path: Dict[str, List[wt.RunRecord]] = {}
    records_by_alias: Dict[str, List[wt.RunRecord]] = {}
    for alias in aliases:
        if alias not in sweeps or not sweeps[alias].ref:
            if verbose:
                print(f"[warn] No sweep configured for alias '{alias}'; using empty rows.", file=sys.stderr)
            records_by_alias[alias] = []
            continue
        spec = sweeps[alias]
        entity = spec.entity or default_entity
        project = spec.project or default_project
        ref = spec.ref.strip()
        sweep_path = resolve_sweep_path(ref, entity, project)
        if sweep_path not in records_by_path:
            records_by_path[sweep_path] = wt.load_runs_from_wandb_api(ref, entity, project)
            if verbose:
                print(
                    f"[info] Loaded {len(records_by_path[sweep_path])} runs from {sweep_path}",
                    file=sys.stderr,
                )
        records_by_alias[alias] = records_by_path[sweep_path]
    return records_by_alias


def dataset_display_name(dataset: str) -> str:
    key = dataset.strip().lower()
    if key == "pacs":
        return "PACS"
    if key == "vlcs":
        return "VLCS"
    if key in {"office-home", "officehome"}:
        return "OfficeHome"
    return dataset


def dataset_label_key(dataset: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", dataset.strip().lower())


def join_with_and(items: Sequence[str]) -> str:
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def attack_rates_text(attack_rates: Sequence[int]) -> str:
    parts = [f"{rate}\\%" for rate in attack_rates]
    return join_with_and(parts)


def domain_names_for_dataset(dataset: str, domain_ids: Sequence[int]) -> List[str]:
    labels = wt.DATASET_DOMAIN_LABELS.get(dataset)
    out: List[str] = []
    for dom_id in domain_ids:
        if labels and 0 <= dom_id < len(labels):
            out.append(labels[dom_id])
        else:
            out.append(f"Domain {dom_id}")
    return out


def normalize_cells(cells: Sequence[str], expected_len: int) -> List[str]:
    data = list(cells)
    if len(data) < expected_len:
        data.extend(["--"] * (expected_len - len(data)))
    elif len(data) > expected_len:
        data = data[:expected_len]
    return data


def build_dataset_row_cells(
    row: MethodRow,
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_ids: Sequence[int],
    attack_rates: Sequence[int],
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    precision: int,
    verbose: bool,
) -> List[str]:
    rows = wt.build_rows(
        records=records,
        methods={row.label: row.filters},
        filters={"dataset": [dataset]},
        domain_ids=domain_ids,
        attack_rates=attack_rates,
        mean_key=mean_key,
        std_key=std_key,
        select_mode=select_mode,
        select_metric=select_metric,
        precision=precision,
        verbose=verbose,
        std_style="subscript",
    )
    if not rows:
        return ["--"] * (len(domain_ids) * len(attack_rates))
    return rows[0][1]


def build_domain_avg_row_cells(
    row: MethodRow,
    records: Sequence[wt.RunRecord],
    datasets: Sequence[str],
    domain_ids: Sequence[int],
    attack_rates: Sequence[int],
    mean_key: str,
    select_mode: str,
    select_metric: str,
    precision: int,
    verbose: bool,
) -> List[str]:
    rows = wt.build_rows_dataset_domain_avg(
        records=records,
        methods={row.label: row.filters},
        filters={},
        datasets=datasets,
        domain_ids=domain_ids,
        attack_rates=attack_rates,
        mean_key=mean_key,
        select_mode=select_mode,
        select_metric=select_metric,
        precision=precision,
        verbose=verbose,
        include_domain_std=False,
        std_style="subscript",
    )
    if not rows:
        return ["--"] * (len(datasets) * len(attack_rates))
    return rows[0][1]


def render_table(
    caption: str,
    label: str,
    column_groups: Sequence[str],
    attack_rates: Sequence[int],
    row_cells: Mapping[str, Sequence[str]],
    placement: str,
) -> str:
    num_rates = len(attack_rates)
    expected_cells = len(column_groups) * num_rates
    total_columns = 1 + expected_cells
    col_spec = "l|" + "|".join("c" * num_rates for _ in column_groups)

    lines = [
        f"\\begin{{table*}}[{placement}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\textwidth}{!}{",
        "\\setlength{\\tabcolsep}{3pt}",
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]

    header_groups: List[str] = []
    for idx, name in enumerate(column_groups):
        suffix = "|" if idx < len(column_groups) - 1 else ""
        header_groups.append(f"\\multicolumn{{{num_rates}}}{{c{suffix}}}{{{name}}}")
    lines.append("\\multirow{2}{*}{Method} & " + " & ".join(header_groups) + " \\\\")

    rate_cells: List[str] = []
    for _ in column_groups:
        rate_cells.extend([f"{rate}\\%" for rate in attack_rates])
    lines.append("& " + " & ".join(rate_cells) + " \\\\")
    lines.append("\\hline")
    lines.append("")

    for group in METHOD_GROUPS:
        lines.append(f"%\\multicolumn{{{total_columns}}}{{l}}{{\\textit{{{group.title}}}}} \\\\")
        for row in group.rows:
            cells = normalize_cells(row_cells.get(row.label, []), expected_cells)
            lines.append(f"{row.label} & " + " & ".join(cells) + " \\\\")
        lines.append("")
        lines.append("\\hline")

    lines.extend(
        [
            "\\end{tabular}",
            "}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate full ready-to-paste LaTeX paper tables from W&B sweeps "
            "with sweep IDs cached in a YAML file."
        )
    )
    parser.add_argument(
        "--cache-file",
        type=Path,
        default=DEFAULT_CACHE_PATH,
        help=f"Sweep-ID cache file (default: {DEFAULT_CACHE_PATH}).",
    )
    parser.add_argument(
        "--init-cache",
        action="store_true",
        help="Write a template cache file and exit.",
    )
    parser.add_argument(
        "--force-init-cache",
        action="store_true",
        help="Allow --init-cache to overwrite an existing file.",
    )
    parser.add_argument("--entity", help="Default W&B entity (overrides cache file).")
    parser.add_argument("--project", help="Default W&B project (overrides cache file).")

    parser.add_argument("--datasets", help="Comma-separated datasets (default: cache file).")
    parser.add_argument("--attack-rates", help="Comma-separated attack rates (default: cache file).")
    parser.add_argument("--domain-ids", help="Comma-separated domain ids (default: cache file).")

    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for mean accuracy.")
    parser.add_argument("--std-key", default="acc_std", help="Summary key for std accuracy.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric used for run selection (default: mean key).")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision.")

    parser.add_argument(
        "--label-suffix",
        default="main",
        help="Per-dataset label suffix used in tab:<dataset>-<suffix>.",
    )
    parser.add_argument(
        "--domain-avg-label",
        default=None,
        help="Label for the domain-averaged table (default: tab:domain-avg-<label-suffix>).",
    )
    parser.add_argument("--placement", default="t", help="LaTeX placement for table*.")
    parser.add_argument(
        "--no-domain-avg",
        action="store_true",
        help="Disable the dataset-column domain-averaged table.",
    )
    parser.add_argument("--output", type=Path, help="Optional output .tex file (stdout if omitted).")
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")

    args = parser.parse_args()

    if args.init_cache:
        write_cache_template(args.cache_file, force=args.force_init_cache)
        print(f"Wrote sweep cache template to {args.cache_file}")
        return 0

    cache_cfg = load_cache(args.cache_file, args.entity, args.project)
    datasets = wt.parse_csv_strings(args.datasets) if args.datasets else cache_cfg.datasets
    attack_rates = wt.parse_csv_ints(args.attack_rates) if args.attack_rates else cache_cfg.attack_rates
    domain_ids = wt.parse_csv_ints(args.domain_ids) if args.domain_ids else cache_cfg.domain_ids

    if not datasets:
        raise ValueError("No datasets configured.")
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
        verbose=args.verbose,
    )

    select_metric = args.select_metric or args.mean_key
    rate_text = attack_rates_text(attack_rates)
    tables: List[str] = []

    for dataset in datasets:
        domain_names = domain_names_for_dataset(dataset, domain_ids)
        row_cells: Dict[str, List[str]] = {}
        for group in METHOD_GROUPS:
            for row in group.rows:
                row_cells[row.label] = build_dataset_row_cells(
                    row=row,
                    records=records_by_alias[row.sweep_alias],
                    dataset=dataset,
                    domain_ids=domain_ids,
                    attack_rates=attack_rates,
                    mean_key=args.mean_key,
                    std_key=args.std_key,
                    select_mode=args.select,
                    select_metric=select_metric,
                    precision=args.precision,
                    verbose=args.verbose,
                )

        caption = (
            f"{dataset_display_name(dataset)} accuracy (\\%) under "
            f"$\\ell_\\infty$ attacks with attack rates {rate_text} per target domain."
        )
        label = f"tab:{dataset_label_key(dataset)}-{args.label_suffix}"
        tables.append(
            render_table(
                caption=caption,
                label=label,
                column_groups=domain_names,
                attack_rates=attack_rates,
                row_cells=row_cells,
                placement=args.placement,
            )
        )

    if not args.no_domain_avg:
        dataset_headers = [dataset_display_name(dataset) for dataset in datasets]
        row_cells = {}
        for group in METHOD_GROUPS:
            for row in group.rows:
                row_cells[row.label] = build_domain_avg_row_cells(
                    row=row,
                    records=records_by_alias[row.sweep_alias],
                    datasets=datasets,
                    domain_ids=domain_ids,
                    attack_rates=attack_rates,
                    mean_key=args.mean_key,
                    select_mode=args.select,
                    select_metric=select_metric,
                    precision=args.precision,
                    verbose=args.verbose,
                )

        dataset_text = join_with_and(dataset_headers)
        caption = (
            f"Domain-averaged accuracy (\\%) under $\\ell_\\infty$ attacks with "
            f"attack rates {rate_text} across {dataset_text}."
        )
        domain_avg_label = args.domain_avg_label or f"tab:domain-avg-{args.label_suffix}"
        tables.append(
            render_table(
                caption=caption,
                label=domain_avg_label,
                column_groups=dataset_headers,
                attack_rates=attack_rates,
                row_cells=row_cells,
                placement=args.placement,
            )
        )

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
