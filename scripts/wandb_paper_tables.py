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
            MethodRow("Tent + SAFER-A", "tent_safer_sig", {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
            MethodRow("PL + SAFER-A", "pl_safer_sig", {"adapt_alg": ["PL"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
            #MethodRow("EATA + SAFER-A", "eata_safer_sig", {"adapt_alg": ["EATA"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
            MethodRow("TSD + SAFER-A", "tsd_safer_sig", {"adapt_alg": ["TSD"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
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
    # MethodGroup(
    #     title="SAFER Standalone",
    #     rows=(
    #         MethodRow("Standalone SAFER-A", "safer_sig", {"adapt_alg": ["SAFER"], "s_wrap_alg": [0], "s_alpha_mode": ["sigmoid"]}),
    #         MethodRow("Standalone SAFER", "safer", {"adapt_alg": ["SAFER"], "s_wrap_alg": [0], "s_alpha_mode": ["none"]}),
    #     ),
    # ),
)


# METHOD_GROUPS: Sequence[MethodGroup] = (
#     MethodGroup(
#         title="ERM",
#         rows=(
#             MethodRow("ERM", "base", {"adapt_alg": ["ERM"]}),
#             MethodRow("+FFT", "fft", {"adapt_alg": ["ERM"]}),
#             MethodRow("+Gauss", "blur", {"adapt_alg": ["ERM"]}),
#             MethodRow("+JPEG", "jpeg", {"adapt_alg": ["ERM"]}),
#             MethodRow("+Adv", "adv", {"adapt_alg": ["ERM"], "use_adv_source": [1]}),
#         ),
#     ),
#     MethodGroup(
#         title="Tent and Defenses",
#         rows=(
#             MethodRow("Tent", "base", {"adapt_alg": ["Tent"]}),
#             MethodRow("+FFT", "fft", {"adapt_alg": ["Tent"]}),
#             MethodRow("+Gauss", "blur", {"adapt_alg": ["Tent"]}),
#             MethodRow("+JPEG", "jpeg", {"adapt_alg": ["Tent"]}),
#             MethodRow("+Adv", "adv", {"adapt_alg": ["Tent"], "use_adv_source": [1]}),
#         ),
#     ),
#     MethodGroup(
#         title="TSD and Defenses",
#         rows=(
#             MethodRow("TSD", "base", {"adapt_alg": ["TSD"]}),
#             MethodRow("+FFT", "fft", {"adapt_alg": ["TSD"]}),
#             MethodRow("+Gauss", "blur", {"adapt_alg": ["TSD"]}),
#             MethodRow("+JPEG", "jpeg", {"adapt_alg": ["TSD"]}),
#             MethodRow("+Adv", "adv", {"adapt_alg": ["TSD"], "use_adv_source": [1]}),
#         ),
#     ),
# )


CACHE_TEMPLATE = """# Sweep-ID cache used by scripts/wandb_paper_tables.py
# Fill each alias with a W&B sweep id (abcd1234) or full sweep path (entity/project/abcd1234).
# Aliases can point to the same sweep when the run is selected by filters (e.g., adapt_alg).
entity: bigslav
project: safer

datasets:
  - PACS

attack_rates: [0, 100]
domain_ids: [0, 1, 2, 3]

sweeps:
  base: ""
  adv: ""
  jpeg: ""
  blur: ""
  fft: ""
"""


# CACHE_TEMPLATE = """# Sweep-ID cache used by scripts/wandb_paper_tables.py
# # Fill each alias with a W&B sweep id (abcd1234) or full sweep path (entity/project/abcd1234).
# # Aliases can point to the same sweep when the run is selected by filters (e.g., adapt_alg).
# entity: bigslav
# project: safer

# datasets:
#   - PACS

# attack_rates: [0, 100]
# domain_ids: [0, 1, 2, 3]

# sweeps:
#   base: ""
#   adv: ""
#   jpeg: ""
#   blur: ""
#   fft: ""
# """


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
    wandb_timeout: Optional[int],
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


def row_key(group_idx: int, row_idx: int, label: str) -> str:
    return f"g{group_idx}r{row_idx}:{label}"


def all_method_ids() -> List[str]:
    row_ids: List[str] = []
    for group_idx, group in enumerate(METHOD_GROUPS):
        for row_idx, row in enumerate(group.rows):
            row_ids.append(row_key(group_idx, row_idx, row.label))
    return row_ids


def parse_cell_mean(cell: str) -> Optional[float]:
    text = cell.strip()
    if not text or text == "--":
        return None
    match = re.match(r"^([-+]?\d+(?:\.\d+)?)", text)
    if not match:
        return None
    try:
        return float(match.group(1))
    except ValueError:
        return None


def split_cell_mean_suffix(cell: str) -> Optional[tuple[str, str]]:
    text = cell.strip()
    if not text or text == "--":
        return None
    match = re.match(r"^([-+]?\d+(?:\.\d+)?)(.*)$", text)
    if not match:
        return None
    mean_text = match.group(1)
    suffix = match.group(2) or ""
    return mean_text, suffix


def integer_digit_count(mean_text: str) -> int:
    if not mean_text:
        return 1
    int_part = mean_text.split(".", 1)[0]
    if int_part.startswith("-") or int_part.startswith("+"):
        int_part = int_part[1:]
    return max(len(int_part), 1)


def decimal_pad_prefix(mean_text: str, target_digits: int) -> str:
    missing = target_digits - integer_digit_count(mean_text)
    if missing <= 0:
        return ""
    return f"\\phantom{{{'0' * missing}}}"


def style_cells_by_rank(raw_cells_by_row: Mapping[str, Sequence[str]], row_ids: Sequence[str]) -> Dict[str, List[str]]:
    if not row_ids:
        return {}

    expected_cells = 0
    for row_id in row_ids:
        expected_cells = max(expected_cells, len(raw_cells_by_row.get(row_id, [])))

    best_rows_by_col: List[set[str]] = [set() for _ in range(expected_cells)]
    second_rows_by_col: List[set[str]] = [set() for _ in range(expected_cells)]
    max_int_digits_by_col: List[int] = [1 for _ in range(expected_cells)]

    for col_idx in range(expected_cells):
        values: List[tuple[str, float]] = []
        int_width = 1
        for row_id in row_ids:
            raw = raw_cells_by_row.get(row_id, [])
            cell = raw[col_idx] if col_idx < len(raw) else "--"
            mean_val = parse_cell_mean(cell)
            if mean_val is not None:
                values.append((row_id, mean_val))
                mean_suffix = split_cell_mean_suffix(cell)
                if mean_suffix is not None:
                    int_width = max(int_width, integer_digit_count(mean_suffix[0]))
        max_int_digits_by_col[col_idx] = int_width
        if not values:
            continue

        unique_desc = sorted({value for _, value in values}, reverse=True)
        best = unique_desc[0]
        second = unique_desc[1] if len(unique_desc) > 1 else None

        for row_id, mean_val in values:
            if abs(mean_val - best) < 1e-12:
                best_rows_by_col[col_idx].add(row_id)
            elif second is not None and abs(mean_val - second) < 1e-12:
                second_rows_by_col[col_idx].add(row_id)

    styled: Dict[str, List[str]] = {}
    for row_id in row_ids:
        raw = list(raw_cells_by_row.get(row_id, []))
        if len(raw) < expected_cells:
            raw.extend(["--"] * (expected_cells - len(raw)))

        out_cells: List[str] = []
        for col_idx, cell in enumerate(raw):
            mean_suffix = split_cell_mean_suffix(cell)
            if mean_suffix is None:
                out_cells.append(cell)
                continue

            mean_text, suffix = mean_suffix
            prefix = decimal_pad_prefix(mean_text, max_int_digits_by_col[col_idx])
            styled_mean = mean_text
            if row_id in best_rows_by_col[col_idx]:
                styled_mean = f"\\textbf{{{mean_text}}}"
            elif row_id in second_rows_by_col[col_idx]:
                styled_mean = f"\\underline{{{mean_text}}}"
            out_cells.append(f"{prefix}{styled_mean}{suffix}")

        styled[row_id] = out_cells

    return styled


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
    include_std: bool,
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
        include_std=include_std,
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
    include_all_domains_column: bool,
    verbose: bool,
) -> List[str]:
    matched = [
        record
        for record in records
        if mean_key in record.summary and wt.record_matches_filters(record, row.filters)
    ]

    grouped: Dict[tuple[str, int, int], List[wt.RunRecord]] = {}
    for record in matched:
        dataset_name = wt.get_value(record, "dataset")
        if not isinstance(dataset_name, str):
            continue
        dom_val = wt.get_value(record, "test_envs", "test_env")
        if isinstance(dom_val, list):
            dom_val = dom_val[0] if dom_val else None
        dom_id = wt.to_int(dom_val)
        rate = wt.to_int(wt.get_value(record, "attack_rate"))
        if dom_id is None or rate is None:
            continue
        grouped.setdefault((dataset_name, dom_id, rate), []).append(record)

    cells: List[str] = []
    overall_by_rate: Dict[int, List[float]] = {rate: [] for rate in attack_rates} if include_all_domains_column else {}

    for dataset in datasets:
        for rate in attack_rates:
            domain_means: List[float] = []
            for dom_id in domain_ids:
                runs = grouped.get((dataset, dom_id, rate), [])
                if not runs:
                    continue
                chosen = wt.select_run(runs, select_mode, select_metric)
                mean_val = chosen.summary.get(mean_key)
                try:
                    mean_float = float(mean_val)
                except (TypeError, ValueError):
                    continue
                domain_means.append(mean_float)
                if include_all_domains_column:
                    overall_by_rate[rate].append(mean_float)

            if not domain_means:
                if verbose:
                    print(
                        f"[warn] Missing dataset {dataset} rate {rate} for {row.label}",
                        file=sys.stderr,
                    )
                cells.append("--")
                continue

            mean_val = sum(domain_means) / len(domain_means)
            cells.append(
                wt.format_cell(
                    mean_val,
                    None,
                    precision,
                    show_std=False,
                    std_style="subscript",
                )
            )

    if include_all_domains_column:
        for rate in attack_rates:
            values = overall_by_rate.get(rate, [])
            if not values:
                if verbose:
                    print(
                        f"[warn] Missing overall rate {rate} for {row.label}",
                        file=sys.stderr,
                    )
                cells.append("--")
                continue
            mean_val = sum(values) / len(values)
            cells.append(
                wt.format_cell(
                    mean_val,
                    None,
                    precision,
                    show_std=False,
                    std_style="subscript",
                )
            )

    return cells


def render_table(
    caption: str,
    label: str,
    column_groups: Sequence[str],
    attack_rates: Sequence[int],
    row_cells: Mapping[str, Sequence[str]],
    placement: str,
    style_rank: bool = True,
) -> str:
    num_rates = len(attack_rates)
    expected_cells = len(column_groups) * num_rates
    total_columns = 1 + expected_cells
    col_spec = "l|" + "|".join("r" * num_rates for _ in column_groups)

    row_id_to_label: Dict[str, str] = {}
    raw_cells_by_row: Dict[str, List[str]] = {}
    for group_idx, group in enumerate(METHOD_GROUPS):
        for row_idx, row in enumerate(group.rows):
            row_id = row_key(group_idx, row_idx, row.label)
            row_id_to_label[row_id] = row.label
            cells = row_cells.get(row_id, row_cells.get(row.label, []))
            raw_cells_by_row[row_id] = normalize_cells(cells, expected_cells)
    row_ids = all_method_ids()
    styled_cells_by_row = style_cells_by_rank(raw_cells_by_row, row_ids) if style_rank else raw_cells_by_row

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

    for group_idx, group in enumerate(METHOD_GROUPS):
        lines.append(f"%\\multicolumn{{{total_columns}}}{{l}}{{\\textit{{{group.title}}}}} \\\\")
        for row_idx, row in enumerate(group.rows):
            row_id = row_key(group_idx, row_idx, row.label)
            cells = styled_cells_by_row.get(row_id, ["--"] * expected_cells)
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
    parser.add_argument("--precision", type=int, default=1, help="Decimal precision.")
    parser.add_argument(
        "--include-std",
        action="store_true",
        help="Include std in non-domain-avg dataset tables (default: mean-only).",
    )
    parser.add_argument(
        "--no-rank-style",
        action="store_true",
        help="Disable bold/underline rank styling in table cells.",
    )

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
    parser.add_argument(
        "--no-all-domains-column",
        action="store_true",
        help="Exclude the final 'All Domains' column block in the domain-averaged table.",
    )
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
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )

    select_metric = args.select_metric or args.mean_key
    rate_text = attack_rates_text(attack_rates)
    tables: List[str] = []

    for dataset in datasets:
        domain_names = domain_names_for_dataset(dataset, domain_ids)
        row_cells: Dict[str, List[str]] = {}
        for group_idx, group in enumerate(METHOD_GROUPS):
            for row_idx, row in enumerate(group.rows):
                row_cells[row_key(group_idx, row_idx, row.label)] = build_dataset_row_cells(
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
                    include_std=args.include_std,
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
                style_rank=not args.no_rank_style,
            )
        )

    if not args.no_domain_avg:
        dataset_headers = [dataset_display_name(dataset) for dataset in datasets]
        include_all_domains_column = not args.no_all_domains_column
        if include_all_domains_column:
            dataset_headers.append("All Domains")
        row_cells = {}
        for group_idx, group in enumerate(METHOD_GROUPS):
            for row_idx, row in enumerate(group.rows):
                row_cells[row_key(group_idx, row_idx, row.label)] = build_domain_avg_row_cells(
                    row=row,
                    records=records_by_alias[row.sweep_alias],
                    datasets=datasets,
                    domain_ids=domain_ids,
                    attack_rates=attack_rates,
                    mean_key=args.mean_key,
                    select_mode=args.select,
                    select_metric=select_metric,
                    precision=args.precision,
                    include_all_domains_column=include_all_domains_column,
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
                style_rank=not args.no_rank_style,
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
