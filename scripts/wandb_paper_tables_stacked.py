#!/usr/bin/env python3
from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import wandb_table as wt  # noqa: E402


DEFAULT_ENTITY = "bigslav"
DEFAULT_PROJECT = "safer"
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


@dataclass(frozen=True)
class SweepRef:
    ref: str
    entity: Optional[str] = None
    project: Optional[str] = None


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
            MethodRow("TSD + SAFER-A", "tsd_safer_sig", {"adapt_alg": ["TSD"], "s_wrap_alg": [1], "s_alpha_mode": ["sigmoid"]}),
        ),
    ),
    MethodGroup(
        title="SAFER plug-in variants (main claim)",
        rows=(
            MethodRow("Tent + SAFER", "tent_safer", {"adapt_alg": ["Tent"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
            MethodRow("PL + SAFER", "pl_safer", {"adapt_alg": ["PL"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
            MethodRow("TSD + SAFER", "tsd_safer", {"adapt_alg": ["TSD"], "s_wrap_alg": [1], "s_alpha_mode": ["none"]}),
        ),
    ),
)


# Same sweep IDs as sweeps/wandb_paper_table_cache.yaml.
SWEEPS: Dict[str, SweepRef] = {
    "erm_baseline": SweepRef(ref="4hsk0s2k"),
    "robust_erm_source": SweepRef(ref="mw633n64"),
    "tent_baseline": SweepRef(ref="4hsk0s2k"),
    "pl_baseline": SweepRef(ref="4hsk0s2k"),
    "eata_baseline": SweepRef(ref="41h9zln6"),
    "tsd_baseline": SweepRef(ref="4hsk0s2k"),
    "tesla_baseline": SweepRef(ref="4hsk0s2k"),
    "medbn_tent": SweepRef(ref="ae0oxsym"),
    "tent_safer": SweepRef(ref="fpvy9adq"),
    "pl_safer": SweepRef(ref="fpvy9adq"),
    "eata_safer": SweepRef(ref="fpvy9adq"),
    "tsd_safer": SweepRef(ref="fpvy9adq"),
    "tesla_safer": SweepRef(ref="fpvy9adq"),
    "tent_safer_sig": SweepRef(ref="qo6yo28k"),
    "pl_safer_sig": SweepRef(ref="qo6yo28k"),
    "eata_safer_sig": SweepRef(ref="qo6yo28k"),
    "tsd_safer_sig": SweepRef(ref="qo6yo28k"),
    "tesla_safer_sig": SweepRef(ref="qo6yo28k"),
    "safer": SweepRef(ref="4r8v4lcw"),
    "safer_sig": SweepRef(ref="8r91gthz"),
}


def required_aliases() -> List[str]:
    aliases: List[str] = []
    for group in METHOD_GROUPS:
        for row in group.rows:
            if row.sweep_alias not in aliases:
                aliases.append(row.sweep_alias)
    return aliases


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
        spec = sweeps.get(alias)
        if spec is None or not spec.ref:
            if verbose:
                print(f"[warn] No sweep configured for alias '{alias}'; using empty rows.", file=sys.stderr)
            records_by_alias[alias] = []
            continue

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
                print(f"[info] Loaded {len(records_by_path[sweep_path])} runs from {sweep_path}", file=sys.stderr)
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


def build_dataset_rate_row_cells(
    row: MethodRow,
    records: Sequence[wt.RunRecord],
    dataset: str,
    domain_ids: Sequence[int],
    attack_rate: int,
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
        attack_rates=[attack_rate],
        mean_key=mean_key,
        std_key=std_key,
        select_mode=select_mode,
        select_metric=select_metric,
        precision=precision,
        verbose=verbose,
        std_style="subscript",
    )
    if not rows:
        return ["--"] * len(domain_ids)
    return normalize_cells(rows[0][1], len(domain_ids))


def attack_rate_title(attack_rate: int) -> str:
    if attack_rate == 0:
        return "Clean (0\\%)"
    if attack_rate == 50:
        return "Mixed (50\\%)"
    if attack_rate == 100:
        return "Fully attacked (100\\%)"
    return f"Attack rate ({attack_rate}\\%)"


def render_stacked_table(
    caption: str,
    label: str,
    domain_names: Sequence[str],
    attack_rates: Sequence[int],
    row_cells_by_rate: Mapping[int, Mapping[str, Sequence[str]]],
    placement: str,
    style_rank: bool = True,
) -> str:
    expected_cells = len(domain_names)
    total_columns = 1 + expected_cells
    col_spec = "l|" + "r" * len(domain_names)
    row_ids = all_method_ids()

    lines = [
        f"\\begin{{table*}}[{placement}]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\setlength{\\tabcolsep}{3pt}",
    ]

    for idx, rate in enumerate(attack_rates):
        if idx > 0:
            lines.extend(["", "\\vspace{0.5em}"])

        lines.extend(
            [
                f"\\textbf{{{attack_rate_title(rate)}}}\\\\",
                "\\vspace{0.2em}",
                f"\\begin{{tabular}}{{{col_spec}}}",
                "\\hline",
                "Method & " + " & ".join(domain_names) + " \\\\",
                "\\hline",
                "",
            ]
        )

        raw_cells_by_row: Dict[str, List[str]] = {}
        row_cells = row_cells_by_rate.get(rate, {})
        for group_idx, group in enumerate(METHOD_GROUPS):
            for row_idx, row in enumerate(group.rows):
                row_id = row_key(group_idx, row_idx, row.label)
                cells = row_cells.get(row_id, row_cells.get(row.label, []))
                raw_cells_by_row[row_id] = normalize_cells(cells, expected_cells)

        styled_cells_by_row = style_cells_by_rank(raw_cells_by_row, row_ids) if style_rank else raw_cells_by_row

        for group_idx, group in enumerate(METHOD_GROUPS):
            lines.append(f"%\\multicolumn{{{total_columns}}}{{l}}{{\\textit{{{group.title}}}}} \\\\")
            for row_idx, row in enumerate(group.rows):
                row_id = row_key(group_idx, row_idx, row.label)
                cells = styled_cells_by_row.get(row_id, ["--"] * expected_cells)
                lines.append(f"{row.label} & " + " & ".join(cells) + " \\\\")
            lines.extend(["", "\\hline"])

        lines.append("\\end{tabular}")

    lines.append("\\end{table*}")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Generate ready-to-paste LaTeX paper tables from W&B sweeps using "
            "hardcoded sweep IDs, rendered as stacked Clean/Mixed/Fully-attacked subtables."
        )
    )
    parser.add_argument("--entity", default=DEFAULT_ENTITY, help=f"W&B entity (default: {DEFAULT_ENTITY}).")
    parser.add_argument("--project", default=DEFAULT_PROJECT, help=f"W&B project (default: {DEFAULT_PROJECT}).")

    parser.add_argument("--datasets", help="Comma-separated datasets.")
    parser.add_argument("--attack-rates", help="Comma-separated attack rates.")
    parser.add_argument("--domain-ids", help="Comma-separated domain ids.")

    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for mean accuracy.")
    parser.add_argument("--std-key", default="acc_std", help="Summary key for std accuracy.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric used for run selection (default: mean key).")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision.")
    parser.add_argument(
        "--no-rank-style",
        action="store_true",
        help="Disable bold/underline rank styling in table cells.",
    )

    parser.add_argument(
        "--label-suffix",
        default="main-stacked",
        help="Per-dataset label suffix used in tab:<dataset>-<suffix>.",
    )
    parser.add_argument("--placement", default="t", help="LaTeX placement for table*.")
    parser.add_argument("--output", type=Path, help="Optional output .tex file (stdout if omitted).")
    parser.add_argument(
        "--wandb-timeout",
        type=int,
        default=60,
        help="W&B API timeout in seconds (default: 60).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print diagnostics to stderr.")

    args = parser.parse_args()

    datasets = wt.parse_csv_strings(args.datasets) if args.datasets else list(DEFAULT_DATASETS)
    attack_rates = wt.parse_csv_ints(args.attack_rates) if args.attack_rates else list(DEFAULT_ATTACK_RATES)
    domain_ids = wt.parse_csv_ints(args.domain_ids) if args.domain_ids else list(DEFAULT_DOMAIN_IDS)

    if not datasets:
        raise ValueError("No datasets configured.")
    if not attack_rates:
        raise ValueError("No attack rates configured.")
    if not domain_ids:
        raise ValueError("No domain ids configured.")

    aliases = required_aliases()
    missing_aliases = [alias for alias in aliases if alias not in SWEEPS]
    if missing_aliases:
        missing_str = ", ".join(missing_aliases)
        raise ValueError(f"Missing hardcoded sweep IDs for aliases: {missing_str}")

    records_by_alias = fetch_records(
        sweeps=SWEEPS,
        aliases=aliases,
        default_entity=args.entity,
        default_project=args.project,
        wandb_timeout=args.wandb_timeout,
        verbose=args.verbose,
    )

    select_metric = args.select_metric or args.mean_key
    tables: List[str] = []

    for dataset in datasets:
        domain_names = domain_names_for_dataset(dataset, domain_ids)

        row_cells_by_rate: Dict[int, Dict[str, List[str]]] = {}
        for attack_rate in attack_rates:
            row_cells: Dict[str, List[str]] = {}
            for group_idx, group in enumerate(METHOD_GROUPS):
                for row_idx, row in enumerate(group.rows):
                    row_cells[row_key(group_idx, row_idx, row.label)] = build_dataset_rate_row_cells(
                        row=row,
                        records=records_by_alias[row.sweep_alias],
                        dataset=dataset,
                        domain_ids=domain_ids,
                        attack_rate=attack_rate,
                        mean_key=args.mean_key,
                        std_key=args.std_key,
                        select_mode=args.select,
                        select_metric=select_metric,
                        precision=args.precision,
                        verbose=args.verbose,
                    )
            row_cells_by_rate[attack_rate] = row_cells

        caption = (
            f"{dataset_display_name(dataset)} accuracy (\\%) under $\\ell_\\infty$ attacks "
            "with vertically stacked clean/mixed/fully attacked subtables per target domain."
        )
        label = f"tab:{dataset_label_key(dataset)}-{args.label_suffix}"

        tables.append(
            render_stacked_table(
                caption=caption,
                label=label,
                domain_names=domain_names,
                attack_rates=attack_rates,
                row_cells_by_rate=row_cells_by_rate,
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
