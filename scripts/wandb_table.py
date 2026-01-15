#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
import json

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

try:
    from adapt_presets import ADAPT_ALG_ORDER
except Exception:
    ADAPT_ALG_ORDER = []


DATASET_DOMAIN_LABELS: Mapping[str, List[str]] = {
    "PACS": ["Art Painting", "Cartoon", "Photo", "Sketch"],
    "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "office-home": ["Art", "Clipart", "Product", "RealWorld"],
    #"DomainNet": ["clipart", "infograph", "painting", "quickdraw", "real", "sketch"],
}


@dataclass
class RunRecord:
    path: str
    config: Dict[str, Any]
    summary: Dict[str, Any]
    mtime: float


def coerce_value(text: str) -> Any:
    text = text.strip()
    if not text:
        return ""
    lower = text.lower()
    if lower in ("true", "false"):
        return lower == "true"
    if lower in ("none", "null"):
        return None
    try:
        return int(text)
    except ValueError:
        pass
    try:
        return float(text)
    except ValueError:
        return text


def normalize_value(value: Any) -> Any:
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, str)):
        return value
    if isinstance(value, float):
        if abs(value - round(value)) < 1e-9:
            return int(round(value))
        return round(value, 8)
    if isinstance(value, (list, tuple)):
        return [normalize_value(v) for v in value]
    return value


def value_matches(config_value: Any, allowed_values: Iterable[Any]) -> bool:
    if isinstance(config_value, (list, tuple)):
        return any(value_matches(v, allowed_values) for v in config_value)
    config_value = normalize_value(config_value)
    for allowed in allowed_values:
        allowed = normalize_value(allowed)
        if isinstance(config_value, float) and isinstance(allowed, float):
            if abs(config_value - allowed) < 1e-6:
                return True
        elif config_value == allowed:
            return True
    return False


def parse_filter_args(filter_args: Sequence[str]) -> Dict[str, List[Any]]:
    filters: Dict[str, List[Any]] = {}
    for raw in filter_args:
        if "=" not in raw:
            raise ValueError(f"Invalid filter (expected key=value): {raw}")
        key, values = raw.split("=", 1)
        key = key.strip()
        if not key:
            continue
        tokens = [v for v in values.replace("|", ",").split(",") if v != ""]
        parsed = [coerce_value(v) for v in tokens] if tokens else [""]
        filters.setdefault(key, []).extend(parsed)
    return filters


def load_yaml_file(path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML files.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Unexpected YAML format: {path}")
    return data


def load_sweep_filters(path: Path) -> Dict[str, List[Any]]:
    data = load_yaml_file(path)
    params = data.get("parameters", {})
    filters: Dict[str, List[Any]] = {}
    if not isinstance(params, Mapping):
        return filters
    for key, spec in params.items():
        if not isinstance(spec, Mapping):
            continue
        values = None
        if "values" in spec:
            values = spec["values"]
        elif "value" in spec:
            values = [spec["value"]]
        if values is None:
            continue
        if isinstance(values, list):
            filters[key] = [coerce_value(str(v)) if isinstance(v, str) else v for v in values]
        else:
            filters[key] = [values]
    return filters




def load_runs_from_wandb_api(sweep_id: str, entity: Optional[str], project: Optional[str]) -> List[RunRecord]:
    import wandb  # type: ignore

    api = wandb.Api()
    if "/" in sweep_id:
        sweep_path = sweep_id
    else:
        if not entity or not project:
            raise ValueError("Provide --entity and --project when using a bare sweep id.")
        sweep_path = f"{entity}/{project}/{sweep_id}"
    sweep = api.sweep(sweep_path)
    records: List[RunRecord] = []
    for run in sweep.runs:
        config = {k: v for k, v in run.config.items() if not k.startswith("_")}
        summary = dict(run.summary)
        records.append(RunRecord(path=run.id, config=config, summary=summary, mtime=0.0))
    return records


def get_value(record: RunRecord, *keys: str) -> Any:
    for key in keys:
        if key in record.config:
            return record.config[key]
        if key in record.summary:
            return record.summary[key]
    return None


def record_matches_filters(record: RunRecord, filters: Mapping[str, List[Any]]) -> bool:
    for key, allowed in filters.items():
        value = get_value(record, key)
        if value is None:
            return False
        if not value_matches(value, allowed):
            return False
    return True


def to_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and abs(value - round(value)) < 1e-8:
        return int(round(value))
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def select_run(runs: Sequence[RunRecord], mode: str, metric: str) -> RunRecord:
    if mode == "latest":
        return max(runs, key=lambda r: r.mtime)
    if mode == "first":
        return min(runs, key=lambda r: r.mtime)
    return max(runs, key=lambda r: r.summary.get(metric, float("-inf")))


def format_cell(mean: Optional[float], std: Optional[float], precision: int) -> str:
    if mean is None:
        return "--"
    try:
        mean_val = float(mean)
    except (TypeError, ValueError):
        return str(mean)
    if std is None:
        std_val = 0.0
    else:
        try:
            std_val = float(std)
        except (TypeError, ValueError):
            std_val = 0.0
    fmt = f"{{:.{precision}f}}"
    return f"{fmt.format(mean_val)}$\\pm${fmt.format(std_val)}"


def resolve_domain_names(dataset: Optional[str], domain_names: Optional[List[str]]) -> Optional[List[str]]:
    if domain_names:
        return domain_names
    if dataset and dataset in DATASET_DOMAIN_LABELS:
        return list(DATASET_DOMAIN_LABELS[dataset])
    return None


def latex_header(domain_names: Sequence[str], attack_rates: Sequence[int]) -> List[str]:
    num_rates = len(attack_rates)
    col_groups = ["c" * num_rates for _ in domain_names]
    col_spec = "l|" + "|".join(col_groups)
    lines = [
        f"\\begin{{tabular}}{{{col_spec}}}",
        "\\hline",
    ]
    header_cells = []
    for idx, name in enumerate(domain_names):
        suffix = "|" if idx < len(domain_names) - 1 else ""
        header_cells.append(f"\\multicolumn{{{num_rates}}}{{c{suffix}}}{{{name}}}")
    lines.append("\\multirow{2}{*}{Method} & " + " & ".join(header_cells) + " \\\\")
    rate_cells = []
    for _ in domain_names:
        rate_cells.extend([f"{rate}\\%" for rate in attack_rates])
    lines.append("& " + " & ".join(rate_cells) + " \\\\")
    lines.append("\\hline")
    return lines


def build_rows(
    records: Sequence[RunRecord],
    methods: Mapping[str, Mapping[str, List[Any]]],
    filters: Mapping[str, List[Any]],
    domain_ids: Sequence[int],
    attack_rates: Sequence[int],
    mean_key: str,
    std_key: str,
    select_mode: str,
    select_metric: str,
    precision: int,
    verbose: bool,
) -> List[Tuple[str, List[str]]]:
    rows: List[Tuple[str, List[str]]] = []
    for method_name, method_filters in methods.items():
        merged = dict(filters)
        merged.update(method_filters)
        matched = [
            record
            for record in records
            if mean_key in record.summary and record_matches_filters(record, merged)
        ]
        if verbose:
            print(f"[info] {method_name}: {len(matched)} runs matched", file=sys.stderr)
        grouped: Dict[Tuple[int, int], List[RunRecord]] = {}
        for record in matched:
            dom_val = get_value(record, "test_envs", "test_env")
            if isinstance(dom_val, list):
                dom_val = dom_val[0] if dom_val else None
            dom_id = to_int(dom_val)
            rate = to_int(get_value(record, "attack_rate"))
            if dom_id is None or rate is None:
                continue
            grouped.setdefault((dom_id, rate), []).append(record)

        cells: List[str] = []
        for dom_id in domain_ids:
            for rate in attack_rates:
                runs = grouped.get((dom_id, rate), [])
                if not runs:
                    if verbose:
                        print(f"[warn] Missing dom {dom_id} rate {rate} for {method_name}", file=sys.stderr)
                    cells.append("--")
                    continue
                chosen = select_run(runs, select_mode, select_metric)
                mean = chosen.summary.get(mean_key)
                std = chosen.summary.get(std_key)
                cells.append(format_cell(mean, std, precision))
        rows.append((method_name, cells))
    return rows


def parse_methods_file(path: Path) -> Dict[str, Mapping[str, List[Any]]]:
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    else:
        data = load_yaml_file(path)
    if isinstance(data, Mapping) and "methods" in data:
        data = data["methods"]
    if isinstance(data, Mapping):
        methods = {}
        for name, filt in data.items():
            if not isinstance(filt, Mapping):
                continue
            cleaned: Dict[str, List[Any]] = {}
            for key, value in filt.items():
                values = value if isinstance(value, list) else [value]
                cleaned[key] = [coerce_value(v) if isinstance(v, str) else v for v in values]
            methods[name] = cleaned
        return methods
    raise ValueError("Methods file must be a mapping of name -> filters.")


def parse_csv_ints(text: str) -> List[int]:
    values = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        values.append(int(chunk))
    return values


def parse_csv_strings(text: str) -> List[str]:
    return [chunk.strip() for chunk in text.split(",") if chunk.strip()]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build LaTeX rows/tables from W&B runs of unsupervise_adapt.py."
    )
    parser.add_argument(
        "--sweep-id",
        required=True,
        help="W&B sweep id or entity/project/sweep.",
    )
    parser.add_argument("--sweep-config", type=Path, help="Sweep YAML to restrict allowed values.")
    parser.add_argument("--entity", default="bigslav", help="W&B entity for --sweep-id.")
    parser.add_argument("--project", default="safer_loss", help="W&B project for --sweep-id.")

    parser.add_argument("--dataset", help="Dataset name (e.g., PACS).")
    parser.add_argument(
        "--datasets",
        help="Comma-separated datasets to emit rows for (e.g., PACS,VLCS,office-home).",
    )
    parser.add_argument("--attack-rates", help="Comma-separated attack rates (e.g., 0,50,100).")
    parser.add_argument("--domain-ids", help="Comma-separated domain indices (e.g., 0,1,2,3).")

    parser.add_argument("--filter", action="append", default=[], help="Filter runs: key=value (repeatable).")
    parser.add_argument("--methods-file", type=Path, help="YAML/JSON mapping of method name -> filters.")
    parser.add_argument("--method-name", help="Name for the output row.")
    parser.add_argument(
        "--algorithms",
        help="Comma-separated adapt_alg list to print in order (overrides --method-name).",
    )
    parser.add_argument(
        "--all-algorithms",
        action="store_true",
        help="Print all algorithms in the default order.",
    )

    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for the mean metric.")
    parser.add_argument("--std-key", default="acc_std", help="Summary key for the std metric.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric used to select runs (defaults to mean key).")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision in table cells.")

    parser.add_argument("--verbose", action="store_true", help="Print matching diagnostics to stderr.")

    args = parser.parse_args()

    sweep_filters: Dict[str, List[Any]] = {}
    if args.sweep_config:
        sweep_filters = load_sweep_filters(args.sweep_config)

    filters = parse_filter_args(args.filter)
    for key, values in sweep_filters.items():
        filters.setdefault(key, []).extend(values)

    records = load_runs_from_wandb_api(args.sweep_id, args.entity, args.project)

    if args.verbose:
        print(f"[info] Loaded {len(records)} runs", file=sys.stderr)

    if args.attack_rates:
        attack_rates = parse_csv_ints(args.attack_rates)
    else:
        attack_rates = [0, 50, 100]

    methods: Dict[str, Mapping[str, List[Any]]]
    if args.methods_file:
        methods = parse_methods_file(args.methods_file)
    elif args.algorithms or args.all_algorithms:
        if args.algorithms:
            alg_list = parse_csv_strings(args.algorithms)
        else:
            alg_list = list(ADAPT_ALG_ORDER)
        if not alg_list:
            raise ValueError("No algorithms provided for --algorithms/--all-algorithms.")
        methods = {name: {"adapt_alg": [name]} for name in alg_list}
    else:
        method_name = args.method_name
        if not method_name:
            if "adapt_alg" in filters and len(filters["adapt_alg"]) == 1:
                method_name = str(filters["adapt_alg"][0])
            else:
                method_name = "Method"
        methods = {method_name: {}}

    datasets: List[str] = []
    if args.datasets:
        datasets = parse_csv_strings(args.datasets)
    elif args.dataset:
        datasets = [args.dataset]
    elif "dataset" in filters and len(filters["dataset"]) == 1:
        datasets = [str(filters["dataset"][0])]

    if not datasets:
        datasets = ["PACS", "VLCS", "office-home"]

    select_metric = args.select_metric or args.mean_key
    printed_any = False
    for dataset in datasets:
        print(dataset)
        dataset_filters = dict(filters)
        dataset_filters["dataset"] = [dataset]

        if args.domain_ids:
            domain_ids = parse_csv_ints(args.domain_ids)
        else:
            domain_ids = [0, 1, 2, 3]

        rows = build_rows(
            records=records,
            methods=methods,
            filters=dataset_filters,
            domain_ids=domain_ids,
            attack_rates=attack_rates,
            mean_key=args.mean_key,
            std_key=args.std_key,
            select_mode=args.select,
            select_metric=select_metric,
            precision=args.precision,
            verbose=args.verbose,
        )
        # if printed_any:
        #     print()
        for name, cells in rows:
            print(f"{name} & " + " & ".join(cells) + " \\\\")
        printed_any = True

    if not printed_any:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
