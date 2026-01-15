#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

try:
    import yaml  # type: ignore
except Exception:
    yaml = None

import wandb  # type: ignore

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
import wandb_table as wt  # noqa: E402

try:
    from adapt_presets import ADAPT_ALG_ORDER
except Exception:
    ADAPT_ALG_ORDER = []

TERMINAL_STATES = {"finished", "failed", "crashed", "killed"}


def load_yaml_file(path: Path) -> Mapping[str, Any]:
    if yaml is None:
        raise RuntimeError("PyYAML is required to read YAML files.")
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, Mapping):
        raise ValueError(f"Unexpected YAML format: {path}")
    return data


def _param_values(params: Mapping[str, Any], key: str) -> List[Any]:
    spec = params.get(key)
    if not isinstance(spec, Mapping):
        return []
    if "values" in spec:
        values = spec["values"]
        return list(values) if isinstance(values, list) else [values]
    if "value" in spec:
        return [spec["value"]]
    return []


def _coerce_list(values: Iterable[Any]) -> List[Any]:
    out = []
    for value in values:
        if isinstance(value, str):
            out.append(wt.coerce_value(value))
        else:
            out.append(value)
    return out


def _fetch_sweep_runs(
    api: wandb.Api,
    sweep_path: str,
    sweep_id: str,
    entity: str,
    project: str,
) -> List[wandb.apis.public.Run]:
    try:
        sweep = api.sweep(sweep_path)
        runs = list(sweep.runs)
        if runs:
            return runs
    except Exception:
        runs = []
    try:
        runs = list(api.runs(f"{entity}/{project}", filters={"sweep": sweep_id}))
    except Exception:
        runs = []
    return runs


def _wait_for_sweep(
    api: wandb.Api,
    sweep_path: str,
    sweep_id: str,
    entity: str,
    project: str,
    poll_interval: int,
) -> None:
    last_state = None
    while True:
        runs = _fetch_sweep_runs(api, sweep_path, sweep_id, entity, project)
        if runs:
            states = [run.state for run in runs]
            done = sum(state in TERMINAL_STATES for state in states)
            if done == len(states):
                return
            state_line = f"[wait] {done}/{len(states)} runs finished"
        else:
            state_line = "[wait] no runs yet"
        if state_line != last_state:
            print(state_line, file=sys.stderr)
            last_state = state_line
        time.sleep(poll_interval)


def _group_runs(
    records: Sequence[wt.RunRecord],
    param_names: Sequence[str],
    allow_vary: Sequence[str],
) -> Dict[Tuple[Tuple[str, Any], ...], List[wt.RunRecord]]:
    allow = set(allow_vary)
    grouped: Dict[Tuple[Tuple[str, Any], ...], List[wt.RunRecord]] = {}
    for record in records:
        key_items: List[Tuple[str, Any]] = []
        for name in param_names:
            if name in allow:
                continue
            value = record.config.get(name)
            if value is None:
                continue
            key_items.append((name, wt.normalize_value(value)))
        key = tuple(sorted(key_items))
        grouped.setdefault(key, []).append(record)
    if not grouped:
        grouped[tuple()] = list(records)
    return grouped


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a W&B sweep from YAML, wait for completion, then emit tables."
    )
    parser.add_argument("--sweep-config", type=Path, required=True, help="Sweep YAML config path.")
    parser.add_argument("--entity", help="W&B entity (overrides YAML).")
    parser.add_argument("--project", help="W&B project (overrides YAML).")
    parser.add_argument("--no-wait", action="store_true", help="Skip waiting for sweep completion.")
    parser.add_argument("--poll-interval", type=int, default=60, help="Seconds between sweep status checks.")

    parser.add_argument("--algorithms", help="Comma-separated adapt_alg list to print in order.")
    parser.add_argument("--all-algorithms", action="store_true", help="Print all algorithms in default order.")
    parser.add_argument("--datasets", help="Comma-separated datasets to emit rows for.")
    parser.add_argument("--attack-rates", help="Comma-separated attack rates (e.g., 0,50,100).")
    parser.add_argument("--domain-ids", help="Comma-separated domain indices (e.g., 0,1,2,3).")

    parser.add_argument("--mean-key", default="acc_mean", help="Summary key for the mean metric.")
    parser.add_argument("--std-key", default="acc_std", help="Summary key for the std metric.")
    parser.add_argument("--select", choices=["best", "latest", "first"], default="best")
    parser.add_argument("--select-metric", help="Metric used to select runs (defaults to mean key).")
    parser.add_argument("--precision", type=int, default=2, help="Decimal precision in table cells.")
    parser.add_argument("--verbose", action="store_true", help="Print matching diagnostics to stderr.")

    args = parser.parse_args()

    sweep_cfg = load_yaml_file(args.sweep_config)
    project = args.project or sweep_cfg.get("project")
    if not project:
        raise ValueError("Project must be provided via --project or sweep config.")

    api = wandb.Api()
    entity = args.entity or sweep_cfg.get("entity") or os.environ.get("WANDB_ENTITY") or api.default_entity
    if not entity:
        raise ValueError("Entity must be provided via --entity, sweep config, or WANDB_ENTITY.")

    sweep_id = wandb.sweep(sweep_cfg, project=project, entity=entity)
    sweep_path = f"{entity}/{project}/{sweep_id}"
    print(f"Sweep created: {sweep_path}")
    print(f"Start with: wandb agent {sweep_path}")

    if not args.no_wait:
        _wait_for_sweep(api, sweep_path, sweep_id, entity, project, args.poll_interval)

    records = wt.load_runs_from_wandb_api(sweep_path, entity, project)
    if args.verbose:
        print(f"[info] Loaded {len(records)} runs", file=sys.stderr)

    params = sweep_cfg.get("parameters", {})
    param_names = list(params.keys()) if isinstance(params, Mapping) else []
    allow_vary = ["adapt_alg", "dataset", "test_env", "test_envs", "attack_rate"]
    grouped = _group_runs(records, param_names, allow_vary)

    select_metric = args.select_metric or args.mean_key
    for key_items, group_records in grouped.items():
        if key_items:
            config_line = ", ".join(f"{k}={v}" for k, v in key_items)
            print(f"Config: {config_line}")

        if args.algorithms:
            algorithms = wt.parse_csv_strings(args.algorithms)
        elif args.all_algorithms:
            algorithms = list(ADAPT_ALG_ORDER)
        else:
            algorithms = _coerce_list(_param_values(params, "adapt_alg"))
            if not algorithms:
                algorithms = sorted({str(wt.get_value(r, "adapt_alg")) for r in group_records if wt.get_value(r, "adapt_alg")})
        if not algorithms:
            raise ValueError("No algorithms available for table output.")
        methods = {name: {"adapt_alg": [name]} for name in algorithms}

        datasets = []
        if args.datasets:
            datasets = wt.parse_csv_strings(args.datasets)
        else:
            datasets = _coerce_list(_param_values(params, "dataset"))
            if not datasets:
                datasets = sorted({str(wt.get_value(r, "dataset")) for r in group_records if wt.get_value(r, "dataset")})

        if args.attack_rates:
            attack_rates = wt.parse_csv_ints(args.attack_rates)
        else:
            attack_rates = _coerce_list(_param_values(params, "attack_rate"))
            if not attack_rates:
                attack_rates = [0, 50, 100]

        if args.domain_ids:
            domain_ids = wt.parse_csv_ints(args.domain_ids)
        else:
            domain_ids = _coerce_list(_param_values(params, "test_envs"))
            if not domain_ids:
                doms = []
                for record in group_records:
                    val = wt.get_value(record, "test_envs", "test_env")
                    if isinstance(val, list):
                        val = val[0] if val else None
                    if val is not None:
                        doms.append(val)
                domain_ids = sorted({int(v) for v in doms})

        group_filters = {k: [v] for k, v in key_items}
        for dataset in datasets:
            print(dataset)
            dataset_filters = dict(group_filters)
            dataset_filters["dataset"] = [dataset]
            rows = wt.build_rows(
                records=group_records,
                methods=methods,
                filters=dataset_filters,
                domain_ids=[int(v) for v in domain_ids],
                attack_rates=[int(v) for v in attack_rates],
                mean_key=args.mean_key,
                std_key=args.std_key,
                select_mode=args.select,
                select_metric=select_metric,
                precision=args.precision,
                verbose=args.verbose,
            )
            for name, cells in rows:
                print(f"{name} & " + " & ".join(cells) + " \\\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
