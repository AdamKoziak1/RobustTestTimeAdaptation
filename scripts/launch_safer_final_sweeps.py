#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import subprocess
from pathlib import Path


def load_queue(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = [row for row in reader]
    if not rows:
        raise ValueError(f"No sweep entries found in {path}")
    return rows


def run_wandb_sweep(config_path: str) -> str:
    proc = subprocess.run(
        ["wandb", "sweep", config_path],
        check=False,
        text=True,
        capture_output=True,
    )
    output = (proc.stdout or "") + (proc.stderr or "")
    print(output, end="")
    if proc.returncode != 0:
        raise RuntimeError(f"wandb sweep failed for {config_path}")
    match = re.findall(r"wandb agent ([^\s]+)", output)
    if not match:
        raise RuntimeError(f"Could not parse sweep path from output for {config_path}")
    return match[-1]


def create_map(queue_path: Path, map_path: Path) -> None:
    rows = load_queue(queue_path)
    with map_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["label", "config", "description", "sweep_path"])
        for row in rows:
            label = row["label"]
            config = row["config"]
            desc = row.get("description", "")
            print(f"== [{label}] {config}")
            sweep_path = run_wandb_sweep(config)
            writer.writerow([label, config, desc, sweep_path])
            print(f"-> {sweep_path}")
    print(f"Saved: {map_path}")


def run_agents(map_path: Path) -> None:
    with map_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No sweep entries found in {map_path}")

    for row in rows:
        label = row["label"]
        config = row["config"]
        sweep_path = row["sweep_path"]
        print(f"== Running [{label}] {sweep_path} ({config})")
        subprocess.run(["wandb", "agent", sweep_path], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Create and/or run the safer_final sweep queue.")
    parser.add_argument(
        "mode",
        choices=["create", "run", "create-and-run"],
        default="create",
        nargs="?",
    )
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("sweeps/safer_final_queue.tsv"),
        help="Queue file with label/config/description columns.",
    )
    parser.add_argument(
        "--map",
        type=Path,
        default=Path("sweeps/safer_final_sweep_map.tsv"),
        help="Output/read mapping file (label -> sweep path).",
    )
    args = parser.parse_args()

    if args.mode in {"create", "create-and-run"}:
        create_map(args.queue, args.map)
    if args.mode in {"run", "create-and-run"}:
        run_agents(args.map)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
