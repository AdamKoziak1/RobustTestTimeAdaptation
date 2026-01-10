#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple


@dataclass
class RunResult:
    label: str
    acc_mean: Optional[float]
    acc_std: Optional[float]
    cost_time: Optional[float]
    wall_time: float
    returncode: int


ACC_RE = re.compile(r"Accuracy:\s*([0-9.]+)")
STD_RE = re.compile(r"Accuracy std:\s*([0-9.]+)")
COST_RE = re.compile(r"Cost time:\s*([0-9.]+)")


def _parse_metrics(output: str) -> Tuple[Optional[float], Optional[float], Optional[float]]:
    acc = None
    std = None
    cost = None
    for line in output.splitlines():
        m = ACC_RE.search(line)
        if m:
            acc = float(m.group(1))
        m = STD_RE.search(line)
        if m:
            std = float(m.group(1))
        m = COST_RE.search(line)
        if m:
            cost = float(m.group(1))
    return acc, std, cost


def _run(
    label: str,
    cmd: List[str],
    env: dict,
) -> RunResult:
    start = time.perf_counter()
    proc = subprocess.run(cmd, capture_output=True, text=True, env=env)
    wall = time.perf_counter() - start
    output = (proc.stdout or "") + "\n" + (proc.stderr or "")
    acc, std, cost = _parse_metrics(output)
    return RunResult(
        label=label,
        acc_mean=acc,
        acc_std=std,
        cost_time=cost,
        wall_time=wall,
        returncode=proc.returncode,
    )


def _format_value(value: Optional[float]) -> str:
    if value is None:
        return "--"
    return f"{value:.2f}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare precomputed vs on-the-fly Linf attack runtime on PACS."
    )
    parser.add_argument("--dataset", default="PACS", help="Dataset to evaluate.")
    parser.add_argument("--test-env", type=int, default=0, help="Target domain index.")
    parser.add_argument("--adapt-alg", default="TTA3", help="Adaptation algorithm.")
    parser.add_argument("--net", default="resnet18", help="Backbone name.")
    parser.add_argument("--attack-id", default="linf_eps-8.0_steps-20", help="Attack id suffix.")
    parser.add_argument("--attack-preset", default="linf8", help="Named attack preset for on-the-fly attacks.")
    parser.add_argument("--attack-rate", type=int, default=100, help="Attack percentage.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for evaluation.")
    parser.add_argument("--seeds", default="0", help="Comma-separated seed list for unsupervise_adapt.py.")
    parser.add_argument(
        "--extra",
        nargs=argparse.REMAINDER,
        help="Additional args forwarded to unsupervise_adapt.py.",
    )

    args = parser.parse_args()

    base_cmd = [
        sys.executable,
        "unsupervise_adapt.py",
        "--dataset",
        args.dataset,
        "--test_envs",
        str(args.test_env),
        "--adapt_alg",
        args.adapt_alg,
        "--net",
        args.net,
        "--attack",
        args.attack_id,
        "--attack_preset",
        args.attack_preset,
        "--attack_rate",
        str(args.attack_rate),
        "--batch_size",
        str(args.batch_size),
        "--seeds",
        args.seeds,
    ]
    if args.extra:
        base_cmd.extend(args.extra)

    env = os.environ.copy()
    env.setdefault("WANDB_MODE", "disabled")

    pre_cmd = base_cmd + ["--attack_source", "precomputed"]
    on_cmd = base_cmd + ["--attack_source", "on_the_fly"]

    pre = _run("precomputed", pre_cmd, env)
    on = _run("on_the_fly", on_cmd, env)

    print("Attack runtime comparison")
    print(f"dataset={args.dataset} test_env={args.test_env} attack_rate={args.attack_rate}%")
    print(f"attack_id={args.attack_id} attack_preset={args.attack_preset}")
    print("")
    for res in (pre, on):
        print(
            f"{res.label:12s} acc={_format_value(res.acc_mean)} "
            f"std={_format_value(res.acc_std)} cost_time={_format_value(res.cost_time)}s "
            f"wall_time={res.wall_time:.2f}s"
        )
    if pre.wall_time > 0 and on.wall_time > 0:
        ratio = on.wall_time / pre.wall_time
        print(f"wall_time ratio (on_the_fly / precomputed): {ratio:.2f}x")
    if pre.cost_time and on.cost_time:
        ratio = on.cost_time / pre.cost_time
        print(f"cost_time ratio (on_the_fly / precomputed): {ratio:.2f}x")

    if pre.returncode != 0 or on.returncode != 0:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
