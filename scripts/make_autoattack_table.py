#!/usr/bin/env python3
"""Render the E5 AutoAttack-transfer table from measured TTA accuracies.

Companion to the compute_overhead csv->tex pattern (scripts/bench_safer_overhead.py):
the measured numbers live in a tidy CSV (sweeps/autoattack_transfer_dom0.csv) produced
by the four TTA runs in sweeps/supp_autoattack_transfer_tent_pacs_dom0.yaml (Tent vs
Tent+SAFER x PGD-transfer vs AutoAttack-transfer, PACS:Art, attack_rate=100, seeds
0,1,2), and this script formats them into the LaTeX table body
(sweeps/autoattack_transfer_rows.tex) inlined by Table~\\ref{tab:supp-autoattack}.

Input CSV columns: method,attack,acc_mean,acc_std
  method in {Tent, Tent+SAFER}; attack in {pgd, autoattack}

Usage:
  python scripts/make_autoattack_table.py \
      --csv sweeps/autoattack_transfer_dom0.csv \
      --output-tex sweeps/autoattack_transfer_rows.tex
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, Tuple

METHOD_ORDER = ["Tent", "Tent+SAFER"]
ATTACK_ORDER = ["pgd", "autoattack"]


def load(csv_path: Path) -> Dict[Tuple[str, str], Tuple[float, float]]:
    out: Dict[Tuple[str, str], Tuple[float, float]] = {}
    with csv_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["method"].strip(), row["attack"].strip().lower())
            out[key] = (float(row["acc_mean"]), float(row["acc_std"]))
    return out


def fmt(cell: Tuple[float, float]) -> str:
    mean, std = cell
    return f"{mean:.2f} $\\pm$ {std:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=Path, default=Path("sweeps/autoattack_transfer_dom0.csv"))
    ap.add_argument("--output-tex", type=Path,
                    default=Path("sweeps/autoattack_transfer_rows.tex"))
    args = ap.parse_args()

    data = load(args.csv)
    lines = []
    for method in METHOD_ORDER:
        cells = [fmt(data[(method, atk)]) for atk in ATTACK_ORDER]
        lines.append(f"\\texttt{{{method}}} & " + " & ".join(cells) + r" \\")

    args.output_tex.parent.mkdir(parents=True, exist_ok=True)
    args.output_tex.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {args.output_tex}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
