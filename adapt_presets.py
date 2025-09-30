from __future__ import annotations

import sys
from typing import Dict, Iterable, Mapping

ADAPT_ALG_PRESETS: Mapping[str, Dict[str, object]] = {
    "Tent": {
        #"lr": 1e-3,
        "lr": 1e-4,
    },
    "PL": {
        #"lr": 1e-5,
        "lr": 1e-4,
    },
    "SHOT-IM": {
        "lr": 1e-5,
    },
    "T3A": {
        "filter_K": 100,
    },
    "TSD": {
        "lr": 1e-4,
        "filter_K": 100,
    },
    "TTA3": {
        "lr": 1e-3,
        "lam_flat": 0,
        "lam_adv": 0,
        "lam_cr": 0,
        "lam_pl": 0,
        "cr_start": 0,
        "update_param": "tent",
        "use_mi": "em",
        "ema": 0.9,
        "lam_reg": 0.001,
        "reg_type": "klprob",
        "x_lr": 0.0001,
        "x_steps": 3,
    },
}


def _cli_overrides(tokens: Iterable[str]) -> Dict[str, None]:
    overrides = {}
    for token in tokens:
        if not token.startswith("--"):
            continue
        key = token[2:]
        if "=" in key:
            key = key.split("=", 1)[0]
        key = key.replace("-", "_")
        overrides[key] = None
    return overrides


def apply_adapt_preset(args, disable: bool = False, argv: Iterable[str] | None = None):
    if disable:
        return {}

    preset = ADAPT_ALG_PRESETS.get(getattr(args, "adapt_alg", None))
    if not preset:
        return {}

    cli_tokens = list(argv) if argv is not None else sys.argv[1:]
    explicit = _cli_overrides(cli_tokens)

    applied = {}
    for hyperparam, value in preset.items():
        if hyperparam in explicit:
            continue
        setattr(args, hyperparam, value)
        applied[hyperparam] = value

    return applied
