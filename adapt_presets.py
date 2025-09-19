from __future__ import annotations

import sys
from typing import Dict, Iterable, Mapping

ADAPT_ALG_PRESETS: Mapping[str, Dict[str, object]] = {
    "Tent": {
        "lr": 1e-3,
        "steps": 3,
    },
    "PL": {
        "lr": 1e-5,
        "steps": 3,
    },
    "SHOT-IM": {
        "lr": 1e-5,
        "steps": 1,
    },
    "T3A": {
        "filter_K": 100,
    },
    "TSD": {
        "lr": 1e-4,
        "filter_K": 100,
        "steps": 10,
    },
    # "TTA3": {
    #     "lr": 1e-3,
    #     "steps": 3,
    #     "lam_flat": 1e-3,
    #     "lam_adv": 1e-3,
    #     "lam_cr": 1e-1,
    #     "lam_pl": 50,
    #     "cr_start": 0,
    #     "update_param": "tent",
    #     "use_mi": "em",
    # },
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
