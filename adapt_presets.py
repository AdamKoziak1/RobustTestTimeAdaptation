from __future__ import annotations

import sys
from typing import Dict, Iterable, Mapping

ADAPT_ALG_ORDER = [
    "SAFER",
    "TeSLA",
    "TSD",
    "Tent",
    "T3A",
    "PL",
    "ERM",
    # "BN",
    # "PLC",
    # "SHOT-IM",
    # "TTA3",
    # "AMTDC",
]

ADAPT_ALG_PRESETS: Mapping[str, Dict[str, object]] = {
    "Tent": {
        "lr": 1e-3,
    },
    "PL": {
        "lr": 1e-4,
    },
    "SHOT-IM": {
        "lr": 1e-4,
    },
    "T3A": {
        "filter_K": 100,
    },
    "TSD": {
        "lr": 1e-4,
        "filter_K": 100,
        "update_param": "all"
    },
    "AMTDC": {
        "lr": 1e-4,
        "mt_alpha": 0.02,
        "mt_gamma": 0.99,
        "mt_gamma_y": 0.5,
        "mt_kl_weight": 0.1,
        "mt_ce_weight": 1.0,
        "mt_ent_weight": 0.0,
        "mt_mixup_weight": 0.1,
        "mt_mixup_beta": 0.5,
        "mt_use_teacher_pred": 1,
        "update_param": "tent",
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
        # if hyperparam in explicit:
        #     continue
        setattr(args, hyperparam, value)
        applied[hyperparam] = value

    return applied
