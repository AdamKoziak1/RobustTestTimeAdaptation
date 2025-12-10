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
    "SAFER": {
        "lr": 1e-4,
        "s_num_views": 4,
        "s_aug_prob": 0.7,
        "s_aug_max_ops": 4,
        "s_js_weight": 1.0,
        "s_cc_weight": 1.0,
        "s_cc_offdiag": 0.01,
        "s_include_original": 1,
        "s_sup_type": "none",
        "s_sup_weight": 0.0,
        "s_sup_view_pool": "mean",
        "s_sup_pl_weighted": 0,
        "s_sup_conf_scale": 1,
        "s_js_mode": "pooled",
        "s_js_view_pool": "matching",
        "s_view_weighting": 1,
        "s_cc_mode": "pairwise",
        "s_cc_view_pool": "matching",
        "s_tta_loss": "none",
        "s_tta_weight": 0.0,
        "s_tta_target": "views",
        "s_tta_view_pool": "matching",
        "update_param": "tent",
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
        if hyperparam in explicit:
            continue
        setattr(args, hyperparam, value)
        applied[hyperparam] = value

    return applied
