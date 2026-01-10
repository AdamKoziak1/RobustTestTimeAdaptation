from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Dict, Optional


@dataclass(frozen=True)
class AttackConfig:
    norm: str
    eps: float
    alpha: float
    steps: int
    fft_rho: float = 1.0
    fft_alpha: float = 1.0
    attack_id: str = ""


def _fmt_float(val: float) -> str:
    val = float(val)
    if abs(val - round(val)) < 1e-8:
        return f"{val:.1f}"
    return f"{val:g}"


def format_attack_id(
    norm: str,
    eps: float,
    steps: int,
    fft_rho: float = 1.0,
    fft_alpha: float = 1.0,
) -> str:
    base = f"{norm}_eps-{_fmt_float(eps)}_steps-{int(steps)}"
    if fft_rho < 1.0:
        base = f"{base}_rho-{_fmt_float(fft_rho)}_a-{_fmt_float(fft_alpha)}"
    return base


def parse_attack_id(attack_id: str) -> Optional[AttackConfig]:
    if not attack_id or attack_id == "clean":
        return None
    match = re.search(r"(linf|l2)_eps-([0-9.]+)_steps-([0-9]+)", attack_id)
    if not match:
        return None
    norm = match.group(1)
    eps = float(match.group(2))
    steps = int(match.group(3))
    fft_rho = 1.0
    fft_alpha = 1.0
    match_fft = re.search(r"_rho-([0-9.]+)_a-([0-9.]+)", attack_id)
    if match_fft:
        fft_rho = float(match_fft.group(1))
        fft_alpha = float(match_fft.group(2))
    else:
        match_fft = re.search(r"_fft-spatial_k-([0-9.]+)_a-([0-9.]+)", attack_id)
        if match_fft:
            fft_rho = float(match_fft.group(1))
            fft_alpha = float(match_fft.group(2))
    alpha = eps * 0.25
    return AttackConfig(
        norm=norm,
        eps=eps,
        alpha=alpha,
        steps=steps,
        fft_rho=fft_rho,
        fft_alpha=fft_alpha,
        attack_id=format_attack_id(norm, eps, steps, fft_rho, fft_alpha),
    )


ATTACK_PRESETS: Dict[str, Dict[str, object]] = {
    "linf8": {
        "norm": "linf",
        "eps": 8.0,
        "alpha": 2.0,
        "steps": 20,
        "fft_rho": 1.0,
        "fft_alpha": 1.0,
    },
    "linf8_fft0.3": {
        "norm": "linf",
        "eps": 8.0,
        "alpha": 2.0,
        "steps": 20,
        "fft_rho": 0.3,
        "fft_alpha": 1.0,
    },
    "l2_112": {
        "norm": "l2",
        "eps": 112.0,
        "alpha": 2.0,
        "steps": 100,
        "fft_rho": 1.0,
        "fft_alpha": 1.0,
    },
}

DEFAULT_ATTACK_PRESET = "linf8"


def resolve_attack_config(
    *,
    preset_name: Optional[str] = None,
    attack_id: Optional[str] = None,
    norm: Optional[str] = None,
    eps: Optional[float] = None,
    steps: Optional[int] = None,
    alpha: Optional[float] = None,
    fft_rho: Optional[float] = None,
    fft_alpha: Optional[float] = None,
    default_alpha_factor: float = 0.25,
) -> AttackConfig:
    if preset_name:
        preset = ATTACK_PRESETS.get(preset_name)
        if preset is None:
            raise ValueError(f"Unknown attack preset: {preset_name}")
        norm = norm or str(preset.get("norm", "linf"))
        eps = eps if eps is not None else float(preset.get("eps", 8.0))
        steps = steps if steps is not None else int(preset.get("steps", 20))
        alpha = alpha if alpha is not None else float(preset.get("alpha", 2.0))
        fft_rho = fft_rho if fft_rho is not None else float(preset.get("fft_rho", 1.0))
        fft_alpha = fft_alpha if fft_alpha is not None else float(preset.get("fft_alpha", 1.0))

    parsed = parse_attack_id(attack_id) if attack_id else None
    if parsed:
        norm = norm or parsed.norm
        eps = eps if eps is not None else parsed.eps
        steps = steps if steps is not None else parsed.steps
        fft_rho = fft_rho if fft_rho is not None else parsed.fft_rho
        fft_alpha = fft_alpha if fft_alpha is not None else parsed.fft_alpha

    norm = norm or "linf"
    eps = float(eps if eps is not None else 8.0)
    steps = int(steps if steps is not None else 20)
    fft_rho = float(fft_rho if fft_rho is not None else 1.0)
    fft_alpha = float(fft_alpha if fft_alpha is not None else 1.0)
    if alpha is None:
        alpha = eps * float(default_alpha_factor)
    alpha = float(alpha)

    attack_id = format_attack_id(norm, eps, steps, fft_rho, fft_alpha)
    return AttackConfig(
        norm=norm,
        eps=eps,
        alpha=alpha,
        steps=steps,
        fft_rho=fft_rho,
        fft_alpha=fft_alpha,
        attack_id=attack_id,
    )
