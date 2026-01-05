"""
Randomized augmentation pipelines for SAFER test-time adaptation.

This module defines a lightweight sampler over a library of atomic
augmentations. Each augmentation provides a parameter sampler and an
application callable that operates directly on tensor images in
range [0, 1]. The augmenter composes a random subset of operations
per view and applies them sequentially.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import torch
import torchvision.transforms.functional as TF


Tensor = torch.Tensor


@dataclass
class _OpSpec:
    """Container for augmentation metadata."""

    sample_params: Callable[[random.Random], Dict[str, float]]
    apply: Callable[[Tensor, Dict[str, float]], Tensor]


def _clamp_img(x: Tensor) -> Tensor:
    return x.clamp_(0.0, 1.0)


def _sample_uniform(rng: random.Random, low: float, high: float) -> float:
    return float(rng.uniform(low, high))


def _fft_low_pass(x: Tensor, keep_ratio: float) -> Tensor:
    if keep_ratio >= 1.0:
        return x.clone()
    c, h, w = x.shape
    freq = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

    keep_h = max(1, int(math.ceil(h * keep_ratio)))
    keep_w = max(1, int(math.ceil(w * keep_ratio)))
    start_h = (h - keep_h) // 2
    end_h = start_h + keep_h
    start_w = (w - keep_w) // 2
    end_w = start_w + keep_w

    mask = torch.zeros_like(freq_shifted.real)
    mask[..., start_h:end_h, start_w:end_w] = 1.0
    filtered_shifted = freq_shifted * mask
    filtered = torch.fft.ifftshift(filtered_shifted, dim=(-2, -1))
    return torch.fft.ifft2(filtered, dim=(-2, -1), norm="ortho").real

def _equalize(x: Tensor) -> Tensor:
    c, h, w = x.shape
    out = torch.empty_like(x)
    bins = 256
    for ch in range(c):
        channel = x[ch]
        hist = torch.histc(channel, bins=bins, min=0.0, max=1.0)
        cdf = hist.cumsum(0)
        non_zero = cdf > 0
        if not bool(non_zero.any()):
            out[ch] = channel
            continue
        cdf_min = cdf[non_zero].min()
        denom = (cdf[-1] - cdf_min).clamp_min(1e-6)
        cdf_norm = (cdf - cdf_min) / denom
        idx = torch.clamp((channel * (bins - 1)).round().long(), 0, bins - 1)
        eq = torch.gather(cdf_norm, 0, idx.view(-1))
        out[ch] = eq.view(h, w)
    return out.clamp_(0.0, 1.0)


def _posterize(x: Tensor, bits: int) -> Tensor:
    levels = max(2, 2 ** bits)
    return torch.floor(x * (levels - 1)) / (levels - 1)


def _solarize(x: Tensor, threshold: float) -> Tensor:
    return torch.where(x < threshold, x, 1.0 - x)


def _auto_contrast(x: Tensor) -> Tensor:
    cmins = x.amin(dim=(1, 2), keepdim=True)
    cmaxs = x.amax(dim=(1, 2), keepdim=True)
    denom = (cmaxs - cmins).clamp_min(1e-6)
    return ((x - cmins) / denom).clamp_(0.0, 1.0)


def _gaussian_noise(x: Tensor, std: float) -> Tensor:
    noise = torch.randn_like(x) * std
    return (x + noise).clamp_(0.0, 1.0)


def _make_affine(
    x: Tensor,
    angle: float = 0.0,
    translate: Tuple[float, float] = (0.0, 0.0),
    shear: Tuple[float, float] = (0.0, 0.0),
) -> Tensor:
    return TF.affine(x, angle=angle, translate=translate, scale=1.0, shear=shear)


def _build_registry() -> Dict[str, _OpSpec]:
    return {
        "gaussian_blur": _OpSpec(
            sample_params=lambda rng: {
                "kernel_size": int(rng.choice([7,9])),
                "sigma": _sample_uniform(rng, 0.1, 2.5),
            },
            apply=lambda x, p: TF.gaussian_blur(
                x,
                kernel_size=p["kernel_size"],
                sigma=(p["sigma"], p["sigma"]),
            ),
        ),
        "gaussian_noise": _OpSpec(
            sample_params=lambda rng: {"std": _sample_uniform(rng, 0.01, 0.1)},
            apply=lambda x, p: _gaussian_noise(x, p["std"]),
        ),
        "fft_low_pass": _OpSpec(
            sample_params=lambda rng: {"keep_ratio": _sample_uniform(rng, 0.2, 0.7)},
            apply=lambda x, p: _fft_low_pass(x, p["keep_ratio"]),
        ),
        "equalize": _OpSpec(
            sample_params=lambda _: {},
            apply=lambda x, _: _equalize(x),
        ),
        "invert": _OpSpec(
            sample_params=lambda _: {},
            apply=lambda x, _: (1.0 - x).clamp_(0.0, 1.0),
        ),
        "solarize": _OpSpec(
            sample_params=lambda rng: {"threshold": _sample_uniform(rng, 0.1, 0.9)},
            apply=lambda x, p: _solarize(x, p["threshold"]),
        ),
        "posterize": _OpSpec(
            sample_params=lambda rng: {"bits": int(rng.choice([2, 3, 4, 5]))},
            apply=lambda x, p: _posterize(x, p["bits"]),
        ),
        "contrast": _OpSpec(
            sample_params=lambda rng: {"factor": _sample_uniform(rng, 0.4, 1.6)},
            apply=lambda x, p: TF.adjust_contrast(x, p["factor"]),
        ),
        "brightness": _OpSpec(
            sample_params=lambda rng: {"factor": _sample_uniform(rng, 0.4, 1.6)},
            apply=lambda x, p: TF.adjust_brightness(x, p["factor"]),
        ),
        "saturation": _OpSpec(
            sample_params=lambda rng: {"factor": _sample_uniform(rng, 0.4, 1.6)},
            apply=lambda x, p: TF.adjust_saturation(x, p["factor"]),
        ),
        "sharpness": _OpSpec(
            sample_params=lambda rng: {"factor": _sample_uniform(rng, 0, 10.0)},
            apply=lambda x, p: TF.adjust_sharpness(x, p["factor"]),
        ),
        "shear_x": _OpSpec(
            sample_params=lambda rng: {"shear": _sample_uniform(rng, -20.0, 20.0)},
            apply=lambda x, p: _make_affine(
                x, shear=(p["shear"], 0.0)
            ),
        ),
        "shear_y": _OpSpec(
            sample_params=lambda rng: {"shear": _sample_uniform(rng, -20.0, 20.0)},
            apply=lambda x, p: _make_affine(
                x, shear=(0.0, p["shear"])
            ),
        ),
        "translate_x": _OpSpec(
            sample_params=lambda rng: {"shift": _sample_uniform(rng, -0.2, 0.2)},
            apply=lambda x, p: _make_affine(
                x,
                translate=(
                    int(round(p["shift"] * x.shape[2])),
                    0,
                ),
            ),
        ),
        "translate_y": _OpSpec(
            sample_params=lambda rng: {"shift": _sample_uniform(rng, -0.2, 0.2)},
            apply=lambda x, p: _make_affine(
                x,
                translate=(
                    0,
                    int(round(p["shift"] * x.shape[1])),
                ),
            ),
        ),
        "rotate": _OpSpec(
            sample_params=lambda rng: {"angle": _sample_uniform(rng, -30.0, 30.0)},
            apply=lambda x, p: TF.rotate(x, p["angle"]),
        ),
    }


class SAFERAugmenter:
    """
    Random augmentation pipeline generator.

    Parameters:
        num_views: how many views to synthesise per input batch.
        augmentations: ordered list of augmentation names to sample from.
        max_ops: cap on how many operations to include in one pipeline (None = unlimited).
        prob: Bernoulli probability per augmentation.
        seed: optional seed for determinism.
        force_noise_first: ensure the noise_op is the first operation in the pipeline.
        require_freq_or_blur: enforce at least one frequency/blur op per pipeline.
        noise_op: name of the noise operation to place first when forced.
        freq_or_blur_ops: candidate ops used to satisfy the frequency/blur requirement.
    """

    def __init__(
        self,
        num_views: int,
        augmentations: Optional[Sequence[str]] = None,
        max_ops: Optional[int] = None,
        prob: float = 0.7,
        seed: Optional[int] = None,
        force_noise_first: bool = False,
        require_freq_or_blur: bool = False,
        noise_op: str = "gaussian_noise",
        freq_or_blur_ops: Optional[Sequence[str]] = None,
    ) -> None:
        assert num_views >= 1, "num_views must be ≥ 1"
        assert 0.0 <= prob <= 1.0, "prob must be in [0, 1]"
        self.num_views = num_views
        self.max_ops = max_ops
        self.prob = prob
        self.rng = random.Random(seed)
        self.registry = _build_registry()
        self.force_noise_first = bool(force_noise_first)
        self.require_freq_or_blur = bool(require_freq_or_blur)
        self.noise_op = noise_op
        self.freq_or_blur_ops = list(freq_or_blur_ops) if freq_or_blur_ops is not None else [
            "fft_low_pass",
            "gaussian_blur",
        ]
        default_ops: Tuple[str, ...] = (
            "gaussian_blur",
            "gaussian_noise",
            "fft_low_pass",
            "equalize",
            "invert",
            "solarize",
            "posterize",
            "contrast",
            "brightness",
            "saturation",
            "sharpness",
            "shear_x",
            "shear_y",
            "translate_x",
            "translate_y",
            "rotate",
        )
        self.augmentations = list(augmentations) if augmentations is not None else list(default_ops)

    def _sample_pipeline(self) -> List[Tuple[str, Dict[str, float]]]:
        ops: List[Tuple[str, Dict[str, float]]] = []
        for name in self.augmentations:
            if name not in self.registry:
                continue
            if self.rng.random() <= self.prob:
                spec = self.registry[name]
                params = spec.sample_params(self.rng)
                ops.append((name, params))

        mandatory: List[Tuple[str, Dict[str, float]]] = []
        if self.force_noise_first:
            noise_spec = self.registry.get(self.noise_op)
            if noise_spec is not None:
                noise_params = None
                for i, (name, params) in enumerate(ops):
                    if name == self.noise_op:
                        noise_params = params
                        ops.pop(i)
                        break
                if noise_params is None:
                    noise_params = noise_spec.sample_params(self.rng)
                mandatory.append((self.noise_op, noise_params))

        if self.require_freq_or_blur:
            candidates = [name for name in self.freq_or_blur_ops if name in self.registry]
            if candidates:
                chosen_name = None
                chosen_params = None
                for i, (name, params) in enumerate(ops):
                    if name in candidates:
                        chosen_name = name
                        chosen_params = params
                        ops.pop(i)
                        break
                if chosen_name is None:
                    chosen_name = self.rng.choice(candidates)
                    chosen_params = self.registry[chosen_name].sample_params(self.rng)
                mandatory.append((chosen_name, chosen_params))

        if self.max_ops is not None:
            budget = max(self.max_ops - len(mandatory), 0)
            if len(ops) > budget:
                ops = self.rng.sample(ops, budget)
        self.rng.shuffle(ops)
        return mandatory + ops

    def _apply_ops(self, x: Tensor, ops: List[Tuple[str, Dict[str, float]]]) -> Tensor:
        out = x.clone()
        for name, params in ops:
            spec = self.registry.get(name)
            if spec is None:
                continue
            out = spec.apply(out, params)
            out = _clamp_img(out)
        return out

    def sample_pipelines(self, num_views: Optional[int] = None) -> List[List[Tuple[str, Dict[str, float]]]]:
        total_views = num_views or self.num_views
        return [self._sample_pipeline() for _ in range(total_views)]

    def apply_pipeline(self, x: Tensor, pipeline: List[Tuple[str, Dict[str, float]]]) -> Tensor:
        return self._apply_ops(x, pipeline)

    def augment(self, x: Tensor, num_views: Optional[int] = None) -> Tensor:
        """
        Generate augmentation views for a batch.

        Returns:
            Tensor with shape (B, V, C, H, W).
        """
        b = x.size(0)
        views = []
        pipelines = self.sample_pipelines(num_views)
        for ops in pipelines:
            augmented = [self._apply_ops(x[i], ops) for i in range(b)]
            if not augmented:
                stacked = x.clone()
            else:
                stacked = torch.stack(augmented, dim=0)
            views.append(stacked)
        return torch.stack(views, dim=1)
