#!/usr/bin/env python3
"""
Visualize SAFER augmentation operations across parameter grids.

This script loads an RGB image, sweeps each augmentation from
``utils.safer_aug`` across predefined parameter ranges, and writes
the resulting grids to disk (and/or displays them).
"""

from __future__ import annotations

import argparse
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence

import matplotlib.pyplot as plt
import torch
import torchvision.transforms.functional as TF
from PIL import Image

from utils.safer_aug import _build_registry, _clamp_img, _OpSpec


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "range" or "choices"
    min: float | None = None
    max: float | None = None
    choices: Sequence[float] | None = None
    num: int | None = None
    dtype: str = "float"


PARAM_METADATA: dict[str, Sequence[ParamSpec]] = {
    "gaussian_blur": (
        ParamSpec(
            name="kernel_size",
            kind="choices",
            choices=(7,9),
            dtype="int",
        ),
        ParamSpec(
            name="sigma",
            kind="range",
            min=0.1,
            max=2.5,
        ),
    ),
    "gaussian_noise": (
        ParamSpec(
            name="std",
            kind="range",
            min=0.01,
            max=0.1,
        ),
    ),
    "fft_low_pass": (
        ParamSpec(
            name="keep_ratio",
            kind="range",
            min=0.2,
            max=0.7,
        ),
    ),
    "fft_high_pass": (
        ParamSpec(
            name="keep_ratio",
            kind="range",
            min=0.3,
            max=0.6,
        ),
    ),
    "solarize": (
        ParamSpec(
            name="threshold",
            kind="range",
            min=0.1,
            max=0.9,
        ),
    ),
    "posterize": (
        ParamSpec(
            name="bits",
            kind="choices",
            choices=(2, 3, 4, 5),
            dtype="int",
        ),
    ),
    "contrast": (
        ParamSpec(
            name="factor",
            kind="range",
            min=0.4,
            max=1.6,
        ),
    ),
    "brightness": (
        ParamSpec(
            name="factor",
            kind="range",
            min=0.4,
            max=1.6,
        ),
    ),
    "saturation": (
        ParamSpec(
            name="factor",
            kind="range",
            min=0.4,
            max=1.6,
        ),
    ),
    "sharpness": (
        ParamSpec(
            name="factor",
            kind="range",
            min=0.0,
            max=10.0,
        ),
    ),
    "shear_x": (
        ParamSpec(
            name="shear",
            kind="range",
            min=-20.0,
            max=20.0,
        ),
    ),
    "shear_y": (
        ParamSpec(
            name="shear",
            kind="range",
            min=-20.0,
            max=20.0,
        ),
    ),
    "translate_x": (
        ParamSpec(
            name="shift",
            kind="range",
            min=-0.2,
            max=0.2,
        ),
    ),
    "translate_y": (
        ParamSpec(
            name="shift",
            kind="range",
            min=-0.2,
            max=0.2,
        ),
    ),
    "rotate": (
        ParamSpec(
            name="angle",
            kind="range",
            min=-30.0,
            max=30.0,
        ),
    ),
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize SAFER augmentation operations."
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="Path to an RGB image used as the augmentation source.",
    )
    parser.add_argument(
        "--transforms",
        type=str,
        nargs="*",
        default=None,
        help="Optional subset of transform names to visualize. "
        "Defaults to all registered transforms.",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=4,
        help="Fallback number of random samples when no parameter grid is defined (default: 4).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed controlling the random parameter draws.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("safer_aug_viz"),
        help="Directory where figures will be saved (default: ./safer_aug_viz).",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display figures interactively in addition to saving them.",
    )
    parser.add_argument(
        "--max-size",
        type=int,
        default=None,
        help="Optional maximum edge length to resize the input (preserves aspect).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=5,
        help="Number of interpolation steps for range parameters (default: 5).",
    )
    parser.add_argument(
        "--include-original",
        action="store_true",
        help="Include the original image (inline for 1D grids, saved separately for 2D grids).",
    )
    return parser.parse_args()


def _load_image(path: Path, max_size: int | None = None) -> torch.Tensor:
    image = Image.open(path).convert("RGB")
    tensor = TF.to_tensor(image)
    if max_size is not None:
        h, w = tensor.shape[1:]
        longest = max(h, w)
        if longest > max_size:
            scale = max_size / float(longest)
            new_h = max(1, int(round(h * scale)))
            new_w = max(1, int(round(w * scale)))
            tensor = TF.resize(tensor, size=[new_h, new_w], antialias=True)
    return tensor


def _ensure_transforms(
    requested: Sequence[str] | None, available: Iterable[str]
) -> List[str]:
    available_set = {name for name in available}
    if not requested:
        return sorted(available_set)
    missing = [name for name in requested if name not in available_set]
    if missing:
        raise ValueError(
            f"Unknown transform(s): {', '.join(missing)}. "
            f"Available transforms: {', '.join(sorted(available_set))}"
        )
    return list(dict.fromkeys(requested))


def _format_value(value: float, dtype: str) -> str:
    if dtype == "int":
        return f"{int(round(value))}"
    if abs(value) >= 100 or (abs(value) <= 1e-3 and value != 0):
        return f"{value:.2e}"
    return f"{value:.3f}"


def _generate_values(spec: ParamSpec, steps: int) -> List[float]:
    if spec.kind == "choices":
        values = list(spec.choices or [])
    else:
        num = spec.num or steps
        if num <= 1:
            values = [spec.min or 0.0]
        else:
            values = torch.linspace(spec.min or 0.0, spec.max or 0.0, num).tolist()
    if spec.dtype == "int":
        return [int(round(v)) for v in values]
    return values


def _visualize_transform(
    name: str,
    base_img: torch.Tensor,
    rng: random.Random,
    output_dir: Path,
    show: bool,
    spec: _OpSpec,
    param_specs: Sequence[ParamSpec],
    steps: int,
    include_original: bool,
    fallback_samples: int,
) -> None:
    param_specs = list(param_specs)
    if len(param_specs) > 2:
        param_specs = param_specs[:2]
    num_params = len(param_specs)

    if num_params == 0:
        panels = fallback_samples
        total = panels + int(include_original)
        fig, axes = plt.subplots(
            1,
            total,
            figsize=(2.4 * total, 2.4),
            squeeze=False,
        )
        axes = axes[0]
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        idx = 0
        if include_original:
            axes[0].imshow(TF.to_pil_image(base_img))
            axes[0].set_title("original", fontsize=10)
            idx = 1
        for col in range(panels):
            params = spec.sample_params(rng)
            augmented = spec.apply(base_img.clone(), params)
            augmented = _clamp_img(augmented)
            axes[idx + col].imshow(TF.to_pil_image(augmented))
            axes[idx + col].set_title(f"sample {col + 1}", fontsize=9)
        fig.suptitle(name, fontsize=14)
        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.88)
    elif num_params == 1:
        param = param_specs[0]
        values = _generate_values(param, steps)
        total = len(values) + int(include_original)
        fig, axes = plt.subplots(
            1,
            total,
            figsize=(2.4 * total, 2.4),
            squeeze=False,
        )
        axes = axes[0]
        for ax in axes:
            ax.set_xticks([])
            ax.set_yticks([])
        col = 0
        if include_original:
            axes[0].imshow(TF.to_pil_image(base_img))
            axes[0].set_title("original", fontsize=10)
            col = 1
        for value in values:
            params = {param.name: value}
            augmented = spec.apply(base_img.clone(), params)
            augmented = _clamp_img(augmented)
            axes[col].imshow(TF.to_pil_image(augmented))
            axes[col].set_title(
                f"{param.name}={_format_value(value, param.dtype)}",
                fontsize=9,
            )
            col += 1
        fig.suptitle(name, fontsize=14)
        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.88)
    else:
        row_param = param_specs[0]
        col_param = param_specs[1]
        row_values = _generate_values(row_param, steps)
        col_values = _generate_values(col_param, steps)
        fig, axes = plt.subplots(
            len(row_values),
            len(col_values),
            figsize=(2.2 * len(col_values), 2.2 * len(row_values)),
            squeeze=False,
        )
        for r, rv in enumerate(row_values):
            for c, cv in enumerate(col_values):
                ax = axes[r][c]
                ax.set_xticks([])
                ax.set_yticks([])
                params = {
                    row_param.name: rv,
                    col_param.name: cv,
                }
                augmented = spec.apply(base_img.clone(), params)
                augmented = _clamp_img(augmented)
                ax.imshow(TF.to_pil_image(augmented))
        for c, value in enumerate(col_values):
            axes[0][c].set_title(
                f"{col_param.name}={_format_value(value, col_param.dtype)}",
                fontsize=9,
            )
        for r, value in enumerate(row_values):
            axes[r][0].set_ylabel(
                f"{row_param.name}={_format_value(value, row_param.dtype)}",
                fontsize=9,
                rotation=90,
                labelpad=6,
            )
        fig.suptitle(name, fontsize=14)
        fig.subplots_adjust(wspace=0.02, hspace=0.02, top=0.9)

        if include_original:
            preview_dir = output_dir / "originals"
            preview_dir.mkdir(parents=True, exist_ok=True)
            preview_path = preview_dir / f"{name}.png"
            if not preview_path.exists():
                preview_fig, preview_ax = plt.subplots(figsize=(2.4, 2.4))
                preview_ax.set_xticks([])
                preview_ax.set_yticks([])
                preview_ax.imshow(TF.to_pil_image(base_img))
                preview_ax.set_title("original", fontsize=10)
                preview_fig.tight_layout()
                preview_fig.savefig(preview_path, dpi=200)
                plt.close(preview_fig)

    output_dir.mkdir(parents=True, exist_ok=True)
    figure_path = output_dir / f"{name}.png"
    fig.savefig(figure_path, dpi=200)

    if show:
        plt.show()
    plt.close(fig)


def main() -> None:
    args = _parse_args()

    base_img = _load_image(args.image, args.max_size)
    registry = _build_registry()
    transform_names = _ensure_transforms(args.transforms, registry.keys())
    rng = random.Random(args.seed)

    for name in transform_names:
        spec = registry[name]
        _visualize_transform(
            name=name,
            base_img=base_img,
            rng=rng,
            output_dir=args.output_dir,
            show=args.show,
            spec=spec,
            param_specs=PARAM_METADATA.get(name, ()),
            steps=args.steps,
            include_original=args.include_original,
            fallback_samples=args.num_samples,
        )


if __name__ == "__main__":
    main()
