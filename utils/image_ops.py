# coding=utf-8
"""Shared blur/compression operators for input-level defenses."""
from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.io import image as tv_image
from utils.fft import FFTDrop2D
from utils.svd import SVDDrop2D

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

__all__ = [
    "GaussianBlur2D",
    "GaussianBlurLoader",
    "InputDefense",
    "JPEGCompression",
    "JPEGCompressionLoader",
]


class GaussianBlur2D(nn.Module):
    """Depthwise Gaussian blur implemented with a fixed conv kernel."""

    def __init__(self, sigma: float, kernel_size: int | None = None) -> None:
        super().__init__()
        self.sigma = float(sigma)
        self.kernel_size = self._resolve_kernel_size(kernel_size)
        if self.sigma > 0:
            kernel = self._build_kernel(self.kernel_size, self.sigma)
        else:
            kernel = torch.zeros(1, 1, 1, 1, dtype=torch.float32)
        self.register_buffer("_kernel", kernel, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.sigma <= 0:
            return x
        kernel = self._kernel.to(dtype=x.dtype, device=x.device)
        padding = self.kernel_size // 2
        c = x.size(1)
        kernel = kernel.expand(c, 1, self.kernel_size, self.kernel_size)
        return F.conv2d(x, kernel, padding=padding, groups=c)

    def _resolve_kernel_size(self, kernel_size: int | None) -> int:
        if self.sigma <= 0:
            return max(3, kernel_size or 3)
        if kernel_size is None or kernel_size <= 0:
            kernel_size = int(2 * math.ceil(3 * self.sigma) + 1)
        if kernel_size % 2 == 0:
            kernel_size += 1
        return max(3, kernel_size)

    def _build_kernel(self, kernel_size: int, sigma: float) -> torch.Tensor:
        coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
        grid = coords[:, None] ** 2 + coords[None, :] ** 2
        kernel = torch.exp(-0.5 * grid / (sigma ** 2))
        kernel /= kernel.sum()
        return kernel.view(1, 1, kernel_size, kernel_size)


class GaussianBlurLoader:
    """Wraps a dataloader and blurs the inputs on-the-fly."""

    def __init__(self, dataloader, sigma: float, device: str = "cuda") -> None:
        self.loader = dataloader
        self.device = torch.device(device)
        self.enabled = sigma > 0
        self.blur = GaussianBlur2D(sigma).to(self.device) if self.enabled else None

    def __iter__(self):
        for xb, yb in self.loader:
            xb = xb.to(self.device, non_blocking=True)
            if self.enabled and self.blur is not None:
                with torch.no_grad():
                    xb = self.blur(xb)
            yield xb, yb

    def __len__(self):
        if hasattr(self.loader, "__len__"):
            return len(self.loader)
        raise TypeError("Wrapped dataloader does not define __len__")


class JPEGCompressionLoader:
    """Applies JPEG re-encoding (quality∈[1,100]) before moving data to GPU."""

    def __init__(
        self,
        dataloader,
        quality: int,
        device: str = "cuda",
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        assert 1 <= quality <= 100, "quality must be in [1, 100]"
        self.loader = dataloader
        self.quality = int(quality)
        self.device = torch.device(device)
        self.mean = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)

    def __iter__(self):
        for xb, yb in self.loader:
            xb = self._compress_batch(xb)
            xb = xb.to(self.device, non_blocking=True)
            yield xb, yb

    def __len__(self):
        if hasattr(self.loader, "__len__"):
            return len(self.loader)
        raise TypeError("Wrapped dataloader does not define __len__")

    def _compress_batch(self, xb: torch.Tensor) -> torch.Tensor:
        if self.quality >= 100:
            return xb

        with torch.no_grad():
            batch = xb.detach()
            dtype = batch.dtype
            need_denorm = (batch.amin().item() < -0.05) or (batch.amax().item() > 1.05)
            if need_denorm:
                mean = self.mean.to(batch.device, dtype=batch.dtype)
                std = self.std.to(batch.device, dtype=batch.dtype)
                batch = batch * std + mean

            batch = batch.clamp(0.0, 1.0)
            batch_uint8 = (batch * 255.0).round().to(torch.uint8).cpu()

            compressed = []
            for sample in batch_uint8:
                encoded = tv_image.encode_jpeg(sample, quality=self.quality)
                decoded = tv_image.decode_jpeg(encoded, device="cpu")
                compressed.append(decoded)
            comp = torch.stack(compressed, dim=0).float() / 255.0

            if need_denorm:
                comp = (comp - self.mean) / self.std

            return comp.to(dtype=dtype)


class JPEGCompression(nn.Module):
    """JPEG re-encoding module for input-level defenses."""

    def __init__(
        self,
        quality: int,
        backprop_mode: str = "exact",
        mean: Sequence[float] = IMAGENET_MEAN,
        std: Sequence[float] = IMAGENET_STD,
    ) -> None:
        super().__init__()
        assert 1 <= quality <= 100, "quality must be in [1, 100]"
        assert backprop_mode in ("exact", "bpda"), "backprop_mode must be 'exact' or 'bpda'"
        self.quality = int(quality)
        self.backprop_mode = backprop_mode
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1))
        self.register_buffer("std", torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1))

    def forward(self, xb: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.quality >= 100:
            return xb

        with torch.no_grad():
            batch = xb.detach()
            dtype = batch.dtype
            device = batch.device
            need_denorm = (batch.amin().item() < -0.05) or (batch.amax().item() > 1.05)
            if need_denorm:
                mean = self.mean.to(device=device, dtype=batch.dtype)
                std = self.std.to(device=device, dtype=batch.dtype)
                batch = batch * std + mean

            batch = batch.clamp(0.0, 1.0)
            batch_uint8 = (batch * 255.0).round().to(torch.uint8).cpu()

            compressed = []
            for sample in batch_uint8:
                encoded = tv_image.encode_jpeg(sample, quality=self.quality)
                decoded = tv_image.decode_jpeg(encoded, device="cpu")
                compressed.append(decoded)
            comp = torch.stack(compressed, dim=0).float() / 255.0

            if need_denorm:
                mean = self.mean.to(device=comp.device, dtype=comp.dtype)
                std = self.std.to(device=comp.device, dtype=comp.dtype)
                comp = (comp - mean) / std

            comp = comp.to(dtype=dtype, device=device)

            if self.backprop_mode == "bpda":
                # Forward uses JPEG output; backward treats JPEG as identity.
                return xb + (comp - xb).detach()
            return comp


class InputDefense(nn.Module):
    """Composable input defense pipeline applied inside the model."""

    def __init__(
        self,
        *,
        jpeg_quality: int = 100,
        jpeg_backprop: str = "exact",
        gauss_sigma: float = 0.0,
        svd_rank_ratio: float = 1.0,
        svd_mode: str = "spatial",
        fft_keep_ratio: float = 1.0,
        fft_mode: str = "spatial",
        fft_alpha: float = 1.0,
        fft_use_residual: bool = True,
        fft_learn_alpha: bool = False,
    ) -> None:
        super().__init__()
        self.jpeg = (
            JPEGCompression(jpeg_quality, backprop_mode=jpeg_backprop)
            if 1 <= jpeg_quality < 100
            else None
        )
        self.blur = GaussianBlur2D(gauss_sigma) if gauss_sigma > 0 else None

        if svd_rank_ratio < 1.0:
            self.svd = SVDDrop2D(
                rank_ratio=svd_rank_ratio,
                mode=svd_mode,
                backprop_mode="exact",
            )
            self.fft = None
        elif fft_keep_ratio < 1.0:
            self.svd = None
            self.fft = FFTDrop2D(
                keep_ratio=fft_keep_ratio,
                mode=fft_mode,
                backprop_mode="exact",
                use_residual=fft_use_residual,
                alpha=fft_alpha,
                learn_alpha=fft_learn_alpha,
            )
        else:
            self.svd = None
            self.fft = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.jpeg is not None:
            x = self.jpeg(x)
        if self.blur is not None:
            x = self.blur(x)
        if self.svd is not None:
            x = self.svd(x)
        if self.fft is not None:
            x = self.fft(x)
        return x
