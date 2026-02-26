# coding=utf-8
"""Shared FFT-based projection layers."""
from __future__ import annotations

import math

import torch
import torch.nn as nn

__all__ = ["FFTDrop2D", "FFTLoader"]


class FFTDrop2D(nn.Module):
    """Spatial frequency-domain filtering with fixed-alpha residual mixing."""

    def __init__(
        self,
        keep_ratio: float,
        backprop_mode: str = "exact",
        alpha: float = 1.0,
    ) -> None:
        super().__init__()
        assert 0.0 <= keep_ratio <= 1.0, "keep_ratio must be in [0, 1]"
        assert backprop_mode in ("exact", "ste"), "backprop_mode must be 'exact' or 'ste'"
        self.keep_ratio = keep_ratio
        self.backprop_mode = backprop_mode
        self.register_buffer("alpha_buffer", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.keep_ratio >= 1.0:
            return x

        filtered = self._apply_spatial_fft(x)

        if self.backprop_mode == "ste":
            filtered = x + (filtered - x).detach()

        alpha = torch.clamp(self.alpha_buffer, 0.0, 1.0).to(dtype=filtered.dtype, device=filtered.device)
        return alpha * filtered + (1.0 - alpha) * x
    
    def _apply_spatial_fft(self, x: torch.Tensor) -> torch.Tensor:
        """Spatial FFT drop without allocating a mask tensor."""
        _, _, h, w = x.shape
        freq = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
        freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

        keep_h = max(1, math.ceil(h * self.keep_ratio))
        keep_w = max(1, math.ceil(w * self.keep_ratio))
        start_h = (h - keep_h) // 2
        end_h = start_h + keep_h
        start_w = (w - keep_w) // 2
        end_w = start_w + keep_w

        freq_shifted[..., :start_h, :] = 0
        freq_shifted[..., end_h:, :] = 0
        freq_shifted[..., :, :start_w] = 0
        freq_shifted[..., :, end_w:] = 0

        filtered = torch.fft.ifftshift(freq_shifted, dim=(-2, -1))
        return torch.fft.ifft2(filtered, dim=(-2, -1), norm="ortho").real


class FFTLoader:
    def __init__(
        self,
        dataloader,
        keep_ratio: float,
        device: str = "cuda",
        use_ste: bool = False,
        alpha: float = 1.0,
    ) -> None:
        self.loader = dataloader
        self.device = device
        backprop_mode = "ste" if use_ste else "exact"
        self.keep_ratio = keep_ratio
        self.proj = FFTDrop2D(
            keep_ratio=keep_ratio,
            backprop_mode=backprop_mode,
            alpha=alpha,
        ).to(self.device)

    def __iter__(self):
        for xb, yb in self.loader:
            xb = xb.to(self.device, non_blocking=True)
            if self.keep_ratio < 1.0:
                with torch.no_grad():
                    xb = self.proj(xb)
            yield xb, yb
