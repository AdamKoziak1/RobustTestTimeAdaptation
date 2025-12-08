# coding=utf-8
"""Shared FFT-based projection layers."""
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn

__all__ = ["FFTDrop2D", "FFTLoader"]

_FFTMode = Literal["spatial", "channel"]


def _init_logit(alpha: float) -> torch.Tensor:
    eps = 1e-4
    clipped = min(max(alpha, eps), 1.0 - eps)
    return torch.logit(torch.tensor(clipped, dtype=torch.float32))


class FFTDrop2D(nn.Module):
    """Frequency-domain filtering with optional residual mixing."""

    def __init__(
        self,
        keep_ratio: float,
        mode: _FFTMode = "spatial",
        backprop_mode: str = "exact",
        use_residual: bool = False,
        alpha: float = 1.0,
        learn_alpha: bool = False,
    ) -> None:
        super().__init__()
        assert 0.0 <= keep_ratio <= 1.0, "keep_ratio must be in [0, 1]"
        assert mode in ("spatial", "channel"), "mode must be 'spatial' or 'channel'"
        assert backprop_mode in ("exact", "ste"), "backprop_mode must be 'exact' or 'ste'"
        self.keep_ratio = keep_ratio
        self.mode = mode
        self.backprop_mode = backprop_mode
        self.use_residual = use_residual
        self.learn_alpha = learn_alpha and use_residual

        if self.learn_alpha:
            self.alpha_param = nn.Parameter(_init_logit(alpha))
        else:
            self.register_buffer("alpha_buffer", torch.tensor(float(alpha), dtype=torch.float32))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.keep_ratio >= 1.0:
            return x

        filtered = self._apply_fft_filter(x)

        if self.backprop_mode == "ste":
            filtered = x + (filtered - x).detach()

        if not self.use_residual:
            return filtered

        alpha = self._mix_coeff(filtered.dtype, filtered.device)
        return alpha * filtered + (1.0 - alpha) * x

    def _mix_coeff(self, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        if not self.use_residual:
            return torch.tensor(1.0, dtype=dtype, device=device)
        if self.learn_alpha:
            return torch.sigmoid(self.alpha_param).to(dtype=dtype)
        return torch.clamp(self.alpha_buffer, 0.0, 1.0).to(dtype=dtype, device=device)

    def _apply_fft_filter(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "spatial":
            return self._apply_spatial_fft(x)
        return self._apply_channel_fft(x)
    
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

    def _apply_channel_fft(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        freq = torch.fft.fft(x, dim=1, norm="ortho")
        freq_shifted = torch.fft.fftshift(freq, dim=1)

        keep_c = max(1, int(math.ceil(c * self.keep_ratio)))
        start_c = (c - keep_c) // 2
        end_c = start_c + keep_c

        mask = torch.zeros_like(freq_shifted.real)
        mask[:, start_c:end_c, :, :] = 1.0
        filtered_shifted = freq_shifted * mask
        filtered = torch.fft.ifftshift(filtered_shifted, dim=1)
        return torch.fft.ifft(filtered, dim=1, norm="ortho").real


class FFTLoader:
    def __init__(
        self,
        dataloader,
        keep_ratio: float,
        device: str = "cuda",
        mode: _FFTMode = "spatial",
        use_ste: bool = False,
        use_residual: bool = False,
        alpha: float = 1.0,
        learn_alpha: bool = False,
    ) -> None:
        self.loader = dataloader
        self.device = device
        backprop_mode = "ste" if use_ste else "exact"
        self.keep_ratio = keep_ratio
        self.proj = FFTDrop2D(
            keep_ratio=keep_ratio,
            mode=mode,
            backprop_mode=backprop_mode,
            use_residual=use_residual,
            alpha=alpha,
            learn_alpha=learn_alpha,
        ).to(self.device)

    def __iter__(self):
        for xb, yb in self.loader:
            xb = xb.to(self.device, non_blocking=True)
            if self.keep_ratio < 1.0:
                with torch.no_grad():
                    xb = self.proj(xb)
            yield xb, yb
