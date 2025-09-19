# coding=utf-8
"""Shared SVD-based projection layers."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

__all__ = ["SVDDrop2D"]


class SVDDrop2D(nn.Module):
    def __init__(self, rank_ratio: float, mode: str,
                 backprop_mode: str = 'ste'):
        super().__init__()
        assert mode in ('spatial', 'channel')
        assert 0.0 <= rank_ratio <= 1.0
        assert backprop_mode in ('exact', 'ste')
        self.rank_ratio = rank_ratio
        self.mode = mode
        self.backprop_mode = backprop_mode

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.rank_ratio >= 1.0:
            return x

        b, c, h, w = x.shape
        if self.mode == 'spatial':
            x_flat = x.reshape(b * c, h, w)
            dim = h
        else:  # channel
            x_flat = x.reshape(b, c, h * w)
            dim = c

        rank = max(1, math.ceil(dim * self.rank_ratio))
        u, s, vh = torch.svd_lowrank(x_flat, q=rank, niter=2)
        x_recon = torch.matmul(u * s.unsqueeze(1), torch.transpose(vh, 1, 2)).reshape(b, c, h, w)

        if self.backprop_mode == 'exact':
            return x_recon
        # straight-through estimator: forward uses SVD projection, gradients bypass it
        return x + (x_recon - x).detach()

class SVDLoader:
    def __init__(self, dataloader, rank_ratio: float,
                 device: str = "cuda",
                 mode: str = "spatial",
                 use_ste: bool = False):
        self.loader = dataloader
        self.device = device
        backprop_mode = 'ste' if use_ste else 'exact'
        self.proj = SVDDrop2D(rank_ratio=rank_ratio, mode=mode,
                              backprop_mode=backprop_mode).to(self.device)
        self.rank_ratio = rank_ratio

    def __iter__(self):
        for xb, yb in self.loader:
            xb = xb.to(self.device, non_blocking=True)
            if self.rank_ratio < 1.0:
                with torch.no_grad():
                    xb = self.proj(xb)
            yield xb, yb