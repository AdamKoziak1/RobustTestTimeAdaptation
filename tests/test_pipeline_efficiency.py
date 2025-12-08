import math
import sys
import time
from pathlib import Path
from typing import Callable, Tuple

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapt_algorithm import (
    _barlow_twins_loss,
    _barlow_twins_loss_einsum,
    _entropy_minimization_loss,
    _js_divergence,
    softmax_entropy,
)
from utils.fft import FFTDrop2D

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
RESNET_FEAT_DIM = 512  # ResNet-18 avgpool output
BATCH_SIZE = 64        # default batch size in unsupervise_adapt.py
NUM_CLASSES = 65       # default for OfficeHome in utils/util.py
SAFER_VIEWS = 5        # 1 original + 4 augmented
IMG_SHAPE = (3, 224, 224)


def _maybe_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def benchmark(fn: Callable, *args, repeats: int = 10, **kwargs) -> Tuple[torch.Tensor, float]:
    """Return (result, avg_time_ms) with optional CUDA sync for fair timing."""
    _maybe_sync()
    start = time.perf_counter()
    out = None
    for _ in range(repeats):
        out = fn(*args, **kwargs)
    _maybe_sync()
    elapsed = (time.perf_counter() - start) / repeats * 1e3
    return out, elapsed


def spatial_fft_maskless(x: torch.Tensor, keep_ratio: float) -> torch.Tensor:
    """Spatial FFT drop without allocating a mask tensor."""
    _, _, h, w = x.shape
    freq = torch.fft.fft2(x, dim=(-2, -1), norm="ortho")
    freq_shifted = torch.fft.fftshift(freq, dim=(-2, -1))

    keep_h = max(1, math.ceil(h * keep_ratio))
    keep_w = max(1, math.ceil(w * keep_ratio))
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


def softmax_entropy_opt(logits: torch.Tensor) -> torch.Tensor:
    lse = torch.logsumexp(logits, dim=1, keepdim=True)
    probs = torch.exp(logits - lse)
    entropy = lse.squeeze(1) - torch.sum(probs * logits, dim=1)
    return entropy


def js_divergence_opt(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_prob = probs.mean(dim=1)
    entropy_each = -torch.xlogy(probs, probs.clamp_min(eps)).sum(dim=-1)
    entropy_mean = -torch.xlogy(mean_prob, mean_prob.clamp_min(eps)).sum(dim=-1)
    return (entropy_mean - entropy_each.mean(dim=1)).mean()


def entropy_minimization_opt(mean_prob: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    entropy = -torch.xlogy(mean_prob, mean_prob.clamp_min(eps)).sum(dim=-1)
    return entropy.mean()


@pytest.mark.parametrize("keep_ratio", [0.1, 0.5, 1])
def test_fft_spatial_mask_vs_inplace(keep_ratio):
    x = torch.randn(BATCH_SIZE, *IMG_SHAPE, device=DEVICE)
    layer = FFTDrop2D(keep_ratio=keep_ratio, mode="spatial").to(DEVICE)

    baseline, base_ms = benchmark(layer._apply_spatial_fft, x)
    optimized, opt_ms = benchmark(spatial_fft_maskless, x, keep_ratio)

    torch.testing.assert_close(baseline, optimized, atol=1e-6, rtol=1e-6)
    print(f"[FFT spatial keep={keep_ratio}] baseline={base_ms:.3f}ms optimized={opt_ms:.3f}ms")


def test_barlow_twins_fast_vs_einsum():
    torch.manual_seed(0)
    features = torch.randn(BATCH_SIZE, SAFER_VIEWS, RESNET_FEAT_DIM, device=DEVICE)
    kwargs = {"offdiag_weight": 1.0}
    base, base_ms = benchmark(_barlow_twins_loss, features, **kwargs)
    fast, fast_ms = benchmark(_barlow_twins_loss_einsum, features, **kwargs)
    torch.testing.assert_close(base, fast, atol=1e-5, rtol=1e-5)
    print(f"[Barlow Twins] fast={base_ms:.3f}ms einsum={fast_ms:.3f}ms")


def test_softmax_entropy_efficiency():
    logits = torch.randn(BATCH_SIZE, NUM_CLASSES, device=DEVICE)
    base, base_ms = benchmark(softmax_entropy, logits)
    opt, opt_ms = benchmark(softmax_entropy_opt, logits)
    torch.testing.assert_close(base, opt, atol=1e-6, rtol=1e-6)
    print(f"[Tent entropy] baseline={base_ms:.3f}ms optimized={opt_ms:.3f}ms")


def test_js_divergence_efficiency():
    probs = torch.rand(BATCH_SIZE, SAFER_VIEWS, NUM_CLASSES, device=DEVICE)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    base, base_ms = benchmark(_js_divergence, probs)
    opt, opt_ms = benchmark(js_divergence_opt, probs)
    torch.testing.assert_close(base, opt, atol=1e-6, rtol=1e-6)
    print(f"[SAFER JS] baseline={base_ms:.3f}ms optimized={opt_ms:.3f}ms")


def test_entropy_minimization_efficiency():
    mean_prob = torch.rand(BATCH_SIZE, NUM_CLASSES, device=DEVICE)
    mean_prob = mean_prob / mean_prob.sum(dim=-1, keepdim=True)
    base, base_ms = benchmark(_entropy_minimization_loss, mean_prob)
    opt, opt_ms = benchmark(entropy_minimization_opt, mean_prob)
    torch.testing.assert_close(base, opt, atol=1e-6, rtol=1e-6)
    print(f"[EM loss] baseline={base_ms:.3f}ms optimized={opt_ms:.3f}ms")


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))
