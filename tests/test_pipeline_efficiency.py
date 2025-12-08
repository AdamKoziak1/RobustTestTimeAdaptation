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
    for i in range(repeats):
        torch.manual_seed(i)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(i)
        out = fn(*args, **kwargs)
    _maybe_sync()
    elapsed = (time.perf_counter() - start) / repeats * 1e3
    return out, elapsed


def js_divergence_opt(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    mean_prob = probs.mean(dim=1)
    entropy_each = -torch.xlogy(probs, probs.clamp_min(eps)).sum(dim=-1)
    entropy_mean = -torch.xlogy(mean_prob, mean_prob.clamp_min(eps)).sum(dim=-1)
    return (entropy_mean - entropy_each.mean(dim=1)).mean()


def test_barlow_twins_fast_vs_einsum():
    features = torch.randn(BATCH_SIZE, SAFER_VIEWS, RESNET_FEAT_DIM, device=DEVICE)
    kwargs = {"offdiag_weight": 1.0}
    base, base_ms = benchmark(_barlow_twins_loss, features, **kwargs)
    fast, fast_ms = benchmark(_barlow_twins_loss_einsum, features, **kwargs)
    torch.testing.assert_close(base, fast, atol=1e-5, rtol=1e-5)
    print(f"[Barlow Twins] fast={base_ms:.3f}ms einsum={fast_ms:.3f}ms")


def test_js_divergence_efficiency():
    probs = torch.rand(BATCH_SIZE, SAFER_VIEWS, NUM_CLASSES, device=DEVICE)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    base, base_ms = benchmark(_js_divergence, probs)
    opt, opt_ms = benchmark(js_divergence_opt, probs)
    torch.testing.assert_close(base, opt, atol=1e-6, rtol=1e-6)
    print(f"[SAFER JS] baseline={base_ms:.3f}ms optimized={opt_ms:.3f}ms")


#@pytest.mark.parametrize("keep_ratio", [0.1, 0.5, 1])

if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))
