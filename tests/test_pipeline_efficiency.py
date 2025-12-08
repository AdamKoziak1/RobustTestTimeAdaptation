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


def benchmark(fn: Callable, *args, repeats: int = 10, warmup: int = 3, **kwargs) -> Tuple[torch.Tensor, float]:
    """Return (result, avg_time_ms) with warmup and optional CUDA sync for fair timing."""
    # Warm up to amortize kernel/cudnn/fft plan init.
    for k in range(warmup):
        torch.manual_seed(10_000 + k)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(10_000 + k)
        fn(*args, **kwargs)
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


@pytest.mark.parametrize("offdiag_weight", [0.01, 0.1, 1])
def test_barlow_twins_fast_vs_einsum(offdiag_weight):
    features = torch.randn(BATCH_SIZE, SAFER_VIEWS, RESNET_FEAT_DIM, device=DEVICE)
    kwargs = {"offdiag_weight": offdiag_weight}
    base, base_ms = benchmark(_barlow_twins_loss, features, **kwargs)
    fast, fast_ms = benchmark(_barlow_twins_loss_einsum, features, **kwargs)
    torch.testing.assert_close(base, fast, atol=1e-5, rtol=1e-5)
    print(f"[Barlow Twins] fast={base_ms:.3f}ms einsum={fast_ms:.3f}ms")


@pytest.mark.parametrize("keep_ratio", [0.1, 0.3, 0.7, 1])
def test_fft_spatial_mask_vs_inplace(keep_ratio):
    x = torch.randn(BATCH_SIZE, *IMG_SHAPE, device=DEVICE)
    layer = FFTDrop2D(keep_ratio=keep_ratio, mode="spatial").to(DEVICE)

    optimized, opt_ms = benchmark(layer._apply_spatial_fft, x)
    baseline, base_ms = benchmark(layer._apply_spatial_fft_old, x)

    torch.testing.assert_close(baseline, optimized, atol=1e-6, rtol=1e-6)
    print(f"[FFT spatial keep={keep_ratio}] baseline={base_ms:.3f}ms optimized={opt_ms:.3f}ms")

if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))
