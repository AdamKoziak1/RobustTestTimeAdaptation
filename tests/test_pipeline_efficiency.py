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
    _aggregate_view_probs,
    _js_divergence,
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


def _peak_vram_mb() -> float:
    if DEVICE.type != "cuda":
        return 0.0
    stats = torch.cuda.memory_stats(DEVICE)
    return stats["allocated_bytes.all.peak"] / 1e6


@pytest.mark.parametrize("offdiag_weight", [0.01, 0.1, 1])
def test_barlow_twins_fast_vs_einsum(offdiag_weight):
    features = torch.randn(BATCH_SIZE, SAFER_VIEWS, RESNET_FEAT_DIM, device=DEVICE)
    kwargs = {"offdiag_weight": offdiag_weight}
    base, base_ms = benchmark(_barlow_twins_loss, features, **kwargs)
    fast, fast_ms = benchmark(_barlow_twins_loss_einsum, features, **kwargs)
    torch.testing.assert_close(base, fast, atol=1e-5, rtol=1e-5)
    print(f"[Barlow Twins] fast={base_ms:.3f}ms einsum={fast_ms:.3f}ms")


def _benchmark_js(mode: str, views_list=(2, 3, 5, 7), repeats: int = 5, warmup: int = 2):
    assert mode in {"pooled", "pairwise"}
    header = f"{'views':>5} | {'time (ms)':>10} | {'peak VRAM (MB)':>15} | {'loss':>12}"
    print(header)
    print("-" * len(header))
    results = []
    for views in views_list:
        logits = torch.randn(BATCH_SIZE, views, NUM_CLASSES, device=DEVICE)
        probs = torch.softmax(logits, dim=-1)
        features = torch.randn(BATCH_SIZE, views, RESNET_FEAT_DIM, device=DEVICE)
        pooled, weights = _aggregate_view_probs(probs, features, strategy="entropy")

        def fn():
            if mode == "pairwise":
                return _js_divergence(probs, view_weights=weights, mode="pairwise")
            return _js_divergence(probs, ref_probs=pooled, view_weights=weights, mode="pooled")

        if DEVICE.type == "cuda":
            torch.cuda.reset_peak_memory_stats(DEVICE)
        out, elapsed = benchmark(fn, repeats=repeats, warmup=warmup)
        peak_vram = _peak_vram_mb()
        results.append({"views": views, "time_ms": elapsed, "vram_mb": peak_vram, "loss": out.item()})
        print(f"[JS {mode}] {views:5d} | {elapsed:10.3f} | {peak_vram:15.3f} | {out.item():12.6f}")
    return results


@pytest.mark.parametrize("mode", ["pooled", "pairwise"])
def test_js_runtime_scaling(mode):
    results = _benchmark_js(mode)
    for row in results:
        assert math.isfinite(row["loss"])


if __name__ == "__main__":
    raise SystemExit(pytest.main(["-s", __file__]))
