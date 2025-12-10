import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapt_algorithm import (
    _aggregate_view_probs,
    _weighted_pseudo_label_loss,
    _js_divergence,
)


def _rand_probs(batch: int = 4, views: int = 3, classes: int = 5):
    logits = torch.randn(batch, views, classes)
    probs = torch.softmax(logits, dim=-1)
    return probs, logits


@pytest.mark.parametrize("strategy", ["mean", "entropy", "top1", "cc", "cc_drop", "worst"])
def test_aggregate_view_probs_shapes_and_weights(strategy):
    probs, _ = _rand_probs()
    features = torch.randn(probs.size(0), probs.size(1), 8)
    pooled, weights = _aggregate_view_probs(probs, features, strategy=strategy)

    assert pooled.shape == (probs.size(0), probs.size(-1))
    assert weights.shape == probs.shape[:2]

    weight_sums = weights.sum(dim=1)
    torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-4, rtol=1e-4)

    if strategy == "worst":
        entropy = -(probs * torch.log(probs.clamp_min(1e-9))).sum(dim=-1)
        worst_idx = entropy.argmax(dim=1)
        pooled_ref = probs[torch.arange(probs.size(0)), worst_idx]
        torch.testing.assert_close(pooled, pooled_ref, atol=1e-5, rtol=1e-5)


def test_weighted_pseudo_label_loss_supports_cc_weights():
    batch, views, classes = 2, 3, 6
    logits = torch.randn(batch, views, classes, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    features = torch.randn(batch, views, 10)

    pooled, weights = _aggregate_view_probs(probs, features, strategy="cc_drop")
    loss = _weighted_pseudo_label_loss(logits, pooled, weights, confidence_scale=False)

    assert loss.ndim == 0 and torch.isfinite(loss).item()
    loss.backward()
    assert logits.grad is not None


def test_aggregate_view_probs_uniform_when_disabled():
    probs, _ = _rand_probs()
    features = torch.randn_like(probs[:, :, :1]).expand(probs.size(0), probs.size(1), 3)
    pooled, weights = _aggregate_view_probs(probs, features, strategy="entropy", use_weights=False)

    expected_weights = torch.full_like(weights, 1.0 / probs.size(1))
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(pooled, probs.mean(dim=1), atol=1e-6, rtol=1e-6)


def test_weighted_pseudo_label_loss_accepts_none_weights():
    batch, views, classes = 2, 4, 7
    logits = torch.randn(batch, views, classes, requires_grad=True)
    probs = torch.softmax(logits, dim=-1)
    pooled = probs.mean(dim=1)
    loss = _weighted_pseudo_label_loss(logits, pooled, view_weights=None, confidence_scale=False)
    loss.backward()
    assert loss.item() >= 0
    assert logits.grad is not None


def test_js_divergence_pairwise_weighted_runs():
    probs, _ = _rand_probs(batch=3, views=4, classes=6)
    # simple entropy-based weights
    pooled, weights = _aggregate_view_probs(probs, torch.randn(probs.size(0), probs.size(1), 3), strategy="entropy")
    loss_pair = _js_divergence(probs, view_weights=weights, mode="pairwise")
    assert torch.isfinite(loss_pair)
    loss_pooled = _js_divergence(probs, ref_probs=pooled, view_weights=weights, mode="pooled")
    assert torch.isfinite(loss_pooled)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
