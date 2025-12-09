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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
