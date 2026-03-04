import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from adapt_algorithm import (
    _aggregate_view_probs,
    _barlow_twins_loss,
    _weighted_pseudo_label_loss,
    _js_divergence,
)


def _rand_probs(batch: int = 4, views: int = 3, classes: int = 5):
    logits = torch.randn(batch, views, classes)
    probs = torch.softmax(logits, dim=-1)
    return probs, logits


@pytest.mark.parametrize("strategy", ["mean", "entropy", "top1", "cc", "cc_drop"])
def test_aggregate_view_probs_shapes_and_weights(strategy):
    probs, _ = _rand_probs()
    features = torch.randn(probs.size(0), probs.size(1), 8)
    pooled, weights = _aggregate_view_probs(probs, features, strategy=strategy)

    assert pooled.shape == (probs.size(0), probs.size(-1))
    assert weights.shape == probs.shape[:2]

    weight_sums = weights.sum(dim=1)
    torch.testing.assert_close(weight_sums, torch.ones_like(weight_sums), atol=1e-4, rtol=1e-4)


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


def test_js_divergence_pairwise_normalizes_by_pair_weight_sum():
    probs = torch.tensor(
        [
            [[0.80, 0.20], [0.55, 0.45], [0.25, 0.75]],
            [[0.60, 0.40], [0.20, 0.80], [0.90, 0.10]],
        ],
        dtype=torch.float64,
    )
    weights = torch.tensor(
        [
            [0.20, 0.30, 0.50],
            [0.10, 0.60, 0.30],
        ],
        dtype=torch.float64,
    )
    eps = 1e-12

    actual = _js_divergence(probs, view_weights=weights, mode="pairwise", eps=eps)

    weighted_sum = probs.new_tensor(0.0)
    pair_weight_sum = probs.new_tensor(0.0)
    for i in range(probs.size(1)):
        for j in range(i + 1, probs.size(1)):
            p = probs[:, i]
            q = probs[:, j]
            m = 0.5 * (p + q)
            kl_pm = (p * (p.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
            kl_qm = (q * (q.clamp_min(eps).log() - m.clamp_min(eps).log())).sum(dim=-1)
            js_pair = 0.5 * (kl_pm + kl_qm)
            pair_weight = weights[:, i] * weights[:, j]
            weighted_sum += (js_pair * pair_weight).sum()
            pair_weight_sum += pair_weight.sum()

    expected = weighted_sum / pair_weight_sum
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


def test_barlow_twins_pairwise_normalizes_by_number_of_pairs():
    features = torch.tensor(
        [
            [[1.0, 0.2, 1.5], [0.4, 1.2, 0.3], [1.4, 0.5, 0.9]],
            [[0.7, 1.1, 0.8], [1.5, 0.2, 1.0], [0.6, 1.3, 1.4]],
            [[1.3, 0.4, 0.6], [0.8, 1.4, 1.2], [1.1, 0.7, 0.2]],
            [[0.2, 1.5, 1.1], [1.1, 0.6, 0.5], [0.9, 1.0, 1.3]],
        ],
        dtype=torch.float64,
    )
    offdiag_weight = 0.25
    eps = 1e-12

    actual = _barlow_twins_loss(features, offdiag_weight=offdiag_weight, eps=eps)

    bsz, num_views, dim = features.shape
    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True)
    z = (features - mean) / (std + eps)
    eye = torch.eye(dim, dtype=features.dtype)

    pair_loss_sum = features.new_tensor(0.0)
    num_pairs = 0
    for i in range(num_views):
        for j in range(i + 1, num_views):
            c = (z[:, i].T @ z[:, j]) / float(bsz)
            c_diff = (c - eye).pow(2)
            diag_loss = torch.diagonal(c_diff).sum()
            offdiag_loss = c_diff.sum() - diag_loss
            pair_loss_sum += diag_loss + offdiag_weight * offdiag_loss
            num_pairs += 1

    expected = pair_loss_sum / num_pairs
    torch.testing.assert_close(actual, expected, atol=1e-12, rtol=1e-12)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
