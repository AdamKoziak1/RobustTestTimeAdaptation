import sys
from pathlib import Path

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.safer_view import SAFERViewPooler


def test_fixed_conf_threshold_alpha_weights_high_conf_attack():
    probs = torch.tensor(
        [
            [
                [0.995, 0.003, 0.002],  # original: high confidence -> attacked
                [0.1, 0.8, 0.1],
                [0.2, 0.2, 0.6],
            ],
            [
                [0.6, 0.2, 0.2],  # original: lower confidence -> clean
                [0.2, 0.7, 0.1],
                [0.3, 0.2, 0.5],
            ],
        ],
        dtype=torch.float32,
    )
    feats = torch.randn(2, 3, 8)
    pooler = SAFERViewPooler(
        probs=probs,
        features=feats,
        primary_pool="mean",
        use_weights=True,
        include_original=True,
        adaptive_alpha_mode="fixed_conf_threshold",
        adaptive_alpha_conf_threshold=0.99,
        adaptive_alpha_attack_value=0.0,
        adaptive_alpha_clean_value=1.0,
        adaptive_alpha_attack_high_conf=True,
    )
    pooled, weights = pooler.pool("mean")
    assert weights is not None

    expected_weights = torch.tensor(
        [
            [0.0, 0.5, 0.5],
            [1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-6)
    expected_pooled = (probs * expected_weights.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(pooled, expected_pooled, atol=1e-6, rtol=1e-6)
    assert pooler.attack_detected is not None
    torch.testing.assert_close(pooler.attack_detected, torch.tensor([True, False]))


def test_fixed_conf_threshold_alpha_applies_when_view_weighting_disabled():
    probs = torch.tensor(
        [
            [
                [0.98, 0.01, 0.01],  # low conf under threshold -> attacked when attack_high_conf=False
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
            ]
        ],
        dtype=torch.float32,
    )
    feats = torch.randn(1, 3, 4)
    pooler = SAFERViewPooler(
        probs=probs,
        features=feats,
        primary_pool="cc_drop",
        use_weights=False,
        include_original=True,
        adaptive_alpha_mode="fixed_conf_threshold",
        adaptive_alpha_conf_threshold=0.99,
        adaptive_alpha_attack_value=0.0,
        adaptive_alpha_clean_value=1.0,
        adaptive_alpha_attack_high_conf=False,
    )
    pooled, weights = pooler.pool("cc_drop")
    assert weights is not None
    expected_weights = torch.tensor([[0.0, 0.5, 0.5]], dtype=torch.float32)
    torch.testing.assert_close(weights, expected_weights, atol=1e-6, rtol=1e-6)
    expected_pooled = (probs * expected_weights.unsqueeze(-1)).sum(dim=1)
    torch.testing.assert_close(pooled, expected_pooled, atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("mode", ["none", "fixed_conf_threshold"])
def test_pooler_weights_sum_to_one(mode):
    probs = torch.softmax(torch.randn(4, 5, 7), dim=-1)
    feats = torch.randn(4, 5, 11)
    pooler = SAFERViewPooler(
        probs=probs,
        features=feats,
        primary_pool="entropy",
        use_weights=True,
        include_original=True,
        adaptive_alpha_mode=mode,
        adaptive_alpha_conf_threshold=0.9,
        adaptive_alpha_attack_value=0.0,
        adaptive_alpha_clean_value=1.0,
        adaptive_alpha_attack_high_conf=True,
    )
    _, weights = pooler.pool("entropy")
    assert weights is not None
    sums = weights.sum(dim=1)
    torch.testing.assert_close(sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
