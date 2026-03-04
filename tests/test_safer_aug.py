import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.safer_aug import SAFERAugmenter


def test_require_freq_or_blur_keeps_unchosen_candidate_available():
    augmenter = SAFERAugmenter(
        num_views=1,
        augmentations=["gaussian_blur", "fft_low_pass", "equalize"],
        max_ops=3,
        prob=1.0,
        seed=0,
        require_freq_or_blur=True,
        noise_std=0.0,
    )

    pipeline = augmenter.sample_pipelines()[0]
    op_names = [name for name, _ in pipeline]

    assert len(op_names) == 3
    assert op_names[0] in {"gaussian_blur", "fft_low_pass"}
    assert "gaussian_blur" in op_names
    assert "fft_low_pass" in op_names
    assert "equalize" in op_names


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
