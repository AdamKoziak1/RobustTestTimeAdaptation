import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapt_algorithm import _barlow_twins_loss  # noqa: E402
from test_barlow_twins_equivalence import resolve_device, barlow_twins_paper


def resnet18_feature_dim() -> int:
    """Return the flattened feature dimension produced by ResNet-18's avgpool."""
    return 512  # network/img_network.ResBase sets in_features=model_resnet.fc.in_features (512)


def barlow_twins_pairwise(features: torch.Tensor, offdiag_weight: float = 1.0, eps: float = 1e-6) -> torch.Tensor:
    """Apply the two-view reference formula to every pair and average."""
    _, num_views, _ = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    losses = []
    for i, j in combinations(range(num_views), r=2):
        losses.append(barlow_twins_paper(features[:, i], features[:, j], offdiag_weight, eps))
    return torch.stack(losses).mean()


@torch.no_grad()
def benchmark(num_views_list, device, batch_size=64, repeats=10, offdiag_weight=5e-3, eps=1e-6):
    dim = resnet18_feature_dim()
    header = f"{'views':>5} | {'repo impl (ms)':>15} | {'pairwise (ms)':>15} | diff"
    print(header)
    print("-" * len(header))
    for nv in num_views_list:
        features = torch.randn(batch_size, nv, dim, device=device)
        # warm up
        _barlow_twins_loss(features, offdiag_weight=offdiag_weight, eps=eps)
        barlow_twins_pairwise(features, offdiag_weight=offdiag_weight, eps=eps)

        start = time.perf_counter()
        for _ in range(repeats):
            _barlow_twins_loss(features, offdiag_weight=offdiag_weight, eps=eps)
        repo_ms = (time.perf_counter() - start) / repeats * 1e3

        start = time.perf_counter()
        for _ in range(repeats):
            barlow_twins_pairwise(features, offdiag_weight=offdiag_weight, eps=eps)
        pairwise_ms = (time.perf_counter() - start) / repeats * 1e3

        diff = repo_ms - pairwise_ms
        print(f"{nv:5d} | {repo_ms:15.3f} | {pairwise_ms:15.3f} | {diff: .3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Barlow Twins runtime across #views.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device to use, e.g., cpu, cuda, cuda:0 (use 'auto' to pick cuda if available).")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for synthetic tensors.")
    parser.add_argument("--repeats", type=int, default=10, help="Number of timing repetitions.")
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10],
                        help="List of augmentation counts to benchmark.")
    parser.add_argument("--offdiag-weight", type=float, default=5e-3, help="Off-diagonal penalty weight.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical stability epsilon.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    benchmark(args.views, device=device, batch_size=args.batch_size, repeats=args.repeats,
              offdiag_weight=args.offdiag_weight, eps=args.eps)
