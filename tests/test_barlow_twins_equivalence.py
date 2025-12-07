import argparse
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapt_algorithm import _barlow_twins_loss  # noqa: E402


def resolve_device(device_str: str) -> torch.device:
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device_str)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def off_diagonal(x: torch.Tensor) -> torch.Tensor:
    # Returns a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def barlow_twins_paper(z_a: torch.Tensor, z_b: torch.Tensor, offdiag_weight: float = 1, eps: float = 0, center: bool = True) -> torch.Tensor:
    """
    Args:
        z_a, z_b: tensors of shape [N, D]
    """
    assert z_a.ndim == 2 and z_b.ndim == 2
    assert z_a.shape == z_b.shape
    N, D = z_a.shape

    # Mean-center along the batch dimension (matches paper's assumption)
    if center:
        z_a = z_a - z_a.mean(dim=0, keepdim=True)
        z_b = z_b - z_b.mean(dim=0, keepdim=True)

    # L2 norms per feature across the batch
    norm_a = torch.sqrt(torch.sum(z_a ** 2, dim=0) + eps)  # [D]
    norm_b = torch.sqrt(torch.sum(z_b ** 2, dim=0) + eps)  # [D]

    # Cross-correlation matrix per Eq. 2
    # numerator: [D, D]
    num = z_a.T @ z_b
    # denom: outer product of per-feature norms
    denom = norm_a.unsqueeze(1) * norm_b.unsqueeze(0)
    c = num / denom  # [D, D]

    # Loss per Eq. 1
    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = off_diagonal(c).pow_(2).sum()
    loss = on_diag + offdiag_weight * off_diag
    return loss


def main():
    parser = argparse.ArgumentParser(description="Compare repo Barlow Twins loss with reference implementation.")
    parser.add_argument("--device", default="cpu",
                        help="Torch device, e.g., cpu, cuda, cuda:0 (use 'auto' to pick cuda if available).")
    parser.add_argument("--trials", type=int, default=10, help="Number of random trials.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for random features.")
    parser.add_argument("--dim", type=int, default=512, help="Feature dimension (ResNet-18 avgpool outputs 512).")
    parser.add_argument("--offdiag-weight", type=float, default=5e-3, help="Off-diagonal weight lambda.")
    parser.add_argument("--eps", type=float, default=1e-6, help="Numerical stability epsilon.")
    args = parser.parse_args()

    device = resolve_device(args.device)
    torch.manual_seed(1)
    trials = args.trials
    batch_size = args.batch_size
    dim = args.dim
    offdiag_weight = args.offdiag_weight
    eps = args.eps
    max_diff = 0.0

    for trial in range(trials):
        z_a = torch.randn(batch_size, dim, device=device, dtype=torch.float64)
        z_b = torch.randn(batch_size, dim, device=device, dtype=torch.float64)
        feats = torch.stack([z_a, z_b], dim=1)

        loss_repo = _barlow_twins_loss(feats, offdiag_weight=offdiag_weight, eps=eps)
        loss_paper = barlow_twins_paper(z_a, z_b, offdiag_weight=offdiag_weight, eps=eps)
        diff = (loss_repo - loss_paper).abs().item()
        max_diff = max(max_diff, diff)
        print(f"trial {trial + 1}: repo={loss_repo.item():.6f}, paper={loss_paper.item():.6f}, |diff|={diff:.6e}")
        #assert diff < 1e-6, "Implementation deviates from reference more than tolerance"

    print(f"Maximum absolute difference across trials: {max_diff:.6e}")


if __name__ == "__main__":
    main()
