import argparse
import sys
import time
from itertools import combinations
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from adapt_algorithm import _barlow_twins_loss  # noqa: E402

try:
    import matplotlib.pyplot as plt
except ImportError:
    plt = None


def _barlow_twins_loss_fast_alg1_style(
    features: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    bsz, num_views, dim = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    mean = features.mean(dim=0, keepdim=True)
    std  = features.std(dim=0, unbiased=False, keepdim=True)
    z = (features - mean) / (std + eps)

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        zi = z[:, i]
        for j in range(i + 1, num_views):
            zj = z[:, j]
            c = (zi.T @ zj) / float(bsz)

            c_diff = (c - eye).pow(2)
            # scale only off-diagonal entries in-place
            off_diagonal(c_diff).mul_(offdiag_weight)

            loss = loss + c_diff.sum()
            pairs += 1

    return loss / pairs


def _barlow_twins_loss_einsum(
    features: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    bsz, num_views, dim = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True)
    z = (features - mean) / (std + eps)
    z = z.permute(1, 0, 2)  # (V, B, D)

    c = torch.einsum('vbi,wbj->vij', z, z) / bsz  # (V, V, D, D)

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        for j in range(i + 1, num_views):
            c_ij = c[i, j]
            c_diff = (c_ij - eye).pow(2)
            off_diagonal(c_diff).mul_(offdiag_weight)
            loss += c_diff.sum()
            pairs += 1

    return loss / pairs


def _barlow_twins_loss_einsum_v2(
    features: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    bsz, num_views, dim = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)

    mean = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, unbiased=False, keepdim=True)
    z = (features - mean) / (std + eps)
    z = z.permute(1, 0, 2)  # (V, B, D)

    # The line the user requested
    C = torch.einsum('bvi,bwj->vwij', features, features) / bsz

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        for j in range(i + 1, num_views):
            c_ij = C[i, j]
            c_diff = (c_ij - eye).pow(2)
            off_diagonal(c_diff).mul_(offdiag_weight)
            loss += c_diff.sum()
            pairs += 1

    return loss / pairs


def get_gpu_memory_info(device) -> tuple[int, int] | None:
    """Returns (current, peak) memory usage in bytes, or None if not a CUDA device."""
    if device.type != "cuda":
        return None
    torch.cuda.synchronize(device)
    stats = torch.cuda.memory_stats(device)
    return stats["allocated_bytes.all.current"], stats["allocated_bytes.all.peak"]


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


def test_equivalence(device, trials, batch_size, dim, offdiag_weight, eps):
    torch.manual_seed(1)
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

    print(f"Maximum absolute difference across trials: {max_diff:.6e}")


def resnet18_feature_dim() -> int:
    """Return the flattened feature dimension produced by ResNet-18's avgpool."""
    return 512  # network/img_network.ResBase sets in_features=model_resnet.fc.in_features (512)


def barlow_twins_pairwise(features: torch.Tensor, offdiag_weight: float = 1.0, eps: float = 1e-12) -> torch.Tensor:
    """Apply the two-view reference formula to every pair and average."""
    _, num_views, _ = features.shape
    if num_views < 2:
        return torch.tensor(0.0, device=features.device, dtype=features.dtype)
    losses = []
    for i, j in combinations(range(num_views), r=2):
        losses.append(barlow_twins_paper(features[:, i], features[:, j], offdiag_weight, eps))
    return torch.stack(losses).mean()


def plot_results(results):
    if plt is None:
        print("Matplotlib not found. Skipping plot.")
        return

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle('Barlow Twins Implementation Benchmark')

    impls = sorted(list(results[0].keys()))
    views = sorted([r['views'] for r in results])

    for impl in impls:
        if impl == "views":
            continue
        times = [r[impl]['time'] for r in results]
        losses = [r[impl]['loss'] for r in results]
        ax1.plot(views, times, marker='o', linestyle='-', label=impl)
        ax2.plot(views, losses, marker='o', linestyle='-', label=impl)

    ax1.set_xlabel('Number of Views')
    ax1.set_ylabel('Time (ms)')
    ax1.set_title('Runtime vs. Number of Views')
    ax1.legend()
    ax1.grid(True)

    ax2.set_xlabel('Number of Views')
    ax2.set_ylabel('Loss Value')
    ax2.set_title('Loss Value vs. Number of Views')
    ax2.legend()
    ax2.grid(True)
    ax2.set_yscale('log')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()


@torch.no_grad()
def benchmark_runtime(num_views_list, device, batch_size, repeats, offdiag_weight, eps, do_plot=False):
    dim = resnet18_feature_dim()
    
    implementations = {
        "repo": _barlow_twins_loss,
        "pairwise": barlow_twins_pairwise,
        "fast_alg1": _barlow_twins_loss_fast_alg1_style,
        "einsum": _barlow_twins_loss_einsum,
        "einsum_v2": _barlow_twins_loss_einsum_v2,
    }

    header = f"{ 'views':>5} | { 'impl':>10} | { 'time (ms)':>12} | { 'peak VRAM (MB)':>15} | { 'loss_val':>12} | { 'rel_diff':>10}"
    print(header)
    print("-" * len(header))
    
    if device.type != 'cuda':
        print("Warning: Not running on a CUDA device. VRAM usage will be 0.")

    results_for_plot = []

    for nv in num_views_list:
        features = torch.randn(batch_size, nv, dim, device=device)
        
        base_loss = implementations["repo"](features, offdiag_weight=offdiag_weight, eps=eps)
        
        view_results = {'views': nv}

        for name, func in implementations.items():
            # Warm up
            func(features, offdiag_weight=offdiag_weight, eps=eps)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)

            start = time.perf_counter()
            for _ in range(repeats):
                loss_val = func(features, offdiag_weight=offdiag_weight, eps=eps)
            elapsed_ms = (time.perf_counter() - start) / repeats * 1e3

            mem_info = get_gpu_memory_info(device)
            peak_vram_mb = mem_info[1] / 1e6 if mem_info else 0

            rel_diff = (loss_val - base_loss).abs() / base_loss if base_loss != 0 else 0

            print(f"{nv:5d} | {name:>10} | {elapsed_ms:12.3f} | {peak_vram_mb:15.3f} | {loss_val.item():12.6f} | {rel_diff.item():10.4e}")
            
            view_results[name] = {'time': elapsed_ms, 'loss': loss_val.item()}
        
        results_for_plot.append(view_results)
        print("-" * len(header))

    if do_plot:
        plot_results(results_for_plot)


def main():
    parser = argparse.ArgumentParser(description="Run Barlow Twins tests: equivalence and runtime benchmark.")
    parser.add_argument("--test", nargs="+", default=["benchmark"],
                        help="Which tests to run: 'equivalence' or 'benchmark'.")
    parser.add_argument("--device", default="auto",
                        help="Torch device, e.g., cpu, cuda, cuda:0 (use 'auto' to pick cuda if available).")

    # Equivalence test args
    parser.add_argument("--trials", type=int, default=10, help="[Equivalence] Number of random trials.")
    parser.add_argument("--dim", type=int, default=512, help="[Equivalence] Feature dimension.")

    # Benchmark test args
    parser.add_argument("--repeats", type=int, default=10, help="[Benchmark] Number of timing repetitions.")
    parser.add_argument("--views", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10],
                        help="[Benchmark] List of augmentation counts to benchmark.")
    parser.add_argument("--plot", action='store_true', help="Show a plot of the benchmark results.")


    # Common args
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size for synthetic tensors.")
    parser.add_argument("--offdiag-weight", type=float, default=5e-3, help="Off-diagonal penalty weight.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical stability epsilon.")

    args = parser.parse_args()
    device = resolve_device(args.device)

    if "equivalence" in args.test:
        print("--- Running Equivalence Test ---")
        test_equivalence(device=device, trials=args.trials, batch_size=args.batch_size, dim=args.dim,
                         offdiag_weight=args.offdiag_weight, eps=args.eps)
        print("-" * 30)

    if "benchmark" in args.test:
        print("\n--- Running Runtime Benchmark ---")
        benchmark_runtime(num_views_list=args.views, device=device, batch_size=args.batch_size,
                          repeats=args.repeats, offdiag_weight=args.offdiag_weight, eps=args.eps,
                          do_plot=args.plot)
        print("-" * 30)


if __name__ == "__main__":
    main()
