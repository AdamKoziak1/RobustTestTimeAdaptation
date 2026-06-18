from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F

from utils.fft import FFTDrop2D


def _project_linf(x0: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    return (x0 + (x - x0).clamp(min=-eps, max=eps)).clamp(0.0, 1.0)


def _project_l2(x0: torch.Tensor, x: torch.Tensor, eps: float) -> torch.Tensor:
    b = x.size(0)
    delta = (x - x0)
    flat = delta.view(b, -1)
    nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(nrm), eps / nrm)
    delta = (flat * scale).view_as(delta)
    return (x0 + delta).clamp(0.0, 1.0)


def _forward_model(model, x: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "predict"):
        return model.predict(x)
    return model(x)


def build_attack_transform(
    *,
    fft_rho: float,
    fft_alpha: float,
    device: torch.device,
) -> Optional[FFTDrop2D]:
    if fft_rho >= 1.0:
        return None
    transform = FFTDrop2D(
        keep_ratio=fft_rho,
        alpha=fft_alpha,
    ).to(device)
    transform.eval()
    for param in transform.parameters():
        param.requires_grad_(False)
    return transform


def pgd_attack(
    model,
    x_pix: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    *,
    norm: str = "linf",
    input_transform: Optional[torch.nn.Module] = None,
) -> torch.Tensor:
    """
    x_pix: pixel input in [0,1]
    eps, alpha: pixel units (e.g., 8/255, 2/255)
    returns: adversarial example in *pixel* space
    """
    x0 = x_pix.detach()
    x = x0.clone()

    b = x.size(0)
    if norm == "linf":
        delta = torch.empty_like(x).uniform_(-eps, eps)
    elif norm == "l2":
        d = torch.randn_like(x).view(b, -1)
        d = d / d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand(b, 1, device=x.device) * eps
        delta = (d * radius).view_as(x)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")

    x = (x0 + delta).clamp(0.0, 1.0)
    x = _project_linf(x0, x, eps) if norm == "linf" else _project_l2(x0, x, eps)

    for _ in range(steps):
        x.requires_grad_(True)
        x_in = input_transform(x) if input_transform is not None else x
        logits = _forward_model(model, x_in)
        loss = F.cross_entropy(logits, y)
        (g,) = torch.autograd.grad(loss, x, only_inputs=True)

        with torch.no_grad():
            if norm == "linf":
                x = _project_linf(x0, x + alpha * g.sign(), eps)
            else:
                g_flat = g.view(b, -1)
                g_dir = g_flat / g_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                x = _project_l2(x0, x + alpha * g_dir.view_as(g), eps)

    return x.detach()


def pgd_attack_eot(
    model,
    x_pix: torch.Tensor,
    y: torch.Tensor,
    eps: float,
    alpha: float,
    steps: int,
    *,
    norm: str = "linf",
    input_transform: Optional[torch.nn.Module] = None,
    eot_samples: int = 1,
) -> torch.Tensor:
    """Expectation-over-Transformation PGD for defense-aware attacks on SAFER.

    Identical to pgd_attack, except each step's ascent direction is the
    *average* gradient over `eot_samples` independent stochastic forward
    passes through `model`. When `model` is a SAFERPooledPredictor, every
    forward pass samples a fresh set of augmented views, so a single-sample
    gradient (eot_samples=1, i.e. plain pgd_attack) is a noisy estimate of
    "the direction that fools SAFER" - it mostly reflects whichever view
    realization happened to be drawn that step. Averaging over several
    realizations targets the *expected* SAFER-pooled prediction directly,
    which is the standard EOT recipe (Athalye et al., 2018) for attacking
    randomized defenses, and yields perturbations optimized to survive
    SAFER's stochastic view pooling rather than ones that merely overfit to
    one lucky draw of it.

    x_pix: pixel input in [0,1]
    eps, alpha: pixel units (e.g., 8/255, 2/255)
    returns: adversarial example in *pixel* space
    """
    x0 = x_pix.detach()
    x = x0.clone()

    b = x.size(0)
    if norm == "linf":
        delta = torch.empty_like(x).uniform_(-eps, eps)
    elif norm == "l2":
        d = torch.randn_like(x).view(b, -1)
        d = d / d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand(b, 1, device=x.device) * eps
        delta = (d * radius).view_as(x)
    else:
        raise ValueError("norm must be 'linf' or 'l2'")

    x = (x0 + delta).clamp(0.0, 1.0)
    x = _project_linf(x0, x, eps) if norm == "linf" else _project_l2(x0, x, eps)

    k = max(1, int(eot_samples))
    for _ in range(steps):
        x.requires_grad_(True)
        grad_sum = torch.zeros_like(x)
        for _ in range(k):
            x_in = input_transform(x) if input_transform is not None else x
            logits = _forward_model(model, x_in)
            loss = F.cross_entropy(logits, y)
            (g,) = torch.autograd.grad(loss, x, only_inputs=True)
            grad_sum = grad_sum + g
        g = grad_sum / k

        with torch.no_grad():
            if norm == "linf":
                x = _project_linf(x0, x + alpha * g.sign(), eps)
            else:
                g_flat = g.view(b, -1)
                g_dir = g_flat / g_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                x = _project_l2(x0, x + alpha * g_dir.view_as(g), eps)

    return x.detach()
