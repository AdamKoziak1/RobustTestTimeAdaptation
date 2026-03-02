from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import wandb
except Exception:
    wandb = None

from utils.safer_aug import SAFERAugmenter


VIEW_POOL_STRATEGIES = {"mean", "entropy", "top1", "cc", "cc_drop"}
VIEW_POOL_STRATEGIES_WITH_MATCHING = VIEW_POOL_STRATEGIES | {"matching"}


def _resolve_input_stats(
    mean: Optional[Sequence[float]],
    std: Optional[Sequence[float]],
    modules: Sequence[nn.Module],
) -> tuple[torch.Tensor, torch.Tensor]:
    if mean is not None and std is not None:
        mean_t = torch.tensor(mean, dtype=torch.float32).view(1, -1, 1, 1)
        std_t = torch.tensor(std, dtype=torch.float32).view(1, -1, 1, 1)
        return mean_t, std_t
    for module in modules:
        if hasattr(module, "_in_mean") and hasattr(module, "_in_std"):
            mean_t = module._in_mean.detach().clone().to(dtype=torch.float32)
            std_t = module._in_std.detach().clone().to(dtype=torch.float32)
            return mean_t, std_t
    default_mean = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(1, -1, 1, 1)
    default_std = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(1, -1, 1, 1)
    return default_mean, default_std


def _normalize_input(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return (x - mean) / std


def _denormalize_input(x: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    return x * std + mean


def _detect_normalized_input(x: torch.Tensor, override: Optional[bool] = None) -> bool:
    if override is not None:
        return override
    x_min = x.amin().item()
    x_max = x.amax().item()
    return (x_min < -0.05) or (x_max > 1.05)


def _maybe_denormalize(
    x: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    normalized: bool,
) -> tuple[torch.Tensor, bool]:
    if normalized:
        return _denormalize_input(x, mean, std), True
    return x, False


def _normalize_views(views: torch.Tensor, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
    b, v, c, h, w = views.shape
    flat = views.reshape(-1, c, h, w)
    flat = _normalize_input(flat, mean, std)
    return flat.reshape(b, v, c, h, w)


def _prob_kl_divergence(p: torch.Tensor, q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p = p.clamp_min(eps)
    q = q.clamp_min(eps)
    return (p * (p.log() - q.log())).sum(dim=-1)


def _weighted_mean(values: torch.Tensor, weights: Optional[torch.Tensor], eps: float = 1e-6) -> torch.Tensor:
    if weights is None:
        return values.mean()
    weight_sum = weights.sum().clamp_min(eps)
    return (values * weights).sum() / weight_sum


def _js_divergence(
    probs: torch.Tensor,
    ref_probs: Optional[torch.Tensor] = None,
    view_weights: Optional[torch.Tensor] = None,
    mode: str = "pooled",
    eps: float = 1e-12,
) -> torch.Tensor:
    mode = mode.lower()
    if mode not in {"pooled", "pairwise"}:
        raise ValueError(f"Unknown JS computation mode '{mode}'.")

    if mode == "pairwise":
        bsz, num_views, _ = probs.shape
        if num_views < 2:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        total = probs.new_tensor(0.0)
        weight_total = probs.new_tensor(0.0)
        for i in range(num_views):
            for j in range(i + 1, num_views):
                p, q = probs[:, i], probs[:, j]
                m = 0.5 * (p + q)
                js_pair = 0.5 * _prob_kl_divergence(p, m, eps=eps) + 0.5 * _prob_kl_divergence(q, m, eps=eps)
                if view_weights is not None:
                    pair_weight = view_weights[:, i] * view_weights[:, j]
                    total += (js_pair * pair_weight).sum()
                    weight_total += pair_weight.sum()
                else:
                    total += js_pair.sum()
                    weight_total += js_pair.numel()
        return total / weight_total.clamp_min(eps)

    if ref_probs is None:
        ref_probs = probs.mean(dim=1)
    entropy_each = -torch.xlogy(probs, probs.clamp_min(eps)).sum(dim=-1)
    entropy_mean = -torch.xlogy(ref_probs, ref_probs.clamp_min(eps)).sum(dim=-1)
    if view_weights is None:
        entropy_each = entropy_each.mean(dim=1)
    else:
        entropy_each = (entropy_each * view_weights).sum(dim=1)
    return (entropy_mean - entropy_each).mean()


def _barlow_twins_loss(
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

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        zi = z[:, i]
        for j in range(i + 1, num_views):
            zj = z[:, j]
            c = (zi.T @ zj) / float(bsz)
            c_diff = (c - eye).pow(2)
            diag_loss = torch.diagonal(c_diff).sum()
            offdiag_loss = c_diff.sum() - diag_loss
            loss = loss + diag_loss + offdiag_weight * offdiag_loss
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

    z = features - features.mean(dim=0, keepdim=True)
    norm = torch.sqrt(torch.sum(z ** 2, dim=0) + eps)

    C = torch.einsum("bvi,bwj->vwij", z, z)
    denom = torch.einsum("vi,wj->vwij", norm, norm)
    C = C / (denom + eps)

    loss = features.new_tensor(0.0)
    pairs = 0
    eye = torch.eye(dim, device=features.device, dtype=features.dtype)

    for i in range(num_views):
        for j in range(i + 1, num_views):
            c_ij = C[i, j]
            c_diff = (c_ij - eye).pow(2)
            diag_loss = torch.diagonal(c_diff).sum()
            offdiag_loss = c_diff.sum() - diag_loss
            loss += diag_loss + offdiag_weight * offdiag_loss
            pairs += 1

    return loss / pairs if pairs > 0 else loss


def _barlow_twins_against_pooled(
    features: torch.Tensor,
    pooled: torch.Tensor,
    offdiag_weight: float = 1.0,
    eps: float = 1e-12,
    view_weights: Optional[torch.Tensor] = None,
    impl: str = "fast",
) -> torch.Tensor:
    bsz, num_views, _ = features.shape
    losses = []
    impl_fn = _barlow_twins_loss if impl == "fast" else _barlow_twins_loss_einsum
    for i in range(num_views):
        pair_feats = torch.stack([features[:, i], pooled], dim=1)
        losses.append(impl_fn(pair_feats, offdiag_weight=offdiag_weight, eps=eps))
    view_weights_reduced: Optional[torch.Tensor] = None
    if view_weights is not None:
        view_weights_reduced = view_weights.mean(dim=0)
    return _weighted_mean(torch.stack(losses), view_weights_reduced)


def _cross_view_reliability(features: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    bsz, num_views, _ = features.shape
    if num_views < 2:
        return torch.ones(num_views, device=features.device, dtype=features.dtype)

    z = features - features.mean(dim=0, keepdim=True)
    std = torch.sqrt(torch.mean(z ** 2, dim=0) + eps)
    z = z / (std + eps)

    reliabilities = []
    for i in range(num_views):
        zi = z[:, i]
        pair_scores = []
        for j in range(num_views):
            if i == j:
                continue
            zj = z[:, j]
            diag_corr = (zi * zj).mean(dim=0)
            pair_scores.append(diag_corr.mean())
        reliabilities.append(torch.stack(pair_scores).mean())
    return torch.stack(reliabilities)


def _view_pool_weights(
    probs: torch.Tensor,
    features: Optional[torch.Tensor],
    strategy: str = "mean",
    eps: float = 1e-6,
) -> torch.Tensor:
    bsz, num_views, num_classes = probs.shape
    strategy = strategy.lower()
    if strategy == "mean":
        weights = torch.ones(bsz, num_views, device=probs.device, dtype=probs.dtype)
    elif strategy == "entropy":
        entropy = -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)
        weights = 1.0 - entropy / math.log(num_classes)
    elif strategy == "top1":
        weights = probs.max(dim=-1).values
    elif strategy in {"cc", "cc_drop"}:
        if features is None:
            raise ValueError("Feature tensor is required for cc-based pooling.")
        rel = _cross_view_reliability(features, eps=eps).clamp_min(0.0)
        weights = rel.unsqueeze(0).expand(bsz, -1)
    else:
        raise ValueError(f"Unknown SAFER view pooling strategy '{strategy}'.")

    weights = weights.clamp_min(0.0)
    if strategy == "cc_drop":
        min_idx = weights.argmin(dim=1, keepdim=True)
        weights = weights.clone()
        weights.scatter_(1, min_idx, 0.0)

    zero_mask = weights.sum(dim=1, keepdim=True) <= eps
    weights = torch.where(zero_mask, torch.ones_like(weights), weights)
    weight_sum = weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return weights / weight_sum


def _prediction_entropy(probs: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    return -(probs * torch.log(probs.clamp_min(eps))).sum(dim=-1)


def _top1_margin(probs: torch.Tensor) -> torch.Tensor:
    topk = torch.topk(probs, k=min(2, probs.size(-1)), dim=-1).values
    if topk.size(-1) == 1:
        return topk[..., 0]
    return topk[..., 0] - topk[..., 1]


def _symmetric_prob_divergence(
    p: torch.Tensor,
    q: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    return 0.5 * (
        _prob_kl_divergence(p, q, eps=eps) + _prob_kl_divergence(q, p, eps=eps)
    )


def _compute_alpha_signal_metrics(
    probs: torch.Tensor,
    features: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if probs.ndim != 3 or probs.size(1) < 1:
        raise ValueError("Expected probs with shape (B, V, C) and at least one view.")
    orig_prob = probs[:, 0]
    orig_feat = features[:, 0]

    if probs.size(1) > 1:
        aug_prob = probs[:, 1:].mean(dim=1)
        aug_feat = features[:, 1:].mean(dim=1)
    else:
        aug_prob = orig_prob
        aug_feat = orig_feat

    orig_conf = orig_prob.max(dim=-1).values
    orig_entropy = _prediction_entropy(orig_prob)
    aug_entropy = _prediction_entropy(aug_prob)
    orig_margin = _top1_margin(orig_prob)
    aug_margin = _top1_margin(aug_prob)
    prob_disagreement = _symmetric_prob_divergence(orig_prob, aug_prob)
    feat_cos = F.cosine_similarity(orig_feat, aug_feat, dim=-1).clamp(-1.0, 1.0)
    feat_disagreement = (1.0 - feat_cos).clamp_min(0.0)

    return {
        "orig_conf": orig_conf,
        "orig_margin": orig_margin,
        "aug_margin": aug_margin,
        "orig_entropy": orig_entropy,
        "aug_entropy": aug_entropy,
        "margin_gap": orig_margin - aug_margin,
        "entropy_gap": orig_entropy - aug_entropy,
        "prob_disagreement": prob_disagreement,
        "feat_disagreement": feat_disagreement,
    }


def _adaptive_alpha_from_signal(
    signal: torch.Tensor,
    mode: str,
    threshold: float,
    attack_alpha: float,
    clean_alpha: float,
    attack_when_high_signal: bool = True,
    transition_width: float = 0.1,
    sigmoid_slope: float = 10.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    mode = mode.lower()
    if attack_when_high_signal:
        oriented = signal - threshold
    else:
        oriented = threshold - signal

    if mode in {"step", "fixed_conf_threshold"}:
        attack_score = (oriented >= 0).to(signal.dtype)
    elif mode == "linear":
        width = max(float(transition_width), 1e-6)
        attack_score = (0.5 + (oriented / width)).clamp(0.0, 1.0)
    elif mode == "sigmoid":
        attack_score = torch.sigmoid(float(sigmoid_slope) * oriented)
    else:
        raise ValueError(
            f"Unknown adaptive alpha mode '{mode}'. Expected one of ['none', 'step', 'linear', 'sigmoid']."
        )

    attacked = attack_score >= 0.5
    alpha = clean_alpha + attack_score * (attack_alpha - clean_alpha)
    return alpha.clamp(0.0, 1.0), attack_score, attacked


def _pool_features(
    features: torch.Tensor,
    view_weights: Optional[torch.Tensor],
    eps: float = 1e-6,
) -> torch.Tensor:
    if view_weights is None:
        return features.mean(dim=1)
    weight_sum = view_weights.sum(dim=1, keepdim=True).clamp_min(eps)
    return (features * view_weights.unsqueeze(-1)).sum(dim=1) / weight_sum


class SAFERViewPooler:
    def __init__(
        self,
        probs: torch.Tensor,
        features: torch.Tensor,
        primary_pool: str,
        use_weights: bool = True,
        include_original: bool = True,
        adaptive_alpha_mode: str = "none",
        adaptive_alpha_conf_threshold: float = 0.99,
        adaptive_alpha_attack_value: float = 0.0,
        adaptive_alpha_clean_value: float = 1.0,
        adaptive_alpha_attack_high_conf: bool = True,
        adaptive_alpha_signal: str = "orig_conf",
        adaptive_alpha_transition_width: float = 0.1,
        adaptive_alpha_sigmoid_slope: float = 10.0,
    ) -> None:
        self.probs = probs
        self.features = features
        self.primary_pool = primary_pool
        self.use_weights = bool(use_weights)
        self.include_original = True
        self.adaptive_alpha_mode = adaptive_alpha_mode.lower()
        self.adaptive_alpha_signal_name = adaptive_alpha_signal.lower()
        self._cache: dict[tuple[str, bool], torch.Tensor] = {}
        self.adaptive_alpha: Optional[torch.Tensor] = None
        self.attack_score: Optional[torch.Tensor] = None
        self.alpha_signal: Optional[torch.Tensor] = None
        self.attack_confidence: Optional[torch.Tensor] = None
        self.attack_detected: Optional[torch.Tensor] = None
        metrics = _compute_alpha_signal_metrics(probs, features)
        valid_signals = set(metrics.keys())
        if self.adaptive_alpha_signal_name not in valid_signals:
            raise ValueError(
                f"Unknown adaptive alpha signal '{adaptive_alpha_signal}'. Expected one of {sorted(valid_signals)}."
            )
        self.alpha_signal = metrics[self.adaptive_alpha_signal_name]
        self.orig_confidence = metrics["orig_conf"]
        self.orig_margin = metrics["orig_margin"]
        self.aug_margin = metrics["aug_margin"]
        self.orig_entropy = metrics["orig_entropy"]
        self.aug_entropy = metrics["aug_entropy"]
        self.margin_gap = metrics["margin_gap"]
        self.entropy_gap = metrics["entropy_gap"]
        self.prob_disagreement = metrics["prob_disagreement"]
        self.feat_disagreement = metrics["feat_disagreement"]
        self.attack_confidence = self.orig_confidence

        if self.adaptive_alpha_mode != "none":
            if self.include_original and probs.size(1) > 1:
                alpha, attack_score, attacked = _adaptive_alpha_from_signal(
                    signal=self.alpha_signal,
                    mode=self.adaptive_alpha_mode,
                    threshold=adaptive_alpha_conf_threshold,
                    attack_alpha=adaptive_alpha_attack_value,
                    clean_alpha=adaptive_alpha_clean_value,
                    attack_when_high_signal=adaptive_alpha_attack_high_conf,
                    transition_width=adaptive_alpha_transition_width,
                    sigmoid_slope=adaptive_alpha_sigmoid_slope,
                )
                self.adaptive_alpha = alpha
                self.attack_score = attack_score
                self.attack_detected = attacked

    def _apply_adaptive_alpha(self, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        if self.adaptive_alpha is None or not self.include_original or weights.size(1) < 2:
            return weights
        alpha = self.adaptive_alpha.to(weights.dtype).unsqueeze(1).clamp(0.0, 1.0)
        other = weights[:, 1:].clamp_min(0.0)
        other_sum = other.sum(dim=1, keepdim=True)
        uniform_other = torch.full_like(other, 1.0 / float(other.size(1)))
        other_norm = torch.where(other_sum > eps, other / other_sum.clamp_min(eps), uniform_other)
        mixed = torch.zeros_like(weights)
        mixed[:, 0] = alpha.squeeze(1)
        mixed[:, 1:] = (1.0 - alpha) * other_norm
        return mixed

    def pool(self, strategy: str, apply_alpha: bool = True) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        strategy = strategy.lower()
        if strategy == "matching":
            strategy = self.primary_pool
        cache_key = (strategy, apply_alpha)
        if cache_key in self._cache:
            weights = self._cache[cache_key]
            pooled = (self.probs * weights.unsqueeze(-1)).sum(dim=1)
            return pooled, weights

        weights = None
        if self.use_weights:
            weights = _view_pool_weights(
                probs=self.probs,
                features=self.features,
                strategy=strategy,
            )
        elif apply_alpha and self.adaptive_alpha is not None and self.include_original and self.probs.size(1) > 1:
            bsz, num_views, _ = self.probs.shape
            uniform = torch.ones(bsz, num_views, device=self.probs.device, dtype=self.probs.dtype)
            weights = uniform / uniform.sum(dim=1, keepdim=True)
        if apply_alpha and weights is not None:
            weights = self._apply_adaptive_alpha(weights)
        if weights is None:
            pooled = self.probs.mean(dim=1)
        else:
            pooled = (self.probs * weights.unsqueeze(-1)).sum(dim=1)
            self._cache[cache_key] = weights
        return pooled, weights


@dataclass
class SAFERViewOutput:
    views: torch.Tensor
    logits: torch.Tensor
    probs: torch.Tensor
    features: torch.Tensor
    pooled_prob: torch.Tensor
    pooled_weights: Optional[torch.Tensor]
    loss_pooled_prob: torch.Tensor
    loss_pooled_weights: Optional[torch.Tensor]
    js_loss: torch.Tensor
    cc_loss: torch.Tensor
    pooler: SAFERViewPooler
    adaptive_alpha: Optional[torch.Tensor] = None
    attack_score: Optional[torch.Tensor] = None
    alpha_signal: Optional[torch.Tensor] = None
    attack_confidence: Optional[torch.Tensor] = None
    attack_detected: Optional[torch.Tensor] = None
    orig_confidence: Optional[torch.Tensor] = None
    orig_margin: Optional[torch.Tensor] = None
    aug_margin: Optional[torch.Tensor] = None
    orig_entropy: Optional[torch.Tensor] = None
    aug_entropy: Optional[torch.Tensor] = None
    margin_gap: Optional[torch.Tensor] = None
    entropy_gap: Optional[torch.Tensor] = None
    prob_disagreement: Optional[torch.Tensor] = None
    feat_disagreement: Optional[torch.Tensor] = None


class SAFERViewModule(nn.Module):
    def __init__(
        self,
        num_aug_views: int,
        include_original: bool,
        aug_prob: float,
        aug_max_ops: Optional[int],
        augmentations: Optional[Sequence[str]],
        require_freq_or_blur: bool,
        aug_seed: Optional[int],
        feature_normalize: bool,
        view_weighting: bool,
        primary_view_pool: str,
        js_weight: float,
        js_mode: str,
        js_view_pool: str,
        cc_weight: float,
        cc_mode: str,
        cc_view_pool: str,
        cc_impl: str,
        offdiag_weight: float,
        adaptive_alpha_mode: str = "none",
        adaptive_alpha_conf_threshold: float = 0.99,
        adaptive_alpha_attack_value: float = 0.0,
        adaptive_alpha_clean_value: float = 1.0,
        adaptive_alpha_attack_high_conf: bool = True,
        adaptive_alpha_signal: str = "orig_conf",
        adaptive_alpha_transition_width: float = 0.1,
        adaptive_alpha_sigmoid_slope: float = 10.0,
        mean: Optional[Sequence[float]] = None,
        std: Optional[Sequence[float]] = None,
        input_is_normalized: Optional[bool] = None,
        stat_modules: Optional[Sequence[nn.Module]] = None,
        fixed_ops: Optional[Sequence[str]] = None,
        fixed_op: Optional[str] = None,
        fixed_blur_kernel: Optional[int] = None,
        fixed_blur_sigma: Optional[float] = None,
        fixed_fft_keep_ratio: Optional[float] = None,
        noise_std: float = -1.0,
        debug: bool = False,
        log_pipelines: bool = False,
    ) -> None:
        super().__init__()
        self.include_original = True
        self.num_aug_views = int(num_aug_views)
        self.input_is_normalized = input_is_normalized
        self.feature_normalize = bool(feature_normalize)
        self.use_view_weights = bool(view_weighting)
        primary_view_pool = primary_view_pool.lower()
        self.primary_view_pool = primary_view_pool
        self.js_weight = float(js_weight)
        js_mode = js_mode.lower()
        if js_mode not in {"pooled", "pairwise"}:
            raise ValueError("js_mode must be 'pooled' or 'pairwise'.")
        self.js_mode = js_mode
        js_view_pool = js_view_pool.lower()
        self.js_view_pool = js_view_pool
        self.cc_weight = float(cc_weight)
        cc_mode = cc_mode.lower()
        if cc_mode not in {"pairwise", "pooled"}:
            raise ValueError("cc_mode must be 'pairwise' or 'pooled'.")
        self.cc_mode = cc_mode
        cc_view_pool = cc_view_pool.lower()
        self.cc_view_pool = cc_view_pool
        cc_impl = cc_impl.lower()
        if cc_impl not in {"fast", "einsum"}:
            raise ValueError("cc_impl must be 'fast' or 'einsum'.")
        self.cc_impl = cc_impl
        self.offdiag_weight = float(offdiag_weight)
        self.log_pipelines = bool(log_pipelines)
        adaptive_alpha_mode = adaptive_alpha_mode.lower()
        if adaptive_alpha_mode not in {"none", "fixed_conf_threshold", "step", "linear", "sigmoid"}:
            raise ValueError(
                "adaptive_alpha_mode must be one of ['none', 'fixed_conf_threshold', 'step', 'linear', 'sigmoid']."
            )
        if not (0.0 <= adaptive_alpha_attack_value <= 1.0):
            raise ValueError("adaptive_alpha_attack_value must be in [0, 1].")
        if not (0.0 <= adaptive_alpha_clean_value <= 1.0):
            raise ValueError("adaptive_alpha_clean_value must be in [0, 1].")
        if adaptive_alpha_transition_width <= 0:
            raise ValueError("adaptive_alpha_transition_width must be > 0.")
        if adaptive_alpha_sigmoid_slope <= 0:
            raise ValueError("adaptive_alpha_sigmoid_slope must be > 0.")
        self.adaptive_alpha_mode = adaptive_alpha_mode
        self.adaptive_alpha_conf_threshold = float(adaptive_alpha_conf_threshold)
        self.adaptive_alpha_attack_value = float(adaptive_alpha_attack_value)
        self.adaptive_alpha_clean_value = float(adaptive_alpha_clean_value)
        self.adaptive_alpha_attack_high_conf = bool(adaptive_alpha_attack_high_conf)
        self.adaptive_alpha_signal = adaptive_alpha_signal.lower()
        self.adaptive_alpha_transition_width = float(adaptive_alpha_transition_width)
        self.adaptive_alpha_sigmoid_slope = float(adaptive_alpha_sigmoid_slope)
        self.use_pool_weights_for_losses = self.use_view_weights

        if primary_view_pool not in VIEW_POOL_STRATEGIES:
            raise ValueError(
                f"Unknown view pooling strategy '{primary_view_pool}'. Expected one of {sorted(VIEW_POOL_STRATEGIES)}."
            )
        for name, pool in {
            "js_view_pool": js_view_pool,
            "cc_view_pool": cc_view_pool,
        }.items():
            if pool not in VIEW_POOL_STRATEGIES_WITH_MATCHING:
                raise ValueError(
                    f"Unknown view pooling strategy '{pool}' for {name}. Expected one of {sorted(VIEW_POOL_STRATEGIES_WITH_MATCHING)}."
                )

        aug_views = max(1, self.num_aug_views)
        fixed_params = None
        fixed_ops_list = None
        fixed_ops_params = None
        if fixed_ops is not None and fixed_op is not None:
            raise ValueError("fixed_ops and fixed_op are mutually exclusive.")
        if fixed_ops is not None:
            fixed_ops_list = []
            for op in fixed_ops:
                if op is None:
                    fixed_ops_list.append(None)
                    continue
                op_name = op.lower()
                if op_name == "none":
                    fixed_ops_list.append(None)
                    continue
                if op_name not in {"gaussian_blur", "fft_low_pass"}:
                    raise ValueError("fixed_ops must be one of ['gaussian_blur', 'fft_low_pass', 'none'].")
                fixed_ops_list.append(op_name)
            if self.num_aug_views <= 0:
                raise ValueError("fixed_ops requires num_aug_views > 0.")
            if len(fixed_ops_list) != self.num_aug_views:
                raise ValueError("fixed_ops length must match num_aug_views.")
            params_map = {}
            if "gaussian_blur" in fixed_ops_list and fixed_blur_sigma is not None and fixed_blur_sigma > 0:
                if fixed_blur_kernel is None or fixed_blur_kernel <= 0:
                    raise ValueError("fixed_blur_kernel must be a positive odd integer.")
                if fixed_blur_kernel % 2 == 0:
                    raise ValueError("fixed_blur_kernel must be odd.")
                params_map["gaussian_blur"] = {
                    "kernel_size": int(fixed_blur_kernel),
                    "sigma": float(fixed_blur_sigma),
                }
            if "fft_low_pass" in fixed_ops_list and fixed_fft_keep_ratio is not None and fixed_fft_keep_ratio > 0:
                if not (0.0 < fixed_fft_keep_ratio <= 1.0):
                    raise ValueError("fixed_fft_keep_ratio must be in (0, 1].")
                params_map["fft_low_pass"] = {"keep_ratio": float(fixed_fft_keep_ratio)}
            fixed_ops_params = params_map or None
            fixed_op = None
        else:
            if fixed_op is not None:
                fixed_op = fixed_op.lower()
                if fixed_op == "none":
                    fixed_op = None
            if fixed_op is not None:
                if fixed_op == "gaussian_blur":
                    if fixed_blur_kernel is None or fixed_blur_kernel <= 0:
                        raise ValueError("fixed_blur_kernel must be a positive odd integer.")
                    if fixed_blur_kernel % 2 == 0:
                        raise ValueError("fixed_blur_kernel must be odd.")
                    if fixed_blur_sigma is None:
                        raise ValueError("fixed_blur_sigma must be provided for fixed gaussian blur.")
                    fixed_params = {
                        "kernel_size": int(fixed_blur_kernel),
                        "sigma": float(fixed_blur_sigma),
                    }
                elif fixed_op == "fft_low_pass":
                    if fixed_fft_keep_ratio is None:
                        raise ValueError("fixed_fft_keep_ratio must be provided for fixed FFT low-pass.")
                    fixed_params = {"keep_ratio": float(fixed_fft_keep_ratio)}
                else:
                    raise ValueError("fixed_op must be one of ['gaussian_blur', 'fft_low_pass'].")

        self.augmenter = SAFERAugmenter(
            num_views=aug_views,
            augmentations=augmentations,
            max_ops=aug_max_ops,
            prob=aug_prob,
            seed=aug_seed,
            require_freq_or_blur=require_freq_or_blur,
            fixed_ops=fixed_ops_list,
            fixed_ops_params=fixed_ops_params,
            fixed_op=fixed_op,
            fixed_op_params=fixed_params,
            noise_std=noise_std,
            debug=debug,
            log_pipelines=log_pipelines,
        )

        mean_t, std_t = _resolve_input_stats(mean, std, stat_modules or [])
        self.register_buffer("norm_mean", mean_t)
        self.register_buffer("norm_std", std_t)

    def _build_views(self, x: torch.Tensor) -> torch.Tensor:
        views: list[torch.Tensor] = []
        normalized = _detect_normalized_input(x, self.input_is_normalized)
        x_raw, was_norm = _maybe_denormalize(x, self.norm_mean, self.norm_std, normalized)
        if self.include_original:
            views.append(x.unsqueeze(1))

        if self.num_aug_views > 0:
            aug = self.augmenter.augment(x_raw, num_views=self.num_aug_views)
            if was_norm:
                aug = _normalize_views(aug, self.norm_mean, self.norm_std)
            views.append(aug)
            self._maybe_log_pipelines()

        if not views:
            raise RuntimeError("SAFER could not construct any views.")
        return torch.cat(views, dim=1)

    def _maybe_log_pipelines(self) -> None:
        if not self.log_pipelines:
            return
        lines = self.augmenter.pop_log_lines()
        if not lines:
            return
        payload = "\n".join(lines)
        if wandb is not None and wandb.run is not None:
            summary = wandb.run.summary
            if "SAFER/aug_pipelines" not in summary:
                summary["SAFER/aug_pipelines"] = payload
        else:
            print(payload)

    def _cc_loss(self, feats: torch.Tensor, pooler: SAFERViewPooler) -> torch.Tensor:
        if self.cc_weight <= 1e-8:
            return torch.tensor(0.0, device=feats.device, dtype=feats.dtype)
        if self.cc_mode == "pairwise":
            impl = _barlow_twins_loss if self.cc_impl == "fast" else _barlow_twins_loss_einsum
            return impl(feats, self.offdiag_weight)
        _, cc_weights = pooler.pool(self.cc_view_pool, apply_alpha=False)
        pooled_feats = _pool_features(feats, cc_weights if self.use_pool_weights_for_losses else None)
        return _barlow_twins_against_pooled(
            feats,
            pooled_feats,
            offdiag_weight=self.offdiag_weight,
            eps=1e-12,
            view_weights=cc_weights if self.use_pool_weights_for_losses else None,
            impl=self.cc_impl,
        )

    def _js_loss(self, probs: torch.Tensor, pooler: SAFERViewPooler) -> torch.Tensor:
        if self.js_weight <= 1e-8:
            return torch.tensor(0.0, device=probs.device, dtype=probs.dtype)
        js_prob, js_weights = pooler.pool(self.js_view_pool, apply_alpha=False)
        return _js_divergence(
            probs,
            ref_probs=js_prob if self.js_mode == "pooled" else None,
            view_weights=js_weights if self.use_pool_weights_for_losses else None,
            mode=self.js_mode,
        )

    def forward(
        self,
        x: torch.Tensor,
        model: Optional[nn.Module] = None,
        *,
        featurizer: Optional[nn.Module] = None,
        classifier: Optional[nn.Module] = None,
    ) -> SAFERViewOutput:
        if model is not None:
            featurizer = getattr(model, "featurizer", None)
            classifier = getattr(model, "classifier", None)
        if featurizer is None or classifier is None:
            raise ValueError("SAFERViewModule requires a model with featurizer/classifier or explicit modules.")

        batch_views = self._build_views(x)
        bsz, total_views, c, h, w = batch_views.shape
        flat_input = batch_views.view(bsz * total_views, c, h, w)

        feats = featurizer(flat_input)
        if feats.dim() > 2:
            feats_flat = torch.flatten(feats, start_dim=1)
        else:
            feats_flat = feats

        logits = classifier(feats_flat)
        logits = logits.view(bsz, total_views, -1)
        feats_bt = feats_flat.view(bsz, total_views, -1)
        if self.feature_normalize:
            feats_bt = F.normalize(feats_bt, dim=-1)

        probs = F.softmax(logits, dim=-1)
        pooler = SAFERViewPooler(
            probs=probs,
            features=feats_bt,
            primary_pool=self.primary_view_pool,
            use_weights=self.use_view_weights,
            include_original=self.include_original,
            adaptive_alpha_mode=self.adaptive_alpha_mode,
            adaptive_alpha_conf_threshold=self.adaptive_alpha_conf_threshold,
            adaptive_alpha_attack_value=self.adaptive_alpha_attack_value,
            adaptive_alpha_clean_value=self.adaptive_alpha_clean_value,
            adaptive_alpha_attack_high_conf=self.adaptive_alpha_attack_high_conf,
            adaptive_alpha_signal=self.adaptive_alpha_signal,
            adaptive_alpha_transition_width=self.adaptive_alpha_transition_width,
            adaptive_alpha_sigmoid_slope=self.adaptive_alpha_sigmoid_slope,
        )
        loss_prob, loss_weights = pooler.pool(self.primary_view_pool, apply_alpha=False)
        base_prob, base_weights = pooler.pool(self.primary_view_pool, apply_alpha=True)
        js_loss = self._js_loss(probs, pooler)
        cc_loss = self._cc_loss(feats_bt, pooler)

        return SAFERViewOutput(
            views=batch_views,
            logits=logits,
            probs=probs,
            features=feats_bt,
            pooled_prob=base_prob,
            pooled_weights=base_weights,
            loss_pooled_prob=loss_prob,
            loss_pooled_weights=loss_weights,
            js_loss=js_loss,
            cc_loss=cc_loss,
            pooler=pooler,
            adaptive_alpha=pooler.adaptive_alpha,
            attack_score=pooler.attack_score,
            alpha_signal=pooler.alpha_signal,
            attack_confidence=pooler.attack_confidence,
            attack_detected=pooler.attack_detected,
            orig_confidence=pooler.orig_confidence,
            orig_margin=pooler.orig_margin,
            aug_margin=pooler.aug_margin,
            orig_entropy=pooler.orig_entropy,
            aug_entropy=pooler.aug_entropy,
            margin_gap=pooler.margin_gap,
            entropy_gap=pooler.entropy_gap,
            prob_disagreement=pooler.prob_disagreement,
            feat_disagreement=pooler.feat_disagreement,
        )
