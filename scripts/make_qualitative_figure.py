#!/usr/bin/env python3
"""
Build fig:qualitative - a real attacked PACS:Art sample run through the SAFER
view pipeline.

Loads a batch of precomputed Linf eps=8/255 PGD-20 adversarial PACS:Art
images, runs it through SAFERViewModule on the domain-0 source model, and
finds a batch where the cross-view reliability r_v (a BATCH-level statistic -
see utils/safer_view._cross_view_reliability) is minimized at view 0 (the
unaugmented adversarial input), i.e. the view cc_drop zeroes out for the whole
batch. It then renders one sample from that batch:
  - the clean original
  - the adversarial input actually fed to the model (= view 0)
  - several stochastic augmented views (views 1..N)
  - a bar chart of the batch-level r_v per view, with the dropped view marked

Example:
  python scripts/make_qualitative_figure.py \
      --seed 0 --batch-size 64 --num-views 4 \
      --output sweeps/qualitative_pacs_art.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from alg.alg import get_algorithm_class  # noqa: E402
from utils.util import img_param_init, load_ckpt  # noqa: E402
from utils.safer_view import SAFERViewModule, _cross_view_reliability  # noqa: E402

DOMAIN_LABELS = {
    "PACS": ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS": ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "office-home": ["Art", "Clipart", "Product", "RealWorld"],
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render the SAFER qualitative figure.")
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--dataset", default="PACS")
    parser.add_argument("--domain", type=int, default=0, help="0 = PACS:Art (art_painting).")
    parser.add_argument("--seed", type=int, default=0, help="Source-model seed / adv-stream seed.")
    parser.add_argument("--attack-config", default="resnet18_linf_eps-8.0_steps-20",
                        help="Subdir of datasets_adv/seed_*/<dataset>/ holding precomputed PGD tensors.")
    parser.add_argument("--net", default="resnet18")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-views", type=int, default=4, help="Number of stochastic augmented views (N).")
    parser.add_argument("--num-views-to-show", type=int, default=3,
                        help="How many augmented views to render in the panel (<= num-views).")
    parser.add_argument("--max-tries", type=int, default=40,
                        help="Random batches to try before giving up on finding one where view 0 is dropped.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("sweeps/qualitative_pacs_art.png"))
    parser.add_argument(
        "--reliability-style", choices=["bars", "numeric"], default="bars",
        help="How to display the per-view cross-view reliability r_v: a bar "
             "chart sub-panel ('bars', the original layout) or compact "
             "numerical entries printed under each thumbnail ('numeric', no "
             "extra sub-panel -- more compact).",
    )
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument(
        "--variations-dir", type=Path, default=None,
        help="If set, instead of rendering one hand-picked figure, scan many "
             "random batches/samples for cases where SAFER's stochastic "
             "augmented views either flip the (misclassified) adversarial "
             "prediction back to the true label, or meaningfully reduce "
             "confidence in the wrong label -- and render one figure per "
             "such sample into this directory.",
    )
    parser.add_argument("--num-variations", type=int, default=6,
                        help="How many variation figures to render (best flips first, then largest confidence drops).")
    parser.add_argument("--variation-max-batches", type=int, default=80,
                        help="Random batches to scan when searching for variation candidates.")
    return parser.parse_args()


def _device(args: argparse.Namespace) -> torch.device:
    if args.cpu or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device("cuda")


def _build_source_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    class _Args:
        pass

    a = _Args()
    a.net = args.net
    a.classifier = "linear"
    a.dataset = args.dataset
    a.data_dir = str(args.root / "datasets" / args.dataset)
    img_param_init(a)

    model = get_algorithm_class("ERM")(a)
    ckpt_path = (
        args.root / "train_output" / args.dataset / f"test_{args.domain}"
        / f"seed_{args.seed}" / "model.pkl"
    )
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Source checkpoint not found: {ckpt_path}")
    model = load_ckpt(model, str(ckpt_path))
    model.eval()
    return model.to(device)


def _build_view_module(model: nn.Module, num_views: int) -> SAFERViewModule:
    return SAFERViewModule(
        num_aug_views=num_views,
        include_original=True,
        aug_prob=1.0,
        aug_max_ops=3,
        augmentations=None,
        require_freq_or_blur=True,
        aug_seed=123,
        feature_normalize=False,
        view_weighting=True,
        primary_view_pool="cc_drop",
        js_weight=0.0,
        js_mode="pooled",
        js_view_pool="cc_drop",
        cc_weight=0.0,
        cc_mode="pairwise",
        cc_view_pool="cc_drop",
        cc_impl="fast",
        offdiag_weight=1.0,
        adaptive_alpha_mode="none",
        mean=None,
        std=None,
        input_is_normalized=None,
        stat_modules=(model.featurizer, model),
    )


def _load_indexed_tensor(adv_root: Path, dataset: str, seed: int, config: str, domain: str, idx: int) -> torch.Tensor:
    path = adv_root / f"seed_{seed}" / dataset / config / domain / f"{idx}.pt"
    return torch.load(path, weights_only=True)


def _to_image(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().clamp(0.0, 1.0).permute(1, 2, 0).numpy()


def _find_batch_with_view0_dropped(
    *,
    view_module: SAFERViewModule,
    model: nn.Module,
    domain_dataset: ImageFolder,
    adv_root: Path,
    dataset: str,
    seed: int,
    attack_config: str,
    domain_name: str,
    batch_size: int,
    max_tries: int,
    device: torch.device,
) -> Tuple[List[int], torch.Tensor, torch.Tensor, np.ndarray]:
    """Search random index batches until view 0 (adversarial original) has the
    minimum batch-level cross-view reliability (i.e. cc_drop removes it)."""
    rng = np.random.default_rng(seed)
    n = len(domain_dataset)

    for attempt in range(max_tries):
        indices = rng.choice(n, size=min(batch_size, n), replace=False).tolist()
        x_adv = torch.stack(
            [_load_indexed_tensor(adv_root, dataset, seed, attack_config, domain_name, idx) for idx in indices]
        ).to(device)

        with torch.no_grad():
            output = view_module(x_adv, model)
            r_v = _cross_view_reliability(output.features).clamp_min(0.0)

        dropped_idx = int(r_v.argmin().item())
        print(f"[try {attempt:02d}] r_v = {r_v.cpu().numpy().round(4)}  dropped_view={dropped_idx}")
        if dropped_idx == 0:
            return indices, output.views.detach(), r_v.detach().cpu(), x_adv.detach().cpu().numpy()

    raise RuntimeError(
        f"Could not find a batch where view 0 is dropped within {max_tries} tries. "
        "Try a different --seed or increase --max-tries."
    )


def _predict_label(img: torch.Tensor, model: nn.Module, class_names: List[str], device: torch.device) -> Tuple[str, float]:
    """Return (predicted class name, confidence) for a single CHW image."""
    with torch.no_grad():
        logits = model.predict(img.unsqueeze(0).to(device))
        probs = torch.softmax(logits, dim=-1)[0]
        conf, pred = probs.max(dim=-1)
    return class_names[int(pred.item())], float(conf.item())


def _pick_display_sample(views: torch.Tensor, model: nn.Module, device: torch.device) -> int:
    """Pick the sample whose adversarial (view-0) prediction the source model
    is most confident about, for a visually clean illustrative panel."""
    with torch.no_grad():
        logits = model.predict(views[:, 0].to(device))
        conf = torch.softmax(logits, dim=-1).max(dim=-1).values
    return int(conf.argmax().item())


def _render_qualitative_panel(
    *,
    sample_idx: int,
    true_label: str,
    clean_tensor: torch.Tensor,
    sample_views: torch.Tensor,
    r_v: torch.Tensor,
    batch_size: int,
    domain_id: int,
    class_names: List[str],
    model: nn.Module,
    device: torch.device,
    num_views_to_show: int,
    reliability_style: str,
    output_path: Path,
    dpi: int,
) -> None:
    """Render one clean/adversarial/augmented-views panel with predicted-class
    captions (green if correct, red if not vs. the true label) and either a
    cross-view-reliability bar chart ('bars') or compact numeric r_v captions
    ('numeric'); shared by the single-figure and --variations-dir code paths."""
    n_show = min(num_views_to_show, sample_views.shape[0] - 1)
    class_lookup = class_names

    panels = []
    clean_pred, clean_conf = _predict_label(clean_tensor.to(device), model, class_lookup, device)
    panels.append(("Clean original", _to_image(clean_tensor), None, clean_pred, clean_conf))
    adv_pred, adv_conf = _predict_label(sample_views[0], model, class_lookup, device)
    panels.append((r"Adversarial input ($\ell_\infty$, $\epsilon{=}8/255$)", _to_image(sample_views[0]), 0, adv_pred, adv_conf))
    for v in range(1, n_show + 1):
        v_pred, v_conf = _predict_label(sample_views[v], model, class_lookup, device)
        panels.append((f"Augmented view {v}", _to_image(sample_views[v]), v, v_pred, v_conf))

    dropped_idx = int(r_v.argmin().item())
    correct_color, wrong_color = "#2ca02c", "#d62728"

    def _pred_caption(pred: str, conf: float) -> Tuple[str, str]:
        text = f"pred: {pred} ({conf * 100:.0f}%)"
        return text, (correct_color if pred == true_label else wrong_color)

    n_panels = len(panels)
    suptitle = (
        f"PACS:Art (domain {domain_id}) - sample {sample_idx} (true: {true_label}), "
        f"batch of {batch_size} fully-attacked inputs"
    )

    if reliability_style == "numeric":
        fig = plt.figure(figsize=(2.35 * n_panels, 3.05))
        gs = fig.add_gridspec(1, n_panels, hspace=0.0, wspace=0.08)
        for col, (title, img, rv_idx, pred, conf) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            cap_text, cap_color = _pred_caption(pred, conf)
            lines = [cap_text]
            if rv_idx is not None:
                marker = "  (dropped)" if rv_idx == dropped_idx else ""
                lines.append(rf"$r_v$ = {r_v[rv_idx].item():.3f}{marker}")
            ax.text(
                0.5, -0.07, "\n".join(lines), transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=cap_color,
                fontweight=("bold" if rv_idx == dropped_idx else "normal"),
            )
        fig.suptitle(suptitle, fontsize=8.5, y=1.01)
    else:
        fig = plt.figure(figsize=(2.35 * n_panels, 4.5))
        gs = fig.add_gridspec(2, n_panels, height_ratios=[2.0, 1.3], hspace=0.55, wspace=0.08)
        for col, (title, img, rv_idx, pred, conf) in enumerate(panels):
            ax = fig.add_subplot(gs[0, col])
            ax.imshow(img)
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            cap_text, cap_color = _pred_caption(pred, conf)
            ax.text(
                0.5, -0.07, cap_text, transform=ax.transAxes,
                ha="center", va="top", fontsize=7.5, color=cap_color,
            )

        ax_bar = fig.add_subplot(gs[1, :])
        view_labels = ["original\n(view 0)"] + [f"aug\nview {v}" for v in range(1, r_v.shape[0])]
        colors = ["#d62728" if v == 0 else "#1f77b4" for v in range(r_v.shape[0])]
        bars = ax_bar.bar(range(r_v.shape[0]), r_v.numpy(), color=colors, width=0.6)
        ax_bar.annotate(
            "dropped by cc_drop",
            xy=(dropped_idx, r_v[dropped_idx].item()),
            xytext=(dropped_idx, float(r_v.max().item()) * 1.18 + 1e-3),
            ha="center", fontsize=8, color="#d62728",
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2),
        )
        bars[dropped_idx].set_hatch("//")
        bars[dropped_idx].set_edgecolor("#d62728")
        ax_bar.set_xticks(range(r_v.shape[0]))
        ax_bar.set_xticklabels(view_labels, fontsize=7.5)
        ax_bar.set_ylabel(r"reliability $r_v$", fontsize=8.5)
        ax_bar.set_title("Batch-level cross-view reliability (cc_drop)", fontsize=8.5)
        ax_bar.grid(True, axis="y", linewidth=0.4, alpha=0.35)
        ax_bar.set_ylim(bottom=0.0)
        fig.suptitle(suptitle, fontsize=8.5)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)


def _search_variation_samples(
    *,
    view_module: SAFERViewModule,
    model: nn.Module,
    domain_dataset: ImageFolder,
    adv_root: Path,
    dataset: str,
    seed: int,
    attack_config: str,
    domain_name: str,
    batch_size: int,
    num_views_to_show: int,
    num_wanted: int,
    max_batches: int,
    device: torch.device,
) -> List[dict]:
    """Scan random batches for misclassified-adversarial samples (true label !=
    view-0 prediction) whose stochastic augmented views either (a) FLIP the
    prediction back to the true label, or (b) meaningfully REDUCE confidence
    in the (still-wrong) adversarial label relative to view 0 -- i.e. cases
    that visually demonstrate the view-pooling defense doing real work, rather
    than a 100%-confident wrong prediction passing through every view
    untouched. Flips are ranked above reductions; ties broken by larger
    confidence swing."""
    class_names = domain_dataset.classes
    rng = np.random.default_rng(seed + 9001)
    n = len(domain_dataset)
    seen: set = set()
    candidates: List[dict] = []

    for attempt in range(max_batches):
        indices = rng.choice(n, size=min(batch_size, n), replace=False).tolist()
        x_adv = torch.stack(
            [_load_indexed_tensor(adv_root, dataset, seed, attack_config, domain_name, idx) for idx in indices]
        ).to(device)
        with torch.no_grad():
            output = view_module(x_adv, model)
            r_v_batch = _cross_view_reliability(output.features).clamp_min(0.0).detach().cpu()
        views_batch = output.views.detach()
        n_show = min(num_views_to_show, views_batch.shape[1] - 1)

        for pos, idx in enumerate(indices):
            if idx in seen:
                continue
            true_lbl = class_names[domain_dataset.targets[idx]]
            adv_pred, adv_conf = _predict_label(views_batch[pos, 0], model, class_names, device)
            if adv_pred == true_lbl:
                continue  # adversarial input wasn't even misclassified -- not an interesting case

            best_score, best_kind = 0.0, None
            for v in range(1, n_show + 1):
                v_pred, v_conf = _predict_label(views_batch[pos, v], model, class_names, device)
                if v_pred == true_lbl:
                    score, kind = 1000.0 + v_conf, "flip"
                elif v_pred == adv_pred and v_conf < adv_conf:
                    score, kind = adv_conf - v_conf, "reduced"
                else:
                    continue
                if score > best_score:
                    best_score, best_kind = score, kind

            if best_kind is not None:
                seen.add(idx)
                candidates.append({
                    "sample_idx": idx,
                    "true_label": true_lbl,
                    "sample_views": views_batch[pos].detach().cpu(),
                    "r_v": r_v_batch,
                    "score": best_score,
                    "kind": best_kind,
                    "adv_pred": adv_pred,
                    "adv_conf": adv_conf,
                })

        n_flips = sum(1 for c in candidates if c["kind"] == "flip")
        print(f"[search try {attempt:02d}] candidates so far: {len(candidates)} "
              f"(flips={n_flips}, reduced={len(candidates) - n_flips})")
        if n_flips >= num_wanted or len(candidates) >= num_wanted * 4:
            break

    candidates.sort(key=lambda c: (0 if c["kind"] == "flip" else 1, -c["score"]))
    return candidates[:num_wanted]


def main() -> int:
    args = _parse_args()
    device = _device(args)
    domain_name = DOMAIN_LABELS[args.dataset][args.domain]
    adv_root = args.root / "datasets_adv"
    data_dir = args.root / "datasets" / args.dataset / domain_name

    print(f"[info] device={device} domain={domain_name} attack_config={args.attack_config} seed={args.seed}")

    model = _build_source_model(args, device)
    view_module = _build_view_module(model, args.num_views).to(device)
    domain_dataset = ImageFolder(root=str(data_dir))
    class_names = domain_dataset.classes

    if args.variations_dir is not None:
        candidates = _search_variation_samples(
            view_module=view_module,
            model=model,
            domain_dataset=domain_dataset,
            adv_root=adv_root,
            dataset=args.dataset,
            seed=args.seed,
            attack_config=args.attack_config,
            domain_name=domain_name,
            batch_size=args.batch_size,
            num_views_to_show=args.num_views_to_show,
            num_wanted=args.num_variations,
            max_batches=args.variation_max_batches,
            device=device,
        )
        if not candidates:
            print("[warn] no flip/confidence-reduction candidates found -- "
                  "try a larger --variation-max-batches", file=sys.stderr)
            return 1

        args.variations_dir.mkdir(parents=True, exist_ok=True)
        for i, cand in enumerate(candidates, start=1):
            clean_tensor = _load_indexed_tensor(
                adv_root, args.dataset, args.seed, "clean", domain_name, cand["sample_idx"]
            )
            out_path = args.variations_dir / f"var{i:02d}_{cand['kind']}_idx{cand['sample_idx']}.png"
            _render_qualitative_panel(
                sample_idx=cand["sample_idx"],
                true_label=cand["true_label"],
                clean_tensor=clean_tensor,
                sample_views=cand["sample_views"],
                r_v=cand["r_v"],
                batch_size=args.batch_size,
                domain_id=args.domain,
                class_names=class_names,
                model=model,
                device=device,
                num_views_to_show=args.num_views_to_show,
                reliability_style="numeric",
                output_path=out_path,
                dpi=args.dpi,
            )
            print(
                f"[ok] [{cand['kind']}] sample {cand['sample_idx']} "
                f"(true={cand['true_label']}, adv_pred={cand['adv_pred']} "
                f"@ {cand['adv_conf'] * 100:.0f}%, score={cand['score']:.3f}) -> {out_path}"
            )
        print(f"[ok] wrote {len(candidates)} variation figure(s) to: {args.variations_dir}")
        return 0

    indices, views, r_v, _ = _find_batch_with_view0_dropped(
        view_module=view_module,
        model=model,
        domain_dataset=domain_dataset,
        adv_root=adv_root,
        dataset=args.dataset,
        seed=args.seed,
        attack_config=args.attack_config,
        domain_name=domain_name,
        batch_size=args.batch_size,
        max_tries=args.max_tries,
        device=device,
    )

    sample_pos = _pick_display_sample(views, model, device)
    sample_idx = indices[sample_pos]
    true_label = class_names[domain_dataset.targets[sample_idx]]
    print(f"[ok] displaying sample dataset-index={sample_idx} (batch position {sample_pos}, true label={true_label})")

    clean_tensor = _load_indexed_tensor(adv_root, args.dataset, args.seed, "clean", domain_name, sample_idx)
    sample_views = views[sample_pos]  # (total_views, 3, H, W); view 0 == adversarial input

    _render_qualitative_panel(
        sample_idx=sample_idx,
        true_label=true_label,
        clean_tensor=clean_tensor,
        sample_views=sample_views,
        r_v=r_v,
        batch_size=len(indices),
        domain_id=args.domain,
        class_names=class_names,
        model=model,
        device=device,
        num_views_to_show=args.num_views_to_show,
        reliability_style=args.reliability_style,
        output_path=args.output,
        dpi=args.dpi,
    )
    print(f"[ok] wrote figure: {args.output}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:  # noqa: BLE001
        print(f"[error] {exc}", file=sys.stderr)
        raise SystemExit(1)
