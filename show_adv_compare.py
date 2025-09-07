#!/usr/bin/env python3
# coding=utf-8
"""
Interactive viewer for clean vs adversarial tensors saved by generate_adv_data.py.

Example:
  python show_adv_compare.py \
      --data_root /home/adam/Downloads/RobustTestTimeAdaptation/datasets \
      --adv_root  /home/adam/Downloads/RobustTestTimeAdaptation/datasets_adv \
      --dataset PACS --domain 0 --seed 0 \
      --config resnet18_linf_eps-8.0_steps-20

UI:
  • Buttons: Prev / Random / Next
  • TextBox: enter an integer to jump to that global index
  • Overlays: Top-3 predictions for both clean and adversarial panels

Notes:
  • If --model is not given, the script tries:
      {adv_root}/seed_{seed}/{dataset}/clean/model_{<domain_name>}_best.pt
  • The backbone net is inferred from --config's prefix (e.g., resnet18_*).
  • Predictions run on pixel tensors in [0,1]; the model featurizer handles
    ImageNet normalization through attach_input_standardization().
"""

import argparse, os, sys, re
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox
from torchvision.datasets import ImageFolder

from alg.alg import get_algorithm_class   # ERM
from utils.util import img_param_init

# ------------------------------ Args ------------------------------------------
def parse_args():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root", default="datasets",
                    help="Root of original ImageFolder datasets (for labels).")
    pa.add_argument("--adv_root",  default="datasets_adv",
                    help="Root of saved clean/adv .pt tensors.")
    pa.add_argument("--dataset",   default="PACS", choices=["PACS", "VLCS", "office-home"],
                    help="Dataset name.")
    pa.add_argument("--domain",    default=0, type=int, choices=[0,1,2,3],
                    help="Domain ID in sorted(os.listdir(data_root/dataset)).")
    pa.add_argument("--seed",      type=int, choices=[0,1,2], default=0)
    pa.add_argument("--config",    choices=[
                        "resnet18_linf_eps-8.0_steps-20",
                        "resnet18_l2_eps-112.0_steps-100",
                        "resnet18_linf_eps-8_steps-20",
                        "resnet18_l2_eps-112.0_steps-100"
                     ], default="resnet18_linf_eps-8.0_steps-20",
                     help="Attack configuration folder name under datasets_adv/seed_S/DATASET/…")
    pa.add_argument("--idx",       type=int, default=-1,
                    help="Initial index to show (global dataset index). -1 => random.")
    pa.add_argument("--cmap",      default="inferno", help="Heatmap colormap for |δ|.")
    pa.add_argument("--figsize",   type=float, nargs=2, default=(10.5, 3.6),
                    help="Matplotlib figure size (W H).")
    pa.add_argument("--model",     default=None,
                    help="Path to model_<domain>_best.pt. If omitted, auto-resolve in adv_root/…/clean/")
    pa.add_argument("--cpu",       action="store_true", help="Force CPU (otherwise use CUDA if available).")
    pa.add_argument("--no_label_lookup", action="store_true",
                    help="Skip ImageFolder label resolution (if original dataset not available).")
    pa.add_argument("--topk", type=int, default=3, help="Show top-K predictions per image (default: 3).")
    return pa.parse_args()

# ------------------------------ Utils -----------------------------------------
def _try_load_tensor(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")

def _img_form(t):  # CxHxW -> HxWxC, clamp to [0,1], numpy
    return t.clamp(0,1).permute(1,2,0).cpu().numpy()

def _infer_net_from_config(cfg: str) -> str:
    # assumes "<net>_<attack…>"
    return cfg.split("_")[0]

def _attack_type(cfg: str) -> str:
    m = re.search(r"(linf|l2)", cfg)
    return m.group(1) if m else "linf"

def resolve_domain_name(args):
    base = os.path.join(args.data_root, args.dataset)
    if not os.path.isdir(base):
        raise FileNotFoundError(f"Data root missing: {base}")
    subdirs = sorted([d for d in os.listdir(base) if os.path.isdir(os.path.join(base, d))])
    if args.domain < 0 or args.domain >= len(subdirs):
        raise IndexError(f"Domain id {args.domain} out of range [0,{len(subdirs)-1}] from {base}")
    return subdirs[args.domain]

def collect_indices(clean_dir, adv_dir):
    if not os.path.isdir(clean_dir):
        raise FileNotFoundError(f"Not found: {clean_dir}")
    if not os.path.isdir(adv_dir):
        raise FileNotFoundError(f"Not found: {adv_dir}")

    def grab(dirpath):
        ids = []
        for fn in os.listdir(dirpath):
            if fn.endswith(".pt"):
                stem = fn[:-3]
                try:
                    ids.append(int(stem))
                except:
                    pass
        return set(ids)

    clean_ids = grab(clean_dir)
    adv_ids   = grab(adv_dir)
    common    = sorted(clean_ids & adv_ids)
    if not common:
        raise RuntimeError(f"No overlapping indices between:\n  {clean_dir}\n  {adv_dir}")
    return common

def maybe_build_label_map(data_root, dataset, domain_name, skip=False):
    if skip:
        return None, None
    root_imgs = os.path.join(data_root, dataset, domain_name)
    if not os.path.isdir(root_imgs):
        print(f"[warn] Original dataset path not found for labels: {root_imgs}", file=sys.stderr)
        return None, None
    ds = ImageFolder(root_imgs)
    idx_to_class = {v:k for k,v in ds.class_to_idx.items()}
    labels = np.array([lab for _, lab in ds.imgs], dtype=np.int64)
    return labels, idx_to_class

def default_model_path(adv_root, seed, dataset, domain_name):
    return os.path.join(adv_root, f"seed_{seed}", dataset, "clean", f"model_{domain_name}_best.pt")

def build_model(args, domain_name, device):
    ckpt_path = args.model or default_model_path(args.adv_root, args.seed, args.dataset, domain_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Model checkpoint not found: {ckpt_path}")

    sd = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(sd, dict):
        raise RuntimeError("Unexpected checkpoint format (expected state_dict).")

    # infer num_classes
    if "classifier.fc.weight" in sd:
        num_classes = sd["classifier.fc.weight"].shape[0]
    else:
        fc_keys = [k for k in sd.keys() if k.endswith("fc.weight")]
        if not fc_keys:
            raise RuntimeError("Could not infer num_classes from checkpoint (no *.fc.weight key).")
        num_classes = sd[fc_keys[0]].shape[0]

    # infer backbone from config
    net = _infer_net_from_config(args.config)

    # minimal args for ERM
    class A: pass
    a = A()
    a.net = net
    a.classifier = "linear"
    a.num_classes = num_classes
    a.dataset = args.dataset
    a.data_dir = os.path.join(args.data_root, args.dataset)
    img_param_init(a)

    ERM = get_algorithm_class("ERM")
    model = ERM(a).to(device)
    model.load_state_dict(sd, strict=True)
    model.eval()
    return model

# ------------------------------ Viewer ----------------------------------------
class Viewer:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")

        self.domain_name = resolve_domain_name(args)
        self.clean_dir = os.path.join(args.adv_root, f"seed_{args.seed}", args.dataset, "clean", self.domain_name)
        self.adv_dir   = os.path.join(args.adv_root, f"seed_{args.seed}", args.dataset, args.config, self.domain_name)

        print(f"[setup] clean dir: {self.clean_dir}")
        print(f"[setup]  adv  dir: {self.adv_dir}")

        self.indices = collect_indices(self.clean_dir, self.adv_dir)
        print(f"[info] {len(self.indices)} indices available (intersection of clean & adv).")

        # Labels
        self.labels, self.idx_to_class = maybe_build_label_map(args.data_root, args.dataset, self.domain_name, args.no_label_lookup)

        # Model
        self.model = build_model(args, self.domain_name, self.device)
        print(f"[info] model loaded on {self.device} (backbone: {_infer_net_from_config(args.config)})")

        # RNG + starting pos
        import time
        self.rng = np.random.default_rng(int(time.time()))
        if args.idx >= 0 and args.idx in self.indices:
            self.cur_pos = self.indices.index(args.idx)
        else:
            self.cur_pos = int(self.rng.integers(0, len(self.indices)))

        self.attack = _attack_type(args.config)  # 'linf' or 'l2'

        # Figure + UI
        self.fig, self.ax = plt.subplots(1, 3, figsize=tuple(args.figsize))
        plt.subplots_adjust(bottom=0.18)

        self.ax_prev  = self.fig.add_axes([0.15, 0.06, 0.12, 0.07])
        self.ax_rand  = self.fig.add_axes([0.31, 0.06, 0.12, 0.07])
        self.ax_next  = self.fig.add_axes([0.47, 0.06, 0.12, 0.07])
        self.ax_goto  = self.fig.add_axes([0.67, 0.06, 0.18, 0.07])
        self.btn_prev = Button(self.ax_prev, "Prev")
        self.btn_rand = Button(self.ax_rand, "Random")
        self.btn_next = Button(self.ax_next, "Next")
        self.txt_goto = TextBox(self.ax_goto, "Go to idx ", initial="")

        self.btn_prev.on_clicked(self.on_prev)
        self.btn_rand.on_clicked(self.on_rand)
        self.btn_next.on_clicked(self.on_next)
        self.txt_goto.on_submit(self.on_goto)

        # artists to clear
        self.clean_text_artist = None
        self.adv_text_artist   = None
        self.heat_im_artist    = None
        self.cbar              = None

        self.redraw()

    def _load_pair(self, gidx):
        ct = _try_load_tensor(os.path.join(self.clean_dir, f"{gidx}.pt"))
        at = _try_load_tensor(os.path.join(self.adv_dir,   f"{gidx}.pt"))
        return ct, at

    @torch.no_grad()
    def _topk_pred(self, t, k):
        x = t.unsqueeze(0).to(self.device)   # pixel space [0,1]
        logits = self.model.predict(x)
        probs  = torch.softmax(logits, dim=1)[0]
        vals, idx = torch.topk(probs, k=min(k, probs.numel()))
        ids  = idx.detach().cpu().tolist()
        conf = vals.detach().cpu().tolist()
        return ids, conf

    def _name_for(self, cls_id):
        if self.idx_to_class is not None and cls_id in self.idx_to_class:
            return self.idx_to_class[cls_id]
        return f"id={cls_id}"

    def _gt_name(self, gidx):
        if self.labels is not None and self.idx_to_class is not None:
            if 0 <= gidx < len(self.labels):
                lab_id = int(self.labels[gidx])
                return f"{self.idx_to_class[lab_id]} (id={lab_id})"
        return None

    def _delta_norm(self, clean_t, adv_t):
        df = (adv_t - clean_t).view(-1)
        if self.attack == "linf":
            val = torch.norm(df, p=float('inf')).item()
            return "‖δ‖∞", val
        else:
            val = torch.norm(df, p=2).item()
            return "‖δ‖₂", val

    def _overlay_topk(self, ax, header, ids, conf):
        lines = [header] + [f"  {self._name_for(i)}  ({p*100:.1f}%)" for i, p in zip(ids, conf)]
        return ax.text(
            0.02, 0.02, "\n".join(lines),
            transform=ax.transAxes, ha="left", va="bottom",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.85, edgecolor="black", boxstyle="round,pad=0.25")
        )

    def redraw(self):
        # clear prior texts/colorbar
        for artist in (self.clean_text_artist, self.adv_text_artist):
            if artist is not None:
                try: artist.remove()
                except Exception: pass
        self.clean_text_artist = None
        self.adv_text_artist   = None

        if self.cbar is not None:
            try: self.cbar.remove()
            except Exception: pass
            self.cbar = None

        gidx = self.indices[self.cur_pos]
        clean_t, adv_t = self._load_pair(gidx)

        img_c = _img_form(clean_t)
        img_a = _img_form(adv_t)
        delta = np.abs(img_a - img_c)

        # Top-K predictions
        k = self.args.topk
        ids_c, conf_c = self._topk_pred(clean_t, k)
        ids_a, conf_a = self._topk_pred(adv_t,   k)

        # GT
        gt_name = self._gt_name(gidx)

        # Clean panel
        self.ax[0].cla()
        self.ax[0].imshow(img_c)
        t0 = f"clean (idx {gidx})"
        if gt_name is not None:
            t0 += f"\nGT: {gt_name}"
        self.ax[0].set_title(t0)
        self.ax[0].axis("off")
        self.clean_text_artist = self._overlay_topk(self.ax[0], "Top-3 (clean)", ids_c, conf_c)

        # Adv panel
        self.ax[1].cla()
        self.ax[1].imshow(img_a)
        self.ax[1].set_title("adversarial")
        self.ax[1].axis("off")
        self.adv_text_artist = self._overlay_topk(self.ax[1], "Top-3 (adv)", ids_a, conf_a)

        # Heatmap panel (max over channels)
        self.ax[2].cla()
        self.heat_im_artist = self.ax[2].imshow(delta.max(-1), cmap=self.args.cmap)
        self.ax[2].set_title("‖δ‖ per-pixel")
        self.ax[2].axis("off")
        self.cbar = self.fig.colorbar(self.heat_im_artist, ax=self.ax[2], fraction=0.046, pad=0.04)

        # Title with only the relevant norm
        sym, val = self._delta_norm(clean_t, adv_t)
        self.fig.suptitle(f"{sym} = {val:.4f}   (pixel scale 0–1)   {self._pretty_eps(self.args.config)}", y=0.99, fontsize=10)

        self.fig.canvas.draw_idle()

    def _pretty_eps(self, config):
        m = re.search(r'(l2|linf)_eps-([0-9.]+)', config)
        if not m: return ""
        p = "ℓ∞" if m.group(1) == "linf" else "ℓ2"
        return f"{p} ε={m.group(2)}"

    # --- Callbacks ---
    def on_prev(self, _):
        self.cur_pos = (self.cur_pos - 1) % len(self.indices)
        self.redraw()

    def on_next(self, _):
        self.cur_pos = (self.cur_pos + 1) % len(self.indices)
        self.redraw()

    def on_rand(self, _):
        self.cur_pos = int(self.rng.integers(0, len(self.indices)))
        self.redraw()

    def on_goto(self, text):
        text = text.strip()
        try:
            j = int(text)
        except:
            return
        if j in self.indices:
            self.cur_pos = self.indices.index(j)
            self.redraw()

# ------------------------------ Main ------------------------------------------
def main():
    args = parse_args()
    Viewer(args)
    plt.show()

if __name__ == "__main__":
    main()
