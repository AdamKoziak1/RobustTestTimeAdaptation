#!/usr/bin/env python3
# svd_spectrum_compare.py
# ---------------------------------------------------------------------------
# Compute mean singular‑value spectra of clean vs attacked images for
# PACS, VLCS and office‑home (all domains) and plot them, plus a percent
# difference curve per singular value.
#
# USAGE
# -----
# python svd_spectrum_compare.py \
#     --data_root      /path/to/datasets \
#     --adv_root       /path/to/datasets_adv \
#     --net            resnet18 \
#     --attack_config  linf_eps-8_steps-20 \
#     --seed           0 \
#     --batch_size     32 \
#     --device         cuda
# ---------------------------------------------------------------------------

import argparse, os, math, torch, numpy as np, matplotlib.pyplot as plt
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from adv.attacked_imagefolder import AttackedImageFolder          # repo local

# ---------------------------------------------------------------------------
DATASETS = {
    "PACS"        : ["art_painting", "cartoon", "photo", "sketch"],
    "VLCS"        : ["Caltech101", "LabelMe", "SUN09", "VOC2007"],
    "office-home" : ["Art", "Clipart", "Product", "RealWorld"],
}

IMG_SIZE   = 224                       # H = W = 224 after resize
N_SINGULAR = IMG_SIZE                  # min(H,W)

@torch.no_grad()
def mean_spectrum(loader, device):
    """Return mean σ‑vector (length 224) across all images in loader."""
    sum_sigma = torch.zeros(N_SINGULAR, device=device)
    cnt       = 0
    for xb, _ in tqdm(loader, leave=False):
        xb = xb.to(device)                     # B × 3 × H × W  in [0,1]
        B  = xb.size(0)
        for c in range(3):                     # loop over RGB channels
            sv = torch.linalg.svdvals(xb[:, c])        # shape: B × 224
            sum_sigma += sv.sum(0)                     # σ already sorted desc
        cnt += 3 * B
    return (sum_sigma / cnt).cpu().numpy()             # length‑224 mean vector

def get_clean_loader(root, batch_size, workers):
    tf = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor()])      # keep raw pixel values
    ds = ImageFolder(root, transform=tf)
    return DataLoader(ds, batch_size=batch_size,
                      num_workers=workers, shuffle=False, pin_memory=True)

def get_attack_loader(root, adv_root, dataset, domain, cfg,
                      rate, seed, batch_size, workers):
    ds = AttackedImageFolder(
        root      = root,
        transform = None,          # tensors saved already → leave as is
        adv_root  = adv_root,
        dataset   = dataset,
        domain    = domain,
        config    = cfg,
        rate      = rate,
        seed      = seed)
    return DataLoader(ds, batch_size=batch_size,
                      num_workers=workers, shuffle=False, pin_memory=True)

# ---------------------------------------------------------------------------
def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data_root",  default="/home/adam/Downloads/RobustTestTimeAdaptation/datasets")
    pa.add_argument("--adv_root",   default="/home/adam/Downloads/RobustTestTimeAdaptation/datasets_adv")
    pa.add_argument("--net",        default="resnet18",
                    help="used only to build <CONFIG> folder names")
    pa.add_argument("--attack_config", default="linf_eps-8_steps-20", choices=["l2_eps-16.0_steps-100", "linf_eps-8_steps-20"])
    pa.add_argument("--seed",       type=int, default=0)
    pa.add_argument("--batch_size", type=int, default=64)
    pa.add_argument("--workers",    type=int, default=4)
    pa.add_argument("--device",     default="cuda", choices=["cuda","cpu"])
    args = pa.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    cfg_id = f"{args.net}_{args.attack_config}"

    for dset, domains in DATASETS.items():
        print(f"\n===== {dset} =====")
        sigma_clean  = torch.zeros(N_SINGULAR)
        sigma_attack = torch.zeros(N_SINGULAR)
        dom_cnt = 0

        for dom in domains:
            print(f"  ↳ {dom} …")
            clean_root = os.path.join(args.data_root, dset, dom)
            attack_root_adv = args.adv_root

            # Use AttackedImageFolder twice: rate=0 → purely clean; rate=100 → all attacked
            clean_loader = get_attack_loader(clean_root, attack_root_adv,
                                             dset, dom, cfg_id,
                                             0, args.seed,
                                             args.batch_size, args.workers)
            attack_loader = get_attack_loader(clean_root, attack_root_adv,
                                              dset, dom, cfg_id,
                                              100, args.seed,
                                              args.batch_size, args.workers)

            sc = mean_spectrum(clean_loader,  device)
            sa = mean_spectrum(attack_loader, device)

            sigma_clean  += torch.from_numpy(sc)
            sigma_attack += torch.from_numpy(sa)
            dom_cnt += 1
            print("     done")

        # average across domains
        sigma_clean  = (sigma_clean  / dom_cnt).numpy()
        sigma_attack = (sigma_attack / dom_cnt).numpy()

        # ---------- plot spectra ------------------------------------------------
        ranks = np.arange(1, N_SINGULAR+1)
        plt.figure(figsize=(6,4))
        plt.semilogy(ranks, sigma_clean,  label="clean")
        plt.semilogy(ranks, sigma_attack, label="attacked")
        plt.xlabel("rank $k$")
        plt.ylabel("mean singular value $\sigma_k$")
        plt.title(f"{dset}: mean spectrum (all domains)")
        plt.grid(alpha=0.3, which="both")
        plt.legend()
        fname = f"svd_spectrum_{dset}.png"
        plt.tight_layout(); plt.savefig(fname, dpi=300)
        print(f"  ↳ figure saved → {fname}")

        # ---------- plot % difference ------------------------------------------
        eps = 1e-12
        pct_diff = 100.0 * (sigma_attack - sigma_clean) / (sigma_clean + eps)
        plt.figure(figsize=(6,4))
        plt.plot(ranks, pct_diff)
        plt.axhline(0.0, linewidth=1.0, linestyle="--")
        plt.xlabel("rank $k$")
        plt.ylabel(r"% difference  $\frac{\sigma^{att}_k-\sigma^{cln}_k}{\sigma^{cln}_k}\times 100$")
        plt.title(f"{dset}: % difference per singular value (attacked vs clean)")
        plt.grid(alpha=0.3)
        fname2 = f"svd_spectrum_{args.attack_config}_pctdiff_{dset}.png"
        plt.tight_layout(); plt.savefig(fname2, dpi=300)
        print(f"  ↳ figure saved → {fname2}")

        # ---------- plot  difference ------------------------------------------
        pct_diff = sigma_clean - sigma_attack
        plt.figure(figsize=(6,4))
        plt.semilogy(ranks, pct_diff)
        plt.axhline(0.0, linewidth=1.0, linestyle="--")
        plt.xlabel("rank $k$")
        plt.ylabel(r"difference  $\sigma^{cln}_k-\sigma^{att}_k$")
        plt.title(f"{dset}: difference per singular value (attacked vs clean)")
        plt.grid(alpha=0.3)
        fname2 = f"svd_spectrum_{args.attack_config}_pct_diff_raw_{dset}.png"
        plt.tight_layout(); plt.savefig(fname2, dpi=300)
        print(f"  ↳ figure saved → {fname2}")
if __name__ == "__main__":
    main()
