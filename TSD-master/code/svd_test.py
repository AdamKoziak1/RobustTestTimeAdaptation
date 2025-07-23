#!/usr/bin/env python3
# svd_linf_analysis_once.py  (batched SVD – one per image)
# -------------------------------------------------------------------
#  ▸ Compute SVD once           ▸ reconstructions for all k
#  ▸ For first attacked image:  ▸ big labeled grid every 12th k
#  ▸ Plot ℓ∞ curves + 8/255 ref line
# -------------------------------------------------------------------
import argparse, os, torch, numpy as np, matplotlib.pyplot as plt, math
from tqdm import tqdm
from adv.attacked_imagefolder import AttackedImageFolder  # ← your repo

plt.rcParams['savefig.bbox'] = 'tight'

def reconstructions_from_svd(U, S, Vh, k_values):
    """Return dict {k: reconstruction} for each k in k_values."""
    out = {}
    for k in k_values:
        rec_chan = []
        for u, s, vh in zip(U, S, Vh):
            s_k = s.clone()
            if k > 0: s_k[-k:] = 0
            rec_chan.append((u * s_k) @ vh)
        out[k] = torch.stack(rec_chan)  # 3×H×W
    return out

def save_labeled_grid(tensors, ks, fname, cmap=None):
    """
    tensors: list of 3×H×W (RGB) or broadcast heat maps
    ks:      matching list of k values
    fname:   output PNG path
    cmap:    e.g. 'inferno' for heat maps, else None
    """
    N = len(tensors)
    cols = 6
    rows = math.ceil(N / cols)
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = axs.flatten()
    for ax in axs[N:]:
        ax.axis('off')
    for i, (img, k) in enumerate(zip(tensors, ks)):
        ax = axs[i]
        arr = img.permute(1,2,0).cpu().numpy()
        if cmap:
            # assume single‑channel broadcast in the first channel
            ax.imshow(arr[:,:,0], cmap=cmap)
        else:
            ax.imshow(arr)
        ax.set_title(f'k={k}', fontsize=8)
        ax.axis('off')
    fig.tight_layout(pad=1.0)
    fig.savefig(fname, dpi=250)
    plt.close(fig)

@torch.no_grad()
def main():
    pa = argparse.ArgumentParser()
    # ---------- paths & params ----------
    pa.add_argument('--data_root', default='../../datasets')
    pa.add_argument('--adv_root',  default='../../datasets_adv')
    pa.add_argument('--dataset',   default='PACS')
    pa.add_argument('--domain',    default='photo',
                    choices=['art_painting', 'cartoon', 'photo', 'sketch'])
    pa.add_argument('--config',    default='resnet18_linf_eps-8_steps-20')
    pa.add_argument('--rate',      type=int, default=100)
    pa.add_argument('--seed',      type=int, default=0)
    # ---------- analysis ----------------
    pa.add_argument('--drop_k', type=int, nargs='+',
                    default=list(range(0, 224, 12)))
    pa.add_argument('--plot_png', default='svd_linf_plot.png')
    args = pa.parse_args()

    ds = AttackedImageFolder(
        root=os.path.join(args.data_root, args.dataset, args.domain),
        transform=None,
        adv_root=args.adv_root,
        dataset=args.dataset,
        domain=args.domain,
        config=args.config,
        rate=args.rate,
        seed=args.seed
    )
    atk_ids = np.where(ds.mask)[0]
    print(f'Analysing {len(atk_ids)} attacked tensors.')

    linf_att, linf_cln = [], []  # we'll index by k-position
    drop_ks = args.drop_k

    first_done = False
    for idx in tqdm(atk_ids):
        name = f'{idx}.pt'
        adv = torch.load(os.path.join(ds.adv_root, name))
        cln = torch.load(os.path.join(ds.cln_root, name))

        # ---- SVD once per channel ----
        U, S, Vh = zip(*(torch.linalg.svd(adv[c], full_matrices=False) for c in range(3)))
        Uc, Sc, Vhc = zip(*(torch.linalg.svd(cln[c], full_matrices=False) for c in range(3)))

        rec_adv = reconstructions_from_svd(U,  S,  Vh,  drop_ks)
        rec_cln = reconstructions_from_svd(Uc, Sc, Vhc, drop_ks)

        # init lists on first pass
        if not linf_att:
            linf_att = {k: [] for k in drop_ks}
            linf_cln = {k: [] for k in drop_ks}

        # ---- accumulate ℓ∞ norms ----
        for k in drop_ks:
            linf_att[k].append((adv - rec_adv[k]).abs().max().item())
            linf_cln[k].append((cln - rec_cln[k]).abs().max().item())

        # ---- save grid for first image ----
        if not first_done:
            rec_list  = [rec_adv[k] for k in drop_ks]
            diff_list = [ (adv - rec_adv[k]).abs().max(0).values.expand(3,-1,-1)
                           for k in drop_ks ]
            save_labeled_grid(rec_list,  drop_ks, f'svd_recon_grid{idx}.png')
            save_labeled_grid(diff_list, drop_ks,f'svd_diff_grid{idx}.png', cmap='inferno')
            if idx > 4:
                first_done = True

    # ---- summarise & plot ----
    ks = drop_ks
    mean_att = [np.mean(linf_att[k]) for k in ks]
    mean_cln = [np.mean(linf_cln[k]) for k in ks]

    plt.figure(figsize=(6,4))
    plt.plot(ks, mean_att, marker='o', label='Attacked')
    plt.plot(ks, mean_cln, marker='s', label='Clean', linestyle='--')
    plt.axhline(8/255, color='red', linestyle=':', label='8/255 threshold')
    plt.xlabel('smallest singular values dropped')
    plt.ylabel(r'average $\ell_\infty$ norm (0–1 pixel scale)')
    plt.title(f'{args.dataset}/{args.domain}: SVD denoising')
    plt.grid(alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(args.plot_png, dpi=250)
    print(f'\nPlot saved → {args.plot_png}')

    # ---- raw table ----
    print('\nAverage ℓ∞ per image')
    for k,a,c in zip(ks, mean_att, mean_cln):
        print(f'  drop {k:3d}:  attacked={a:.6f}   clean={c:.6f}')

if __name__ == '__main__':
    main()
