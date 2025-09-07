# coding=utf‑8
"""
Visualise a clean image, its adversarial counterpart and the perturbation.
Print ‖δ‖₂ and ‖δ‖_∞  in pixel space + Top‑5 probabilities.

Example
-------
python adv/show_adv_compare.py \
       --clean ../../datasets_adv/seed_0/PACS/clean/art_painting/42.pt \
       --adv   ../../datasets_adv/seed_0/PACS/resnet18_linf_eps-8_steps-20/art_painting/42.pt   \
       --model ../../datasets_adv/seed_0/PACS/clean/model_art_painting_best.pt    \
       --dataset PACS
"""
import argparse, os
import torch, torch.nn.functional as F
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
import numpy as np

# ---------- helpers ----------------------------------------------------------
INV_NORM = transforms.Normalize(mean=[-m/s for m,s in zip([0.485,0.456,0.406],
                                                          [0.229,0.224,0.225])],
                                std =[1/s for s in [0.229,0.224,0.225]])

def unnorm(t):                      # C×H×W → H×W×C in [0,1]
    return INV_NORM(t).mul_(1).clamp_(0,1).permute(1,2,0).cpu().numpy()
def img_form(t):                      # C×H×W → H×W×C in [0,1]
    return t.mul_(1).clamp_(0,1).permute(1,2,0).cpu().numpy()

def load_net(ckpt, arch="resnet18"):
    net = (models.resnet18 if arch=="resnet18" else models.resnet50)(weights=None)
    sd  = torch.load(ckpt, map_location="cpu")
    if isinstance(sd,dict) and "fc.weight" in sd:
        net.fc = torch.nn.Linear(net.fc.in_features, sd["fc.weight"].shape[0])
    net.load_state_dict(sd)
    net.eval(); return net

def top5(probs, labels):
    vals, idx = probs.topk(5)
    for v,i in zip(vals.tolist(), idx.tolist()):
        print(f"{labels[i]:25s}: {v*100:6.2f} %")

# ---------- main -------------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--clean", required=True)
    pa.add_argument("--adv",   required=True)
    pa.add_argument("--model", required=True)
    pa.add_argument("--dataset", required=True)
    args = pa.parse_args()

    # tensors are stored *after* normalisation
    x_clean = torch.load(args.clean)          # shape C×H×W, float32
    x_adv   = torch.load(args.adv)

    # ---- visual -------------------------------------------------------------
    # img_c   = unnorm(x_clean)
    # img_a   = unnorm(x_adv)    
    img_c   = img_form(x_clean)
    img_a   = img_form(x_adv)
    delta   = np.abs(img_a - img_c)           # for display



    # ---- norms in pixel space ----------------------------------------------
    #delta_flat = (img_a - img_c).reshape(-1)
    delta_flat = (x_adv - x_clean).reshape(-1)
    l2  = np.linalg.norm(delta_flat, ord=2)
    linf = np.linalg.norm(delta_flat, ord=np.inf)
    print(f"\n‖δ‖₂  = {l2:.4f}   ‖δ‖_∞ = {linf:.4f}  ( pixel scale 0–1 )")

    # ---- Top‑5 predictions --------------------------------------------------
    # build label map from the original dataset structure
    domain_dir = os.path.basename(os.path.dirname(args.clean)).replace("_clean","")
    root_imgs  = os.path.join("..","..","datasets", args.dataset, domain_dir)
    idx_to_class = {v:k for k,v in ImageFolder(root_imgs).class_to_idx.items()}

    # net = load_net(args.model, "resnet18").eval()
    # with torch.no_grad():
    #     prob_c = F.softmax(net(x_clean.unsqueeze(0)),1)[0]
    #     prob_a = F.softmax(net(x_adv  .unsqueeze(0)),1)[0]

    # print("\nTop‑5 clean:")
    # top5(prob_c, idx_to_class)
    # print("\nTop‑5 adversarial:")
    # top5(prob_a, idx_to_class)
    
    fig, ax = plt.subplots(1,3, figsize=(9,3))
    ax[0].imshow(img_c)
    ax[0].set_title("clean")
    ax[0].axis("off")
    ax[1].imshow(img_a)
    ax[1].set_title("adversarial")
    ax[1].axis("off")
    im = ax[2].imshow(delta.max(-1), cmap="inferno");    # heat‑map
    ax[2].set_title("‖δ‖ per‑pixel")
    ax[2].axis("off")
    fig.colorbar(im, ax=ax[2], fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()