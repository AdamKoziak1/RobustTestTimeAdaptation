# coding=utf‑8
"""
Show clean vs. adversarial image, run model, print top‑5 probabilities.
Example:
python adv/show_adv_compare.py \
       --clean ../../datasets_adv/PACS_resnet18_linf8/art_painting_clean/42.pt \
       --adv   ../../datasets_adv/PACS_resnet18_linf8/art_painting_adv/42.pt   \
       --model ../../datasets_adv/PACS_resnet18_linf8/art_painting_model.pt
"""
import torch, argparse, matplotlib.pyplot as plt, torch.nn.functional as F
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from PIL import Image
import numpy as np, json, os, sys

norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])

def load_net(ckpt, arch="resnet18"):
    net = (models.resnet18 if arch=="resnet18" else models.resnet50)(weights=None)
    ck = torch.load(ckpt, map_location="cpu")
    n_cls = ck["fc.weight"].shape[0] if isinstance(ck,dict) else None
    if n_cls: net.fc = torch.nn.Linear(net.fc.in_features, n_cls)
    net.load_state_dict(ck if isinstance(ck,dict) else torch.load(ck))
    net.eval(); return net

def show_pair(clean_t, adv_t):
    inv = transforms.Normalize(mean=[-m/s for m,s in zip([0.485,0.456,0.406],[0.229,0.224,0.225])],
                               std=[1/s for s in [0.229,0.224,0.225]])
    c_img, a_img = inv(clean_t).permute(1,2,0).numpy(), inv(adv_t).permute(1,2,0).numpy()
    f,ax = plt.subplots(1,2,figsize=(6,3)); ax[0].imshow(c_img); ax[0].set_title("clean"); ax[0].axis("off")
    ax[1].imshow(a_img); ax[1].set_title("adversarial"); ax[1].axis("off"); plt.tight_layout(); plt.show()

def top5(logits, classes):
    probs = F.softmax(logits,1)[0]; vals,idx = probs.topk(5)
    for v,i in zip(vals.tolist(), idx.tolist()):
        label = classes[i] if isinstance(classes, dict) else classes[i]
        print(f"{str(label):25s}: {v*100:.2f}%")

if __name__ == "__main__":
    pa = argparse.ArgumentParser(); 
    pa.add_argument("--clean")
    pa.add_argument("--adv")
    pa.add_argument("--model")
    pa.add_argument("--dataset")
    args = pa.parse_args()


    root_domain = os.path.dirname(args.clean)          # …/art_painting_clean
    root_dataset = os.path.dirname(root_domain)                         # …/PACS_resnet18_linf8
    domain_name  = os.path.basename(root_domain).replace("_clean","")   # art_painting
    img_root     = os.path.join("..", "..", "datasets", args.dataset, domain_name)  # adjust if needed
    print(root_domain)
    print(root_dataset)
    print(domain_name)
    print(img_root)
    class_to_idx = ImageFolder(img_root).class_to_idx
    idx_to_class = {v:k for k,v in class_to_idx.items()}
    clean = torch.load(args.clean); adv = torch.load(args.adv)
    show_pair(clean, adv)

    ckpt_dir = os.path.dirname(args.model)
    #meta = json.load(open(os.path.join(ckpt_dir, os.path.basename(ckpt_dir).split('_')[-3]+"_meta.json"))) # hacky

    net = load_net(args.model)
    logits_clean = net(clean.unsqueeze(0))
    logits_adv   = net(adv.unsqueeze(0))

    print("\nTop‑5 clean:")
    top5(logits_clean, idx_to_class)
    print("\nTop‑5 adversarial:")
    top5(logits_adv, idx_to_class)
