# coding=utf‑8
"""
Train per‑domain model, create ℓp‑bounded PGD images,
save *one tensor per image*  (clean & adv)  and print accuracy each epoch.
"""
import argparse, os, random, json, time, math
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
# ----------------------------------------------------------

def seed_everything(seed):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False

def get_model(name, n_cls, load_weights=False):
    weights_init = "IMAGENET1K_V1" if load_weights else None
    m = (models.resnet18 if name=="resnet18" else models.resnet50)(
            weights=weights_init)
    m.fc = nn.Linear(m.fc.in_features, n_cls)
    return m

# -------- PGD (no in‑place ops) --------------------------------------------
def pgd(model, x, y, eps, alpha, steps, norm="linf"):
    x_adv = x.clone().detach()

    for _ in range(steps):
    #while True:
        x_adv = x_adv.clone().detach().requires_grad_(True)
        logits = model(x_adv)
        preds  = logits.argmax(1)
        success = preds.ne(y)  
        #print(logits.shape, preds.shape, y.shape)
        if success.all():
            break
        loss = nn.CrossEntropyLoss()(logits, y)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            if norm == "linf":
                x_adv = x_adv + alpha * grad.sign()
                delta = torch.clamp(x_adv - x, min=-eps, max=eps)
            else:  # l2
                grad_norm = grad.view(grad.size(0),-1).norm(2,1).view(-1,1,1,1)
                x_adv = x_adv + alpha * grad/grad_norm
                delta = x_adv - x
                delta = delta.renorm(2,0,eps)
            x_adv = (x + delta).clamp(0,1)
    return x_adv.detach()
# ---------------------------------------------------------------------------

def main(cfg):
    seed_everything(cfg.seed)

    norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    tf_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), norm])
    tf_test  = transforms.Compose([transforms.Resize(224), transforms.ToTensor(), norm])

    from utils.util import img_param_init
    doms = img_param_init(argparse.Namespace(dataset=cfg.dataset)).img_dataset[cfg.dataset]

    root_out = os.path.join(cfg.output_dir, f"{cfg.dataset}_{cfg.net}_{cfg.attack}{cfg.eps}")
    os.makedirs(root_out, exist_ok=True)

    for dom in doms:
        print(f"\n=== Domain {dom} ===")
        dom_path = os.path.join(cfg.data_dir, dom)

        # ---------- training -------------------------------------------------
        train_set = ImageFolder(dom_path, transform=tf_train)
        train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers)


        test_set = ImageFolder(dom_path, transform=tf_test)
        test_loader = DataLoader(test_set, batch_size=cfg.bs, shuffle=False, num_workers=cfg.workers)

        model = get_model(cfg.net, len(train_set.classes), load_weights=True).to(cfg.dev)
        opt   = optim.SGD(model.parameters(), 0.01, 0.9, weight_decay=5e-4)

        for epoch in range(cfg.ep):
            model.train()
            running = 0
            for img, lab in tqdm(train_loader, leave=False, desc=f"train‑{epoch}"):
                img, lab = img.to(cfg.dev), lab.to(cfg.dev)
                opt.zero_grad()
                loss = nn.CrossEntropyLoss()(model(img), lab)
                loss.backward()
                opt.step()
                running += loss.item()*img.size(0)
            # simple accuracy on *train* (fast) – change loader if you prefer val
            model.eval()
            correct = 0
            with torch.no_grad():
                for img, lab in test_loader:
                    img, lab = img.to(cfg.dev), lab.to(cfg.dev)
                    pred = model(img).argmax(1)
                    correct += (pred==lab).float().sum().item()
            acc = correct/len(train_set)*100
            print(f"  epoch {epoch}:  loss {running/len(train_set):.4f} | acc {acc:.2f}%")

        torch.save(model.state_dict(), os.path.join(root_out, f"{dom}_model.pt"))

        # ---------- generate & save one‑by‑one --------------------------------


        N = len(test_set)
        mask = np.zeros(N, bool)
        mask[np.random.choice(N, int(np.ceil(N*cfg.rate)), replace=False)] = True
        np.save(os.path.join(root_out, f"{dom}_mask.npy"), mask)

        d_adv   = os.path.join(root_out, dom+"_adv");   os.makedirs(d_adv,   exist_ok=True)
        d_clean = os.path.join(root_out, dom+"_clean"); os.makedirs(d_clean, exist_ok=True)

        model.eval();   
        ptr = 0
        for img, lab in tqdm(test_loader, desc="attack"):
            img, lab = img.to(cfg.dev), lab.to(cfg.dev)
            bsz = img.size(0)

            sel = mask[ptr:ptr+bsz]          # NumPy bools for this batch
            if sel.any():
                img_adv = pgd(model, img[sel], lab[sel],
                            cfg.eps/255., cfg.alpha/255., cfg.steps, cfg.attack)

            adv_cursor = 0                   # points into img_adv
            for k in range(bsz):
                gidx = ptr + k               # global index in dataset
                torch.save(img[k].cpu(), os.path.join(d_clean, f"{gidx}.pt"))
                if sel[k]:
                    torch.save(img_adv[adv_cursor].cpu(),
                            os.path.join(d_adv,   f"{gidx}.pt"))
                    adv_cursor += 1

            ptr += bsz

        json.dump(dict(eps=cfg.eps, alpha=cfg.alpha, steps=cfg.steps,
                       rate=cfg.rate, attack=cfg.attack, domain=dom,
                       time=time.asctime()),
                  open(os.path.join(root_out,f"{dom}_meta.json"),"w"))

# ---------------- CLI --------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="PACS")      
    pa.add_argument("--data_dir", default="../../datasets/PACS")
    pa.add_argument("--output_dir", default="../../datasets_adv")
    pa.add_argument("--net", default="resnet18")       
    pa.add_argument("--ep", type=int, default=5)
    pa.add_argument("--bs", type=int, default=64)      
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--attack", choices=["linf","l2"], default="l2")
    pa.add_argument("--eps", type=float, default=8)    
    pa.add_argument("--alpha", type=float, default=2)
    pa.add_argument("--steps", type=int, default=20)   
    pa.add_argument("--rate", type=float, default=1)
    pa.add_argument("--seed", type=int, default=0)     
    pa.add_argument("--gpu", default="0")
    #load_weights
    cfg = pa.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu
    cfg.dev = "cuda:0"
    main(cfg)
