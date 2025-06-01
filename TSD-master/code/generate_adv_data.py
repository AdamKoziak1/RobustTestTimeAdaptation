# coding=utf‑8
"""
Train per‑domain model, create ℓp‑bounded PGD images,
save *one tensor per image*  (clean & adv)  and print accuracy each epoch.
"""
import argparse, os, random, time, math
import numpy as np
from tqdm import tqdm
import torch, torch.nn as nn, torch.optim as optim
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from utils.util import img_param_init, get_config_id

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

def pgd(model, x, y, eps, alpha, steps, n_classes, norm="linf"):
    y_onehot = nn.functional.one_hot(y, num_classes=n_classes).float()
    x_adv = x.clone().detach() 
    delta = torch.zeros_like(x, requires_grad=True)
    
    softmax = nn.Softmax(dim=1)
    for _ in range(steps):
        x_adv = x_adv.clone().detach().requires_grad_(True)
        preds = softmax(model(x_adv))
        pred  = preds.argmax(1)
        success = pred.ne(y)  
        if success.all():
            break
        loss = nn.CrossEntropyLoss()(preds, y_onehot)
        grad = torch.autograd.grad(loss, x_adv)[0]

        with torch.no_grad():
            if norm == "linf":
                delta = delta + alpha * grad.sign()
                delta = torch.clamp(delta, min=-eps, max=eps)
            else:  # l2
                grad_norm = grad.view(grad.size(0),-1).norm(2,1).view(-1,1,1,1)
                delta = delta + alpha * grad/grad_norm
                delta = delta.renorm(2,0,eps)
            x_adv = (x + delta).clamp(0,1)
    return x_adv.detach()
# ---------------------------------------------------------------------------

def main(cfg):
    seed_everything(cfg.seed)

    tf_train = transforms.Compose([transforms.Resize(224), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    tf_test  = transforms.Compose([transforms.Resize(224), transforms.ToTensor()])

    doms = img_param_init(argparse.Namespace(dataset=cfg.dataset)).img_dataset[cfg.dataset]

    dataset_dir = os.path.join(cfg.output_dir, cfg.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    clean_dir = os.path.join(dataset_dir, "clean")
    os.makedirs(dataset_dir, exist_ok=True)

    configuration_id = get_config_id(cfg)

    attack_config_dir = os.path.join(dataset_dir, configuration_id)
    os.makedirs(attack_config_dir, exist_ok=True)


    for dom in doms:
        print(f"\n=== Domain {dom} ===")
        dom_path = os.path.join(cfg.data_dir, cfg.dataset, dom)

        # ---------- training -------------------------------------------------
        train_set = ImageFolder(dom_path, transform=tf_train)
        train_loader = DataLoader(train_set, batch_size=cfg.bs, shuffle=True, num_workers=cfg.workers)


        test_set = ImageFolder(dom_path, transform=tf_test)
        test_loader = DataLoader(test_set, batch_size=cfg.bs, shuffle=False, num_workers=cfg.workers)
        #test_loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=cfg.workers)
        n_classes = len(train_set.classes)
        model = get_model(cfg.net, n_classes, load_weights=True).to(cfg.dev)
        opt   = optim.SGD(model.parameters(), 0.01, 0.9, weight_decay=5e-4)
        softmax = nn.Softmax(dim=1)
        for epoch in range(cfg.ep):
            model.train()
            running = 0
            for img, lab in tqdm(train_loader, leave=False, desc=f"train‑{epoch}"):
                img, lab = img.to(cfg.dev), lab.to(cfg.dev)
                opt.zero_grad()
                pred = softmax(model(img))
                lab = nn.functional.one_hot(lab, num_classes=n_classes).float()

                loss = nn.CrossEntropyLoss()(model(img), lab)
                loss.backward()
                opt.step()
                running += loss.item()*img.size(0)
            
            model.eval()
            correct = 0
            with torch.no_grad():
                for img, lab in train_loader: # change to val?
                    img, lab = img.to(cfg.dev), lab.to(cfg.dev)
                    pred = model(img).argmax(1)
                    correct += (pred==lab).float().sum().item()
            acc = correct/len(train_set)*100
            print(f"  epoch {epoch}:  loss {running/len(train_set):.4f} | acc {acc:.2f}%")

        domain_clean_dir = os.path.join(clean_dir, dom)
        os.makedirs(domain_clean_dir, exist_ok=True)

        torch.save(model.state_dict(), os.path.join(clean_dir, dom, f"model.pt"))

        attack_config_dom_dir = os.path.join(attack_config_dir, dom)
        os.makedirs(attack_config_dom_dir, exist_ok=True)


        model.eval();   
        ptr = 0
        for img, lab in tqdm(test_loader, desc="attack"):
            img, lab = img.to(cfg.dev), lab.to(cfg.dev)
            bsz = img.size(0)

            img_adv = pgd(model, img, lab, cfg.eps/255., cfg.alpha/255., cfg.steps, n_classes, cfg.attack)

            adv_cursor = 0                   # points into img_adv
            for k in range(bsz):
                gidx = ptr + k               # global index in dataset
                torch.save(img[k].cpu(), os.path.join(domain_clean_dir, f"{gidx}.pt"))
                torch.save(img_adv[adv_cursor].cpu(), os.path.join(attack_config_dom_dir, f"{gidx}.pt"))
                adv_cursor += 1

            ptr += bsz

# ---------------- CLI --------------------------------------------------------
if __name__ == "__main__":
    pa = argparse.ArgumentParser()
    pa.add_argument("--dataset", default="PACS")      
    pa.add_argument("--data_dir", default="../../datasets")
    pa.add_argument("--output_dir", default="../../datasets_adv")
    pa.add_argument("--net", default="resnet18")       
    pa.add_argument("--ep", type=int, default=3)
    pa.add_argument("--bs", type=int, default=64)      
    pa.add_argument("--workers", type=int, default=4)
    pa.add_argument("--attack", choices=["linf","l2"], default="linf")
    pa.add_argument("--eps", type=float, default=8)    
    pa.add_argument("--alpha", type=float, default=2)
    pa.add_argument("--steps", type=int, default=20)  
    pa.add_argument("--seed", type=int, default=0)     
    pa.add_argument("--gpu", default="0")
    #load_weights
    cfg = pa.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=cfg.gpu
    cfg.dev = "cuda:0"
    main(cfg)
