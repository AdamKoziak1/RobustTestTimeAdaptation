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
from alg import alg, modelopera
from alg.opt import *
from utils.util import img_param_init, get_config_id
from datautil.getdataloader import get_img_dataloader_adv, get_img_dataloader
import torch.nn.functional as F

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
        preds = softmax(model.predict(x_adv))
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

def main(args):
    seed_everything(args.seed)
    doms = args.img_dataset[args.dataset]
    #args.output = os.path.join(args.output, args.dataset, str(args.test_envs[0]))


    dataset_dir = os.path.join(args.data_file, "datasets_adv", args.dataset)
    os.makedirs(dataset_dir, exist_ok=True)
    
    clean_dir = os.path.join(dataset_dir, "clean")
    os.makedirs(clean_dir, exist_ok=True)

    configuration_id = get_config_id(args)

    attack_config_dir = os.path.join(dataset_dir, configuration_id)
    os.makedirs(attack_config_dir, exist_ok=True)


    for dom_id, dom in enumerate(doms):
        if dom_id == 0:
            continue
        print(f"\n=== Domain {dom_id} {dom} ===")
        args.test_envs = [dom_id]
        # ---------- training -------------------------------------------------
        train_loader, test_loader = get_img_dataloader_adv(args)
        print(len(train_loader)*args.batch_size, len(test_loader)*args.batch_size)
        
        algorithm_class = alg.get_algorithm_class(args.algorithm)
        model = algorithm_class(args).cuda()
        model.train()
        opt = get_optimizer(model, args)

        n_classes = args.num_classes
        best_acc = 0

        domain_clean_dir = os.path.join(clean_dir, dom)
        os.makedirs(domain_clean_dir, exist_ok=True)

        final_model_path = os.path.join(clean_dir, f"model_{dom}.pt")
        torch.save(model.state_dict(), final_model_path)

        attack_config_dom_dir = os.path.join(attack_config_dir, dom)
        os.makedirs(attack_config_dom_dir, exist_ok=True)
        best_model_path = os.path.join(clean_dir, f"model_{dom}_best.pt")
        for epoch in range(args.ep):
            model.train()
            running = 0
            for img, lab, _ in tqdm(train_loader, leave=False, desc=f"train‑{epoch}"):
                img, lab = img.to(args.dev), lab.to(args.dev)
                opt.zero_grad()
                pred = model.predict(img)
                loss = F.cross_entropy(pred, lab)
                loss.backward()
                opt.step()
                running += loss.item()
            
            acc = modelopera.accuracy(model, test_loader)
            print(f"  epoch {epoch}:  loss {running/len(train_loader):.4f} | test acc {acc:.4f}")

            if acc >= best_acc:
                best_acc = acc
                
                torch.save(model.state_dict(), best_model_path)



        model.load_state_dict(torch.load(best_model_path))
        model.eval();   
        ptr = 0
        for img, lab, _ in tqdm(test_loader, desc="attack"):
            img, lab = img.to(args.dev), lab.to(args.dev)
            bsz = img.size(0)

            img_adv = pgd(model, img, lab, args.eps/255., args.alpha_adv/255., args.steps, n_classes, args.attack)

            adv_cursor = 0                   # points into img_adv
            for k in range(bsz):
                gidx = ptr + k               # global index in dataset
                torch.save(img[k].cpu(), os.path.join(domain_clean_dir, f"{gidx}.pt"))
                torch.save(img_adv[adv_cursor].cpu(), os.path.join(attack_config_dom_dir, f"{gidx}.pt"))
                adv_cursor += 1

            ptr += bsz

# ---------------- CLI --------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=108, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam hyper-param')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='/home/adam/Downloads/RobustTestTimeAdaptation',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='office-home')
    parser.add_argument('--data_dir', type=str, default='datasets', help='data dir')
    parser.add_argument('--dis_hidden', type=int,
                        default=256, help='dis hidden dimension')
    parser.add_argument('--gpu_id', type=str, nargs='?',
                        default='1', help="device id to run")
    parser.add_argument('--groupdro_eta', type=float,
                        default=1, help="groupdro eta")
    parser.add_argument('--inner_lr', type=float,
                        default=1e-2, help="learning rate used in MLDG")
    parser.add_argument('--lam', type=float,
                        default=1, help="tradeoff hyperparameter used in VREx")
    parser.add_argument('--lr', type=float, default=1e-2, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=120, help="max iterations")
    parser.add_argument('--mixupalpha', type=float,
                        default=0.2, help='mixup hyper-param')
    parser.add_argument('--mldg_beta', type=float,
                        default=1, help="mldg hyper-param")
    parser.add_argument('--mmd_gamma', type=float,
                        default=1, help='MMD, CORAL hyper-param')
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase,ViT-B16/32,ViT-L16/32,ViT-H14")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--rsc_f_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--rsc_b_drop_factor', type=float,
                        default=1/3, help='rsc hyper-param')
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
    parser.add_argument('--tau', type=float, default=1, help="andmask tau")
    parser.add_argument('--test_envs', type=int, nargs='+',
                        default=[0], help='target domains')
    parser.add_argument('--opt_type',type=str,default='Adam')  #if want to use Adam, please set Adam
    parser.add_argument('--output', type=str, default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
   
    parser.add_argument("--output_dir", default="datasets_adv")  
    parser.add_argument("--ep", type=int, default=50)
    parser.add_argument("--attack", choices=["linf","l2"], default="linf")
    parser.add_argument("--eps", type=float, default=8)    
    parser.add_argument("--alpha_adv", type=float, default=2)
    parser.add_argument("--steps", type=int, default=20)  
    #load_weights
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_id
    args.dev = "cuda:0"
    
    args.data_dir = os.path.join(args.data_file,args.data_dir,args.dataset)
    args = img_param_init(args)
    main(args)
