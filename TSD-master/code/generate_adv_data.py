# coding=utf‑8
"""
Train per‑domain model, create ℓp‑bounded PGD images,
save *one tensor per image*  (clean & adv)  and print accuracy each epoch.
"""
import argparse, os
from tqdm import tqdm
import torch, torch.nn as nn

from alg import alg, modelopera
from alg.opt import *
from datautil.getdataloader import get_img_dataloader_adv
import torch.nn.functional as F

from utils.util import set_random_seed, img_param_init, print_environ, img_param_init, get_config_id
import wandb

def get_args_adv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ERM")
    parser.add_argument('--alpha', type=float,
                        default=1, help='DANN dis alpha')
    parser.add_argument('--anneal_iters', type=int,
                        default=500, help='Penalty anneal iters used in VREx')
    parser.add_argument('--batch_size', type=int,
                        default=32, help='batch_size')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='Adam hyper-param')
    parser.add_argument('--checkpoint_freq', type=int,
                        default=3, help='Checkpoint every N epoch')
    parser.add_argument('--classifier', type=str,
                        default="linear", choices=["linear", "wn"])
    parser.add_argument('--data_file', type=str, default='/home/adam/Downloads/RobustTestTimeAdaptation',
                        help='root_dir')
    parser.add_argument('--dataset', type=str, default='office')
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
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=50, help="max iterations")
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
    parser.add_argument('--output', type=str,
                        default="datasets_adv", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=0)
   
    parser.add_argument("--attack", choices=["linf","l2"], default="linf")
    parser.add_argument("--eps", type=float, default=8)    
    parser.add_argument("--alpha_adv", type=float, default=2)
    parser.add_argument("--steps", type=int, default=20)  

    args = parser.parse_args()
    args.steps_per_epoch = 100
    
    args.data_dir = os.path.join(args.data_file, args.data_dir, args.dataset)

    args = img_param_init(args)
    print_environ()
    return args

@torch.no_grad()
def _linf_project(x0, x, eps):
    # project x back to L_inf ball around x0 and [0,1] box
    return (x0 + (x - x0).clamp(min=-eps, max=eps))

def pgd(model, x, y, eps, alpha, steps, n_classes, norm="linf"):
    x_adv = x.clone().detach() 
    b = x.size(0)
    delta = torch.zeros_like(x)
    if norm == "linf":
        delta.uniform_(-eps, eps)
    elif norm == "l2":
        delta = torch.randn_like(x)
        b = x.size(0)
        delta_flat = delta.view(b, -1)
        delta_flat = delta_flat / (delta_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12))
        r = torch.rand(b, 1, device=x.device)   # <-- match 2D shape for broadcast
        delta_flat = delta_flat * (r * eps)
        delta = delta_flat.view_as(x)

    x_adv = (x + delta)

    #print()
    for _ in range(steps):
        x_in = x_adv
        x_in.requires_grad_(True)

        logits = model.predict(x_in)
        loss = F.cross_entropy(logits, y)

        (grad,) = torch.autograd.grad(loss, x_in, only_inputs=True)

        preds = logits.argmax(1)
        fooled = preds.ne(y) 
        #print(fooled.float().mean().item()) 

        with torch.no_grad():
            if norm == "linf":
                step = alpha * grad.sign()
                x_adv = _linf_project(x, x_adv + step, eps)
            elif norm == "l2":
                g = grad
                g_flat = g.view(b, -1)
                gnorm = g_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                g_dir = (g_flat / gnorm).view_as(g)          # unit L2 direction per sample
                step = alpha * g_dir
                x_adv = (x_adv + step)
    return x_adv.detach()

# ---------------- CLI --------------------------------------------------------
if __name__ == "__main__":
    args = get_args_adv()
    set_random_seed(args.seed)
    
    wandb.init(
        project="tta3_train_attack",         # ← change to your project
        name=f"{args.dataset}_test-env-{args.test_envs[0]}_s{args.seed}",  # run name in W&B
        config=vars(args),                   # log all hyperparameters
    )

    dom_id = args.test_envs[0]
    dom = args.img_dataset[args.dataset][dom_id]

    dataset_dir = os.path.join(args.data_file, "datasets_adv", f"seed_{str(args.seed)}", args.dataset)
    
    clean_dir = os.path.join(dataset_dir, "clean")

    domain_clean_dir = os.path.join(clean_dir, dom)
    os.makedirs(domain_clean_dir, exist_ok=True)

    configuration_id = get_config_id(args)
    attack_config_dom_dir = os.path.join(dataset_dir, configuration_id, dom)
    os.makedirs(attack_config_dom_dir, exist_ok=True)
    best_model_path = os.path.join(clean_dir, f"model_{dom}_best.pt")
    

    # ---------- training -------------------------------------------------
    train_loader, val_loader, attack_loader = get_img_dataloader_adv(args)
    
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    model = algorithm_class(args).cuda()
    model.train()
    opt = get_optimizer(model, args)
    sch = get_scheduler(opt, args)

    n_classes = args.num_classes
    best_acc = 0

    train_minibatches_iterator = zip(*train_loader)
    for epoch in range(args.max_epoch):
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            
            step_vals = model.update(minibatches_device, opt, sch)

            wandb.log({
                f"train_loss": step_vals.get('class', step_vals.get('total', None)),
                "epoch": epoch,
            }, commit=False)

        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            val_acc = modelopera.accuracy(model, val_loader)

            if val_acc >= best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), best_model_path)
            wandb.log({
                "epoch": epoch,
                "lr": opt.param_groups[0]['lr'],
                "acc_val": val_acc,
            }, commit=True)

    print('valid acc: %.4f' % best_acc)
    wandb.summary["best_valid_acc"] = best_acc

    # ---------- attacking -------------------------------------------------
    model.load_state_dict(torch.load(best_model_path))
    model.eval();   
    ptr = 0
    for img, lab, _ in tqdm(attack_loader, desc="attack"):
        img, lab = img.to(torch.device('cuda')), lab.to(torch.device('cuda'))
        bsz = img.size(0)

        img_adv = pgd(model, img, lab, args.eps/255., args.alpha_adv/255., args.steps, n_classes, args.attack)

        adv_cursor = 0                   # points into img_adv
        for k in range(bsz):
            gidx = ptr + k               # global index in dataset
            #torch.save(img[k].cpu(), os.path.join(domain_clean_dir, f"{gidx}.pt"))
            torch.save(img_adv[adv_cursor].cpu(), os.path.join(attack_config_dom_dir, f"{gidx}.pt"))
            adv_cursor += 1

        ptr += bsz
