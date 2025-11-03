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
from utils.fft import FFTDrop2D
import wandb

def get_args_adv():
    parser = argparse.ArgumentParser()
    parser.add_argument('--algorithm', type=str, default="ERM")
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
    parser.add_argument('--lr', type=float, default=5e-5, help="learning rate")
    parser.add_argument('--lr_decay', type=float, default=0.75, help='for sgd')
    parser.add_argument('--lr_decay1', type=float,
                        default=1.0, help='for pretrained featurizer')
    parser.add_argument('--lr_decay2', type=float, default=1.0,
                        help='inital learning rate decay of network')
    parser.add_argument('--lr_gamma', type=float,
                        default=0.0003, help='for optimizer')
    parser.add_argument('--max_epoch', type=int,
                        default=0, help="max iterations")
    parser.add_argument('--momentum', type=float,
                        default=0.9, help='for optimizer')
    parser.add_argument('--net', type=str, default='resnet18',
                        help="featurizer: vgg16, resnet50, resnet101,DTNBase,ViT-B16/32,ViT-L16/32,ViT-H14")
    parser.add_argument('--N_WORKERS', type=int, default=4)
    parser.add_argument('--save_model_every_checkpoint', action='store_true')
    parser.add_argument('--schuse', action='store_true')
    parser.add_argument('--schusech', type=str, default='cos')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--split_style', type=str, default='strat',
                        help="the style to split the train and eval datasets")
    parser.add_argument('--task', type=str, default="img_dg",
                        choices=["img_dg"], help='now only support image tasks')
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
    parser.add_argument("--fft_rho", type=float, default=1.0,
                        help="Frequency keep ratio for FFT input transform (1.0 disables).")
    parser.add_argument("--fft_alpha", type=float, default=1.0,
                        help="Residual mixing weight for FFT attack transform.")

    args = parser.parse_args()
    args.steps_per_epoch = 100
    
    args.data_dir = os.path.join(args.data_file, args.data_dir, args.dataset)

    args = img_param_init(args)
    assert 0.0 <= args.fft_rho <= 1.0, "fft_rho must be in [0, 1]"
    print_environ()
    return args

@torch.no_grad()
def _project_linf(x0, x, eps):
    return (x0 + (x - x0).clamp(min=-eps, max=eps)).clamp(0.0, 1.0)

@torch.no_grad()
def _project_l2(x0, x, eps):
    b = x.size(0)
    delta = (x - x0)
    flat = delta.view(b, -1)
    nrm = flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
    scale = torch.minimum(torch.ones_like(nrm), eps / nrm)
    delta = (flat * scale).view_as(delta)
    return (x0 + delta).clamp(0.0, 1.0)

def pgd(model, x_pix, y, eps, alpha, steps, n_classes, norm="linf", input_transform=None):
    """
    x_pix: pixel input in [0,1]
    eps, alpha: pixel units (e.g., 8/255, 2/255)
    returns: adversarial example in *pixel* space
    """
    x0 = x_pix.detach()
    x = x0.clone()

    b = x.size(0)
    if norm == "linf":
        delta = torch.empty_like(x).uniform_(-eps, eps)
    elif norm == "l2":
        d = torch.randn_like(x).view(b, -1)
        d = d / d.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
        radius = torch.rand(b, 1, device=x.device) * eps
        delta  = (d * radius).view_as(x)
    else:
        raise ValueError("norm ∈ {'linf','l2'}")

    x = (x0 + delta).clamp(0.0, 1.0)
    x = _project_linf(x0, x, eps) if norm == "linf" else _project_l2(x0, x, eps)

    for _ in range(steps):
        x.requires_grad_(True)
        x_in = input_transform(x) if input_transform is not None else x
        logits = model.predict(x_in)  # <-- model handles standardization
        loss = F.cross_entropy(logits, y)
        (g,) = torch.autograd.grad(loss, x, only_inputs=True)

        #preds = logits.argmax(1)
        #fooled = preds.ne(y) 
        #print(fooled.float().mean().item()) 
        with torch.no_grad():
            if norm == "linf":
                x = _project_linf(x0, x + alpha * g.sign(), eps)
            else:
                g_flat = g.view(b, -1)
                g_dir  = g_flat / g_flat.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
                x = _project_l2(x0, x + alpha * g_dir.view_as(g), eps)

    return x.detach()


# ---------------- CLI --------------------------------------------------------
if __name__ == "__main__":
    args = get_args_adv()
    set_random_seed(args.seed)
    
    wandb.init(
        project="tta3_train_attack",         # ← change to your project
        name=f"{args.dataset}_test-env-{args.test_envs[0]}_s{args.seed}_rho{args.fft_rho}_a{args.fft_alpha}",  # run name in W&B
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
    model.eval()

    attack_transform = None
    if args.fft_rho < 1.0:
        attack_transform = FFTDrop2D(
            keep_ratio=args.fft_rho,
            alpha=args.fft_alpha,
        ).cuda()
        attack_transform.eval()
        for param in attack_transform.parameters():
            param.requires_grad_(False)

    ptr = 0
    for img, lab, _ in tqdm(attack_loader, desc="attack"):
        img, lab = img.to(torch.device('cuda')), lab.to(torch.device('cuda'))
        bsz = img.size(0)

        img_adv = pgd(
            model,
            img,
            lab,
            args.eps / 255.,
            args.alpha_adv / 255.,
            args.steps,
            n_classes,
            args.attack,
            input_transform=attack_transform,
        )

        adv_cursor = 0                   # points into img_adv
        for k in range(bsz):
            gidx = ptr + k               # global index in dataset
            #torch.save(img[k].cpu(), os.path.join(domain_clean_dir, f"{gidx}.pt"))
            torch.save(img_adv[adv_cursor].cpu(), os.path.join(attack_config_dom_dir, f"{gidx}.pt"))
            adv_cursor += 1

        ptr += bsz
