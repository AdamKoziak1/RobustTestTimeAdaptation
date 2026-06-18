# coding=utf-8
"""
Train per-domain model, create ℓp-bounded PGD images,
save *one tensor per image*  (clean & adv)  and print accuracy each epoch.
"""
import argparse, os
from typing import Optional
from tqdm import tqdm
import torch, torch.nn as nn

from alg import alg, modelopera
from alg.opt import *
from datautil.getdataloader import get_img_dataloader_adv
import torch.nn.functional as F

from utils.util import set_random_seed, img_param_init, print_environ, img_param_init, get_config_id
from utils.fft import FFTDrop2D
from utils.safer_aug import SAFERAugmenter
from utils.attack_presets import resolve_attack_config
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
    parser.add_argument('--data_file', type=str, default=os.getcwd(),
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
    parser.add_argument("--attack_method", type=str, default="pgd",
                        choices=["pgd", "autoattack"],
                        help="Adversarial generator: 'pgd' (default, used for all "
                             "main-paper streams) or 'autoattack' (Croce & Hein, ICML "
                             "2020) for the E5 transfer-set robustness check. AutoAttack "
                             "tensors are written to a distinct '<net>_autoattack_...' "
                             "config dir so they never collide with the PGD streams.")
    parser.add_argument("--autoattack_version", type=str, default="standard",
                        choices=["standard", "plus", "rand"],
                        help="AutoAttack ensemble version (only used when "
                             "--attack_method autoattack).")
    parser.add_argument("--max_batches", type=int, default=0,
                        help="If >0, only attack this many batches (debug/timing).")
    parser.add_argument("--attack_preset", type=str, default=None, help="Named attack preset to apply.")
    parser.add_argument("--fft_rho", type=float, default=1.0,
                        help="Frequency keep ratio for FFT input transform (1.0 disables).")
    parser.add_argument("--fft_alpha", type=float, default=1.0,
                        help="Residual mixing weight for FFT attack transform.")
    parser.add_argument("--attack_variant", type=str, default="baseline",
                        choices=["baseline", "random_aug", "easy_aug"],
                        help="Strategy for adversarial example generation with augmentations.")
    parser.add_argument("--attack_aug_views", type=int, default=4,
                        help="Number of augmentation candidates to sample per attack step.")
    parser.add_argument("--attack_aug_prob", type=float, default=0.7,
                        help="Per-augmentation application probability in attack pipelines.")
    parser.add_argument("--attack_aug_max_ops", type=int, default=3,
                        help="Max operations per attack pipeline (0 disables the cap).")
    parser.add_argument("--attack_aug_seed", type=int, default=-1,
                        help="Random seed for attack augmentation sampling (-1 disables).")
    parser.add_argument("--attack_aug_list", type=str, nargs="+", default=None,
                        help="Custom augmentation set for attack pipelines.")

    args = parser.parse_args()
    args.steps_per_epoch = 100
    
    args.data_dir = os.path.join(args.data_file, args.data_dir, args.dataset)

    if args.attack_preset:
        attack_cfg = resolve_attack_config(
            preset_name=args.attack_preset,
            attack_id=None,
            norm=args.attack,
            eps=args.eps,
            steps=args.steps,
            alpha=args.alpha_adv,
            fft_rho=args.fft_rho,
            fft_alpha=args.fft_alpha,
        )
        args.attack = attack_cfg.norm
        args.eps = attack_cfg.eps
        args.steps = attack_cfg.steps
        args.alpha_adv = attack_cfg.alpha
        args.fft_rho = attack_cfg.fft_rho
        args.fft_alpha = attack_cfg.fft_alpha
    args = img_param_init(args)
    if args.attack_aug_max_ops is not None and args.attack_aug_max_ops <= 0:
        args.attack_aug_max_ops = None
    if args.attack_aug_seed is not None and args.attack_aug_seed < 0:
        args.attack_aug_seed = None
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

def _apply_pipeline_batch(augmenter: SAFERAugmenter, x: torch.Tensor, pipeline):
    return torch.stack([augmenter.apply_pipeline(x[i], pipeline) for i in range(x.size(0))], dim=0)


def _attack_forward(
    model,
    x: torch.Tensor,
    y: torch.Tensor,
    input_transform,
    variant: str,
    augmenter: Optional[SAFERAugmenter],
    candidates: int,
):
    def _transform(input_tensor: torch.Tensor) -> torch.Tensor:
        return input_transform(input_tensor) if input_transform is not None else input_tensor

    if variant == "baseline" or augmenter is None:
        x_in = _transform(x)
        logits = model.predict(x_in)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    if variant == "random_aug":
        pipeline = augmenter.sample_pipelines(num_views=1)[0]
        aug = _apply_pipeline_batch(augmenter, x, pipeline)
        x_in = _transform(aug)
        logits = model.predict(x_in)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    if variant == "easy_aug":
        sampled = augmenter.sample_pipelines(num_views=1)[0]
        pipeline = sampled[:1]
        if pipeline:
            aug = _apply_pipeline_batch(augmenter, x, pipeline)
        else:
            aug = x
        x_in = _transform(aug)
        logits = model.predict(x_in)
        loss = F.cross_entropy(logits, y)
        return logits, loss

    raise ValueError(f"Unknown attack_variant: {variant}")


def pgd(
    model,
    x_pix,
    y,
    eps,
    alpha,
    steps,
    n_classes,
    norm="linf",
    input_transform=None,
    attack_variant: str = "baseline",
    attack_augmenter: Optional[SAFERAugmenter] = None,
    attack_views: int = 1,
):
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
        logits, loss = _attack_forward(
            model,
            x,
            y,
            input_transform,
            attack_variant,
            attack_augmenter,
            attack_views,
        )
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

    attack_augmenter = None
    if args.attack_variant != "baseline":
        attack_augmenter = SAFERAugmenter(
            num_views=max(1, args.attack_aug_views),
            augmentations=args.attack_aug_list,
            max_ops=args.attack_aug_max_ops,
            prob=args.attack_aug_prob,
            seed=args.attack_aug_seed,
        )
    
    wandb.init(
        project="tta3_train_attack",         # <- change to your project
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
    if args.attack_method == "autoattack":
        # Keep AutoAttack streams in their own config dir (e.g.
        # resnet18_autoattack_linf_eps-8.0_steps-20) so they never overwrite or
        # get confused with the PGD streams that share the same eps/steps. The
        # TTA side selects this dir via --attack_config_id (see unsupervise_adapt).
        configuration_id = configuration_id.replace(
            f"{args.net}_", f"{args.net}_autoattack_", 1
        )
    attack_config_dom_dir = os.path.join(dataset_dir, configuration_id, dom)
    os.makedirs(attack_config_dom_dir, exist_ok=True)
    # E9: clean_dir (the *.pt clean-image cache AttackedImageFolder reads from)
    # is intentionally backbone-agnostic and shared - the images don't depend
    # on --net. Only the white-box source checkpoint trained here for attack
    # generation is net-specific, so suffix *that* filename (and only for
    # non-default backbones) to avoid a resnet50 run silently overwriting /
    # mis-loading the cached resnet18 model_{dom}_best.pt that the existing
    # adversarial datasets were generated from - this lets resnet50 reuse the
    # exact same seeds (0,1,2) as the rest of the paper with no collisions.
    net_tag = "" if args.net == "resnet18" else f"_{args.net}"
    best_model_path = os.path.join(clean_dir, f"model_{dom}_best{net_tag}.pt")
    

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

    adversary = None
    if args.attack_method == "autoattack":
        from autoattack import AutoAttack

        # AutoAttack operates on pixel inputs in [0,1] and expects a callable
        # mapping images -> logits. We reuse the *exact* same forward as PGD
        # (model.predict, optionally pre-composed with the FFT input transform)
        # so the AutoAttack stream is directly comparable to the PGD stream:
        # same surrogate, same input domain, same eps budget.
        def _forward_pass(x):
            x_in = attack_transform(x) if attack_transform is not None else x
            return model.predict(x_in)

        aa_norm = "Linf" if args.attack == "linf" else "L2"
        adversary = AutoAttack(
            _forward_pass,
            norm=aa_norm,
            eps=args.eps / 255.,
            version=args.autoattack_version,
            device=torch.device("cuda"),
            seed=args.seed,
        )
        # These benchmarks have few classes (PACS=7, VLCS=5, OfficeHome=65), so
        # cap the targeted attacks (APGD-T, FAB-T) at num_classes-1 targets;
        # AutoAttack's default of 9 otherwise triggers a warning and wastes work.
        n_target = max(1, args.num_classes - 1)
        adversary.apgd_targeted.n_target_classes = n_target
        adversary.fab.n_target_classes = n_target

    ptr = 0
    for b_i, (img, lab, _) in enumerate(tqdm(attack_loader, desc="attack")):
        if args.max_batches and b_i >= args.max_batches:
            break
        img, lab = img.to(torch.device('cuda')), lab.to(torch.device('cuda'))
        bsz = img.size(0)

        if args.attack_method == "autoattack":
            img_adv = adversary.run_standard_evaluation(img, lab, bs=bsz)
        else:
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
                attack_variant=args.attack_variant,
                attack_augmenter=attack_augmenter,
                attack_views=args.attack_aug_views,
            )

        adv_cursor = 0                   # points into img_adv
        for k in range(bsz):
            gidx = ptr + k               # global index in dataset
            #torch.save(img[k].cpu(), os.path.join(domain_clean_dir, f"{gidx}.pt"))
            torch.save(img_adv[adv_cursor].cpu(), os.path.join(attack_config_dom_dir, f"{gidx}.pt"))
            adv_cursor += 1

        ptr += bsz
