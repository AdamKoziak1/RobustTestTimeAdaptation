import argparse
import os
import sys
import time
import math
import time
import torch
import torch.nn as nn
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from alg.opt import *
from alg import alg
from utils.util import set_random_seed, Tee, img_param_init, print_environ, load_ckpt
from utils.svd import SVDDrop2D, SVDLoader
from utils.fft import FFTDrop2D, FFTLoader
from utils.image_ops import GaussianBlur2D, GaussianBlurLoader, JPEGCompressionLoader
from adapt_algorithm import collect_params, configure_model
from adapt_algorithm import (
    PseudoLabel,
    SHOTIM,
    T3A,
    BN,
    ERM,
    Tent,
    TSD,
    TTA3,
    SAFER,
    MeanTeacherCorrection,
)
from datautil.attacked_imagefolder import AttackedImageFolder
import statistics
from peft import LoraConfig, get_peft_model
import wandb
from adapt_presets import apply_adapt_preset


def get_args():
    parser = argparse.ArgumentParser(description="Test time adaptation")
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam hyper-param")
    parser.add_argument("--checkpoint_freq", type=int, default=3, help="Checkpoint every N epoch")
    parser.add_argument("--classifier", type=str, default="linear", choices=["linear", "wn"])
    parser.add_argument("--data_file", type=str, default="/home/adam/Downloads/RobustTestTimeAdaptation/", help="root_dir")
    parser.add_argument("--dis_hidden", type=int, default=256, help="dis hidden dimension")
    parser.add_argument("--gpu_id", type=str, nargs="?", default="0", help="device id to run")
    parser.add_argument("--lr_decay", type=float, default=0.75, help="for sgd")
    parser.add_argument("--lr_decay1", type=float, default=1.0, help="for pretrained featurizer")
    parser.add_argument("--lr_decay2",type=float,default=1.0,help="inital learning rate decay of network",)
    parser.add_argument("--lr_gamma", type=float, default=0.0003, help="for optimizer")
    parser.add_argument("--max_epoch", type=int, default=120, help="max epoch")
    parser.add_argument("--momentum", type=float, default=0.9, help="for optimizer")
    parser.add_argument("--N_WORKERS", type=int, default=4)
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument("--schuse", action="store_true")
    parser.add_argument("--schusech", type=str, default="cos")
    parser.add_argument("--split_style",type=str,default="strat",help="the style to split the train and eval datasets",)
    parser.add_argument("--task",type=str,default="img_dg",choices=["img_dg"],help="now only support image tasks",)
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument("--batch_size", type=int, default=64, help="batch_size of **test** time")
    parser.add_argument("--dataset", type=str, default="PACS", help="office-home,PACS,VLCS,DomainNet")
    parser.add_argument("--data_dir", type=str, default="datasets", help="data dir")
    parser.add_argument("--attack_data_dir", type=str, default="/home/adam/Downloads/RobustTestTimeAdaptation/datasets_adv", help="attacked data dir")
    parser.add_argument("--lr",type=float,default=1e-4,help="learning rate of **test** time adaptation,important",)
    parser.add_argument("--net",type=str,default="resnet18",help="featurizer: vgg16, resnet18,resnet50, resnet101,DTNBase,ViT-B16,resnext50",)
    parser.add_argument("--test_envs", type=int, nargs="+", default=[0], help="target domains")
    parser.add_argument("--output", type=str, default="./tta_output", help="result output path")
    parser.add_argument("--adapt_alg",type=str,default="TTA3",help="[Tent,PL,PLC,SHOT-IM,T3A,BN,ETA,LAME,ERM,TSD,TTA3,SAFER,AMTDC]",)
    parser.add_argument("--beta", type=float, default=0.9, help="threshold for pseudo label(PL)")
    parser.add_argument("--episodic", action="store_true", help="is episodic or not,default:False")
    parser.add_argument("--steps", type=int, default=1, help="steps of test time, default:1")
    parser.add_argument("--filter_K",type=int,default=100,help="M in T3A/TSD, in [1,5,20,50,100,-1],-1 denotes no selection",)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--update_param", type=str, default="all", help="all / affine / body / head / lora / tent")
    # two hyper-parameters for EATA (ICML22)
    parser.add_argument("--e_margin",type=float,default=math.log(7) * 0.40,help="entropy margin E_0 in Eqn. (3) for filtering reliable samples",)
    parser.add_argument("--d_margin",type=float,default=0.05,help="epsilon in Eqn. (5) for filtering redundant samples",)
    # TTA3
    parser.add_argument("--use_mi", type=str, choices=['mi', 'em'], default='em')   
    parser.add_argument('--lam_em', type=float, default=1.0, help='weight on entropy minimization')
    parser.add_argument("--lam_flat", type=float, default=0.0, help="Coefficient for Flatness Loss")
    parser.add_argument("--lam_adv", type=float, default=0.0, help="Coefficient for Adversarial Loss")
    parser.add_argument("--lam_cr", type=float, default=0.0, help="Coefficient for Consistency Regularization Loss")
    parser.add_argument("--lam_pl", type=float, default=0.0, help="Coefficient for PsuedoLabel Loss")
    parser.add_argument("--cr_type", type=str, choices=['cosine', 'l2'], default='cosine')   
    parser.add_argument("--cr_start", type=int, choices=[0,1,2,3], default=0, help="Which ResNet block to start consistency-regularization at (0=layer1, …, 3=layer4).")

    # SAFER
    parser.add_argument("--s_num_views", type=int, default=4, help="Number of augmented SAFER views per input.")
    parser.add_argument("--s_include_original", type=int, default=1, choices=[0,1], help="Include original sample as one of the SAFER views (1 enables).")
    parser.add_argument("--s_aug_prob", type=float, default=0.7, help="Probability of sampling each augmentation in the SAFER pipeline.")
    parser.add_argument("--s_aug_max_ops", type=int, default=4, help="Max number of operations per SAFER augmentation pipeline (0 disables the cap).")
    parser.add_argument("--s_aug_list", type=str, nargs="+", default=None, help="Optional custom list of SAFER augmentations to sample from.")
    parser.add_argument("--s_js_weight", type=float, default=1.0, help="Weight for SAFER JS divergence consistency loss.")
    parser.add_argument("--s_cc_weight", type=float, default=1.0, help="Weight for SAFER cross-correlation loss.")
    parser.add_argument("--s_cc_offdiag", type=float, default=1.0, help="Weight on off-diagonal terms in SAFER cross-correlation loss.")
    parser.add_argument("--s_cc_impl", type=str, default="fast", choices=["fast", "einsum"], help="Cross-correlation implementation: fast pairwise or einsum-based.")
    parser.add_argument("--s_feat_normalize", type=int, default=0, choices=[0,1], help="L2-normalise features before computing SAFER cross-correlation.")
    parser.add_argument("--s_aug_seed", type=int, default=-1, help="Deterministic seed for SAFER augmentation sampling (-1 disables).")
    parser.add_argument(
        "--s_sup_type",
        type=str.lower,
        default="none",
        choices=["none", "pl", "em"],
        help="Additional SAFER supervision: none, confidence-weighted pseudo-label (pl), or entropy minimisation (em).",
    )
    parser.add_argument(
        "--s_sup_weight",
        type=float,
        default=0.0,
        help="Weight applied to the SAFER pseudo-label / entropy minimization loss.",
    )
    parser.add_argument(
        "--s_sup_view_pool",
        type=str.lower,
        default="mean",
        choices=["mean", "worst", "entropy", "top1", "cc", "cc_drop"],
        help="Pooling strategy for combining SAFER view predictions.",
    )
    parser.add_argument(
        "--s_sup_pl_weighted",
        type=int,
        default=0,
        choices=[0, 1],
        help="Weight pseudo-label loss across views using the pooling weights (1 enables).",
    )
    parser.add_argument(
        "--s_sup_conf_scale",
        type=int,
        default=1,
        choices=[0, 1],
        help="Scale pseudo-label losses by pooled confidence (1 enables).",
    )
    parser.add_argument(
        "--s_js_mode",
        type=str.lower,
        default="pooled",
        choices=["pooled", "pairwise"],
        help="JS divergence computation mode.",
    )
    parser.add_argument(
        "--s_js_view_pool",
        type=str.lower,
        default="matching",
        choices=["matching", "mean", "worst", "entropy", "top1", "cc", "cc_drop"],
        help="Pooling strategy for JS reference; 'matching' reuses the supervision pool.",
    )
    parser.add_argument(
        "--s_view_weighting",
        type=int,
        default=1,
        choices=[0, 1],
        help="Enable view-weighting when pooling for SAFER losses.",
    )
    parser.add_argument(
        "--s_tta_loss",
        type=str.lower,
        default="none",
        choices=["none", "tent", "pl", "tsd"],
        help="Auxiliary TTA loss applied to SAFER predictions.",
    )
    parser.add_argument(
        "--s_tta_weight",
        type=float,
        default=0.0,
        help="Weight applied to SAFER TTA auxiliary loss.",
    )
    parser.add_argument(
        "--s_tta_target",
        type=str.lower,
        default="views",
        choices=["views", "pooled"],
        help="Whether to apply the SAFER TTA loss to individual views or the pooled prediction.",
    )
    parser.add_argument(
        "--s_tta_view_pool",
        type=str.lower,
        default="matching",
        choices=["matching", "mean", "worst", "entropy", "top1", "cc", "cc_drop"],
        help="Pooling strategy for TTA losses; 'matching' reuses the supervision pool.",
    )
    parser.add_argument(
        "--s_cc_mode",
        type=str.lower,
        default="pairwise",
        choices=["pairwise", "pooled"],
        help="Cross-correlation mode: pairwise across views or each view vs pooled features.",
    )
    parser.add_argument(
        "--s_cc_view_pool",
        type=str.lower,
        default="matching",
        choices=["matching", "mean", "worst", "entropy", "top1", "cc", "cc_drop"],
        help="Pooling strategy for pooled-feature cross-correlation; 'matching' reuses the supervision pool.",
    )
    # Adaptive Mean-Teacher Data Correction
    parser.add_argument("--mt_alpha", type=float, default=0.02, help="Step size for data correction (alpha).")
    parser.add_argument("--mt_gamma", type=float, default=0.99, help="EMA momentum for teacher parameters (gamma).")
    parser.add_argument("--mt_gamma_y", type=float, default=0.5, help="Mixing weight between student and teacher pseudo-labels (gamma_y).")
    parser.add_argument("--mt_kl_weight", type=float, default=0.1, help="Weight on KL regularisation between student and teacher.")
    parser.add_argument("--mt_ce_weight", type=float, default=1.0, help="Weight on cross-entropy w.r.t pseudo-labels.")
    parser.add_argument("--mt_ent_weight", type=float, default=0.0, help="Weight on entropy minimisation term.")
    parser.add_argument("--mt_mixup_weight", type=float, default=0.0, help="Weight on mixup consistency regulariser.")
    parser.add_argument("--mt_mixup_beta", type=float, default=0.5, help="Beta distribution parameter for mixup.")
    parser.add_argument("--mt_use_teacher_pred", type=int, default=1, choices=[0,1], help="Use teacher predictions for evaluation (1 enables).")

    parser.add_argument("--attack", choices=["linf_eps-8.0_steps-20", "clean", "l2_eps-112.0_steps-100", "linf_eps-8.0_steps-20_rho-0.3_a-1.0"], default="linf_eps-8.0_steps-20")
    parser.add_argument("--eps", type=float, default=4)  
    parser.add_argument("--attack_rate", type=int, choices=[0, 25, 50, 75, 100], default=0)   
    parser.add_argument("--lora_r", type=int, default=4)  
    parser.add_argument("--lora_alpha", type=int, default=8)  
    parser.add_argument("--lora_dropout", type=float, default=0.0)  

    parser.add_argument("--svd_input_rank_ratio", type=float, default=1.0, help="Rank ratio for input SVD projection (1.0 disables it).")
    parser.add_argument("--svd_input_mode", choices=["spatial","channel"], default="spatial")
    parser.add_argument("--svd_feat_rank_ratio", type=float, default=1.0, help="proportional rank threshold for feature-map SVD.")
    parser.add_argument('--svd_feat_max_layer', type=int, default=0, choices=[0,1,2,3,4], help="ResNet block at which to end lowrank (0=off)")
    parser.add_argument("--svd_feat_mode", choices=["spatial","channel"], default="spatial")

    parser.add_argument("--fft_input_keep_ratio", type=float, default=1.0, help="Frequency keep ratio for input FFT filtering (1.0 disables it).")
    parser.add_argument("--fft_input_mode", choices=["spatial", "channel"], default="spatial")
    parser.add_argument("--fft_input_alpha", type=float, default=1.0, help="Residual mix weight for FFT input filtering.")
    parser.add_argument("--fft_input_learn_alpha", type=int, default=0, choices=[0,1], help="Learn residual alpha for FFT input filtering (1 enables).")
    parser.add_argument("--fft_input_use_residual", type=int, default=1, choices=[0,1], help="Enable residual mixing for FFT input filtering (1 enables).")
    parser.add_argument("--fft_feat_keep_ratio", type=float, default=1.0, help="Frequency keep ratio for FFT feature filtering (1.0 disables it).")
    parser.add_argument('--fft_feat_max_layer', type=int, default=1, choices=[0,1,2,3,4], help="ResNet block at which to end FFT filtering (0=off)")
    parser.add_argument("--fft_feat_mode", choices=["spatial", "channel"], default="spatial")
    parser.add_argument("--fft_feat_alpha", type=float, default=1.0, help="Residual mix weight for FFT feature filtering.")
    parser.add_argument("--fft_feat_learn_alpha", type=int, default=0, choices=[0,1], help="Learn residual alpha for FFT feature filtering (1 enables).")
    parser.add_argument("--fft_feat_use_residual", type=int, default=1, choices=[0,1], help="Enable residual mixing for FFT feature filtering (1 enables).")
    parser.add_argument("--gauss_input_sigma", type=float, default=0.0, help="Gaussian blur σ for input preprocessing (0 disables).")
    parser.add_argument("--gauss_feat_sigma", type=float, default=0.0, help="Gaussian blur σ inserted after the first conv block (0 disables).")
    parser.add_argument("--jpeg_input_quality", type=int, default=100, help="JPEG quality (1-100) for input re-encoding (100 disables).")

    parser.add_argument('--nuc_top', type=int, default=0, help='0..4 stages instrumented (bottom-up)')
    parser.add_argument('--nuc_after_stem', action='store_true', help='also insert after stem (post-maxpool)')
    parser.add_argument('--nuc_kernel', type=int, default=3, help='odd kernel size for NuclearConv2d')
    parser.add_argument('--nuc_lambda', type=float, default=0.0, help='weight on nuclear-norm penalty')
    parser.add_argument('--lam_recon', type=float, default=0.0, help='weight on feature reconstruction penalty')

    parser.add_argument('--lam_reg', type=float, default=1.0, help='weight on student-teacher regularization')
    parser.add_argument("--reg_type", choices=["l2logits","klprob"], default="l2logits")
    parser.add_argument('--ema', type=float, default=0.99, help='EMA coefficient for student-teacher distillation')
    parser.add_argument('--x_lr', type=float, default=0.1, help='learning rate for x_tilde update')
    parser.add_argument('--x_steps', type=int, default=3, help='number of steps for x_tilde update')
    parser.add_argument('--disable_preset_hparams', type=int, default=1, choices=[0,1], help='Disable auto-selection of preset hyperparameters based on adapt_alg.')

    
    args = parser.parse_args()
    if args.svd_input_rank_ratio >=1:
        args.svd_input_rank_ratio = args.svd_feat_rank_ratio
    preset_overrides = apply_adapt_preset(args, disable=args.disable_preset_hparams)
    if preset_overrides:
        print(f"Applying preset hyperparameters for {args.adapt_alg}: {preset_overrides}")
    args.preset_overrides = preset_overrides
    args.steps_per_epoch = 100
    args.data_dir =  os.path.join(args.data_file, args.data_dir, args.dataset)
    args.use_mi = args.use_mi == 'mi'

    args = img_param_init(args)

    args.fft_input_use_residual = bool(args.fft_input_use_residual)
    args.fft_input_learn_alpha = bool(args.fft_input_learn_alpha)
    args.fft_feat_use_residual = bool(args.fft_feat_use_residual)
    args.fft_feat_learn_alpha = bool(args.fft_feat_learn_alpha)

    args.s_include_original = bool(args.s_include_original)
    args.s_feat_normalize = bool(args.s_feat_normalize)
    args.s_cc_impl = args.s_cc_impl.lower()
    args.s_sup_pl_weighted = bool(args.s_sup_pl_weighted)
    args.s_sup_conf_scale = bool(args.s_sup_conf_scale)
    args.s_js_view_pool = args.s_js_view_pool.lower()
    args.s_js_mode = args.s_js_mode.lower()
    args.s_view_weighting = bool(args.s_view_weighting)
    args.s_tta_loss = args.s_tta_loss.lower()
    args.s_tta_target = args.s_tta_target.lower()
    args.s_tta_view_pool = args.s_tta_view_pool.lower()
    args.s_cc_mode = args.s_cc_mode.lower()
    args.s_cc_view_pool = args.s_cc_view_pool.lower()
    if args.s_sup_type == "none" or args.s_sup_weight <= 0.0:
        args.s_sup_type = "none"
        args.s_sup_weight = 0.0
    if args.s_tta_loss == "none" or args.s_tta_weight <= 0.0:
        args.s_tta_loss = "none"
        args.s_tta_weight = 0.0
    args.mt_use_teacher_pred = bool(args.mt_use_teacher_pred)
    if args.mt_mixup_beta <= 0:
        args.mt_mixup_beta = 0.5
    if args.s_aug_max_ops is not None and args.s_aug_max_ops <= 0:
        args.s_aug_max_ops = None
    if args.s_aug_seed is not None and args.s_aug_seed < 0:
        args.s_aug_seed = None

    assert args.filter_K in [1,5,20,50,100,-1], "filter_K must be in [1,5,20,50,100,-1]"
    assert 0.0 <= args.svd_input_rank_ratio <= 1.0, "svd_input_rank_ratio must be in [0,1]"
    assert 0.0 <= args.fft_input_keep_ratio <= 1.0, "fft_input_keep_ratio must be in [0,1]"
    assert 0.0 <= args.fft_feat_keep_ratio <= 1.0, "fft_feat_keep_ratio must be in [0,1]"
    assert args.gauss_input_sigma >= 0.0, "gauss_input_sigma must be non-negative"
    assert args.gauss_feat_sigma >= 0.0, "gauss_feat_sigma must be non-negative"
    assert 0 <= args.jpeg_input_quality <= 100, "jpeg_input_quality must be in [0, 100]"
    print_environ()
    return args


def log_args(args, time_taken_s):
    wandb.log({
        "adapt_algorithm": args.adapt_alg,
        "attack_rate": args.attack_rate,
        # "svd_feat_rank_ratio": args.svd_feat_rank_ratio,
        # "svd_feat_max_layer": args.svd_feat_max_layer,
        # "svd_feat_mode": args.svd_feat_mode,
        # "svd_input_rank_ratio": args.svd_input_rank_ratio,
        # "svd_input_mode": args.svd_input_mode,
        "fft_feat_keep_ratio": args.fft_feat_keep_ratio,
        "fft_feat_max_layer": args.fft_feat_max_layer,
        "fft_feat_alpha": args.fft_feat_alpha,
        "fft_input_keep_ratio": args.fft_input_keep_ratio,
        "fft_input_alpha": args.fft_input_alpha,
        "steps": args.steps,
        "lr": args.lr,
        # "lam_flat": args.lam_flat,
        # "lam_adv": args.lam_adv,
        # "lam_cr": args.lam_cr,
        # "lam_pl": args.lam_pl,
        # "lam_em": args.lam_em,
        # "lam_nuc": args.nuc_lambda,
        # "lam_recon": args.lam_recon,
        # "nuc_kernel": args.nuc_kernel,
        # "nuc_top": args.nuc_top,
        "time_taken_s": time_taken_s,
        # "lam_reg": args.lam_reg,
        # "reg_type": args.reg_type,
        # "ema": args.ema,
        # "x_lr": args.x_lr,
        # "x_steps": args.x_steps
    })
    if args.adapt_alg == "SAFER":
        wandb.log({
            "s_num_views": args.s_num_views,
            "s_include_original": args.s_include_original,
            "s_aug_prob": args.s_aug_prob,
            "s_aug_max_ops": -1 if args.s_aug_max_ops is None else args.s_aug_max_ops,
            "s_js_weight": args.s_js_weight,
            "s_cc_weight": args.s_cc_weight,
            "s_cc_offdiag": args.s_cc_offdiag,
            "s_cc_impl": args.s_cc_impl,
            "s_feat_normalize": args.s_feat_normalize,
            "s_sup_type": args.s_sup_type,
            "s_sup_weight": args.s_sup_weight,
            "s_sup_view_pool": args.s_sup_view_pool,
            "s_sup_pl_weighted": args.s_sup_pl_weighted,
            "s_sup_conf_scale": args.s_sup_conf_scale,
            "s_js_mode": args.s_js_mode,
            "s_js_view_pool": args.s_js_view_pool,
            "s_view_weighting": args.s_view_weighting,
            "s_tta_loss": args.s_tta_loss,
            "s_tta_weight": args.s_tta_weight,
            "s_tta_target": args.s_tta_target,
            "s_tta_view_pool": args.s_tta_view_pool,
            "s_cc_mode": args.s_cc_mode,
            "s_cc_view_pool": args.s_cc_view_pool,
        }, commit=False)
    elif args.adapt_alg == "AMTDC":
        wandb.log({
            "mt_alpha": args.mt_alpha,
            "mt_gamma": args.mt_gamma,
            "mt_gamma_y": args.mt_gamma_y,
            "mt_kl_weight": args.mt_kl_weight,
            "mt_ce_weight": args.mt_ce_weight,
            "mt_ent_weight": args.mt_ent_weight,
            "mt_mixup_weight": args.mt_mixup_weight,
            "mt_mixup_beta": args.mt_mixup_beta,
            "mt_use_teacher_pred": args.mt_use_teacher_pred,
        }, commit=False)

def adapt_loader(args):
    test_envs = args.test_envs[0]
    domain_name = args.img_dataset[args.dataset][test_envs]
    data_root = os.path.join(args.data_dir, args.img_dataset[args.dataset][test_envs])
    if args.attack == "clean":
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )
        test_transform = transforms.Compose(
            [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
        )
        testset = ImageFolder(root=data_root, transform=test_transform)
    else:
        testset = AttackedImageFolder(
            root=data_root, # normal ImageFolder root
            transform=None,
            adv_root=args.attack_data_dir,
            dataset=args.dataset,
            domain=domain_name,
            config=f"{args.net}_{args.attack}",
            rate=args.attack_rate,                            
            seed=args.seed)   

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.N_WORKERS,
        pin_memory=True,
    )

    loader = testloader
    if 1 <= args.jpeg_input_quality < 100:
        loader = JPEGCompressionLoader(
            loader,
            quality=args.jpeg_input_quality,
            device="cuda",
        )
    if args.gauss_input_sigma > 0:
        loader = GaussianBlurLoader(
            loader,
            sigma=args.gauss_input_sigma,
            device="cuda",
        )

    if args.svd_input_rank_ratio < 1.0:
        loader = SVDLoader(
            loader,
            rank_ratio=args.svd_input_rank_ratio,
            device="cuda",
            mode=args.svd_input_mode,
            use_ste=False,
        )
    elif args.fft_input_keep_ratio < 1.0:
        loader = FFTLoader(
            loader,
            keep_ratio=args.fft_input_keep_ratio,
            device="cuda",
            mode=args.fft_input_mode,
            use_ste=False,
            use_residual=args.fft_input_use_residual,
            alpha=args.fft_input_alpha,
            learn_alpha=args.fft_input_learn_alpha,
        )
    return loader


def make_adapt_model(args, algorithm):
    if args.svd_feat_max_layer > 0 and args.svd_feat_rank_ratio < 1.0: 
        rank_ratio = args.svd_feat_rank_ratio
        mode = args.svd_feat_mode
        feat = algorithm.featurizer 
        feat.layer1 = nn.Sequential(feat.layer1, SVDDrop2D(rank_ratio, mode)) if args.svd_feat_max_layer >= 1 else feat.layer1
        feat.layer2 = nn.Sequential(feat.layer2, SVDDrop2D(rank_ratio, mode)) if args.svd_feat_max_layer >= 2 else feat.layer2
        feat.layer3 = nn.Sequential(feat.layer3, SVDDrop2D(rank_ratio, mode)) if args.svd_feat_max_layer >= 3 else feat.layer3
        feat.layer4 = nn.Sequential(feat.layer4, SVDDrop2D(rank_ratio, mode)) if args.svd_feat_max_layer >= 4 else feat.layer4

    if args.fft_feat_max_layer > 0 and args.fft_feat_keep_ratio < 1.0:
        keep_ratio = args.fft_feat_keep_ratio
        mode = args.fft_feat_mode
        use_residual = args.fft_feat_use_residual
        alpha = args.fft_feat_alpha
        learn_alpha = args.fft_feat_learn_alpha
        feat = algorithm.featurizer

        feat.layer1 = nn.Sequential(feat.layer1, FFTDrop2D(keep_ratio, mode=mode, use_residual=use_residual, alpha=alpha, learn_alpha=learn_alpha)) if args.fft_feat_max_layer >= 1 else feat.layer1
        feat.layer2 = nn.Sequential(feat.layer2, FFTDrop2D(keep_ratio, mode=mode, use_residual=use_residual, alpha=alpha, learn_alpha=learn_alpha)) if args.fft_feat_max_layer >= 2 else feat.layer2
        feat.layer3 = nn.Sequential(feat.layer3, FFTDrop2D(keep_ratio, mode=mode, use_residual=use_residual, alpha=alpha, learn_alpha=learn_alpha)) if args.fft_feat_max_layer >= 3 else feat.layer3
        feat.layer4 = nn.Sequential(feat.layer4, FFTDrop2D(keep_ratio, mode=mode, use_residual=use_residual, alpha=alpha, learn_alpha=learn_alpha)) if args.fft_feat_max_layer >= 4 else feat.layer4
    if args.gauss_feat_sigma > 0:
        feat = algorithm.featurizer
        blur = GaussianBlur2D(args.gauss_feat_sigma)
        if hasattr(feat, "layer1"):
            feat.layer1 = nn.Sequential(feat.layer1, blur)
        else:
            print("Warning: gauss_feat_sigma > 0 but featurizer lacks layer1; skipping blur.")

    # set adapt model and optimizer
    if args.adapt_alg == "Tent":
        algorithm = configure_model(algorithm)
        params, _ = collect_params(algorithm)
        optimizer = torch.optim.Adam(params, lr=args.lr)
        adapt_model = Tent(
            algorithm, optimizer, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "ERM":
        adapt_model = ERM(algorithm)
    elif args.adapt_alg == "PL":
        optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
        adapt_model = PseudoLabel(
            algorithm, optimizer, args.beta, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "PLC":
        optimizer = torch.optim.Adam(algorithm.classifier.parameters(), lr=args.lr)
        adapt_model = PseudoLabel(
            algorithm, optimizer, args.beta, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "SHOT-IM":
        optimizer = torch.optim.Adam(algorithm.featurizer.parameters(), lr=args.lr)
        adapt_model = SHOTIM(
            algorithm, optimizer, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "T3A":
        adapt_model = T3A(
            algorithm, filter_K=args.filter_K, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "BN":
        adapt_model = BN(algorithm)
    elif args.adapt_alg == "TSD":
        if args.update_param == "all":
            optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
            sum_params = sum([p.nelement() for p in algorithm.parameters()])
        elif args.update_param == "affine":
            algorithm.train()
            algorithm.requires_grad_(False)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            for m in algorithm.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
            sum_params = sum([p.nelement() for p in params])
        elif args.update_param == "body":
            # only update encoder
            optimizer = torch.optim.Adam(algorithm.featurizer.parameters(), lr=args.lr)
            print("Update encoder")
        elif args.update_param == "head":
            # only update classifier
            optimizer = torch.optim.Adam(algorithm.classifier.parameters(), lr=args.lr)
            print("Update classifier")
        else:
            raise Exception("Do not support update with %s manner." % args.update_param)
        adapt_model = TSD(
            algorithm,
            optimizer,
            filter_K=args.filter_K,
            steps=args.steps,
            episodic=args.episodic,
        )

    elif args.adapt_alg == "TTA3":
        if args.update_param == "all":
            optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
            sum_params = sum([p.nelement() for p in algorithm.parameters()])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "affine":
            algorithm.train()
            algorithm.requires_grad_(False)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            for m in algorithm.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
            sum_params = sum([p.nelement() for p in params])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "tent":
            algorithm = configure_model(algorithm)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            sum_params = sum([p.nelement() for p in params])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "body":
            # only update encoder
            optimizer = torch.optim.Adam(algorithm.featurizer.parameters(), lr=args.lr)
            print("Update encoder")
        elif args.update_param == "head":
            # only update classifier
            optimizer = torch.optim.Adam(algorithm.classifier.parameters(), lr=args.lr)
            print("Update classifier")
        elif args.update_param == "lora":
            def resnet_target_modules(model, depth=(3, 4)):
                targets = []
                for blk in depth:
                    for n, m in model.named_modules():
                        if f"layer{blk}" in n and isinstance(m, (nn.Conv2d, nn.Linear)):
                            targets.append(n)  
                return list(set(targets))
            
            lora_cfg = LoraConfig(
                r=args.lora_r,
                lora_alpha=args.lora_alpha,
                lora_dropout=args.lora_dropout,
                target_modules=resnet_target_modules(algorithm.featurizer, depth=(1,2,3,4)),
                bias="none",
                task_type="FEATURE_EXTRACTION"
            )
            algorithm = get_peft_model(algorithm, lora_cfg)
            algorithm.print_trainable_parameters()  # sanity‑check
            
            optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
        elif args.update_param == "nuc":
            optimizer = torch.optim.Adam(algorithm.featurizer.nuc_parameters(), lr=args.lr)
        else:
            raise Exception("Do not support update with %s manner." % args.update_param)
        
        adapt_model = TTA3(
            algorithm,
            optimizer,
            steps=args.steps,
            episodic=args.episodic,
            lam_flat=args.lam_flat,
            lam_adv=args.lam_adv,
            lam_cr=args.lam_cr,
            lam_pl=args.lam_pl,
            cr_type = args.cr_type,
            cr_start = args.cr_start,
            r=args.eps,
            use_mi=args.use_mi,
            lambda_nuc=args.nuc_lambda,
            lam_em=args.lam_em,
            lam_recon=args.lam_recon,
            lam_reg=args.lam_reg,
            reg_type=args.reg_type,
            ema=args.ema,
            x_lr=args.x_lr,
            x_steps=args.x_steps
        )
    elif args.adapt_alg == "SAFER":
        if args.update_param == "all":
            optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
            sum_params = sum([p.nelement() for p in algorithm.parameters()])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "affine":
            algorithm.train()
            algorithm.requires_grad_(False)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            for m in algorithm.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
            sum_params = sum([p.nelement() for p in params])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "tent":
            algorithm = configure_model(algorithm)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            sum_params = sum([p.nelement() for p in params])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "body":
            optimizer = torch.optim.Adam(algorithm.featurizer.parameters(), lr=args.lr)
            print("Update encoder")
        elif args.update_param == "head":
            optimizer = torch.optim.Adam(algorithm.classifier.parameters(), lr=args.lr)
            print("Update classifier")
        else:
            raise Exception("Do not support update with %s manner." % args.update_param)

        augment_list = args.s_aug_list if args.s_aug_list else None
        adapt_model = SAFER(
            algorithm,
            optimizer,
            steps=args.steps,
            episodic=args.episodic,
            num_aug_views=args.s_num_views,
            include_original=args.s_include_original,
            aug_prob=args.s_aug_prob,
            aug_max_ops=args.s_aug_max_ops,
            augmentations=augment_list,
            js_weight=args.s_js_weight,
            cc_weight=args.s_cc_weight,
            offdiag_weight=args.s_cc_offdiag,
            feature_normalize=args.s_feat_normalize,
            aug_seed=args.s_aug_seed,
            sup_mode=args.s_sup_type,
            sup_weight=args.s_sup_weight,
            cc_impl=args.s_cc_impl,
            sup_view_pool=args.s_sup_view_pool,
            sup_pl_weighted=args.s_sup_pl_weighted,
            sup_confidence_scale=args.s_sup_conf_scale,
            js_view_pool=args.s_js_view_pool,
            js_mode=args.s_js_mode,
            view_weighting=args.s_view_weighting,
            tta_loss=args.s_tta_loss,
            tta_weight=args.s_tta_weight,
            tta_target=args.s_tta_target,
            tta_view_pool=args.s_tta_view_pool,
            cc_mode=args.s_cc_mode,
            cc_view_pool=args.s_cc_view_pool,
        )
    elif args.adapt_alg == "AMTDC":
        if args.update_param == "all":
            optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
            sum_params = sum([p.nelement() for p in algorithm.parameters()])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "affine":
            algorithm.train()
            algorithm.requires_grad_(False)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            for m in algorithm.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.requires_grad_(True)
            sum_params = sum([p.nelement() for p in params])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "tent":
            algorithm = configure_model(algorithm)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
            sum_params = sum([p.nelement() for p in params])
            wandb.log({"sum_params": sum_params})
        elif args.update_param == "body":
            optimizer = torch.optim.Adam(algorithm.featurizer.parameters(), lr=args.lr)
            print("Update encoder")
        elif args.update_param == "head":
            optimizer = torch.optim.Adam(algorithm.classifier.parameters(), lr=args.lr)
            print("Update classifier")
        else:
            raise Exception("Do not support update with %s manner." % args.update_param)

        adapt_model = MeanTeacherCorrection(
            algorithm,
            optimizer,
            steps=args.steps,
            episodic=args.episodic,
            correction_alpha=args.mt_alpha,
            teacher_momentum=args.mt_gamma,
            pseudo_momentum=args.mt_gamma_y,
            kl_weight=args.mt_kl_weight,
            ce_weight=args.mt_ce_weight,
            ent_weight=args.mt_ent_weight,
            mixup_weight=args.mt_mixup_weight,
            mixup_beta=args.mt_mixup_beta,
            use_teacher_prediction=args.mt_use_teacher_pred,
        )
    else:
        raise ValueError(f"Unknown adapt_alg: {args.adapt_alg}")
    return adapt_model.cuda()


def run_one_seed(args):
    pretrain_model_path = os.path.join(args.data_file, "train_output", args.dataset, f"test_{str(dom_id)}", f"seed_{str(args.seed)}", "model.pkl")
    set_random_seed(args.seed)
    
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args)
    algorithm.train()
    algorithm = load_ckpt(algorithm, pretrain_model_path)

    dataloader = adapt_loader(args)
    adapt_model = make_adapt_model(args, algorithm)

    adapt_model.cuda()
    outputs_arr, labels_arr = [], []
    peak_vram_mb = 0.0
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()

    for _, sample in enumerate(dataloader):
        image, label = sample
        image = image.cuda()
        logits = adapt_model(image)
        
        outputs = logits.detach().cpu()
        batch_acc = 100*accuracy_score(label.numpy(), outputs.argmax(1).numpy())
        wandb.log({"batch_acc": batch_acc})
        outputs_arr.append(outputs)
        labels_arr.append(label)

    outputs_arr = torch.cat(outputs_arr, 0).numpy()
    labels_arr = torch.cat(labels_arr).numpy()
    outputs_arr = outputs_arr.argmax(1)
    if use_cuda:
        peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    wandb.log({"max_vram_overhead_mb": peak_vram_mb}, commit=False)

    return 100*accuracy_score(labels_arr, outputs_arr)


if __name__ == "__main__":
    args = get_args()
    
    output_path = os.path.join(args.output, args.dataset, str(args.test_envs[0]), args.adapt_alg, str(args.attack_rate))


    dom_id = args.test_envs[0]
    #run_name = f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}_rate-{args.attack_rate}"
    run_name = f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}-ax{args.fft_input_alpha}-px{args.fft_input_keep_ratio}-az{args.fft_feat_alpha}-pz{args.fft_feat_keep_ratio}_rate-{args.attack_rate}"

    if args.adapt_alg == "TTA3":
        cr_modifier = ""
        if args.lam_cr >= 1e-8:
            cr_modifier = f"-{args.cr_type}"
        #run_name = f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}-{args.lam_flat}-{args.lam_adv}-{args.lam_cr}{cr_modifier}_rate-{args.attack_rate}"
        run_name = f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}-fftin-k{args.fft_input_keep_ratio}-a{args.fft_input_alpha}-feat-k{args.fft_feat_keep_ratio}-a{args.fft_feat_alpha}-l{args.fft_feat_max_layer}_rate-{args.attack_rate}"
    elif args.adapt_alg == "SAFER":
        run_name = (
            f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}"
            f"-v{args.s_num_views}"
            f"-p{args.s_aug_prob:.2f}"
            f"-js{args.s_js_weight:.2f}"
            f"-cc{args.s_cc_weight:.2f}"
        )
        if args.s_sup_type != "none" and args.s_sup_weight > 0:
            run_name += f"-{args.s_sup_type}{args.s_sup_weight:.2f}"
        run_name += (
            f"_rate-{args.attack_rate}"
        )
    elif args.adapt_alg == "AMTDC":
        run_name = (
            f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}"
            f"-a{args.mt_alpha:.3f}"
            f"-g{args.mt_gamma:.2f}"
            f"-gy{args.mt_gamma_y:.2f}"
            f"-kl{args.mt_kl_weight:.2f}"
            f"-mix{args.mt_mixup_weight:.2f}"
            f"_rate-{args.attack_rate}"
        )

    wandb.init(
        project="fft_runs",
        name=run_name,
        config=vars(args),
    )

    all_acc   = []
    time1 = time.time()
    for s in (0,1,2):
        args.seed = s   
        args.output = os.path.join(output_path, f"_s{args.seed}")
        os.makedirs(args.output, exist_ok=True)
        sys.stdout = Tee(os.path.join(args.output, "out.txt"))
        sys.stderr = Tee(os.path.join(args.output, "err.txt"))
        acc_s = run_one_seed(args)
        all_acc.append(acc_s)

    time2 = time.time()
    acc_mean = round(statistics.mean(all_acc), 2)
    acc_std  = round(statistics.stdev(all_acc), 2)

    print("\t Hyper-parameter")
    print("\t Dataset: {}".format(args.dataset))
    print("\t Net: {}".format(args.net))
    print("\t Test domain: {}".format(dom_id))
    print("\t Algorithm: {}".format(args.adapt_alg))
    print("\t Accuracy: %f" % float(acc_mean))
    print("\t Accuracy std: %f" % float(acc_std))
    print("\t Cost time: %f s" % (time2 - time1))

    wandb.log({f"acc_mean": acc_mean, f"acc_std": acc_std}, commit=False)
    log_args(args, time2 - time1)
    

    wandb.finish()
