import argparse
import copy
import os
import sys
import time
import math
import time
from PIL import ImageFile
import torch
import torch.nn as nn
from typing import Optional
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score
from alg.opt import *
from alg import alg
from utils.util import set_random_seed, Tee, img_param_init, print_environ, load_ckpt
from utils.svd import SVDDrop2D
from utils.fft import FFTDrop2D
from utils.image_ops import GaussianBlur2D, InputDefense
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
    TeSLA,
    MeanTeacherCorrection,
    SAFERPooledPredictor,
)
from utils.safer_view import SAFERViewModule
from datautil.attacked_imagefolder import AttackedImageFolder
import statistics
from peft import LoraConfig, get_peft_model
import wandb
from adapt_presets import apply_adapt_preset
from utils.attack_presets import resolve_attack_config, DEFAULT_ATTACK_PRESET
from utils.adv_attack import build_attack_transform, pgd_attack

ImageFile.LOAD_TRUNCATED_IMAGES = True


class InputDefenseWrapper(nn.Module):
    def __init__(self, model: nn.Module, defense: nn.Module) -> None:
        super().__init__()
        self.model = model
        self.defense = defense
        self.defense.requires_grad_(False)
        self.defense.eval()

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.defense(x)
        if hasattr(self.model, "predict"):
            return self.model.predict(x)
        return self.model(x)

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        x = self.defense(x)
        if hasattr(self.model, "predict"):
            return self.model.predict(x)
        return self.model(x)

    def __getattr__(self, name: str):
        if name in {"model", "defense"}:
            return super().__getattr__(name)
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def _build_input_defense(args) -> Optional[nn.Module]:
    use_jpeg = 1 <= args.jpeg_input_quality < 100
    use_gauss = args.gauss_input_sigma > 0
    use_svd = args.svd_input_rank_ratio < 1.0
    use_fft = (not use_svd) and (args.fft_input_keep_ratio < 1.0)
    if not (use_jpeg or use_gauss or use_svd or use_fft):
        return None
    return InputDefense(
        jpeg_quality=args.jpeg_input_quality,
        jpeg_backprop=args.jpeg_input_backprop,
        gauss_sigma=args.gauss_input_sigma,
        svd_rank_ratio=args.svd_input_rank_ratio,
        svd_mode=args.svd_input_mode,
        fft_keep_ratio=args.fft_input_keep_ratio,
        fft_mode=args.fft_input_mode,
        fft_alpha=args.fft_input_alpha,
        fft_use_residual=bool(args.fft_input_use_residual),
        fft_learn_alpha=bool(args.fft_input_learn_alpha),
    )


def wrap_with_input_defense(model: nn.Module, args) -> nn.Module:
    if isinstance(model, InputDefenseWrapper):
        return model
    defense = _build_input_defense(args)
    if defense is None:
        return model
    return InputDefenseWrapper(model, defense)


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
    parser.add_argument("--adapt_alg",type=str,default="TTA3",help="[Tent,PL,PLC,SHOT-IM,T3A,BN,ETA,LAME,ERM,TSD,TTA3,SAFER,TeSLA,AMTDC]",)
    parser.add_argument("--beta", type=float, default=0.9, help="threshold for pseudo label(PL)")
    parser.add_argument("--episodic", action="store_true", help="is episodic or not,default:False")
    parser.add_argument("--steps", type=int, default=1, help="steps of test time, default:1")
    parser.add_argument("--filter_K",type=int,default=100,help="M in T3A/TSD, in [1,5,20,50,100,-1],-1 denotes no selection",)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seeds", type=str, default=None, help="Comma-separated seed list (overrides --seed).")
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
    parser.add_argument(
        "--s_aug_params_per_image",
        type=int,
        default=0,
        choices=[0, 1],
        help="Sample SAFER augmentation parameters per image instead of per batch.",
    )
    parser.add_argument(
        "--s_aug_debug",
        type=int,
        default=0,
        choices=[0, 1],
        help="Print sampled SAFER augmentation pipelines once per run.",
    )
    parser.add_argument(
        "--s_aug_log",
        type=int,
        default=0,
        choices=[0, 1],
        help="Log sampled SAFER augmentation pipelines once per run.",
    )
    parser.add_argument("--s_js_weight", type=float, default=1.0, help="Weight for SAFER JS divergence consistency loss.")
    parser.add_argument("--s_cc_weight", type=float, default=1.0, help="Weight for SAFER cross-correlation loss.")
    parser.add_argument("--s_cc_offdiag", type=float, default=1.0, help="Weight on off-diagonal terms in SAFER cross-correlation loss.")
    parser.add_argument("--s_cc_impl", type=str, default="fast", choices=["fast", "einsum"], help="Cross-correlation implementation: fast pairwise or einsum-based.")
    parser.add_argument("--s_feat_normalize", type=int, default=0, choices=[0,1], help="L2-normalise features before computing SAFER cross-correlation.")
    parser.add_argument("--s_aug_seed", type=int, default=-1, help="Deterministic seed for SAFER augmentation sampling (-1 disables).")
    parser.add_argument(
        "--s_aug_force_noise",
        type=int,
        default=1,
        choices=[0, 1],
        help="Always apply Gaussian noise as the first SAFER augmentation.",
    )
    parser.add_argument(
        "--s_aug_require_freq_blur",
        type=int,
        default=1,
        choices=[0, 1],
        help="Require FFT low-pass or Gaussian blur in each SAFER augmentation pipeline.",
    )
    parser.add_argument(
        "--s_aug_use_noise",
        type=int,
        default=1,
        choices=[0, 1],
        help="Allow Gaussian noise in SAFER augmentations (0 removes it).",
    )
    parser.add_argument(
        "--s_aug_noise_std",
        type=float,
        default=-1.0,
        help="Fixed std for SAFER Gaussian noise (-1 keeps random sampling).",
    )
    parser.add_argument(
        "--s_aug_fixed_op",
        type=str,
        default="none",
        choices=["none", "gaussian_blur", "fft_low_pass"],
        help="Use a fixed SAFER op only (disables other sampled ops).",
    )
    parser.add_argument(
        "--s_aug_fixed_ops",
        type=str,
        nargs="+",
        default=None,
        choices=["none", "gaussian_blur", "fft_low_pass"],
        help="Optional fixed SAFER ops (one per augmented view).",
    )
    parser.add_argument(
        "--s_aug_fixed_blur_kernel",
        type=int,
        default=9,
        help="Kernel size for fixed SAFER Gaussian blur (odd integer).",
    )
    parser.add_argument(
        "--s_aug_fixed_blur_sigma",
        type=float,
        default=1.0,
        help="Sigma for fixed SAFER Gaussian blur.",
    )
    parser.add_argument(
        "--s_aug_fixed_fft_keep_ratio",
        type=float,
        default=0.5,
        help="Keep ratio for fixed SAFER FFT low-pass.",
    )
    parser.add_argument(
        "--s_input_is_normalized",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Override SAFER input normalization detection: -1 auto, 0 raw, 1 normalized.",
    )
    parser.add_argument(
        "--s_cm_weight",
        type=float,
        default=0.0,
        help="Weight for SAFER class-marginal regularisation.",
    )
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
    parser.add_argument(
        "--s_wrap_alg",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use SAFER view pooling with non-SAFER algorithms (1 enables).",
    )
    parser.add_argument(
        "--s_attack_use_views",
        type=int,
        default=0,
        choices=[0, 1],
        help="Use SAFER view pooling during live attacks when adapt_alg=SAFER.",
    )
    # TeSLA
    parser.add_argument("--tesla_sub_policy_dim", type=int, default=2, help="Number of ops per TeSLA sub-policy.")
    parser.add_argument("--tesla_aug_mult", type=int, default=1, help="Number of hard augmentation views per batch.")
    parser.add_argument("--tesla_aug_mult_easy", type=int, default=4, help="Number of easy augmentation views per batch.")
    parser.add_argument("--tesla_lmb_kl", type=float, default=1.0, help="Weight on TeSLA hard-augmentation KL loss.")
    parser.add_argument("--tesla_lmb_norm", type=float, default=1.0, help="Weight on TeSLA augmentation norm loss.")
    parser.add_argument("--tesla_ema", type=float, default=0.99, help="EMA momentum for TeSLA teacher.")
    parser.add_argument("--tesla_no_kl_hard", type=int, default=0, choices=[0,1], help="Disable TeSLA hard-augmentation distillation.")
    parser.add_argument("--tesla_nn_queue_size", type=int, default=256, help="TeSLA nearest-neighbour queue size (0 disables).")
    parser.add_argument("--tesla_n_neigh", type=int, default=10, help="Number of neighbours for TeSLA PLR.")
    parser.add_argument("--tesla_pl_ce", type=int, default=0, choices=[0,1], help="Use CE with teacher as target.")
    parser.add_argument("--tesla_pl_fce", type=int, default=0, choices=[0,1], help="Use CE with student as target.")
    parser.add_argument(
        "--tesla_hard_augment",
        type=str,
        default="optimal",
        choices=["optimal", "aa", "randaugment"],
        help="TeSLA hard augmentation source (optimal policy, AutoAugment, or RandAugment).",
    )
    parser.add_argument(
        "--tesla_input_is_normalized",
        type=int,
        default=-1,
        choices=[-1, 0, 1],
        help="Override input normalization detection: -1 auto, 0 raw, 1 normalized.",
    )
    parser.add_argument(
        "--tesla_view_pool",
        type=str.lower,
        default="mean",
        choices=["mean", "worst", "entropy", "top1", "cc", "cc_drop"],
        help="View pooling strategy for TeSLA teacher view aggregation.",
    )
    parser.add_argument(
        "--tesla_js_weight",
        type=float,
        default=0.0,
        help="Weight for TeSLA JS divergence across easy views.",
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

    parser.add_argument("--attack", type=str, default="linf_eps-8.0_steps-20", help="Attack config suffix or 'clean'.")
    parser.add_argument("--attack_preset", type=str, default=None, help="Named attack preset for on-the-fly attacks.")
    parser.add_argument(
        "--attack_source",
        type=str,
        default="precomputed",
        choices=["precomputed", "on_the_fly", "live"],
        help="Attack source: precomputed tensors, on-the-fly with frozen source model, or live with adapted model.",
    )
    parser.add_argument("--attack_norm", type=str, choices=["linf", "l2"], default=None)
    parser.add_argument("--attack_eps", type=float, default=None)
    parser.add_argument("--attack_steps", type=int, default=None)
    parser.add_argument("--attack_alpha", type=float, default=None)
    parser.add_argument("--attack_fft_rho", type=float, default=None)
    parser.add_argument("--attack_fft_alpha", type=float, default=None)
    parser.add_argument("--eps", type=float, default=4)  
    parser.add_argument("--attack_rate", type=int, choices=[0, 25, 50, 75, 100], default=0)   
    parser.add_argument(
        "--use_adv_source",
        nargs="?",
        const=1,
        default=0,
        type=int,
        choices=[0, 1],
        help="Use adversarially trained source weights (1 enables).",
    )
    parser.add_argument("--adv_source_preset", type=str, default=None, help="Preset name used to locate adv source weights.")
    parser.add_argument("--adv_source_tag", type=str, default=None, help="Override tag for adv source checkpoint filename.")
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
    parser.add_argument(
        "--jpeg_input_backprop",
        type=str,
        default="bpda",
        choices=["exact", "bpda"],
        help="Backprop mode for input JPEG: exact (no grad) or bpda (identity backward).",
    )

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

    attack_id_hint = None
    if args.attack_preset or args.attack != "clean":
        attack_id_hint = args.attack
    attack_cfg = resolve_attack_config(
        preset_name=args.attack_preset,
        attack_id=attack_id_hint,
        norm=args.attack_norm,
        eps=args.attack_eps,
        steps=args.attack_steps,
        alpha=args.attack_alpha,
        fft_rho=args.attack_fft_rho,
        fft_alpha=args.attack_fft_alpha,
    )
    if args.attack == "clean" and not args.attack_preset:
        args.attack_id = "clean"
    else:
        args.attack = attack_cfg.attack_id
        args.attack_id = attack_cfg.attack_id
    args.attack_norm = attack_cfg.norm
    args.attack_eps = attack_cfg.eps
    args.attack_steps = attack_cfg.steps
    args.attack_alpha = attack_cfg.alpha
    args.attack_fft_rho = attack_cfg.fft_rho
    args.attack_fft_alpha = attack_cfg.fft_alpha
    if args.use_adv_source:
        if not args.adv_source_tag:
            preset = args.adv_source_preset or DEFAULT_ATTACK_PRESET
            adv_cfg = resolve_attack_config(preset_name=preset)
            args.adv_source_tag = adv_cfg.attack_id

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
    args.s_wrap_alg = bool(args.s_wrap_alg)
    args.s_attack_use_views = bool(args.s_attack_use_views)
    args.s_aug_force_noise = bool(args.s_aug_force_noise)
    args.s_aug_require_freq_blur = bool(args.s_aug_require_freq_blur)
    args.s_aug_use_noise = bool(args.s_aug_use_noise)
    args.s_aug_params_per_image = bool(args.s_aug_params_per_image)
    args.s_aug_debug = bool(args.s_aug_debug)
    args.s_aug_log = bool(args.s_aug_log)
    if not args.s_aug_use_noise:
        args.s_aug_force_noise = False
    if args.s_aug_noise_std is not None and args.s_aug_noise_std < 0:
        args.s_aug_noise_std = None
    if args.s_aug_fixed_ops:
        fixed_ops = []
        for op in args.s_aug_fixed_ops:
            op_name = op.lower()
            fixed_ops.append(None if op_name == "none" else op_name)
        if args.s_aug_fixed_op is not None and args.s_aug_fixed_op.lower() != "none":
            raise ValueError("s_aug_fixed_ops overrides s_aug_fixed_op; set s_aug_fixed_op=none.")
        if args.s_num_views <= 0:
            raise ValueError("s_num_views must be > 0 when s_aug_fixed_ops is set.")
        if len(fixed_ops) != args.s_num_views:
            raise ValueError("s_aug_fixed_ops length must match s_num_views.")
        args.s_aug_fixed_ops = fixed_ops
        args.s_aug_fixed_op = None
    else:
        if args.s_aug_fixed_op is not None:
            fixed_op = args.s_aug_fixed_op.lower()
            args.s_aug_fixed_op = None if fixed_op == "none" else fixed_op
        if args.s_aug_fixed_op == "gaussian_blur":
            if args.s_aug_fixed_blur_kernel <= 0 or args.s_aug_fixed_blur_kernel % 2 == 0:
                raise ValueError("s_aug_fixed_blur_kernel must be a positive odd integer.")
            if args.s_aug_fixed_blur_sigma <= 0:
                raise ValueError("s_aug_fixed_blur_sigma must be > 0.")
        if args.s_aug_fixed_op == "fft_low_pass":
            if not (0.0 < args.s_aug_fixed_fft_keep_ratio <= 1.0):
                raise ValueError("s_aug_fixed_fft_keep_ratio must be in (0, 1].")
    if args.s_input_is_normalized < 0:
        args.s_input_is_normalized = None
    else:
        args.s_input_is_normalized = bool(args.s_input_is_normalized)
    args.tesla_no_kl_hard = bool(args.tesla_no_kl_hard)
    args.tesla_pl_ce = bool(args.tesla_pl_ce)
    args.tesla_pl_fce = bool(args.tesla_pl_fce)
    args.tesla_hard_augment = args.tesla_hard_augment.lower()
    args.tesla_view_pool = args.tesla_view_pool.lower()
    if args.tesla_input_is_normalized < 0:
        args.tesla_input_is_normalized = None
    else:
        args.tesla_input_is_normalized = bool(args.tesla_input_is_normalized)
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


def _format_fixed_ops(fixed_ops):
    if not fixed_ops:
        return "none"
    return ",".join(op if op is not None else "none" for op in fixed_ops)


def log_args(args, time_taken_s):
    fixed_ops_label = _format_fixed_ops(args.s_aug_fixed_ops)
    wandb.log({
        "adapt_algorithm": args.adapt_alg,
        "attack_rate": args.attack_rate,
        "attack_source": args.attack_source,
        "attack_id": args.attack_id,
        "attack_norm": args.attack_norm,
        "attack_eps": args.attack_eps,
        "attack_steps": args.attack_steps,
        "attack_alpha": args.attack_alpha,
        "attack_fft_rho": args.attack_fft_rho,
        "attack_fft_alpha": args.attack_fft_alpha,
        "use_adv_source": args.use_adv_source,
        "adv_source_tag": args.adv_source_tag,
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
    if args.s_wrap_alg:
        wandb.log({
            "s_wrap_alg": args.s_wrap_alg,
            "s_num_views": args.s_num_views,
            "s_include_original": args.s_include_original,
            "s_aug_prob": args.s_aug_prob,
            "s_aug_max_ops": -1 if args.s_aug_max_ops is None else args.s_aug_max_ops,
            "s_aug_fixed_op": args.s_aug_fixed_op or "none",
            "s_aug_fixed_ops": fixed_ops_label,
            "s_aug_fixed_fft_keep_ratio": args.s_aug_fixed_fft_keep_ratio,
            "s_aug_fixed_blur_kernel": args.s_aug_fixed_blur_kernel,
            "s_aug_fixed_blur_sigma": args.s_aug_fixed_blur_sigma,
            "s_aug_use_noise": args.s_aug_use_noise,
            "s_aug_noise_std": args.s_aug_noise_std,
            "s_aug_params_per_image": args.s_aug_params_per_image,
            "s_sup_view_pool": args.s_sup_view_pool,
            "s_view_weighting": args.s_view_weighting,
        }, commit=False)
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
            "s_aug_force_noise": args.s_aug_force_noise,
            "s_aug_require_freq_blur": args.s_aug_require_freq_blur,
            "s_aug_use_noise": args.s_aug_use_noise,
            "s_aug_noise_std": args.s_aug_noise_std,
            "s_aug_params_per_image": args.s_aug_params_per_image,
            "s_aug_fixed_op": args.s_aug_fixed_op or "none",
            "s_aug_fixed_ops": fixed_ops_label,
            "s_aug_fixed_blur_kernel": args.s_aug_fixed_blur_kernel,
            "s_aug_fixed_blur_sigma": args.s_aug_fixed_blur_sigma,
            "s_aug_fixed_fft_keep_ratio": args.s_aug_fixed_fft_keep_ratio,
            "s_input_is_normalized": args.s_input_is_normalized,
            "s_cm_weight": args.s_cm_weight,
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
    elif args.adapt_alg == "TeSLA":
        wandb.log({
            "tesla_sub_policy_dim": args.tesla_sub_policy_dim,
            "tesla_aug_mult": args.tesla_aug_mult,
            "tesla_aug_mult_easy": args.tesla_aug_mult_easy,
            "tesla_lmb_kl": args.tesla_lmb_kl,
            "tesla_lmb_norm": args.tesla_lmb_norm,
            "tesla_ema": args.tesla_ema,
            "tesla_no_kl_hard": args.tesla_no_kl_hard,
            "tesla_nn_queue_size": args.tesla_nn_queue_size,
            "tesla_n_neigh": args.tesla_n_neigh,
            "tesla_pl_ce": args.tesla_pl_ce,
            "tesla_pl_fce": args.tesla_pl_fce,
            "tesla_hard_augment": args.tesla_hard_augment,
            "tesla_input_is_normalized": args.tesla_input_is_normalized,
            "tesla_view_pool": args.tesla_view_pool,
            "tesla_js_weight": args.tesla_js_weight,
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


def resolve_source_checkpoint(args, dom_id: int) -> str:
    base_dir = os.path.join(
        args.data_file,
        "train_output",
        args.dataset,
        f"test_{str(dom_id)}",
        f"seed_{str(args.seed)}",
    )
    if args.use_adv_source:
        tag = args.adv_source_tag or DEFAULT_ATTACK_PRESET
        return os.path.join(base_dir, f"model_adv_{tag}.pkl")
    return os.path.join(base_dir, "model.pkl")

def adapt_loader(args):
    test_envs = args.test_envs[0]
    domain_name = args.img_dataset[args.dataset][test_envs]
    data_root = os.path.join(args.data_dir, args.img_dataset[args.dataset][test_envs])
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    clean_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    attack_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    if args.attack == "clean":
        testset = ImageFolder(root=data_root, transform=clean_transform)
    elif args.attack_source == "precomputed":
        testset = AttackedImageFolder(
            root=data_root, # normal ImageFolder root
            transform=None,
            adv_root=args.attack_data_dir,
            dataset=args.dataset,
            domain=domain_name,
            config=f"{args.net}_{args.attack}",
            rate=args.attack_rate,                            
            seed=args.seed)
    else:
        testset = ImageFolder(root=data_root, transform=attack_transform)

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.N_WORKERS,
        pin_memory=True,
    )
    return testloader


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

    augment_list = args.s_aug_list if args.s_aug_list else None

    def _build_safer_view_module(model: nn.Module) -> SAFERViewModule:
        return SAFERViewModule(
            num_aug_views=args.s_num_views,
            include_original=args.s_include_original,
            aug_prob=args.s_aug_prob,
            aug_max_ops=args.s_aug_max_ops,
            augmentations=augment_list,
            force_noise_first=args.s_aug_force_noise,
            require_freq_or_blur=args.s_aug_require_freq_blur,
            sample_params_per_image=args.s_aug_params_per_image,
            aug_seed=args.s_aug_seed,
            fixed_ops=args.s_aug_fixed_ops,
            fixed_op=args.s_aug_fixed_op,
            fixed_blur_kernel=args.s_aug_fixed_blur_kernel,
            fixed_blur_sigma=args.s_aug_fixed_blur_sigma,
            fixed_fft_keep_ratio=args.s_aug_fixed_fft_keep_ratio,
            allow_noise=args.s_aug_use_noise,
            noise_std=args.s_aug_noise_std,
            debug=args.s_aug_debug,
            log_pipelines=args.s_aug_log,
            feature_normalize=args.s_feat_normalize,
            view_weighting=args.s_view_weighting,
            primary_view_pool=args.s_sup_view_pool,
            js_weight=0.0,
            js_mode="pooled",
            js_view_pool=args.s_sup_view_pool,
            cc_weight=0.0,
            cc_mode="pairwise",
            cc_view_pool=args.s_sup_view_pool,
            cc_impl=args.s_cc_impl,
            offdiag_weight=args.s_cc_offdiag,
            mean=None,
            std=None,
            input_is_normalized=args.s_input_is_normalized,
            stat_modules=(model.featurizer, model),
        )

    def _maybe_wrap_model(model: nn.Module) -> tuple[nn.Module, Optional[SAFERViewModule]]:
        if not args.s_wrap_alg or args.adapt_alg == "SAFER":
            return model, None
        unsupported = {"TTA3", "TeSLA", "AMTDC"}
        if args.adapt_alg in unsupported:
            raise ValueError(f"s_wrap_alg is not supported for {args.adapt_alg}.")
        view_module = _build_safer_view_module(model)
        if args.adapt_alg in {"T3A", "TSD"}:
            # Keep SAFER view pooling for feature-based algorithms while exposing pooled predict().
            return SAFERPooledPredictor(model, view_module), view_module
        return SAFERPooledPredictor(model, view_module), None

    # set adapt model and optimizer
    if args.adapt_alg == "Tent":
        algorithm = configure_model(algorithm)
        params, _ = collect_params(algorithm)
        optimizer = torch.optim.Adam(params, lr=args.lr)
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = Tent(
            algorithm, optimizer, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "ERM":
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = ERM(algorithm)
    elif args.adapt_alg == "PL":
        optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = PseudoLabel(
            algorithm, optimizer, args.beta, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "PLC":
        optimizer = torch.optim.Adam(algorithm.classifier.parameters(), lr=args.lr)
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = PseudoLabel(
            algorithm, optimizer, args.beta, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "SHOT-IM":
        optimizer = torch.optim.Adam(algorithm.featurizer.parameters(), lr=args.lr)
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = SHOTIM(
            algorithm, optimizer, steps=args.steps, episodic=args.episodic
        )
    elif args.adapt_alg == "T3A":
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = T3A(
            algorithm, filter_K=args.filter_K, steps=args.steps, episodic=args.episodic, view_module=view_module
        )
    elif args.adapt_alg == "BN":
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
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
        elif args.update_param == "tent":
            algorithm = configure_model(algorithm)
            params, _ = collect_params(algorithm)
            optimizer = torch.optim.Adam(params, lr=args.lr)
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
        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
        adapt_model = TSD(
            algorithm,
            optimizer,
            filter_K=args.filter_K,
            steps=args.steps,
            episodic=args.episodic,
            view_module=view_module,
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

        algorithm = wrap_with_input_defense(algorithm, args)
        algorithm, view_module = _maybe_wrap_model(algorithm)
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

        algorithm = wrap_with_input_defense(algorithm, args)
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
            force_noise_first=args.s_aug_force_noise,
            require_freq_or_blur=args.s_aug_require_freq_blur,
            fixed_ops=args.s_aug_fixed_ops,
            fixed_op=args.s_aug_fixed_op,
            fixed_blur_kernel=args.s_aug_fixed_blur_kernel,
            fixed_blur_sigma=args.s_aug_fixed_blur_sigma,
            fixed_fft_keep_ratio=args.s_aug_fixed_fft_keep_ratio,
            allow_noise=args.s_aug_use_noise,
            noise_std=args.s_aug_noise_std,
            js_weight=args.s_js_weight,
            cc_weight=args.s_cc_weight,
            offdiag_weight=args.s_cc_offdiag,
            feature_normalize=args.s_feat_normalize,
            aug_seed=args.s_aug_seed,
            sample_params_per_image=args.s_aug_params_per_image,
            sup_mode=args.s_sup_type,
            sup_weight=args.s_sup_weight,
            class_marginal_weight=args.s_cm_weight,
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
            input_is_normalized=args.s_input_is_normalized,
            debug=args.s_aug_debug,
            log_pipelines=args.s_aug_log,
        )
    elif args.adapt_alg == "TeSLA":
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

        algorithm = wrap_with_input_defense(algorithm, args)
        adapt_model = TeSLA(
            algorithm,
            optimizer,
            steps=args.steps,
            episodic=args.episodic,
            sub_policy_dim=args.tesla_sub_policy_dim,
            aug_mult=args.tesla_aug_mult,
            aug_mult_easy=args.tesla_aug_mult_easy,
            hard_augment=args.tesla_hard_augment,
            lmb_kl=args.tesla_lmb_kl,
            lmb_norm=args.tesla_lmb_norm,
            ema_momentum=args.tesla_ema,
            no_kl_hard=args.tesla_no_kl_hard,
            nn_queue_size=args.tesla_nn_queue_size,
            n_neigh=args.tesla_n_neigh,
            pl_ce=args.tesla_pl_ce,
            pl_fce=args.tesla_pl_fce,
            input_is_normalized=args.tesla_input_is_normalized,
            view_pool=args.tesla_view_pool,
            js_weight=args.tesla_js_weight,
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

        algorithm = wrap_with_input_defense(algorithm, args)
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
    dom_id = args.test_envs[0]
    pretrain_model_path = resolve_source_checkpoint(args, dom_id)
    if not os.path.isfile(pretrain_model_path):
        raise FileNotFoundError(f"Checkpoint not found: {pretrain_model_path}")
    set_random_seed(args.seed)
    
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args)
    algorithm.train()
    algorithm = load_ckpt(algorithm, pretrain_model_path)

    dataloader = adapt_loader(args)
    adapt_model = make_adapt_model(args, algorithm)

    adapt_model.cuda()
    attack_model = None
    if args.attack != "clean" and args.attack_source in ("on_the_fly", "live"):
        if args.attack_source == "on_the_fly":
            attack_model = copy.deepcopy(algorithm)
            attack_model = wrap_with_input_defense(attack_model, args)
            attack_model = attack_model.cuda().eval()
            for param in attack_model.parameters():
                param.requires_grad_(False)
        else:
            if args.adapt_alg == "SAFER" and args.s_attack_use_views:
                base_model = getattr(adapt_model, "model", None)
                view_module = getattr(adapt_model, "view_module", None)
                if base_model is not None and view_module is not None:
                    attack_model = SAFERPooledPredictor(base_model, view_module)
                else:
                    attack_model = getattr(adapt_model, "model", None)
            else:
                attack_model = getattr(adapt_model, "model", None)

    outputs_arr, labels_arr = [], []
    peak_vram_mb = 0.0
    use_cuda = torch.cuda.is_available()
    attack_transform = None
    if attack_model is not None and args.attack_fft_rho < 1.0:
        attack_transform = build_attack_transform(
            fft_rho=args.attack_fft_rho,
            fft_alpha=args.attack_fft_alpha,
            device=torch.device("cuda" if use_cuda else "cpu"),
        )
    if use_cuda:
        torch.cuda.reset_peak_memory_stats()
    attack_rng = torch.Generator(device="cuda" if use_cuda else "cpu")
    attack_rng.manual_seed(args.seed)

    for _, sample in enumerate(dataloader):
        image, label = sample
        image = image.cuda()
        label = label.cuda()
        if attack_model is not None and args.attack_rate > 0:
            if args.attack_rate >= 100:
                image = pgd_attack(
                    attack_model,
                    image,
                    label,
                    args.attack_eps / 255.0,
                    args.attack_alpha / 255.0,
                    args.attack_steps,
                    norm=args.attack_norm,
                    input_transform=attack_transform,
                )
            else:
                mask = torch.rand(
                    (image.size(0),),
                    generator=attack_rng,
                    device=image.device,
                ) < (args.attack_rate / 100.0)
                if mask.any():
                    adv = pgd_attack(
                        attack_model,
                        image[mask],
                        label[mask],
                        args.attack_eps / 255.0,
                        args.attack_alpha / 255.0,
                        args.attack_steps,
                        norm=args.attack_norm,
                        input_transform=attack_transform,
                    )
                    image = image.clone()
                    image[mask] = adv
        logits = adapt_model(image)
        
        outputs = logits.detach().cpu()
        batch_acc = 100*accuracy_score(label.detach().cpu().numpy(), outputs.argmax(1).numpy())
        wandb.log({"batch_acc": batch_acc})
        outputs_arr.append(outputs)
        labels_arr.append(label.detach().cpu())

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
    elif args.adapt_alg == "TeSLA":
        run_name = (
            f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}"
            f"-a{args.tesla_aug_mult}"
            f"-e{args.tesla_aug_mult_easy}"
            f"-k{args.tesla_lmb_kl:.2f}"
            f"-ema{args.tesla_ema:.2f}"
            f"-{args.tesla_hard_augment}"
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

    seed_list = None
    if args.seeds:
        seed_list = [int(s.strip()) for s in args.seeds.split(",") if s.strip() != ""]
    if not seed_list:
        seed_list = [0, 1, 2]

    all_acc   = []
    time1 = time.time()
    for s in seed_list:
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
