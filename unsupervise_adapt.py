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
from adapt_algorithm import collect_params, configure_model
from adapt_algorithm import PseudoLabel, SHOTIM, T3A, BN, ERM, Tent, TSD, TTA3
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
    parser.add_argument("--adapt_alg",type=str,default="TTA3",help="[Tent,PL,PLC,SHOT-IM,T3A,BN,ETA,LAME,ERM,TSD,TTA3]",)
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

    parser.add_argument("--attack", choices=["linf_eps-8.0_steps-20", "clean", "l2_eps-112.0_steps-100"], default="linf_eps-8.0_steps-20")
    parser.add_argument("--eps", type=float, default=4)  
    parser.add_argument("--attack_rate", type=int, choices=[0,50,100], default=0)   
    parser.add_argument("--lora_r", type=int, default=4)  
    parser.add_argument("--lora_alpha", type=int, default=8)  
    parser.add_argument("--lora_dropout", type=float, default=0.0)  

    parser.add_argument("--svd_input_rank_ratio", type=float, default=1.0, help="Rank ratio for input SVD projection (1.0 disables it).")
    parser.add_argument("--svd_input_mode", choices=["spatial","channel"], default="spatial")
    parser.add_argument("--svd_feat_rank_ratio", type=float, default=1.0, help="proportional rank threshold for feature-map SVD.")
    parser.add_argument('--svd_feat_max_layer', type=int, default=0, choices=[0,1,2,3,4], help="ResNet block at which to end lowrank (0=off)")
    parser.add_argument("--svd_feat_mode", choices=["spatial","channel"], default="spatial")

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
    parser.add_argument('--disable_preset_hparams', type=int, default=0, choices=[0,1], help='Disable auto-selection of preset hyperparameters based on adapt_alg.')

    
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

    assert args.filter_K in [1,5,20,50,100,-1], "filter_K must be in [1,5,20,50,100,-1]"
    assert 0.0 <= args.svd_input_rank_ratio <= 1.0, "svd_input_rank_ratio must be in [0,1]"
    print_environ()
    return args


def log_args(args, time_taken_s):
    wandb.log({
        "adapt_algorithm": args.adapt_alg,
        "attack_rate": args.attack_rate,
        "svd_feat_rank_ratio": args.svd_feat_rank_ratio,
        "svd_feat_max_layer": args.svd_feat_max_layer,
        "svd_feat_mode": args.svd_feat_mode,
        "svd_input_rank_ratio": args.svd_input_rank_ratio,
        "svd_input_mode": args.svd_input_mode,
        "steps": args.steps,
        "lr": args.lr,
        "lam_flat": args.lam_flat,
        "lam_adv": args.lam_adv,
        "lam_cr": args.lam_cr,
        "lam_pl": args.lam_pl,
        "lam_em": args.lam_em,
        "lam_nuc": args.nuc_lambda,
        "lam_recon": args.lam_recon,
        "nuc_kernel": args.nuc_kernel,
        "nuc_top": args.nuc_top,
        "time_taken_s": time_taken_s,
        "lam_reg": args.lam_reg,
        "reg_type": args.reg_type,
        "ema": args.ema,
        "x_lr": args.x_lr,
        "x_steps": args.x_steps
    })

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

    if args.svd_input_rank_ratio < 1.0:
        return SVDLoader(
            testloader,
            rank_ratio=args.svd_input_rank_ratio,
            device="cuda",
            mode=args.svd_input_mode,
            use_ste=False,
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

    return 100*accuracy_score(labels_arr, outputs_arr)


if __name__ == "__main__":
    args = get_args()
    
    output_path = os.path.join(args.output, args.dataset, str(args.test_envs[0]), args.adapt_alg, str(args.attack_rate))


    dom_id = args.test_envs[0]
    run_name = f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}_rate-{args.attack_rate}"

    if args.adapt_alg == "TTA3":
        cr_modifier = ""
        if args.lam_cr >= 1e-8:
            cr_modifier = f"-{args.cr_type}"
        run_name = f"{args.dataset}_dom_{dom_id}_{args.adapt_alg}-{args.lam_flat}-{args.lam_adv}-{args.lam_cr}{cr_modifier}_rate-{args.attack_rate}"

    wandb.init(
        project="tta3_adapt_test",
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
