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
import numpy as np
from sklearn.metrics import confusion_matrix

from alg.opt import *
from alg import alg
from utils.util import set_random_seed, Tee, img_param_init, print_environ, load_ckpt
from adapt_algorithm import collect_params, configure_model
from adapt_algorithm import PseudoLabel, SHOTIM, T3A, BN, ERM, Tent, TSD
from adv.attacked_imagefolder import AttackedImageFolder

import wandb

def get_args():
    parser = argparse.ArgumentParser(description="Test time adaptation")
    parser.add_argument("--alpha", type=float, default=1, help="DANN dis alpha")
    parser.add_argument(
        "--anneal_iters",
        type=int,
        default=500,
        help="Penalty anneal iters used in VREx",
    )
    parser.add_argument("--beta1", type=float, default=0.5, help="Adam hyper-param")
    parser.add_argument(
        "--checkpoint_freq", type=int, default=3, help="Checkpoint every N epoch"
    )
    parser.add_argument(
        "--classifier", type=str, default="linear", choices=["linear", "wn"]
    )
    parser.add_argument("--data_file", type=str, default="/home/adam/Downloads/RobustTestTimeAdaptation/", help="root_dir")
    parser.add_argument(
        "--dis_hidden", type=int, default=256, help="dis hidden dimension"
    )
    parser.add_argument(
        "--gpu_id", type=str, nargs="?", default="0", help="device id to run"
    )
    parser.add_argument("--groupdro_eta", type=float, default=1, help="groupdro eta")
    parser.add_argument(
        "--inner_lr", type=float, default=1e-2, help="learning rate used in MLDG"
    )
    parser.add_argument(
        "--lam", type=float, default=1, help="tradeoff hyperparameter used in VREx"
    )

    parser.add_argument("--lr_decay", type=float, default=0.75, help="for sgd")
    parser.add_argument(
        "--lr_decay1", type=float, default=1.0, help="for pretrained featurizer"
    )
    parser.add_argument(
        "--lr_decay2",
        type=float,
        default=1.0,
        help="inital learning rate decay of network",
    )
    parser.add_argument("--lr_gamma", type=float, default=0.0003, help="for optimizer")
    parser.add_argument("--max_epoch", type=int, default=120, help="max epoch")
    parser.add_argument(
        "--mixupalpha", type=float, default=0.2, help="mixup hyper-param"
    )
    parser.add_argument("--mldg_beta", type=float, default=1, help="mldg hyper-param")
    parser.add_argument(
        "--mmd_gamma", type=float, default=1, help="MMD, CORAL hyper-param"
    )
    parser.add_argument("--momentum", type=float, default=0.9, help="for optimizer")
    parser.add_argument("--N_WORKERS", type=int, default=4)
    parser.add_argument(
        "--rsc_f_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument(
        "--rsc_b_drop_factor", type=float, default=1 / 3, help="rsc hyper-param"
    )
    parser.add_argument("--save_model_every_checkpoint", action="store_true")
    parser.add_argument("--schuse", action="store_true")
    parser.add_argument("--schusech", type=str, default="cos")

    parser.add_argument(
        "--split_style",
        type=str,
        default="strat",
        help="the style to split the train and eval datasets",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="img_dg",
        choices=["img_dg"],
        help="now only support image tasks",
    )
    parser.add_argument("--tau", type=float, default=1, help="andmask tau")
    parser.add_argument("--weight_decay", type=float, default=5e-4)
    parser.add_argument("--algorithm", type=str, default="ERM")
    parser.add_argument(
        "--batch_size", type=int, default=64, help="batch_size of **test** time"
    )
    parser.add_argument(
        "--dataset", type=str, default="PACS", help="office-home,PACS,VLCS,DomainNet"
    )
    parser.add_argument(
        "--data_dir", type=str, default="datasets", help="data dir"
    )
    parser.add_argument(
        "--attack_data_dir", type=str, default="/home/adam/Downloads/RobustTestTimeAdaptation/datasets_adv", help="attacked data dir"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-4,
        help="learning rate of **test** time adaptation,important",
    )
    parser.add_argument(
        "--net",
        type=str,
        default="resnet18",
        help="featurizer: vgg16, resnet18,resnet50, resnet101,DTNBase,ViT-B16,resnext50",
    )
    parser.add_argument(
        "--test_envs", type=int, nargs="+", default=[0], help="target domains"
    )
    parser.add_argument(
        "--output", type=str, default="./tta_output", help="result output path"
    )
    parser.add_argument(
        "--adapt_alg",
        type=str,
        default="TTA3",
        help="[Tent,PL,PLC,SHOT-IM,T3A,BN,ETA,LAME,ERM,TSD,TTA3]",
    )
    parser.add_argument(
        "--beta", type=float, default=0.9, help="threshold for pseudo label(PL)"
    )
    parser.add_argument(
        "--episodic", action="store_true", help="is episodic or not,default:False"
    )
    parser.add_argument(
        "--steps", type=int, default=1, help="steps of test time, default:1"
    )
    parser.add_argument(
        "--filter_K",
        type=int,
        default=100,
        help="M in T3A/TSD, \in [1,5,20,50,100,-1],-1 denotes no selectiion",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--source_seed", type=int, default=0, help="source model seed")
    parser.add_argument(
        "--update_param", type=str, default="all", help="all / affine / body / head"
    )
    # two hpyer-parameters for EATA (ICML22)
    parser.add_argument(
        "--e_margin",
        type=float,
        default=math.log(7) * 0.40,
        help="entropy margin E_0 in Eqn. (3) for filtering reliable samples",
    )
    parser.add_argument(
        "--d_margin",
        type=float,
        default=0.05,
        help="\epsilon in Eqn. (5) for filtering redundant samples",
    )
    parser.add_argument("--lambda1", type=float, default=1.0, help="Coefficient for Flatness Loss")
    parser.add_argument("--lambda2", type=float, default=1.0, help="Coefficient for Adversarial Loss")
    parser.add_argument("--lambda3", type=float, default=1.0, help="Coefficient for Consistency Regularization Loss")
    parser.add_argument("--l_adv_iter", type=int, default=1, help="Number of iterations for instance-level flatness")
    parser.add_argument("--attack", choices=["linf_eps-8_steps-20", "clean"], default="linf_eps-8_steps-20")
    parser.add_argument("--eps", type=float, default=4)  
    parser.add_argument("--attack_rate", type=int, choices=[0,10,20,30,40,50,60,70,80,90,100], default=100)   
    parser.add_argument("--mask_id", type=int, choices=[0,1,2,3,4], default=0)   
    parser.add_argument("--cr_type", type=str, choices=['cosine', 'l2'], default='l2')   


    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir =  os.path.join(args.data_file, args.data_dir)

    #args.output = os.path.join(args.output, args.dataset, str(args.test_envs[0]), args.adapt_alg, str(args.attack_rate), str(args.mask_id))
    os.environ["CUDA_VISIBLE_DEVICS"] = args.gpu_id
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    #os.makedirs(args.output, exist_ok=True)
    #sys.stdout = Tee(os.path.join(args.output, "out.txt"))
    #sys.stderr = Tee(os.path.join(args.output, "err.txt"))
    args = img_param_init(args)

    assert args.filter_K in [
        1,
        5,
        20,
        50,
        100,
        -1,
    ], "filter_K must be in [1,5,20,50,100,-1]"
    print_environ()
    return args


def adapt_loader(args):
    """
    easy dataloader
    """
    # transform
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    test_transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor(), normalize]
    )
    # data
    test_envs = args.test_envs[0]
    data_root = os.path.join(args.data_dir, args.img_dataset[args.dataset][test_envs])
    #print(data_root)
    if args.attack == "clean":
        testset = ImageFolder(root=data_root, transform=test_transform)
    else:
        testset = AttackedImageFolder(
            root=data_root,                      # normal ImageFolder root
            #transform=test_transform,
            transform=None,
            adv_root=args.attack_data_dir,
            dataset=args.dataset,
            domain=test_envs,
            config=f"{args.net}_{args.attack}",
            rate=args.attack_rate,                            
            mask_idx=args.mask_id)   

    testloader = DataLoader(
        testset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.N_WORKERS,
        pin_memory=True,
    )
    return testloader

def configure_model_and_optimizer(args, algorithm):
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
        optimizer = torch.optim.Adam(algorithm.parameters(), lr=args.lr)
        from adapt_algorithm import TTA3
        adapt_model = TTA3(
            algorithm,
            optimizer,
            steps=args.steps,
            episodic=args.episodic,
            #episodic=True,
            lambda1=args.lambda1,
            lambda2=args.lambda2,
            lambda3=args.lambda3,
            l_adv_iter=args.l_adv_iter,
            cr_type = args.cr_type,
            r=args.eps,
        )
    return adapt_model

if __name__ == "__main__":
    args = get_args()
    set_random_seed(args.seed)

    run_name = f"{args.dataset}_dom_{args.test_envs[0]}_{args.adapt_alg}_rate-{args.attack_rate}_mask_{args.mask_id}"

    if args.adapt_alg == "TTA3":
        cr_modifier = ""
        if args.lambda3 >= 1e-8:
            cr_modifier = f"-{args.cr_type}"
        run_name = f"{args.dataset}_{args.adapt_alg}-{args.lambda1}-{args.lambda2}-{args.lambda3}{cr_modifier}_rate-{args.attack_rate}"
        
    wandb.init(
        project="tta3_grid",    
        name=run_name,
        config=vars(args),
    )

    labels_arr = []
    outputs_arr = []
    time_global_start = time.time()


    data_dir_outer = args.data_dir
    #for dataset in ["PACS", "VLCS", "office-home"]:
    #print(dataset)
    #args.dataset = dataset
    args = img_param_init(args)
    args.data_dir = os.path.join(data_dir_outer, args.dataset)
    time1 = time.time()
    for dom_id, domain in enumerate(args.domains):
        args.test_envs = [dom_id]
        pretrain_model_path = os.path.join(args.data_file, "TSD-master", "code", "train_output", args.dataset, str(args.test_envs[0]), "model.pkl")
        #print(dom_id, domain)

        algorithm_class = alg.get_algorithm_class(args.algorithm)
        algorithm = algorithm_class(args)
        algorithm.train()
        algorithm = load_ckpt(algorithm, pretrain_model_path)

        dataloader = adapt_loader(args)

        adapt_model = configure_model_and_optimizer(args, algorithm)
        adapt_model.cuda()
        #total, correct = 0, 0
        #acc_arr = []
        #print(len(dataloader))
        for idx, sample in enumerate(dataloader):
            
            #print(sample[0].shape)
            #print(idx)
            image, label = sample
            image = image.cuda()
            logits = adapt_model(image)
            outputs_arr.append(logits.detach().cpu())
            labels_arr.append(label)

    total_time = time.time() - time_global_start
    labels_np = torch.cat(labels_arr).numpy()
    logits_np = torch.cat(outputs_arr).numpy()
    preds     = logits_np.argmax(1)
    acc = 100.0 * (preds == labels_np).mean()

    wandb.log({
        "acc_overall" : round(acc, 2),
        "time_taken_s": total_time,
        "lambda1": args.lambda1,
        "lambda2": args.lambda2,
        "lambda3": args.lambda3,
        "l_adv_iter": args.l_adv_iter,
        "cr_type": args.cr_type,
        "r": args.eps,
    })

    wandb.finish()



    #wandb agent bigslav/RobustTestTimeAdaptation-TSD-master_code/m21c4mcw
