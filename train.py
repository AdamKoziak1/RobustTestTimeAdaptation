# coding=utf-8

import os
import sys
import time
import numpy as np
import argparse
import torch
import torch.nn.functional as F

from alg.opt import *
from alg import alg, modelopera
from utils.util import set_random_seed, save_checkpoint, print_args, train_valid_target_eval_names, alg_loss_dict, Tee, img_param_init, print_environ
from utils.attack_presets import resolve_attack_config, DEFAULT_ATTACK_PRESET
from utils.adv_attack import build_attack_transform, pgd_attack
from datautil.getdataloader import get_img_dataloader

# python train.py --output train_output --dataset PACS

import wandb


def get_args():
    parser = argparse.ArgumentParser(description='DG')
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
                        default=50, help="max iterations")
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
                        default="train_output", help='result output path')
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--adv_train', action='store_true', help='Enable adversarial training on source domains.')
    parser.add_argument('--adv_attack_preset', type=str, default=None, help='Named attack preset for adversarial training.')
    parser.add_argument('--adv_attack_norm', type=str, choices=['linf', 'l2'], default=None)
    parser.add_argument('--adv_attack_eps', type=float, default=None)
    parser.add_argument('--adv_attack_steps', type=int, default=None)
    parser.add_argument('--adv_attack_alpha', type=float, default=None)
    parser.add_argument('--adv_attack_fft_rho', type=float, default=None)
    parser.add_argument('--adv_attack_fft_alpha', type=float, default=None)
    parser.add_argument('--adv_attack_rate', type=int, default=100, help='Percent of each batch to attack.')
    args = parser.parse_args()
    args.steps_per_epoch = 100
    args.data_dir = os.path.join(args.data_file,args.data_dir,args.dataset)
    #os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.output = os.path.join(args.output, args.dataset, f"test_{str(args.test_envs[0])}", f"seed_{str(args.seed)}")
    os.makedirs(args.output, exist_ok=True)
    sys.stdout = Tee(os.path.join(args.output, 'out.txt'))
    sys.stderr = Tee(os.path.join(args.output, 'err.txt'))
    args = img_param_init(args)
    if args.adv_train:
        preset = args.adv_attack_preset or DEFAULT_ATTACK_PRESET
        adv_cfg = resolve_attack_config(
            preset_name=preset,
            attack_id=None,
            norm=args.adv_attack_norm,
            eps=args.adv_attack_eps,
            steps=args.adv_attack_steps,
            alpha=args.adv_attack_alpha,
            fft_rho=args.adv_attack_fft_rho,
            fft_alpha=args.adv_attack_fft_alpha,
        )
        args.adv_attack_norm = adv_cfg.norm
        args.adv_attack_eps = adv_cfg.eps
        args.adv_attack_steps = adv_cfg.steps
        args.adv_attack_alpha = adv_cfg.alpha
        args.adv_attack_fft_rho = adv_cfg.fft_rho
        args.adv_attack_fft_alpha = adv_cfg.fft_alpha
        args.adv_attack_tag = adv_cfg.attack_id
    print_environ()
    return args


if __name__ == '__main__':
    args = get_args()
    set_random_seed(args.seed)

    wandb.init(
        project="tta3_train",        
        name=f"{args.dataset}_test-env-{args.test_envs[0]}_s{args.seed}",  
        config=vars(args),                  
    )

    loss_list = alg_loss_dict(args)
    train_loaders, eval_loaders = get_img_dataloader(args)
    eval_name_dict = train_valid_target_eval_names(args)
    algorithm_class = alg.get_algorithm_class(args.algorithm)
    algorithm = algorithm_class(args).cuda()
    algorithm.train()
    opt = get_optimizer(algorithm, args)
    sch = get_scheduler(opt, args)
    adv_transform = None
    adv_rng = None
    if args.adv_train:
        if args.algorithm != "ERM":
            print(f"[warn] adv_train is tuned for ERM; got {args.algorithm}.")
        if args.adv_attack_fft_rho is not None and args.adv_attack_fft_rho < 1.0:
            adv_transform = build_attack_transform(
                fft_rho=args.adv_attack_fft_rho,
                fft_alpha=args.adv_attack_fft_alpha,
                device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            )
        adv_rng = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu")
        adv_rng.manual_seed(args.seed)

    wandb.config.update({
        "num_train_envs": len(train_loaders),
        "steps_per_epoch": args.steps_per_epoch,
        "num_classes": args.num_classes,
        "adv_train": args.adv_train,
        "adv_attack_tag": getattr(args, "adv_attack_tag", None),
        "adv_attack_rate": args.adv_attack_rate,
    })

    s = print_args(args, [])
    print('=======hyper-parameter used========')
    print(s)
    acc_record = {}
    acc_type_list = ['train', 'valid', 'target']
    train_minibatches_iterator = zip(*train_loaders)
    best_valid_acc, target_acc = 0, 0
    ckpt_prefix = "model"
    if args.adv_train:
        ckpt_prefix = f"model_adv_{args.adv_attack_tag}"
    print('===========start training===========')
    sss = time.time()
    for epoch in range(args.max_epoch):                
        for iter_num in range(args.steps_per_epoch):
            minibatches_device = [(data)
                                  for data in next(train_minibatches_iterator)]
            if args.adv_train:
                all_x = torch.cat([data[0].cuda().float() for data in minibatches_device])
                all_y = torch.cat([data[1].cuda().long() for data in minibatches_device])
                if args.adv_attack_rate <= 0:
                    adv_x = all_x
                elif args.adv_attack_rate >= 100:
                    adv_x = pgd_attack(
                        algorithm,
                        all_x,
                        all_y,
                        args.adv_attack_eps / 255.0,
                        args.adv_attack_alpha / 255.0,
                        args.adv_attack_steps,
                        norm=args.adv_attack_norm,
                        input_transform=adv_transform,
                    )
                else:
                    mask = torch.rand(
                        (all_x.size(0),),
                        generator=adv_rng,
                        device=all_x.device,
                    ) < (args.adv_attack_rate / 100.0)
                    if mask.any():
                        adv = pgd_attack(
                            algorithm,
                            all_x[mask],
                            all_y[mask],
                            args.adv_attack_eps / 255.0,
                            args.adv_attack_alpha / 255.0,
                            args.adv_attack_steps,
                            norm=args.adv_attack_norm,
                            input_transform=adv_transform,
                        )
                        adv_x = all_x.clone()
                        adv_x[mask] = adv
                    else:
                        adv_x = all_x
                loss = F.cross_entropy(algorithm.predict(adv_x), all_y)
                opt.zero_grad()
                loss.backward()
                opt.step()
                if sch:
                    sch.step()
                step_vals = {'class': loss.item()}
            else:
                step_vals = algorithm.update(minibatches_device, opt, sch)

            wandb.log({
                f"train_loss": step_vals.get('class', step_vals.get('total', None)),
                "epoch": epoch,
            }, commit=False)
        if (epoch in [int(args.max_epoch*0.7), int(args.max_epoch*0.9)]) and (not args.schuse):
            print('manually descrease lr')
            for params in opt.param_groups:
                params['lr'] = params['lr']*0.1

        if (epoch == (args.max_epoch-1)) or (epoch % args.checkpoint_freq == 0):
            print('===========epoch %d===========' % (epoch))
            s = ''
            for item in loss_list:
                s += (item+'_loss:%.4f,' % step_vals[item])
            print(s[:-1])
            s = ''
            for item in acc_type_list:
                acc_record[item] = np.mean(np.array([modelopera.accuracy(
                    algorithm, eval_loaders[i]) for i in eval_name_dict[item]]))
                s += (item+'_acc:%.4f,' % acc_record[item])
            print(s[:-1])
            
            if acc_record['valid'] > best_valid_acc:
                best_valid_acc = acc_record['valid']
                target_acc = acc_record['target']
                save_checkpoint(f'{ckpt_prefix}.pkl', algorithm, args)
            if args.save_model_every_checkpoint:
                save_checkpoint(f'{ckpt_prefix}_epoch{epoch}.pkl', algorithm, args)
            print('total cost time: %.4f' % (time.time()-sss))
            algorithm_dict = algorithm.state_dict()
            wandb.log({
                "epoch": epoch,
                "lr": opt.param_groups[0]['lr'],
                "acc_train": acc_record['train'],
                "acc_val": acc_record['valid'],
                "acc_target": acc_record['target'],
            }, commit=True)
    save_checkpoint(f'{ckpt_prefix}_last.pkl', algorithm, args)

    print('valid acc: %.4f' % best_valid_acc)
    print('DG result: %.4f' % target_acc)

    wandb.summary["best_valid_acc"] = best_valid_acc
    wandb.summary["final_target_acc"] = target_acc

    with open(os.path.join(args.output, 'done.txt'), 'w') as f:
        f.write('done\n')
        f.write('total cost time:%s\n' % (str(time.time()-sss)))
        f.write('valid acc:%.4f\n' % (best_valid_acc))
        f.write('target acc:%.4f' % (target_acc))
