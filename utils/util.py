# coding=utf-8
import random
import numpy as np
import torch
import sys
import os
import torchvision
import PIL


def set_random_seed(seed=0):
    # seed setting
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def save_checkpoint(filename, alg, args):
    save_dict = {
        "args": vars(args),
        "model_dict": alg.state_dict()
    }
    torch.save(save_dict, os.path.join(args.output, filename))


def load_ckpt(algorithm,ckpt_dir):
    """
    load pretrain model to adapt
    """
    checkpoint = torch.load(ckpt_dir, weights_only=True)
    missing, unexpected = algorithm.load_state_dict(checkpoint['model_dict'], strict=False)
    return algorithm


def train_valid_target_eval_names(args):
    eval_name_dict = {'train': [], 'valid': [], 'target': []}
    t = 0
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['train'].append(t)
            t += 1
    for i in range(args.domain_num):
        if i not in args.test_envs:
            eval_name_dict['valid'].append(t)
        else:
            eval_name_dict['target'].append(t)
        t += 1
    return eval_name_dict


def alg_loss_dict(args):
    loss_dict = {'ERM': ['class']}
    return loss_dict[args.algorithm]


def print_args(args, print_list):
    s = "==========================================\n"
    l = len(print_list)
    for arg, content in args.__dict__.items():
        if l == 0 or arg in print_list:
            s += "{}:{}\n".format(arg, content)
    return s


def print_environ():
    print("Environment:")
    print("\tPython: {}".format(sys.version.split(" ")[0]))
    print("\tPyTorch: {}".format(torch.__version__))
    print("\tTorchvision: {}".format(torchvision.__version__))
    print("\tCUDA: {}".format(torch.version.cuda))
    print("\tCUDNN: {}".format(torch.backends.cudnn.version()))
    print("\tNumPy: {}".format(np.__version__))
    print("\tPIL: {}".format(PIL.__version__))


class Tee:
    def __init__(self, fname, mode="a"):
        self.stdout = sys.stdout
        self.file = open(fname, mode)

    def write(self, message):
        self.stdout.write(message)
        self.file.write(message)
        self.flush()

    def flush(self):
        self.stdout.flush()
        self.file.flush()
    
    def isatty(self):
        return False


def img_param_init(args):
    dataset = args.dataset
    if dataset == 'office':
        domains = ['amazon', 'dslr', 'webcam']
    elif dataset == 'office-caltech':
        domains = ['amazon', 'dslr', 'webcam', 'caltech']
    elif dataset == 'office-home':
        domains = ['Art', 'Clipart', 'Product', 'RealWorld']
    elif dataset == 'PACS':
        domains = ['art_painting', 'cartoon', 'photo', 'sketch']
    elif dataset == 'VLCS':
        domains = ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007']
    elif dataset == 'DomainNet':
        domains = ['clipart','infograph','painting','quickdraw','real','sketch']
    else:
        print('No such dataset exists!')
    args.domains = domains
    args.img_dataset = {
        'office': ['amazon', 'dslr', 'webcam'],
        'office-caltech': ['amazon', 'dslr', 'webcam', 'caltech'],
        'office-home': ['Art', 'Clipart', 'Product', 'RealWorld'],
        'PACS': ['art_painting', 'cartoon', 'photo', 'sketch'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'DomainNet':['clipart','infograph','painting','quickdraw','real','sketch'],
    }

    args.input_shape = (3, 224, 224)
    if args.dataset == 'office-home':
        args.num_classes = 65
    elif args.dataset == 'office':
        args.num_classes = 31
    elif args.dataset == 'PACS':
        args.num_classes = 7
    elif args.dataset == 'VLCS':
        args.num_classes = 5
    elif args.dataset == 'DomainNet':
        args.num_classes = 345
    return args


def sum_param(model):
    """
    get the number of parameters
    """
    total = sum([param.nelement() for param in model.parameters()])
    return total


def get_config_id(cfg):
    #return f"{cfg.net}_{cfg.attack}_eps-{cfg.eps}_steps-{cfg.steps}"

    def _fmt_float(val):
        val = float(val)
        if abs(val - round(val)) < 1e-8:
            return f"{val:.1f}"
        return f"{val:g}"

    base = f"{cfg.net}_{cfg.attack}_eps-{_fmt_float(cfg.eps)}_steps-{cfg.steps}"

    keep_ratio = getattr(cfg, "fft_rho", 1.0)
    if keep_ratio < 1.0:
        parts = [
            f"rho-{_fmt_float(keep_ratio)}",
            f"a-{_fmt_float(getattr(cfg, 'fft_alpha', 1.0))}",
        ]
        base = f"{base}_" + "_".join(parts)
    return base
