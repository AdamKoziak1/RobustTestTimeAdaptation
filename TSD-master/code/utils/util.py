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
    algorithm.load_state_dict(checkpoint['model_dict'])
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
    loss_dict = {'ANDMask': ['total'],
                 'CORAL': ['class', 'coral', 'total'],
                 'DANN': ['class', 'dis', 'total'],
                 'ERM': ['class'],
                 'Mixup': ['class'],
                 'MLDG': ['total'],
                 'MMD': ['class', 'mmd', 'total'],
                 'GroupDRO': ['group'],
                 'RSC': ['class'],
                 'VREx': ['loss', 'nll', 'penalty']
                 }
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
    elif dataset == 'dg5':
        domains = ['mnist', 'mnist_m', 'svhn', 'syn', 'usps']
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
        'dg5': ['mnist', 'mnist_m', 'svhn', 'syn', 'usps'],
        'VLCS': ['Caltech101', 'LabelMe', 'SUN09', 'VOC2007'],
        'DomainNet':['clipart','infograph','painting','quickdraw','real','sketch'],
    }
    if dataset == 'dg5':
        args.input_shape = (3, 32, 32)
        args.num_classes = 10
    else:
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
    return f"{cfg.net}_{cfg.attack}_eps-{cfg.eps}_steps-{cfg.steps}"


class SVDLoader:
    def __init__(self, dataloader, k, device="cuda", full_decomposition=False):
        self.loader, self.k, self.device = dataloader, k, device
        self.full_decomposition = full_decomposition

    def __iter__(self):
        for xb, yb in self.loader:
            xb = xb.to(self.device, non_blocking=True)
            if self.k:
                xb = drop_low_singular_values(xb, self.k, full_decomposition=self.full_decomposition)
            yield xb, yb                          


@torch.no_grad()
def drop_low_singular_values(x: torch.Tensor, k: int, full_decomposition=False) -> torch.Tensor:
    if k == 0:
        return x                           

    B, C, H, W = x.shape      
    x_flat = x.reshape(B * C, H, W)
    
    if full_decomposition:
        U, S, Vh = torch.linalg.svd(x_flat, full_matrices=True)

        # Zero‑out the k smallest σ ‑ vectorised
        if k > 0:
            S[..., -k:] = 0

        x_recon = (U * S.unsqueeze(-1)) @ Vh

    else:
        q        = H - k
        U,S,Vh   = torch.svd_lowrank(x_flat, q=q, niter=2)
        x_recon = torch.matmul(U * S.unsqueeze(1), torch.transpose(Vh, 1, 2))

    return x_recon.reshape(B, C, H, W)