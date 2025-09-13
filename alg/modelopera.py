# coding=utf-8
import torch
from network import img_network
from typing import Sequence
import torch, torch.nn as nn

def attach_input_standardization(module: nn.Module,
                                 mean: Sequence[float],
                                 std: Sequence[float]) -> nn.Module:
    """
    Registers buffers and a forward_pre_hook to apply: (x - mean)/std
    on the module's first argument. No architecture wrapping, no name changes.
    """
    mean_t = torch.tensor(mean).view(1, len(mean), 1, 1)
    std_t  = torch.tensor(std).view(1, len(std), 1, 1)
    module.register_buffer("_in_mean", mean_t, persistent=False)
    module.register_buffer("_in_std", std_t, persistent=False)

    def _pre_norm(mod, inputs):
        if not inputs:
            return inputs
        x, *rest = inputs
        return ((x - mod._in_mean) / mod._in_std, *rest)

    module.register_forward_pre_hook(_pre_norm, with_kwargs=False)
    return module


def get_fea(args):
    if args.net.startswith('res'):
        nuc_top = getattr(args, 'nuc_top', 0)                  # 0..4
        nuc_k   = getattr(args, 'nuc_kernel', 3)
        nuc_stem= getattr(args, 'nuc_after_stem', False)
        if nuc_top and nuc_top > 0:
            net = img_network.ResBaseNuc(args.net, nuc_top=nuc_top, k=nuc_k,
                                         nuc_after_stem=nuc_stem)
        else:
            net = img_network.ResBase(args.net)
    elif args.net.startswith('vgg'):
        net = img_network.VGGBase(args.net)
    elif args.net.startswith('ViT'):
        net = img_network.ViTBase(args.net)
    elif args.net.startswith('Eff'):
        net = img_network.EfficientBase(args.net)
    elif args.net.startswith('Mix'):
        net = img_network.MLPMixer(args.net)
        
    mean, std = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225) # imagenet stats

    attach_input_standardization(net, mean, std)
    return net

def accuracy(network, loader):
    correct = 0
    total = 0

    network.eval()
    with torch.no_grad():
        for data in loader:
            x = data[0].cuda().float()
            y = data[1].cuda().long()
            p = network.predict(x)

            if p.size(1) == 1:
                correct += (p.gt(0).eq(y).float()).sum().item()
            else:
                correct += (p.argmax(1).eq(y).float()).sum().item()
            total += len(x)
    network.train()
    return correct / total
