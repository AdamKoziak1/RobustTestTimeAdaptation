# coding=utf-8
import torch
from network import img_network
from utils.util import attach_input_standardization

def get_fea(args):
    if args.net.startswith('res'):
        # Optional nuclear-conv instrumentation
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
