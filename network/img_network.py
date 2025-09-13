# coding=utf-8
import torch.nn as nn
from torchvision import models
import torchvision
import torch
import timm  #load ViT or MLP-mixer
from network.common_network import Identity
import math

vgg_dict = {"vgg11": models.vgg11, "vgg13": models.vgg13, "vgg16": models.vgg16, "vgg19": models.vgg19,
            "vgg11bn": models.vgg11_bn, "vgg13bn": models.vgg13_bn, "vgg16bn": models.vgg16_bn, "vgg19bn": models.vgg19_bn}


class VGGBase(nn.Module):
    def __init__(self, vgg_name):
        super(VGGBase, self).__init__()
        model_vgg = vgg_dict[vgg_name](pretrained=True)
        self.features = model_vgg.features
        self.classifier = nn.Sequential()
        for i in range(6):
            self.classifier.add_module(
                "classifier"+str(i), model_vgg.classifier[i])
        self.in_features = model_vgg.classifier[6].in_features

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


res_dict = {"resnet18": models.resnet18, 
            "resnet34": models.resnet34, 
            "resnet50": models.resnet50,
            "resnet101": models.resnet101, 
            "resnet152": models.resnet152, 
            "resnext50": models.resnext50_32x4d, 
            "resnext101": models.resnext101_32x8d}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        #model_resnet = res_dict[res_name](pretrained=True)
        model_resnet = res_dict[res_name](weights=models.ResNet18_Weights.IMAGENET1K_V1)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x


class DTNBase(nn.Module):
    def __init__(self):
        super(DTNBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.Dropout2d(0.1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(128),
            nn.Dropout2d(0.3),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.Dropout2d(0.5),
            nn.ReLU()
        )
        self.in_features = 256*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class LeNetBase(nn.Module):
    def __init__(self):
        super(LeNetBase, self).__init__()
        self.conv_params = nn.Sequential(
            nn.Conv2d(1, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.Dropout2d(p=0.5),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.in_features = 50*4*4

    def forward(self, x):
        x = self.conv_params(x)
        x = x.view(x.size(0), -1)
        return x


class ViTBase(nn.Module):
    def __init__(self,model_name):
        self.KNOWN_MODELS = {
        'ViT-B16': 'vit_base_patch16_224_in21k', 
        'ViT-B32': 'vit_base_patch32_224_in21k',
        'ViT-L16': 'vit_large_patch16_224_in21k',
        'ViT-L32': 'vit_large_patch32_224_in21k',
        'ViT-H14': 'vit_huge_patch14_224_in21k'
    }
    
        self.FEAT_DIM = {
        'ViT-B16': 768, 
        'ViT-B32': 768,
        'ViT-L16': 1024,
        'ViT-L32': 1024,
        'ViT-H14': 1280
    }    
        super().__init__()
        self.vit_backbone = timm.create_model(self.KNOWN_MODELS[model_name],pretrained=True,num_classes=0)
        self.in_features = self.FEAT_DIM[model_name]
    
    def forward(self,x):
        return self.vit_backbone(x)
        


effnet_dict = {"efficientnet_b0": models.efficientnet_b0, 
         "efficientnet_b1": models.efficientnet_b1,
         "efficientnet_b2": models.efficientnet_b2,
         "efficientnet_b3": models.efficientnet_b3,
         "efficientnet_b4": models.efficientnet_b4,
         "efficientnet_b5": models.efficientnet_b5,
         "efficientnet_b6": models.efficientnet_b6,
         "efficientnet_b7": models.efficientnet_b7}


class EfficientBase(nn.Module):
    def __init__(self,backbone="efficientnet_b4"):
        super().__init__()
        self.network = effnet_dict[backbone](pretrained=True)
        self.in_features = self.network.classifier[1].in_features
        self.network.classifier = Identity()
        
        
    def forward(self,x):
        return self.network(x)



class NuclearConv2d(nn.Module):
    """
    Depthwise 3x3 conv initialized as (near-)identity.
    A linear, per-channel spatial filter so the nuclear penalty is well-defined.
    """
    def __init__(self, channels: int, k: int = 3, bias: bool = False):
        super().__init__()
        assert k % 2 == 1, "kernel size should be odd"
        self.k = k
        self.conv = nn.Conv2d(channels, channels, kernel_size=k, padding=k//2,
                              groups=channels, bias=bias)
        # identity init
        with torch.no_grad():
            self.conv.weight.zero_()
            self.conv.weight[:, :, k//2, k//2] = 1.0
            if bias:
                self.conv.bias.zero_()

    def forward(self, x):
        return self.conv(x)

from typing import Iterable, List

class ResBaseNuc(nn.Module):
    """
    ResNet backbone + optional NuclearConv2d inserted AFTER layer1..layer4 (and/or after the stem).
    The 'nuc_top' knob chooses how high (from the bottom) we insert these layers:
        nuc_top = 3 : after layer1, layer2, layer3
    Optionally, set nuc_after_stem=True to also insert one after the stem (maxpool).

    During forward() we accumulate a nuclear-norm penalty on the outputs of each inserted conv.
    The penalty can be consumed via pop_nuc_penalty().
    """
    def __init__(self, res_name: str, nuc_top: int = 0, k: int = 3,
                 nuc_after_stem: bool = False):
        super().__init__()
        base = res_dict[res_name](weights=models.ResNet18_Weights.IMAGENET1K_V1
                                  if res_name == "resnet18" else None)
        # expose the usual pieces
        self.conv1, self.bn1, self.relu, self.maxpool = base.conv1, base.bn1, base.relu, base.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = base.layer1, base.layer2, base.layer3, base.layer4
        self.avgpool = base.avgpool
        self.in_features = base.fc.in_features

        # figure out channel dims at stage outputs
        # figure out channel dims at stage outputs (robust for BasicBlock/Bottleneck)
        def _stage_out_channels(layer: nn.Sequential) -> int:
            last_block = layer[-1]
            if hasattr(last_block, "bn3"):       # Bottleneck (50/101/152, resnext)
                return last_block.bn3.num_features
            if hasattr(last_block, "bn2"):       # BasicBlock (18/34)
                return last_block.bn2.num_features
            # Fallback: scan backwards for the last BN2d inside the block
            for m in reversed(list(last_block.modules())):
                if isinstance(m, nn.BatchNorm2d):
                    return m.num_features
            raise RuntimeError(f"Could not infer channels for {type(last_block).__name__}")

        ch = {
            "stem": self.bn1.num_features,       # channels after stem BN / maxpool
            "l1":   _stage_out_channels(self.layer1),
            "l2":   _stage_out_channels(self.layer2),
            "l3":   _stage_out_channels(self.layer3),
            "l4":   _stage_out_channels(self.layer4),
        }

        self.nuc_after_stem = nuc_after_stem
        self.nuc_top = int(nuc_top)
        assert 0 <= self.nuc_top <= 4, "nuc_top must be in [0,4]"

        # Build convs
        self.nuc_stem = NuclearConv2d(ch["stem"], k) if self.nuc_after_stem else None
        self.nuc_l1 = NuclearConv2d(ch["l1"], k) if self.nuc_top >= 1 else None
        self.nuc_l2 = NuclearConv2d(ch["l2"], k) if self.nuc_top >= 2 else None
        self.nuc_l3 = NuclearConv2d(ch["l3"], k) if self.nuc_top >= 3 else None
        self.nuc_l4 = NuclearConv2d(ch["l4"], k) if self.nuc_top >= 4 else None

        # Freeze ALL base params by default; user can optimize only these convs
        for p in self.parameters():
            p.requires_grad_(False)
        for m in self.nuc_modules():
            for p in m.parameters():
                p.requires_grad_(True)

        # accumulator for the nuclear penalty (scalar tensor on correct device)
        self._nuc_penalty = None
        self._recon_penalty = None

    # ----- utilities -----
    def nuc_modules(self) -> List[nn.Module]:
        mods = []
        if self.nuc_after_stem and self.nuc_stem is not None: mods.append(self.nuc_stem)
        if self.nuc_l1 is not None: mods.append(self.nuc_l1)
        if self.nuc_l2 is not None: mods.append(self.nuc_l2)
        if self.nuc_l3 is not None: mods.append(self.nuc_l3)
        if self.nuc_l4 is not None: mods.append(self.nuc_l4)
        return mods

    def nuc_parameters(self) -> Iterable[nn.Parameter]:
        for m in self.nuc_modules():
            yield from m.parameters()

    @staticmethod
    def _batch_nuclear_norm(feat: torch.Tensor) -> torch.Tensor:
        """
        feat: (B, C, H, W). We compute ||F||_* with matrix F = C x (H*W), batched over B.
        Returns mean over batch (scalar).
        """
        B, C, H, W = feat.shape
        M = feat.view(B, C, H * W)                    # (B, C, HW)
        # svdvals is batched over B
        s = torch.linalg.svdvals(M)                   # (B, min(C, HW))
        nuc_norm = (s.sum(-1)/(C*H*W)).mean()   
        return nuc_norm              

    def _accumulate(self, x: torch.Tensor, x_after: torch.Tensor):
        nuc= self._batch_nuclear_norm(x)
        diff = x_after - x
        recon = torch.sqrt(diff.pow(2).flatten(1).sum(dim=1) + 1e-12).mean()
        if self._nuc_penalty is None:
            self._nuc_penalty = nuc
            self._recon_penalty = recon
        else:
            self._nuc_penalty = self._nuc_penalty + nuc
            self._recon_penalty = self._recon_penalty + recon

    @torch.no_grad()
    def pop_nuc_penalty(self) -> torch.Tensor:
        if self._nuc_penalty is None:
            return torch.zeros((), device=next(self.parameters()).device)
        n = self._nuc_penalty
        r = self._recon_penalty
        self._nuc_penalty = None
        self._recon_penalty = None
        return n, r

    # ----- forward with taps after each stage -----
    def forward(self, x):
        self._nuc_penalty = None
        x = self.conv1(x); x = self.bn1(x); x = self.relu(x); x = self.maxpool(x)
        if self.nuc_after_stem:
            x_before = x.detach()
            x = self.nuc_stem(x); self._accumulate(x_before, x)
        x = self.layer1(x)
        if self.nuc_l1 is not None:
            x_before = x.detach()
            x = self.nuc_l1(x); self._accumulate(x_before, x)
        x = self.layer2(x)
        if self.nuc_l2 is not None:
            x_before = x.detach()
            x = self.nuc_l2(x); self._accumulate(x_before, x)
        x = self.layer3(x)
        if self.nuc_l3 is not None:
            x_before = x.detach()
            x = self.nuc_l3(x); self._accumulate(x_before, x)
        x = self.layer4(x)
        if self.nuc_l4 is not None:
            x_before = x.detach()
            x = self.nuc_l4(x); self._accumulate(x_before, x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x

class SVDDrop2D(nn.Module):
    def __init__(self, rank_ratio: float, mode: str, full: bool = False):
        super().__init__()
        assert mode in ('spatial', 'channel')
        assert rank_ratio >= 0.0 and rank_ratio <= 1.0
        self.rank_ratio = rank_ratio
        self.mode = mode
        self.full = full

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        if self.mode == 'spatial':
            x_flat = x.reshape(B * C, H, W)
            rank = math.ceil(H * self.rank_ratio)
        if self.mode == 'channel':
            x_flat = x.reshape(B, C, H * W)
            rank = math.ceil(C * self.rank_ratio)

        if self.full:
            U, S, Vh = torch.linalg.svd(x_flat, full_matrices=True)
            #S[..., -rank:] = 0
            S[..., :rank] = 0
            x_recon = (U * S.unsqueeze(-1)) @ Vh

        else:
            U,S,Vh   = torch.svd_lowrank(x_flat, q=rank, niter=2)
            x_recon = torch.matmul(U * S.unsqueeze(1), torch.transpose(Vh, 1, 2))

        return x_recon.reshape(B, C, H, W)
        
        # RESIDUAL?
        # HOW TO RECOMBINE: WEIGHTED SUM? LEARNABLE?
