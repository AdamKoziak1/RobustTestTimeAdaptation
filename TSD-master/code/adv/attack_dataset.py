# coding=utf‑8
from torchvision.datasets import ImageFolder
import os, torch, numpy as np

class AttackAwareDataset(ImageFolder):
    """
    Behaves like ImageFolder but transparently swaps in the adversarial tensor
    when mask[idx] == True.
    """
    def __init__(self, root, transform, adv_root, domain):
        super().__init__(root, transform=transform)
        self.adv_root , self.cln_root = os.path.join(adv_root, domain+"_adv"), os.path.join(adv_root, domain+"_clean")
        self.mask = np.load(os.path.join(adv_root, f"{domain}_mask.npy"))

    def __getitem__(self, idx):
        path, target = self.samples[idx]
        if self.mask[idx]:
            img = torch.load(os.path.join(self.adv_root,  f"{idx}.pt"))
        else:
            img = torch.load(os.path.join(self.cln_root, f"{idx}.pt"))
        if self.transform: img = self.transform(img)
        return img, target
