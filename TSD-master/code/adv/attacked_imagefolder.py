# coding=utf-8
"""
ds = AttackAwareDataset(
        root=data_root,                      # normal ImageFolder root
        transform=test_tf,
        adv_root="../../datasets_adv",
        dataset="PACS",
        domain="art_painting",
        config="resnet18_linf_eps-8_steps-20",
        rate=70,                             # 70 % of samples attacked
        mask_idx=3)                          # 4th random vector
"""

from torchvision.datasets import ImageFolder
import os, torch, numpy as np

class AttackedImageFolder(ImageFolder):
    def __init__(self, *,
                 root, transform,
                 adv_root,      # path to datasets_adv
                 dataset,       # "PACS", "office-home", …
                 domain,        # e.g. "photo"
                 config,        # configuration_id that holds the adversarial tensors
                 rate=100,       # attack percentage
                 seed=0):  

        super().__init__(root, transform=transform)

        # Absolute locations of clean & adversarial tensors
        self.cln_root = os.path.join(adv_root, f"seed_{seed}", dataset, "clean", domain)
        self.adv_root = os.path.join(adv_root, f"seed_{seed}", dataset, config, domain)

        rng = np.random.default_rng(seed)
        self.mask = rng.random(len(self.samples)) < (rate / 100.0)

        assert len(self.mask) == len(self.samples), \
            f"Mask length {len(self.mask)} ≠ dataset length {len(self.samples)}"

    def __getitem__(self, idx):
        _, target = self.samples[idx]
        tensor_name = f"{idx}.pt"
        # choose clean or adversarial tensor
        if self.mask[idx]:
            tensor_path = os.path.join(self.adv_root, tensor_name)
        else:
            tensor_path = os.path.join(self.cln_root, tensor_name)

        img = torch.load(tensor_path, weights_only=True)
        return img, target

# class AttackedImageFolder(ImageFolder):
#     def __init__(self, *,
#                  root, transform,
#                  adv_root,      # path to datasets_adv
#                  dataset,       # "PACS", "office-home", …
#                  domain,        # e.g. "photo"
#                  config,        # configuration_id that holds the adversarial tensors
#                  rate=100,       # attack percentage
#                  mask_idx=0):   # which of the 5 random vectors to use

#         super().__init__(root, transform=transform)

#         # Absolute locations of clean & adversarial tensors
#         self.cln_root = os.path.join(adv_root, dataset, "clean", domain)
#         self.adv_root = os.path.join(adv_root, dataset, config, domain)

#         # Load the indicator vector
#         mask_path = os.path.join(
#             adv_root, dataset, "masks", str(rate), f"{domain}_mask_{mask_idx}.npy")
#         if not os.path.isfile(mask_path):
#             raise FileNotFoundError(f"mask file not found: {mask_path}")
#         self.mask = np.load(mask_path)

#         assert len(self.mask) == len(self.samples), \
#             f"Mask length {len(self.mask)} ≠ dataset length {len(self.samples)}"

#     def __getitem__(self, idx):
#         path, target = self.samples[idx]
#         tensor_name = f"{idx}.pt"
#         # choose clean or adversarial tensor
#         if self.mask[idx]:
#             tensor_path = os.path.join(self.adv_root, tensor_name)
#         else:
#             tensor_path = os.path.join(self.cln_root, tensor_name)

#         img = torch.load(tensor_path)
#         return img, target
