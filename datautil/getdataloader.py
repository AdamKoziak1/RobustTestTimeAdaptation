# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader
import torch

def get_img_dataloader(args):
    rate = 0.2
    trdatalist, tedatalist = [], []

    names = args.img_dataset[args.dataset]
    args.domain_num = len(names)
    for i in range(len(names)):
        if i in args.test_envs:
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(), test_envs=args.test_envs))
        else:
            tmpdatay = ImageDataset(args.dataset, args.task, args.data_dir,
                                    names[i], i, transform=imgutil.image_train(), test_envs=args.test_envs).labels
            l = len(tmpdatay)
            if args.split_style == 'strat':
                lslist = np.arange(l)
                stsplit = ms.StratifiedShuffleSplit(
                    2, test_size=rate, train_size=1-rate, random_state=args.seed)
                stsplit.get_n_splits(lslist, tmpdatay)
                indextr, indexte = next(stsplit.split(lslist, tmpdatay))
            else:
                indexall = np.arange(l)
                np.random.seed(args.seed)
                np.random.shuffle(indexall)
                ted = int(l*rate)
                indextr, indexte = indexall[:-ted], indexall[-ted:]

            trdatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_train(), indices=indextr, test_envs=args.test_envs))
            tedatalist.append(ImageDataset(args.dataset, args.task, args.data_dir,
                                           names[i], i, transform=imgutil.image_test(), indices=indexte, test_envs=args.test_envs))

    train_loaders = [InfiniteDataLoader(
        dataset=env,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)
        for env in trdatalist]

    eval_loaders = [DataLoader(
        dataset=env,
        batch_size=64,
        num_workers=args.N_WORKERS,
        drop_last=False,
        shuffle=False)
        for env in trdatalist+tedatalist]

    return train_loaders, eval_loaders


def get_img_dataloader_adv(args):
    """
    Builds three loaders for the held-out test domain:
        • train_loader – stratified split, image_train transform
        • val_loader   – complementary split, image_test transform
        • test_loader  – full domain,  image_test transform
    """
    # --- domain metadata ----------------------------------------------------
    dom_id   = args.test_envs[0]                        # single held-out env
    dom_name = args.img_dataset[args.dataset][dom_id]

    # --- full dataset (image_train transform only for label collection) -----
    full_ds = ImageDataset(                    
        args.dataset, args.task, args.data_dir,
        dom_name, dom_id,
        transform=imgutil.image_test(),
        test_envs=args.test_envs
    )

    labels = full_ds.labels
    n      = len(labels)
    idx    = np.arange(n)

    # --- stratified split ----------------------------------------------------
    
    rate = 0.2
    splitter  = ms.StratifiedShuffleSplit(
        n_splits=1,
        test_size=rate,
        train_size=1 - rate,
        random_state=args.seed
    )
    train_idx, val_idx = next(splitter.split(idx, labels))

    # --- dataset objects -----------------------------------------------------
    train_ds = ImageDataset(
        args.dataset, args.task, args.data_dir,
        dom_name, dom_id,
        transform=imgutil.image_train(),
        indices=train_idx,
        test_envs=args.test_envs
    )

    val_ds = ImageDataset(
        args.dataset, args.task, args.data_dir,
        dom_name, dom_id,
        transform=imgutil.image_test(),
        indices=val_idx,
        test_envs=args.test_envs
    )


    # --- loaders -------------------------------------------------------------
    train_loader = [InfiniteDataLoader(
        dataset=train_ds,
        weights=None,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS)]

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        drop_last=False
    )

    attack_loader = DataLoader(
        full_ds,
        batch_size=args.batch_size,
        num_workers=args.N_WORKERS,
        shuffle=False,
        drop_last=False
    )

    return train_loader, val_loader, attack_loader


class SVDLoader:
    def __init__(self, dataloader, k, device="cuda",
                    tau: float = 0.0,
                    thresh_mode: str = "abs"
                ):
        self.loader, self.k, self.device = dataloader, k, device
        self.tau = float(tau)
        self.thresh_mode = thresh_mode
    def __iter__(self):
        for xb, yb in self.loader:
            xb = xb.to(self.device, non_blocking=True)
            if self.tau > 0.0:
                xb = drop_small_singular_values_threshold(xb, self.tau, mode=self.thresh_mode)
            elif self.k:
                xb = drop_low_singular_values(xb, self.k)
            yield xb, yb                          


@torch.no_grad()
def drop_low_singular_values(x: torch.Tensor, k: int) -> torch.Tensor:
    if k == 0:
        return x                           

    B, C, H, W = x.shape      
    x_flat = x.reshape(B * C, H, W)
    q        = H - k
    U,S,Vh   = torch.svd_lowrank(x_flat, q=q, niter=2)
    x_recon = torch.matmul(U * S.unsqueeze(1), torch.transpose(Vh, 1, 2))

    return x_recon.reshape(B, C, H, W)


# ---- Value-thresholded exact SVD --------------------------------------------
@torch.no_grad()
def drop_small_singular_values_threshold(
    x: torch.Tensor,
    tau: float,
    mode: str = "abs",
) -> torch.Tensor:
    if tau <= 0:
        return x

    B, C, H, W = x.shape
    X = x.reshape(B * C, H, W)

    U, S, Vh = torch.linalg.svd(X, full_matrices=False)  # (BC,H,r), (BC,r), (BC,r,W) when full_matrices=False
    if mode == "abs":
        thr = tau
    elif mode == "rel":
        sigma_max = S.max(dim=1, keepdim=True).values
        thr = sigma_max * tau  # (BC,1) -> broadcast
    else:
        raise ValueError(f"Unknown threshold mode: {mode}. Use 'abs' or 'rel'.")

    mask = S >= thr     # (BC, r)

    S = S * mask
    X_hat = (U * S.unsqueeze(-2)) @ Vh  # (BC,H,r) * (BC,1,r) @ (BC,r,W)
    return X_hat.reshape(B, C, H, W)
