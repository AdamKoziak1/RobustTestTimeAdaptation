# coding=utf-8
import numpy as np
import sklearn.model_selection as ms
from torch.utils.data import DataLoader

import datautil.imgdata.util as imgutil
from datautil.imgdata.imgdataload import ImageDataset
from datautil.mydataloader import InfiniteDataLoader

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


