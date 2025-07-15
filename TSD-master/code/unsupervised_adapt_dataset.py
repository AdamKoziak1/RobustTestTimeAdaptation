# coding=utf-8
"""
Unsupervised hyper-parameter sweep on *source* domains.

For each seed ∈ {0,1,2} and each source domain (all domains ≠ test_envs[0])
    1. load the pretrained source model (seed-specific)
    2. build attacked train/val splits (stratified 80/20, **no extra transforms**)
    3. adapt on the train split with the chosen TTA algorithm
    4. evaluate on the val split
Finally, print and log mean ± std accuracy across domains and seeds.

Example
-------
python unsupervised_adapt_dataset.py \
    --dataset PACS --net resnet18 --adapt_alg TTA3 \
    --lambda1 10 --lambda2 69 --lambda3 1
"""
import os, time, statistics
import numpy as np
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit

from alg import alg
from alg.opt import *
from utils.util import set_random_seed, load_ckpt

from adv.attacked_imagefolder import AttackedImageFolder
import wandb
from unsupervise_adapt import get_args, make_adapt_model


def build_split_loaders(args, dom_id):
    """Return attacked (train_loader, val_loader) for a single domain."""
    dom_name = args.img_dataset[args.dataset][dom_id]
    dom_root = os.path.join(args.data_dir, dom_name)

    # Full attacked dataset (100 % rate, tensors already normalised in file)
    full_ds = AttackedImageFolder(
        root=dom_root,
        transform=None,                 # tensors are already normalised
        adv_root=args.attack_data_dir,
        dataset=args.dataset,
        domain=dom_name,
        config=f"{args.net}_{args.attack}",
        rate=args.attack_rate,
        seed=args.seed,
    )

    labels = np.array([t[1] for t in full_ds.samples])
    idx_all = np.arange(len(labels))

    splitter = StratifiedShuffleSplit(
        n_splits=1, test_size=0.2, train_size=0.8, random_state=args.seed
    )
    idx_tr, idx_val = next(splitter.split(idx_all, labels))

    train_ds = Subset(full_ds, idx_tr)
    val_ds   = Subset(full_ds, idx_val)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.N_WORKERS, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=64, shuffle=False,
        num_workers=args.N_WORKERS, pin_memory=True, drop_last=False
    )
    return train_loader, val_loader, len(val_ds)

def build_all_source_loaders(args):
    """
    Returns
        train_loaders : list[DataLoader]  – one per *source* domain
        val_loaders   : dict[dom_id → DataLoader]
    """
    train_loaders, val_loaders = [], {}
    for dom_id in range(len(args.domains)):
        if dom_id in args.test_envs:
            continue                        # skip held-out test domain
        tl, vl, _ = build_split_loaders(args, dom_id)
        train_loaders.append(tl)
        val_loaders[dom_id] = vl
    return train_loaders, val_loaders

def accuracy(model, loader):
    """Simple top-1 accuracy over a loader (no gradient updates)."""
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            logits = model.predict(x)   # adapt_model returns logits
            correct += (logits.argmax(1) == y).sum().item()
            total   += y.size(0)
    model.train()
    return 100.0 * correct / total

if __name__ == "__main__":
    args = get_args()

    run_name = (f"{args.adapt_alg}_{args.lambda3}_{args.cr_type}_start-{args.cr_start}_{args.steps}steps")    
    wandb.init(project="tta3_src", name=run_name, config=vars(args))

    global_accs = []          # store mean acc for each seed
    domain_names = [d for i, d in enumerate(args.domains)
                    if i not in args.test_envs]   # source domains only

    overall_start = time.time()
    for seed in (0, 1, 2):
        args.seed = seed
        set_random_seed(seed)

        seed_accs = []
        seed_start = time.time()
        for dom_id in [i for i in range(len(args.domains))
                       if i not in args.test_envs]:
            # 1. Load pretrained source model for *this* seed
            src_ckpt = os.path.join(
                args.data_file, "TSD-master", "code", "train_output",
                args.dataset, f"test_{args.test_envs[0]}", f"seed_{seed}",
                "model.pkl")
            alg_class = alg.get_algorithm_class("ERM")   # same as training
            base_model = alg_class(args).cuda()
            base_model.train()
            base_model = load_ckpt(base_model, src_ckpt)

            # 2. Build attacked loaders
            train_loader, val_loader, n_val = build_split_loaders(args, dom_id)

            # 3. Wrap with chosen test-time adaptation algorithm
            adapt_model = make_adapt_model(args, base_model)

            # 4. One pass over train_loader (unsupervised adaptation)
            for xb, _ in train_loader:
                adapt_model(xb.cuda())   # forward = adapt; labels unused

            # 5. Validation accuracy
            base_model.eval()
            acc_dom = accuracy(base_model, val_loader)
            seed_accs.append(acc_dom)

            wandb.log({f"acc_{domain_names[len(seed_accs)-1]}": acc_dom})
            print(f"[seed {seed}] {args.domains[dom_id]:12s}: {acc_dom:.2f} %")

        seed_mean = round(statistics.mean(seed_accs), 2)
        global_accs.append(seed_mean)
        print(f"[seed {seed}] mean over {len(seed_accs)} source domains: "
              f"{seed_mean:.2f} %  (took {time.time()-seed_start:.1f}s)")
        wandb.log({f"acc_seed_{seed}": seed_mean})

    overall_mean = round(statistics.mean(global_accs), 2)
    overall_std  = round(statistics.stdev(global_accs), 2)
    elapsed      = time.time() - overall_start

    print("\n==================  Summary  ==================")
    print(f"Seeds:      {global_accs}")
    print(f"Overall:    {overall_mean} ± {overall_std}  %")
    print(f"Total time: {elapsed/60:.1f} min")
    wandb.log({
        "acc_mean": overall_mean,
        "acc_std":  overall_std,
        "time_taken_s": elapsed,
        "adapt_alg": args.adapt_alg,
        "steps": args.steps,
        "lr": args.lr,
        "cr_type": args.cr_type,
        "cr_start": args.cr_start,
        "update_param": args.update_param,
    })
    wandb.finish()
