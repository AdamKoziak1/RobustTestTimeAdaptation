# RobustTestTimeAdaptation

This repository builds on [TSD](https://github.com/SakurajimaMaiii/TSD) and contains code/results for robust test-time adaptation with SAFER.

## ECML Reproducibility Checklist
1. Create environment and install dependencies.
```bash
conda create -n tta-ecml python=3.13.5 -y
conda activate tta-ecml
pip install -r requirements.txt
pip install pyyaml
wandb login
```

2. Set repository root for all commands.
```bash
export RTTA_ROOT="$(pwd)"
```

3. Download datasets and place them as:
```text
datasets/
  office-home/
    Art/ Clipart/ Product/ RealWorld/
  PACS/
    art_painting/ cartoon/ photo/ sketch/
  VLCS/
    Caltech101/ LabelMe/ SUN09/ VOC2007/
```
Links:
- PACS: https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd
- OfficeHome: https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC
- VLCS: https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8

4. Reproduce paper artifacts from logged W&B sweeps (recommended for review).  
These commands regenerate the `.tex/.csv/.png` files already tracked in `sweeps/`.

Main paper tables (PACS, VLCS, OfficeHome + domain average):
```bash
python scripts/wandb_paper_tables.py \
  --cache-file sweeps/wandb_paper_table_cache.yaml \
  --output sweeps/paper_tables_main.tex
```

Ablation tables:
```bash
python scripts/wandb_ablation_tables.py \
  --cache-file sweeps/wandb_ablation_table_cache.yaml \
  --output sweeps/ablation_tables.tex
```

Alpha-mode and alpha-signal ablation tables (PACS domain 0):
```bash
python scripts/wandb_alpha_mode_table.py \
  --mode-sweep-ids kqab7u5b \
  --baseline-sweep-ids 3da10eth \
  --entity bigslav --project safer \
  --dataset PACS --domain-ids 0 --attack-rates 0,50,100 \
  --output sweeps/alpha_mode_table_dom0.tex

python scripts/wandb_alpha_signal_table.py \
  --signal-sweep-ids zgj6v28w \
  --baseline-sweep-ids 3da10eth \
  --entity bigslav --project safer \
  --dataset PACS --domain-ids 0 --attack-rates 0,50,100 \
  --output sweeps/alpha_signal_table_dom0.tex
```

Sensitivity plots used in `paper.tex`:
```bash
python scripts/plot_wandb_param_curve.py \
  --sweep-ids mxif4s7w \
  --entity bigslav --project safer \
  --dataset PACS --domain-id 0 \
  --x-key s_num_views --attack-rates 0,50,100 \
  --filter adapt_alg=Tent --filter s_wrap_alg=1 \
  --filter s_alpha_mode=none --filter s_primary_view_pool=cc_drop \
  --output sweeps/views_sensitivity_dom0.png \
  --csv-output sweeps/views_sensitivity_dom0.csv

python scripts/plot_wandb_param_curve.py \
  --sweep-ids t55sacwf \
  --entity bigslav --project safer \
  --dataset PACS --domain-id 0 \
  --x-key s_aug_max_ops --attack-rates 0,50,100 \
  --filter adapt_alg=Tent --filter s_wrap_alg=1 \
  --filter s_alpha_mode=none --filter s_primary_view_pool=cc_drop \
  --output sweeps/maxops_sensitivity_dom0.png \
  --csv-output sweeps/maxops_sensitivity_dom0.csv

python scripts/plot_wandb_param_curve.py \
  --sweep-ids 161jsdp8 \
  --entity bigslav --project safer \
  --dataset PACS --domain-id 0 \
  --x-key s_alpha_conf_threshold --attack-rates 0,100 \
  --filter adapt_alg=Tent --filter s_wrap_alg=1 \
  --filter s_alpha_mode=sigmoid --filter s_alpha_signal=feat_disagreement \
  --filter s_primary_view_pool=cc_drop --filter s_alpha_attack_high_conf=1 \
  --output sweeps/alpha_tau_sensitivity_dom0_0_100.png \
  --csv-output sweeps/alpha_tau_sensitivity_dom0_0_100.csv

python scripts/plot_wandb_param_curve.py \
  --sweep-ids 161jsdp8 \
  --entity bigslav --project safer \
  --dataset PACS --domain-id 0 \
  --x-key s_alpha_sigmoid_slope --attack-rates 0,100 \
  --filter adapt_alg=Tent --filter s_wrap_alg=1 \
  --filter s_alpha_mode=sigmoid --filter s_alpha_signal=feat_disagreement \
  --filter s_primary_view_pool=cc_drop --filter s_alpha_attack_high_conf=1 \
  --output sweeps/alpha_kappa_sensitivity_dom0_0_100.png \
  --csv-output sweeps/alpha_kappa_sensitivity_dom0_0_100.csv

python scripts/plot_wandb_param_curve.py \
  --sweep-ids 161jsdp8 \
  --entity bigslav --project safer \
  --dataset PACS --domain-id 0 \
  --x-key s_alpha_attack_value --attack-rates 0,100 \
  --filter adapt_alg=Tent --filter s_wrap_alg=1 \
  --filter s_alpha_mode=sigmoid --filter s_alpha_signal=feat_disagreement \
  --filter s_primary_view_pool=cc_drop --filter s_alpha_attack_high_conf=1 \
  --output sweeps/alpha_attackval_sensitivity_dom0_0_100.png \
  --csv-output sweeps/alpha_attackval_sensitivity_dom0_0_100.csv

python scripts/plot_wandb_param_curve.py \
  --sweep-ids 161jsdp8 \
  --entity bigslav --project safer \
  --dataset PACS --domain-id 0 \
  --x-key s_alpha_clean_value --attack-rates 0,100 \
  --filter adapt_alg=Tent --filter s_wrap_alg=1 \
  --filter s_alpha_mode=sigmoid --filter s_alpha_signal=feat_disagreement \
  --filter s_primary_view_pool=cc_drop --filter s_alpha_attack_high_conf=1 \
  --output sweeps/alpha_cleanval_sensitivity_dom0_0_100.png \
  --csv-output sweeps/alpha_cleanval_sensitivity_dom0_0_100.csv
```

Batch-stability plot used in `paper.tex`:
```bash
python scripts/plot_wandb_batch_acc_history.py \
  --run-ids 7x4fwbep,wf4s03uf,ta93e6fh,2pd4iyfq,6c31hz8j,nfri3cya \
  --labels "Tent (0%),Tent (100%),Tent+SAFER (0%),Tent+SAFER (100%),Tent+SAFER-A (0%),Tent+SAFER-A (100%)" \
  --metrics batch_acc --window 16 \
  --entity bigslav --project safer \
  --output sweeps/batch_stability_dom0_0_100_batch_acc.png \
  --csv-output sweeps/batch_stability_dom0_0_100_batch_acc.csv
```

5. Compile the paper:
```bash
latexmk -pdf paper.tex
```

## Full Rerun From Scratch (Compute-Heavy)
1. Train source models (all datasets, domains, seeds).
```bash
for d in PACS VLCS office-home; do
  for env in 0 1 2 3; do
    for s in 0 1 2; do
      python train.py --dataset "$d" --data_file "$RTTA_ROOT" --data_dir datasets \
        --test_envs "$env" --seed "$s" --opt_type Adam --lr 5e-5 --max_epoch 50
    done
  done
done
```

2. Generate adversarial test tensors (example: Linf 8/255 PGD-20).
```bash
for d in PACS VLCS office-home; do
  for env in 0 1 2 3; do
    for s in 0 1 2; do
      python generate_adv_data.py --dataset "$d" --data_file "$RTTA_ROOT" --data_dir datasets \
        --test_envs "$env" --seed "$s" --attack linf --eps 8 --alpha_adv 2 --steps 20
    done
  done
done
```

3. Launch adaptation sweeps with W&B (`wandb sweep <yaml>`, then `wandb agent <entity/project/sweep_id>`).  
For non-author paths, include these in each sweep YAML for `unsupervise_adapt.py`:
```yaml
data_file:
  value: /absolute/path/to/RobustTestTimeAdaptation
attack_data_dir:
  value: /absolute/path/to/RobustTestTimeAdaptation/datasets_adv
```

4. Re-run the artifact-generation commands from the checklist above.

## Single-Run Examples
Source training:
```bash
python train.py --dataset PACS --data_file "$RTTA_ROOT" --data_dir datasets \
  --test_envs 0 --seed 0 --opt_type Adam --lr 5e-5 --max_epoch 50
```

Attack generation:
```bash
python generate_adv_data.py --dataset PACS --data_file "$RTTA_ROOT" --data_dir datasets \
  --test_envs 0 --seed 0 --attack linf --eps 8 --alpha_adv 2 --steps 20
```

TTA evaluation:
```bash
python unsupervise_adapt.py --dataset PACS --data_file "$RTTA_ROOT" --data_dir datasets \
  --attack_data_dir "$RTTA_ROOT/datasets_adv" \
  --test_envs 0 --adapt_alg Tent --attack linf_eps-8.0_steps-20 \
  --attack_rate 100 --seeds 0,1,2 --batch_size 64 --steps 1
```

## Contact
adamkoziak@cmail.carleton.ca
