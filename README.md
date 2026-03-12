# Robust Test-Time Adaptation with FFT Defenses

This branch is prepared for the ECCV 2026 supplementary submission of the FFT-specific paper.
It focuses on reproducibility for FFT-based robust test-time adaptation (RTTA) experiments.

This is an anonymized artifact release:
- personal contact details were removed,
- machine-specific absolute paths were removed,
- W&B account identifiers were replaced with placeholders.

## Scope for This Supplement
Included:
- source-model training (`train.py`),
- adversarial stream generation (`generate_adv_data.py`),
- test-time adaptation/evaluation (`unsupervise_adapt.py`),
- sweep configs under `sweeps/` for FFT-paper runs,
- scripts that export tables/figures from run logs.

Out of scope:
- SAFER-specific claims and SAFER-only ablations from older drafts.
  Legacy SAFER files are kept for code compatibility/history, but are not required for the FFT supplementary package.

## Environment Setup
Option 1 (recommended):
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

Option 2 (conda lockfile-style environment):
```bash
conda env create -f tta-full.yml
conda activate tta-peft
```

## Data Setup
Download datasets:
- PACS: https://drive.google.com/uc?id=1JFr8f805nMUelQWWmfnJR3y4_SYoN5Pd
- OfficeHome: https://drive.google.com/uc?id=1uY0pj7oFsjMxRwaD3Sxy0jgel0fsYXLC
- VLCS: https://drive.google.com/uc?id=1skwblH1_okBwxWxmRsp9_qi15hyPpxg8

Expected layout:
```text
datasets/
  PACS/
    art_painting/ cartoon/ photo/ sketch/
  VLCS/
    Caltech101/ LabelMe/ SUN09/ VOC2007/
  office-home/
    Art/ Clipart/ Product/ RealWorld/
```

## Reproducing Core FFT Experiments
Use the same seeds/domains/rates as the paper (typically seeds `0,1,2`, domains `0,1,2,3`, attack rates `0,50,100`).

1. Train source model checkpoints:
```bash
python train.py \
  --dataset PACS \
  --data_file . \
  --data_dir datasets \
  --test_envs 0 \
  --seed 0 \
  --opt_type Adam \
  --lr 5e-5 \
  --max_epoch 50 \
  --output train_output
```

2. Generate adversarial streams (baseline PGD):
```bash
python generate_adv_data.py \
  --dataset PACS \
  --data_file . \
  --data_dir datasets \
  --test_envs 0 \
  --seed 0 \
  --attack linf \
  --eps 8 \
  --alpha_adv 2 \
  --steps 20
```

3. Generate FFT-aware adversarial streams (when needed by the paper setup):
```bash
python generate_adv_data.py \
  --dataset PACS \
  --data_file . \
  --data_dir datasets \
  --test_envs 0 \
  --seed 0 \
  --attack linf \
  --eps 8 \
  --alpha_adv 2 \
  --steps 20 \
  --fft_rho 0.6 \
  --fft_alpha 1.0
```

4. Evaluate TTA baseline:
```bash
python unsupervise_adapt.py \
  --dataset PACS \
  --data_file . \
  --data_dir datasets \
  --attack_data_dir datasets_adv \
  --test_envs 0 \
  --seed 0 \
  --adapt_alg Tent \
  --attack linf_eps-8.0_steps-20 \
  --attack_rate 100 \
  --fft_input_keep_ratio 1.0 \
  --fft_feat_keep_ratio 1.0
```

5. Evaluate FFT defense:
```bash
python unsupervise_adapt.py \
  --dataset PACS \
  --data_file . \
  --data_dir datasets \
  --attack_data_dir datasets_adv \
  --test_envs 0 \
  --seed 0 \
  --adapt_alg Tent \
  --attack linf_eps-8.0_steps-20 \
  --attack_rate 100 \
  --fft_input_keep_ratio 0.6 \
  --fft_input_alpha 0.7 \
  --fft_feat_keep_ratio 0.8 \
  --fft_feat_alpha 1.0
```

## Sweep-Based Reproduction
FFT-paper sweep configs are provided (examples):
- `sweeps/FFT_PAPER_tta_linf8_fft_paper_all_tta_baseline.yaml`
- `sweeps/FFT_PAPER_tta_linf8_fft_paper_defense_all_tta.yaml`
- `sweeps/ablation_tent_baseline_pacs.yaml`
- `sweeps/ablation_tent_fft_defense_pacs.yaml`

Run sweeps:
```bash
wandb sweep sweeps/FFT_PAPER_tta_linf8_fft_paper_all_tta_baseline.yaml
wandb sweep sweeps/FFT_PAPER_tta_linf8_fft_paper_defense_all_tta.yaml
wandb agent <entity>/<project>/<sweep_id>
```

Use your own W&B entity/project (do not rely on defaults).

## Table/Figure Export
Use the scripts under `scripts/` to export LaTeX tables and plots from run logs.
Start from:
- `scripts/wandb_table.py`
- `scripts/wandb_ablation_tables.py`
- `scripts/plot_wandb_attack_rate_curve.py`
- `scripts/wandb_compute_overhead_table.py`

An ECCV-focused reproducibility checklist and command map is provided in:
- `docs/ECCV2026_SUPPLEMENTARY_REPRODUCIBILITY.md`
