# ECCV 2026 Supplementary Reproducibility (FFT Paper)

This document defines the reproducibility protocol for the FFT-specific paper submission.

## 1. Reproducibility Targets
Primary targets:
- main robustness tables across PACS, VLCS, OfficeHome,
- FFT-vs-baseline deltas under attack rates `{0, 50, 100}`,
- key ablations (Tent baseline, JPEG, Gaussian blur, FFT defense),
- attack-rate robustness curves and compute-overhead summaries.

## 2. Execution Envelope
Suggested envelope used for runs:
- GPU: 1x CUDA GPU (>= 12 GB VRAM recommended),
- CPU workers: 4 (`--N_WORKERS 4` default),
- seeds: `0,1,2`,
- batch size: `64` for adaptation/evaluation.

For strict repeatability, keep:
- dataset order unchanged,
- identical seed set,
- same attack config IDs (e.g., `linf_eps-8.0_steps-20`),
- same FFT hyperparameters as in sweep YAML files.

## 3. End-to-End Command Plan
1. Train source models for each dataset/domain/seed:
```bash
python train.py --dataset <DATASET> --test_envs <DOMAIN_ID> --seed <SEED> --data_file . --data_dir datasets --output train_output
```

2. Generate adversarial streams for each dataset/domain/seed:
```bash
python generate_adv_data.py --dataset <DATASET> --test_envs <DOMAIN_ID> --seed <SEED> --data_file . --data_dir datasets --attack linf --eps 8 --alpha_adv 2 --steps 20
```

3. Run adaptation baseline and FFT defense:
```bash
python unsupervise_adapt.py --dataset <DATASET> --test_envs <DOMAIN_ID> --seeds 0,1,2 --data_file . --data_dir datasets --attack_data_dir datasets_adv --adapt_alg <ALG> --attack linf_eps-8.0_steps-20 --attack_rate <RATE>
```

4. Run FFT-paper sweep grids:
```bash
wandb sweep sweeps/FFT_PAPER_tta_linf8_fft_paper_all_tta_baseline.yaml
wandb sweep sweeps/FFT_PAPER_tta_linf8_fft_paper_defense_all_tta.yaml
wandb sweep sweeps/ablation_tent_baseline_pacs.yaml
wandb sweep sweeps/ablation_tent_fft_defense_pacs.yaml
wandb sweep sweeps/ablation_tent_blur_defense_pacs.yaml
wandb sweep sweeps/ablation_tent_jpeg_defense_pacs.yaml
```

5. Launch agents:
```bash
wandb agent <entity>/<project>/<sweep_id>
```

## 4. Exporting Supplement Tables/Figures
Ablation tables:
```bash
python scripts/wandb_ablation_tables.py \
  --cache-file sweeps/wandb_ablation_table_cache.yaml \
  --entity <your_wandb_entity> \
  --project <your_wandb_project> \
  --output sweeps/ablation_tables_fft.tex
```

Attack-rate curve:
```bash
python scripts/plot_wandb_attack_rate_curve.py \
  --sweep-ids <comma_separated_sweep_ids_or_paths> \
  --entity <your_wandb_entity> \
  --project <your_wandb_project> \
  --dataset PACS \
  --domain-id 0 \
  --output sweeps/attack_rate_curve_fft.png \
  --csv-output sweeps/attack_rate_curve_fft_points.csv
```

Compute overhead table:
```bash
python scripts/wandb_compute_overhead_table.py \
  --sweep-ids <comma_separated_sweep_ids_or_paths> \
  --entity <your_wandb_entity> \
  --project <your_wandb_project> \
  --dataset PACS \
  --domain-id 0 \
  --output sweeps/compute_overhead_fft.tex
```

## 5. Output Manifest (Expected)
During a full reproduction run, expect these folders/files:
- `train_output/<dataset>/test_<domain>/seed_<seed>/model*.pkl`
- `datasets_adv/seed_<seed>/<dataset>/clean/` and attack-config subfolders
- `tta_output/...` evaluation summaries
- `wandb/` local run logs (if W&B online/offline logging enabled)
- generated `*.tex`, `*.png`, and `*.csv` files under `sweeps/`

## 6. Anonymization Notes
To support double-blind review:
- no personal email is included in user-facing docs,
- no hard-coded local absolute path is required,
- W&B defaults use placeholders (`your_wandb_entity`, `your_wandb_project`).

## 7. Legacy SAFER Artifacts
SAFER-named scripts/configs remain in the repository for backward compatibility.
For the FFT ECCV supplementary package, treat them as legacy unless a section in the paper explicitly cites them.
