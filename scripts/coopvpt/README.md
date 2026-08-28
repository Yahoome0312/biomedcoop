# DermaMNIST CoOp + VPT-Deep experiments

## Current from-scratch MT-TCP + soft confusion protocol

The current retained MT-TCP baseline jointly trains CoOp, Visual Deep Prompt,
MT-TCP and its internal TextDeep prompts from epoch 1. Historical TextDeep
warm-start checkpoints and KG/anchor/XProto losses are not used.

Build support-only Soft Banks:

```powershell
python scripts/coopvpt/build_confusion_prior.py `
  --data-root D:\Data\dermamnist `
  --output-root output\soft_confusion_banks `
  --shots 4 8 16 32 --seeds 1 2 3
```

Train the from-scratch MT-TCP B0 validation grid, then independently test the
best-validation-ACC and best-validation-BACC checkpoints:

```powershell
python scripts/coopvpt/run_tcp_fullval.py `
  --data-root D:\Data\dermamnist `
  --output-root output\confusion_aware_from_scratch `
  --variant b0 --shots 4 8 16 32 --seeds 1 2 3 `
  --epochs 100 --validation-only

python scripts/coopvpt/run_tcp_fullval.py `
  --data-root D:\Data\dermamnist `
  --output-root output\confusion_aware_from_scratch `
  --variant b0 --shots 4 8 16 32 --seeds 1 2 3 `
  --epochs 100 --evaluate-existing
```

Run all ablations with the same support and seeds:

```powershell
$variants = 'pair','semantic','semantic_global','semantic_local','global_local','full'
foreach ($variant in $variants) {
  python scripts/coopvpt/run_tcp_fullval.py `
    --data-root D:\Data\dermamnist `
    --output-root output\confusion_aware_from_scratch `
    --bank-root output\soft_confusion_banks `
    --variant $variant --shots 4 8 16 32 --seeds 1 2 3 `
    --epochs 100 --validation-only

  python scripts/coopvpt/run_tcp_fullval.py `
    --data-root D:\Data\dermamnist `
    --output-root output\confusion_aware_from_scratch `
    --bank-root output\soft_confusion_banks `
    --variant $variant --shots 4 8 16 32 --seeds 1 2 3 `
    --epochs 100 --evaluate-existing
}
```

Aggregate all seven completed variants, including both checkpoint-selection
rules, per-seed/per-class metrics, parameter counts and the full gate audit:

```powershell
python scripts/coopvpt/aggregate_confusion_results.py `
  --root output\confusion_aware_from_scratch `
  --bank-root output\soft_confusion_banks
```

Every run saves both selection checkpoints, raw/normalized confusion matrices,
per-class recall, compressed pair/gate analysis, loss terms, parameter counts,
initialization fingerprints, and per-seed/mean±std summaries.
The aggregate report additionally exposes the soft-prior matrix and hard-count
diagnostic, selected-pair frequencies/coverage, selected prior and score,
training `L_confuse` (overall and epoch-100), and aggregate raw/normalized
confusion matrices in `experiment_report.md` and `experiment_report.json`.

The trainer is `CoOpVPT_BiomedCLIP`. CoOp, Visual VPT and optional TextDeep
prompts share one AdamW optimizer, one scheduler and one learning rate. Few-shot
sampling applies only to training; validation uses the complete official split
with its natural class distribution.

To export the complete B₀/full tables (including B₀ confusion matrices, all
per-class recalls, per-seed values, full pair/loss diagnostics and matrix file
links) from an already completed output tree:

```powershell
python scripts/coopvpt/export_b0_full_metrics.py `
  --report output\confusion_aware_from_scratch\experiment_report.json `
  --output-dir output\confusion_aware_from_scratch
```

This writes `b0_full_metrics.md` and `b0_full_metrics.json`; it does not rerun
training or evaluation.

Every training run performs one validation pass per epoch and retains the best
validation-ACC and best validation-BACC prompt checkpoints from that same
trajectory. Their JSON records contain ACC, BACC, AUC, macro-F1 and per-class
recall. No checkpoint or logit ensemble is used.

## Baseline workflow

```powershell
# Search and reproduce Visual VPT-Deep baselines.
.\scripts\coopvpt\search_dermamnist.ps1
.\scripts\coopvpt\reproduce_dermamnist.ps1

# Aggregate baseline results.
D:\Anaconda\python.exe scripts\coopvpt\aggregate_results.py

# Add four TextDeep tokens per BERT layer.
python scripts\coopvpt\reproduce_text_deep_dermamnist.py
```

## Legacy MT-TCP results

Earlier LayerBasis+XProto runs used a TextDeep checkpoint warm-start and extra
KG/anchor/prototype losses. Those outputs remain historical artifacts only and
are incompatible with the current runner's `from_scratch_joint_ce` checkpoint
metadata. The current trainer rejects them rather than silently mixing the two
protocols.
