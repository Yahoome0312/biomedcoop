# From-scratch MT-TCP + Soft Confusion Prior implementation notes

## Protocol boundary

- The production baseline starts from original BiomedCLIP weights. No CoOp,
  VPT, TextDeep, or MT-TCP checkpoint is used for initialization.
- CoOp, Visual Deep Prompt, MT-TCP (including internal TextDeep), and active
  confusion modules share one AdamW optimizer and one backward pass.
- The BiomedCLIP towers and `logit_scale` stay frozen.
- KG, prompt-anchor, cross-modal prototype, epoch-zero warm-start, and
  baseline-checkpoint copying are absent from the active training path.

## Soft Bank

Each `(shot, seed)` Bank contains only that run's support samples. It averages
the full frozen-BiomedCLIP probability vector by true class, clears the
diagonal, and does not renormalize the remaining row. Hard error counts are
diagnostic only. Dataset, shot, seed, class order, support identity, and prior
tensor fingerprints are verified before training.

There is no exemplar retrieval, SAM, Diff-Manner adapter, MGDE, MoE, router,
or historical-image forward.

## Dimensions and projections

- Projected image/text features: 512.
- Final normalized image patch tokens: `196 x 768`.
- Semantic input `[t_i-t_j; t_i*t_j]`: 1024 to 512.
- Local query: 512 to 768; local output: 768 to 512.
- Global gating and final fusion remain in 512-D, so MT-TCP needs no extra
  compatibility projection.

## Selection and checkpoints

The complete official validation split is used. Accuracy and balanced
accuracy independently save two checkpoints from the same trajectory. Test is
locked until the complete requested validation grid exists, then each
checkpoint is loaded and tested separately.

## Initialization

- CoOp: frozen BERT embeddings for `a photo of a`.
- Visual Deep Prompt: existing uniform initialization (`±0.0625`).
- TextDeep: `Normal(0, 0.02)` inside MT-TCP.
- MT-TCP down projection: PyTorch default; up projection: zeros;
  layer-residual sigmoid value: 0.1; late gates: 0.05.
- New MLP Linear layers: PyTorch defaults; LayerNorm: weight 1/bias 0.
- Local query: Xavier uniform; patch projection: `Normal(0, 1e-3)`, bias 0.

The B0 initialization fingerprint is computed before confusion modules are
created, making same-seed cross-variant initialization directly auditable.

## Confusion diagnostics

`scripts/coopvpt/aggregate_confusion_results.py` aggregates the saved
support-only soft prior and hard-count diagnostic, selected-pair frequencies
and coverage, selected prior/score, training `L_confuse`, and raw/normalized
test confusion matrices. The generated `experiment_report.md` is the compact
human-readable report; `experiment_report.json` retains all per-seed and
per-class details.
