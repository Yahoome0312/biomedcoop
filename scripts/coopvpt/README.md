# DermaMNIST CoOp + VPT-Deep experiments

The trainer is `CoOpVPT_BiomedCLIP`. CoOp, Visual VPT and optional TextDeep
prompts share one AdamW optimizer, one scheduler and one learning rate. Few-shot
sampling applies only to training; validation uses the complete official split
with its natural class distribution.

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

## Final retained MT-TCP method

Only the reported configuration remains in production code:

`CoOp_VPT_Deep_TextDeep_MT-TCP_LayerBasis_XProto_Nv4_Nt4_K50`

- CoOp tokens `Nc=4`, Visual VPT-Deep tokens `Nv=4`, internal TextDeep tokens
  `Nt=4`;
- 50 fixed BiomedCoOp descriptions per class;
- descriptions encoded in the frozen PubMedBERT representation before block 8;
- five ordered groups of ten descriptions produce four layer-aligned bases;
- a zero-initialized `768 -> 128 -> 4x768` QuickGELU residual TKE;
- centered, norm-matched residual injection before BERT blocks 8--11;
- gate initialization `0.05` and ten-epoch residual warm-up;
- TextDeep baseline initialization selected by validation BACC;
- `CE + 4*centered-KG + 4*prompt-anchor + 0.5*cross-modal-prototype`;
- AdamW, LR `0.005`, weight decay `0.0005`, 100 epochs;
- one fused model and the same image/text similarity logits for train, val and
  test.

The runner defaults to the reported 4/8/16/32-shot, seeds 1/2/3 grid:

```powershell
D:\Anaconda\python.exe scripts\coopvpt\run_tcp_fullval.py `
  --init-baseline-root output\dermamnist_fullval_acc_bacc `
  --comparison-baseline-root output\dermamnist_fullval_acc_bacc
```

For strict test isolation, first add `--validation-only`; after the frozen
validation gate passes, rerun with `--evaluate-existing`. The second command
loads the already-trained ACC/BACC checkpoints and never retrains the model.

On the 12 paired test runs, the supported best-validation-ACC selection changed
BACC by `+2.76`, AUC by `+1.56`, macro-F1 by `+2.60`, and ACC by `-0.27`
percentage points relative to the same-shot, same-seed TextDeep baseline. The
separately retained best-validation-BACC selection did not establish an
improvement and is kept only for audit.

Historical failed TCP variants remain recoverable from Git commit `b14f6e9`;
their search/refinement entry points are intentionally absent from the current
code.
