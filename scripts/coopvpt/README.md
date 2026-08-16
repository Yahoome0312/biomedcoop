# DermaMNIST CoOp + VPT-Deep reproduction

The trainer is `CoOpVPT_BiomedCLIP` and uses cross-entropy only. With VPT
enabled, the CoOp context and VPT-Deep prompts are optimized together by one
AdamW optimizer, one scheduler, and one shared learning rate.

The dataset CLI root is `D:\Data\dermamnist`; the existing junction below it
points to `DermaMNIST-224`.

Run the stages in order:

```powershell
# 1. Search one shared AdamW LR for each VPT-Deep prompt budget.
.\scripts\coopvpt\search_dermamnist.ps1

# 2. Run VPT-Deep for budgets 1/2/5/10/20,
#    shots 1/2/4/8/16/32 and seeds 1/2/3.
.\scripts\coopvpt\reproduce_dermamnist.ps1

# Or select the checkpoint with the highest validation accuracy before test.
.\scripts\coopvpt\reproduce_dermamnist.ps1 `
    -FinalModel best_val `
    -OutputRoot output\dermamnist_coop_native_vptdeep_adamw_bestval

# 3. Produce JSON, CSV, and Markdown summaries.
D:\Anaconda\python.exe scripts\coopvpt\aggregate_results.py

# 4. Fixed Nv=5 comparison with four text Deep Prompt tokens per BERT layer.
python scripts/coopvpt/reproduce_text_deep_dermamnist.py

# Aggregate the text-Deep comparison.
D:\Anaconda\python.exe scripts\coopvpt\aggregate_results.py `
    --output-root output\dermamnist_coop_native_vptdeep_textdeep_adamw_lr005 `
    --methods CoOp_VPT_Deep_TextDeep_Nv5_Nt4
```

Few-shot sampling is applied only to the training split. Validation always uses
the complete official validation split with its natural class distribution.
Every epoch performs one validation pass; that same result independently
updates `model-best-accuracy.pth.tar` and
`model-best-balanced_accuracy.pth.tar`. The corresponding
`best_validation_accuracy.json` and
`best_validation_balanced_accuracy.json` files record all metrics from the
selected epoch, including accuracy, balanced accuracy, AUC, and macro F1.
`best_validation_all.json` combines both records, while the legacy
`best_validation.json` remains an alias of the metric configured by
`TEST.BEST_METRIC`.

Search runs rank configurations by validation accuracy and skip the test split.
Final runs default to the last epoch and evaluate the test split once. Passing
`-FinalModel best_val` instead evaluates the checkpoint with the highest
validation accuracy. Both best checkpoints are retained in either mode, and
the model is trained only once. Shallow is not part of this experiment
workflow.

Every output directory contains `run_manifest.json`, `console.log`, Dassl's
`log.txt`, the two best-validation records, and prompt-only checkpoints.
Re-running a command skips complete runs and continues the remaining grid.

## CoOp + VPT-Deep + TCP

`CoOp_VPT_Deep_TCP_Nv4_Ntke4` reimplements the original
[Textual-based Class-aware Prompt Tuning paper](https://arxiv.org/abs/2209.10190)
and its [official MIT-licensed repository](https://github.com/htyao89/Textual-based_Class-aware_prompt_tuning)
for BiomedCLIP's PubMedBERT text tower. It does not copy the repository's
custom CLIP implementation. The first protocol is fixed to 16-shot, seeds
1/2/3, 100 epochs, four CoOp tokens, four visual VPT tokens, and four TKE
tokens:

```powershell
D:\Anaconda\python.exe scripts\coopvpt\run_tcp_fullval.py --shots 16 --seeds 1 2 3
```

TCP uses the single frozen prior template `a photo of a {class}.`, the original
`prior_dim -> 128 -> 4x768` QuickGELU TKE, exact replacement before zero-based
BERT layer 8, and `CE + 8 * knowledge-consistency loss`. CoOp, visual VPT, and
TKE remain jointly trainable; all BiomedCLIP backbone and projection weights
remain frozen. The fixed prior is excluded from checkpoints, while its
fingerprint and compatibility metadata are validated when a TCP checkpoint is
loaded.

### Full original-TCP baseline

The TCP runner now defaults to the complete paired baseline grid: 4/8/16/32
shots and seeds 1–5. Existing complete runs are skipped safely:

```powershell
D:\Anaconda\python.exe scripts\coopvpt\run_tcp_fullval.py
```

The complete original-TCP table is written to
`output/dermamnist_fullval_acc_bacc/tcp_fullgrid_summary.json` (and `.md`).

The active implementation deliberately contains no model/logit ensemble and no
post-hoc prior-calibration entry point. The protocol forbids combining two
independently trained models and forbids selecting shot-specific fusion or
calibration parameters.

### Single-model 50-description TCP

The current refinement uses all 50 fixed BiomedCoOp descriptions available for
each of DermaMNIST's seven classes (350 descriptions total). Vanilla frozen
BiomedCLIP encodes them once into a non-trainable `[7, 50, prior_dim]` bank.
One trainable model then maps each class set to four TKE tokens. The implemented
validation candidates are:

- `feature_mean`: average the 50 frozen description features first, then TKE;
- `tke_mean`: run the shared TKE on every description, then average its tokens;
- `consensus_weighted`: weight descriptions by agreement with their class
  centroid before TKE;
- `set_attention`: use four learned queries to pool the unordered 50-description
  set directly into four class-aware tokens;
- `cosine_set_attention`: normalize the query/key directions to prevent the
  learned selectors from collapsing to the uniform 50-description mean;
- `grouped10_cosine_attention`: average the five ordered blocks of ten first,
  then use four learned selectors over those five description groups;
- `grouped10_layer_residual`: encode all descriptions to the exact pre-block-8
  PubMedBERT space, form four frozen token bases from the five ten-description
  means, and learn a zero-initialized TKE residual on top of those bases;
- `grouped10_layer_projected_hybrid`: retain those four stable layer bases and
  add a gated, unit-normalized correction from the original projected-text TKE
  inside the same text branch;
- `grouped10_layer_projected_residual`: use the same two frozen description
  spaces but zero-initialize the projected TKE correction, making the initial
  injection exactly equal to the layer basis before the correction is learned.

The first differential in-place connection centers the seven class token sets,
normalizes each residual, and adds it once to the original four CoOp context
positions immediately before zero-based BERT block 8. The deep in-place
refinement applies the same operation before blocks 8, 9, 10, and 11, with one
trainable scalar gate per block. Both variants reuse the existing CoOp slots:
they add no sequence positions and no separate persistent Deep Text Prompt
slots, so sequence length, attention mask, and position IDs remain unchanged.
CoOp, visual VPT, and the description aggregator/TKE are trained together by
one optimizer. There is one image branch, one text branch, and one logit tensor;
no external logits, model ensemble, checkpoint ensemble, or post-hoc
shot-dependent parameter exists.

The deep refinement can evaluate the exact initialized CoOp+VPT checkpoint as
epoch 0. This is a real loadable dual-best checkpoint, not a copied metric: a
zero TCP residual reproduces the baseline text path exactly. A fixed cosine plus
relative-L2 prompt anchor limits drift of both CoOp and VPT while TKE learns the
class-specific directions. These settings are global and identical for every
shot and seed.

The optional ramped deep connection keeps the same four injections but
initializes their global, shot-independent gates to `[0.05, 0.10, 0.15, 0.20]`.
This compensates for the fact that an early injected residual passes through
more frozen BERT blocks than a late residual. The gates remain trainable model
parameters; the initialization profile is identical for every run.
The balanced-ramp variant uses `[0.075, 0.125, 0.175, 0.225]`: it has the same
late-block emphasis but preserves the uniform four-gate total and mean strength
(`0.60` and `0.15`, respectively).
The terminal-boost profile keeps the successful ramp's first three gates
unchanged and increases only the final pre-block-11 gate, producing
`[0.05, 0.10, 0.15, 0.25]`. This isolates added semantic strength at the
shortest residual path instead of increasing every layer.
The separately fingerprinted terminal-peak refinement changes only that final
gate to `0.30`, producing `[0.05, 0.10, 0.15, 0.30]`. It is evaluated as a new
global configuration for the complete shot/seed grid; existing terminal-boost
checkpoints cannot be loaded as terminal-peak checkpoints.

For a 100-epoch run, schedule-normalized residual warmup uses 67 epochs so the
warmup occupies the same two-thirds fraction as the 20-of-30 screening run.
The value is one global setting shared by every shot and seed; LR, optimizer,
loss weights, model, description bank, and gate profile remain unchanged.

The validation-only search compares four permutation-invariant description
aggregators (`feature_mean`, `tke_mean`, `consensus_weighted`, and
`set_attention`) and then compares the supported internal TKE/TextPrompt
connections for the best aggregator. Every candidate uses exactly the same
shots, seeds, optimizer, loss weights, and 50-epoch budget. Test evaluation is
disabled until one method is frozen from validation:

```powershell
D:\Anaconda\python.exe scripts\coopvpt\search_multitext_tcp.py
```

The selected method is subsequently trained once per run for 100 epochs on
the full 4/8/16/32-shot, seed 1/2/3 grid. Every shot and seed uses the same
optimizer, LR, loss weights, gate initialization, warmup, anchor, description
count, and architecture. Both best-validation-ACC and
best-validation-BACC checkpoints are retained from that single training run;
the final report evaluates the two checkpoint selections independently and
never combines their logits. The pre-test validation gate remains tied to the
frozen best-validation-BACC rule, so adding the second reporting verdict cannot
retroactively authorize test evaluation.

The full grid is first run with `--validation-only`. If and only if the global
validation rule passes, `--evaluate-existing` builds the same architecture and
loads the already trained ACC/BACC prompt checkpoints for test evaluation. It
never calls the training loop. The runner also verifies that the matching
full-grid validation summary exists, was produced without test evaluation, and
has a passing frozen global gate; a failed or mismatched summary makes test
evaluation abort. This keeps test isolated from method selection and avoids a
second training run.

If the first search does not improve validation BACC, the refinement runner
tests global knowledge-consistency weights for the single SetAttention model:

```powershell
D:\Anaconda\python.exe scripts\coopvpt\refine_multitext_tcp.py
```

Each weight is shared by every shot and seed. The runner freezes a method only
when mean validation BACC improves and the predefined ACC/AUC guards pass; it
never evaluates test data. All runs load the same fingerprint-checked frozen
description-bank cache, which avoids re-encoding the identical 350 texts.

Checkpoint diagnostics showed that ordinary `set_attention` can collapse to
four nearly identical uniform distributions over the 50 descriptions. The
optional `cosine_set_attention` refinement normalizes queries and keys and uses
dimension-derived scaling, making the four description selectors non-uniform
without introducing a shot-specific temperature. It remains permutation
invariant and changes only the internal single-model description aggregation.
`grouped10_cosine_attention` first averages each ordered block of ten
BiomedCoOp descriptions, then applies the same four-query attention over the
five resulting semantic groups. This retains the requested ten-description
means without collapsing all 50 descriptions into one vector.
`grouped10_layer_residual` additionally addresses representation mismatch: the
frozen grouped bases already live in PubMedBERT's injection space, while the
trainable two-layer TKE learns only their residual correction. Its fifth group
contributes through the shared mean, so four prompt positions still consume all
50 descriptions. The basis, description bank, and class prior remain frozen;
only the TKE correction, CoOp context, and visual VPT are optimized.

### Frozen TextPrompt-baseline refinement

The final single-model configuration is
`CoOp_VPT_Deep_TextDeep_MT-TCP_LayerBasis_XProto_Nv4_Nt4_K50`. It starts from
the paired `CoOp_VPT_Deep_TextDeep_Nv4_Nt4` prompt checkpoint and preserves its
four CoOp tokens, four deep visual tokens, and four deep text-prompt tokens.
All 50 BiomedCoOp descriptions per class are encoded at the pre-block-8
PubMedBERT representation. Five ordered ten-description means form four
layer-aligned semantic bases; the fifth group enters through their shared mean.
A shared `768 -> 128 -> 4x768` QuickGELU TKE learns a residual correction.

Before PubMedBERT blocks 8–11, the class-common component is removed from the
four TKE tokens, their norms are matched to the existing TextPrompt tokens, and
the resulting class-differential residual is added with one trainable gate per
layer. The gate initialization is `0.05` for every shot and seed and uses the
same ten-epoch warmup. Training uses one model and one optimizer with the fixed
global objective: CE, centered knowledge consistency (weight 4), prompt anchor
(weight 4; relative-L2 coefficient 0.5), and class-balanced cross-modal
prototype alignment (weight 0.5, temperature 0.1). The prototype term is
training-only; inference still produces exactly one image/text similarity-logit
tensor.

The complete natural-distribution validation grid contains 4/8/16/32 shots,
seeds 1/2/3, and 100 epochs per run. All 12 manifests have the same architecture,
LR `0.005`, AdamW weight decay `0.0005`, loss weights, description count, gate,
and warmup; there are no shot-specific parameters. Each run is trained once and
retains both best-validation-ACC and best-validation-BACC prompt checkpoints.

On the 12 paired test runs, the best-validation-ACC checkpoint is the supported
selection: relative to the same-shot, same-seed TextPrompt baseline, mean BACC
changes from `48.04` to `50.79` (`+2.76` points), AUC from `83.92` to `85.48`
(`+1.56`), macro-F1 from `38.40` to `41.00` (`+2.60`), and ACC from `61.16` to
`60.89` (`-0.27`). BACC wins in 8/12 paired runs. The gains by shot are `+5.00`,
`+5.83`, `+1.70`, and `-1.51` BACC points for 4/8/16/32 shots respectively, so
the 32-shot limitation must remain visible.

The separately retained best-validation-BACC checkpoint does not establish an
improvement: its overall test BACC delta is `-0.05` points with 6/12 wins. It is
kept for audit and reproducibility but must not be reported as the successful
selection. The machine-readable and Markdown paired reports are under
`output/tcp_textbaseline_layerbasis_xproto_full100/` with prefix
`full100_layerbasis_xproto_validation`.
