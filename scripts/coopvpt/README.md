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
