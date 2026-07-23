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
```

Search runs use validation accuracy, matching the original BiomedCoOp
checkpoint-selection protocol, and skip the test split. Final runs default to
the last epoch and evaluate the test split once. Passing `-FinalModel
best_val` instead evaluates the checkpoint with the highest validation
accuracy. Shallow is not part of this experiment workflow.

Every output directory contains `run_manifest.json`, `console.log`, Dassl's
`log.txt`, and prompt-only checkpoints. Re-running a command skips complete
runs and continues the remaining grid.
