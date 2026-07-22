# DermaMNIST CoOp + VPT reproduction

The new trainer is `CoOpVPT_BiomedCLIP`. It always uses cross-entropy only.
With VPT disabled, only the CoOp text context is trainable. With VPT enabled,
the text context and visual prompts use separate AdamW optimizers, schedulers,
and checkpoints.

The dataset CLI root is `D:\Data\dermamnist`; the existing junction below it
points to `DermaMNIST-224`.

Run the stages in order:

```powershell
# 1. Select the fair pure-CoOp AdamW baseline using validation only.
.\scripts\coopvpt\search_dermamnist.ps1 -Phase coop

# 2. Run all 1/2/4/8/16/32-shot pure-CoOp final experiments.
.\scripts\coopvpt\reproduce_dermamnist.ps1 -Methods PureCoOp_AdamW

# 3. Search shallow and deep VPT after the pure-CoOp baseline is complete.
.\scripts\coopvpt\search_dermamnist.ps1 -Phase vpt

# 4. Run the selected VPT configurations.
.\scripts\coopvpt\reproduce_dermamnist.ps1 `
    -Methods CoOp_VPT_Shallow,CoOp_VPT_Deep

# 5. Produce JSON, CSV, and Markdown summaries.
D:\Anaconda\python.exe scripts\coopvpt\aggregate_results.py
```

Search runs set `TEST.SKIP_FINAL_TEST=True`; they select checkpoints using
validation balanced accuracy and never evaluate the test split. Final runs
load the best validation checkpoint and evaluate the test split once.

Every output directory contains `run_manifest.json`, `console.log`, Dassl's
`log.txt`, and prompt-only checkpoints. Re-running a command skips complete
runs and resumes the remaining grid.
