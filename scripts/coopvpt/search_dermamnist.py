"""Validation-only shared-LR search for native CoOp + VPT-Deep."""

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
TRAINER = "CoOpVPT_BiomedCLIP"
DATASET_CONFIG = "configs/datasets/dermamnist.yaml"
TRAINER_CONFIG = "configs/trainers/CoOp/dermamnist_native_vpt.yaml"
VPT_TOKENS = (1, 2, 5, 10, 20)
SHARED_LRS = (1e-4, 5e-4, 1e-3, 2e-3, 5e-3)


def number(value):
    return "{:.0e}".format(value).replace("-0", "-").replace("+0", "+")


def candidate_name(candidate):
    return "deep_v{}_lr{}".format(
        candidate["vpt_tokens"], number(candidate["shared_lr"])
    )


def trial_dir(root, stage, candidate, seed):
    return root / stage / candidate_name(candidate) / "seed{}".format(seed)


def load_json(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_trial(args, root, stage, candidate, seed):
    output = trial_dir(root, stage, candidate, seed)
    result_path = output / "best_validation_accuracy.json"
    complete_path = output / "search_complete.json"
    final_checkpoint = output / "prompt_parameters" / "model.pth.tar-100"
    log_path = output / "log.txt"

    if (
        not complete_path.exists()
        and result_path.exists()
        and final_checkpoint.exists()
        and log_path.exists()
        and "Finished training" in log_path.read_text(
            encoding="utf-8", errors="replace"
        )
    ):
        complete_path.write_text(
            json.dumps({"status": "complete", "exit_code": 0}, indent=2),
            encoding="utf-8",
        )

    if complete_path.exists() and result_path.exists():
        print(
            "SKIP {} seed={} (complete)".format(candidate_name(candidate), seed),
            flush=True,
        )
        return load_json(result_path)

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "seed": seed,
        "shots": 4,
        "batch_size": args.batch_size,
        "method": "CoOp_VPT_Deep",
        "text_tokens": 4,
        "ctx_init": "a photo of a",
        **candidate,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    opts = [
        "DATASET.NUM_SHOTS", "4",
        "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
        "DATALOADER.NUM_WORKERS", "4",
        "TEST.FINAL_MODEL", "best_val",
        "TEST.BEST_METRIC", "accuracy",
        "TEST.SKIP_FINAL_TEST", "True",
        "TEST.COMPUTE_CMAT", "False",
        "TRAINER.COOP.N_CTX", "4",
        "TRAINER.COOP.CTX_INIT", "a photo of a",
        "TRAINER.COOPVPT.PREC", "fp32",
        "TRAINER.COOPVPT.VPT_ENABLED", "True",
        "TRAINER.COOPVPT.VPT_MODE", "deep",
        "TRAINER.COOPVPT.VPT_N_CTX", str(candidate["vpt_tokens"]),
        "OPTIM.LR", str(candidate["shared_lr"]),
        "TRAINER.COOPVPT.OPTIM.LR", str(candidate["shared_lr"]),
    ]
    command = [
        args.python,
        "-u",
        "train.py",
        "--root", str(args.data_root),
        "--seed", str(seed),
        "--trainer", TRAINER,
        "--dataset-config-file", DATASET_CONFIG,
        "--config-file", TRAINER_CONFIG,
        "--output-dir", str(output),
        *opts,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "Dassl.pytorch")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    print("RUN {} seed={}".format(candidate_name(candidate), seed), flush=True)
    with (output / "console.log").open("w", encoding="utf-8") as console:
        process = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=console,
            stderr=subprocess.STDOUT,
            text=True,
        )
    if process.returncode:
        tail = (output / "console.log").read_text(encoding="utf-8", errors="replace")[-8000:]
        raise RuntimeError("Trial failed (exit={}):\n{}".format(process.returncode, tail))
    if not result_path.exists():
        raise RuntimeError(
            "Trial completed without best_validation_accuracy.json: {}".format(output)
        )
    balanced_result_path = output / "best_validation_balanced_accuracy.json"
    if not balanced_result_path.exists():
        raise RuntimeError(
            "Trial completed without best_validation_balanced_accuracy.json: {}".format(
                output
            )
        )
    complete_path.write_text(
        json.dumps({"status": "complete", "exit_code": 0}, indent=2),
        encoding="utf-8",
    )
    return load_json(result_path)


def score_records(records, candidate):
    metrics = [record["metrics"] for record in records]
    return (
        float(np.mean([item["accuracy"] for item in metrics])),
        float(np.mean([item["auc"] for item in metrics])),
        float(np.mean([item["balanced_accuracy"] for item in metrics])),
        -float(candidate["shared_lr"]),
    )


def rank_candidates(args, root, stage, candidates, seeds):
    ranked = []
    for candidate in candidates:
        records = [run_trial(args, root, stage, candidate, seed) for seed in seeds]
        ranked.append((score_records(records, candidate), candidate, records))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def search_one_budget(args, root, tokens):
    candidates = [
        {"vpt_mode": "deep", "vpt_tokens": tokens, "shared_lr": lr}
        for lr in SHARED_LRS
    ]
    stage = "vpt_deep_v{}_screen".format(tokens)
    screen = rank_candidates(args, root, stage, candidates, (1,))
    top2 = [item[1] for item in screen[:2]]
    ranked = rank_candidates(args, root, stage, top2, (1, 2, 3))
    score, candidate, records = ranked[0]
    return {
        "score": {
            "accuracy": score[0],
            "auc": score[1],
            "balanced_accuracy": score[2],
        },
        "parameters": candidate,
        "validation_runs": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "dermamnist_coop_native_vptdeep_search",
    )
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    selections = {}
    for tokens in VPT_TOKENS:
        selections[str(tokens)] = search_one_budget(
            args, args.output_root, tokens
        )

    payload = {
        "method": "CoOp_VPT_Deep",
        "text_tokens": 4,
        "ctx_init": "a photo of a",
        "shared_lr_candidates": list(SHARED_LRS),
        "selections": selections,
    }
    output = args.output_root / "selected_vpt_deep.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SELECTED VPT-Deep configurations -> {}".format(output), flush=True)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
