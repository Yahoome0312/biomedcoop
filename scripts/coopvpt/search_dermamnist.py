"""Resumable validation-only hyperparameter search for CoOp and VPT."""

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
TRAINER_CONFIG = "configs/trainers/CoOpVPT/few_shot/dermamnist.yaml"


def number(value):
    return "{:.0e}".format(value).replace("-0", "-").replace("+0", "+")


def candidate_name(candidate):
    parts = ["t{}".format(candidate["text_tokens"]), "tlr{}".format(number(candidate["coop_lr"]))]
    parts.append("twd{}".format(number(candidate["coop_wd"])))
    if candidate.get("vpt_enabled"):
        parts.extend(
            [
                candidate["vpt_mode"],
                "v{}".format(candidate["vpt_tokens"]),
                "vlr{}".format(number(candidate["vpt_lr"])),
                "vwd{}".format(number(candidate["vpt_wd"])),
            ]
        )
    return "_".join(parts)


def trial_dir(root, stage, candidate, seed):
    return root / stage / candidate_name(candidate) / "seed{}".format(seed)


def load_validation(path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_trial(args, root, stage, candidate, seed):
    output = trial_dir(root, stage, candidate, seed)
    result_path = output / "best_validation.json"
    complete_path = output / "search_complete.json"
    manifest_path = output / "run_manifest.json"
    # Backfill the marker for runs produced by the first version of this
    # driver, which used the final epoch checkpoint as the only completion
    # evidence.
    final_checkpoint = output / "prompt_learner" / "model.pth.tar-100"
    log_path = output / "log.txt"
    if (
        not complete_path.exists()
        and result_path.exists()
        and final_checkpoint.exists()
        and log_path.exists()
        and "Finished training" in log_path.read_text(encoding="utf-8", errors="replace")
    ):
        complete_path.write_text(
            json.dumps({"status": "complete", "exit_code": 0}, indent=2),
            encoding="utf-8",
        )
    if complete_path.exists() and result_path.exists():
        print("SKIP {} seed={} (complete)".format(candidate_name(candidate), seed), flush=True)
        return load_validation(result_path)

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "stage": stage,
        "seed": seed,
        "shots": 4,
        "batch_size": args.batch_size,
        **candidate,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    opts = [
        "DATASET.NUM_SHOTS", "4",
        "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
        "DATALOADER.NUM_WORKERS", "4",
        "TEST.SKIP_FINAL_TEST", "True",
        "TEST.COMPUTE_CMAT", "False",
        "TRAINER.COOP.N_CTX", str(candidate["text_tokens"]),
        "TRAINER.COOPVPT.COOP_OPTIM.LR", str(candidate["coop_lr"]),
        "TRAINER.COOPVPT.COOP_OPTIM.WEIGHT_DECAY", str(candidate["coop_wd"]),
        "TRAINER.COOPVPT.VPT_ENABLED", str(bool(candidate.get("vpt_enabled", False))),
    ]
    if candidate.get("vpt_enabled"):
        opts.extend(
            [
                "TRAINER.COOPVPT.VPT_MODE", candidate["vpt_mode"],
                "TRAINER.COOPVPT.VPT_N_CTX", str(candidate["vpt_tokens"]),
                "TRAINER.COOPVPT.VPT_OPTIM.LR", str(candidate["vpt_lr"]),
                "TRAINER.COOPVPT.VPT_OPTIM.WEIGHT_DECAY", str(candidate["vpt_wd"]),
            ]
        )

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
        raise RuntimeError("Trial completed without best_validation.json: {}".format(output))
    complete_path.write_text(
        json.dumps({"status": "complete", "exit_code": 0}, indent=2),
        encoding="utf-8",
    )
    return load_validation(result_path)


def score_records(records, candidate):
    metrics = [record["metrics"] for record in records]
    return (
        float(np.mean([item["balanced_accuracy"] for item in metrics])),
        float(np.mean([item["auc"] for item in metrics])),
        float(np.mean([item["accuracy"] for item in metrics])),
        -int(candidate.get("vpt_tokens", candidate["text_tokens"])),
    )


def rank_candidates(args, root, stage, candidates, seeds):
    ranked = []
    for candidate in candidates:
        records = [run_trial(args, root, stage, candidate, seed) for seed in seeds]
        ranked.append((score_records(records, candidate), candidate, records))
    ranked.sort(key=lambda item: item[0], reverse=True)
    return ranked


def save_selection(path, score, candidate, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "score": {
            "balanced_accuracy": score[0],
            "auc": score[1],
            "accuracy": score[2],
        },
        "parameters": candidate,
        "validation_runs": records,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("SELECTED {} -> {}".format(candidate_name(candidate), path), flush=True)
    return payload


def search_coop(args, root):
    screen = [
        {
            "vpt_enabled": False,
            "text_tokens": tokens,
            "coop_lr": lr,
            "coop_wd": 1e-4,
        }
        for tokens, lr in product((4, 8, 16), (1e-4, 5e-4, 1e-3))
    ]
    seed1 = rank_candidates(args, root, "coop_screen", screen, (1,))
    top3 = [item[1] for item in seed1[:3]]
    ranked_top = rank_candidates(args, root, "coop_screen", top3, (1, 2, 3))
    base = ranked_top[0][1]

    wd_candidates = [{**base, "coop_wd": wd} for wd in (0.0, 1e-4, 1e-3)]
    ranked_wd = rank_candidates(args, root, "coop_weight_decay", wd_candidates, (1, 2, 3))
    return save_selection(root / "selected_coop.json", *ranked_wd[0])


def search_vpt_mode(args, root, coop, mode):
    base = coop["parameters"]
    screen = [
        {
            **base,
            "vpt_enabled": True,
            "vpt_mode": mode,
            "vpt_tokens": tokens,
            "vpt_lr": lr,
            "vpt_wd": 1e-4,
        }
        for tokens, lr in product((1, 5, 10, 20), (1e-4, 5e-4, 1e-3))
    ]
    stage = "{}_screen".format(mode)
    seed1 = rank_candidates(args, root, stage, screen, (1,))
    top3 = [item[1] for item in seed1[:3]]
    ranked_top = rank_candidates(args, root, stage, top3, (1, 2, 3))
    candidate = ranked_top[0][1]

    joint = [
        {
            **candidate,
            "coop_lr": candidate["coop_lr"] * text_factor,
            "vpt_lr": candidate["vpt_lr"] * visual_factor,
        }
        for text_factor, visual_factor in product((0.5, 1.0, 2.0), repeat=2)
    ]
    joint_stage = "{}_joint_lr".format(mode)
    joint_seed1 = rank_candidates(args, root, joint_stage, joint, (1,))
    top2 = [item[1] for item in joint_seed1[:2]]
    ranked_joint = rank_candidates(args, root, joint_stage, top2, (1, 2, 3))
    candidate = ranked_joint[0][1]

    wd_candidates = [{**candidate, "vpt_wd": wd} for wd in (0.0, 1e-4, 1e-3)]
    ranked_wd = rank_candidates(
        args, root, "{}_weight_decay".format(mode), wd_candidates, (1, 2, 3)
    )
    return save_selection(root / "selected_{}.json".format(mode), *ranked_wd[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument("--output-root", type=Path, default=REPO / "output" / "dermamnist_coopvpt_search")
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--phase", choices=("coop", "vpt", "all"), default="all")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()

    coop_path = args.output_root / "selected_coop.json"
    if args.phase in {"coop", "all"}:
        coop = search_coop(args, args.output_root)
    elif coop_path.exists():
        coop = json.loads(coop_path.read_text(encoding="utf-8"))
    else:
        raise FileNotFoundError("Run --phase coop first: {}".format(coop_path))

    if args.phase in {"vpt", "all"}:
        shallow = search_vpt_mode(args, args.output_root, coop, "shallow")
        deep = search_vpt_mode(args, args.output_root, coop, "deep")
        summary = {"coop": coop, "shallow": shallow, "deep": deep}
        (args.output_root / "search_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
