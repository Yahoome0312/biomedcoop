"""Run the fixed CoOp + visual VPT-Deep + text Deep Prompt comparison."""

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRAINER = "CoOpVPT_BiomedCLIP"
DATASET_CONFIG = "configs/datasets/dermamnist.yaml"
TRAINER_CONFIG = "configs/trainers/CoOp/dermamnist_native_text_vpt.yaml"
METHOD = "CoOp_VPT_Deep_TextDeep_Nv5_Nt4"
SHARED_LR = 5e-3


def run_one(args, shots, seed):
    output = args.output_root / METHOD / "shots_{}".format(shots) / "seed{}".format(seed)
    final_checkpoint = output / "prompt_parameters" / "model.pth.tar-100"
    best_checkpoint = output / "prompt_parameters" / "model-best.pth.tar"
    complete_path = output / "run_complete.json"
    checkpoint_ready = final_checkpoint.exists()
    if args.final_model == "best_val":
        checkpoint_ready = checkpoint_ready and best_checkpoint.exists()
    if complete_path.exists() and checkpoint_ready:
        print(
            "SKIP {} shots={} seed={} (complete)".format(METHOD, shots, seed),
            flush=True,
        )
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": METHOD,
        "shots": shots,
        "seed": seed,
        "batch_size": args.batch_size,
        "num_workers": 4,
        "final_model": args.final_model,
        "best_metric": "accuracy",
        "loss": "cross_entropy",
        "optimizer": "AdamW",
        "shared_lr": SHARED_LR,
        "weight_decay": 5e-4,
        "epochs": 100,
        "text_tokens": 4,
        "text_prompt_mode": "deep",
        "visual_tokens": 5,
        "visual_prompt_mode": "deep",
        "ctx_init": "a photo of a",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    opts = [
        "DATASET.NUM_SHOTS", str(shots),
        "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
        "DATALOADER.NUM_WORKERS", "4",
        "TEST.FINAL_MODEL", args.final_model,
        "TEST.BEST_METRIC", "accuracy",
        "TEST.SKIP_FINAL_TEST", "False",
        "TEST.COMPUTE_CMAT", "True",
        "TRAINER.COOP.N_CTX", "4",
        "TRAINER.COOP.CTX_INIT", "a photo of a",
        "TRAINER.COOPVPT.PREC", "fp32",
        "TRAINER.COOPVPT.VPT_ENABLED", "True",
        "TRAINER.COOPVPT.VPT_MODE", "deep",
        "TRAINER.COOPVPT.VPT_N_CTX", "5",
        "TRAINER.COOPVPT.TEXT_VPT_ENABLED", "True",
        "TRAINER.COOPVPT.TEXT_VPT_MODE", "deep",
        "TRAINER.COOPVPT.TEXT_VPT_N_CTX", "4",
        "TRAINER.COOPVPT.TEXT_VPT_DROPOUT", "0.0",
        "TRAINER.COOPVPT.TEXT_VPT_INIT", "normal",
        "OPTIM.NAME", "adamw",
        "OPTIM.LR", str(SHARED_LR),
        "TRAINER.COOPVPT.OPTIM.NAME", "adamw",
        "TRAINER.COOPVPT.OPTIM.LR", str(SHARED_LR),
        "TRAINER.COOPVPT.OPTIM.WEIGHT_DECAY", "0.0005",
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
    print(
        "RUN {} shots={} seed={} lr={}".format(
            METHOD, shots, seed, SHARED_LR
        ),
        flush=True,
    )
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
        tail = (output / "console.log").read_text(
            encoding="utf-8", errors="replace"
        )[-12000:]
        raise RuntimeError(
            "Final run failed (exit={}):\n{}".format(process.returncode, tail)
        )
    complete_path.write_text(
        json.dumps({"status": "complete", "exit_code": 0}, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "dermamnist_coop_native_vptdeep_textdeep_adamw_lr005",
    )
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--final-model",
        choices=("last_step", "best_val"),
        default="last_step",
        help="Checkpoint used for final test evaluation.",
    )
    parser.add_argument("--shots", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    args = parser.parse_args()
    args.data_root = args.data_root.resolve()
    args.output_root = args.output_root.resolve()

    for shots in args.shots:
        for seed in args.seeds:
            run_one(args, shots, seed)


if __name__ == "__main__":
    main()
