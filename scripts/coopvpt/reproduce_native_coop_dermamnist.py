"""Run the repository-native CE-only CoOp protocol on DermaMNIST."""

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRAINER = "CoOp_BiomedCLIP"
CONFIG = "configs/trainers/CoOp/dermamnist_native.yaml"
LAYOUT = "nctx4_cscFalse_ctpend"


def run_one(args, shots, seed):
    output = (
        args.output_root
        / "shots_{}".format(shots)
        / TRAINER
        / LAYOUT
        / "seed{}".format(seed)
    )
    checkpoint = output / "prompt_learner" / "model.pth.tar-100"
    complete = output / "run_complete.json"
    if complete.exists() and checkpoint.exists():
        print("SKIP shots={} seed={} (complete)".format(shots, seed), flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "protocol": "repository_native_coop",
        "trainer": TRAINER,
        "shots": shots,
        "seed": seed,
        "text_context_tokens": 4,
        "context_init": "a photo of a",
        "optimizer": "sgd",
        "lr": 0.002,
        "momentum": 0.9,
        "batch_size": 32,
        "workers": 4,
        "persistent_workers": True,
        "scheduler": "cosine",
        "epochs": 100,
        "checkpoint_policy": "last_step",
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    command = [
        args.python,
        "-u",
        "train.py",
        "--root", str(args.data_root),
        "--seed", str(seed),
        "--trainer", TRAINER,
        "--dataset-config-file", "configs/datasets/dermamnist.yaml",
        "--config-file", CONFIG,
        "--output-dir", str(output),
        "DATASET.NUM_SHOTS", str(shots),
        "DATALOADER.TRAIN_X.BATCH_SIZE", "32",
        "DATALOADER.NUM_WORKERS", "4",
        "TEST.PER_CLASS_RESULT", "True",
        "TEST.COMPUTE_CMAT", "True",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "Dassl.pytorch")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    print("RUN shots={} seed={}".format(shots, seed), flush=True)
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
        )[-8000:]
        raise RuntimeError("Native CoOp run failed (exit={}):\n{}".format(
            process.returncode, tail
        ))
    complete.write_text(
        json.dumps({"status": "complete", "exit_code": 0}, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "dermamnist_coop_native_sgd_lr002_bs32",
    )
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
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
