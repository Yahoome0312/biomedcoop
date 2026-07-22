"""Run selected shared-LR CoOp + VPT-Deep configurations on all shots/seeds."""

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRAINER = "CoOpVPT_BiomedCLIP"
DATASET_CONFIG = "configs/datasets/dermamnist.yaml"
TRAINER_CONFIG = "configs/trainers/CoOp/dermamnist_native_vpt.yaml"


def run_one(args, tokens, selected, shots, seed):
    parameters = selected["parameters"]
    method = "CoOp_VPT_Deep_Nv{}".format(tokens)
    output = args.output_root / method / "shots_{}".format(shots) / "seed{}".format(seed)
    final_checkpoint = output / "prompt_parameters" / "model.pth.tar-100"
    complete_path = output / "run_complete.json"
    if complete_path.exists() and final_checkpoint.exists():
        print(
            "SKIP {} shots={} seed={} (complete)".format(method, shots, seed),
            flush=True,
        )
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": method,
        "shots": shots,
        "seed": seed,
        "batch_size": args.batch_size,
        "text_tokens": 4,
        "ctx_init": "a photo of a",
        **parameters,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    shared_lr = parameters["shared_lr"]
    opts = [
        "DATASET.NUM_SHOTS", str(shots),
        "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
        "DATALOADER.NUM_WORKERS", "4",
        "TEST.FINAL_MODEL", "last_step",
        "TEST.SKIP_FINAL_TEST", "False",
        "TEST.COMPUTE_CMAT", "True",
        "TRAINER.COOP.N_CTX", "4",
        "TRAINER.COOP.CTX_INIT", "a photo of a",
        "TRAINER.COOPVPT.PREC", "fp32",
        "TRAINER.COOPVPT.VPT_ENABLED", "True",
        "TRAINER.COOPVPT.VPT_MODE", "deep",
        "TRAINER.COOPVPT.VPT_N_CTX", str(tokens),
        "OPTIM.LR", str(shared_lr),
        "TRAINER.COOPVPT.OPTIM.LR", str(shared_lr),
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
            method, shots, seed, shared_lr
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
        )[-8000:]
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
        "--search-root",
        type=Path,
        default=REPO / "output" / "dermamnist_coop_native_vptdeep_search",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "dermamnist_coop_native_vptdeep_adamw",
    )
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--budgets", nargs="+", type=int, default=(1, 2, 5, 10, 20))
    parser.add_argument("--shots", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    args = parser.parse_args()
    args.search_root = args.search_root.resolve()
    args.output_root = args.output_root.resolve()

    selection_path = args.search_root / "selected_vpt_deep.json"
    if not selection_path.exists():
        raise FileNotFoundError("Missing selected parameters: {}".format(selection_path))
    selected_all = json.loads(selection_path.read_text(encoding="utf-8"))

    for tokens in args.budgets:
        selected = selected_all["selections"].get(str(tokens))
        if selected is None:
            raise KeyError("No selected VPT-Deep configuration for budget {}".format(tokens))
        for shots in args.shots:
            for seed in args.seeds:
                run_one(args, tokens, selected, shots, seed)


if __name__ == "__main__":
    main()
