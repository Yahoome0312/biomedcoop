"""Run the selected CoOp/VPT configurations on every requested shot/seed."""

import argparse
import json
import os
import subprocess
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
TRAINER = "CoOpVPT_BiomedCLIP"
METHOD_FILES = {
    "PureCoOp_AdamW": "selected_coop.json",
    "CoOp_VPT_Shallow": "selected_shallow.json",
    "CoOp_VPT_Deep": "selected_deep.json",
}


def run_one(args, method, selected, shots, seed):
    parameters = selected["parameters"]
    output = args.output_root / method / "shots_{}".format(shots) / "seed{}".format(seed)
    prompt_checkpoint = output / "prompt_learner" / "model-best.pth.tar"
    vpt_checkpoint = output / "vpt_prompt" / "model-best.pth.tar"
    complete_path = output / "run_complete.json"
    expected = [prompt_checkpoint]
    if parameters.get("vpt_enabled"):
        expected.append(vpt_checkpoint)
    if complete_path.exists() and all(path.exists() for path in expected):
        print("SKIP {} shots={} seed={} (complete)".format(method, shots, seed), flush=True)
        return

    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "method": method,
        "shots": shots,
        "seed": seed,
        "batch_size": args.batch_size,
        **parameters,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    opts = [
        "DATASET.NUM_SHOTS", str(shots),
        "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
        "DATALOADER.NUM_WORKERS", "4",
        "TEST.SKIP_FINAL_TEST", "False",
        "TEST.COMPUTE_CMAT", "True",
        "TRAINER.COOP.N_CTX", str(parameters["text_tokens"]),
        "TRAINER.COOPVPT.COOP_OPTIM.LR", str(parameters["coop_lr"]),
        "TRAINER.COOPVPT.COOP_OPTIM.WEIGHT_DECAY", str(parameters["coop_wd"]),
        "TRAINER.COOPVPT.VPT_ENABLED", str(bool(parameters.get("vpt_enabled", False))),
    ]
    if parameters.get("vpt_enabled"):
        opts.extend(
            [
                "TRAINER.COOPVPT.VPT_MODE", parameters["vpt_mode"],
                "TRAINER.COOPVPT.VPT_N_CTX", str(parameters["vpt_tokens"]),
                "TRAINER.COOPVPT.VPT_OPTIM.LR", str(parameters["vpt_lr"]),
                "TRAINER.COOPVPT.VPT_OPTIM.WEIGHT_DECAY", str(parameters["vpt_wd"]),
            ]
        )

    command = [
        args.python,
        "-u",
        "train.py",
        "--root", str(args.data_root),
        "--seed", str(seed),
        "--trainer", TRAINER,
        "--dataset-config-file", "configs/datasets/dermamnist.yaml",
        "--config-file", "configs/trainers/CoOpVPT/few_shot/dermamnist.yaml",
        "--output-dir", str(output),
        *opts,
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(REPO / "Dassl.pytorch")
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    print("RUN {} shots={} seed={}".format(method, shots, seed), flush=True)
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
        raise RuntimeError("Final run failed (exit={}):\n{}".format(process.returncode, tail))
    complete_path.write_text(
        json.dumps({"status": "complete", "exit_code": 0}, indent=2),
        encoding="utf-8",
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument("--search-root", type=Path, default=REPO / "output" / "dermamnist_coopvpt_search")
    parser.add_argument("--output-root", type=Path, default=REPO / "output" / "dermamnist_coopvpt_final")
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--methods", nargs="+", choices=tuple(METHOD_FILES), default=list(METHOD_FILES))
    # 32-shot is an explicit extension of the upstream 1/2/4/8/16-shot scripts.
    parser.add_argument("--shots", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    args = parser.parse_args()
    args.search_root = args.search_root.resolve()
    args.output_root = args.output_root.resolve()

    for method in args.methods:
        selection_path = args.search_root / METHOD_FILES[method]
        if not selection_path.exists():
            raise FileNotFoundError("Missing selected parameters: {}".format(selection_path))
        selected = json.loads(selection_path.read_text(encoding="utf-8"))
        for shots in args.shots:
            for seed in args.seeds:
                run_one(args, method, selected, shots, seed)


if __name__ == "__main__":
    main()
