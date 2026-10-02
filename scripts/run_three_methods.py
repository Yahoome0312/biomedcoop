"""Run the 108 from-scratch CoOp/Deep Prompt/Class Text Token experiments."""

import csv
import json
import os
import queue
import statistics
import subprocess
import threading
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output/coop_deep_prompt_class_text_token_seed123"
PYTHON = "/mnt/nas1/disk09/yuejianwu/.conda/envs/biocoop/bin/python"
METHODS = ("coop", "deep_prompt", "class_text_token")
DATASETS = ("DermaMNIST", "Kvasir", "CHMNIST")
SHOTS = (4, 8, 16, 32)
SEEDS = (1, 2, 3)
GPUS = (0, 1, 2, 3, 4, 6, 7)
CASES = [(method, dataset, shots, seed) for dataset in DATASETS
         for shots in SHOTS for seed in SEEDS for method in METHODS]


def run_dir(case):
    method, dataset, shots, seed = case
    return OUT / method / dataset / f"shots_{shots}" / f"seed{seed}"


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(path)


def run_case(gpu, case, insert_layer=7, output_dir=None, cvp_fusion_weight=0.5):
    method, dataset, shots, seed = case
    dest = Path(output_dir) if output_dir is not None else run_dir(case)
    if (dest / "test_metrics.json").exists():
        return "already_complete"
    dest.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=str(ROOT), TMPDIR="/tmp",
               HF_HUB_OFFLINE="1")
    is_coop = method == "coop"
    opts = ["DATASET.NUM_SHOTS", str(shots), "TEST.SKIP_FINAL_TEST", "True",
            "TEST.SAVE_BEST_METRICS", "['accuracy']", "TRAINER.TCP.INSERT_LAYER", str(insert_layer)]
    if not is_coop:
        opts += ["TRAINER.TCP.ENABLED", str(method in ("class_text_token", "fusion", "cvp")),
                 "TRAINER.TCP.FUSION_WEIGHT",
                 "0.5" if method in ("fusion", "cvp") else "1.0"]
    if method == "cvp":
        opts += ["TRAINER.CVP.ENABLED", "True", "TRAINER.CVP.INSERT_LAYER", "7",
                 "TRAINER.CVP.NUM_TOKENS", "4", "TRAINER.CVP.BOTTLENECK_DIM", "128",
                 "TRAINER.CVP.FUSION_WEIGHT", str(cvp_fusion_weight)]
    checkpoint_dir = "prompt_learner" if is_coop else "prompt_parameters"
    if not (dest / checkpoint_dir / "model.pth.tar-100").exists():
        command = [PYTHON, "-u", str(ROOT / "train.py"), "--root", str(ROOT / "data"),
                   "--output-dir", str(dest), "--seed", str(seed),
                   "--trainer", "CoOp_BiomedCLIP" if is_coop else "CoOpVPT_BiomedCLIP",
                   "--dataset-config-file", str(ROOT / "configs/datasets" / (dataset.lower() + ".yaml")),
                   "--config-file", str(ROOT / "configs/trainers/CoOp" / (
                       "dermamnist_native.yaml" if is_coop else "dermamnist_native_vpt_tcp.yaml")),
                   *opts]
        with (dest / "train.stdout.log").open("a") as log:
            code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                                  stderr=subprocess.STDOUT).returncode
        if code:
            raise RuntimeError(f"Training exited {code}: {dest}")
    command = [PYTHON, "-u", str(ROOT / "scripts/evaluate_three_methods.py"),
               "--run-dir", str(dest), "--method", method, "--dataset", dataset,
               "--shots", str(shots), "--seed", str(seed), "--insert-layer", str(insert_layer)]
    if method == "cvp":
        command += ["--cvp-fusion-weight", str(cvp_fusion_weight)]
    with (dest / "test.stdout.log").open("a") as log:
        code = subprocess.run(command, cwd=ROOT, env=env, stdout=log,
                              stderr=subprocess.STDOUT).returncode
    if code:
        raise RuntimeError(f"Test exited {code}: {dest}")
    return "complete"


def summarize():
    rows = []
    for case in CASES:
        path = run_dir(case) / "test_metrics.json"
        if path.exists():
            rows.append(json.loads(path.read_text()))
    if rows:
        fields = ("method", "dataset", "shots", "seed", "selected_epoch",
                  "validation_accuracy", "test_accuracy")
        with (OUT / "results_detailed.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        summary = []
        for method in METHODS:
            for dataset in DATASETS:
                for shots in SHOTS:
                    values = [row["test_accuracy"] for row in rows if
                              (row["method"], row["dataset"], row["shots"]) ==
                              (method, dataset, shots)]
                    if len(values) == 3:
                        summary.append(dict(method=method, dataset=dataset, shots=shots,
                                            mean=statistics.mean(values),
                                            std=statistics.stdev(values)))
        with (OUT / "results_summary.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=("method", "dataset", "shots", "mean", "std"))
            writer.writeheader()
            writer.writerows(summary)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for case in CASES:
        pending.put(case)
    lock = threading.Lock()
    records = []

    def worker(gpu):
        while True:
            try:
                case = pending.get_nowait()
            except queue.Empty:
                return
            try:
                status = run_case(gpu, case)
                record = dict(case=case, gpu=gpu, status=status)
            except Exception as exc:
                record = dict(case=case, gpu=gpu, status="failed", error=str(exc))
            with lock:
                records.append(record)
                write_json(OUT / "_manager/status.json", dict(total=len(CASES),
                           completed=sum(item["status"] != "failed" for item in records),
                           processed=len(records), records=records))
                summarize()
                print(json.dumps(record), flush=True)
            pending.task_done()

    threads = [threading.Thread(target=worker, args=(gpu,)) for gpu in GPUS]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()


if __name__ == "__main__":
    main()
