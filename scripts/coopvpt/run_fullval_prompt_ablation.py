"""Complete and summarize the three 4-token full-validation experiments.

The missing arm is CoOp + visual VPT-Deep (Nv=4, no text VPT). Each run is
trained once and evaluated twice: once from the best validation-accuracy
checkpoint and once from the best validation-balanced-accuracy checkpoint.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from argparse import Namespace
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DASSL = REPO / "Dassl.pytorch"
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(DASSL) not in sys.path:
    sys.path.insert(0, str(DASSL))

METHOD = "CoOp_VPT_Deep_Nv4"
SELECTION_METRICS = ("accuracy", "balanced_accuracy")
REPORT_METRICS = ("accuracy", "balanced_accuracy", "auc", "macro_f1")


def _json_read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _run_dir(output_root, shots, seed):
    return output_root / METHOD / "shots_{}".format(shots) / "seed{}".format(seed)


def _is_complete(run_dir):
    required = [
        run_dir / "run_complete.json",
        run_dir / "results.json",
        run_dir / "test_by_selection.json",
        run_dir / "best_validation_accuracy.json",
        run_dir / "best_validation_balanced_accuracy.json",
    ]
    required.extend(
        run_dir / "prompt_parameters" / "model-best-{}.pth.tar".format(metric)
        for metric in SELECTION_METRICS
    )
    return all(path.exists() for path in required)


def _worker_args(args):
    resume_dir = ""
    if (args.run_dir / "prompt_parameters" / "checkpoint").exists():
        resume_dir = str(args.run_dir)
    return Namespace(
        root=str(args.data_root),
        output_dir=str(args.run_dir),
        resume=resume_dir,
        seed=args.seed,
        source_domains=None,
        target_domains=None,
        transforms=None,
        trainer="CoOpVPT_BiomedCLIP",
        backbone="",
        head="",
        dataset_config_file=str(REPO / "configs/datasets/dermamnist.yaml"),
        config_file=str(REPO / "configs/trainers/CoOp/dermamnist_native_vpt.yaml"),
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=[
            "DATASET.NUM_SHOTS", str(args.shots),
            "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
            "DATALOADER.NUM_WORKERS", str(args.num_workers),
            "DATALOADER.PERSISTENT_WORKERS", "True" if args.num_workers > 0 else "False",
            "OPTIM.NAME", "adamw",
            "OPTIM.LR", str(args.learning_rate),
            "OPTIM.WEIGHT_DECAY", str(args.weight_decay),
            "OPTIM.MAX_EPOCH", str(args.epochs),
            "TRAIN.CHECKPOINT_FREQ", "10",
            "TEST.FINAL_MODEL", "last_step",
            "TEST.BEST_METRIC", "accuracy",
            "TEST.SKIP_FINAL_TEST", "True",
            "TEST.PER_CLASS_RESULT", "True",
            "TEST.COMPUTE_CMAT", "True",
            "TRAINER.COOP.N_CTX", "4",
            "TRAINER.COOP.CTX_INIT", "a photo of a",
            "TRAINER.COOPVPT.PREC", "fp32",
            "TRAINER.COOPVPT.VPT_ENABLED", "True",
            "TRAINER.COOPVPT.VPT_MODE", "deep",
            "TRAINER.COOPVPT.VPT_N_CTX", "4",
            "TRAINER.COOPVPT.VPT_DROPOUT", "0.0",
            "TRAINER.COOPVPT.VPT_INIT", "uniform",
            "TRAINER.COOPVPT.TEXT_VPT_ENABLED", "False",
            "TRAINER.COOPVPT.OPTIM.NAME", "adamw",
            "TRAINER.COOPVPT.OPTIM.LR", str(args.learning_rate),
            "TRAINER.COOPVPT.OPTIM.WEIGHT_DECAY", str(args.weight_decay),
            "TRAINER.COOPVPT.OPTIM.MAX_EPOCH", str(args.epochs),
        ],
    )


def run_worker(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    import torch
    import train
    from dassl.engine import build_trainer
    from dassl.utils import load_checkpoint, set_random_seed, setup_logger

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if _is_complete(run_dir) and not args.force:
        print("SKIP shots={} seed={} (complete)".format(args.shots, args.seed), flush=True)
        return

    manifest = {
        "method": METHOD,
        "shots": args.shots,
        "seed": args.seed,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "final_model": "best_val",
        "selection_metrics": list(SELECTION_METRICS),
        "validation_split": "complete official validation split",
        "checkpoint_count": 2,
        "loss": "cross_entropy",
        "optimizer": "AdamW",
        "shared_lr": args.learning_rate,
        "weight_decay": args.weight_decay,
        "epochs": args.epochs,
        "text_tokens": 0,
        "text_prompt_mode": "disabled",
        "visual_tokens": 4,
        "visual_prompt_mode": "deep",
        "ctx_init": "a photo of a",
    }
    _json_write(run_dir / "run_manifest.json", manifest)

    train_args = _worker_args(args)
    cfg = train.setup_cfg(train_args)
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print(
        "RUN {} shots={} seed={} epochs={} lr={}".format(
            METHOD, args.shots, args.seed, args.epochs, args.learning_rate
        ),
        flush=True,
    )
    trainer = build_trainer(cfg)
    trainer.train()
    # after_train closes TensorBoard; suppress scalar writes during the two
    # explicit deployment evaluations below.
    trainer._writer = None

    test_by_selection = {}
    selections = {}
    for metric in SELECTION_METRICS:
        record = _json_read(run_dir / "best_validation_{}.json".format(metric))
        checkpoint_path = (
            run_dir
            / "prompt_parameters"
            / "model-best-{}.pth.tar".format(metric)
        )
        checkpoint = load_checkpoint(str(checkpoint_path))
        trainer.prompt_parameters.load_state_dict(checkpoint["state_dict"], strict=True)
        trainer.test(split="test")
        test_metrics = {
            key: float(value) for key, value in trainer.last_eval_results.items()
        }
        test_by_selection[metric] = test_metrics
        selections[metric] = {
            "best_epoch": int(record["epoch"]),
            "selection_value": float(record["selection_value"]),
            "validation": {
                key: float(value) for key, value in record["metrics"].items()
            },
            "test": test_metrics,
        }

        selection_dir = run_dir / "selection_{}".format(metric) / "prompt_parameters"
        selection_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint_path, selection_dir / "model-best.pth.tar")
        (selection_dir / "checkpoint").write_text(
            "model_checkpoint_path: model-best.pth.tar\n", encoding="utf-8"
        )
        cmat = run_dir / "cmat.pt"
        if cmat.exists():
            shutil.copyfile(cmat, run_dir / "test_cmat_selected_by_{}.pt".format(metric))

    _json_write(run_dir / "test_by_selection.json", test_by_selection)
    _json_write(
        run_dir / "results.json",
        {
            "method": METHOD,
            "shots": args.shots,
            "seed": args.seed,
            "selection_metrics": list(SELECTION_METRICS),
            "selections": selections,
        },
    )
    _json_write(
        run_dir / "run_complete.json",
        {
            "status": "complete",
            "exit_code": 0,
            "selection_metrics": list(SELECTION_METRICS),
        },
    )
    print("DONE shots={} seed={}".format(args.shots, args.seed), flush=True)


def launch_all(args):
    pending = []
    for shots in args.shots:
        for seed in args.seeds:
            run_dir = _run_dir(args.output_root, shots, seed)
            if _is_complete(run_dir) and not args.force:
                print("SKIP shots={} seed={} (complete)".format(shots, seed), flush=True)
                continue
            pending.append((shots, seed, run_dir))

    active = []
    failures = []
    while pending or active:
        while pending and len(active) < args.max_parallel:
            shots, seed, run_dir = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_path = run_dir / "console.log"
            log_file = log_path.open("w", encoding="utf-8")
            command = [
                args.python,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--data-root", str(args.data_root),
                "--run-dir", str(run_dir),
                "--shots", str(shots),
                "--seed", str(seed),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--epochs", str(args.epochs),
                "--learning-rate", str(args.learning_rate),
                "--weight-decay", str(args.weight_decay),
            ]
            if args.force:
                command.append("--force")
            env = os.environ.copy()
            env["PYTHONPATH"] = str(DASSL)
            env["HF_HUB_OFFLINE"] = "1"
            env["TRANSFORMERS_OFFLINE"] = "1"
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=env,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((process, shots, seed, run_dir, log_file))
            print("START shots={} seed={} pid={}".format(shots, seed, process.pid), flush=True)

        time.sleep(2)
        still_active = []
        for process, shots, seed, run_dir, log_file in active:
            return_code = process.poll()
            if return_code is None:
                still_active.append((process, shots, seed, run_dir, log_file))
                continue
            log_file.close()
            if return_code == 0 and _is_complete(run_dir):
                print("COMPLETE shots={} seed={}".format(shots, seed), flush=True)
            else:
                tail = (run_dir / "console.log").read_text(
                    encoding="utf-8", errors="replace"
                )[-5000:]
                failures.append((shots, seed, return_code, tail))
                print("FAILED shots={} seed={} exit={}".format(shots, seed, return_code), flush=True)
        active = still_active

    if failures:
        details = "\n\n".join(
            "shots={} seed={} exit={}\n{}".format(*failure)
            for failure in failures
        )
        raise RuntimeError("One or more runs failed:\n{}".format(details))


def _source_dir(output_root, method, shots, seed):
    if method == "CoOp":
        return (
            output_root
            / "shots_{}".format(shots)
            / "CoOp_BiomedCLIP"
            / "nctx4_cscFalse_ctpend"
            / "seed{}".format(seed)
        )
    if method == "CoOp+VPT":
        return _run_dir(output_root, shots, seed)
    if method == "CoOp+VPT+Text":
        return (
            output_root
            / "CoOp_VPT_Deep_TextDeep_Nv4_Nt4"
            / "shots_{}".format(shots)
            / "seed{}".format(seed)
        )
    raise KeyError(method)


def aggregate(args):
    methods = ("CoOp", "CoOp+VPT", "CoOp+VPT+Text")
    summary = {
        "protocol": {
            "dataset": "DermaMNIST",
            "validation": "complete official validation split (1003 samples), natural class distribution",
            "test": "complete official test split (2005 samples)",
            "shots": list(args.shots),
            "seeds": list(args.seeds),
            "coop_tokens": 4,
            "visual_tokens": 4,
            "text_tokens": 4,
            "selection_metrics": list(SELECTION_METRICS),
            "reported_scale": "percent",
            "std": "population standard deviation across seeds",
        },
        "shots": {},
    }
    for shots in args.shots:
        shot_result = {}
        for method in methods:
            runs = []
            for seed in args.seeds:
                path = _source_dir(args.output_root, method, shots, seed) / "test_by_selection.json"
                if not path.exists():
                    raise FileNotFoundError("Missing new-protocol result: {}".format(path))
                runs.append(_json_read(path))
            selections = {}
            for selection_metric in SELECTION_METRICS:
                metrics = {}
                for report_metric in REPORT_METRICS:
                    values = [float(run[selection_metric][report_metric]) for run in runs]
                    metrics[report_metric] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values, ddof=0)),
                        "values": values,
                    }
                selections[selection_metric] = metrics
            shot_result[method] = {"n": len(runs), "selections": selections}
        summary["shots"][str(shots)] = shot_result

    json_path = args.output_root / "prompt_ablation_fullval_summary.json"
    md_path = args.output_root / "prompt_ablation_fullval_summary.md"
    _json_write(json_path, summary)

    lines = [
        "# 4-token prompt ablation — full natural validation",
        "",
        "Protocol: DermaMNIST official validation (1003 samples, natural class distribution), "
        "official test (2005 samples), 5 seeds. Values are test mean ± population std (%).",
        "",
    ]
    for selection_metric in SELECTION_METRICS:
        label = "best validation ACC" if selection_metric == "accuracy" else "best validation BACC"
        lines.extend(
            [
                "## Checkpoint selected by {}".format(label),
                "",
                "| Shots | Method | ACC | BACC | AUC | Macro-F1 |",
                "|---:|---|---:|---:|---:|---:|",
            ]
        )
        for shots in args.shots:
            for method in methods:
                metrics = summary["shots"][str(shots)][method]["selections"][selection_metric]
                cells = []
                for metric in REPORT_METRICS:
                    item = metrics[metric]
                    cells.append("{:.2f} ± {:.2f}".format(item["mean"], item["std"]))
                lines.append("| {} | {} | {} | {} | {} | {} |".format(shots, method, *cells))
        lines.append("")
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE {}".format(json_path), flush=True)
    print("WROTE {}".format(md_path), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "dermamnist_fullval_acc_bacc",
    )
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--shots", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Use 0 on Windows when several GPU runs execute concurrently.",
    )
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=0.005)
    parser.add_argument("--weight-decay", type=float, default=0.0005)
    parser.add_argument("--max-parallel", type=int, default=4)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.worker:
        if args.run_dir is None or args.seed is None or len(args.shots) != 1:
            raise ValueError("Worker mode requires --run-dir, --seed, and exactly one --shots value")
        args.shots = args.shots[0]
        run_worker(args)
        return

    args.output_root = args.output_root.resolve()
    if not args.aggregate_only:
        launch_all(args)
    aggregate(args)


if __name__ == "__main__":
    main()
