"""Run from-scratch MT-TCP B0 and support-only confusion-aware variants.

Training and test evaluation are intentionally separate.  A complete
validation grid and both ACC/BACC checkpoints are required before any test
image is evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from argparse import Namespace
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
DASSL = REPO / "Dassl.pytorch"
for location in (REPO, DASSL):
    if str(location) not in sys.path:
        sys.path.insert(0, str(location))

from models.confusion_aware import CONFUSION_VARIANTS


SELECTION_METRICS = ("accuracy", "balanced_accuracy")
DEFAULT_SHOTS = (4, 8, 16, 32)
DEFAULT_SEEDS = (1, 2, 3)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def method_name(variant):
    return "FromScratch_MT-TCP_CE_{}".format(variant)


def _run_dir(output_root, method, shots, seed):
    return Path(output_root) / method / "shots_{}".format(shots) / "seed{}".format(seed)


def _complete(run_dir, validation_only=False, required_epochs=None):
    run_dir = Path(run_dir)
    if validation_only:
        marker = run_dir / "validation_search_complete.json"
        required = [marker]
        for metric in SELECTION_METRICS:
            required.extend(
                [
                    run_dir / "best_validation_{}.json".format(metric),
                    run_dir
                    / "prompt_parameters"
                    / "model-best-{}.pth.tar".format(metric),
                ]
            )
    else:
        marker = run_dir / "run_complete.json"
        required = [marker, run_dir / "results.json", run_dir / "test_by_selection.json"]
    if not all(path.exists() for path in required):
        return False
    if required_epochs is None:
        return True
    try:
        return int(_read_json(marker).get("epochs", -1)) == int(required_epochs)
    except (OSError, ValueError, TypeError):
        return False


def _train_args(args):
    resume = ""
    if (args.run_dir / "prompt_parameters" / "checkpoint").exists():
        resume = str(args.run_dir)
    options = [
        "DATASET.NUM_SHOTS",
        str(args.shots),
        "DATALOADER.TRAIN_X.BATCH_SIZE",
        str(args.batch_size),
        "DATALOADER.NUM_WORKERS",
        str(args.num_workers),
        "DATALOADER.PERSISTENT_WORKERS",
        "True" if args.num_workers else "False",
        "OPTIM.MAX_EPOCH",
        str(args.epochs),
        "TRAINER.COOPVPT.OPTIM.MAX_EPOCH",
        str(args.epochs),
        "OPTIM.LR",
        "0.005",
        "TRAINER.COOPVPT.OPTIM.LR",
        "0.005",
        "TRAIN.CHECKPOINT_FREQ",
        str(args.checkpoint_freq),
        "TRAINER.TCP.DESCRIPTION_CACHE",
        str(args.description_cache),
        "TRAINER.TCP.LAYER_DESCRIPTION_CACHE",
        str(args.layer_description_cache),
        "TRAINER.CONFUSION_AWARE.VARIANT",
        str(args.variant),
        "TRAINER.CONFUSION_AWARE.BANK_ROOT",
        str(args.bank_root or ""),
        "TRAINER.CONFUSION_AWARE.PRIOR_ALPHA",
        str(args.prior_alpha),
        "TRAINER.CONFUSION_AWARE.GAMMA",
        str(args.gamma),
        "TRAINER.CONFUSION_AWARE.LAMBDA_CONF",
        str(args.lambda_conf),
    ]
    return Namespace(
        root=str(args.data_root),
        output_dir=str(args.run_dir),
        resume=resume,
        seed=args.seed,
        source_domains=None,
        target_domains=None,
        transforms=None,
        trainer="CoOpVPT_BiomedCLIP",
        backbone="",
        head="",
        dataset_config_file=str(REPO / "configs/datasets/dermamnist.yaml"),
        config_file=str(
            REPO
            / "configs/trainers/CoOp"
            / "dermamnist_native_vpt_multitext_tcp.yaml"
        ),
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=options,
    )


def _validation_grid_path(args):
    return Path(args.output_root) / args.method / "validation_grid_complete.json"


def _require_validation_grid(args):
    path = _validation_grid_path(args)
    if not path.exists():
        raise RuntimeError(
            "Test evaluation requires a frozen complete validation grid: {}".format(path)
        )
    marker = _read_json(path)
    expected = {
        "method": args.method,
        "variant": args.variant,
        "shots": list(args.shots),
        "seeds": list(args.seeds),
        "epochs": int(args.epochs),
        "test_evaluated": False,
    }
    mismatches = {
        key: (marker.get(key), value)
        for key, value in expected.items()
        if marker.get(key) != value
    }
    if mismatches:
        raise RuntimeError("Validation-grid marker mismatch: {}".format(mismatches))
    return path


def _copy_confusion_matrices(run_dir, selection):
    for source_name in ("cmat.pt", "cmat_raw.pt"):
        source = Path(run_dir) / source_name
        if source.exists():
            destination = Path(run_dir) / "test_{}_selected_by_{}.pt".format(
                source_name.removesuffix(".pt"), selection
            )
            shutil.copyfile(source, destination)


def _evaluate_existing_selections(trainer, args, run_dir, training_reused=True):
    selections = {}
    test_by_selection = {}
    for metric in SELECTION_METRICS:
        validation_path = Path(run_dir) / "best_validation_{}.json".format(metric)
        checkpoint_path = (
            Path(run_dir)
            / "prompt_parameters"
            / "model-best-{}.pth.tar".format(metric)
        )
        if not validation_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                "Both validation record and checkpoint are required: {} / {}".format(
                    validation_path, checkpoint_path
                )
            )
        validation = _read_json(validation_path)
        checkpoint = trainer.load_prompt_checkpoint(checkpoint_path)
        trainer._analysis_tag = "test_selected_by_{}".format(metric)
        trainer.test(split="test")
        test_metrics = {
            key: float(value) for key, value in trainer.last_eval_results.items()
        }
        _copy_confusion_matrices(run_dir, metric)
        test_by_selection[metric] = test_metrics
        selections[metric] = {
            "best_epoch": int(validation["epoch"]),
            "checkpoint_epoch": int(checkpoint["epoch"]),
            "selection_value": float(validation["selection_value"]),
            "validation": {
                key: float(value) for key, value in validation["metrics"].items()
            },
            "test": test_metrics,
        }
        selection_dir = Path(run_dir) / "selection_{}".format(metric) / "prompt_parameters"
        selection_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint_path, selection_dir / "model-best.pth.tar")
        (selection_dir / "checkpoint").write_text(
            "model-best.pth.tar\n", encoding="utf-8"
        )

    _write_json(Path(run_dir) / "test_by_selection.json", test_by_selection)
    _write_json(
        Path(run_dir) / "results.json",
        {
            "method": args.method,
            "variant": args.variant,
            "shots": int(args.shots),
            "seed": int(args.seed),
            "training_reused": bool(training_reused),
            "selections": selections,
        },
    )
    _write_json(
        Path(run_dir) / "run_complete.json",
        {
            "status": "complete",
            "epochs": int(args.epochs),
            "evaluated_existing_checkpoints_only": bool(training_reused),
        },
    )


def run_worker(args):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    import torch
    import train
    from dassl.engine import build_trainer
    from dassl.utils import set_random_seed, setup_logger

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    if _complete(run_dir, args.validation_only, args.epochs) and not args.force:
        print("SKIP shots={} seed={} (complete)".format(args.shots, args.seed))
        return

    cfg = train.setup_cfg(_train_args(args))
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True
    trainer = build_trainer(cfg)

    if args.evaluate_existing:
        trainer._writer = None
        _evaluate_existing_selections(trainer, args, run_dir, training_reused=True)
        return

    manifest = {
        "protocol": "from_scratch_joint_ce",
        "method": args.method,
        "variant": args.variant,
        "dataset": "DermaMNIST",
        "shots": int(args.shots),
        "seed": int(args.seed),
        "epochs": int(args.epochs),
        "split_sizes": {
            "train": len(trainer.dm.dataset.train_x),
            "validation": len(trainer.dm.dataset.val),
            "test": len(trainer.dm.dataset.test),
        },
        "validation_split": "complete official split with natural class distribution",
        "optimizer": "AdamW",
        "lr": 0.005,
        "weight_decay": 0.0005,
        "scheduler": "cosine",
        "loss": "CE" if args.variant == "b0" else "CE + lambda_conf * softplus_margin",
        "lambda_conf": float(args.lambda_conf),
        "gamma": float(args.gamma),
        "prior_alpha": float(args.prior_alpha),
        "warm_start": False,
        "selection_metrics": list(SELECTION_METRICS),
    }
    _write_json(run_dir / "run_manifest.json", manifest)
    trainer.train()
    if not _complete(run_dir, validation_only=True, required_epochs=None):
        # Marker does not exist yet; verify the actual dual artifacts first.
        for metric in SELECTION_METRICS:
            required = [
                run_dir / "best_validation_{}.json".format(metric),
                run_dir / "prompt_parameters" / "model-best-{}.pth.tar".format(metric),
            ]
            if not all(path.exists() for path in required):
                raise RuntimeError("Training did not produce dual best checkpoints")
    _write_json(
        run_dir / "validation_search_complete.json",
        {
            "status": "complete",
            "epochs": int(args.epochs),
            "test_evaluated": False,
            "variant": args.variant,
        },
    )
    if not args.validation_only:
        _evaluate_existing_selections(trainer, args, run_dir, training_reused=False)


def _worker_command(args, shots, seed):
    run_dir = _run_dir(args.output_root, args.method, shots, seed)
    command = [
        str(args.python),
        "-B",
        str(Path(__file__).resolve()),
        "--worker",
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--run-dir",
        str(run_dir),
        "--method",
        args.method,
        "--variant",
        args.variant,
        "--shots",
        str(shots),
        "--seed",
        str(seed),
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--description-cache",
        str(args.description_cache),
        "--layer-description-cache",
        str(args.layer_description_cache),
        "--prior-alpha",
        str(args.prior_alpha),
        "--gamma",
        str(args.gamma),
        "--lambda-conf",
        str(args.lambda_conf),
    ]
    if args.bank_root:
        command.extend(["--bank-root", str(args.bank_root)])
    if args.validation_only:
        command.append("--validation-only")
    if args.evaluate_existing:
        command.append("--evaluate-existing")
    if args.force:
        command.append("--force")
    return command, run_dir


def launch(args):
    pending = []
    for shots in args.shots:
        for seed in args.seeds:
            command, run_dir = _worker_command(args, shots, seed)
            if _complete(run_dir, args.validation_only, args.epochs) and not args.force:
                print("SKIP {}".format(run_dir), flush=True)
                continue
            pending.append((command, run_dir))

    def run_one(item):
        command, run_dir = item
        print("RUN {}".format(" ".join(command)), flush=True)
        subprocess.run(command, cwd=str(REPO), check=True)
        return run_dir

    if int(args.max_parallel) <= 1:
        for item in pending:
            run_one(item)
        return
    with ThreadPoolExecutor(max_workers=int(args.max_parallel)) as executor:
        futures = {executor.submit(run_one, item): item[1] for item in pending}
        for future in as_completed(futures):
            run_dir = futures[future]
            future.result()
            print("DONE {}".format(run_dir), flush=True)


def _numeric_metrics(records):
    common = set(records[0])
    for record in records[1:]:
        common &= set(record)
    return sorted(
        key
        for key in common
        if all(isinstance(record[key], (int, float)) for record in records)
    )


def aggregate(args, validation_only=False):
    split = "validation" if validation_only else "test"
    summary = {
        "protocol": {
            "method": args.method,
            "variant": args.variant,
            "shots": list(args.shots),
            "seeds": list(args.seeds),
            "epochs": int(args.epochs),
            "selection_metrics": list(SELECTION_METRICS),
            "split": split,
        },
        "shots": {},
    }
    for shots in args.shots:
        shot_result = {}
        for selection in SELECTION_METRICS:
            rows = []
            per_seed = []
            for seed in args.seeds:
                run_dir = _run_dir(args.output_root, args.method, shots, seed)
                if validation_only:
                    record = _read_json(
                        run_dir / "best_validation_{}.json".format(selection)
                    )
                    metrics = record["metrics"]
                    epoch = record["epoch"]
                else:
                    result = _read_json(run_dir / "results.json")
                    item = result["selections"][selection]
                    metrics = item["test"]
                    epoch = item["best_epoch"]
                metrics = {key: float(value) for key, value in metrics.items()}
                rows.append(metrics)
                per_seed.append({"seed": int(seed), "best_epoch": int(epoch), **metrics})
            metric_summary = {}
            for metric in _numeric_metrics(rows):
                values = np.asarray([row[metric] for row in rows], dtype=np.float64)
                metric_summary[metric] = {
                    "mean": float(np.nanmean(values)),
                    "std": float(np.nanstd(values, ddof=0)),
                    "values": [float(value) for value in values],
                }
            shot_result[selection] = {
                "per_seed": per_seed,
                "metrics": metric_summary,
            }
        summary["shots"][str(shots)] = shot_result
    method_dir = Path(args.output_root) / args.method
    _write_json(method_dir / "{}_summary.json".format(split), summary)
    if validation_only:
        _write_json(
            method_dir / "validation_grid_complete.json",
            {
                "method": args.method,
                "variant": args.variant,
                "shots": list(args.shots),
                "seeds": list(args.seeds),
                "epochs": int(args.epochs),
                "test_evaluated": False,
            },
        )
    print("WROTE {} summary for {}".format(split, args.method), flush=True)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "confusion_aware_from_scratch",
    )
    parser.add_argument("--bank-root", type=Path)
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--method")
    parser.add_argument("--variant", choices=CONFUSION_VARIANTS, default="b0")
    parser.add_argument("--description-cache", type=Path, default=REPO / "output" / "_tcp_prior_cache" / "dermamnist_biomedcoop50.pt")
    parser.add_argument("--layer-description-cache", type=Path, default=REPO / "output" / "_tcp_prior_cache" / "dermamnist_biomedcoop50_layer8_cls.pt")
    parser.add_argument("--shots", nargs="+", type=int, default=DEFAULT_SHOTS)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--prior-alpha", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.2)
    parser.add_argument("--lambda-conf", type=float, default=1.0)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument("--evaluate-existing", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validation_only and args.evaluate_existing:
        raise ValueError("--validation-only and --evaluate-existing are exclusive")
    if args.variant != "b0" and args.bank_root is None:
        raise ValueError("Non-b0 variants require --bank-root")
    args.output_root = args.output_root.resolve()
    args.method = args.method or method_name(args.variant)
    if args.worker:
        if args.run_dir is None or args.seed is None or len(args.shots) != 1:
            raise ValueError("Worker requires --run-dir, --seed and exactly one shot")
        args.shots = args.shots[0]
        run_worker(args)
        return
    if args.evaluate_existing:
        _require_validation_grid(args)
    if not args.aggregate_only:
        launch(args)
    if not args.skip_aggregate:
        aggregate(args, validation_only=args.validation_only)


if __name__ == "__main__":
    main()
