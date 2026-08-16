"""Run and compare the fixed CoOp+VPT-Deep+TCP experiment.

The model is trained once per seed. Both best-validation-ACC and
best-validation-BACC prompt bundles are then loaded and evaluated on test.
"""

import argparse
import hashlib
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
for path in (REPO, DASSL):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

METHOD = "CoOp_VPT_Deep_TCP_Nv4_Ntke4"
BASELINE_METHOD = "CoOp_VPT_Deep_Nv4"
SELECTION_METRICS = ("accuracy", "balanced_accuracy")
CORE_METRICS = ("accuracy", "balanced_accuracy", "auc", "macro_f1")


INTERNAL_TEXT_PROMPT_CONNECTIONS = {
    "late_residual",
    "late_norm_residual",
    "late_centered_norm_residual",
    "late_centered_classlayer_norm_residual",
    "late_replace",
    "all_residual",
    "original_coop_replace",
}


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run_dir(output_root, shots, seed, method=METHOD):
    return output_root / method / "shots_{}".format(shots) / "seed{}".format(seed)


def _baseline_dir(output_root, shots, seed, method=BASELINE_METHOD):
    return output_root / method / "shots_{}".format(shots) / "seed{}".format(seed)


def _baseline_checkpoint(run_dir, selection):
    """Resolve both current dual-checkpoint and legacy selection-copy layouts."""

    candidates = (
        run_dir
        / "prompt_parameters"
        / "model-best-{}.pth.tar".format(selection),
        run_dir
        / "selection_{}".format(selection)
        / "prompt_parameters"
        / "model-best.pth.tar",
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "Missing baseline {} checkpoint; checked {}".format(
            selection, ", ".join(str(path) for path in candidates)
        )
    )


def _complete(run_dir, validation_only=False, required_epochs=None):
    if validation_only:
        completion_path = run_dir / "validation_search_complete.json"
        paths = [
            completion_path,
            run_dir / "best_validation_accuracy.json",
            run_dir / "best_validation_balanced_accuracy.json",
        ]
        paths.extend(
            run_dir / "prompt_parameters" / "model-best-{}.pth.tar".format(metric)
            for metric in SELECTION_METRICS
        )
    else:
        completion_path = run_dir / "run_complete.json"
        paths = [
            completion_path,
            run_dir / "results.json",
            run_dir / "test_by_selection.json",
            run_dir / "best_validation_accuracy.json",
            run_dir / "best_validation_balanced_accuracy.json",
        ]
        paths.extend(
            run_dir / "prompt_parameters" / "model-best-{}.pth.tar".format(metric)
            for metric in SELECTION_METRICS
        )

    if not all(path.exists() for path in paths):
        return False
    if required_epochs is None:
        return True
    try:
        completion = _read_json(completion_path)
    except (OSError, ValueError, TypeError):
        return False
    return int(completion.get("epochs", -1)) == int(required_epochs)


def _require_validation_gate(args):
    """Refuse test evaluation until the frozen full-grid validation gate passes."""

    if args.worker:
        if args.run_dir is None:
            raise ValueError("Worker gate validation requires --run-dir")
        output_root = args.run_dir.resolve().parents[2]
    else:
        output_root = args.output_root.resolve()
    gate_path = output_root / "{}_validation_summary.json".format(
        args.summary_prefix
    )
    if not gate_path.exists():
        raise RuntimeError(
            "Test evaluation requires a validation-gate summary: {}".format(
                gate_path
            )
        )
    summary = _read_json(gate_path)
    protocol = summary.get("protocol", {})
    verdict = summary.get("overall", {}).get("effectiveness", {})
    if protocol.get("method") != args.method:
        raise RuntimeError(
            "Validation-gate method mismatch: expected {!r}, got {!r}".format(
                args.method, protocol.get("method")
            )
        )
    if protocol.get("test_evaluated") is not False:
        raise RuntimeError("Validation gate must be produced without test evaluation")
    if not args.worker:
        expected_grid = (list(args.shots), list(args.seeds))
        actual_grid = (protocol.get("shots"), protocol.get("seeds"))
        if actual_grid != expected_grid:
            raise RuntimeError(
                "Validation-gate grid mismatch: expected {}, got {}".format(
                    expected_grid, actual_grid
                )
            )
    if verdict.get("effective") is not True:
        raise RuntimeError(
            "Frozen global validation gate did not pass; test remains untouched: {}".format(
                gate_path
            )
        )
    return gate_path


def _train_args(args):
    resume = ""
    if (args.run_dir / "prompt_parameters" / "checkpoint").exists():
        resume = str(args.run_dir)
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
            / (
                "dermamnist_native_vpt_multitext_tcp.yaml"
                if args.prior_source == "biomedcoop_50"
                else "dermamnist_native_vpt_tcp.yaml"
            )
        ),
        eval_only=False,
        model_dir="",
        load_epoch=None,
        no_train=False,
        opts=[
            "DATASET.NUM_SHOTS", str(args.shots),
            "DATALOADER.TRAIN_X.BATCH_SIZE", str(args.batch_size),
            "DATALOADER.NUM_WORKERS", str(args.num_workers),
            "DATALOADER.PERSISTENT_WORKERS", "True" if args.num_workers else "False",
            "OPTIM.MAX_EPOCH", str(args.epochs),
            "TRAINER.COOPVPT.OPTIM.MAX_EPOCH", str(args.epochs),
            "OPTIM.LR", "0.005",
            "TRAINER.COOPVPT.OPTIM.LR", "0.005",
            "TRAIN.CHECKPOINT_FREQ", str(args.checkpoint_freq),
            "TRAINER.TCP.FUSION_MODE", args.fusion_mode,
            "TRAINER.TCP.FUSION_WEIGHT", str(args.fusion_weight),
            "TRAINER.TCP.KG_WEIGHT", str(args.kg_weight),
            "TRAINER.TCP.KG_MODE", args.kg_mode,
            "TRAINER.TCP.PRIOR_SOURCE", args.prior_source,
            "TRAINER.TCP.DESCRIPTION_COUNT", str(args.description_count),
            "TRAINER.TCP.DESCRIPTION_BATCH_SIZE", str(args.description_batch_size),
            "TRAINER.TCP.DESCRIPTION_CACHE", str(args.description_cache),
            "TRAINER.TCP.LAYER_DESCRIPTION_CACHE",
            str(args.layer_description_cache),
            "TRAINER.TCP.PRIOR_REPRESENTATION", args.prior_representation,
            "TRAINER.TCP.AGGREGATION", args.aggregation,
            "TRAINER.TCP.CONNECTION", args.connection,
            "TRAINER.TCP.CONSENSUS_TEMPERATURE", str(args.consensus_temperature),
            "TRAINER.TCP.GATE_INIT", str(args.gate_init),
            "TRAINER.TCP.RESIDUAL_WARMUP_EPOCHS",
            str(args.residual_warmup_epochs),
            "TRAINER.TCP.PROMPT_ANCHOR_WEIGHT",
            str(args.prompt_anchor_weight),
            "TRAINER.TCP.PROMPT_ANCHOR_L2_WEIGHT",
            str(args.prompt_anchor_l2_weight),
            "TRAINER.TCP.EVAL_WARMSTART",
            "True" if args.eval_warmstart else "False",
            "TRAINER.TCP.DESCRIPTION_KD_WEIGHT",
            str(args.description_kd_weight),
            "TRAINER.TCP.DESCRIPTION_KD_TEMPERATURE",
            str(args.description_kd_temperature),
            "TRAINER.TCP.DESCRIPTION_KD_TAU",
            str(args.description_kd_tau),
            "TRAINER.TCP.IMAGE_PRIOR_WEIGHT",
            str(args.image_prior_weight),
            "TRAINER.TCP.PRIOR_CONTRASTIVE_WEIGHT",
            str(args.prior_contrastive_weight),
            "TRAINER.TCP.PRIOR_CONTRASTIVE_TEMPERATURE",
            str(args.prior_contrastive_temperature),
            "TRAINER.TCP.LAYER_TOKEN_ALIGNMENT_WEIGHT",
            str(args.layer_token_alignment_weight),
            "TRAINER.TCP.CROSS_MODAL_PROTO_WEIGHT",
            str(args.cross_modal_proto_weight),
            "TRAINER.TCP.CROSS_MODAL_PROTO_TEMPERATURE",
            str(args.cross_modal_proto_temperature),
            "TRAINER.TCP.HARD_NEGATIVE_MARGIN_WEIGHT",
            str(args.hard_negative_margin_weight),
            "TRAINER.TCP.HARD_NEGATIVE_MARGIN",
            str(args.hard_negative_margin),
            "TRAINER.TCP.HARD_NEGATIVE_TEMPERATURE",
            str(args.hard_negative_temperature),
            "TRAINER.TCP.BASE_PROMPT_FREEZE_EPOCHS",
            str(args.base_prompt_freeze_epochs),
            "TRAINER.TCP.INIT_BASELINE_CHECKPOINT",
            str(args.init_baseline_checkpoint or ""),
        ],
    )


def _evaluate_existing_selections(trainer, args, run_dir, training_reused):
    """Evaluate the two saved validation selections without any training."""

    selections = {}
    test_by_selection = {}
    for metric in SELECTION_METRICS:
        validation_path = run_dir / "best_validation_{}.json".format(metric)
        checkpoint_path = (
            run_dir
            / "prompt_parameters"
            / "model-best-{}.pth.tar".format(metric)
        )
        if not validation_path.exists() or not checkpoint_path.exists():
            raise FileNotFoundError(
                "Existing evaluation requires validation record and checkpoint: "
                "{} / {}".format(validation_path, checkpoint_path)
            )
        validation = _read_json(validation_path)
        checkpoint = trainer.load_prompt_checkpoint(checkpoint_path)
        trainer.test(split="test")
        test_metrics = {
            key: float(value) for key, value in trainer.last_eval_results.items()
        }
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

        selection_dir = run_dir / "selection_{}".format(metric) / "prompt_parameters"
        selection_dir.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(checkpoint_path, selection_dir / "model-best.pth.tar")
        (selection_dir / "checkpoint").write_text(
            "model-best.pth.tar\n", encoding="utf-8"
        )
        cmat = run_dir / "cmat.pt"
        if cmat.exists():
            shutil.copyfile(
                cmat, run_dir / "test_cmat_selected_by_{}.pt".format(metric)
            )

    _write_json(run_dir / "test_by_selection.json", test_by_selection)
    _write_json(
        run_dir / "results.json",
        {
            "method": args.method,
            "shots": args.shots,
            "seed": args.seed,
            "training_reused": bool(training_reused),
            "selections": selections,
        },
    )
    _write_json(
        run_dir / "run_complete.json",
        {
            "status": "complete",
            "exit_code": 0,
            "epochs": args.epochs,
            "training_reused": bool(training_reused),
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
    if _complete(
        run_dir, args.validation_only, required_epochs=args.epochs
    ) and not args.force:
        print("SKIP shots={} seed={} (complete)".format(args.shots, args.seed), flush=True)
        return

    cfg = train.setup_cfg(_train_args(args))
    set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)
    if torch.cuda.is_available() and cfg.USE_CUDA:
        torch.backends.cudnn.benchmark = True

    print(
        "RUN {} shots={} seed={} epochs={}".format(
            args.method, args.shots, args.seed, args.epochs
        ),
        flush=True,
    )
    trainer = build_trainer(cfg)
    model = trainer._unwrapped_model()
    if args.evaluate_existing:
        if args.validation_only:
            raise ValueError("--evaluate-existing and --validation-only are exclusive")
        if not _complete(
            run_dir, validation_only=True, required_epochs=args.epochs
        ):
            raise RuntimeError(
                "Cannot evaluate existing checkpoints before validation training is complete: "
                "{}".format(run_dir)
            )
        existing_manifest = _read_json(run_dir / "run_manifest.json")
        expected = {
            "method": args.method,
            "shots": args.shots,
            "seed": args.seed,
            "epochs": args.epochs,
        }
        mismatches = {
            key: (existing_manifest.get(key), value)
            for key, value in expected.items()
            if existing_manifest.get(key) != value
        }
        if mismatches:
            raise RuntimeError(
                "Evaluation arguments do not match the trained run manifest: {}".format(
                    mismatches
                )
            )
        trainer._writer = None
        _evaluate_existing_selections(
            trainer, args, run_dir, training_reused=True
        )
        print(
            "DONE existing-checkpoint evaluation shots={} seed={}".format(
                args.shots, args.seed
            ),
            flush=True,
        )
        return

    manifest = {
        "method": args.method,
        "dataset": "DermaMNIST",
        "shots": args.shots,
        "seed": args.seed,
        "epochs": args.epochs,
        "split_sizes": {
            "train": len(trainer.dm.dataset.train_x),
            "validation": len(trainer.dm.dataset.val),
            "test": len(trainer.dm.dataset.test),
        },
        "validation_split": "complete official split with natural class distribution",
        "optimizer": "AdamW",
        "shared_lr": 0.005,
        "weight_decay": 0.0005,
        "coop_tokens": 4,
        "visual_vpt": {"mode": "deep", "tokens": 4},
        "ctx_init": "a photo of a",
        "selection_metrics": list(SELECTION_METRICS),
        "loss": "CE + {} * KG[{}] + {} * prompt_anchor(cosine + {} * relative_L2) + {} * description_KD + {} * image_prior_CE + {} * prior_contrastive(T={}) + {} * layer_token_alignment + {} * cross_modal_proto(T={}) + {} * hard_negative_margin(m={}, T={})".format(
            args.kg_weight,
            args.kg_mode,
            args.prompt_anchor_weight,
            args.prompt_anchor_l2_weight,
            args.description_kd_weight,
            args.image_prior_weight,
            args.prior_contrastive_weight,
            args.prior_contrastive_temperature,
            args.layer_token_alignment_weight,
            args.cross_modal_proto_weight,
            args.cross_modal_proto_temperature,
            args.hard_negative_margin_weight,
            args.hard_negative_margin,
            args.hard_negative_temperature,
        ),
        "tcp_knowledge_loss_mode": args.kg_mode,
        "tcp_prior_representation": args.prior_representation,
        "single_model_only": True,
        "shot_specific_hyperparameters": False,
        "baseline_prompt_initialization": (
            {
                "checkpoint": str(args.init_baseline_checkpoint),
                "selection": args.init_baseline_selection,
                "sha256": _sha256_file(args.init_baseline_checkpoint),
                "copied_parameters": (
                    [
                        "prompt_learner.ctx",
                        "visual_prompt.prompt_embeddings",
                        "text_prompt.prompt_embeddings -> tcp.text_prompt.prompt_embeddings",
                    ]
                    if args.connection in INTERNAL_TEXT_PROMPT_CONNECTIONS
                    else [
                        "prompt_learner.ctx",
                        "visual_prompt.prompt_embeddings",
                    ]
                ),
                "optimizer_state_loaded": False,
            }
            if args.init_baseline_checkpoint
            else None
        ),
        "tcp_residual_warmup_epochs": args.residual_warmup_epochs,
        "tcp_prompt_anchor_weight": args.prompt_anchor_weight,
        "tcp_prompt_anchor_l2_weight": args.prompt_anchor_l2_weight,
        "tcp_eval_warmstart_epoch0": bool(args.eval_warmstart),
        "base_prompt_freeze_epochs": args.base_prompt_freeze_epochs,
        "description_kd": {
            "weight": args.description_kd_weight,
            "temperature": args.description_kd_temperature,
            "robust_selection_tau": args.description_kd_tau,
            "training_only": True,
            "inference_logit_fusion": False,
        },
        "image_description_prior": {
            "weight": args.image_prior_weight,
            "training_only": True,
            "updates": "same Visual VPT branch",
            "inference_logit_fusion": False,
        },
        "prior_contrastive": {
            "weight": args.prior_contrastive_weight,
            "temperature": args.prior_contrastive_temperature,
            "frozen_targets": "50-description class-mean priors",
            "inference_logit_fusion": False,
        },
        "layer_token_alignment": {
            "weight": args.layer_token_alignment_weight,
            "target": "four layer-8 facets from five ordered groups of ten descriptions",
            "uses_all_50_descriptions": True,
            "inference_logit_fusion": False,
        },
        "cross_modal_prototype": {
            "weight": args.cross_modal_proto_weight,
            "temperature": args.cross_modal_proto_temperature,
            "class_balanced_batch_centroids": True,
            "training_only": True,
            "inference_logit_fusion": False,
        },
        "hard_negative_margin": {
            "weight": args.hard_negative_margin_weight,
            "cosine_margin": args.hard_negative_margin,
            "temperature": args.hard_negative_temperature,
            "class_balanced": True,
            "negative": "single most-confusable wrong text prototype",
            "training_only": True,
            "inference_logit_fusion": False,
        },
        "tcp": model.text_encoder.metadata(),
    }
    _write_json(run_dir / "run_manifest.json", manifest)

    trainer.train()
    trainer._writer = None

    if args.validation_only:
        _write_json(
            run_dir / "validation_search_complete.json",
            {
                "status": "complete",
                "epochs": args.epochs,
                "test_evaluated": False,
            },
        )
        print(
            "DONE validation-only shots={} seed={}".format(args.shots, args.seed),
            flush=True,
        )
        return

    _evaluate_existing_selections(
        trainer, args, run_dir, training_reused=False
    )
    print("DONE shots={} seed={}".format(args.shots, args.seed), flush=True)


def launch(args):
    pending = [
        (shots, seed, _run_dir(args.output_root, shots, seed, args.method))
        for shots in args.shots
        for seed in args.seeds
        if args.force
        or not _complete(
            _run_dir(args.output_root, shots, seed, args.method),
            args.validation_only,
            required_epochs=args.epochs,
        )
    ]
    active = []
    failures = []
    while pending or active:
        while pending and len(active) < args.max_parallel:
            shots, seed, run_dir = pending.pop(0)
            run_dir.mkdir(parents=True, exist_ok=True)
            log_file = (run_dir / "console.log").open("w", encoding="utf-8")
            command = [
                args.python,
                "-u",
                str(Path(__file__).resolve()),
                "--worker",
                "--data-root", str(args.data_root),
                "--run-dir", str(run_dir),
                "--shots", str(shots),
                "--seed", str(seed),
                "--epochs", str(args.epochs),
                "--batch-size", str(args.batch_size),
                "--num-workers", str(args.num_workers),
                "--checkpoint-freq", str(args.checkpoint_freq),
                "--method", args.method,
                "--baseline-method", args.baseline_method,
                "--init-baseline-selection", args.init_baseline_selection,
                "--fusion-mode", args.fusion_mode,
                "--fusion-weight", str(args.fusion_weight),
                "--kg-weight", str(args.kg_weight),
                "--kg-mode", args.kg_mode,
                "--prior-source", args.prior_source,
                "--description-count", str(args.description_count),
                "--description-batch-size", str(args.description_batch_size),
                "--description-cache", str(args.description_cache),
                "--layer-description-cache", str(args.layer_description_cache),
                "--prior-representation", args.prior_representation,
                "--aggregation", args.aggregation,
                "--connection", args.connection,
                "--consensus-temperature", str(args.consensus_temperature),
                "--gate-init", str(args.gate_init),
                "--residual-warmup-epochs", str(args.residual_warmup_epochs),
                "--prompt-anchor-weight", str(args.prompt_anchor_weight),
                "--prompt-anchor-l2-weight", str(args.prompt_anchor_l2_weight),
                "--description-kd-weight", str(args.description_kd_weight),
                "--description-kd-temperature", str(args.description_kd_temperature),
                "--description-kd-tau", str(args.description_kd_tau),
                "--image-prior-weight", str(args.image_prior_weight),
                "--prior-contrastive-weight", str(args.prior_contrastive_weight),
                "--prior-contrastive-temperature",
                str(args.prior_contrastive_temperature),
                "--layer-token-alignment-weight",
                str(args.layer_token_alignment_weight),
                "--cross-modal-proto-weight",
                str(args.cross_modal_proto_weight),
                "--cross-modal-proto-temperature",
                str(args.cross_modal_proto_temperature),
                "--hard-negative-margin-weight",
                str(args.hard_negative_margin_weight),
                "--hard-negative-margin",
                str(args.hard_negative_margin),
                "--hard-negative-temperature",
                str(args.hard_negative_temperature),
                "--base-prompt-freeze-epochs",
                str(args.base_prompt_freeze_epochs),
                "--summary-prefix", args.summary_prefix,
            ]
            if args.init_baseline_root:
                baseline_checkpoint = _baseline_checkpoint(
                    _baseline_dir(
                        args.init_baseline_root,
                        shots,
                        seed,
                        args.baseline_method,
                    ),
                    args.init_baseline_selection,
                )
                command.extend(
                    ["--init-baseline-checkpoint", str(baseline_checkpoint)]
                )
            if args.validation_only:
                command.append("--validation-only")
            if args.eval_warmstart:
                command.append("--eval-warmstart")
            if args.evaluate_existing:
                command.append("--evaluate-existing")
            if args.force:
                command.append("--force")
            environment = os.environ.copy()
            environment["PYTHONPATH"] = str(DASSL)
            environment["HF_HUB_OFFLINE"] = "1"
            environment["TRANSFORMERS_OFFLINE"] = "1"
            process = subprocess.Popen(
                command,
                cwd=REPO,
                env=environment,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                text=True,
            )
            active.append((process, shots, seed, run_dir, log_file))
            print(
                "START shots={} seed={} pid={}".format(shots, seed, process.pid),
                flush=True,
            )

        time.sleep(2)
        remaining = []
        for process, shots, seed, run_dir, log_file in active:
            code = process.poll()
            if code is None:
                remaining.append((process, shots, seed, run_dir, log_file))
                continue
            log_file.close()
            if code == 0 and _complete(
                run_dir,
                args.validation_only,
                required_epochs=args.epochs,
            ):
                print("COMPLETE shots={} seed={}".format(shots, seed), flush=True)
            else:
                tail = (run_dir / "console.log").read_text(
                    encoding="utf-8", errors="replace"
                )[-6000:]
                failures.append((shots, seed, code, tail))
                print(
                    "FAILED shots={} seed={} exit={}".format(shots, seed, code),
                    flush=True,
                )
        active = remaining

    if failures:
        raise RuntimeError(
            "One or more TCP runs failed:\n{}".format(
                "\n\n".join(
                    "shots={} seed={} exit={}\n{}".format(*failure)
                    for failure in failures
                )
            )
        )


def _all_report_metrics(run):
    return tuple(
        key
        for key in run["balanced_accuracy"]
        if key in CORE_METRICS or key.startswith("recall_class_")
    )


def _aggregate_runs(tcp_runs, baseline_runs, seeds, split_name="test"):
    result = {"selections": {}}
    for selection in SELECTION_METRICS:
        report_metrics = _all_report_metrics(tcp_runs[0])
        selection_result = {"per_seed": [], "metrics": {}}
        for index, seed in enumerate(seeds):
            tcp = tcp_runs[index][selection]
            baseline = baseline_runs[index][selection]
            selection_result["per_seed"].append(
                {
                    "seed": seed,
                    "tcp": {key: float(tcp[key]) for key in report_metrics},
                    "baseline": {key: float(baseline[key]) for key in report_metrics},
                    "delta": {
                        key: float(tcp[key]) - float(baseline[key])
                        for key in report_metrics
                    },
                }
            )
        for metric in report_metrics:
            tcp_values = np.asarray(
                [float(run[selection][metric]) for run in tcp_runs]
            )
            base_values = np.asarray(
                [float(run[selection][metric]) for run in baseline_runs]
            )
            deltas = tcp_values - base_values
            selection_result["metrics"][metric] = {
                "tcp_mean": float(tcp_values.mean()),
                "tcp_std": float(tcp_values.std(ddof=0)),
                "baseline_mean": float(base_values.mean()),
                "baseline_std": float(base_values.std(ddof=0)),
                "delta_mean": float(deltas.mean()),
                "delta_std": float(deltas.std(ddof=0)),
                "delta_by_seed": deltas.tolist(),
            }
        result["selections"][selection] = selection_result

    result["effectiveness_by_selection"] = {}
    for selection in SELECTION_METRICS:
        selected = result["selections"][selection]["metrics"]
        seed_wins = sum(
            delta > 0
            for delta in selected["balanced_accuracy"]["delta_by_seed"]
        )
        required_wins = int(np.ceil(2.0 * len(seeds) / 3.0))
        bacc_delta = selected["balanced_accuracy"]["delta_mean"]
        acc_delta = selected["accuracy"]["delta_mean"]
        auc_delta = selected["auc"]["delta_mean"]
        criteria = {
            "mean_{}_bacc_gain_at_least_1pp".format(split_name): bacc_delta >= 1.0,
            "bacc_wins_at_least_two_thirds": seed_wins >= required_wins,
            "mean_{}_acc_drop_no_more_than_2pp".format(split_name): acc_delta >= -2.0,
            "mean_{}_auc_drop_no_more_than_1pp".format(split_name): auc_delta >= -1.0,
        }
        result["effectiveness_by_selection"][selection] = {
            "effective": all(criteria.values()),
            "criteria": criteria,
            "bacc_seed_wins": int(seed_wins),
            "evaluated_runs": len(seeds),
            "required_seed_wins": required_wins,
            "primary_deltas_pp": {
                "accuracy": acc_delta,
                "balanced_accuracy": bacc_delta,
                "auc": auc_delta,
            },
        }

    # Backward-compatible alias used by the frozen pre-test validation gate.
    # The gate remains tied to best-validation-BACC so adding the reporting
    # audit above cannot retroactively change whether test evaluation was allowed.
    result["effectiveness"] = result["effectiveness_by_selection"][
        "balanced_accuracy"
    ]
    return result


def _read_validation_selections(run_dir):
    return {
        selection: _read_json(
            run_dir / "best_validation_{}.json".format(selection)
        )["metrics"]
        for selection in SELECTION_METRICS
    }


def aggregate_validation(args):
    """Compare both validation-selected checkpoints without touching test."""

    baseline_root = (
        args.comparison_baseline_root
        or args.init_baseline_root
        or args.output_root
    ).resolve()
    summary = {
        "protocol": {
            "method": args.method,
            "baseline": args.baseline_method,
            "shots": list(args.shots),
            "seeds": list(args.seeds),
            "reported_split": "complete natural-distribution validation",
            "test_evaluated": False,
            "selection_metrics": list(SELECTION_METRICS),
            "reported_scale": "percent",
        },
        "shots": {},
    }
    all_tcp_runs = []
    all_baseline_runs = []
    all_run_ids = []
    for shots in args.shots:
        tcp_runs = []
        baseline_runs = []
        for seed in args.seeds:
            tcp_runs.append(
                _read_validation_selections(
                    _run_dir(args.output_root, shots, seed, args.method)
                )
            )
            baseline_runs.append(
                _read_validation_selections(
                    _baseline_dir(
                        baseline_root, shots, seed, args.baseline_method
                    )
                )
            )
        summary["shots"][str(shots)] = _aggregate_runs(
            tcp_runs, baseline_runs, args.seeds, split_name="validation"
        )
        all_tcp_runs.extend(tcp_runs)
        all_baseline_runs.extend(baseline_runs)
        all_run_ids.extend(
            "shots{}_seed{}".format(shots, seed) for seed in args.seeds
        )
    summary["overall"] = _aggregate_runs(
        all_tcp_runs,
        all_baseline_runs,
        all_run_ids,
        split_name="validation",
    )

    json_path = args.output_root / "{}_validation_summary.json".format(
        args.summary_prefix
    )
    md_path = args.output_root / "{}_validation_summary.md".format(
        args.summary_prefix
    )
    _write_json(json_path, summary)
    lines = [
        "# {} — validation-only paired result".format(args.method),
        "",
        "No test split was evaluated. Values are means across paired seeds (%).",
        "",
    ]
    for selection in SELECTION_METRICS:
        label = "ACC" if selection == "accuracy" else "BACC"
        lines.extend(
            [
                "## Selected by best validation {}".format(label),
                "",
                "| Shots | ΔACC | ΔBACC | ΔAUC | BACC wins |",
                "|---:|---:|---:|---:|---:|",
            ]
        )
        for shots in args.shots:
            item = summary["shots"][str(shots)]["selections"][selection]
            metrics = item["metrics"]
            wins = sum(
                value > 0
                for value in metrics["balanced_accuracy"]["delta_by_seed"]
            )
            lines.append(
                "| {} | {:+.2f} | {:+.2f} | {:+.2f} | {}/{} |".format(
                    shots,
                    metrics["accuracy"]["delta_mean"],
                    metrics["balanced_accuracy"]["delta_mean"],
                    metrics["auc"]["delta_mean"],
                    wins,
                    len(args.seeds),
                )
            )
        overall = summary["overall"]["selections"][selection]
        metrics = overall["metrics"]
        wins = sum(
            value > 0
            for value in metrics["balanced_accuracy"]["delta_by_seed"]
        )
        lines.append(
            "| Overall | {:+.2f} | {:+.2f} | {:+.2f} | {}/{} |".format(
                metrics["accuracy"]["delta_mean"],
                metrics["balanced_accuracy"]["delta_mean"],
                metrics["auc"]["delta_mean"],
                wins,
                len(all_run_ids),
            )
        )
        lines.append("")
    verdict = summary["overall"]["effectiveness"]
    delta = verdict["primary_deltas_pp"]
    lines.extend(
        [
            "## Frozen global validation gate",
            "",
            "Primary selection: best validation BACC checkpoint.",
            "",
            "| Verdict | BACC wins | Required | ΔACC | ΔBACC | ΔAUC |",
            "|---|---:|---:|---:|---:|---:|",
            "| {} | {}/{} | {} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
                "PASS" if verdict["effective"] else "FAIL",
                verdict["bacc_seed_wins"],
                verdict["evaluated_runs"],
                verdict["required_seed_wins"],
                delta["accuracy"],
                delta["balanced_accuracy"],
                delta["auc"],
            ),
            "",
            (
                "Gate passed: the existing dual checkpoints may now be evaluated on test."
                if verdict["effective"]
                else "Test remains untouched because this gate failed."
            ),
            "",
        ]
    )
    md_path.write_text("\n".join(lines), encoding="utf-8")
    print("WROTE {}".format(json_path), flush=True)
    print("WROTE {}".format(md_path), flush=True)


def aggregate(args):
    baseline_root = (
        args.comparison_baseline_root
        or args.init_baseline_root
        or args.output_root
    ).resolve()
    summary = {
        "protocol": {
            "method": args.method,
            "baseline": args.baseline_method,
            "shots": list(args.shots),
            "seeds": list(args.seeds),
            "compared_checkpoints": [
                "best validation accuracy",
                "best validation balanced_accuracy",
            ],
            "effectiveness_evaluated_separately_by_selection": True,
            "reported_scale": "percent",
            "std": "population standard deviation across seeds",
        },
        "shots": {},
    }
    all_tcp_runs = []
    all_baseline_runs = []
    all_run_ids = []
    for shots in args.shots:
        tcp_runs = []
        baseline_runs = []
        for seed in args.seeds:
            tcp_path = (
                _run_dir(args.output_root, shots, seed, args.method)
                / "test_by_selection.json"
            )
            baseline_path = (
                _baseline_dir(
                    baseline_root, shots, seed, args.baseline_method
                )
                / "test_by_selection.json"
            )
            if not tcp_path.exists():
                raise FileNotFoundError("Missing TCP result: {}".format(tcp_path))
            if not baseline_path.exists():
                raise FileNotFoundError(
                    "Missing baseline result: {}".format(baseline_path)
                )
            tcp_runs.append(_read_json(tcp_path))
            baseline_runs.append(_read_json(baseline_path))
        summary["shots"][str(shots)] = _aggregate_runs(
            tcp_runs, baseline_runs, args.seeds
        )
        all_tcp_runs.extend(tcp_runs)
        all_baseline_runs.extend(baseline_runs)
        all_run_ids.extend(
            "shots{}_seed{}".format(shots, seed) for seed in args.seeds
        )
    summary["overall"] = _aggregate_runs(
        all_tcp_runs, all_baseline_runs, all_run_ids
    )

    json_path = args.output_root / "{}_summary.json".format(args.summary_prefix)
    md_path = args.output_root / "{}_summary.md".format(args.summary_prefix)
    _write_json(json_path, summary)
    lines = [
        "# {} — paired result".format(args.method),
        "",
        (
            "Values are test mean ± population std (%); delta is TCP minus "
            "the same-shot, same-seed `{}` baseline."
        ).format(args.baseline_method),
        "",
    ]
    for selection in SELECTION_METRICS:
        label = "ACC" if selection == "accuracy" else "BACC"
        lines.extend(
            [
                "## Selected by best validation {}".format(label),
                "",
                "| Shots | Metric | Baseline | TCP | Mean delta | Seed deltas |",
                "|---:|---|---:|---:|---:|---|",
            ]
        )
        for shots in args.shots:
            metrics = summary["shots"][str(shots)]["selections"][selection]["metrics"]
            for metric in CORE_METRICS:
                item = metrics[metric]
                lines.append(
                    "| {} | {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:+.2f} | {} |".format(
                        shots,
                        metric,
                        item["baseline_mean"], item["baseline_std"],
                        item["tcp_mean"], item["tcp_std"], item["delta_mean"],
                        ", ".join(
                            "{:+.2f}".format(value)
                            for value in item["delta_by_seed"]
                        ),
                    )
                )
        lines.append("")
        metrics = summary["overall"]["selections"][selection]["metrics"]
        lines.extend(
            [
                "### Overall across all paired shot/seed runs",
                "",
                "| Metric | Baseline | TCP | Mean delta | Run deltas |",
                "|---|---:|---:|---:|---|",
            ]
        )
        for metric in CORE_METRICS:
            item = metrics[metric]
            lines.append(
                "| {} | {:.2f} ± {:.2f} | {:.2f} ± {:.2f} | {:+.2f} | {} |".format(
                    metric,
                    item["baseline_mean"], item["baseline_std"],
                    item["tcp_mean"], item["tcp_std"], item["delta_mean"],
                    ", ".join(
                        "{:+.2f}".format(value)
                        for value in item["delta_by_seed"]
                    ),
                )
            )
        lines.append("")

    lines.extend(
        [
            "## Fixed effectiveness rule by checkpoint selection",
            "",
            "Each checkpoint is judged independently; results from the two models are never fused.",
            "",
            "| Selection | Shots | Verdict | BACC wins | ΔACC | ΔBACC | ΔAUC |",
            "|---|---:|---|---:|---:|---:|---:|",
        ]
    )
    for selection in SELECTION_METRICS:
        label = "ACC" if selection == "accuracy" else "BACC"
        for shots in args.shots:
            verdict = summary["shots"][str(shots)][
                "effectiveness_by_selection"
            ][selection]
            delta = verdict["primary_deltas_pp"]
            lines.append(
                "| {} | {} | {} | {}/{} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
                    label,
                    shots,
                    "Effective" if verdict["effective"] else "Not effective",
                    verdict["bacc_seed_wins"],
                    verdict["evaluated_runs"],
                    delta["accuracy"],
                    delta["balanced_accuracy"],
                    delta["auc"],
                )
            )
        verdict = summary["overall"]["effectiveness_by_selection"][selection]
        delta = verdict["primary_deltas_pp"]
        lines.append(
            "| {} | Overall | {} | {}/{} | {:+.2f} | {:+.2f} | {:+.2f} |".format(
                label,
                "Effective" if verdict["effective"] else "Not effective",
                verdict["bacc_seed_wins"],
                verdict["evaluated_runs"],
                delta["accuracy"],
                delta["balanced_accuracy"],
                delta["auc"],
            )
        )
    accuracy_effective = summary["overall"]["effectiveness_by_selection"][
        "accuracy"
    ]["effective"]
    bacc_effective = summary["overall"]["effectiveness_by_selection"][
        "balanced_accuracy"
    ]["effective"]
    lines.extend(
        [
            "",
            "## Selection conclusion",
            "",
            (
                "The best-validation-ACC checkpoint satisfies the fixed test rule; "
                "it is the supported inference checkpoint for this experiment."
                if accuracy_effective
                else "The best-validation-ACC checkpoint does not satisfy the fixed test rule."
            ),
            "",
            (
                "The separately retained best-validation-BACC checkpoint also satisfies the rule."
                if bacc_effective
                else (
                    "The separately retained best-validation-BACC checkpoint does not satisfy "
                    "the rule and must not be presented as an improvement."
                )
            ),
        ]
    )
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
    parser.add_argument("--method", default=METHOD)
    parser.add_argument("--baseline-method", default=BASELINE_METHOD)
    parser.add_argument("--init-baseline-root", type=Path)
    parser.add_argument(
        "--comparison-baseline-root",
        type=Path,
        help="Root containing paired baseline validation records for summaries",
    )
    parser.add_argument("--init-baseline-checkpoint", type=Path)
    parser.add_argument(
        "--init-baseline-selection",
        choices=SELECTION_METRICS,
        default="balanced_accuracy",
    )
    parser.add_argument("--fusion-mode", choices=("replace", "gated_residual"), default="replace")
    parser.add_argument("--fusion-weight", type=float, default=1.0)
    parser.add_argument("--kg-weight", type=float, default=8.0)
    parser.add_argument(
        "--kg-mode",
        choices=("raw_cosine", "centered_cosine"),
        default="raw_cosine",
    )
    parser.add_argument(
        "--prior-source",
        choices=("single_template", "biomedcoop_50"),
        default="single_template",
    )
    parser.add_argument("--description-count", type=int, default=50)
    parser.add_argument("--description-batch-size", type=int, default=64)
    parser.add_argument(
        "--description-cache",
        type=Path,
        default=REPO / "output" / "_tcp_prior_cache" / "dermamnist_biomedcoop50.pt",
    )
    parser.add_argument(
        "--layer-description-cache",
        type=Path,
        default=(
            REPO
            / "output"
            / "_tcp_prior_cache"
            / "dermamnist_biomedcoop50_layer8_cls.pt"
        ),
    )
    parser.add_argument(
        "--prior-representation",
        choices=("projected_text", "layer_cls"),
        default="projected_text",
    )
    parser.add_argument(
        "--aggregation",
        choices=(
            "feature_mean",
            "tke_mean",
            "consensus_weighted",
            "set_attention",
            "cosine_set_attention",
            "grouped10_cosine_attention",
            "grouped10_layer_residual",
            "grouped10_layer_projected_hybrid",
            "grouped10_layer_projected_residual",
            "layer_cosine_set_hybrid",
            "layer_cosine_set_hybrid_light",
            "layer_cosine_set_residual",
        ),
        default="feature_mean",
    )
    parser.add_argument(
        "--connection",
        choices=(
            "late_residual",
            "late_norm_residual",
            "late_centered_norm_residual",
            "late_centered_classlayer_norm_residual",
            "inplace_once_norm_residual",
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
            "late_replace",
            "all_residual",
            "original_coop_replace",
        ),
        default="late_residual",
    )
    parser.add_argument("--consensus-temperature", type=float, default=0.07)
    parser.add_argument("--gate-init", type=float, default=0.1)
    parser.add_argument("--residual-warmup-epochs", type=int, default=0)
    parser.add_argument("--prompt-anchor-weight", type=float, default=0.0)
    parser.add_argument("--prompt-anchor-l2-weight", type=float, default=0.0)
    parser.add_argument("--eval-warmstart", action="store_true")
    parser.add_argument("--description-kd-weight", type=float, default=0.0)
    parser.add_argument("--description-kd-temperature", type=float, default=1.5)
    parser.add_argument("--description-kd-tau", type=float, default=1.5)
    parser.add_argument("--image-prior-weight", type=float, default=0.0)
    parser.add_argument("--prior-contrastive-weight", type=float, default=0.0)
    parser.add_argument(
        "--prior-contrastive-temperature", type=float, default=0.1
    )
    parser.add_argument("--layer-token-alignment-weight", type=float, default=0.0)
    parser.add_argument("--cross-modal-proto-weight", type=float, default=0.0)
    parser.add_argument("--cross-modal-proto-temperature", type=float, default=0.1)
    parser.add_argument("--hard-negative-margin-weight", type=float, default=0.0)
    parser.add_argument("--hard-negative-margin", type=float, default=0.05)
    parser.add_argument("--hard-negative-temperature", type=float, default=0.02)
    parser.add_argument("--base-prompt-freeze-epochs", type=int, default=0)
    parser.add_argument("--summary-prefix", default="tcp_fullgrid")
    parser.add_argument("--shots", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3, 4, 5))
    parser.add_argument("--seed", type=int)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--validation-only", action="store_true")
    parser.add_argument(
        "--evaluate-existing",
        action="store_true",
        help="Evaluate already-trained dual validation checkpoints; never train",
    )
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.validation_only and args.evaluate_existing:
        raise ValueError("--validation-only and --evaluate-existing are exclusive")
    if args.evaluate_existing:
        _require_validation_gate(args)
    if args.worker:
        if args.run_dir is None or args.seed is None or len(args.shots) != 1:
            raise ValueError(
                "Worker mode requires --run-dir, --seed, and exactly one --shots"
            )
        args.shots = args.shots[0]
        run_worker(args)
        return
    args.output_root = args.output_root.resolve()
    if not args.aggregate_only:
        launch(args)
    if not args.skip_aggregate:
        if args.validation_only:
            aggregate_validation(args)
        else:
            aggregate(args)


if __name__ == "__main__":
    main()
