"""Test the validation-accuracy-selected checkpoint for one experiment."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import train
from dassl.engine import build_trainer
from dassl.utils import set_random_seed


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--method", choices=("coop", "deep_prompt", "class_text_token", "fusion", "semantic_distill"), required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--shots", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--insert-layer", type=int, default=7)
    args = parser.parse_args()
    run_dir = Path(args.run_dir)
    (run_dir / "evaluation").mkdir(parents=True, exist_ok=True)
    selection = json.loads((run_dir / "best_validation_accuracy.json").read_text())
    is_coop = args.method == "coop"
    opts = ["DATASET.NUM_SHOTS", str(args.shots), "TEST.SKIP_FINAL_TEST", "True",
            "TEST.SAVE_BEST_METRICS", "['accuracy']", "TRAINER.TCP.INSERT_LAYER", str(args.insert_layer)]
    if not is_coop:
        opts += ["TRAINER.TCP.ENABLED", str(args.method in ("class_text_token", "fusion", "semantic_distill")),
                 "TRAINER.TCP.FUSION_WEIGHT",
                 "0.5" if args.method == "fusion" else "1.0"]
    opts += ["TRAINER.SEMANTIC_DISTILL.ENABLED", str(args.method == "semantic_distill")]
    cfg_args = SimpleNamespace(
        root=str(ROOT / "data"), output_dir=str(run_dir / "evaluation"), resume="",
        source_domains=None, target_domains=None, transforms=None,
        trainer="CoOp_BiomedCLIP" if is_coop else "CoOpVPT_BiomedCLIP",
        backbone="", head="", seed=args.seed,
        dataset_config_file=str(ROOT / "configs/datasets" / (args.dataset.lower() + ".yaml")),
        config_file=str(ROOT / "configs/trainers/CoOp" / (
            "dermamnist_native.yaml" if is_coop else "dermamnist_native_vpt_tcp.yaml")),
        opts=opts,
    )
    cfg = train.setup_cfg(cfg_args)
    set_random_seed(args.seed)
    trainer = build_trainer(cfg)
    trainer.load_model(str(run_dir))
    trainer.test(split="test")
    result = {
        "method": args.method, "dataset": args.dataset, "shots": args.shots,
        "seed": args.seed, "selected_epoch": selection["epoch"],
        "insert_layer": args.insert_layer,
        "semantic_weight": cfg.TRAINER.SEMANTIC_DISTILL.WEIGHT,
        "semantic_temperature": cfg.TRAINER.SEMANTIC_DISTILL.TEMPERATURE,
        "validation_accuracy": selection["selection_value"],
        "test_accuracy": float(trainer.last_eval_results["accuracy"]),
        "test_metrics": {key: float(value) for key, value in trainer.last_eval_results.items()},
    }
    (run_dir / "test_metrics.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
