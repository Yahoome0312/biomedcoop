"""Validation-only refinement of the single-model 50-description TCP.

The refinement changes one global knowledge-consistency weight at a time.
Every candidate uses the same weight for every shot and seed, and test data is
never evaluated by this script.
"""

import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.coopvpt.search_multitext_tcp import (
    BASELINE_METHOD,
    _write_json,
    select_candidate,
    summarize_candidate,
)


RUNNER = REPO / "scripts" / "coopvpt" / "run_tcp_fullval.py"


def _weight_label(value):
    return ("{:g}".format(float(value))).replace(".", "p")


def method_name(weight):
    return (
        "CoOp_VPT_Deep_MT-TCP_SetAttention_LateResidual_"
        "KG{}_Nv4_Nt4_K50".format(_weight_label(weight))
    )


def run_candidate(args, weight):
    method = method_name(weight)
    command = [
        args.python,
        "-u",
        str(RUNNER),
        "--data-root", str(args.data_root),
        "--output-root", str(args.output_root),
        "--method", method,
        "--prior-source", "biomedcoop_50",
        "--description-count", "50",
        "--description-cache", str(args.description_cache),
        "--aggregation", "set_attention",
        "--connection", "late_residual",
        "--shots", *[str(value) for value in args.shots],
        "--seeds", *[str(value) for value in args.seeds],
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--num-workers", str(args.num_workers),
        "--checkpoint-freq", str(args.checkpoint_freq),
        "--max-parallel", str(args.max_parallel),
        "--kg-weight", str(weight),
        "--consensus-temperature", "0.07",
        "--gate-init", "0.1",
        "--validation-only",
        "--skip-aggregate",
    ]
    if args.force:
        command.append("--force")
    subprocess.run(command, cwd=REPO, check=True)
    return {
        "method": method,
        "aggregation": "set_attention",
        "connection": "late_residual",
        "kg_weight": float(weight),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "tcp_multitext_refine_50ep",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO / "output" / "dermamnist_fullval_acc_bacc",
    )
    parser.add_argument("--baseline-method", default=BASELINE_METHOD)
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--kg-weights", nargs="+", type=float, default=(2.0, 4.0))
    parser.add_argument("--shots", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=3)
    parser.add_argument(
        "--description-cache",
        type=Path,
        default=REPO / "output" / "_tcp_prior_cache" / "dermamnist_biomedcoop50.pt",
    )
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.baseline_root = args.baseline_root.resolve()

    candidates = [
        {
            "method": method_name(weight),
            "aggregation": "set_attention",
            "connection": "late_residual",
            "kg_weight": float(weight),
        }
        for weight in args.kg_weights
    ]
    if not args.summarize_only:
        candidates = [run_candidate(args, weight) for weight in args.kg_weights]

    summaries = [summarize_candidate(args, candidate) for candidate in candidates]
    selected = select_candidate(summaries)
    bacc_improved = selected["overall"]["balanced_accuracy"]["delta_mean"] > 0.0
    selection_frozen = bool(selected["feasible"] and bacc_improved)
    payload = {
        "stage": "global_kg_weight_refinement",
        "test_evaluated": False,
        "protocol": {
            "shots": list(args.shots),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "description_count": 50,
            "aggregation": "set_attention",
            "connection": "late_residual",
            "shared_hyperparameters_across_shots_and_seeds": True,
            "single_model_only": True,
        },
        "candidates": summaries,
        "selected": selected,
        "selection_frozen": selection_frozen,
        "freeze_gate": {
            "acc_auc_constraints_met": bool(selected["feasible"]),
            "mean_validation_bacc_improved": bacc_improved,
        },
    }
    _write_json(args.output_root / "kg_refinement_summary.json", payload)
    _write_json(
        args.output_root / "selected_refinement.json",
        {
            "selection_frozen": selection_frozen,
            "test_evaluated": False,
            "method": selected["method"] if selection_frozen else None,
            "candidate": selected,
        },
    )
    print(
        "{} {}".format(
            "SELECTED" if selection_frozen else "NO_REFINEMENT_FROZEN",
            selected["method"],
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
