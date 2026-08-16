"""Two-stage validation-only search for the single-model multi-text TCP.

Every candidate uses the same 50 BiomedCoOp descriptions, optimizer, loss,
epochs, shot grid, and seeds.  No test loader is evaluated during search.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RUNNER = REPO / "scripts" / "coopvpt" / "run_tcp_fullval.py"
BASELINE_METHOD = "CoOp_VPT_Deep_Nv4"
AGGREGATIONS = (
    "feature_mean",
    "tke_mean",
    "consensus_weighted",
    "set_attention",
)
EXTRA_CONNECTIONS = (
    "late_replace",
    "all_residual",
    "original_coop_replace",
)


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def method_name(aggregation, connection):
    labels = {
        "feature_mean": "FeatureMean",
        "tke_mean": "TKEMean",
        "consensus_weighted": "Consensus",
        "set_attention": "SetAttention",
        "late_residual": "LateResidual",
        "late_norm_residual": "LateNormResidual",
        "late_centered_norm_residual": "LateCenteredNormResidual",
        "late_replace": "LateReplace",
        "all_residual": "AllResidual",
        "original_coop_replace": "CoOpReplace",
    }
    return "CoOp_VPT_Deep_MT-TCP_{}_{}_Nv4_Nt4_K50".format(
        labels[aggregation], labels[connection]
    )


def run_candidate(args, aggregation, connection):
    method = method_name(aggregation, connection)
    command = [
        args.python,
        "-u",
        str(RUNNER),
        "--data-root",
        str(args.data_root),
        "--output-root",
        str(args.output_root),
        "--method",
        method,
        "--prior-source",
        "biomedcoop_50",
        "--description-count",
        "50",
        "--aggregation",
        aggregation,
        "--connection",
        connection,
        "--shots",
        *[str(value) for value in args.shots],
        "--seeds",
        *[str(value) for value in args.seeds],
        "--epochs",
        str(args.epochs),
        "--batch-size",
        str(args.batch_size),
        "--num-workers",
        str(args.num_workers),
        "--checkpoint-freq",
        str(args.checkpoint_freq),
        "--max-parallel",
        str(args.max_parallel),
        "--kg-weight",
        str(args.kg_weight),
        "--consensus-temperature",
        str(args.consensus_temperature),
        "--gate-init",
        str(args.gate_init),
        "--validation-only",
        "--skip-aggregate",
    ]
    if args.force:
        command.append("--force")
    print("RUN CANDIDATE {}".format(method), flush=True)
    subprocess.run(command, cwd=REPO, check=True)
    return {
        "method": method,
        "aggregation": aggregation,
        "connection": connection,
    }


def _best_validation_path(root, method, shots, seed):
    return (
        root
        / method
        / "shots_{}".format(shots)
        / "seed{}".format(seed)
        / "best_validation_balanced_accuracy.json"
    )


def summarize_candidate(args, candidate):
    rows = []
    for shots in args.shots:
        for seed in args.seeds:
            candidate_record = _read_json(
                _best_validation_path(
                    args.output_root, candidate["method"], shots, seed
                )
            )["metrics"]
            baseline_record = _read_json(
                _best_validation_path(
                    args.baseline_root, args.baseline_method, shots, seed
                )
            )["metrics"]
            rows.append(
                {
                    "shots": int(shots),
                    "seed": int(seed),
                    "candidate": candidate_record,
                    "baseline": baseline_record,
                    "delta": {
                        metric: float(candidate_record[metric])
                        - float(baseline_record[metric])
                        for metric in (
                            "accuracy",
                            "balanced_accuracy",
                            "auc",
                            "macro_f1",
                        )
                    },
                }
            )

    summary = dict(candidate)
    summary["runs"] = rows
    summary["overall"] = {}
    for metric in ("accuracy", "balanced_accuracy", "auc", "macro_f1"):
        values = np.asarray([row["candidate"][metric] for row in rows])
        deltas = np.asarray([row["delta"][metric] for row in rows])
        summary["overall"][metric] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=0)),
            "delta_mean": float(deltas.mean()),
            "delta_std": float(deltas.std(ddof=0)),
        }
    summary["by_shot"] = {}
    for shots in args.shots:
        shot_rows = [row for row in rows if row["shots"] == shots]
        summary["by_shot"][str(shots)] = {
            metric: {
                "mean": float(np.mean([row["candidate"][metric] for row in shot_rows])),
                "delta_mean": float(np.mean([row["delta"][metric] for row in shot_rows])),
            }
            for metric in ("accuracy", "balanced_accuracy", "auc", "macro_f1")
        }

    overall_acc = summary["overall"]["accuracy"]["delta_mean"]
    overall_auc = summary["overall"]["auc"]["delta_mean"]
    worst_shot_acc = min(
        item["accuracy"]["delta_mean"] for item in summary["by_shot"].values()
    )
    worst_shot_auc = min(
        item["auc"]["delta_mean"] for item in summary["by_shot"].values()
    )
    constraints = {
        "overall_accuracy_drop_no_more_than_2pp": overall_acc >= -2.0,
        "overall_auc_drop_no_more_than_1pp": overall_auc >= -1.0,
        "each_shot_accuracy_drop_no_more_than_4pp": worst_shot_acc >= -4.0,
        "each_shot_auc_drop_no_more_than_2pp": worst_shot_auc >= -2.0,
    }
    summary["constraints"] = constraints
    summary["feasible"] = all(constraints.values())
    summary["constraint_violation"] = float(
        max(0.0, -2.0 - overall_acc)
        + max(0.0, -1.0 - overall_auc)
        + max(0.0, -4.0 - worst_shot_acc)
        + max(0.0, -2.0 - worst_shot_auc)
    )
    return summary


def select_candidate(summaries):
    feasible = [summary for summary in summaries if summary["feasible"]]
    pool = feasible if feasible else summaries
    return max(
        pool,
        key=lambda summary: (
            -summary["constraint_violation"],
            summary["overall"]["balanced_accuracy"]["delta_mean"],
            -summary["overall"]["balanced_accuracy"]["delta_std"],
            summary["overall"]["accuracy"]["delta_mean"],
            summary["overall"]["auc"]["delta_mean"],
        ),
    )


def write_stage_summary(args, name, summaries, selected):
    payload = {
        "stage": name,
        "test_evaluated": False,
        "unified_protocol": {
            "shots": list(args.shots),
            "seeds": list(args.seeds),
            "epochs": args.epochs,
            "description_count": 50,
            "kg_weight": args.kg_weight,
            "shared_hyperparameters_across_shots_and_seeds": True,
        },
        "candidates": summaries,
        "selected": selected,
        "selection_warning": (
            None
            if selected["feasible"]
            else "No candidate met every ACC/AUC guard; selection minimizes violation before BACC."
        ),
    }
    _write_json(args.output_root / "{}_summary.json".format(name), payload)
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=Path, default=Path(r"D:\Data\dermamnist"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=REPO / "output" / "tcp_multitext_search_50ep",
    )
    parser.add_argument(
        "--baseline-root",
        type=Path,
        default=REPO / "output" / "dermamnist_fullval_acc_bacc",
    )
    parser.add_argument("--baseline-method", default=BASELINE_METHOD)
    parser.add_argument("--python", default=r"D:\Anaconda\python.exe")
    parser.add_argument("--shots", nargs="+", type=int, default=(4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--checkpoint-freq", type=int, default=10)
    parser.add_argument("--max-parallel", type=int, default=2)
    parser.add_argument("--kg-weight", type=float, default=8.0)
    parser.add_argument("--consensus-temperature", type=float, default=0.07)
    parser.add_argument("--gate-init", type=float, default=0.1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    args = parser.parse_args()
    args.output_root = args.output_root.resolve()
    args.baseline_root = args.baseline_root.resolve()

    stage_a = [
        {
            "method": method_name(aggregation, "late_residual"),
            "aggregation": aggregation,
            "connection": "late_residual",
        }
        for aggregation in AGGREGATIONS
    ]
    if not args.summarize_only:
        stage_a = [
            run_candidate(args, item["aggregation"], item["connection"])
            for item in stage_a
        ]
    stage_a_summaries = [summarize_candidate(args, item) for item in stage_a]
    stage_a_selected = select_candidate(stage_a_summaries)
    write_stage_summary(args, "stage_a_aggregation", stage_a_summaries, stage_a_selected)

    winning_aggregation = stage_a_selected["aggregation"]
    stage_b = [
        {
            "method": method_name(winning_aggregation, connection),
            "aggregation": winning_aggregation,
            "connection": connection,
        }
        for connection in EXTRA_CONNECTIONS
    ]
    if not args.summarize_only:
        stage_b = [
            run_candidate(args, item["aggregation"], item["connection"])
            for item in stage_b
        ]
    stage_b_summaries = [summarize_candidate(args, item) for item in stage_b]
    write_stage_summary(
        args,
        "stage_b_connection",
        stage_b_summaries,
        select_candidate(stage_b_summaries),
    )

    all_summaries = stage_a_summaries + stage_b_summaries
    final_selected = select_candidate(all_summaries)
    final_payload = write_stage_summary(
        args, "multitext_tcp_search", all_summaries, final_selected
    )
    _write_json(
        args.output_root / "selected_method.json",
        {
            "selection_frozen": True,
            "test_evaluated": False,
            "method": final_selected["method"],
            "aggregation": final_selected["aggregation"],
            "connection": final_selected["connection"],
            "feasible": final_selected["feasible"],
            "validation_summary": final_selected["overall"],
            "protocol": final_payload["unified_protocol"],
        },
    )
    print("SELECTED {}".format(final_selected["method"]), flush=True)


if __name__ == "__main__":
    main()
