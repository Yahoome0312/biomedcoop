"""Aggregate final CoOp/VPT test logs into JSON, CSV, and Markdown."""

import argparse
import csv
import json
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.biomedcoop.aggregate_results import aggregate, parse_log


DEFAULT_METHODS = tuple("CoOp_VPT_Deep_Nv{}".format(tokens) for tokens in (1, 2, 5, 10, 20))


def parse_runtime_metadata(log_path):
    text = log_path.read_text(encoding="utf-8", errors="replace")
    trainable_matches = re.findall(r"Total trainable parameters:\s*([\d,]+)", text)
    elapsed_matches = re.findall(r"Elapsed:\s*(\d+):(\d+):(\d+)", text)
    memory_matches = re.findall(r"Peak CUDA memory allocated:\s*([\d.]+) MiB", text)
    if not (trainable_matches and elapsed_matches and memory_matches):
        raise ValueError("Missing runtime metadata in {}".format(log_path))
    hours, minutes, seconds = map(int, elapsed_matches[-1])
    return {
        "trainable_parameters": int(trainable_matches[-1].replace(",", "")),
        "elapsed_seconds": hours * 3600 + minutes * 60 + seconds,
        "peak_cuda_memory_mib": float(memory_matches[-1]),
    }


def load_hyperparameters(search_root, method):
    if method == "CoOp_VPT_Deep_TextDeep_Nv5_Nt4":
        return {
            "text_tokens": 4,
            "ctx_init": "a photo of a",
            "vpt_mode": "deep",
            "vpt_tokens": 5,
            "text_prompt_mode": "deep",
            "text_prompt_tokens": 4,
            "shared_lr": 5e-3,
            "validation_score": None,
        }
    prefix = "CoOp_VPT_Deep_Nv"
    if not method.startswith(prefix):
        raise ValueError("Only VPT-Deep methods are supported: {}".format(method))
    tokens = int(method[len(prefix):])
    selected_path = search_root / "selected_vpt_deep.json"
    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    record = selected["selections"][str(tokens)]
    return {
        "text_tokens": selected["text_tokens"],
        "ctx_init": selected["ctx_init"],
        "vpt_mode": "deep",
        "vpt_tokens": tokens,
        "text_prompt_mode": "off",
        "text_prompt_tokens": 0,
        "shared_lr": record["parameters"]["shared_lr"],
        "validation_score": record["score"],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/dermamnist_coop_native_vptdeep_adamw"),
    )
    parser.add_argument(
        "--search-root",
        type=Path,
        default=Path("output/dermamnist_coop_native_vptdeep_search"),
    )
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument("--shots", nargs="+", type=int, default=(1, 2, 4, 8, 16, 32))
    parser.add_argument("--seeds", nargs="+", type=int, default=(1, 2, 3))
    args = parser.parse_args()

    result = {}
    csv_rows = []
    for method in args.methods:
        result[method] = {}
        hyperparameters = load_hyperparameters(args.search_root, method)
        for shots in args.shots:
            records = []
            for seed in args.seeds:
                log = args.output_root / method / "shots_{}".format(shots) / "seed{}".format(seed) / "log.txt"
                summary, classes = parse_log(log)
                records.append({
                    "seed": seed,
                    "summary": summary,
                    "classes": classes,
                    **parse_runtime_metadata(log),
                })

            summary = {
                metric: aggregate([record["summary"][metric] for record in records])
                for metric in records[0]["summary"]
            }
            classes = {}
            for label in sorted({item["label"] for record in records for item in record["classes"]}):
                class_records = [
                    item for record in records for item in record["classes"] if item["label"] == label
                ]
                classes[str(label)] = {
                    "classname": class_records[0]["classname"],
                    "support": class_records[0]["support"],
                    "correct": aggregate([item["correct"] for item in class_records]),
                    **{
                        metric: aggregate([item[metric] for item in class_records])
                        for metric in ("accuracy", "precision", "recall", "specificity", "f1", "auc")
                    },
                }
            resources = {
                "trainable_parameters": records[0]["trainable_parameters"],
                "elapsed_seconds": aggregate([record["elapsed_seconds"] for record in records]),
                "peak_cuda_memory_mib": aggregate(
                    [record["peak_cuda_memory_mib"] for record in records]
                ),
                "runs": [
                    {
                        "seed": record["seed"],
                        "elapsed_seconds": record["elapsed_seconds"],
                        "peak_cuda_memory_mib": record["peak_cuda_memory_mib"],
                    }
                    for record in records
                ],
            }
            if any(
                record["trainable_parameters"] != resources["trainable_parameters"]
                for record in records
            ):
                raise ValueError("Trainable parameter count changed within {} {}-shot".format(method, shots))
            result[method][str(shots)] = {
                "hyperparameters": hyperparameters,
                "resources": resources,
                "summary": summary,
                "classes": classes,
            }

            for metric, values in summary.items():
                csv_rows.append(
                    {
                        "method": method, "shots": shots, "scope": "overall",
                        "label": "", "classname": "", "support": "",
                        "correct_mean": "", "correct_std": "",
                        "metric": metric, "mean": values["mean"],
                        "std": values["std"],
                        "ci95_half_width": values["ci95_half_width"],
                    }
                )
            for metric in ("elapsed_seconds", "peak_cuda_memory_mib"):
                values = resources[metric]
                csv_rows.append(
                    {
                        "method": method, "shots": shots, "scope": "resource",
                        "label": "", "classname": "", "support": "",
                        "correct_mean": "", "correct_std": "",
                        "metric": metric, "mean": values["mean"],
                        "std": values["std"],
                        "ci95_half_width": values["ci95_half_width"],
                    }
                )
            csv_rows.append(
                {
                    "method": method, "shots": shots, "scope": "resource",
                    "label": "", "classname": "", "support": "",
                    "correct_mean": "", "correct_std": "",
                    "metric": "trainable_parameters",
                    "mean": resources["trainable_parameters"], "std": 0,
                    "ci95_half_width": 0,
                }
            )
            for label, class_result in classes.items():
                for metric in ("accuracy", "precision", "recall", "specificity", "f1", "auc"):
                    csv_rows.append(
                        {
                            "method": method, "shots": shots, "scope": "class",
                            "label": label, "classname": class_result["classname"],
                            "support": class_result["support"],
                            "correct_mean": class_result["correct"]["mean"],
                            "correct_std": class_result["correct"]["std"],
                            "metric": metric,
                            "mean": class_result[metric]["mean"],
                            "std": class_result[metric]["std"],
                            "ci95_half_width": class_result[metric]["ci95_half_width"],
                        }
                    )

    args.output_root.mkdir(parents=True, exist_ok=True)
    json_path = args.output_root / "metrics_summary.json"
    csv_path = args.output_root / "metrics_summary.csv"
    md_path = args.output_root / "results_summary.md"
    json_path.write_text(json.dumps(result, indent=2, allow_nan=True), encoding="utf-8")
    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)

    lines = [
        "# DermaMNIST CoOp/VPT results",
        "",
        "| Method | Shot | Accuracy | Balanced accuracy | AUC | Macro F1 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for method in args.methods:
        for shots in args.shots:
            summary = result[method][str(shots)]["summary"]
            def cell(metric):
                values = summary[metric]
                return "{:.2f} ± {:.2f}".format(values["mean"], values["std"])
            lines.append(
                "| {} | {} | {} | {} | {} | {} |".format(
                    method, shots, cell("accuracy"), cell("balanced_accuracy"),
                    cell("auc"), cell("macro_f1")
                )
            )
    lines.extend([
        "",
        "## Selected hyperparameters",
        "",
        "| Method | CoOp tokens | Shared LR | VPT mode | Visual tokens | Text mode | Text tokens | Trainable params |",
        "|---|---:|---:|---|---:|---|---:|---:|",
    ])
    for method in args.methods:
        first = result[method][str(args.shots[0])]
        params = first["hyperparameters"]
        lines.append(
            "| {} | {} | {} | {} | {} | {} | {} | {:,} |".format(
                method, params["text_tokens"], params["shared_lr"],
                params.get("vpt_mode", "off"), params.get("vpt_tokens", 0),
                params.get("text_prompt_mode", "off"),
                params.get("text_prompt_tokens", 0),
                first["resources"]["trainable_parameters"],
            )
        )
    lines.extend([
        "",
        "## Runtime and peak memory",
        "",
        "Values are mean ± sample standard deviation across three seeds.",
        "",
        "| Method | Shot | Elapsed (s) | Peak CUDA memory (MiB) |",
        "|---|---:|---:|---:|",
    ])
    for method in args.methods:
        for shots in args.shots:
            resources = result[method][str(shots)]["resources"]
            lines.append(
                "| {} | {} | {:.1f} ± {:.1f} | {:.2f} ± {:.2f} |".format(
                    method, shots,
                    resources["elapsed_seconds"]["mean"], resources["elapsed_seconds"]["std"],
                    resources["peak_cuda_memory_mib"]["mean"],
                    resources["peak_cuda_memory_mib"]["std"],
                )
            )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("Wrote {}, {}, and {}".format(json_path, csv_path, md_path))


if __name__ == "__main__":
    main()
