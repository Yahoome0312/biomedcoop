"""Aggregate BiomedCoOp DermaMNIST logs, including per-class metrics.

The evaluator writes the final test metrics to each seed's ``log.txt``.  This
script keeps the three-seed aggregation reproducible and emits both JSON and
CSV summaries with mean, sample standard deviation, and 95% CI half-width.
"""

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


SUMMARY_PATTERNS = {
    "accuracy": re.compile(r"\* accuracy: ([.\deE+-]+)%"),
    "balanced_accuracy": re.compile(r"\* balanced_accuracy: ([.\deE+-]+)%"),
    "auc": re.compile(r"\* auc: ([.\deE+-]+)%"),
    "macro_f1": re.compile(r"\* macro_f1: ([.\deE+-]+)%"),
    "error_rate": re.compile(r"\* error: ([.\deE+-]+)%"),
}

CLASS_PATTERN = re.compile(
    r"\* class: (\d+) \((.*?)\)\s+"
    r"total: ([\d,]+)\s+correct: ([\d,]+)\s+"
    r"acc: ([.\deE+-]+)%\s+"
    r"precision: ([.\deE+-]+)%\s+"
    r"recall: ([.\deE+-]+)%\s+"
    r"specificity: ([.\deE+-]+)%\s+"
    r"f1: ([.\deE+-]+)%\s+"
    r"auc: ([.\deE+-]+)%"
)


def _last_float(pattern, text):
    matches = pattern.findall(text)
    if not matches:
        raise ValueError("Metric not found: {}".format(pattern.pattern))
    return float(matches[-1])


def parse_log(path):
    text = path.read_text(encoding="utf-8", errors="replace")
    result_text = text.rsplit("=> result", 1)[-1]
    summary = {
        name: _last_float(pattern, result_text)
        for name, pattern in SUMMARY_PATTERNS.items()
    }
    classes = []
    for match in CLASS_PATTERN.finditer(result_text):
        (
            label,
            classname,
            support,
            correct,
            acc,
            precision,
            recall,
            specificity,
            f1,
            auc,
        ) = match.groups()
        classes.append(
            {
                "label": int(label),
                "classname": classname,
                "support": int(support.replace(",", "")),
                "correct": int(correct.replace(",", "")),
                "accuracy": float(acc),
                "precision": float(precision),
                "recall": float(recall),
                "specificity": float(specificity),
                "f1": float(f1),
                "auc": float(auc),
            }
        )
    if not classes:
        raise ValueError("No per-class metrics found in {}".format(path))
    return summary, classes


def aggregate(values):
    arr = np.asarray(values, dtype=float)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0
    return {
        "mean": mean,
        "std": std,
        "ci95_half_width": 1.96 * std / math.sqrt(len(arr))
        if len(arr) > 1 else 0.0,
        "values": [float(x) for x in arr],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", default="output")
    parser.add_argument("--dataset", default="dermamnist")
    parser.add_argument("--trainer", default="BiomedCoOp_BiomedCLIP")
    parser.add_argument("--shots", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3])
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    root = Path(args.output_root)
    # Support both the upstream layout (root/dataset/shots_X) and the
    # Windows reproduction driver's layout (root/shots_X).
    dataset_root = root
    if not any((root / "shots_{}".format(shots)).exists() for shots in args.shots):
        dataset_root = root / args.dataset
    records = []
    for shots in args.shots:
        for seed in args.seeds:
            log_path = (
                dataset_root
                / "shots_{}".format(shots)
                / args.trainer
                / "nctx4_cscFalse_ctpend"
                / "seed{}".format(seed)
                / "log.txt"
            )
            summary, classes = parse_log(log_path)
            records.append(
                {
                    "shots": shots,
                    "seed": seed,
                    "log": str(log_path),
                    "summary": summary,
                    "classes": classes,
                }
            )

    grouped = {}
    for shots in args.shots:
        shot_records = [r for r in records if r["shots"] == shots]
        summary = {
            key: aggregate([r["summary"][key] for r in shot_records])
            for key in SUMMARY_PATTERNS
        }
        class_labels = sorted({c["label"] for r in shot_records for c in r["classes"]})
        classes = {}
        for label in class_labels:
            class_records = [
                c
                for r in shot_records
                for c in r["classes"]
                if c["label"] == label
            ]
            classes[str(label)] = {
                "classname": class_records[0]["classname"],
                "support": class_records[0]["support"],
                "correct": aggregate([c["correct"] for c in class_records]),
                **{
                    key: aggregate([c[key] for c in class_records])
                    for key in (
                        "accuracy",
                        "precision",
                        "recall",
                        "specificity",
                        "f1",
                        "auc",
                    )
                },
            }
        grouped[str(shots)] = {"summary": summary, "classes": classes}

    output = Path(args.output) if args.output else dataset_root / "metrics_summary.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(grouped, indent=2, allow_nan=True), encoding="utf-8")

    csv_path = output.with_suffix(".csv")
    rows = []
    for shots, result in grouped.items():
        for metric, values in result["summary"].items():
            rows.append(
                {
                    "shots": shots,
                    "scope": "overall",
                    "label": "",
                    "classname": "",
                    "support": "",
                    "correct_mean": "",
                    "correct_std": "",
                    "metric": metric,
                    "mean": values["mean"],
                    "std": values["std"],
                    "ci95_half_width": values["ci95_half_width"],
                }
            )
        for label, class_result in result["classes"].items():
            for metric in ("accuracy", "precision", "recall", "specificity", "f1", "auc"):
                values = class_result[metric]
                rows.append(
                    {
                        "shots": shots,
                        "scope": "class",
                        "label": label,
                        "classname": class_result["classname"],
                        "support": class_result["support"],
                        "correct_mean": class_result["correct"]["mean"],
                        "correct_std": class_result["correct"]["std"],
                        "metric": metric,
                        "mean": values["mean"],
                        "std": values["std"],
                        "ci95_half_width": values["ci95_half_width"],
                    }
                )
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print("Wrote {} and {}".format(output, csv_path))


if __name__ == "__main__":
    main()
