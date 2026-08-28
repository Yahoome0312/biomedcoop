"""Export a focused, complete B0/full evaluation report.

The main experiment report contains all variants.  This utility extracts the
two requested endpoints (``b0`` and ``full``) and adds the confusion matrices
for B0 as well as the pair/loss diagnostics for ``full``.  It never reruns
training or evaluation; all values are read from the completed run artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import fmean
from typing import Any


VARIANTS = ("b0", "full")
SHOTS = (4, 8, 16, 32)
SELECTIONS = ("accuracy", "balanced_accuracy")
METRICS = (
    "accuracy",
    "error_rate",
    "balanced_accuracy",
    "macro_f1",
    "macro_recall",
    "auc",
)


def _load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_tensor(path: Path) -> list[list[float]]:
    """Load a small saved confusion matrix without requiring a fixed torch API."""

    import torch

    try:
        matrix = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch < 2.6
        matrix = torch.load(path, map_location="cpu")
    if hasattr(matrix, "tolist"):
        matrix = matrix.tolist()
    return [[float(value) for value in row] for row in matrix]


def _class_names(report: dict[str, Any]) -> list[str]:
    for shot in SHOTS:
        for seed in (1, 2, 3):
            entry = report.get("soft_confusion_bank", {}).get(str(shot), {}).get(str(seed))
            if entry and entry.get("class_order"):
                return [str(value) for value in entry["class_order"]]
    return [f"class_{index}" for index in range(7)]


def _fmt_pm(metric: dict[str, Any] | None, scale: float = 1.0, digits: int = 2) -> str:
    if not metric:
        return "—"
    return f"{float(metric['mean']) * scale:.{digits}f} ± {float(metric['std']) * scale:.{digits}f}"


def _fmt_value(value: Any, digits: int = 2, scale: float = 1.0) -> str:
    if value is None:
        return "—"
    return f"{float(value) * scale:.{digits}f}"


def _top_pairs_text(pairs: list[dict[str, Any]] | None, limit: int = 3) -> str:
    if not pairs:
        return "—"
    rendered: list[str] = []
    for pair in pairs[:limit]:
        first = pair.get("first_name", pair.get("first", "?"))
        second = pair.get("second_name", pair.get("second", "?"))
        fraction = pair.get("fraction")
        suffix = f" {float(fraction):.1%}" if fraction is not None else ""
        rendered.append(f"{first}→{second}{suffix}")
    return "; ".join(rendered)


def _matrix_stats(
    report: dict[str, Any],
    variant: str,
    shot: int,
    selection: str,
    class_names: list[str],
) -> dict[str, Any]:
    """Aggregate the raw and row-normalized matrices from all three seeds."""

    method_dir = Path(report["methods"][variant]["method_dir"])
    raw_by_seed: list[list[list[float]]] = []
    normalized_by_seed: list[list[list[float]]] = []
    paths: list[dict[str, Any]] = []
    for seed in (1, 2, 3):
        run_dir = method_dir / f"shots_{shot}" / f"seed{seed}"
        raw_path = run_dir / f"test_cmat_raw_selected_by_{selection}.pt"
        normalized_path = run_dir / f"test_cmat_selected_by_{selection}.pt"
        raw = _load_tensor(raw_path)
        normalized = _load_tensor(normalized_path)
        raw_by_seed.append(raw)
        normalized_by_seed.append(normalized)
        paths.append(
            {
                "seed": seed,
                "raw": str(raw_path.resolve()),
                "normalized": str(normalized_path.resolve()),
            }
        )

    num_classes = len(raw_by_seed[0])
    raw_sum = [
        [sum(matrix[row][column] for matrix in raw_by_seed) for column in range(num_classes)]
        for row in range(num_classes)
    ]
    raw_mean = [
        [value / len(raw_by_seed) for value in row]
        for row in raw_sum
    ]
    normalized_mean = [
        [
            fmean(matrix[row][column] for matrix in normalized_by_seed)
            for column in range(num_classes)
        ]
        for row in range(num_classes)
    ]
    total = sum(sum(row) for row in raw_sum)
    diagonal = sum(raw_sum[index][index] for index in range(num_classes))
    largest_value = -1.0
    largest_pair: tuple[int, int] | None = None
    for row in range(num_classes):
        for column in range(num_classes):
            if row == column:
                continue
            if normalized_mean[row][column] > largest_value:
                largest_value = normalized_mean[row][column]
                largest_pair = (row, column)
    largest_text = None
    if largest_pair is not None:
        first, second = largest_pair
        largest_text = {
            "first": first,
            "second": second,
            "first_name": class_names[first],
            "second_name": class_names[second],
            "value": largest_value,
        }
    return {
        "seed_count": len(raw_by_seed),
        "paths": paths,
        "raw_sum": raw_sum,
        "raw_mean": raw_mean,
        "normalized_mean": normalized_mean,
        "total": total,
        "diagonal": diagonal,
        "off_diagonal_error_rate": 1.0 - diagonal / total if total else 0.0,
        "largest_normalized_off_diagonal": largest_text,
    }


def _matrix_link(path: str, label: str) -> str:
    target = Path(path).resolve().as_posix()
    # The app resolves Windows absolute paths from a leading slash (``/D:/``).
    if not target.startswith("/"):
        target = "/" + target
    return f"[{label}]({target})"


def _matrix_table(matrix: list[list[float]], class_names: list[str], integer: bool) -> list[str]:
    lines = [
        "| true\\pred | " + " | ".join(class_names) + " |",
        "|---|" + "---:|" * len(class_names),
    ]
    for name, row in zip(class_names, matrix):
        if integer:
            values = [str(int(round(value))) for value in row]
        else:
            values = [f"{value:.2%}" for value in row]
        lines.append("| " + name + " | " + " | ".join(values) + " |")
    return lines


def export(report: dict[str, Any], output_dir: Path) -> tuple[Path, Path]:
    class_names = _class_names(report)
    focused: dict[str, Any] = {
        "source_report": str(Path(report["experiment_root"]) / "experiment_report.json"),
        "variants": list(VARIANTS),
        "shots": list(SHOTS),
        "seeds": [1, 2, 3],
        "selection_metrics": list(SELECTIONS),
        "class_order": class_names,
        "metrics": {},
        "confusion_pair_metrics": {},
        "confusion_matrices": {},
        "full_gate_analysis": report.get("full_gate_analysis", {}),
    }
    markdown: list[str] = [
        "# B₀ / full 完整测试指标",
        "",
        "本文件从已完成的 `experiment_report.json` 和每个 run 的测试产物导出；不重新训练。",
        "所有 scalar 指标为 test 三个 seed 的 mean ± population std；百分数以 `%` 展示。",
        "类别顺序：" + "、".join(f"{index}={name}" for index, name in enumerate(class_names)) + "。",
        "",
    ]

    for selection in SELECTIONS:
        selection_label = "ACC-selected" if selection == "accuracy" else "BACC-selected"
        markdown.extend(
            [
                f"## {selection_label}：aggregate scalar + per-class recall",
                "",
                "| Variant | Shot | Accuracy | Error rate | BACC | Macro-F1 | Macro Recall | AUC | "
                + " | ".join(f"R{index}" for index in range(len(class_names)))
                + " |",
                "|---|---:|---:|---:|---:|---:|---:|---:|"
                + "---:|" * len(class_names),
            ]
        )
        for variant in VARIANTS:
            variant_data = report["methods"][variant]["shots"]
            focused["metrics"].setdefault(selection, {})
            focused["metrics"][selection].setdefault(variant, {})
            for shot in SHOTS:
                entry = variant_data[str(shot)][selection]
                metrics = entry["metrics"]
                focused["metrics"][selection][variant][str(shot)] = entry
                scalar_cells = [
                    _fmt_pm(metrics.get("accuracy"), 1.0, 2),
                    _fmt_pm(metrics.get("error_rate"), 1.0, 2),
                    _fmt_pm(metrics.get("balanced_accuracy"), 1.0, 2),
                    _fmt_pm(metrics.get("macro_f1"), 1.0, 2),
                    _fmt_pm(metrics.get("macro_recall"), 1.0, 2),
                    _fmt_pm(metrics.get("auc"), 1.0, 2),
                ]
                recalls = [
                    _fmt_pm(metrics.get(f"recall_class_{index}"), 1.0, 2)
                    for index in range(len(class_names))
                ]
                markdown.append(
                    "| " + " | ".join([variant, str(shot)] + scalar_cells + recalls) + " |"
                )
        markdown.append("")

        markdown.extend(
            [
                f"### {selection_label}：逐 seed scalar + per-class recall",
                "",
                "| Variant | Shot | Seed | Best epoch | Accuracy | Error rate | BACC | Macro-F1 | Macro Recall | AUC | "
                + " | ".join(f"R{index}" for index in range(len(class_names)))
                + " |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
                + "---:|" * len(class_names),
            ]
        )
        for variant in VARIANTS:
            for shot in SHOTS:
                entry = report["methods"][variant]["shots"][str(shot)][selection]
                for seed_entry in entry["per_seed"]:
                    values = [
                        _fmt_value(seed_entry.get("accuracy")),
                        _fmt_value(seed_entry.get("error_rate")),
                        _fmt_value(seed_entry.get("balanced_accuracy")),
                        _fmt_value(seed_entry.get("macro_f1")),
                        _fmt_value(seed_entry.get("macro_recall")),
                        _fmt_value(seed_entry.get("auc")),
                    ]
                    recalls = [
                        _fmt_value(seed_entry.get(f"recall_class_{index}"))
                        for index in range(len(class_names))
                    ]
                    markdown.append(
                        "| "
                        + " | ".join(
                            [
                                variant,
                                str(shot),
                                str(seed_entry.get("seed")),
                                str(seed_entry.get("best_epoch")),
                            ]
                            + values
                            + recalls
                        )
                        + " |"
                    )
        markdown.append("")

    markdown.extend(
        [
            "## Confusion pair / confusion loss（full）",
            "",
            "B₀ 不启用 pair branch，因此 pair 与 confusion loss 对 B₀ 标记为 disabled；B₀ 的实际分类混淆矩阵见下一节。",
            "",
                "| Selection | Shot | Test pair coverage | Val pair coverage | Test B0 top-1 in pair | Unique undirected | Selected prior | Selected score | L_conf mean | L_conf final | Top-3 test pairs | Train selected top-3 | Train epoch-100 top-3 |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|",
        ]
    )
    for selection in SELECTIONS:
        for shot in SHOTS:
            entry = report["methods"]["full"]["confusion_metrics"]["shots"][str(shot)][selection]
            test = entry.get("test") or {}
            validation = entry.get("validation") or {}
            loss = entry.get("loss") or {}
            loss_metric = loss.get("metrics", {}).get("loss_confuse") or {}
            focused["confusion_pair_metrics"].setdefault(selection, {})[str(shot)] = entry
            markdown.append(
                "| full | {shot} | {coverage} | {val} | {top1} | {unique} | {prior} | {score} | {loss_mean} | {loss_final} | {test_pairs} | {train_selected} | {train_final} |".format(
                    shot=shot,
                    coverage=_fmt_value(test.get("pair_coverage_true_label", {}).get("mean"), 2, 100.0)
                    + (f" ± {float(test['pair_coverage_true_label']['std']) * 100.0:.2f}" if test.get("pair_coverage_true_label") else ""),
                    val=_fmt_value(validation.get("pair_coverage_true_label", {}).get("mean"), 2, 100.0)
                    + (f" ± {float(validation['pair_coverage_true_label']['std']) * 100.0:.2f}" if validation.get("pair_coverage_true_label") else ""),
                    top1=_fmt_value(test.get("pair_first_matches_true", {}).get("mean"), 2, 100.0)
                    + (f" ± {float(test['pair_first_matches_true']['std']) * 100.0:.2f}" if test.get("pair_first_matches_true") else ""),
                    unique=_fmt_value(test.get("pair_unique_undirected", {}).get("mean"), 2)
                    + (f" ± {float(test['pair_unique_undirected']['std']):.2f}" if test.get("pair_unique_undirected") else ""),
                    prior=_fmt_value(test.get("selected_prior_mean", {}).get("mean"), 4)
                    + (f" ± {float(test['selected_prior_mean']['std']):.4f}" if test.get("selected_prior_mean") else ""),
                    score=_fmt_value(test.get("selected_score_mean", {}).get("mean"), 4)
                    + (f" ± {float(test['selected_score_mean']['std']):.4f}" if test.get("selected_score_mean") else ""),
                    loss_mean=_fmt_pm(loss_metric.get("mean"), 1.0, 4)
                    if isinstance(loss_metric.get("mean"), dict)
                    else _fmt_value(loss_metric.get("mean"), 4),
                    loss_final=_fmt_pm(loss_metric.get("final_epoch_mean"), 1.0, 4),
                    test_pairs=_top_pairs_text(test.get("top_pairs")),
                    train_selected=_top_pairs_text((entry.get("train_selected") or {}).get("top_pairs")),
                    train_final=_top_pairs_text((entry.get("train_final") or {}).get("top_pairs")),
                )
            )
    markdown.append("")

    markdown.extend(
        [
            "## full gate（global/local sample-level weights）",
            "",
            "`alpha_local = 1 - alpha_global`; values are aggregated over the three-seed test streams.",
            "",
            "| Selection | Shot | α_global mean ± std | α_local mean | Global-dominant fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for selection in SELECTIONS:
        for shot in SHOTS:
            gate = report.get("full_gate_analysis", {}).get(str(shot), {}).get(selection)
            if not gate:
                markdown.append(f"| {selection} | {shot} | — | — | — |")
                continue
            markdown.append(
                "| {selection} | {shot} | {mean:.6f} ± {std:.6f} | {local:.6f} | {fraction:.2%} |".format(
                    selection=selection,
                    shot=shot,
                    mean=float(gate["alpha_global_mean"]),
                    std=float(gate["alpha_global_std"]),
                    local=float(gate["alpha_local_mean"]),
                    fraction=float(gate["global_dominant_fraction"]),
                )
            )
    markdown.append("")

    markdown.extend(
        [
            "## Test confusion matrices（B₀ 与 full）",
            "",
            "raw matrix 为三 seed 合计（行=true，列=prediction）；normalized matrix 为三个 seed 的 row-normalized 矩阵平均。",
            "",
            "| Variant | Selection | Shot | Off-diagonal error rate | Largest normalized off-diagonal | Matrix files |",
            "|---|---|---:|---:|---|---|",
        ]
    )
    for variant in VARIANTS:
        focused["confusion_matrices"].setdefault(variant, {})
        for selection in SELECTIONS:
            focused["confusion_matrices"][variant].setdefault(selection, {})
            for shot in SHOTS:
                stats = _matrix_stats(report, variant, shot, selection, class_names)
                focused["confusion_matrices"][variant][selection][str(shot)] = stats
                largest = stats["largest_normalized_off_diagonal"]
                largest_text = (
                    f"{largest['first_name']}→{largest['second_name']} {largest['value']:.2%}"
                    if largest
                    else "—"
                )
                links = ", ".join(
                    f"seed{path['seed']} "
                    + _matrix_link(path["raw"], "raw")
                    + "/"
                    + _matrix_link(path["normalized"], "norm")
                    for path in stats["paths"]
                )
                markdown.append(
                    f"| {variant} | {selection} | {shot} | {stats['off_diagonal_error_rate']:.2%} | {largest_text} | {links} |"
                )
                markdown.extend(
                    [
                        "",
                        f"### {variant} / {selection} / {shot}-shot：raw sum",
                        "",
                    ]
                    + _matrix_table(stats["raw_sum"], class_names, integer=True)
                    + [
                        "",
                        f"### {variant} / {selection} / {shot}-shot：row-normalized mean",
                        "",
                    ]
                    + _matrix_table(stats["normalized_mean"], class_names, integer=False)
                    + ["" ]
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "b0_full_metrics.json"
    markdown_path = output_dir / "b0_full_metrics.md"
    json_path.write_text(json.dumps(focused, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text("\n".join(markdown), encoding="utf-8")
    return markdown_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("output/confusion_aware_from_scratch/experiment_report.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output/confusion_aware_from_scratch"),
    )
    args = parser.parse_args()
    report = _load(args.report.resolve())
    markdown_path, json_path = export(report, args.output_dir.resolve())
    print(f"WROTE {markdown_path}")
    print(f"WROTE {json_path}")


if __name__ == "__main__":
    main()
