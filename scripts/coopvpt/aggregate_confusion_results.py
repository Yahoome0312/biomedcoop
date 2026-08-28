"""Aggregate completed from-scratch MT-TCP/confusion-aware experiments.

The runner writes one ``test_summary.json`` per variant. This utility combines
those files into a machine-readable report and a compact Markdown table while
preserving the complete per-seed and per-class metrics from the source files.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any


VARIANTS = (
    "b0",
    "pair",
    "semantic",
    "semantic_global",
    "semantic_local",
    "global_local",
    "full",
)
SHOTS = (4, 8, 16, 32)
SELECTIONS = ("accuracy", "balanced_accuracy")
METRICS = ("accuracy", "balanced_accuracy", "macro_f1", "macro_recall", "auc")
DEFAULT_CLASS_NAMES = tuple("class_{}".format(index) for index in range(7))


def _numeric_summary(values: list[float]) -> dict[str, Any] | None:
    """Return a compact mean/std summary, or ``None`` for missing values."""

    if not values:
        return None
    values = [float(value) for value in values]
    return {
        "count": len(values),
        "mean": fmean(values),
        "std": pstdev(values),
    }


def _safe_mean(values: list[float]) -> float | None:
    return float(fmean(values)) if values else None


def _class_names(bank: dict[str, Any] | None, num_classes: int = 7) -> list[str]:
    if bank:
        names = bank.get("metadata", {}).get("class_order")
        if names and len(names) == num_classes:
            return [str(name) for name in names]
    return list(DEFAULT_CLASS_NAMES[:num_classes])


def _load(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_gzip_json(path: Path) -> Any:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def _bank_path(
    root: Path,
    shot: int,
    seed: int,
    dataset: str = "DermaMNIST",
    bank_root: Path | None = None,
) -> Path | None:
    """Locate the immutable soft-prior Bank for a run.

    The default layout is the sibling ``soft_confusion_banks`` directory used
    by the experiment commands.  ``--bank-root`` is available for reports
    generated from a copied output tree.
    """

    candidates: list[Path] = []
    if bank_root is not None:
        candidates.append(bank_root)
    candidates.extend(
        [
            root.parent / "soft_confusion_banks",
            root / "soft_confusion_banks",
        ]
    )
    for candidate_root in candidates:
        path = (
            candidate_root
            / dataset
            / f"shots_{shot}"
            / f"seed{seed}"
            / "soft_confusion_bank.json"
        )
        if path.exists():
            return path
    return None


def _pair_matrix_from_samples(
    samples: list[dict[str, Any]],
    num_classes: int = 7,
    class_names: list[str] | None = None,
) -> dict[str, Any] | None:
    """Summarize selected pairs in a validation/test analysis stream.

    ``pair_first`` is the detached B0 top-1 class and ``pair_second`` is the
    static-prior-adjusted competitor.  Both directional and canonical
    (unordered) frequencies are retained so that a pair is not split merely
    because the top-1 direction changed.
    """

    if not samples:
        return None
    class_names = class_names or list(DEFAULT_CLASS_NAMES[:num_classes])
    directed = [[0 for _ in range(num_classes)] for _ in range(num_classes)]
    undirected: dict[str, int] = {}
    prior_values: list[float] = []
    score_values: list[float] = []
    alpha_global: list[float] = []
    alpha_local: list[float] = []
    true_in_pair = 0
    first_is_true = 0
    second_is_true = 0

    for sample in samples:
        first = int(sample["pair_first"])
        second = int(sample["pair_second"])
        if 0 <= first < num_classes and 0 <= second < num_classes:
            directed[first][second] += 1
            left, right = sorted((first, second))
            key = f"{left}-{right}"
            undirected[key] = undirected.get(key, 0) + 1
        true_label = int(sample["true_label"])
        true_in_pair += int(true_label == first or true_label == second)
        first_is_true += int(true_label == first)
        second_is_true += int(true_label == second)
        if "selected_prior" in sample:
            prior_values.append(float(sample["selected_prior"]))
        if "selected_score" in sample:
            score_values.append(float(sample["selected_score"]))
        if "alpha_global" in sample:
            alpha_global.append(float(sample["alpha_global"]))
        if "alpha_local" in sample:
            alpha_local.append(float(sample["alpha_local"]))

    sample_count = len(samples)
    directed_items = [
        (count, first, second)
        for first, row in enumerate(directed)
        for second, count in enumerate(row)
        if count
    ]
    directed_items.sort(reverse=True)
    top_pairs = [
        {
            "first": first,
            "second": second,
            "first_name": class_names[first] if first < len(class_names) else str(first),
            "second_name": class_names[second]
            if second < len(class_names)
            else str(second),
            "count": count,
            "fraction": count / sample_count,
        }
        for count, first, second in directed_items[:10]
    ]
    undirected_items = sorted(undirected.items(), key=lambda item: (-item[1], item[0]))
    top_undirected = []
    for key, count in undirected_items[:10]:
        first, second = (int(part) for part in key.split("-"))
        top_undirected.append(
            {
                "pair": key,
                "first": first,
                "second": second,
                "first_name": class_names[first] if first < len(class_names) else str(first),
                "second_name": class_names[second]
                if second < len(class_names)
                else str(second),
                "count": count,
                "fraction": count / sample_count,
            }
        )
    probabilities = [count / sample_count for count, _, _ in directed_items]
    pair_entropy = -sum(probability * math.log2(probability) for probability in probabilities)
    result: dict[str, Any] = {
        "sample_count": sample_count,
        "pair_count_total": sample_count,
        "pair_unique_directed": len(directed_items),
        "pair_unique_undirected": len(undirected),
        "pair_coverage_true_label": true_in_pair / sample_count,
        "pair_first_matches_true": first_is_true / sample_count,
        "pair_second_matches_true": second_is_true / sample_count,
        "pair_entropy_bits": pair_entropy,
        "selected_prior_mean": _safe_mean(prior_values),
        "selected_prior_std": pstdev(prior_values) if len(prior_values) > 1 else 0.0
        if prior_values
        else None,
        "selected_score_mean": _safe_mean(score_values),
        "selected_score_std": pstdev(score_values) if len(score_values) > 1 else 0.0
        if score_values
        else None,
        "alpha_global_mean": _safe_mean(alpha_global),
        "alpha_local_mean": _safe_mean(alpha_local),
        "pair_counts_directed": directed,
        "top_pairs": top_pairs,
        "top_undirected_pairs": top_undirected,
    }
    return result


def _pair_matrix_from_counts(
    counts: list[list[int]],
    class_names: list[str] | None = None,
) -> dict[str, Any]:
    """Summarize a train-epoch directional pair-count matrix."""

    class_names = class_names or list(DEFAULT_CLASS_NAMES[: len(counts)])
    total = sum(sum(int(value) for value in row) for row in counts)
    items = [
        (int(count), first, second)
        for first, row in enumerate(counts)
        for second, count in enumerate(row)
        if count
    ]
    items.sort(reverse=True)
    probabilities = [count / total for count, _, _ in items] if total else []
    top_pairs = [
        {
            "first": first,
            "second": second,
            "first_name": class_names[first] if first < len(class_names) else str(first),
            "second_name": class_names[second]
            if second < len(class_names)
            else str(second),
            "count": count,
            "fraction": count / total if total else 0.0,
        }
        for count, first, second in items[:10]
    ]
    return {
        "pair_count_total": total,
        "pair_unique_directed": len(items),
        "pair_entropy_bits": (
            -sum(probability * math.log2(probability) for probability in probabilities)
            if probabilities
            else 0.0
        ),
        "pair_counts_directed": [[int(value) for value in row] for row in counts],
        "top_pairs": top_pairs,
    }


def _load_train_pair_summary(
    run_dir: Path,
    epoch: int,
    class_names: list[str] | None = None,
) -> dict[str, Any] | None:
    path = run_dir / "confusion_analysis" / f"train_epoch_{epoch:03d}.json"
    if not path.exists():
        return None
    summary = _load(path)
    counts = summary.get("pair_counts")
    if counts is None:
        return None
    result = _pair_matrix_from_counts(counts, class_names)
    result["epoch"] = int(summary.get("epoch", epoch))
    if "alpha_global" in summary:
        result["alpha_global_mean"] = float(summary["alpha_global"])
        result["alpha_local_mean"] = float(summary.get("alpha_local", 1.0 - summary["alpha_global"]))
    return result


_LOG_EPOCH_RE = re.compile(r"epoch\s+\[(\d+)/(\d+)\]")
_LOG_NUMBER_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?"


def _load_loss_summary(run_dir: Path, final_epoch: int = 100) -> dict[str, Any]:
    """Parse scalar loss diagnostics emitted by the trainer's log.txt."""

    path = run_dir / "log.txt"
    records: list[tuple[int, dict[str, float]]] = []
    if path.exists():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            epoch_match = _LOG_EPOCH_RE.search(line)
            if not epoch_match:
                continue
            epoch = int(epoch_match.group(1))
            values: dict[str, float] = {}
            for key in ("loss", "loss_ce", "loss_confuse", "alpha_global", "alpha_local"):
                match = re.search(rf"\b{key}\s+({_LOG_NUMBER_RE})", line)
                if match:
                    values[key] = float(match.group(1))
            if values:
                records.append((epoch, values))

    result: dict[str, Any] = {
        "record_count": len(records),
        "final_epoch": final_epoch,
        "metrics": {},
    }
    for key in ("loss", "loss_ce", "loss_confuse", "alpha_global", "alpha_local"):
        values = [row.get(key) for _, row in records if key in row]
        values = [float(value) for value in values]
        final_values = [
            row[key] for epoch, row in records if epoch == final_epoch and key in row
        ]
        if not values:
            result["metrics"][key] = None
            continue
        result["metrics"][key] = {
            "mean": fmean(values),
            "std": pstdev(values) if len(values) > 1 else 0.0,
            "final_epoch_mean": fmean(final_values) if final_values else None,
            "final_epoch_count": len(final_values),
        }
    return result


def _aggregate_scalar_records(
    records: list[dict[str, Any]],
    keys: tuple[str, ...],
) -> dict[str, Any] | None:
    if not records:
        return None
    result: dict[str, Any] = {}
    for key in keys:
        values = [record[key] for record in records if record.get(key) is not None]
        if values:
            result[key] = _numeric_summary([float(value) for value in values])
    return result


def _aggregate_pair_summaries(
    summaries: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any] | None:
    if not summaries:
        return None
    scalar_keys = (
        "sample_count",
        "pair_count_total",
        "pair_unique_directed",
        "pair_unique_undirected",
        "pair_coverage_true_label",
        "pair_first_matches_true",
        "pair_second_matches_true",
        "pair_entropy_bits",
        "selected_prior_mean",
        "selected_prior_std",
        "selected_score_mean",
        "selected_score_std",
        "alpha_global_mean",
        "alpha_local_mean",
    )
    result = _aggregate_scalar_records(summaries, scalar_keys) or {}
    counts = [[0 for _ in class_names] for _ in class_names]
    for summary in summaries:
        for first, row in enumerate(summary.get("pair_counts_directed", [])):
            if first >= len(counts):
                break
            for second, value in enumerate(row):
                if second < len(counts[first]):
                    counts[first][second] += int(value)
    result["pair_counts_directed_total"] = counts
    total = sum(sum(row) for row in counts)
    items = [
        (count, first, second)
        for first, row in enumerate(counts)
        for second, count in enumerate(row)
        if count
    ]
    items.sort(reverse=True)
    result["top_pairs"] = [
        {
            "first": first,
            "second": second,
            "first_name": class_names[first],
            "second_name": class_names[second],
            "count": count,
            "fraction": count / total if total else 0.0,
        }
        for count, first, second in items[:10]
    ]
    return result


def _aggregate_train_summaries(
    summaries: list[dict[str, Any]],
    class_names: list[str],
) -> dict[str, Any] | None:
    if not summaries:
        return None
    result = _aggregate_pair_summaries(summaries, class_names) or {}
    for key in ("alpha_global_mean", "alpha_local_mean"):
        values = [summary[key] for summary in summaries if key in summary]
        if values:
            result[key] = _numeric_summary([float(value) for value in values])
    return result


def _aggregate_loss_summaries(summaries: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not summaries:
        return None
    result: dict[str, Any] = {"record_count": sum(item.get("record_count", 0) for item in summaries), "metrics": {}}
    for key in ("loss", "loss_ce", "loss_confuse", "alpha_global", "alpha_local"):
        metric_items = [item["metrics"].get(key) for item in summaries]
        metric_items = [item for item in metric_items if item is not None]
        if not metric_items:
            result["metrics"][key] = None
            continue
        metric_result: dict[str, Any] = {}
        for stat in ("mean", "std", "final_epoch_mean"):
            values = [item[stat] for item in metric_items if item.get(stat) is not None]
            if values:
                metric_result[stat] = _numeric_summary([float(value) for value in values])
        metric_result["final_epoch_count"] = sum(
            int(item.get("final_epoch_count", 0)) for item in metric_items
        )
        result["metrics"][key] = metric_result
    return result


def _aggregate_matrices(
    per_seed: list[dict[str, Any]],
    key: str,
) -> dict[str, Any] | None:
    matrices = [entry.get(key) for entry in per_seed if entry.get(key) is not None]
    if not matrices:
        return None
    rows = len(matrices[0])
    cols = len(matrices[0][0]) if rows else 0
    total = [[0 for _ in range(cols)] for _ in range(rows)]
    for matrix in matrices:
        for row_index, row in enumerate(matrix):
            for col_index, value in enumerate(row):
                if row_index < rows and col_index < cols:
                    total[row_index][col_index] += value
    mean = [
        [value / len(matrices) for value in row]
        for row in total
    ]
    return {
        "seed_count": len(matrices),
        "sum": total,
        "mean": mean,
    }


def _load_bank_diagnostics(
    root: Path,
    bank_root: Path | None = None,
) -> dict[str, Any]:
    """Load and expose the fixed soft prior plus hard-count diagnostics."""

    diagnostics: dict[str, Any] = {}
    for shot in SHOTS:
        diagnostics[str(shot)] = {}
        for seed in (1, 2, 3):
            # Dataset is stable for this experiment; use run metadata when it
            # is available so a copied report remains self-describing.
            run_manifest = (
                _method_dir(root, "b0")
                / f"shots_{shot}"
                / f"seed{seed}"
                / "run_manifest.json"
            )
            dataset = "DermaMNIST"
            if run_manifest.exists():
                dataset = str(_load(run_manifest).get("dataset", dataset))
            path = _bank_path(root, shot, seed, dataset, bank_root)
            if path is None:
                diagnostics[str(shot)][str(seed)] = {"missing": True}
                continue
            bank = _load(path)
            metadata = bank.get("metadata", {})
            soft_prior = bank.get("soft_prior", [])
            hard_counts = bank.get("hard_confusion_counts", [])
            class_names = _class_names(bank, len(soft_prior) or 7)
            support_sizes = [int(value) for value in metadata.get("support_size_per_class", [])]
            support_total = sum(support_sizes)
            hard_error_total = sum(
                int(value)
                for row_index, row in enumerate(hard_counts)
                for col_index, value in enumerate(row)
                if row_index != col_index
            )
            soft_items = [
                (float(value), row_index, col_index)
                for row_index, row in enumerate(soft_prior)
                for col_index, value in enumerate(row)
                if row_index != col_index
            ]
            soft_items.sort(reverse=True)
            top_soft_pairs = [
                {
                    "first": first,
                    "second": second,
                    "first_name": class_names[first],
                    "second_name": class_names[second],
                    "value": value,
                }
                for value, first, second in soft_items[:10]
            ]
            row_sums = [sum(float(value) for value in row) for row in soft_prior]
            diagnostics[str(shot)][str(seed)] = {
                "path": str(path.resolve()),
                "bank_fingerprint": metadata.get("bank_fingerprint"),
                "support_fingerprint": metadata.get("support_fingerprint"),
                "class_order": class_names,
                "support_size_per_class": support_sizes,
                "support_record_count": len(bank.get("support_records", [])),
                "hard_error_total": hard_error_total,
                "hard_error_rate": hard_error_total / support_total if support_total else None,
                "soft_prior_row_sums": row_sums,
                "soft_prior_diagonal": [
                    float(soft_prior[index][index]) for index in range(len(soft_prior))
                ],
                "soft_prior_off_diagonal_mean": (
                    fmean(value for value, _, _ in soft_items) if soft_items else 0.0
                ),
                "top_soft_pairs": top_soft_pairs,
                "soft_prior": soft_prior,
                "hard_confusion_counts": hard_counts,
                "checks": {
                    "prior_type_soft_probability_mean": metadata.get("prior_type")
                    == "soft_probability_mean",
                    "diagonal_zero": all(
                        abs(float(soft_prior[index][index])) == 0.0
                        for index in range(len(soft_prior))
                    ),
                    "off_diagonal_renormalized": bool(
                        metadata.get("off_diagonal_renormalized", True)
                    ),
                    "support_only": bool(metadata.get("support_only", False)),
                    "val_test_images_encoded": bool(
                        metadata.get("val_test_images_encoded", True)
                    ),
                },
            }
    return diagnostics


def _load_saved_confusion_matrix(
    run_dir: Path,
    selection: str,
    normalized: bool = False,
) -> list[list[int | float]] | None:
    """Load a runner-emitted raw or row-normalized test confusion matrix."""

    suffix = "" if normalized else "_raw"
    path = run_dir / f"test_cmat{suffix}_selected_by_{selection}.pt"
    if not path.exists():
        return None
    try:
        import torch

        try:
            matrix = torch.load(path, map_location="cpu", weights_only=False)
        except TypeError:  # PyTorch < 2.6
            matrix = torch.load(path, map_location="cpu")
        if hasattr(matrix, "tolist"):
            matrix = matrix.tolist()
        return matrix
    except Exception:
        # Matrix files are supplementary to the scalar report.  Keep report
        # generation usable in a minimal environment and expose the missing
        # matrix as ``None`` rather than silently fabricating values.
        return None


def _confusion_analysis(
    root: Path,
    variant: str,
    bank_diagnostics: dict[str, Any],
) -> dict[str, Any]:
    """Aggregate pair selection, pair frequency, and confusion-loss logs."""

    enabled = variant != "b0"
    result: dict[str, Any] = {"enabled": enabled, "shots": {}, "per_seed": []}
    method_dir = _method_dir(root, variant)
    for shot in SHOTS:
        result["shots"][str(shot)] = {}
        for selection in SELECTIONS:
            per_seed: list[dict[str, Any]] = []
            for seed in (1, 2, 3):
                run_dir = method_dir / f"shots_{shot}" / f"seed{seed}"
                bank = bank_diagnostics.get(str(shot), {}).get(str(seed), {})
                class_names = bank.get("class_order", list(DEFAULT_CLASS_NAMES))
                run_entry: dict[str, Any] = {
                    "shot": shot,
                    "seed": seed,
                    "selection": selection,
                    "enabled": enabled,
                }
                # Classification confusion matrices are meaningful for B₀ as
                # well.  Pair selection/loss remains disabled for B₀, but the
                # raw and row-normalized test matrices must still be exposed
                # in the report instead of being hidden behind that flag.
                run_entry.update(
                    {
                        "test_confusion_matrix_raw": _load_saved_confusion_matrix(
                            run_dir, selection, normalized=False
                        ),
                        "test_confusion_matrix_normalized": _load_saved_confusion_matrix(
                            run_dir, selection, normalized=True
                        ),
                        "test_confusion_matrix_raw_path": str(
                            run_dir / f"test_cmat_raw_selected_by_{selection}.pt"
                        ).replace("\\", "/"),
                        "test_confusion_matrix_normalized_path": str(
                            run_dir / f"test_cmat_selected_by_{selection}.pt"
                        ).replace("\\", "/"),
                    }
                )
                if not enabled:
                    per_seed.append(run_entry)
                    result["per_seed"].append(run_entry)
                    continue
                results_path = run_dir / "results.json"
                results = _load(results_path)
                best_epoch = int(results["selections"][selection]["best_epoch"])
                test_path = (
                    run_dir
                    / "confusion_analysis"
                    / f"test_selected_by_{selection}.json.gz"
                )
                val_path = run_dir / "confusion_analysis" / f"val_epoch_{best_epoch:03d}.json.gz"
                test_pair = (
                    _pair_matrix_from_samples(
                        _load_gzip_json(test_path), len(class_names), class_names
                    )
                    if test_path.exists()
                    else None
                )
                val_pair = (
                    _pair_matrix_from_samples(
                        _load_gzip_json(val_path), len(class_names), class_names
                    )
                    if val_path.exists()
                    else None
                )
                train_selected = _load_train_pair_summary(run_dir, best_epoch, class_names)
                train_final = _load_train_pair_summary(run_dir, 100, class_names)
                loss = _load_loss_summary(run_dir, final_epoch=100)
                run_entry.update(
                    {
                        "best_epoch": best_epoch,
                        "test": test_pair,
                        "validation": val_pair,
                        "train_selected": train_selected,
                        "train_final": train_final,
                        "loss": loss,
                        "test_analysis_path": str(test_path.resolve())
                        if test_path.exists()
                        else None,
                        "validation_analysis_path": str(val_path.resolve())
                        if val_path.exists()
                        else None,
                    }
                )
                per_seed.append(run_entry)
                result["per_seed"].append(run_entry)
            if enabled:
                test_summaries = [entry["test"] for entry in per_seed if entry.get("test")]
                val_summaries = [entry["validation"] for entry in per_seed if entry.get("validation")]
                train_selected_summaries = [
                    entry["train_selected"] for entry in per_seed if entry.get("train_selected")
                ]
                train_final_summaries = [
                    entry["train_final"] for entry in per_seed if entry.get("train_final")
                ]
                losses = [entry["loss"] for entry in per_seed if entry.get("loss")]
                class_names = next(
                    (
                        bank_diagnostics[str(shot)][str(seed)].get(
                            "class_order", list(DEFAULT_CLASS_NAMES)
                        )
                        for seed in (1, 2, 3)
                        if not bank_diagnostics[str(shot)][str(seed)].get("missing")
                    ),
                    list(DEFAULT_CLASS_NAMES),
                )
                result["shots"][str(shot)][selection] = {
                    "best_epoch": _aggregate_scalar_records(per_seed, ("best_epoch",)),
                    "test": _aggregate_pair_summaries(test_summaries, class_names),
                    "validation": _aggregate_pair_summaries(val_summaries, class_names),
                    "train_selected": _aggregate_train_summaries(
                        train_selected_summaries, class_names
                    ),
                    "train_final": _aggregate_train_summaries(
                        train_final_summaries, class_names
                    ),
                    "loss": _aggregate_loss_summaries(losses),
                    "test_confusion_matrix_raw": _aggregate_matrices(
                        per_seed, "test_confusion_matrix_raw"
                    ),
                    "test_confusion_matrix_normalized": _aggregate_matrices(
                        per_seed, "test_confusion_matrix_normalized"
                    ),
                    "per_seed": per_seed,
                }
            else:
                result["shots"][str(shot)][selection] = {
                    "enabled": False,
                    "per_seed": per_seed,
                    "test_confusion_matrix_raw": _aggregate_matrices(
                        per_seed, "test_confusion_matrix_raw"
                    ),
                    "test_confusion_matrix_normalized": _aggregate_matrices(
                        per_seed, "test_confusion_matrix_normalized"
                    ),
                }
    return result


def _fmt(metric: dict[str, Any]) -> str:
    return f"{metric['mean']:.2f} ± {metric['std']:.2f}"


def _fmt_summary(metric: dict[str, Any] | None, scale: float = 1.0, digits: int = 3) -> str:
    if not metric:
        return "—"
    return f"{metric['mean'] * scale:.{digits}f} ± {metric['std'] * scale:.{digits}f}"


def _fmt_scalar(value: Any, digits: int = 3, scale: float = 1.0) -> str:
    if value is None:
        return "—"
    return f"{float(value) * scale:.{digits}f}"


def _top_pair_text(pairs: list[dict[str, Any]] | None, limit: int = 3) -> str:
    if not pairs:
        return "—"
    values = []
    for pair in pairs[:limit]:
        left = pair.get("first_name", pair.get("first", "?"))
        right = pair.get("second_name", pair.get("second", "?"))
        fraction = pair.get("fraction")
        suffix = f" {float(fraction):.1%}" if fraction is not None else ""
        values.append(f"{left}→{right}{suffix}")
    return "; ".join(values)


def _top_soft_pair_text(pairs: list[dict[str, Any]] | None, limit: int = 3) -> str:
    if not pairs:
        return "—"
    values = []
    for pair in pairs[:limit]:
        left = pair.get("first_name", pair.get("first", "?"))
        right = pair.get("second_name", pair.get("second", "?"))
        values.append(f"{left}→{right} {float(pair['value']):.4f}")
    return "; ".join(values)


def _method_dir(root: Path, variant: str) -> Path:
    return root / f"FromScratch_MT-TCP_CE_{variant}"


def _audit_variant(root: Path, variant: str) -> dict[str, Any]:
    method_dir = _method_dir(root, variant)
    runs: list[dict[str, Any]] = []
    fingerprints: set[str] = set()
    parameter_counts: dict[str, int] | None = None
    fingerprints_by_run: dict[str, str] = {}

    for shot in SHOTS:
        for seed in (1, 2, 3):
            run_dir = method_dir / f"shots_{shot}" / f"seed{seed}"
            init_manifest = _load(run_dir / "initialization_manifest.json")
            fingerprint = init_manifest["b0_initialization_fingerprint"]
            fingerprints.add(fingerprint)
            fingerprints_by_run[f"shots_{shot}/seed{seed}"] = fingerprint
            parameter_counts = parameter_counts or init_manifest["parameter_counts"]
            expected = {
                "accuracy_checkpoint": run_dir
                / "prompt_parameters"
                / "model-best-accuracy.pth.tar",
                "balanced_accuracy_checkpoint": run_dir
                / "prompt_parameters"
                / "model-best-balanced_accuracy.pth.tar",
                "results": run_dir / "results.json",
                "run_complete": run_dir / "run_complete.json",
                "accuracy_confusion_matrix": run_dir
                / "test_cmat_raw_selected_by_accuracy.pt",
                "balanced_accuracy_confusion_matrix": run_dir
                / "test_cmat_raw_selected_by_balanced_accuracy.pt",
            }
            missing = [name for name, path in expected.items() if not path.exists()]
            runs.append(
                {
                    "shot": shot,
                    "seed": seed,
                    "run_dir": str(run_dir.resolve()),
                    "missing": missing,
                }
            )

    return {
        "run_count": len(runs),
        "complete_run_count": sum(not run["missing"] for run in runs),
        "runs": runs,
        "b0_initialization_fingerprints": sorted(fingerprints),
        "b0_initialization_fingerprints_by_run": fingerprints_by_run,
        "parameter_counts": parameter_counts,
    }


def _full_gate_analysis(root: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    full_dir = _method_dir(root, "full")
    for shot in SHOTS:
        result[str(shot)] = {}
        for selection in SELECTIONS:
            all_global: list[float] = []
            per_seed: list[dict[str, Any]] = []
            for seed in (1, 2, 3):
                path = (
                    full_dir
                    / f"shots_{shot}"
                    / f"seed{seed}"
                    / "confusion_analysis"
                    / f"test_selected_by_{selection}.json.gz"
                )
                with gzip.open(path, "rt", encoding="utf-8") as handle:
                    samples = json.load(handle)
                values = [float(sample["alpha_global"]) for sample in samples]
                all_global.extend(values)
                per_seed.append(
                    {
                        "seed": seed,
                        "sample_count": len(values),
                        "alpha_global_mean": fmean(values),
                        "alpha_global_std": pstdev(values),
                        "alpha_global_min": min(values),
                        "alpha_global_max": max(values),
                        "global_dominant_fraction": sum(value > 0.5 for value in values)
                        / len(values),
                    }
                )
            result[str(shot)][selection] = {
                "sample_count": len(all_global),
                "alpha_global_mean": fmean(all_global),
                "alpha_global_std": pstdev(all_global),
                "alpha_global_min": min(all_global),
                "alpha_global_max": max(all_global),
                "alpha_local_mean": 1.0 - fmean(all_global),
                "global_dominant_fraction": sum(value > 0.5 for value in all_global)
                / len(all_global),
                "per_seed": per_seed,
            }
    return result


def aggregate(root: Path, bank_root: Path | None = None) -> dict[str, Any]:
    methods: dict[str, Any] = {}
    all_fingerprints: set[str] = set()
    fingerprints_by_run: dict[str, set[str]] = {
        f"shots_{shot}/seed{seed}": set()
        for shot in SHOTS
        for seed in (1, 2, 3)
    }

    bank_diagnostics = _load_bank_diagnostics(root, bank_root)

    for variant in VARIANTS:
        method_dir = _method_dir(root, variant)
        summary_path = method_dir / "test_summary.json"
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing completed test summary: {summary_path}")
        summary = _load(summary_path)
        audit = _audit_variant(root, variant)
        all_fingerprints.update(audit["b0_initialization_fingerprints"])
        for run_key, fingerprint in audit["b0_initialization_fingerprints_by_run"].items():
            fingerprints_by_run[run_key].add(fingerprint)
        methods[variant] = {
            "method_dir": str(method_dir.resolve()),
            "protocol": summary["protocol"],
            "shots": summary["shots"],
            "audit": audit,
        }
        methods[variant]["confusion_metrics"] = _confusion_analysis(
            root, variant, bank_diagnostics
        )

    fingerprint_audit = {
        run_key: sorted(values) for run_key, values in fingerprints_by_run.items()
    }
    return {
        "experiment_root": str(root.resolve()),
        "variants": list(VARIANTS),
        "shots": list(SHOTS),
        "seeds": [1, 2, 3],
        "selection_metrics": list(SELECTIONS),
        "reported_metrics": list(METRICS),
        "all_b0_initialization_fingerprints": sorted(all_fingerprints),
        "b0_initialization_fingerprints_by_shot_seed": fingerprint_audit,
        "initialization_consistent_across_variants_per_shot_seed": all(
            len(values) == 1 for values in fingerprint_audit.values()
        ),
        "full_gate_analysis": _full_gate_analysis(root),
        "soft_confusion_bank": bank_diagnostics,
        "methods": methods,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# From-scratch MT-TCP + Soft Confusion Prior 实验汇总",
        "",
        "全部数值均为 test 集三 seed 的 mean ± population std（百分数）。",
        "",
        f"- 运行总数：{len(VARIANTS) * len(SHOTS) * 3}",
        "- 每组训练：100 epochs；完整官方 val 选模；test 仅在训练完成后执行",
        "- 选模：Accuracy 与 Balanced Accuracy 两套 checkpoint 独立测试",
        "- 同一 `(shot, seed)` 的 B₀ 初始化在各 variants 间一致："
        f"{report['initialization_consistent_across_variants_per_shot_seed']}",
        "- 三个 seed 的 B₀ 初始化 fingerprints："
        + ", ".join(f"`{value}`" for value in report["all_b0_initialization_fingerprints"]),
        "",
    ]

    for selection in SELECTIONS:
        label = "ACC-selected" if selection == "accuracy" else "BACC-selected"
        lines.extend(
            [
                f"## {label}",
                "",
                "| Variant | Shot | Accuracy | BACC | Macro-F1 | Macro Recall | AUC |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for variant in VARIANTS:
            shots = report["methods"][variant]["shots"]
            for shot in SHOTS:
                metrics = shots[str(shot)][selection]["metrics"]
                lines.append(
                    "| {variant} | {shot} | {acc} | {bacc} | {f1} | {recall} | {auc} |".format(
                        variant=variant,
                        shot=shot,
                        acc=_fmt(metrics["accuracy"]),
                        bacc=_fmt(metrics["balanced_accuracy"]),
                        f1=_fmt(metrics["macro_f1"]),
                        recall=_fmt(metrics["macro_recall"]),
                        auc=_fmt(metrics["auc"]),
                    )
                )
        lines.append("")

    lines.extend(
        [
            "## 选模目标下的最佳 Variant",
            "",
            "| Selection | Shot | Best variant | Objective mean |",
            "|---|---:|---|---:|",
        ]
    )
    for selection in SELECTIONS:
        objective = selection
        for shot in SHOTS:
            candidates = {
                variant: report["methods"][variant]["shots"][str(shot)][selection][
                    "metrics"
                ][objective]["mean"]
                for variant in VARIANTS
            }
            best_variant = max(candidates, key=candidates.get)
            lines.append(
                f"| {selection} | {shot} | {best_variant} | {candidates[best_variant]:.2f} |"
            )

    lines.extend(
        [
            "",
            "## Full 相对 B₀",
            "",
            "| Selection | Shot | Δ Accuracy | Δ BACC | Δ Macro-F1 | Δ AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for selection in SELECTIONS:
        for shot in SHOTS:
            full_metrics = report["methods"]["full"]["shots"][str(shot)][selection][
                "metrics"
            ]
            b0_metrics = report["methods"]["b0"]["shots"][str(shot)][selection][
                "metrics"
            ]
            lines.append(
                "| {selection} | {shot} | {acc:+.2f} | {bacc:+.2f} | {f1:+.2f} | {auc:+.2f} |".format(
                    selection=selection,
                    shot=shot,
                    acc=full_metrics["accuracy"]["mean"] - b0_metrics["accuracy"]["mean"],
                    bacc=full_metrics["balanced_accuracy"]["mean"]
                    - b0_metrics["balanced_accuracy"]["mean"],
                    f1=full_metrics["macro_f1"]["mean"] - b0_metrics["macro_f1"]["mean"],
                    auc=full_metrics["auc"]["mean"] - b0_metrics["auc"]["mean"],
                )
            )

    lines.extend(
        [
            "",
            "## Full Test Gate 分布",
            "",
            "三个 seed 合并；每个 selection 共 6,015 个 test 样本。",
            "",
            "| Selection | Shot | α_global mean ± std | α_local mean | Global-dominant fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for selection in SELECTIONS:
        for shot in SHOTS:
            gate = report["full_gate_analysis"][str(shot)][selection]
            lines.append(
                "| {selection} | {shot} | {mean:.6f} ± {std:.6f} | {local:.6f} | {fraction:.2%} |".format(
                    selection=selection,
                    shot=shot,
                    mean=gate["alpha_global_mean"],
                    std=gate["alpha_global_std"],
                    local=gate["alpha_local_mean"],
                    fraction=gate["global_dominant_fraction"],
                )
            )

    lines.extend(
        [
            "",
            "## Soft Confusion Bank 诊断",
            "",
            "Bank 使用全部 K-shot support 的冻结 BiomedCLIP 概率均值；对角线清零后不再归一化。",
            "hard error 只作诊断，不参与当前 prior。",
            "",
            "| Shot | Seed | Support | Hard errors | Hard error rate | Soft prior off-diagonal mean | Top soft prior pairs | Checks |",
            "|---:|---:|---:|---:|---:|---:|---|---|",
        ]
    )
    for shot in SHOTS:
        for seed in (1, 2, 3):
            bank = report["soft_confusion_bank"][str(shot)][str(seed)]
            if bank.get("missing"):
                lines.append(f"| {shot} | {seed} | — | — | — | — | missing | — |")
                continue
            checks = bank.get("checks", {})
            check_text = "prior={} diag={} no-renorm={} support-only={} no-val/test={}".format(
                checks.get("prior_type_soft_probability_mean"),
                checks.get("diagonal_zero"),
                not checks.get("off_diagonal_renormalized", True),
                checks.get("support_only"),
                not checks.get("val_test_images_encoded", True),
            )
            lines.append(
                "| {shot} | {seed} | {support} | {errors} | {rate:.2%} | {mean:.4f} | {top} | {checks} |".format(
                    shot=shot,
                    seed=seed,
                    support=bank.get("support_record_count", "—"),
                    errors=bank.get("hard_error_total", "—"),
                    rate=bank.get("hard_error_rate") or 0.0,
                    mean=bank.get("soft_prior_off_diagonal_mean") or 0.0,
                    top=_top_soft_pair_text(bank.get("top_soft_pairs")),
                    checks=check_text,
                )
            )

    lines.extend(
        [
            "",
            "## Confusion pair 指标（validation/test）",
            "",
            "pair coverage = 真实标签落在所选 `(pair_first, pair_second)` 的比例；B0-top1-in-pair 是 `pair_first == y` 的比例。",
            "表中 pair 频次和 prior/score 均来自实际保存的逐样本 analysis，不使用真实标签选择 pair。",
            "",
            "| Selection | Variant | Shot | Test coverage | Val coverage | Test B0 top-1 | Test unique undirected | Selected prior | Selected score | Top-3 test pairs |",
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for selection in SELECTIONS:
        for variant in VARIANTS:
            confusion = report["methods"][variant]["confusion_metrics"]
            for shot in SHOTS:
                entry = confusion["shots"][str(shot)][selection]
                test = entry.get("test")
                validation = entry.get("validation")
                if not test:
                    lines.append(f"| {selection} | {variant} | {shot} | — | — | — | — | — | — | disabled |")
                    continue
                lines.append(
                    "| {selection} | {variant} | {shot} | {coverage} | {val} | {top1} | {unique} | {prior} | {score} | {pairs} |".format(
                        selection=selection,
                        variant=variant,
                        shot=shot,
                        coverage=_fmt_summary(test.get("pair_coverage_true_label"), 100.0, 2),
                        val=_fmt_summary(
                            validation.get("pair_coverage_true_label") if validation else None,
                            100.0,
                            2,
                        ),
                        top1=_fmt_summary(test.get("pair_first_matches_true"), 100.0, 2),
                        unique=_fmt_summary(test.get("pair_unique_undirected"), 1.0, 2),
                        prior=_fmt_summary(test.get("selected_prior_mean"), 1.0, 4),
                        score=_fmt_summary(test.get("selected_score_mean"), 1.0, 4),
                        pairs=_top_pair_text(test.get("top_pairs")),
                    )
                )

    lines.extend(
        [
            "",
            "## Confusion loss 与训练 pair 频次",
            "",
            "`L_confuse` 为训练日志中实际优化的 `softplus(z[h] - z[y])`；mean 是全训练日志均值，final 是 epoch 100 均值。",
            "",
            "| Selection | Variant | Shot | L_confuse mean | L_confuse final | Train selected epoch | Train selected top-3 pairs | Train epoch-100 top-3 pairs |",
            "|---|---|---:|---:|---:|---:|---|---|",
        ]
    )
    for selection in SELECTIONS:
        for variant in VARIANTS:
            confusion = report["methods"][variant]["confusion_metrics"]
            for shot in SHOTS:
                entry = confusion["shots"][str(shot)][selection]
                loss = entry.get("loss")
                loss_metric = loss.get("metrics", {}).get("loss_confuse") if loss else None
                selected = entry.get("train_selected")
                final_train = entry.get("train_final")
                epoch = entry.get("best_epoch")
                epoch_metric = epoch.get("best_epoch") if isinstance(epoch, dict) else epoch
                epoch_text = _fmt_summary(epoch_metric, 1.0, 1) if epoch_metric else "—"
                lines.append(
                    "| {selection} | {variant} | {shot} | {mean} | {final} | {epoch} | {selected} | {train_final} |".format(
                        selection=selection,
                        variant=variant,
                        shot=shot,
                        mean=_fmt_summary(loss_metric.get("mean") if loss_metric else None, 1.0, 4),
                        final=_fmt_summary(
                            loss_metric.get("final_epoch_mean") if loss_metric else None,
                            1.0,
                            4,
                        ),
                        epoch=epoch_text,
                        selected=_top_pair_text(selected.get("top_pairs") if selected else None),
                        train_final=_top_pair_text(
                            final_train.get("top_pairs") if final_train else None
                        ),
                    )
                )

    lines.extend(
        [
            "",
            "## Test confusion matrix 诊断",
            "",
            "每个方法/shot/selection 的 raw 与 row-normalized confusion matrix 已嵌入 `experiment_report.json` 的对应 `test_confusion_matrix_*` 字段（行=true，列=prediction），并保留逐 seed 矩阵文件。",
            "",
            "| Selection | Variant | Shot | Off-diagonal error rate | Largest normalized off-diagonal | Matrix seeds |",
            "|---|---|---:|---:|---|---:|",
        ]
    )
    for selection in SELECTIONS:
        for variant in VARIANTS:
            for shot in SHOTS:
                entry = report["methods"][variant]["confusion_metrics"]["shots"][str(shot)][selection]
                matrix = entry.get("test_confusion_matrix_raw")
                normalized = entry.get("test_confusion_matrix_normalized")
                if not matrix:
                    lines.append(f"| {selection} | {variant} | {shot} | — | disabled | 0 |")
                    continue
                raw_sum = matrix["sum"]
                total = sum(sum(float(value) for value in row) for row in raw_sum)
                diagonal = sum(float(raw_sum[i][i]) for i in range(min(len(raw_sum), len(raw_sum[0]))))
                error_rate = 1.0 - diagonal / total if total else 0.0
                norm_mean = normalized["mean"] if normalized else []
                largest = None
                largest_pair = None
                class_names = next(
                    (
                        report["soft_confusion_bank"][str(shot)][str(seed)].get(
                            "class_order", list(DEFAULT_CLASS_NAMES)
                        )
                        for seed in (1, 2, 3)
                        if not report["soft_confusion_bank"][str(shot)][str(seed)].get(
                            "missing"
                        )
                    ),
                    list(DEFAULT_CLASS_NAMES),
                )
                for first, row in enumerate(norm_mean):
                    for second, value in enumerate(row):
                        if first == second:
                            continue
                        if largest is None or value > largest:
                            largest = float(value)
                            largest_pair = f"{first}→{second}"
                if largest is not None and largest_pair is not None:
                    first, second = (int(item) for item in largest_pair.split("→"))
                    first_name = class_names[first] if first < len(class_names) else first
                    second_name = class_names[second] if second < len(class_names) else second
                    largest_text = f"{first_name}→{second_name} {largest:.2%}"
                else:
                    largest_text = "—"
                lines.append(
                    f"| {selection} | {variant} | {shot} | {error_rate:.2%} | {largest_text} | {matrix['seed_count']} |"
                )

    lines.extend(
        [
            "",
            "## 可训练参数",
            "",
            "| Variant | CoOp | Visual Deep Prompt | TextDeep + MT-TCP | Confusion-aware | Total trainable | Frozen |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for variant in VARIANTS:
        counts = report["methods"][variant]["audit"]["parameter_counts"]
        lines.append(
            "| {variant} | {coop:,} | {vpt:,} | {tcp:,} | {conf:,} | {trainable:,} | {frozen:,} |".format(
                variant=variant,
                coop=counts["coop"],
                vpt=counts["visual_deep_prompt"],
                tcp=counts["mt_tcp_including_text_deep"],
                conf=counts["confusion_aware"],
                trainable=counts["total_trainable"],
                frozen=counts["total_frozen"],
            )
        )

    lines.extend(
        [
            "",
            "完整逐 seed 指标、最佳 epoch、每类 recall、各指标 values/mean/std 和产物审计见同目录 `experiment_report.json`。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--bank-root",
        type=Path,
        help="Optional soft_confusion_banks root (default: sibling of --root)",
    )
    parser.add_argument("--json", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    bank_root = args.bank_root.resolve() if args.bank_root else None
    report = aggregate(root, bank_root=bank_root)
    json_path = (args.json or root / "experiment_report.json").resolve()
    markdown_path = (args.markdown or root / "experiment_report.md").resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    print(f"WROTE {json_path}")
    print(f"WROTE {markdown_path}")


if __name__ == "__main__":
    main()
