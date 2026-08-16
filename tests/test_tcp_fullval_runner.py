import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.coopvpt.run_tcp_fullval import (
    _aggregate_runs,
    _baseline_checkpoint,
    _complete,
    _evaluate_existing_selections,
    _require_validation_gate,
)


def test_baseline_checkpoint_supports_dual_and_legacy_selection_layouts(tmp_path):
    current = tmp_path / "prompt_parameters" / "model-best-balanced_accuracy.pth.tar"
    current.parent.mkdir()
    current.write_bytes(b"current")
    assert _baseline_checkpoint(tmp_path, "balanced_accuracy") == current

    current.unlink()
    legacy = (
        tmp_path
        / "selection_balanced_accuracy"
        / "prompt_parameters"
        / "model-best.pth.tar"
    )
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    assert _baseline_checkpoint(tmp_path, "balanced_accuracy") == legacy

    with pytest.raises(FileNotFoundError, match="Missing baseline accuracy"):
        _baseline_checkpoint(tmp_path, "accuracy")


class _FakeTrainer:
    def __init__(self):
        self.loaded = []
        self.test_calls = 0
        self.last_eval_results = {}

    def load_prompt_checkpoint(self, path):
        self.loaded.append(Path(path).name)
        return {"epoch": 3 if "accuracy.pth" in str(path) else 7}

    def test(self, split):
        assert split == "test"
        self.test_calls += 1
        offset = float(self.test_calls)
        self.last_eval_results = {
            "accuracy": 70.0 + offset,
            "balanced_accuracy": 60.0 + offset,
            "auc": 80.0 + offset,
            "macro_f1": 50.0 + offset,
        }


def _write_json(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_evaluate_existing_reuses_both_checkpoints_without_training(tmp_path):
    prompt_dir = tmp_path / "prompt_parameters"
    prompt_dir.mkdir()
    for metric, epoch, value in (
        ("accuracy", 3, 71.0),
        ("balanced_accuracy", 7, 62.0),
    ):
        (prompt_dir / f"model-best-{metric}.pth.tar").write_bytes(metric.encode())
        _write_json(
            tmp_path / f"best_validation_{metric}.json",
            {
                "epoch": epoch,
                "selection_value": value,
                "metrics": {
                    "accuracy": 71.0,
                    "balanced_accuracy": 62.0,
                    "auc": 81.0,
                    "macro_f1": 51.0,
                },
            },
        )

    trainer = _FakeTrainer()
    args = Namespace(method="method", shots=16, seed=2, epochs=100)
    _evaluate_existing_selections(
        trainer, args, tmp_path, training_reused=True
    )

    assert trainer.test_calls == 2
    assert trainer.loaded == [
        "model-best-accuracy.pth.tar",
        "model-best-balanced_accuracy.pth.tar",
    ]
    results = json.loads((tmp_path / "results.json").read_text())
    assert results["training_reused"] is True
    assert set(results["selections"]) == {"accuracy", "balanced_accuracy"}
    complete = json.loads((tmp_path / "run_complete.json").read_text())
    assert complete["evaluated_existing_checkpoints_only"] is True
    for metric in ("accuracy", "balanced_accuracy"):
        copied = (
            tmp_path
            / f"selection_{metric}"
            / "prompt_parameters"
            / "model-best.pth.tar"
        )
        assert copied.exists()


def test_evaluate_existing_requires_passing_full_grid_validation_gate(tmp_path):
    method = "single_model_tcp"
    summary_path = tmp_path / "screen_validation_summary.json"
    summary = {
        "protocol": {
            "method": method,
            "shots": [4, 8, 16, 32],
            "seeds": [1, 2, 3],
            "test_evaluated": False,
        },
        "overall": {"effectiveness": {"effective": False}},
    }
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    args = Namespace(
        worker=False,
        run_dir=None,
        output_root=tmp_path,
        summary_prefix="screen",
        method=method,
        shots=[4, 8, 16, 32],
        seeds=[1, 2, 3],
    )

    with pytest.raises(RuntimeError, match="did not pass"):
        _require_validation_gate(args)

    summary["overall"]["effectiveness"]["effective"] = True
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    assert _require_validation_gate(args) == summary_path

    args.seeds = [1, 2]
    with pytest.raises(RuntimeError, match="grid mismatch"):
        _require_validation_gate(args)

    run_dir = tmp_path / method / "shots_4" / "seed1"
    run_dir.mkdir(parents=True)
    args.worker = True
    args.run_dir = run_dir
    # Workers validate the same root-level PASS record; the parent process is
    # responsible for the exact full-grid comparison above.
    assert _require_validation_gate(args) == summary_path


def test_global_effectiveness_counts_all_paired_runs():
    def run(offset):
        values = {
            "accuracy": 60.0 + offset,
            "balanced_accuracy": 55.0 + offset,
            "auc": 80.0 + offset,
            "macro_f1": 50.0 + offset,
        }
        return {"accuracy": dict(values), "balanced_accuracy": dict(values)}

    baseline = [run(0.0) for _ in range(12)]
    tcp = [run(1.1) for _ in range(9)] + [run(-0.1) for _ in range(3)]
    summary = _aggregate_runs(
        tcp, baseline, [f"run{index}" for index in range(12)], "validation"
    )

    verdict = summary["effectiveness"]
    assert verdict["evaluated_runs"] == 12
    assert verdict["required_seed_wins"] == 8
    assert verdict["bacc_seed_wins"] == 9
    assert verdict["effective"] is False  # mean gain is below the frozen 1 pp rule


def test_effectiveness_is_audited_independently_for_both_selections():
    def run(accuracy_offset, bacc_offset):
        base = {
            "accuracy": 60.0,
            "balanced_accuracy": 55.0,
            "auc": 80.0,
            "macro_f1": 50.0,
        }
        return {
            "accuracy": {
                key: value + accuracy_offset for key, value in base.items()
            },
            "balanced_accuracy": {
                key: value + bacc_offset for key, value in base.items()
            },
        }

    baseline = [run(0.0, 0.0) for _ in range(12)]
    tcp = [run(2.0, -0.1) for _ in range(12)]
    summary = _aggregate_runs(
        tcp, baseline, [f"run{index}" for index in range(12)]
    )

    by_selection = summary["effectiveness_by_selection"]
    assert by_selection["accuracy"]["effective"] is True
    assert by_selection["accuracy"]["bacc_seed_wins"] == 12
    assert by_selection["balanced_accuracy"]["effective"] is False
    assert by_selection["balanced_accuracy"]["bacc_seed_wins"] == 0
    assert summary["effectiveness"] == by_selection["balanced_accuracy"]


def test_validation_completion_requires_matching_epoch_budget(tmp_path):
    prompt_dir = tmp_path / "prompt_parameters"
    prompt_dir.mkdir()
    for metric in ("accuracy", "balanced_accuracy"):
        (prompt_dir / f"model-best-{metric}.pth.tar").write_bytes(b"checkpoint")
        _write_json(tmp_path / f"best_validation_{metric}.json", {"epoch": 7})
    _write_json(
        tmp_path / "validation_search_complete.json",
        {"status": "complete", "epochs": 30, "test_evaluated": False},
    )

    assert _complete(tmp_path, validation_only=True)
    assert _complete(tmp_path, validation_only=True, required_epochs=30)
    assert not _complete(tmp_path, validation_only=True, required_epochs=100)
