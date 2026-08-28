import json
from argparse import Namespace
from pathlib import Path

import pytest

from scripts.coopvpt.run_tcp_fullval import (
    _complete,
    _evaluate_existing_selections,
    _require_validation_grid,
)


class _FakeTrainer:
    def __init__(self):
        self.loaded = []
        self.last_eval_results = {}
        self._analysis_tag = None

    def load_prompt_checkpoint(self, path):
        self.loaded.append(Path(path).name)
        return {"epoch": 3 if "accuracy.pth" in str(path) else 7}

    def test(self, split):
        assert split == "test"
        offset = float(len(self.loaded))
        self.last_eval_results = {
            "accuracy": 70 + offset,
            "balanced_accuracy": 60 + offset,
            "macro_f1": 50 + offset,
            "macro_recall": 60 + offset,
            "auc": 80 + offset,
        }


def _write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def test_evaluate_existing_uses_both_selection_checkpoints(tmp_path):
    prompt_dir = tmp_path / "prompt_parameters"
    prompt_dir.mkdir()
    for metric in ("accuracy", "balanced_accuracy"):
        (prompt_dir / f"model-best-{metric}.pth.tar").write_bytes(b"x")
        _write(
            tmp_path / f"best_validation_{metric}.json",
            {"epoch": 2, "selection_value": 70, "metrics": {"accuracy": 70}},
        )
    args = Namespace(method="m", variant="b0", shots=4, seed=1, epochs=100)
    trainer = _FakeTrainer()
    _evaluate_existing_selections(trainer, args, tmp_path)
    assert trainer.loaded == [
        "model-best-accuracy.pth.tar",
        "model-best-balanced_accuracy.pth.tar",
    ]
    result = json.loads((tmp_path / "results.json").read_text())
    assert set(result["selections"]) == {"accuracy", "balanced_accuracy"}


def test_validation_grid_is_required_before_test(tmp_path):
    args = Namespace(
        output_root=tmp_path, method="m", variant="full",
        shots=[4, 8], seeds=[1, 2], epochs=100
    )
    with pytest.raises(RuntimeError, match="requires"):
        _require_validation_grid(args)
    marker = tmp_path / "m" / "validation_grid_complete.json"
    marker.parent.mkdir()
    _write(
        marker,
        {"method": "m", "variant": "full", "shots": [4, 8],
         "seeds": [1, 2], "epochs": 100, "test_evaluated": False},
    )
    assert _require_validation_grid(args) == marker


def test_completion_requires_both_best_checkpoints(tmp_path):
    _write(tmp_path / "validation_search_complete.json", {"epochs": 100})
    for metric in ("accuracy", "balanced_accuracy"):
        _write(tmp_path / f"best_validation_{metric}.json", {"epoch": 1})
        directory = tmp_path / "prompt_parameters"
        directory.mkdir(exist_ok=True)
        (directory / f"model-best-{metric}.pth.tar").write_bytes(b"x")
    assert _complete(tmp_path, validation_only=True, required_epochs=100)
    (tmp_path / "prompt_parameters" / "model-best-balanced_accuracy.pth.tar").unlink()
    assert not _complete(tmp_path, validation_only=True, required_epochs=100)
