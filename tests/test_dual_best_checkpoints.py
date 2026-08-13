import json
from pathlib import Path
from types import SimpleNamespace

from dassl.engine.trainer import SimpleTrainer


class _DualBestTrainer(SimpleTrainer):
    def __init__(self, output_dir, validation_results):
        self.output_dir = str(output_dir)
        self.epoch = 0
        self.max_epoch = 10
        self.val_loader = object()
        self.best_result = float("-inf")
        self.best_results = {}
        self.best_validation_records = {}
        self.validation_results = iter(validation_results)
        self.validation_calls = 0
        self.saved_checkpoints = []
        self.cfg = SimpleNamespace(
            TEST=SimpleNamespace(
                NO_TEST=False,
                FINAL_MODEL="last_step",
                BEST_METRIC="accuracy",
                SAVE_BEST_METRICS=("accuracy", "balanced_accuracy"),
            ),
            TRAIN=SimpleNamespace(CHECKPOINT_FREQ=0),
        )

    def test(self, split=None):
        assert split == "val"
        self.validation_calls += 1
        self.last_eval_results = next(self.validation_results)
        return self.last_eval_results[self.cfg.TEST.BEST_METRIC]

    def get_model_names(self, names=None):
        return ["prompt_parameters"]

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        self.saved_checkpoints.append(model_name)
        model_dir = Path(directory) / "prompt_parameters"
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / model_name).write_bytes(b"checkpoint")


def test_one_validation_pass_tracks_accuracy_and_balanced_accuracy(tmp_path):
    trainer = _DualBestTrainer(
        tmp_path,
        [
            {
                "accuracy": 70.0,
                "balanced_accuracy": 60.0,
                "auc": 81.0,
                "macro_f1": 55.0,
            },
            {
                "accuracy": 69.0,
                "balanced_accuracy": 65.0,
                "auc": 82.0,
                "macro_f1": 57.0,
            },
        ],
    )

    trainer.after_epoch()
    trainer.epoch = 1
    trainer.after_epoch()

    assert trainer.validation_calls == 2
    assert trainer.saved_checkpoints == [
        "model-best-accuracy.pth.tar",
        "model-best-balanced_accuracy.pth.tar",
        "model-best-balanced_accuracy.pth.tar",
    ]
    assert (
        tmp_path / "prompt_parameters" / "model-best.pth.tar"
    ).read_bytes() == b"checkpoint"

    accuracy_record = json.loads(
        (tmp_path / "best_validation_accuracy.json").read_text(encoding="utf-8")
    )
    balanced_record = json.loads(
        (tmp_path / "best_validation_balanced_accuracy.json").read_text(
            encoding="utf-8"
        )
    )
    combined = json.loads(
        (tmp_path / "best_validation_all.json").read_text(encoding="utf-8")
    )
    legacy = json.loads(
        (tmp_path / "best_validation.json").read_text(encoding="utf-8")
    )

    assert accuracy_record["epoch"] == 1
    assert accuracy_record["selection_value"] == 70.0
    assert accuracy_record["metrics"]["balanced_accuracy"] == 60.0
    assert accuracy_record["metrics"]["auc"] == 81.0
    assert balanced_record["epoch"] == 2
    assert balanced_record["selection_value"] == 65.0
    assert balanced_record["metrics"]["accuracy"] == 69.0
    assert balanced_record["metrics"]["macro_f1"] == 57.0
    assert set(combined) == {"accuracy", "balanced_accuracy"}
    assert legacy == accuracy_record
