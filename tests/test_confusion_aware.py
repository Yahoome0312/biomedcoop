import json
from types import SimpleNamespace

import pytest
import torch

from scripts.coopvpt import build_confusion_prior
from models.confusion_aware import (
    ConfusionAwareAdapter,
    build_frozen_pair_description_bank,
    compute_hard_confusion_counts,
    compute_soft_confusion_prior,
    confusion_margin_loss,
    load_soft_confusion_bank,
    select_confusion_pairs,
    support_fingerprint,
)


class _PairTextEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))

    def encode_text(self, tokens, normalize=True):
        tokens = tokens.float()
        features = torch.stack(
            (tokens[:, 0], tokens[:, 1], tokens.sum(dim=1), tokens[:, 0] - tokens[:, 1]),
            dim=1,
        )
        return torch.nn.functional.normalize(features, dim=-1) if normalize else features


def _pair_tokenizer(text):
    return torch.tensor([[len(text), sum(text.encode("utf-8")) % 97]], dtype=torch.long)


def test_soft_prior_uses_every_support_probability_without_renormalizing():
    probabilities = torch.tensor(
        [[0.60, 0.30, 0.10], [0.50, 0.42, 0.08],
         [0.70, 0.20, 0.10], [0.50, 0.38, 0.12]]
    )
    prior, counts = compute_soft_confusion_prior(
        probabilities, torch.zeros(4, dtype=torch.long), 3
    )
    assert counts.tolist() == [4, 0, 0]
    assert prior[0, 1].item() == pytest.approx(0.325)
    assert prior[0, 2].item() == pytest.approx(0.10)
    assert prior[0].sum().item() == pytest.approx(0.425)
    assert torch.equal(torch.diag(prior), torch.zeros(3))


def test_hard_count_is_diagnostic_only():
    probabilities = torch.tensor(
        [[0.6, 0.4], [0.7, 0.3], [0.4, 0.6], [0.8, 0.2]]
    )
    labels = torch.zeros(4, dtype=torch.long)
    prior, _ = compute_soft_confusion_prior(probabilities, labels, 2)
    hard = compute_hard_confusion_counts(probabilities, labels, 2)
    assert hard[0, 1].item() == 1
    assert prior[0, 1].item() == pytest.approx(0.375)


def test_bank_builder_uses_requested_dataset_config(monkeypatch, tmp_path):
    requested_config = tmp_path / "kvasir.yaml"
    captured = {}

    def _capture_setup_cfg(namespace):
        captured["namespace"] = namespace
        return namespace

    monkeypatch.setattr(build_confusion_prior.train, "setup_cfg", _capture_setup_cfg)
    args = SimpleNamespace(
        data_root=tmp_path / "data",
        output_root=tmp_path / "output",
        dataset_config_file=requested_config,
    )

    cfg = build_confusion_prior._cfg(args, shots=4, seed=1)

    assert cfg is captured["namespace"]
    assert cfg.dataset_config_file == str(requested_config)
    assert cfg.opts == ["DATASET.NUM_SHOTS", "4"]
    assert cfg.seed == 1


def test_llm_pair_bank_supports_dynamic_classes_and_description_counts(tmp_path):
    classnames = ["class c", "class_a", "class b"]
    payload = {
        "class a": {
            "class b": ["a to b one", "a to b two"],
            "class c": ["a to c"],
        },
        "class b": {
            "class a": ["b to a"],
            "class c": ["b to c"],
        },
        "class c": {
            "class a": ["c to a"],
            "class b": ["c to b"],
        },
    }
    path = tmp_path / "dataset.txt"
    path.write_text(json.dumps(payload), encoding="utf-8")

    bank, metadata = build_frozen_pair_description_bank(
        _PairTextEncoder(), _pair_tokenizer, classnames, path, batch_size=2
    )

    assert bank.shape == (3, 3, 4)
    assert torch.equal(bank.diagonal(dim1=0, dim2=1), torch.zeros(4, 3))
    assert metadata["pair_count"] == 6
    assert metadata["description_count"] == 7
    assert len(metadata["description_fingerprint"]) == 64
    assert len(metadata["feature_fingerprint"]) == 64


def test_llm_pair_bank_rejects_incomplete_pairs(tmp_path):
    path = tmp_path / "dataset.txt"
    path.write_text(
        json.dumps({"a": {"b": ["a to b"]}, "b": {}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="do not match the other dataset classes"):
        build_frozen_pair_description_bank(
            _PairTextEncoder(), _pair_tokenizer, ["a", "b"], path
        )


def test_pair_selection_uses_explicit_anchor_and_detaches_probabilities():
    logits = torch.tensor([[1.0, 3.0, 2.0]], requires_grad=True)
    first = torch.tensor([0])
    first, second, probabilities, _ = select_confusion_pairs(
        logits, torch.zeros(3, 3), first, prior_alpha=1.0
    )
    assert first.tolist() == [0]
    assert second.tolist() == [1]
    assert not probabilities.requires_grad


def test_confusion_margin_compares_true_class_with_selected_negative():
    logits = torch.zeros(3, 4, requires_grad=True)
    labels = torch.tensor([1, 2, 3])
    competitor = torch.tensor([2, 1, 1])
    loss = confusion_margin_loss(logits, labels, competitor)
    assert loss.item() == pytest.approx(torch.log(torch.tensor(2.0)).item())
    loss.backward()
    assert logits.grad is not None


def test_full_confusion_uses_ground_truth_anchor_and_backpropagates():
    torch.manual_seed(1)
    prior = torch.zeros(3, 3)
    prior[0, 1] = 0.325
    pair_bank = torch.randn(3, 3, 512)
    adapter = ConfusionAwareAdapter(
        prior, "a" * 64, pair_bank, "b" * 64
    )
    global_features = torch.randn(2, 512, requires_grad=True)
    patches = torch.randn(2, 196, 768, requires_grad=True)
    text = torch.randn(3, 512, requires_grad=True)
    base_logits = torch.tensor(
        [[0.1, 3.0, 0.2], [2.0, 0.3, 0.1]], requires_grad=True
    )
    labels = torch.tensor([0, 2])
    logits, details = adapter(
        global_features,
        patches,
        text,
        base_logits,
        torch.tensor(10.0),
        labels,
    )
    assert logits.shape == (2, 3)
    assert torch.isfinite(logits).all()
    assert details["pair_first"].tolist() == labels.tolist()
    assert details["pair_second"].ne(labels).all()
    logits.sum().backward()
    assert text.grad is not None
    assert global_features.grad is not None
    assert patches.grad is not None


def test_bank_loader_rejects_support_fingerprint_mismatch(tmp_path):
    import hashlib

    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    item = SimpleNamespace(impath=str(image), label=0, classname="a")
    prior = torch.tensor([[0.0]])
    fingerprint = hashlib.sha256(prior.numpy().tobytes()).hexdigest()
    path = tmp_path / "bank.pt"
    torch.save(
        {
            "metadata": {
                "schema_version": 1,
                "prior_type": "soft_probability_mean",
                "dataset": "DermaMNIST",
                "shots": 1,
                "seed": 1,
                "class_order": ["a"],
                "support_fingerprint": support_fingerprint([item]),
                "bank_fingerprint": fingerprint,
            },
            "soft_prior": prior,
        },
        path,
    )
    load_soft_confusion_bank(
        path, dataset_name="DermaMNIST", shots=1, seed=1,
        classnames=["a"], support_items=[item]
    )
    other = SimpleNamespace(impath=str(tmp_path / "b.png"), label=0, classname="a")
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        load_soft_confusion_bank(
            path, dataset_name="DermaMNIST", shots=1, seed=1,
            classnames=["a"], support_items=[other]
        )
