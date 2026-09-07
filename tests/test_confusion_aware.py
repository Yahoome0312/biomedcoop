import json

import pytest
import torch

from models.confusion_aware import (
    ConfusionAwareAdapter,
    build_frozen_pair_description_bank,
    confusion_margin_loss,
    select_confusion_pairs,
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
        logits, first
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
    pair_bank = torch.randn(3, 3, 512)
    adapter = ConfusionAwareAdapter(
        pair_bank, "b" * 64
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


def test_online_pairs_follow_current_logits_and_exclude_prediction_anchor():
    logits = torch.tensor([[3., 2., 1.], [3., 1., 2.]])
    first, second, probabilities, scores = select_confusion_pairs(logits, logits.argmax(-1))
    assert first.tolist() == [0, 0]
    assert second.tolist() == [1, 2]
    assert torch.isneginf(scores[:, 0]).all()
    assert torch.allclose(probabilities.sum(-1), torch.ones(2))


def test_online_adapter_checkpoint_has_no_offline_prior():
    adapter = ConfusionAwareAdapter(torch.randn(3, 3, 512), "b" * 64)
    state = adapter.state_dict()
    assert "soft_prior" not in state
    assert "_bank_fingerprint" not in state
    restored = ConfusionAwareAdapter(torch.randn(3, 3, 512), "b" * 64)
    restored.load_state_dict(state, strict=True)
