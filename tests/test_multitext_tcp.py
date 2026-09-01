import copy

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel

from models.multitext_tcp import (
    FINAL_AGGREGATION,
    FINAL_CONNECTION,
    MultiTextTCPBertTextEncoder,
    build_frozen_description_bank,
    build_frozen_layer_description_bank,
    validate_tcp_checkpoint_state,
)
from models.text_vpt import BertTextDeepPromptEncoder


class _Pooler(nn.Module):
    def forward(self, output, attention_mask):
        return output.pooler_output


class _TinyTextTower(nn.Module):
    def __init__(self, num_hidden_layers=3):
        super().__init__()
        config = BertConfig(
            vocab_size=101,
            hidden_size=32,
            num_hidden_layers=num_hidden_layers,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        self.transformer = BertModel(config, add_pooling_layer=True)
        self.pooler = _Pooler()
        self.proj = nn.Linear(32, 16, bias=False)
        self.output_tokens = False


def _descriptions(classnames, count=50):
    return {
        classname: tuple(
            "{} description {}".format(classname, index) for index in range(count)
        )
        for classname in classnames
    }


def _adapter(num_hidden_layers=3, gate_init=0.05):
    torch.manual_seed(31)
    tower = _TinyTextTower(num_hidden_layers=num_hidden_layers)
    classnames = ("zero", "one", "two")
    adapter = MultiTextTCPBertTextEncoder(
        tower,
        layer_description_bank=torch.randn(3, 50, 32),
        projected_description_bank=torch.randn(3, 50, 16),
        descriptions=_descriptions(classnames),
        classnames=classnames,
        num_tokens=4,
        bottleneck_dim=8,
        insert_layer=1,
        gate_init=gate_init,
        model_id="tiny",
    )
    tokenized = torch.zeros(3, 16, dtype=torch.long)
    tokenized[:, :8] = torch.tensor([2, 10, 11, 12, 13, 20, 3, 21])
    prompts = tower.transformer.embeddings.word_embeddings(tokenized)
    return tower, adapter, prompts, tokenized


def test_only_reported_architecture_is_exposed():
    _, adapter, _, _ = _adapter()
    assert adapter.aggregation == FINAL_AGGREGATION == "grouped10_layer_residual"
    assert adapter.connection == FINAL_CONNECTION == "late_centered_norm_residual"
    assert adapter.description_count == 50
    assert adapter.num_tokens == 4


def test_zero_residual_exactly_reproduces_textdeep_baseline():
    torch.manual_seed(47)
    baseline_tower = _TinyTextTower(num_hidden_layers=3)
    tcp_tower = copy.deepcopy(baseline_tower)
    baseline = BertTextDeepPromptEncoder(
        baseline_tower, num_tokens=4, dropout=0.0
    )
    classnames = ("zero", "one", "two")
    tcp = MultiTextTCPBertTextEncoder(
        tcp_tower,
        layer_description_bank=torch.randn(3, 50, 32),
        projected_description_bank=torch.randn(3, 50, 16),
        descriptions=_descriptions(classnames),
        classnames=classnames,
        num_tokens=4,
        bottleneck_dim=8,
        insert_layer=1,
        gate_init=0.05,
        model_id="tiny",
    )
    with torch.no_grad():
        tcp.tcp_prompt.text_prompt.prompt_embeddings.copy_(
            baseline.text_prompt.prompt_embeddings
        )
    tcp.tcp_prompt.set_residual_scale(0.0)
    baseline.eval()
    tcp.eval()

    tokenized = torch.zeros(3, 16, dtype=torch.long)
    tokenized[:, :8] = torch.tensor([2, 10, 11, 12, 13, 20, 3, 21])
    baseline_prompts = baseline_tower.transformer.embeddings.word_embeddings(tokenized)
    tcp_prompts = tcp_tower.transformer.embeddings.word_embeddings(tokenized)
    assert torch.equal(
        baseline(baseline_prompts, tokenized), tcp(tcp_prompts, tokenized)
    )


def test_layerbasis_starts_exactly_from_five_groups_and_trains_residual():
    _, adapter, _, _ = _adapter()
    bank = adapter.description_bank
    groups = bank.reshape(3, 5, 10, 32).mean(dim=2)
    expected = torch.nn.functional.normalize(
        groups[:, :4] + groups.mean(dim=1, keepdim=True), dim=-1
    )
    output = adapter.aggregate_class_tokens()
    assert torch.allclose(output, expected, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(adapter.tcp_prompt.up_projection.weight) == 0
    (output * torch.randn_like(output)).sum().backward()
    assert adapter.tcp_prompt.up_projection.weight.grad.norm() > 0


def test_layerbasis_is_invariant_within_each_ordered_group():
    _, adapter, _, _ = _adapter()
    original = adapter.aggregate_class_tokens()
    permutation = torch.cat(
        [torch.arange(end - 1, end - 11, -1) for end in range(10, 51, 10)]
    )
    adapter.description_bank = adapter.description_bank[:, permutation]
    assert torch.allclose(original, adapter.aggregate_class_tokens(), atol=1e-6)
    assert "description_bank" not in adapter.state_dict()
    assert "class_prior" not in adapter.state_dict()


def test_centered_norm_residual_is_scale_controlled_and_common_invariant():
    _, adapter, _, _ = _adapter()
    tokens = adapter.aggregate_class_tokens()
    layer = adapter.insert_layer
    shared = adapter.tcp_prompt.text_prompt.for_layer(
        layer, tokens.shape[0], tokens.dtype, tokens.device
    )
    fused = adapter.tcp_prompt.prompt_for_layer(
        layer, tokens, tokens.dtype, tokens.device
    )
    alpha = adapter.tcp_prompt.gate_logits[0].sigmoid()
    assert torch.allclose(
        (fused - shared).norm(dim=-1), alpha * shared.norm(dim=-1), atol=1e-6
    )
    common = torch.randn_like(tokens[:1]) * 10
    shifted = adapter.tcp_prompt.prompt_for_layer(
        layer, tokens + common, tokens.dtype, tokens.device
    )
    assert torch.allclose(fused, shifted, atol=2e-5, rtol=1e-5)

    adapter.tcp_prompt.set_residual_scale(0.25)
    quarter = adapter.tcp_prompt.prompt_for_layer(
        layer, tokens, tokens.dtype, tokens.device
    )
    assert torch.allclose(quarter - shared, 0.25 * (fused - shared), atol=1e-6)
    with pytest.raises(ValueError, match="residual_scale"):
        adapter.tcp_prompt.set_residual_scale(1.1)


def test_prompt_parameters_receive_gradients_but_backbone_does_not():
    _, adapter, prompts, tokenized = _adapter()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.tcp_prompt.parameters():
        parameter.requires_grad_(True)
    adapter(prompts, tokenized).square().sum().backward()
    prompt_ids = {id(parameter) for parameter in adapter.tcp_prompt.parameters()}
    assert all(parameter.grad is not None for parameter in adapter.tcp_prompt.parameters())
    assert all(
        parameter.grad is None
        for parameter in adapter.parameters()
        if id(parameter) not in prompt_ids
    )


def test_tcp_off_trains_only_existing_text_vpt_tokens():
    _, adapter, prompts, tokenized = _adapter()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.tcp_prompt.text_prompt.parameters():
        parameter.requires_grad_(True)
    adapter.tcp_prompt.set_residual_scale(0.0)

    adapter(prompts, tokenized).square().sum().backward()

    assert adapter.tcp_prompt.text_prompt.prompt_embeddings.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in adapter.tcp_prompt.named_parameters()
        if not name.startswith("text_prompt.")
    )


def test_checkpoint_validates_final_method_identity():
    _, adapter, _, _ = _adapter()
    state = {
        "tcp." + key: value for key, value in adapter.tcp_prompt.state_dict().items()
    }
    validate_tcp_checkpoint_state(state, adapter.tcp_prompt)
    bad = copy.deepcopy(state)
    bad["tcp._meta_connection_fingerprint"] = torch.tensor(
        list(("0" * 64).encode("ascii")), dtype=torch.uint8
    )
    with pytest.raises(RuntimeError, match="mismatch"):
        validate_tcp_checkpoint_state(bad, adapter.tcp_prompt)


class _DescriptionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.encode_calls = 0

    def encode_text(self, tokenized, normalize=True):
        self.encode_calls += 1
        values = tokenized.float().sum(dim=1, keepdim=True)
        features = torch.cat((values, values + 1, values + 2), dim=1)
        return torch.nn.functional.normalize(features, dim=-1)


def test_projected_description_bank_validates_count_and_cache(tmp_path):
    model = _DescriptionModel()
    tokenizer = lambda text: torch.tensor([[len(text), 1]], dtype=torch.long)
    descriptions = {"zero": ("zero a", "zero b"), "one": ("one a", "one b")}
    cache = tmp_path / "bank.pt"
    bank, ordered = build_frozen_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        expected_count=2,
        batch_size=1,
        cache_path=cache,
        model_id="tiny",
    )
    calls = model.encode_calls
    cached, _ = build_frozen_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        expected_count=2,
        cache_path=cache,
        model_id="tiny",
    )
    assert tuple(ordered) == ("zero", "one")
    assert torch.equal(bank, cached)
    assert model.encode_calls == calls
    with pytest.raises(ValueError, match="Expected 3 descriptions"):
        build_frozen_description_bank(
            model, tokenizer, ("zero", "one"), descriptions, expected_count=3
        )


def test_layer_description_bank_matches_injection_space_and_cache(tmp_path):
    class _FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = _TinyTextTower()

    model = _FullModel().eval()
    tokenizer = lambda text: torch.tensor(
        [[2, 10 + len(text) % 10, 11, 3] + [0] * 12], dtype=torch.long
    )
    descriptions = {"zero": ("zero a", "zero b"), "one": ("one a", "one b")}
    cache = tmp_path / "layer_bank.pt"
    bank, ordered = build_frozen_layer_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        insert_layer=1,
        expected_count=2,
        cache_path=cache,
        model_id="tiny",
    )
    cached, _ = build_frozen_layer_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        insert_layer=1,
        expected_count=2,
        cache_path=cache,
        model_id="tiny",
    )
    assert bank.shape == (2, 2, 32)
    assert tuple(ordered) == ("zero", "one")
    assert torch.equal(bank, cached)
    assert torch.allclose(bank.norm(dim=-1), torch.ones(2, 2), atol=1e-6)
