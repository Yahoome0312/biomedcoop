import copy

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel

from models.multitext_tcp import (
    AGGREGATION_MODES,
    CONNECTION_MODES,
    MultiTextTCPBertTextEncoder,
    build_frozen_description_bank,
    build_frozen_layer_description_bank,
)
from models.tcp import validate_tcp_checkpoint_state
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


def _adapter(
    aggregation="feature_mean",
    connection="late_residual",
    num_hidden_layers=3,
    gate_init=0.1,
):
    torch.manual_seed(31)
    tower = _TinyTextTower(num_hidden_layers=num_hidden_layers)
    classnames = ("zero", "one", "two")
    grouped_layer_modes = {
        "grouped10_layer_residual",
        "grouped10_layer_projected_hybrid",
        "grouped10_layer_projected_residual",
        "layer_cosine_set_hybrid",
        "layer_cosine_set_hybrid_light",
        "layer_cosine_set_residual",
    }
    description_count = 50 if aggregation in grouped_layer_modes else 10
    descriptions = {
        classname: tuple(
            "{} description {}".format(classname, index)
            for index in range(description_count)
        )
        for classname in classnames
    }
    if aggregation in grouped_layer_modes:
        bank = torch.randn(3, description_count, 32)
        projected_bank = torch.randn(3, description_count, 16)
        class_prior = torch.nn.functional.normalize(
            projected_bank.mean(dim=1), dim=-1
        )
        prior_representation = "layer_cls"
    else:
        bank = torch.randn(3, description_count, 16)
        projected_bank = None
        class_prior = None
        prior_representation = "projected_text"
    adapter = MultiTextTCPBertTextEncoder(
        tower,
        description_bank=bank,
        descriptions=descriptions,
        classnames=classnames,
        num_tokens=4,
        bottleneck_dim=8,
        insert_layer=1,
        aggregation=aggregation,
        connection=connection,
        gate_init=gate_init,
        class_prior=class_prior,
        prior_representation=prior_representation,
        projected_description_bank=projected_bank,
    )
    tokenized = torch.zeros(3, 16, dtype=torch.long)
    tokenized[:, :8] = torch.tensor([2, 10, 11, 12, 13, 20, 3, 21])
    prompts = tower.transformer.embeddings.word_embeddings(tokenized)
    return tower, adapter, prompts, tokenized


def test_zero_residual_warmstart_exactly_reproduces_text_deep_baseline():
    torch.manual_seed(47)
    baseline_tower = _TinyTextTower(num_hidden_layers=3)
    tcp_tower = copy.deepcopy(baseline_tower)
    baseline = BertTextDeepPromptEncoder(
        baseline_tower, num_tokens=4, dropout=0.0
    )
    classnames = ("zero", "one", "two")
    descriptions = {
        classname: tuple(
            "{} description {}".format(classname, index) for index in range(10)
        )
        for classname in classnames
    }
    tcp = MultiTextTCPBertTextEncoder(
        tcp_tower,
        description_bank=torch.randn(3, 10, 16),
        descriptions=descriptions,
        classnames=classnames,
        num_tokens=4,
        bottleneck_dim=8,
        insert_layer=1,
        aggregation="feature_mean",
        connection="late_centered_norm_residual",
        gate_init=0.1,
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
    baseline_prompts = baseline_tower.transformer.embeddings.word_embeddings(
        tokenized
    )
    tcp_prompts = tcp_tower.transformer.embeddings.word_embeddings(tokenized)

    baseline_output = baseline(baseline_prompts, tokenized)
    tcp_output = tcp(tcp_prompts, tokenized)
    assert torch.equal(baseline_output, tcp_output)


@pytest.mark.parametrize("aggregation", AGGREGATION_MODES)
def test_all_description_aggregators_are_permutation_invariant(aggregation):
    _, adapter, _, _ = _adapter(aggregation=aggregation)
    original = adapter.aggregate_class_tokens()
    if aggregation in {
        "grouped10_cosine_attention",
        "grouped10_layer_residual",
        "grouped10_layer_projected_hybrid",
        "grouped10_layer_projected_residual",
        "layer_cosine_set_hybrid",
        "layer_cosine_set_hybrid_light",
        "layer_cosine_set_residual",
    }:
        permutation = torch.cat(
            [torch.arange(end - 1, end - 11, -1) for end in range(10, adapter.description_count + 1, 10)]
        )
    else:
        permutation = torch.tensor([9, 4, 1, 7, 3, 8, 0, 6, 2, 5])
    adapter.description_bank = adapter.description_bank[:, permutation]
    adapter.projected_description_bank = adapter.projected_description_bank[:, permutation]
    permuted = adapter.aggregate_class_tokens()
    assert original.shape == (3, 4, 32)
    assert not torch.equal(original[0], original[1])
    assert torch.allclose(original, permuted, atol=1e-6, rtol=1e-5)
    assert adapter.description_bank.requires_grad is False
    assert "description_bank" not in adapter.state_dict()
    assert "class_prior" not in adapter.state_dict()


def test_cosine_set_attention_avoids_uniform_query_collapse():
    _, adapter, _, _ = _adapter(aggregation="cosine_set_attention")
    prompt = adapter.tcp_prompt
    bank = adapter.description_bank
    keys = prompt.key_projection(bank)
    queries = torch.nn.functional.normalize(prompt.set_queries.float(), dim=-1)
    keys = torch.nn.functional.normalize(keys.float(), dim=-1)
    score = torch.einsum("nd,ckd->cnk", queries, keys)
    score = score * prompt.bottleneck_dim**0.5
    attention = score.softmax(dim=-1)

    uniform = torch.full_like(attention, 1.0 / attention.shape[-1])
    normalized = torch.nn.functional.normalize(attention, dim=-1)
    pairwise = normalized @ normalized.transpose(-1, -2)
    off_diagonal = pairwise[:, ~torch.eye(4, dtype=torch.bool)]
    assert not torch.allclose(attention, uniform, atol=1e-3, rtol=1e-3)
    assert off_diagonal.mean() < 0.99


def test_grouped10_attention_averages_each_ten_description_block():
    _, adapter, _, _ = _adapter(aggregation="grouped10_cosine_attention")
    prompt = adapter.tcp_prompt
    assert prompt.description_count == 10
    output = prompt.aggregate_descriptions(adapter.description_bank)
    repeated = adapter.description_bank[:, :1].expand(-1, 10, -1).clone()
    repeated_output = prompt.aggregate_descriptions(repeated)
    assert output.shape == (3, 4, 32)
    assert not torch.allclose(output, repeated_output)


def test_grouped10_layer_residual_starts_from_frozen_layer_basis():
    _, adapter, _, _ = _adapter(aggregation="grouped10_layer_residual")
    prompt = adapter.tcp_prompt
    bank = adapter.description_bank
    groups = bank.reshape(3, 5, 10, 32).mean(dim=2)
    expected = torch.nn.functional.normalize(
        groups[:, :4] + groups.mean(dim=1, keepdim=True), dim=-1
    )

    output = prompt.aggregate_descriptions(bank)
    assert torch.allclose(output, expected, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(prompt.up_projection.weight) == 0
    assert bank.requires_grad is False

    target = torch.randn_like(output)
    (output * target).sum().backward()
    assert prompt.up_projection.weight.grad.norm() > 0


def test_grouped10_layer_projected_hybrid_uses_both_frozen_and_learned_spaces():
    _, adapter, _, _ = _adapter(
        aggregation="grouped10_layer_projected_hybrid"
    )
    prompt = adapter.tcp_prompt
    output = adapter.aggregate_class_tokens()
    basis = adapter.grouped_layer_targets()

    assert output.shape == basis.shape == (3, 4, 32)
    assert not torch.allclose(output, basis)
    changed_projected = adapter.projected_description_bank.clone()
    changed_projected[:, :10] += torch.randn_like(changed_projected[:, :10])
    adapter.projected_description_bank = changed_projected
    assert not torch.allclose(output, adapter.aggregate_class_tokens())

    adapter.description_bank = adapter.description_bank + torch.randn_like(
        adapter.description_bank
    )
    assert not torch.allclose(output, adapter.aggregate_class_tokens())
    adapter.aggregate_class_tokens().sum().backward()
    assert prompt.up_projection.weight.grad.norm() > 0


def test_grouped10_layer_projected_residual_has_exact_stable_start():
    _, adapter, _, _ = _adapter(
        aggregation="grouped10_layer_projected_residual"
    )
    prompt = adapter.tcp_prompt
    output = adapter.aggregate_class_tokens()
    basis = adapter.grouped_layer_targets()

    assert torch.allclose(output, basis, atol=1e-7, rtol=1e-6)
    assert torch.count_nonzero(prompt.up_projection.weight) == 0
    target = torch.randn_like(output)
    (output * target).sum().backward()
    assert prompt.up_projection.weight.grad.norm() > 0


def test_layer_cosine_set_hybrid_has_one_token_level_output_and_both_sources():
    _, adapter, _, _ = _adapter(aggregation="layer_cosine_set_hybrid")
    prompt = adapter.tcp_prompt
    output = adapter.aggregate_class_tokens()
    basis = adapter.grouped_layer_targets()

    assert output.shape == basis.shape == (3, 4, 32)
    assert not torch.allclose(output, basis)
    changed_projected = adapter.projected_description_bank.clone()
    changed_projected[:, :10] += torch.randn_like(changed_projected[:, :10])
    adapter.projected_description_bank = changed_projected
    assert not torch.allclose(output, adapter.aggregate_class_tokens())
    adapter.aggregate_class_tokens().sum().backward()
    assert prompt.set_queries.grad.norm() > 0
    assert prompt.output_projection.weight.grad.norm() > 0
    assert prompt.layer_cosine_mix_logit.grad.abs().item() > 0


def test_light_layer_cosine_hybrid_has_distinct_fingerprinted_mix():
    _, regular, _, _ = _adapter(aggregation="layer_cosine_set_hybrid")
    _, light, _, _ = _adapter(aggregation="layer_cosine_set_hybrid_light")
    regular_mix = regular.tcp_prompt.layer_cosine_mix_logit.sigmoid()
    light_mix = light.tcp_prompt.layer_cosine_mix_logit.sigmoid()
    assert torch.allclose(regular_mix, torch.tensor(0.1))
    assert torch.allclose(light_mix, torch.tensor(0.05))
    assert (
        regular.tcp_prompt.checkpoint_metadata()["aggregation_fingerprint"]
        != light.tcp_prompt.checkpoint_metadata()["aggregation_fingerprint"]
    )


def test_layer_cosine_set_residual_has_exact_layer_start_and_learnable_map():
    _, adapter, _, _ = _adapter(aggregation="layer_cosine_set_residual")
    prompt = adapter.tcp_prompt
    output = adapter.aggregate_class_tokens()
    basis = adapter.grouped_layer_targets()
    assert torch.equal(output, basis)
    assert torch.count_nonzero(prompt.output_projection.weight) == 0
    target = torch.randn_like(output)
    (output * target).sum().backward()
    assert prompt.output_projection.weight.grad.norm() > 0
    # Attention/value paths become active after the zero output map moves;
    # the exact-start batch intentionally trains only that final map.
    assert prompt.set_queries.grad is None or prompt.set_queries.grad.norm() == 0


def test_layer_targets_use_all_five_ten_description_groups():
    _, adapter, _, _ = _adapter(aggregation="set_attention")
    adapter.prior_representation = "layer_cls"
    # Tiny tests contain one ten-description group, so expand it to the exact
    # five-group protocol while preserving the hidden dimension.
    base = adapter.description_bank
    adapter.description_bank = torch.cat(
        [base + float(index) for index in range(5)], dim=1
    )
    adapter.description_count = 50
    adapter.tcp_prompt.description_count = 50
    targets = adapter.grouped_layer_targets()
    changed = adapter.description_bank.clone()
    changed[:, 40:] += torch.randn_like(changed[:, 40:]) * 5.0
    adapter.description_bank = changed
    changed_targets = adapter.grouped_layer_targets()
    assert targets.shape == (3, 4, 16)
    assert not torch.allclose(targets, changed_targets)


@pytest.mark.parametrize("connection", CONNECTION_MODES)
def test_all_connections_preserve_length_and_use_exact_slots(connection):
    tower, adapter, prompts, tokenized = _adapter(connection=connection)
    seen = []

    def capture(_module, args, kwargs):
        seen.append((args[0].detach().clone(), kwargs["attention_mask"].detach().clone()))

    hooks = [
        layer.register_forward_pre_hook(capture, with_kwargs=True)
        for layer in tower.transformer.encoder.layer
    ]
    try:
        output = adapter(prompts, tokenized)
    finally:
        for hook in hooks:
            hook.remove()

    class_tokens = adapter.tcp_prompt.aggregate_descriptions(
        adapter.description_bank
    ).detach()
    assert output.shape == (3, 16)
    expected_length = 16 if connection.startswith("inplace_") else 20
    assert [item[0].shape[1] for item in seen] == [expected_length] * 3
    assert torch.equal(seen[0][1], seen[1][1])
    assert torch.equal(seen[1][1], seen[2][1])
    if connection.startswith("inplace_"):
        assert seen[1][0][:, 1:5].shape == class_tokens.shape
    elif connection == "late_replace":
        assert torch.allclose(seen[1][0][:, 1:5], class_tokens)
    elif connection == "original_coop_replace":
        assert torch.allclose(seen[1][0][:, 5:9], class_tokens)
    else:
        expected = adapter.tcp_prompt.prompt_for_layer(
            1, class_tokens, seen[1][0].dtype, seen[1][0].device
        ).detach()
        assert torch.allclose(seen[1][0][:, 1:5], expected)


def test_inplace_zero_residual_is_exact_standard_coop_text_path():
    tower, adapter, prompts, tokenized = _adapter(
        aggregation="set_attention", connection="inplace_once_norm_residual"
    )
    tower.eval()
    adapter.eval()
    adapter.tcp_prompt.set_residual_scale(0.0)
    attention_mask = tokenized.ne(0).long()
    with torch.no_grad():
        expected_output = tower.transformer(
            inputs_embeds=prompts, attention_mask=attention_mask
        )
        expected = tower.proj(tower.pooler(expected_output, attention_mask))
        actual = adapter(prompts, tokenized)
    assert adapter.tcp_prompt.text_prompt is None
    assert actual.shape == expected.shape
    assert torch.equal(actual, expected)


def test_inplace_residual_is_norm_controlled_and_has_tke_gradients():
    _, adapter, prompts, tokenized = _adapter(
        aggregation="set_attention", connection="inplace_once_norm_residual"
    )
    hidden = torch.randn(3, 4, 32)
    tokens = adapter.tcp_prompt.aggregate_descriptions(adapter.description_bank)
    fused = adapter.tcp_prompt.inplace_context(1, hidden, tokens)
    alpha = adapter.tcp_prompt.gate_logits[0].sigmoid()
    assert torch.allclose(
        (fused - hidden).norm(dim=-1),
        alpha * hidden.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )

    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.tcp_prompt.parameters():
        parameter.requires_grad_(True)
    adapter(prompts, tokenized).square().sum().backward()
    assert all(
        parameter.grad is not None
        for parameter in adapter.tcp_prompt.parameters()
    )


def test_inplace_centered_residual_ignores_common_tke_direction():
    _, adapter, _, _ = _adapter(
        aggregation="set_attention",
        connection="inplace_once_centered_norm_residual",
    )
    hidden = torch.randn(3, 4, 32)
    tokens = adapter.tcp_prompt.aggregate_descriptions(adapter.description_bank)
    original = adapter.tcp_prompt.inplace_context(1, hidden, tokens)
    common = torch.randn(1, 4, 32) * 10.0
    shifted = adapter.tcp_prompt.inplace_context(1, hidden, tokens + common)
    assert torch.allclose(original, shifted, atol=5e-5, rtol=1e-5)


def test_inplace_class_gates_have_one_independent_strength_per_class():
    _, adapter, _, _ = _adapter(
        aggregation="set_attention",
        connection="inplace_once_centered_classgate_norm_residual",
    )
    hidden = torch.randn(3, 4, 32)
    tokens = adapter.tcp_prompt.aggregate_descriptions(adapter.description_bank)
    with torch.no_grad():
        adapter.tcp_prompt.gate_logits.copy_(torch.tensor([-2.0, 0.0, 2.0]))
    fused = adapter.tcp_prompt.inplace_context(1, hidden, tokens)
    expected = adapter.tcp_prompt.gate_logits.sigmoid().reshape(3, 1)
    assert tuple(adapter.tcp_prompt.gate_logits.shape) == (3,)
    assert torch.allclose(
        (fused - hidden).norm(dim=-1),
        expected * hidden.norm(dim=-1),
        atol=1e-5,
        rtol=1e-5,
    )


def test_inplace_deep_uses_one_gate_for_each_remaining_bert_block():
    _, adapter, _, _ = _adapter(
        aggregation="set_attention",
        connection="inplace_deep_centered_norm_residual",
    )
    hidden = torch.randn(3, 4, 32)
    tokens = adapter.tcp_prompt.aggregate_descriptions(adapter.description_bank)
    with torch.no_grad():
        adapter.tcp_prompt.gate_logits.copy_(torch.tensor([-2.0, 1.0]))

    assert tuple(adapter.tcp_prompt.gate_logits.shape) == (2,)
    assert adapter.tcp_prompt.inplace_context(0, hidden, tokens) is hidden
    for layer_idx, gate in ((1, 0), (2, 1)):
        fused = adapter.tcp_prompt.inplace_context(layer_idx, hidden, tokens)
        expected = adapter.tcp_prompt.gate_logits[gate].sigmoid()
        assert torch.allclose(
            (fused - hidden).norm(dim=-1),
            expected * hidden.norm(dim=-1),
            atol=1e-5,
            rtol=1e-5,
        )


def test_terminal_boost_gate_profile_is_depth_safe_and_late_weighted():
    _, adapter, _, _ = _adapter(
        aggregation="feature_mean",
        connection="inplace_deep_terminal_boost_centered_norm_residual",
    )
    gates = adapter.tcp_prompt.gate_logits.sigmoid()
    # The tiny test tower has two conditioned blocks.  The production tower
    # has four and follows the same construction, yielding
    # [.05, .10, .15, .25] at gate_init=.15.
    expected = torch.tensor([0.1, 0.1 * 5.0 / 3.0])
    assert torch.allclose(gates, expected, atol=1e-7, rtol=1e-6)
    assert gates[-1] > gates[0]


def test_terminal_peak_has_distinct_checkpoint_semantics_and_stronger_last_gate():
    _, boost, _, _ = _adapter(
        aggregation="feature_mean",
        connection="inplace_deep_terminal_boost_centered_norm_residual",
        num_hidden_layers=5,
        gate_init=0.15,
    )
    _, peak, _, _ = _adapter(
        aggregation="feature_mean",
        connection="inplace_deep_terminal_peak_centered_norm_residual",
        num_hidden_layers=5,
        gate_init=0.15,
    )
    boost_gates = boost.tcp_prompt.gate_logits.sigmoid()
    peak_gates = peak.tcp_prompt.gate_logits.sigmoid()

    assert torch.allclose(
        boost_gates, torch.tensor([0.05, 0.10, 0.15, 0.25]), atol=1e-7
    )
    assert torch.allclose(
        peak_gates, torch.tensor([0.05, 0.10, 0.15, 0.30]), atol=1e-7
    )
    assert (
        peak.tcp_prompt.checkpoint_metadata()["connection_fingerprint"]
        != boost.tcp_prompt.checkpoint_metadata()["connection_fingerprint"]
    )


def test_multitext_prompt_parameters_receive_gradients_but_backbone_does_not():
    _, adapter, prompts, tokenized = _adapter(
        aggregation="set_attention", connection="late_residual"
    )
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


def test_late_norm_residual_matches_shared_token_scale():
    _, adapter, _, _ = _adapter(
        aggregation="set_attention", connection="late_norm_residual"
    )
    class_tokens = adapter.tcp_prompt.aggregate_descriptions(
        adapter.description_bank
    )
    layer_idx = adapter.tcp_prompt.insert_layer
    shared = adapter.tcp_prompt.text_prompt.for_layer(
        layer_idx, class_tokens.shape[0], class_tokens.dtype, class_tokens.device
    )
    fused = adapter.tcp_prompt.prompt_for_layer(
        layer_idx, class_tokens, class_tokens.dtype, class_tokens.device
    )
    alpha = adapter.tcp_prompt.gate_logits[0].sigmoid()
    residual = fused - shared
    expected_norm = alpha * shared.norm(dim=-1)
    assert torch.allclose(residual.norm(dim=-1), expected_norm, atol=1e-6)

    scaled = adapter.tcp_prompt.prompt_for_layer(
        layer_idx, class_tokens * 100.0, class_tokens.dtype, class_tokens.device
    )
    assert torch.allclose(fused, scaled, atol=1e-6, rtol=1e-5)

    adapter.tcp_prompt.set_residual_scale(0.25)
    quarter = adapter.tcp_prompt.prompt_for_layer(
        layer_idx, class_tokens, class_tokens.dtype, class_tokens.device
    )
    assert torch.allclose(quarter - shared, 0.25 * residual, atol=1e-6)
    with pytest.raises(ValueError, match="residual_scale"):
        adapter.tcp_prompt.set_residual_scale(1.1)


def test_centered_norm_residual_ignores_common_class_direction():
    _, adapter, _, _ = _adapter(
        aggregation="set_attention", connection="late_centered_norm_residual"
    )
    class_tokens = adapter.tcp_prompt.aggregate_descriptions(
        adapter.description_bank
    )
    layer_idx = adapter.tcp_prompt.insert_layer
    fused = adapter.tcp_prompt.prompt_for_layer(
        layer_idx, class_tokens, class_tokens.dtype, class_tokens.device
    )
    common = torch.randn_like(class_tokens[:1]) * 10.0
    shifted = adapter.tcp_prompt.prompt_for_layer(
        layer_idx,
        class_tokens + common,
        class_tokens.dtype,
        class_tokens.device,
    )
    assert torch.allclose(fused, shifted, atol=2e-5, rtol=1e-5)


def test_late_classlayer_gates_start_equal_then_adapt_one_class_only():
    _, shared_gate, _, _ = _adapter(
        aggregation="set_attention", connection="late_centered_norm_residual"
    )
    _, class_gate, _, _ = _adapter(
        aggregation="set_attention",
        connection="late_centered_classlayer_norm_residual",
    )
    layer_idx = class_gate.tcp_prompt.insert_layer
    tokens = class_gate.tcp_prompt.aggregate_descriptions(
        class_gate.description_bank
    )
    shared = shared_gate.tcp_prompt.prompt_for_layer(
        layer_idx, tokens, tokens.dtype, tokens.device
    )
    initial = class_gate.tcp_prompt.prompt_for_layer(
        layer_idx, tokens, tokens.dtype, tokens.device
    )
    assert class_gate.tcp_prompt.gate_logits.shape == (2, 3)
    assert torch.allclose(initial, shared, atol=1e-7, rtol=1e-6)

    with torch.no_grad():
        class_gate.tcp_prompt.gate_logits[0, 0].add_(1.0)
    changed = class_gate.tcp_prompt.prompt_for_layer(
        layer_idx, tokens, tokens.dtype, tokens.device
    )
    assert not torch.allclose(changed[0], initial[0])
    assert torch.equal(changed[1:], initial[1:])
    changed.sum().backward()
    assert class_gate.tcp_prompt.gate_logits.grad.norm() > 0


def test_multitext_checkpoint_validates_description_and_method_identity():
    _, adapter, _, _ = _adapter()
    state = {
        "tcp." + key: value for key, value in adapter.tcp_prompt.state_dict().items()
    }
    validate_tcp_checkpoint_state(state, adapter.tcp_prompt)
    for suffix in (
        "_meta_description_count",
        "_meta_description_fingerprint",
        "_meta_aggregation_fingerprint",
        "_meta_connection_fingerprint",
    ):
        bad = copy.deepcopy(state)
        key = "tcp." + suffix
        if bad[key].numel() == 1:
            bad[key] = bad[key] + 1
        else:
            bad[key] = torch.tensor(
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


def test_description_bank_requires_exact_nonduplicate_count():
    model = _DescriptionModel()
    tokenizer = lambda text: torch.tensor([[len(text), 1]], dtype=torch.long)
    descriptions = {
        "zero": ("zero a", "zero b"),
        "one": ("one a", "one b"),
    }
    bank, ordered = build_frozen_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        expected_count=2,
        batch_size=1,
    )
    assert bank.shape == (2, 2, 3)
    assert tuple(ordered) == ("zero", "one")
    with pytest.raises(ValueError, match="Expected 3 descriptions"):
        build_frozen_description_bank(
            model,
            tokenizer,
            ("zero", "one"),
            descriptions,
            expected_count=3,
        )


def test_description_bank_cache_is_reused_and_identity_checked(tmp_path):
    model = _DescriptionModel()
    tokenizer = lambda text: torch.tensor([[len(text), 1]], dtype=torch.long)
    descriptions = {
        "zero": ("zero a", "zero b"),
        "one": ("one a", "one b"),
    }
    cache = tmp_path / "bank.pt"
    first, _ = build_frozen_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        expected_count=2,
        batch_size=1,
        cache_path=cache,
        model_id="tiny",
    )
    first_calls = model.encode_calls
    second, _ = build_frozen_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        expected_count=2,
        batch_size=1,
        cache_path=cache,
        model_id="tiny",
    )
    assert first_calls > 0
    assert model.encode_calls == first_calls
    assert torch.equal(first, second)

    changed = dict(descriptions)
    changed["one"] = ("one a", "one changed")
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        build_frozen_description_bank(
            model,
            tokenizer,
            ("zero", "one"),
            changed,
            expected_count=2,
            cache_path=cache,
            model_id="tiny",
        )


def test_layer_aligned_description_bank_matches_injection_space_and_cache(tmp_path):
    class _FullModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.text = _TinyTextTower()

    model = _FullModel().eval()
    tokenizer = lambda text: torch.tensor(
        [[2, 10 + len(text) % 10, 11, 3] + [0] * 12], dtype=torch.long
    )
    descriptions = {
        "zero": ("zero a", "zero b"),
        "one": ("one a", "one b"),
    }
    cache = tmp_path / "layer_bank.pt"
    bank, ordered = build_frozen_layer_description_bank(
        model,
        tokenizer,
        ("zero", "one"),
        descriptions,
        insert_layer=1,
        expected_count=2,
        batch_size=2,
        cache_path=cache,
        model_id="tiny",
    )
    assert bank.shape == (2, 2, 32)
    assert tuple(ordered) == ("zero", "one")
    assert torch.allclose(bank.norm(dim=-1), torch.ones(2, 2), atol=1e-6)
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
    assert torch.equal(bank, cached)
    with pytest.raises(RuntimeError, match="metadata mismatch"):
        build_frozen_layer_description_bank(
            model,
            tokenizer,
            ("zero", "one"),
            descriptions,
            insert_layer=2,
            expected_count=2,
            cache_path=cache,
            model_id="tiny",
        )
