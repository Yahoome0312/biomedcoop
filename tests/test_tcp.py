import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from transformers import BertConfig, BertModel

from dassl.utils import resume_from_checkpoint, save_checkpoint
from models.tcp import TCPBertTextEncoder, validate_tcp_checkpoint_state
from trainers.CoOp.coop_vpt_biomedclip import (
    CoOpVPT_BiomedCLIP,
    PromptParameterBundle,
    base_prompts_frozen,
    class_balanced_hard_negative_margin_loss,
    class_balanced_cross_modal_prototype_loss,
    description_distillation_loss,
    image_description_prior_loss,
    prior_contrastive_loss,
    layer_token_alignment_loss,
    select_description_teacher_prototypes,
    tcp_knowledge_loss,
)


class _Pooler(nn.Module):
    def forward(self, output, attention_mask):
        return output.pooler_output


def test_base_prompt_settling_phase_uses_zero_based_epoch_boundary():
    assert base_prompts_frozen(0, 5)
    assert base_prompts_frozen(4, 5)
    assert not base_prompts_frozen(5, 5)
    assert not base_prompts_frozen(0, 0)
    with pytest.raises(ValueError):
        base_prompts_frozen(-1, 5)


class _TinyTextTower(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(
            vocab_size=101,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=16,
            pad_token_id=0,
        )
        self.transformer = BertModel(config, add_pooling_layer=True)
        self.pooler = _Pooler()
        self.proj = nn.Linear(32, 16, bias=False)
        self.output_tokens = False


def _adapter_and_inputs():
    torch.manual_seed(11)
    tower = _TinyTextTower()
    prior = torch.randn(3, 16)
    adapter = TCPBertTextEncoder(
        tower,
        class_prior=prior,
        classnames=("zero", "one", "two"),
        num_tokens=4,
        bottleneck_dim=8,
        insert_layer=1,
    )
    tokenized = torch.zeros(3, 16, dtype=torch.long)
    tokenized[:, :8] = torch.tensor([2, 10, 11, 12, 13, 20, 3, 21])
    prompts = tower.transformer.embeddings.word_embeddings(tokenized)
    return tower, adapter, prompts, tokenized


def test_tke_shape_class_difference_and_fixed_prior():
    _, adapter, _, _ = _adapter_and_inputs()
    tke = adapter.tcp_prompt(adapter.class_prior)
    assert tke.shape == (3, 4, 32)
    assert not torch.equal(tke[0], tke[1])
    assert adapter.class_prior.requires_grad is False
    assert "class_prior" not in adapter.state_dict()


def test_tcp_replaces_exact_context_before_selected_block_without_length_change():
    tower, adapter, prompts, tokenized = _adapter_and_inputs()
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

    expected = adapter.tcp_prompt(adapter.class_prior).detach()
    assert output.shape == (3, 16)
    assert [value[0].shape[1] for value in seen] == [16, 16, 16]
    assert torch.equal(seen[0][1], seen[1][1])
    assert torch.equal(seen[1][1], seen[2][1])
    assert torch.allclose(seen[1][0][:, 1:5, :], expected, atol=0, rtol=0)


def test_gated_residual_preserves_context_and_learns_bounded_alpha():
    tower, _, prompts, tokenized = _adapter_and_inputs()
    prior = torch.randn(3, 16)
    adapter = TCPBertTextEncoder(
        tower,
        class_prior=prior,
        classnames=("zero", "one", "two"),
        num_tokens=4,
        bottleneck_dim=8,
        insert_layer=1,
        fusion_mode="gated_residual",
        fusion_weight=0.25,
    )
    hidden = torch.randn(3, 16, 32)
    tcp = adapter.tcp_prompt(adapter.class_prior)
    fused = adapter.replace_context(hidden, tcp)
    alpha = adapter.tcp_prompt.fusion_alpha()
    expected = hidden[:, 1:5, :] + alpha * (tcp - hidden[:, 1:5, :])
    assert 0.0 < alpha.item() < 1.0
    assert alpha.item() == pytest.approx(0.25)
    assert torch.allclose(fused[:, 1:5, :], expected)
    fused.sum().backward()
    assert adapter.tcp_prompt.fusion_logit.grad is not None


def test_only_tke_receives_gradient_and_all_tke_projections_are_used():
    _, adapter, prompts, tokenized = _adapter_and_inputs()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    for parameter in adapter.tcp_prompt.parameters():
        parameter.requires_grad_(True)

    adapter(prompts, tokenized).square().sum().backward()
    tke_ids = {id(parameter) for parameter in adapter.tcp_prompt.parameters()}
    assert all(parameter.grad is not None for parameter in adapter.tcp_prompt.parameters())
    assert all(
        parameter.grad is None
        for parameter in adapter.parameters()
        if id(parameter) not in tke_ids
    )


def test_checkpoint_validation_rejects_plain_or_incompatible_bundle():
    _, adapter, _, _ = _adapter_and_inputs()
    state = {"tcp." + key: value for key, value in adapter.tcp_prompt.state_dict().items()}
    validate_tcp_checkpoint_state(state, adapter.tcp_prompt)

    with pytest.raises(RuntimeError, match="not a complete TCP"):
        validate_tcp_checkpoint_state({}, adapter.tcp_prompt)

    corruptions = {
        "tcp._meta_num_tokens": torch.tensor(5),
        "tcp._meta_hidden_dim": torch.tensor(64),
        "tcp._meta_category_order_fingerprint": torch.tensor(
            list(("0" * 64).encode("ascii")), dtype=torch.uint8
        ),
        "tcp._meta_template_fingerprint": torch.tensor(
            list(("1" * 64).encode("ascii")), dtype=torch.uint8
        ),
    }
    for key, value in corruptions.items():
        bad_state = copy.deepcopy(state)
        bad_state[key] = value
        with pytest.raises(RuntimeError, match="mismatch"):
            validate_tcp_checkpoint_state(bad_state, adapter.tcp_prompt)


def test_tcp_disabled_training_loss_is_the_original_ce_path():
    class _LogitModel(nn.Module):
        def forward(self, image):
            return image

    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.tcp_enabled = False
    trainer.model = _LogitModel()
    logits = torch.tensor([[1.0, 2.0], [3.0, -1.0]])
    labels = torch.tensor([1, 0])
    (
        output,
        loss,
        loss_ce,
        loss_kg,
        loss_anchor,
        loss_kd,
        loss_image_prior,
        loss_prior_contrastive,
        loss_layer_token_alignment,
        loss_cross_modal_proto,
        loss_hard_negative_margin,
    ) = trainer._compute_training_loss(logits, labels)
    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.equal(output, logits)
    assert torch.equal(loss, expected)
    assert torch.equal(loss_ce, expected)
    assert loss_kg.item() == 0.0
    assert loss_anchor.item() == 0.0
    assert loss_kd.item() == 0.0
    assert loss_image_prior.item() == 0.0
    assert loss_prior_contrastive.item() == 0.0
    assert loss_layer_token_alignment.item() == 0.0
    assert loss_cross_modal_proto.item() == 0.0
    assert loss_hard_negative_margin.item() == 0.0


def test_cross_modal_prototype_loss_is_class_balanced_and_bidirectional():
    torch.manual_seed(101)
    text = torch.nn.functional.normalize(torch.randn(4, 12), dim=-1)
    labels = torch.tensor([0, 0, 0, 1, 2, 2])
    matching = torch.stack(
        [text[0], text[0], text[0], text[1], text[2], text[2]]
    ).clone().requires_grad_(True)
    shuffled = matching.detach().roll(1, dims=0).requires_grad_(True)

    matching_loss = class_balanced_cross_modal_prototype_loss(
        matching, text, labels, temperature=0.1
    )
    shuffled_loss = class_balanced_cross_modal_prototype_loss(
        shuffled, text, labels, temperature=0.1
    )
    assert matching_loss < shuffled_loss
    matching_loss.backward()
    assert matching.grad is not None
    assert matching.grad.norm().item() > 0.0
    with pytest.raises(ValueError, match="temperature"):
        class_balanced_cross_modal_prototype_loss(matching, text, labels, 0.0)


def test_hard_negative_margin_prefers_separated_matching_features():
    text = torch.eye(4, dtype=torch.float32)
    labels = torch.tensor([0, 0, 1, 2, 2, 2])
    matching = text[labels].clone().requires_grad_(True)
    confused = (0.55 * text[labels] + 0.45 * text[(labels + 1) % 4]).requires_grad_(
        True
    )
    matching_loss = class_balanced_hard_negative_margin_loss(
        matching, text, labels, margin=0.05, temperature=0.02
    )
    confused_loss = class_balanced_hard_negative_margin_loss(
        confused, text, labels, margin=0.05, temperature=0.02
    )
    assert matching_loss < confused_loss
    confused_loss.backward()
    assert confused.grad is not None
    assert confused.grad.norm().item() > 0.0
    with pytest.raises(ValueError, match="temperature"):
        class_balanced_hard_negative_margin_loss(
            matching, text, labels, temperature=0.0
        )


def test_prior_contrastive_loss_uses_fixed_class_negatives():
    torch.manual_seed(71)
    prior = torch.randn(5, 12)
    matching = prior.clone().requires_grad_(True)
    shuffled = prior.roll(1, dims=0).requires_grad_(True)

    matching_loss = prior_contrastive_loss(matching, prior, temperature=0.1)
    shuffled_loss = prior_contrastive_loss(shuffled, prior, temperature=0.1)
    assert matching_loss < shuffled_loss
    matching_loss.backward()
    assert matching.grad is not None
    assert prior.grad is None
    with pytest.raises(ValueError, match="temperature"):
        prior_contrastive_loss(matching, prior, temperature=0.0)


def test_layer_token_alignment_is_centered_and_target_is_fixed():
    torch.manual_seed(73)
    targets = torch.randn(5, 4, 12)
    learned = targets.clone().requires_grad_(True)
    common = torch.randn(1, 4, 12)
    base = layer_token_alignment_loss(learned, targets)
    shifted = layer_token_alignment_loss(learned + common, targets)
    assert base.item() == pytest.approx(0.0, abs=1e-6)
    assert torch.allclose(base, shifted, atol=1e-6)
    base.backward()
    assert learned.grad is not None
    assert targets.grad is None


def test_prompt_bundle_resume_restores_tke_optimizer_and_scheduler(tmp_path):
    _, adapter, _, _ = _adapter_and_inputs()
    bundle = PromptParameterBundle(
        nn.Linear(2, 2),
        visual_prompt=nn.Linear(2, 2),
        tcp_prompt=adapter.tcp_prompt,
    )
    optimizer = torch.optim.AdamW(bundle.parameters(), lr=0.01)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
    loss = sum(parameter.square().sum() for parameter in bundle.parameters())
    loss.backward()
    optimizer.step()
    scheduler.step()
    expected_parameters = {
        key: value.detach().clone() for key, value in bundle.state_dict().items()
    }
    expected_lr = optimizer.param_groups[0]["lr"]
    expected_scheduler_epoch = scheduler.last_epoch
    save_checkpoint(
        {
            "state_dict": bundle.state_dict(),
            "epoch": 1,
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
        },
        str(tmp_path),
    )

    with torch.no_grad():
        for parameter in bundle.parameters():
            parameter.zero_()
    optimizer.param_groups[0]["lr"] = 0.123
    scheduler.last_epoch = 99
    start_epoch = resume_from_checkpoint(
        str(tmp_path), bundle, optimizer, scheduler
    )

    assert start_epoch == 1
    assert optimizer.param_groups[0]["lr"] == expected_lr
    assert scheduler.last_epoch == expected_scheduler_epoch
    for key, value in bundle.state_dict().items():
        assert torch.equal(value, expected_parameters[key])


class _PromptLearnerForInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Parameter(torch.zeros(4, 3))


class _VisualPromptForInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.zeros(2, 4, 3))


class _TextPromptForInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_embeddings = nn.Parameter(torch.zeros(3, 4, 3))


class _TCPPromptWithInternalTextForInit(nn.Module):
    def __init__(self):
        super().__init__()
        self.text_prompt = _TextPromptForInit()
        self.tke_weight = nn.Parameter(torch.zeros(2, 3))


class _PromptAnchorModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = _PromptLearnerForInit()
        self.image_encoder = nn.Module()
        self.image_encoder.visual_prompt = _VisualPromptForInit()


def test_prompt_anchor_is_zero_at_warmstart_and_penalizes_directional_drift():
    torch.manual_seed(9)
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.model = _PromptAnchorModel()
    trainer.cfg = SimpleNamespace(
        TRAINER=SimpleNamespace(
            TCP=SimpleNamespace(PROMPT_ANCHOR_L2_WEIGHT=0.0)
        )
    )
    with torch.no_grad():
        trainer.model.prompt_learner.ctx.normal_()
        trainer.model.image_encoder.visual_prompt.prompt_embeddings.normal_()
    trainer._prompt_anchor = (
        trainer.model.prompt_learner.ctx.detach().clone(),
        trainer.model.image_encoder.visual_prompt.prompt_embeddings.detach().clone(),
    )
    assert trainer._prompt_anchor_loss().item() == pytest.approx(0.0, abs=1e-6)

    with torch.no_grad():
        trainer.model.prompt_learner.ctx[0].add_(torch.tensor([1.0, -2.0, 0.5]))
        trainer.model.image_encoder.visual_prompt.prompt_embeddings[0, 0].add_(
            torch.tensor([-0.5, 1.0, 2.0])
        )
    loss = trainer._prompt_anchor_loss()
    assert loss.item() > 0.0
    loss.backward()
    assert trainer.model.prompt_learner.ctx.grad is not None
    assert (
        trainer.model.image_encoder.visual_prompt.prompt_embeddings.grad is not None
    )


def test_centered_knowledge_loss_ignores_common_semantic_direction():
    torch.manual_seed(12)
    text_features = torch.randn(7, 16, requires_grad=True)
    prior = torch.randn(7, 16)
    common = torch.randn(1, 16) * 20.0
    original = tcp_knowledge_loss(
        text_features, prior, mode="centered_cosine"
    )
    shifted = tcp_knowledge_loss(
        text_features + common, prior, mode="centered_cosine"
    )
    assert torch.allclose(original, shifted, atol=2e-6, rtol=1e-5)
    assert not torch.allclose(
        tcp_knowledge_loss(text_features, prior, mode="raw_cosine"),
        tcp_knowledge_loss(text_features + common, prior, mode="raw_cosine"),
    )
    original.backward()
    assert text_features.grad is not None
    with pytest.raises(ValueError, match="knowledge-loss"):
        tcp_knowledge_loss(text_features, prior, mode="unknown")


def test_description_teacher_selection_is_permutation_invariant_and_kd_trains():
    torch.manual_seed(22)
    image_features = torch.nn.functional.normalize(torch.randn(4, 8), dim=-1)
    bank = torch.nn.functional.normalize(torch.randn(3, 6, 8), dim=-1)
    prototypes, mask = select_description_teacher_prototypes(
        image_features, bank, torch.tensor(10.0), tau=1.5
    )
    permutation = torch.tensor([4, 1, 5, 0, 3, 2])
    permuted, permuted_mask = select_description_teacher_prototypes(
        image_features, bank[:, permutation], torch.tensor(10.0), tau=1.5
    )
    assert prototypes.shape == (3, 8)
    assert mask.shape == (6,)
    assert mask.any()
    assert torch.equal(mask[permutation], permuted_mask)
    assert torch.allclose(prototypes, permuted, atol=1e-6, rtol=1e-5)

    student = torch.randn(4, 3, requires_grad=True)
    teacher = torch.randn(4, 3)
    loss = description_distillation_loss(student, teacher, temperature=1.5)
    assert loss.item() > 0.0
    loss.backward()
    assert student.grad is not None
    assert description_distillation_loss(teacher, teacher).item() == pytest.approx(
        0.0, abs=1e-6
    )


def test_image_description_prior_loss_updates_only_image_features():
    torch.manual_seed(31)
    images = torch.randn(5, 8, requires_grad=True)
    prior = torch.randn(3, 8, requires_grad=True)
    labels = torch.tensor([0, 1, 2, 1, 0])
    loss = image_description_prior_loss(
        images, prior, labels, torch.tensor(10.0)
    )
    assert loss.item() > 0.0
    loss.backward()
    assert images.grad is not None
    assert images.grad.norm().item() > 0.0
    assert prior.grad is None


def test_baseline_initialization_copies_only_coop_and_visual_vpt():
    _, adapter, _, _ = _adapter_and_inputs()
    bundle = PromptParameterBundle(
        _PromptLearnerForInit(),
        visual_prompt=_VisualPromptForInit(),
        tcp_prompt=adapter.tcp_prompt,
    )
    tcp_before = {
        key: value.clone()
        for key, value in bundle.tcp.state_dict().items()
    }
    baseline = {
        "prompt_learner.ctx": torch.full((4, 3), 2.0),
        "prompt_learner.token_prefix": torch.randn(1),
        "prompt_learner.token_suffix": torch.randn(1),
        "visual_prompt.prompt_embeddings": torch.full((2, 4, 3), 3.0),
    }
    CoOpVPT_BiomedCLIP._initialize_bundle_from_baseline_state(bundle, baseline)
    assert torch.equal(bundle.prompt_learner.ctx, baseline["prompt_learner.ctx"])
    assert torch.equal(
        bundle.visual_prompt.prompt_embeddings,
        baseline["visual_prompt.prompt_embeddings"],
    )
    for key, value in bundle.tcp.state_dict().items():
        assert torch.equal(value, tcp_before[key])

    bad = dict(baseline)
    bad["tcp.gate_logits"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match="non-TCP"):
        CoOpVPT_BiomedCLIP._initialize_bundle_from_baseline_state(bundle, bad)


def test_textprompt_baseline_initialization_maps_into_internal_tcp_prompt():
    tcp = _TCPPromptWithInternalTextForInit()
    bundle = PromptParameterBundle(
        _PromptLearnerForInit(),
        visual_prompt=_VisualPromptForInit(),
        tcp_prompt=tcp,
    )
    tke_before = tcp.tke_weight.detach().clone()
    baseline = {
        "prompt_learner.ctx": torch.full((4, 3), 2.0),
        "visual_prompt.prompt_embeddings": torch.full((2, 4, 3), 3.0),
        "text_prompt.prompt_embeddings": torch.full((3, 4, 3), 4.0),
    }

    values = CoOpVPT_BiomedCLIP._validated_baseline_prompt_tensors(
        bundle, baseline
    )
    assert len(values) == 3
    CoOpVPT_BiomedCLIP._initialize_bundle_from_baseline_state(bundle, baseline)

    assert torch.equal(bundle.prompt_learner.ctx, baseline["prompt_learner.ctx"])
    assert torch.equal(
        bundle.visual_prompt.prompt_embeddings,
        baseline["visual_prompt.prompt_embeddings"],
    )
    assert torch.equal(
        bundle.tcp.text_prompt.prompt_embeddings,
        baseline["text_prompt.prompt_embeddings"],
    )
    assert torch.equal(bundle.tcp.tke_weight, tke_before)


def test_textprompt_and_plain_baseline_kinds_cannot_be_cross_loaded():
    plain_bundle = PromptParameterBundle(
        _PromptLearnerForInit(),
        visual_prompt=_VisualPromptForInit(),
        tcp_prompt=_adapter_and_inputs()[1].tcp_prompt,
    )
    text_bundle = PromptParameterBundle(
        _PromptLearnerForInit(),
        visual_prompt=_VisualPromptForInit(),
        tcp_prompt=_TCPPromptWithInternalTextForInit(),
    )
    plain = {
        "prompt_learner.ctx": torch.zeros(4, 3),
        "visual_prompt.prompt_embeddings": torch.zeros(2, 4, 3),
    }
    with_text = dict(plain)
    with_text["text_prompt.prompt_embeddings"] = torch.zeros(3, 4, 3)

    with pytest.raises(RuntimeError, match="TextPrompt baseline"):
        CoOpVPT_BiomedCLIP._validated_baseline_prompt_tensors(text_bundle, plain)
    with pytest.raises(RuntimeError, match="plain CoOp\\+VPT"):
        CoOpVPT_BiomedCLIP._validated_baseline_prompt_tensors(
            plain_bundle, with_text
        )
