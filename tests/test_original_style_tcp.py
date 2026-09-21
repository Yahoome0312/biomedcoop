import copy

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from transformers import BertConfig, BertModel

from models.original_style_tcp import (
    OriginalStyleTCPBertTextEncoder,
    OriginalStyleTCPPromptParameters,
    validate_tcp_checkpoint_state,
)


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


def _original_encoder(num_hidden_layers=12):
    torch.manual_seed(71)
    tower = _TinyTextTower(num_hidden_layers=num_hidden_layers)
    tower.requires_grad_(False)
    bank = F.normalize(torch.randn(3, 50, 16), dim=-1).requires_grad_()
    encoder = OriginalStyleTCPBertTextEncoder(
        tower, bank, ["a", "b", "c"]
    ).eval()
    ids = torch.zeros(3, 16, dtype=torch.long)
    ids[:, :8] = torch.tensor([2, 10, 11, 12, 13, 20, 3, 21])
    prompts = tower.transformer.embeddings.word_embeddings(ids).detach()
    prompts.requires_grad_()
    return encoder, bank, prompts, ids


def test_mean50_order_independence_and_shared_tke():
    encoder, bank, _, _ = _original_encoder()
    torch.testing.assert_close(encoder.class_prior, F.normalize(bank.mean(1), dim=-1))
    shuffled = bank.detach()[:, torch.randperm(50)]
    other = OriginalStyleTCPBertTextEncoder(
        copy.deepcopy(encoder.base_text_encoder), shuffled, ["a", "b", "c"]
    )
    other.tcp_prompt.load_state_dict(encoder.tcp_prompt.state_dict())
    torch.testing.assert_close(other.class_prior, encoder.class_prior)
    tokens = encoder.aggregate_class_tokens()
    torch.testing.assert_close(other.aggregate_class_tokens(), tokens)
    assert tokens.shape == (3, 4, 32)
    assert not torch.allclose(tokens[0], tokens[1])
    assert not encoder.description_bank.requires_grad
    assert not encoder.class_prior.requires_grad
    assert set(dict(encoder.tcp_prompt.named_parameters())) == {
        "text_prompt.prompt_embeddings",
        "down_projection.weight",
        "down_projection.bias",
        "up_projection.weight",
        "up_projection.bias",
    }


def test_fusion_preserves_initialization_and_uses_fixed_half_weights():
    torch.manual_seed(9)
    original = OriginalStyleTCPPromptParameters(16, 32, 12)
    original_rng = torch.random.get_rng_state()
    torch.manual_seed(9)
    fused = OriginalStyleTCPPromptParameters(16, 32, 12, fusion=True)
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for name, value in original.state_dict().items():
        torch.testing.assert_close(value, fused.state_dict()[name])
    tokens = fused.class_tokens(torch.randn(3, 16))
    result = fused.prompt_for_layer(8, tokens, torch.float32, "cpu")
    torch.testing.assert_close(result, 0.5 * tokens + 0.5 * fused.fusion_prompt)
    result.square().sum().backward()
    assert fused.fusion_prompt.grad.norm() > 0
    assert fused.up_projection.weight.grad.norm() > 0
    restored = OriginalStyleTCPPromptParameters(16, 32, 12, fusion=True)
    restored.load_state_dict(fused.state_dict(), strict=True)
    torch.testing.assert_close(restored.fusion_prompt, fused.fusion_prompt)


@pytest.mark.parametrize("fusion", [False, True])
def test_once_at_insert_layer_and_natural_propagation_with_gradients(fusion):
    encoder, bank, prompts, ids = _original_encoder()
    if fusion:
        encoder.tcp_prompt = OriginalStyleTCPPromptParameters(16, 32, 12, fusion=True)
    inputs, outputs = {}, {}
    handles = []
    for i, layer in enumerate(encoder.transformer.encoder.layer):
        handles.append(
            layer.register_forward_pre_hook(
                lambda _module, args, i=i: inputs.__setitem__(
                    i, args[0].detach().clone()
                )
            )
        )
        handles.append(
            layer.register_forward_hook(
                lambda _module, _args, output, i=i: outputs.__setitem__(
                    i, output[0].detach().clone()
                )
            )
        )
    try:
        result = encoder(prompts, ids)
        assert result.shape == (3, 16)
        expected = encoder.aggregate_class_tokens()
        if fusion:
            expected = 0.5 * expected + 0.5 * encoder.tcp_prompt.fusion_prompt
        torch.testing.assert_close(inputs[8][:, 1:5], expected)
        torch.testing.assert_close(inputs[8][:, :1], outputs[7][:, :1])
        torch.testing.assert_close(inputs[8][:, 5:], outputs[7][:, 5:])
        for i in range(9, 12):
            assert torch.equal(inputs[i], outputs[i - 1])
        for i in range(1, 8):
            expected = encoder.tcp_prompt.text_prompt.for_layer(
                i, 3, prompts.dtype, prompts.device
            )
            torch.testing.assert_close(inputs[i][:, 1:5], expected)
        loss = result.square().mean()
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    if fusion:
        assert encoder.tcp_prompt.fusion_prompt.grad.norm() > 0
    assert encoder.tcp_prompt.down_projection.weight.grad.norm() > 0
    assert encoder.tcp_prompt.up_projection.weight.grad.norm() > 0
    assert encoder.tcp_prompt.text_prompt.prompt_embeddings.grad.norm() > 0
    assert prompts.grad.norm() > 0
    assert bank.grad is None
    assert all(parameter.grad is None for parameter in encoder.base_text_encoder.parameters())


def test_parameter_count_and_checkpoint_metadata():
    prompt = OriginalStyleTCPPromptParameters(512, 768, 12)
    assert sum(
        parameter.numel()
        for name, parameter in prompt.named_parameters()
        if not name.startswith("text_prompt.")
    ) == 461952
    assert sum(parameter.numel() for parameter in prompt.parameters()) == 486528

    encoder, _, _, _ = _original_encoder()
    state = {"tcp." + key: value for key, value in encoder.tcp_prompt.state_dict().items()}
    validate_tcp_checkpoint_state(state, encoder.tcp_prompt)
    tampered = copy.deepcopy(state)
    tampered["tcp._meta_insert_layer"] = torch.tensor(7)
    with pytest.raises(RuntimeError, match="insert_layer"):
        validate_tcp_checkpoint_state(tampered, encoder.tcp_prompt)


def test_tcp_config_has_no_implementation_mode():
    from dassl.config import get_cfg_default
    from train import extend_cfg

    cfg = get_cfg_default()
    extend_cfg(cfg)
    assert "MODE" not in cfg.TRAINER.TCP
    assert "CONFUSION_AWARE" not in cfg.TRAINER
    assert "EXPERT_MOE" not in cfg.TRAINER
    assert cfg.TRAINER.TCP.INSERT_LAYER == 8
