import torch
from torch import nn
from transformers import BertConfig, BertModel

from models.text_vpt import BertTextDeepPromptEncoder


class _TinyTextTower(nn.Module):
    def __init__(self):
        super().__init__()
        config = BertConfig(
            vocab_size=101,
            hidden_size=32,
            num_hidden_layers=3,
            num_attention_heads=4,
            intermediate_size=64,
            max_position_embeddings=32,
            pad_token_id=0,
        )
        self.transformer = BertModel(config, add_pooling_layer=True)
        self.pooler = _Pooler()
        self.proj = nn.Linear(32, 16, bias=False)
        self.output_tokens = False


class _Pooler(nn.Module):
    def forward(self, output, attention_mask):
        return output.pooler_output


def _inputs(tower):
    input_ids = torch.zeros(2, 16, dtype=torch.long)
    input_ids[:, 0] = 2
    input_ids[:, 1:6] = torch.tensor([10, 11, 12, 13, 3])
    prompts = tower.transformer.embeddings.word_embeddings(input_ids)
    return prompts, input_ids


def test_text_deep_prompt_replaces_prompt_and_keeps_length():
    torch.manual_seed(1)
    tower = _TinyTextTower()
    adapter = BertTextDeepPromptEncoder(tower, num_tokens=4)
    prompts, tokenized = _inputs(tower)
    lengths = []
    hooks = [
        layer.register_forward_pre_hook(
            lambda _module, args: lengths.append(args[0].shape[1])
        )
        for layer in tower.transformer.encoder.layer
    ]
    try:
        output = adapter(prompts, tokenized)
    finally:
        for hook in hooks:
            hook.remove()

    assert output.shape == (2, 16)
    assert tuple(adapter.text_prompt.prompt_embeddings.shape) == (3, 4, 32)
    assert lengths == [20, 20, 20]


def test_only_text_prompt_receives_gradients():
    torch.manual_seed(2)
    tower = _TinyTextTower()
    adapter = BertTextDeepPromptEncoder(tower, num_tokens=4)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    adapter.text_prompt.prompt_embeddings.requires_grad_(True)

    prompts, tokenized = _inputs(tower)
    adapter(prompts, tokenized).sum().backward()

    assert adapter.text_prompt.prompt_embeddings.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in adapter.named_parameters()
        if name != "text_prompt.prompt_embeddings"
    )


def test_text_deep_prompt_rejects_truncating_valid_tokens():
    tower = _TinyTextTower()
    adapter = BertTextDeepPromptEncoder(tower, num_tokens=20)
    prompts, tokenized = _inputs(tower)
    tokenized[:, -1] = 7

    try:
        adapter(prompts, tokenized)
    except ValueError as error:
        assert "truncate a valid token" in str(error)
    else:
        raise AssertionError("Expected a descriptive valid-token truncation error")
