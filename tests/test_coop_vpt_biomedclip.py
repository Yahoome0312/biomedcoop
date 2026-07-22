import os

import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from models.biomedclip_loader import load_biomedclip
from models.vpt import TimmViTVisualPromptEncoder


class _TinyTimmVisual(nn.Module):
    def __init__(self):
        super().__init__()
        self.trunk = VisionTransformer(
            img_size=32,
            patch_size=8,
            in_chans=3,
            num_classes=0,
            global_pool="token",
            embed_dim=32,
            depth=3,
            num_heads=4,
            mlp_ratio=2,
        )
        self.head = nn.Linear(32, 16, bias=False)

    def forward(self, image):
        return self.head(self.trunk(image))


@pytest.mark.parametrize("mode", ["shallow", "deep"])
def test_vpt_keeps_sequence_length_and_output_contract(mode):
    visual = _TinyTimmVisual()
    adapter = TimmViTVisualPromptEncoder(visual, num_tokens=5, mode=mode)
    lengths = []
    hooks = [
        block.register_forward_pre_hook(
            lambda _module, args: lengths.append(args[0].shape[1])
        )
        for block in visual.trunk.blocks
    ]
    try:
        output = adapter(torch.randn(2, 3, 32, 32))
    finally:
        for hook in hooks:
            hook.remove()

    assert output.shape == (2, 16)
    assert lengths == [22, 22, 22]  # CLS + 16 patches + 5 prompts


@pytest.mark.parametrize(
    ("mode", "expected_shape"),
    [("shallow", (1, 5, 32)), ("deep", (3, 5, 32))],
)
def test_only_visual_prompt_receives_gradients(mode, expected_shape):
    visual = _TinyTimmVisual()
    adapter = TimmViTVisualPromptEncoder(visual, num_tokens=5, mode=mode)
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    adapter.visual_prompt.prompt_embeddings.requires_grad_(True)

    adapter(torch.randn(2, 3, 32, 32)).sum().backward()

    assert tuple(adapter.visual_prompt.prompt_embeddings.shape) == expected_shape
    assert adapter.visual_prompt.prompt_embeddings.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in adapter.named_parameters()
        if name != "visual_prompt.prompt_embeddings"
    )


def test_text_and_visual_optimizers_are_independent():
    text_prompt = nn.Parameter(torch.randn(4, 32))
    visual_prompt = nn.Parameter(torch.randn(3, 5, 32))
    text_optimizer = torch.optim.AdamW([text_prompt], lr=1e-4, weight_decay=0.0)
    visual_optimizer = torch.optim.AdamW([visual_prompt], lr=1e-3, weight_decay=1e-3)

    text_ids = {id(p) for group in text_optimizer.param_groups for p in group["params"]}
    visual_ids = {id(p) for group in visual_optimizer.param_groups for p in group["params"]}
    assert text_ids.isdisjoint(visual_ids)
    assert text_optimizer.param_groups[0]["lr"] != visual_optimizer.param_groups[0]["lr"]
    assert text_optimizer.param_groups[0]["weight_decay"] != visual_optimizer.param_groups[0]["weight_decay"]


@pytest.mark.skipif(
    os.environ.get("RUN_BIOMEDCLIP_INTEGRATION") != "1",
    reason="Set RUN_BIOMEDCLIP_INTEGRATION=1 to load cached BiomedCLIP weights",
)
def test_cached_biomedclip_deep_vpt_forward():
    model, _ = load_biomedclip(
        vpt_enabled=True, vpt_mode="deep", vpt_num_tokens=5
    )
    model.eval()
    output = model.visual(torch.randn(1, 3, 224, 224))
    assert output.shape == (1, 512)
