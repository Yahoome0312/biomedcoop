import os

import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from models.biomedclip_loader import load_biomedclip
from models.vpt import TimmViTVisualPromptEncoder
from trainers.CoOp.coop_vpt_biomedclip import (
    checkpoint_residual_scale,
    cosine_l2_prompt_anchor_loss,
)


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


def test_vpt_deep_keeps_sequence_length_and_output_contract():
    visual = _TinyTimmVisual()
    adapter = TimmViTVisualPromptEncoder(visual, num_tokens=5, mode="deep")
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


def test_only_visual_prompt_receives_gradients():
    visual = _TinyTimmVisual()
    adapter = TimmViTVisualPromptEncoder(visual, num_tokens=5, mode="deep")
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    adapter.visual_prompt.prompt_embeddings.requires_grad_(True)

    adapter(torch.randn(2, 3, 32, 32)).sum().backward()

    assert tuple(adapter.visual_prompt.prompt_embeddings.shape) == (3, 5, 32)
    assert adapter.visual_prompt.prompt_embeddings.grad is not None
    assert all(
        parameter.grad is None
        for name, parameter in adapter.named_parameters()
        if name != "visual_prompt.prompt_embeddings"
    )


def test_coop_and_visual_prompt_use_one_adamw_group():
    text_prompt = nn.Parameter(torch.randn(4, 32))
    visual_prompt = nn.Parameter(torch.randn(3, 5, 32))
    optimizer = torch.optim.AdamW(
        [text_prompt, visual_prompt], lr=2e-3, weight_decay=5e-4
    )

    assert len(optimizer.param_groups) == 1
    parameter_ids = {
        id(p) for group in optimizer.param_groups for p in group["params"]
    }
    assert parameter_ids == {id(text_prompt), id(visual_prompt)}
    assert optimizer.param_groups[0]["lr"] == 2e-3


def test_checkpoint_restores_the_residual_scale_used_at_its_epoch():
    assert checkpoint_residual_scale(0, 20) == pytest.approx(0.0)
    assert checkpoint_residual_scale(1, 20) == pytest.approx(0.05)
    assert checkpoint_residual_scale(10, 20) == pytest.approx(0.5)
    assert checkpoint_residual_scale(20, 20) == pytest.approx(1.0)
    assert checkpoint_residual_scale(80, 20) == pytest.approx(1.0)
    assert checkpoint_residual_scale(1, 0) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="negative"):
        checkpoint_residual_scale(-1, 20)


def test_cosine_l2_anchor_penalizes_scale_drift_and_keeps_gradients():
    reference = torch.tensor([1.0, 2.0, 3.0])
    unchanged = reference.clone().requires_grad_(True)
    assert cosine_l2_prompt_anchor_loss(
        (unchanged,), (reference,), l2_weight=0.5
    ).item() == pytest.approx(0.0, abs=2e-7)

    scaled = (2.0 * reference).requires_grad_(True)
    cosine_only = cosine_l2_prompt_anchor_loss(
        (scaled,), (reference,), l2_weight=0.0
    )
    scale_aware = cosine_l2_prompt_anchor_loss(
        (scaled,), (reference,), l2_weight=0.5
    )
    assert cosine_only.item() == pytest.approx(0.0, abs=1e-6)
    assert scale_aware.item() == pytest.approx(0.5, abs=1e-6)
    scale_aware.backward()
    assert scaled.grad is not None
    assert scaled.grad.norm().item() > 0.0


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
