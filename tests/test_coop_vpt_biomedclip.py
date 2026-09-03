import os

import torch
import pytest
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from dassl.config import get_cfg_default
from models.biomedclip_loader import load_biomedclip
from models.vpt import TimmViTVisualPromptEncoder
from train import extend_cfg
from trainers.CoOp.coop_vpt_biomedclip import (
    CoOpVPT_BiomedCLIP,
    _pair_description_file,
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


def test_vpt_can_return_only_image_patch_tokens():
    visual = _TinyTimmVisual()
    adapter = TimmViTVisualPromptEncoder(visual, num_tokens=5, mode="deep")
    output, patches = adapter(torch.randn(2, 3, 32, 32), return_tokens=True)

    assert output.shape == (2, 16)
    assert patches.shape == (2, 16, 32)


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


def _tcp_ablation_cfg(enabled):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.OPTIM.NAME = "adamw"
    cfg.TRAINER.TCP.ENABLED = enabled
    cfg.TRAINER.CONFUSION_AWARE.BANK_ROOT = "bank"
    return cfg


def test_full_confusion_accepts_tcp_on_and_off():
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.check_cfg(_tcp_ablation_cfg(True))
    trainer.check_cfg(_tcp_ablation_cfg(False))


def test_full_confusion_requires_bank_root():
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    cfg = _tcp_ablation_cfg(False)
    cfg.TRAINER.CONFUSION_AWARE.BANK_ROOT = ""
    with pytest.raises(ValueError, match="requires CONFUSION_AWARE.BANK_ROOT"):
        trainer.check_cfg(cfg)


@pytest.mark.parametrize("tcp_enabled", [False, True])
def test_confusion_off_does_not_require_bank_root(tcp_enabled):
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    cfg = _tcp_ablation_cfg(tcp_enabled)
    cfg.TRAINER.CONFUSION_AWARE.ENABLED = False
    cfg.TRAINER.CONFUSION_AWARE.BANK_ROOT = ""

    trainer.check_cfg(cfg)


def test_confusion_off_uses_only_cross_entropy():
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.confusion_enabled = False
    trainer.model = lambda image: image
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    labels = torch.tensor([0, 1])

    output, losses, details = trainer._compute_training_loss(logits, labels)

    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert output is logits
    assert set(losses) == {"loss", "loss_ce"}
    assert torch.equal(losses["loss"], expected)
    assert torch.equal(losses["loss_ce"], expected)
    assert details is None


@pytest.mark.parametrize(
    ("dataset_name", "filename"),
    [
        ("DermaMNIST", "DermaMNIST.txt"),
        ("CHMNIST", "CHMNIST.txt"),
        ("Kvasir", "Kvasir.txt"),
    ],
)
def test_pair_description_file_matches_repository_filename(
    dataset_name, filename
):
    path = _pair_description_file(dataset_name)
    assert path.name == filename
    assert path.is_file()


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
