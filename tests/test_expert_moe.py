from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F

from models.expert_moe import ExpertMoE, router_loss
from trainers.CoOp.coop_biomedclip import CustomCLIP


class ImageEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = nn.Linear(3, 3, bias=False)

    def forward(self, image):
        return self.proj(image)


class Prompts(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Parameter(torch.eye(3))

    def forward(self):
        return self.ctx


class TextEncoder(nn.Module):
    def forward(self, prompts, tokens):
        return prompts


class ConfusionSpy(nn.Module):
    needs_patch_tokens = False

    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(0.2))
        self.calls = []

    def forward(self, images, patches, text, base, scale, first):
        self.calls.append((base.clone(), first.clone()))
        return base + self.scale * base.roll(1, dims=-1), {}


def expert(confusion=False):
    # Exercise the actual CustomCLIP forward, with small towers to avoid downloads.
    model = CustomCLIP.__new__(CustomCLIP)
    nn.Module.__init__(model)
    model.image_encoder = ImageEncoder()
    model.prompt_learner = Prompts()
    model.text_encoder = TextEncoder()
    model.tokenized_prompts = None
    model.dtype = torch.float32
    model.logit_scale = nn.Parameter(torch.tensor(1.0))
    model.confusion_adapter = ConfusionSpy() if confusion else None
    return model


def test_frozen_experts_router_gradients_and_exact_parameter_preservation():
    torch.manual_seed(12)
    tcp, conf = expert(), expert(True)
    model = ExpertMoE(tcp, conf, 3).train()
    assert not ({id(p) for p in tcp.parameters()} & {id(p) for p in conf.parameters()})
    assert all(not m.training for e in (tcp, conf) for m in e.modules())
    assert all(not p.requires_grad for e in (tcp, conf) for p in e.parameters())
    before = [p.detach().clone() for e in (tcp, conf) for p in e.parameters()]
    image = torch.randn(7, 3)
    output, weights = model(image, return_weights=True)
    assert torch.equal(weights, torch.full((7, 2), 0.5))
    assert torch.allclose(weights.sum(-1), torch.ones(7))
    expected = (tcp(image).softmax(-1) + conf(image).softmax(-1)) / 2
    assert torch.allclose(output.exp(), expected)
    optimizer = torch.optim.AdamW(model.router.parameters(), lr=0.1)
    loss = router_loss(output, torch.arange(7) % 3)
    assert torch.allclose(loss, F.nll_loss(expected.clamp_min(1e-8).log(), torch.arange(7) % 3))
    loss.backward()
    assert all(p.grad is not None for p in model.router.parameters())
    assert model.router.weight.grad.abs().sum() > 0
    assert all(p.grad is None for e in (tcp, conf) for p in e.parameters())
    optimizer.step()
    assert all(torch.equal(old, p) for old, p in zip(before, [p for e in (tcp, conf) for p in e.parameters()]))
    assert not torch.equal(model.router.weight, torch.zeros_like(model.router.weight))
    assert torch.allclose(model(image, True)[1].sum(-1), torch.ones(7))
    assert sum(p.numel() for p in model.parameters() if p.requires_grad) == 20
    assert ExpertMoE(expert(), expert(True), 8).router.weight.numel() + 2 == 50


def test_original_prediction_routing_is_independent_of_tcp_and_gt():
    tcp, conf = expert(), expert(True)
    model = ExpertMoE(tcp, conf, 3)
    image = torch.randn(4, 3)
    output = model(image)
    base, first = conf.confusion_adapter.calls[-1]
    assert torch.equal(first, base.argmax(-1))
    with torch.no_grad():
        tcp.image_encoder.proj.weight.neg_()
    model(image)
    new_base, new_first = conf.confusion_adapter.calls[-1]
    assert torch.equal(base, new_base)
    assert torch.equal(first, new_first)
    # Labels are only accepted by the loss, after routing has finished.
    router_loss(output, (first + 1) % 3)
    assert torch.equal(first, conf.confusion_adapter.calls[-1][1])
    assert tcp.confusion_adapter is None
    with pytest.raises(TypeError):
        model(image, label=first)


@pytest.mark.parametrize("mode", ["tcp_only", "conf_only"])
def test_single_modes_return_original_logits_and_skip_other_expert(mode):
    tcp, conf = expert(), expert(True)
    model = ExpertMoE(tcp, conf, 3, mode)
    image = torch.randn(2, 3)
    active, inactive = (tcp, conf) if mode == "tcp_only" else (conf, tcp)
    expected = active(image)
    def fail(*args):
        raise AssertionError("Inactive expert must not run")
    inactive.forward = fail
    assert torch.equal(model(image), expected)


def test_shared_parameters_rejected():
    model = expert()
    with pytest.raises(ValueError, match="independent"):
        ExpertMoE(model, model, 3)


@pytest.mark.parametrize("separate_banks", [False, True])
def test_loader_preserves_config_and_uses_strict_historical_loader(monkeypatch, tmp_path, separate_banks):
    from dassl.config import get_cfg_default
    from train import extend_cfg
    from trainers.CoOp.expert_moe_biomedclip import load_expert, CoOpVPT_BiomedCLIP
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.OPTIM.NAME = "adamw"
    cfg.TRAINER.TCP.DESCRIPTION_CACHE = "shared_projected.pt"
    cfg.TRAINER.TCP.LAYER_DESCRIPTION_CACHE = "shared_layer.pt"
    if separate_banks:
        for prefix in ("TCP", "CONF"):
            for bank in ("DESCRIPTION_CACHE", "LAYER_DESCRIPTION_CACHE"):
                setattr(cfg.TRAINER.EXPERT_MOE, f"{prefix}_{bank}", f"{prefix}_{bank}.pt")
    cfg.freeze()
    path = tmp_path / "model.pth.tar"
    path.touch()
    calls = []
    def build(self, expert_checkpoint=None, rebuild_banks=False):
        prefix = "TCP" if self.cfg.TRAINER.TCP.ENABLED else "CONF"
        for bank in ("DESCRIPTION_CACHE", "LAYER_DESCRIPTION_CACHE"):
            expected = f"{prefix}_{bank}.pt" if separate_banks else getattr(cfg.TRAINER.TCP, bank)
            assert getattr(self.cfg.TRAINER.TCP, bank) == expected
        calls.append((self.cfg.TRAINER.TCP.ENABLED, self.cfg.TRAINER.CONFUSION_AWARE.ENABLED, expert_checkpoint))
        self.model = expert(not self.cfg.TRAINER.TCP.ENABLED).requires_grad_(False).eval()
    monkeypatch.setattr(CoOpVPT_BiomedCLIP, "build_model", build)
    load_expert(cfg, SimpleNamespace(), "cpu", str(path), True)
    load_expert(cfg, SimpleNamespace(), "cpu", str(path), False)
    assert calls == [(True, False, str(path)), (False, True, str(path))]
    assert cfg.TRAINER.TCP.ENABLED and cfg.TRAINER.CONFUSION_AWARE.ENABLED



def test_rebuilding_confusion_requires_original_description_source(tmp_path):
    import json
    from trainers.CoOp.coop_vpt_biomedclip import CoOpVPT_BiomedCLIP
    builder = CoOpVPT_BiomedCLIP.__new__(CoOpVPT_BiomedCLIP)
    builder.confusion_enabled = True
    builder.pair_description_metadata = {"description_fingerprint": "original-source"}
    builder.prompt_parameters = nn.Linear(2, 2)
    checked = []
    builder._validate_checkpoint_metadata = lambda checkpoint, check_bank_fingerprints: checked.append(check_bank_fingerprints)
    folder = tmp_path / "prompt_parameters"
    folder.mkdir()
    checkpoint = folder / "model-best.pth.tar"
    torch.save({"state_dict": builder.prompt_parameters.state_dict()}, checkpoint)
    manifest = tmp_path / "initialization_manifest.json"
    manifest.write_text(json.dumps({"pair_description_fingerprint": "original-source"}))
    builder.load_prompt_checkpoint(checkpoint, check_bank_fingerprints=False)
    assert checked == [False]
    manifest.write_text(json.dumps({"pair_description_fingerprint": "different-source"}))
    with pytest.raises(RuntimeError, match="description source"):
        builder.load_prompt_checkpoint(checkpoint, check_bank_fingerprints=False)
    assert checked == [False]
