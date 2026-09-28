import os

import pytest
import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from dassl.config import get_cfg_default
from models.biomedclip_loader import load_biomedclip
from models.vpt import TimmViTVisualPromptEncoder
from train import extend_cfg
from trainers.CoOp.coop_vpt_biomedclip import CoOpVPT_BiomedCLIP


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
    assert lengths == [22, 22, 22]


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


def _tcp_cfg(enabled=True):
    cfg = get_cfg_default()
    extend_cfg(cfg)
    cfg.OPTIM.NAME = "adamw"
    cfg.TRAINER.TCP.ENABLED = enabled
    return cfg


@pytest.mark.parametrize("enabled", [False, True])
def test_tcp_check_cfg_accepts_both_injection_settings(enabled):
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.check_cfg(_tcp_cfg(enabled))


def test_tcp_is_the_only_optional_prompt_component():
    cfg = _tcp_cfg()
    assert "MODE" not in cfg.TRAINER.TCP
    assert "CONFUSION_AWARE" not in cfg.TRAINER
    assert "EXPERT_MOE" not in cfg.TRAINER
    assert cfg.TRAINER.TCP.INSERT_LAYER == 7


def test_classification_loss_has_no_auxiliary_branch():
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.model = lambda image: image
    logits = torch.tensor([[2.0, -1.0], [-0.5, 1.5]], requires_grad=True)
    labels = torch.tensor([0, 1])

    output, losses = trainer._compute_training_loss(logits, labels)

    expected = torch.nn.functional.cross_entropy(logits, labels)
    assert output is logits
    assert set(losses) == {"loss", "loss_ce"}
    torch.testing.assert_close(losses["loss"], expected)
    torch.testing.assert_close(losses["loss_ce"], expected)
    losses["loss"].backward()
    assert logits.grad is not None


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


def test_semantic_loss_matches_kl_and_isolates_text_gradients(tmp_path):
    weight, temperature = 0.1, 0.5
    from torch.nn import functional as F
    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Module()
            self.image_encoder.visual_prompt = nn.Linear(512, 512, bias=False)
            self.text_prompt = nn.Parameter(torch.randn(3, 512))
            self.tke = nn.Linear(512, 512)
            self.backbone = nn.Linear(512, 512, bias=False)
            self.backbone.requires_grad_(False)
        def forward(self, image, return_features=False):
            v = F.normalize(self.image_encoder.visual_prompt(self.backbone(image)), dim=-1)
            t = F.normalize(self.tke(self.text_prompt), dim=-1)
            logits = 7 * v @ t.t()
            return (logits, t, v) if return_features else logits
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.OUTPUT_DIR = str(tmp_path)
    trainer.cfg.TRAINER.SEMANTIC_DISTILL.ENABLED = True
    assert trainer.cfg.TRAINER.SEMANTIC_DISTILL.WEIGHT == weight
    assert trainer.cfg.TRAINER.SEMANTIC_DISTILL.TEMPERATURE == temperature
    trainer.model = Model()
    trainer._semantic_audit_complete = False
    trainer._semantic_step = 0
    trainer.model_zero_grad = lambda: trainer.model.zero_grad(set_to_none=True)
    images, labels = torch.randn(2, 512), torch.tensor([0, 2])
    output, losses = trainer._compute_training_loss(images, labels)
    _, t, v = trainer.model(images, return_features=True)
    expected = F.kl_div(F.log_softmax(v @ t.detach().t() / temperature, dim=-1),
                        F.softmax((t.detach() @ t.detach().t())[labels] / temperature, dim=-1),
                        reduction='batchmean')
    torch.testing.assert_close(losses['loss_sem'], expected)
    torch.testing.assert_close(losses['loss'], F.cross_entropy(output, labels) + weight * expected)
    losses['loss_sem'].backward(retain_graph=True)
    assert trainer.model.image_encoder.visual_prompt.weight.grad.norm() > 0
    assert trainer.model.text_prompt.grad is None
    assert all(p.grad is None for p in trainer.model.tke.parameters())
    assert trainer.model.backbone.weight.grad is None
    trainer.model_zero_grad()
    losses['loss'].backward()
    assert trainer.model.text_prompt.grad.norm() > 0
    assert trainer.model.tke.weight.grad.norm() > 0


@pytest.mark.parametrize('interval, expected_calls', [(0, 1), (1, 4), (2, 2)])
def test_semantic_gradient_sampling_preserves_loss_gradients_and_updates(tmp_path, monkeypatch, interval, expected_calls):
    import copy
    from torch.nn import functional as F

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.image_encoder = nn.Module()
            self.image_encoder.visual_prompt = nn.Linear(8, 8)
            self.text = nn.Parameter(torch.randn(3, 8))

        def forward(self, image, return_features=False):
            v = F.normalize(self.image_encoder.visual_prompt(image), dim=-1)
            t = F.normalize(self.text, dim=-1)
            logits = 7 * v @ t.t()
            return logits, t, v

    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.OUTPUT_DIR = str(tmp_path)
    trainer.cfg.TRAINER.SEMANTIC_DISTILL.ENABLED = True
    trainer.cfg.TRAINER.SEMANTIC_DISTILL.GRAD_NORM_INTERVAL = interval
    trainer.model = Model()
    trainer._semantic_audit_complete = False
    trainer._semantic_step = 0
    trainer.model_zero_grad = lambda: trainer.model.zero_grad(set_to_none=True)
    reference = copy.deepcopy(trainer.model)
    optimizer = torch.optim.AdamW(trainer.model.parameters(), lr=.001)
    reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=.001)
    original_grad = torch.autograd.grad
    calls = []

    def counted_grad(*args, **kwargs):
        calls.append(1)
        return original_grad(*args, **kwargs)

    monkeypatch.setattr(torch.autograd, 'grad', counted_grad)
    for step in range(4):
        images, labels = torch.randn(2, 8), torch.tensor([0, 2])
        trainer.model_zero_grad()
        reference.zero_grad(set_to_none=True)
        output, losses = trainer._compute_training_loss(images, labels)
        ref_output, t, v = reference(images)
        teacher = F.softmax((t.detach() @ t.detach().t())[labels] / .5, dim=-1)
        semantic = F.kl_div(F.log_softmax(v @ t.detach().t() / .5, dim=-1), teacher, reduction='batchmean')
        expected = F.cross_entropy(ref_output, labels) + .1 * semantic
        torch.testing.assert_close(output, ref_output, rtol=0, atol=0)
        torch.testing.assert_close(losses['loss'], expected, rtol=0, atol=0)
        assert ('semantic_grad_norm' in losses) == (step == 0 or interval > 0 and step % interval == 0)
        losses['loss'].backward()
        expected.backward()
        for p, q in zip(trainer.model.parameters(), reference.parameters()):
            torch.testing.assert_close(p.grad, q.grad, rtol=0, atol=0)
        optimizer.step()
        reference_optimizer.step()
        for p, q in zip(trainer.model.parameters(), reference.parameters()):
            torch.testing.assert_close(p, q, rtol=0, atol=0)
    assert len(calls) == expected_calls


def test_resume_accepts_old_semantic_metadata_and_changed_monitor_interval(tmp_path, monkeypatch):
    import trainers.CoOp.coop_vpt_biomedclip as module
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.TRAINER.SEMANTIC_DISTILL.ENABLED = True
    trainer.cfg.TRAINER.SEMANTIC_DISTILL.GRAD_NORM_INTERVAL = 100
    trainer.prompt_parameters = nn.Linear(1, 1)
    trainer.optim = trainer.sched = trainer.scaler = None
    trainer._validate_checkpoint_metadata = lambda checkpoint: None
    prompt = tmp_path/'prompt_parameters'
    prompt.mkdir()
    (prompt/'checkpoint').write_text('model.pth.tar-10')
    saved = {'ENABLED': True, 'WEIGHT': .1, 'TEMPERATURE': .5}
    monkeypatch.setattr(module, 'load_checkpoint', lambda path: {'semantic_distill': saved})
    monkeypatch.setattr(module, 'resume_from_checkpoint', lambda *args: 10)
    assert trainer.resume_model_if_exist(str(tmp_path)) == 10
    saved['GRAD_NORM_INTERVAL'] = 1
    assert trainer.resume_model_if_exist(str(tmp_path)) == 10
    saved['WEIGHT'] = 1.
    with pytest.raises(RuntimeError, match='configuration does not match'):
        trainer.resume_model_if_exist(str(tmp_path))


def test_load_model_returns_training_checkpoint_metadata(tmp_path):
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    folder = tmp_path/'prompt_parameters'
    folder.mkdir()
    (folder/'model-best.pth.tar').touch()
    expected = {'semantic_distill': {'ENABLED': True, 'WEIGHT': 1., 'TEMPERATURE': .2}}
    trainer.load_prompt_checkpoint = lambda path: expected
    assert trainer.load_model(str(tmp_path)) is expected


def test_sparse_gradient_logging_does_not_repeat_stale_values():
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    parameter = nn.Parameter(torch.tensor([[2., 0.], [0., 2.]]))
    trainer.optim = torch.optim.AdamW([parameter], lr=.001)
    trainer._semantic_step = 0
    trainer.parse_batch_train = lambda batch: batch
    trainer.model_zero_grad = lambda: trainer.optim.zero_grad(set_to_none=True)
    trainer.model_backward = lambda loss: loss.backward()
    trainer._audit_gradients_once = lambda: None
    trainer.epoch, trainer.batch_idx, trainer.num_batches = 3, 0, 4
    records = []
    trainer.write_scalar = lambda *args: records.append(args)

    def losses(image, labels):
        step = trainer._semantic_step
        trainer._semantic_step += 1
        result = {'loss': torch.nn.functional.cross_entropy(parameter, labels)}
        if step % 2 == 0:
            result['semantic_grad_norm'] = torch.tensor(step + .5)
        return parameter, result

    trainer._compute_training_loss = losses
    for step in range(3):
        trainer.batch_idx = step
        summary = trainer.forward_backward((None, torch.tensor([0, 1])))
        assert 'semantic_grad_norm' not in summary
    assert records == [('train/semantic_grad_norm', .5, 12), ('train/semantic_grad_norm', 2.5, 14)]
