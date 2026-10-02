import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.nn import functional as F
from timm.models.vision_transformer import VisionTransformer

from models.class_conditioned_visual_prompt import ClassConditionedVisualPrompt
from models.vpt import TimmViTVisualPromptEncoder
from trainers.CoOp.coop_biomedclip import CustomCLIP
from trainers.CoOp.coop_vpt_biomedclip import CVPCustomCLIP, CoOpVPT_BiomedCLIP, PromptParameterBundle
from test_coop_vpt_biomedclip import _tcp_cfg


def test_mlp_shape_count_detach_and_gradient():
    module = ClassConditionedVisualPrompt()
    prior = torch.randn(3, 512, requires_grad=True)
    tokens = module(prior)
    assert tokens.shape == (3, 4, 768)
    assert sum(p.numel() for p in module.parameters()) == 461952
    tokens.square().mean().backward()
    assert prior.grad is None
    assert all(torch.isfinite(p.grad).all() and p.grad.norm() > 0 for p in module.parameters())


def visual():
    base = nn.Module()
    base.trunk = VisionTransformer(img_size=16, patch_size=8, embed_dim=32,
        depth=12, num_heads=4, mlp_ratio=2, num_classes=0)
    base.head = nn.Linear(32, 512, bias=False)
    return TimmViTVisualPromptEncoder(base, 4, mode='deep')


def test_injection_before_block7_and_propagation_without_replacement():
    adapter = visual().eval()
    adapter.requires_grad_(False)
    adapter.visual_prompt.requires_grad_(True)
    prompts = torch.randn(2, 4, 32, requires_grad=True)
    before, after = {}, {}
    handles = []
    for index, block in enumerate(adapter.trunk.blocks):
        handles += [block.register_forward_pre_hook(
            lambda module, args, i=index: before.update({i: args[0].detach().clone()})),
            block.register_forward_hook(
            lambda module, args, output, i=index: after.update({i: output.detach().clone()}))]
    output = adapter.forward_with_class_prompt(torch.randn(2, 3, 16, 16), prompts, fusion_weight=1.0)
    for handle in handles:
        handle.remove()
    assert output.shape == (2, 512)
    assert all(x.shape == (2, 9, 32) for x in before.values())
    torch.testing.assert_close(before[7][:, 1:5], prompts, rtol=0, atol=0)
    for index in range(7):
        torch.testing.assert_close(before[index][:, 1:5],
            adapter.visual_prompt.for_layer(index, 2, prompts.dtype, prompts.device), rtol=0, atol=0)
    for index in range(8, 12):
        torch.testing.assert_close(before[index], after[index - 1], rtol=0, atol=0)
    output.square().mean().backward()
    assert prompts.grad.norm() > 0 and torch.isfinite(prompts.grad).all()
    assert adapter.visual_prompt.prompt_embeddings.grad[:7].norm() > 0
    assert adapter.visual_prompt.prompt_embeddings.grad[7:].count_nonzero() == 0
    assert all(p.grad is None for p in adapter.base_visual.parameters())


class Prompts(nn.Module):
    def __init__(self):
        super().__init__()
        self.ctx = nn.Parameter(torch.randn(3, 512))
    def forward(self):
        return self.ctx


class Text(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('class_prior', torch.randn(3, 512))
        self.tke = nn.Linear(512, 512)
    def forward(self, prompts, ids):
        return prompts + self.tke(self.class_prior)


def model(cls):
    result = cls.__new__(cls)
    nn.Module.__init__(result)
    result.image_encoder = visual().eval()
    result.prompt_learner = Prompts()
    result.text_encoder = Text()
    result.tokenized_prompts = None
    result.logit_scale = nn.Parameter(torch.tensor(1.), requires_grad=False)
    result.dtype = torch.float32
    if cls is CVPCustomCLIP:
        result.cvp = ClassConditionedVisualPrompt(hidden_dim=32)
        result.cvp_insert_layer = 7
        result.cvp_fusion_weight = 1.0
    return result


def test_all_class_shapes_correspondence_labels_and_ce_gradient():
    network = model(CVPCustomCLIP).eval()
    network.image_encoder.base_visual.requires_grad_(False)
    calls = []
    original = network.image_encoder.forward_with_class_prompt
    def counted(image, prompts, insert_layer, fusion_weight=1.0):
        calls.append(prompts.detach().clone())
        return original(image, prompts, insert_layer, fusion_weight)
    network.image_encoder.forward_with_class_prompt = counted
    images = torch.randn(2, 3, 16, 16)
    logits, text, image = network(images, return_features=True)
    tokens = network.cvp(network.text_encoder.class_prior)
    assert (tokens.shape, image.shape, text.shape, logits.shape) == (
        (3, 4, 32), (2, 3, 512), (3, 512), (2, 3))
    assert len(calls) == 3
    for index, prompt in enumerate(calls):
        torch.testing.assert_close(prompt, tokens[index].unsqueeze(0).expand(2, -1, -1))
    torch.testing.assert_close(logits, network.logit_scale.exp() * (image * text.unsqueeze(0)).sum(-1))
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.TRAINER.CVP.ENABLED = True
    trainer.model = network
    out1, loss1 = trainer._compute_training_loss(images, torch.tensor([0, 1]))
    out2, loss2 = trainer._compute_training_loss(images, torch.tensor([1, 2]))
    torch.testing.assert_close(out1, out2, rtol=0, atol=0)
    assert set(loss1) == {'loss', 'loss_ce'}
    assert loss1['loss'].item() != loss2['loss'].item()
    loss1['loss'].backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() and p.grad.norm() > 0
               for p in network.cvp.parameters())
    assert all(p.grad is None for p in network.image_encoder.base_visual.parameters())


def test_disabled_original_forward_is_exactly_unchanged():
    network = model(CustomCLIP).eval()
    image = torch.randn(2, 3, 16, 16)
    with torch.no_grad():
        v = network.image_encoder(image)
        t = network.text_encoder(network.prompt_learner(), None)
        expected = network.logit_scale.exp() * (v / v.norm(dim=-1, keepdim=True)) @ (t / t.norm(dim=-1, keepdim=True)).t()
        torch.testing.assert_close(network(image), expected, rtol=0, atol=0)
        before = network(image)
        network.cvp = ClassConditionedVisualPrompt(hidden_dim=32)
        torch.testing.assert_close(network(image), before, rtol=0, atol=0)


@pytest.mark.parametrize('failure', ['tcp', 'slots', 'layer', 'bottleneck'])
def test_invalid_config(failure):
    cfg = _tcp_cfg()
    cfg.TRAINER.CVP.ENABLED = True
    cfg.TRAINER.COOPVPT.VPT_N_CTX = 4
    if failure == 'tcp':
        cfg.TRAINER.TCP.ENABLED = False
    elif failure == 'slots':
        cfg.TRAINER.COOPVPT.VPT_N_CTX = 5
    elif failure == 'layer':
        cfg.TRAINER.CVP.INSERT_LAYER = 12
    else:
        cfg.TRAINER.CVP.BOTTLENECK_DIM = 0
    with pytest.raises(ValueError):
        object.__new__(CoOpVPT_BiomedCLIP).check_cfg(cfg)


def test_optional_bundle_and_checkpoint_compatibility(tmp_path):
    plain = PromptParameterBundle(nn.Linear(2, 2), nn.Linear(2, 2), nn.Linear(2, 2))
    assert not any(k.startswith('cvp.') for k in plain.state_dict())
    bundle = PromptParameterBundle(plain.prompt_learner, plain.visual_prompt, plain.tcp,
                                  ClassConditionedVisualPrompt())
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer._validate_cvp_checkpoint({'state_dict': plain.state_dict()})
    trainer.cfg.TRAINER.CVP.ENABLED = True
    checkpoint = dict(state_dict=bundle.state_dict(), **trainer._cvp_checkpoint_metadata())
    path = tmp_path / 'cvp.pt'
    torch.save(checkpoint, path)
    restored = torch.load(path, weights_only=False)
    trainer._validate_cvp_checkpoint(restored)
    other = copy.deepcopy(bundle)
    other.load_state_dict(restored['state_dict'], strict=True)
    with pytest.raises(RuntimeError):
        trainer._validate_cvp_checkpoint({'state_dict': plain.state_dict()})
    changed = dict(restored, cvp_insert_layer=8)
    with pytest.raises(RuntimeError):
        trainer._validate_cvp_checkpoint(changed)
    trainer.cfg.TRAINER.CVP.ENABLED = False
    with pytest.raises(RuntimeError):
        trainer._validate_cvp_checkpoint(restored)
    with pytest.raises(RuntimeError):
        trainer._validate_cvp_checkpoint({'state_dict': bundle.state_dict()})


def test_training_entry_sets_cvp_flags(tmp_path, monkeypatch):
    from scripts import run_three_methods as runner
    calls = []
    def run(command, **kwargs):
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner.subprocess, 'run', run)
    runner.run_case(1, ('cvp', 'DermaMNIST', 4, 1), output_dir=tmp_path, cvp_fusion_weight=.5)
    command = calls[0]
    for key, expected in {'TRAINER.CVP.ENABLED': 'True', 'TRAINER.CVP.INSERT_LAYER': '7',
        'TRAINER.CVP.NUM_TOKENS': '4', 'TRAINER.CVP.BOTTLENECK_DIM': '128',
        'TRAINER.CVP.FUSION_WEIGHT': '0.5',
        'TRAINER.TCP.ENABLED': 'True', 'TRAINER.TCP.INSERT_LAYER': '7',
        'TRAINER.TCP.FUSION_WEIGHT': '0.5'}.items():
        assert command[command.index(key) + 1] == expected
    assert calls[1][calls[1].index('--method') + 1] == 'cvp'
    assert calls[1][calls[1].index('--cvp-fusion-weight') + 1] == '0.5'


def test_eval_cvp_configuration_and_checkpoint_metadata(tmp_path, monkeypatch):
    import json
    import sys
    from scripts import evaluate_three_methods as evaluation
    (tmp_path / 'best_validation_accuracy.json').write_text(json.dumps({'epoch': 8, 'selection_value': 60.}))
    metadata = dict(cvp_enabled=True, cvp_insert_layer=7, cvp_num_tokens=4, cvp_bottleneck_dim=128, cvp_fusion_weight=.5)
    checkpoint = dict(fusion_weight=.5, **metadata)
    captured = []
    trainer = SimpleNamespace(load_model=lambda path: checkpoint, test=lambda split: None,
                              last_eval_results={'accuracy': 61.})
    monkeypatch.setattr(evaluation, 'build_trainer', lambda cfg: captured.append(cfg) or trainer)
    monkeypatch.setattr(evaluation, 'set_random_seed', lambda seed: None)
    monkeypatch.setattr(sys, 'argv', ['eval', '--run-dir', str(tmp_path), '--method', 'cvp',
        '--dataset', 'DermaMNIST', '--shots', '4', '--seed', '1', '--cvp-fusion-weight', '0.5'])
    evaluation.main()
    cfg = captured[0]
    assert cfg.TRAINER.CVP.ENABLED and cfg.TRAINER.TCP.ENABLED
    assert cfg.TRAINER.TCP.FUSION_WEIGHT == .5
    assert cfg.TRAINER.CVP.FUSION_WEIGHT == .5
    assert cfg.TRAINER.CVP.INSERT_LAYER == cfg.TRAINER.TCP.INSERT_LAYER == 7
    result = json.loads((tmp_path / 'test_metrics.json').read_text())
    assert all(result[k] == v for k, v in metadata.items())


@pytest.mark.parametrize('enabled', [False, True])
def test_resume_checks_cvp_before_restoring_optimizer(tmp_path, monkeypatch, enabled):
    from trainers.CoOp import coop_vpt_biomedclip as module
    directory = tmp_path / 'prompt_parameters'
    directory.mkdir()
    (directory / 'checkpoint').write_text('model.pth.tar-1')
    state = {'cvp.down_projection.weight': torch.randn(128, 512)} if not enabled else {}
    checkpoint = dict(state_dict=state, cvp_enabled=not enabled)
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.TRAINER.CVP.ENABLED = enabled
    monkeypatch.setattr(module, 'load_checkpoint', lambda path: checkpoint)
    monkeypatch.setattr(module, 'resume_from_checkpoint', lambda *args: pytest.fail('restore must not run'))
    with pytest.raises(RuntimeError, match='CVP mode'):
        trainer.resume_model_if_exist(str(tmp_path))


@pytest.mark.parametrize('enabled', [False, True])
def test_load_entry_rejects_opposite_cvp_mode_before_state_restore(tmp_path, monkeypatch, enabled):
    from trainers.CoOp import coop_vpt_biomedclip as module
    checkpoint = dict(cvp_enabled=not enabled,
        state_dict={'cvp.weight': torch.ones(1)} if not enabled else {})
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.TRAINER.CVP.ENABLED = enabled
    monkeypatch.setattr(module, 'load_checkpoint', lambda path: checkpoint)
    with pytest.raises(RuntimeError, match='CVP mode'):
        trainer.load_prompt_checkpoint(tmp_path / 'model.pt')


def test_complete_report_preserves_actual_checkpoint_configuration(tmp_path, monkeypatch):
    import json
    from scripts import run_cvp as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path / 'cvp_results')
    monkeypatch.setattr(runner, 'BASELINE', tmp_path / 'baseline')
    for dataset in runner.experiments.DATASETS:
        for shots in runner.experiments.SHOTS:
            for seed in runner.experiments.SEEDS:
                rel = f'{dataset}/shots_{shots}/seed{seed}'
                for root, value in ((runner.BASELINE, {'test_accuracy': 60.}),
                    (runner.OUT / 'cvp', dict(test_accuracy=61., selected_epoch=8,
                     validation_accuracy=59., cvp_enabled=True, cvp_insert_layer=7,
                     cvp_num_tokens=4, cvp_bottleneck_dim=128, fusion_weight=.5, cvp_fusion_weight=.5))):
                    path = root / rel / 'test_metrics.json'
                    path.parent.mkdir(parents=True)
                    path.write_text(json.dumps(value))
    runner.summarize()
    report = (runner.OUT / 'comparison_report.md').read_text()
    assert '36/36' in report and '12/12' in report and '12/0/0' in report
    assert 'Original 60.0000%' in report and 'CVP 61.0000%' in report
    header = (runner.OUT / 'results_detailed.csv').read_text().splitlines()[0]
    assert all(name in header for name in ('cvp_enabled', 'cvp_insert_layer',
        'cvp_num_tokens', 'cvp_bottleneck_dim', 'fusion_weight'))


def test_cvp_gpu_wait_uses_measured_memory_and_fifteen_minute_interval(tmp_path, monkeypatch):
    from scripts import run_cvp as runner
    monkeypatch.setattr(runner.experiments, 'OUT', tmp_path)
    monkeypatch.setitem(runner.MEMORY_REQUIRED, 'DermaMNIST', 20000.)
    memory = iter([10000, 30000])
    monkeypatch.setattr(runner, 'gpu_capacity', lambda gpu: next(memory))
    waits, calls = [], []
    monkeypatch.setattr(runner.time, 'sleep', lambda seconds: waits.append(seconds))
    monkeypatch.setattr(runner, '_run_case', lambda gpu, case, **kwargs: calls.append((gpu, case)) or 'complete')
    case = ('cvp', 'DermaMNIST', 4, 1)
    assert runner.run_case(1, case) == 'complete'
    assert waits == [900] and calls == [(1, case)]
    dest = runner.experiments.run_dir(case)
    dest.mkdir(parents=True)
    (dest / 'test_metrics.json').write_text('{}')
    assert runner.run_case(1, case) == 'already_complete'
    assert len(calls) == 1


def test_runner_final_validation_reads_actual_checkpoint_epoch(tmp_path, monkeypatch):
    import json
    from scripts import run_cvp as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path / 'out')
    monkeypatch.setattr(runner, 'VALIDATION', tmp_path / 'validation.json')
    monkeypatch.setattr(runner, 'summarize', lambda: None)
    monkeypatch.setattr(runner.experiments, 'main', lambda: None)
    metadata = dict(cvp_enabled=True, cvp_insert_layer=7, cvp_num_tokens=4,
                    cvp_bottleneck_dim=128, fusion_weight=.5, cvp_fusion_weight=.5)
    runner.VALIDATION.write_text(json.dumps([
        dict(dataset=d, formal_batch=32, batch32_peak_mib=20000., cvp_fusion_weight=.5)
        for d in runner.experiments.DATASETS]))
    for dataset in runner.experiments.DATASETS:
        for shots in runner.experiments.SHOTS:
            for seed in runner.experiments.SEEDS:
                dest = runner.OUT / 'cvp' / dataset / f'shots_{shots}' / f'seed{seed}'
                (dest / 'prompt_parameters').mkdir(parents=True)
                (dest / 'test_metrics.json').write_text(json.dumps(dict(
                    selected_epoch=8, validation_accuracy=60., **metadata)))
                (dest / 'best_validation_accuracy.json').write_text(json.dumps(dict(epoch=8, selection_value=60.)))
                torch.save(dict(epoch=8, **metadata),
                           dest / 'prompt_parameters/model-best.pth.tar')
    # Restore shared launcher globals automatically after invoking the CVP main.
    for attr in ('OUT', 'METHODS', 'GPUS', 'CASES', 'summarize', 'run_case'):
        monkeypatch.setattr(runner.experiments, attr, getattr(runner.experiments, attr))
    runner.main(jobs_per_gpu=1)
    assert json.loads((runner.OUT / '_manager/final_validation.json').read_text())['completed'] == 36
    checkpoint = runner.OUT / 'cvp/DermaMNIST/shots_4/seed1/prompt_parameters/model-best.pth.tar'
    torch.save(dict(epoch=9, **metadata), checkpoint)
    with pytest.raises(RuntimeError, match='saved checkpoint metadata mismatch'):
        runner.main(jobs_per_gpu=1)


def test_gpu_reservations_account_for_other_cvp_job(tmp_path, monkeypatch):
    from scripts import run_cvp as runner
    monkeypatch.setattr(runner.experiments, 'OUT', tmp_path)
    monkeypatch.setattr(runner, 'RESERVATIONS', {2: {('cvp', 'DermaMNIST', 4, 1): 20000.}})
    monkeypatch.setitem(runner.MEMORY_REQUIRED, 'DermaMNIST', 20000.)
    capacity = iter([35000, 45000])
    monkeypatch.setattr(runner, 'gpu_capacity', lambda gpu: next(capacity))
    waits = []
    monkeypatch.setattr(runner.time, 'sleep', waits.append)
    case = ('cvp', 'DermaMNIST', 8, 2)
    monkeypatch.setattr(runner, '_run_case', lambda gpu, case, **kwargs: 'complete')
    assert runner.run_case(2, case) == 'complete'
    assert waits == [900]
    assert case not in runner.RESERVATIONS[2]


def test_two_slot_handoff_keeps_active_case_on_its_gpu_and_never_duplicates(tmp_path, monkeypatch):
    import os
    from scripts import run_cvp as runner
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    cases = [('cvp', 'DermaMNIST', 4, seed) for seed in range(1, 5)]
    monkeypatch.setattr(runner.experiments, 'CASES', cases)
    monkeypatch.setattr(runner.experiments, 'GPUS', (1, 2))
    read_fd, write_fd = os.pipe()
    os.write(write_fd, b'done')
    os.close(write_fd)
    monkeypatch.setattr(runner, 'adopt_children', lambda pid, slots: {1: [(cases[0], 999, read_fd)]})
    monkeypatch.setattr(runner, 'RESERVATIONS', {1: {cases[0]: 20000.}, 2: {}})
    monkeypatch.setattr(runner.os, 'kill', lambda *args: None)
    monkeypatch.setattr(runner, 'summarize', lambda: None)
    calls = []
    monkeypatch.setattr(runner, 'run_case', lambda gpu, case, **kwargs: calls.append((gpu, case)) or 'complete')
    runner.run_queue(2, previous_manager=123)
    assert len(calls) == len(cases)
    assert len({case for gpu, case in calls}) == len(cases)
    assert (1, cases[0]) in calls


def test_failed_handoff_resumes_original_manager(tmp_path, monkeypatch):
    import signal
    import subprocess
    import sys
    from scripts import run_cvp as runner
    proc = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)', 'scripts.run_cvp'])
    monkeypatch.setattr(runner, 'OUT', tmp_path)
    def fail_write(*args):
        raise OSError('test handoff write failure')
    monkeypatch.setattr(runner.experiments, 'write_json', fail_write)
    signals = []
    original_kill = runner.os.kill
    def kill(pid, sig):
        signals.append(sig)
        original_kill(pid, sig)
    monkeypatch.setattr(runner.os, 'kill', kill)
    try:
        with pytest.raises(OSError, match='handoff write failure'):
            runner.adopt_children(proc.pid, 2)
        assert signals == [signal.SIGSTOP, signal.SIGCONT]
    finally:
        original_kill(proc.pid, signal.SIGCONT)
        proc.terminate()
        proc.wait(timeout=5)


def test_visual_half_fusion_uses_layer7_parameter_and_both_branches_receive_gradients():
    adapter = visual().eval()
    adapter.base_visual.requires_grad_(False)
    prompts = torch.randn(2, 4, 32, requires_grad=True)
    seen = []
    handle = adapter.trunk.blocks[7].register_forward_pre_hook(
        lambda module, args: seen.append(args[0][:, 1:5].detach().clone()))
    output = adapter.forward_with_class_prompt(torch.randn(2, 3, 16, 16), prompts, 7)
    handle.remove()
    expected = .5 * adapter.visual_prompt.for_layer(7, 2, prompts.dtype, prompts.device) + .5 * prompts
    torch.testing.assert_close(seen[0], expected, rtol=0, atol=0)
    output.square().mean().backward()
    assert prompts.grad.norm() > 0
    assert adapter.visual_prompt.prompt_embeddings.grad[7].norm() > 0
    assert adapter.visual_prompt.prompt_embeddings.grad[8:].count_nonzero() == 0


def test_visual_fusion_checkpoint_rejects_direct_replacement_but_preserves_legacy():
    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_cfg()
    trainer.cfg.TRAINER.CVP.ENABLED = True
    trainer.cfg.TRAINER.CVP.FUSION_WEIGHT = 1.0
    module = ClassConditionedVisualPrompt()
    checkpoint = dict(state_dict={'cvp.' + k: v for k, v in module.state_dict().items()},
                      **trainer._cvp_checkpoint_metadata())
    del checkpoint['cvp_fusion_weight']
    trainer._validate_cvp_checkpoint(checkpoint)
    trainer.cfg.TRAINER.CVP.FUSION_WEIGHT = .5
    with pytest.raises(RuntimeError, match='cvp_fusion_weight'):
        trainer._validate_cvp_checkpoint(checkpoint)
