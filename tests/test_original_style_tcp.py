import copy

import pytest
import torch
from torch.nn import functional as F

from models.original_style_tcp import OriginalStyleTCPBertTextEncoder, OriginalStyleTCPPromptParameters
from models.multitext_tcp import validate_tcp_checkpoint_state
from test_multitext_tcp import _TinyTextTower, _adapter


def original():
    torch.manual_seed(71)
    tower = _TinyTextTower(num_hidden_layers=12)
    tower.requires_grad_(False)
    bank = F.normalize(torch.randn(3, 50, 16), dim=-1).requires_grad_()
    encoder = OriginalStyleTCPBertTextEncoder(tower, bank, ['a', 'b', 'c']).eval()
    ids = torch.zeros(3, 16, dtype=torch.long)
    ids[:, :8] = torch.tensor([2, 10, 11, 12, 13, 20, 3, 21])
    prompts = tower.transformer.embeddings.word_embeddings(ids).detach().requires_grad_()
    return encoder, bank, prompts, ids


def test_mean50_order_independence_and_shared_tke():
    encoder, bank, _, _ = original()
    torch.testing.assert_close(encoder.class_prior, F.normalize(bank.mean(1), dim=-1))
    shuffled = bank.detach()[:, torch.randperm(50)]
    other = OriginalStyleTCPBertTextEncoder(copy.deepcopy(encoder.base_text_encoder), shuffled, ['a', 'b', 'c'])
    other.tcp_prompt.load_state_dict(encoder.tcp_prompt.state_dict())
    torch.testing.assert_close(other.class_prior, encoder.class_prior)
    tokens = encoder.aggregate_class_tokens()
    torch.testing.assert_close(other.aggregate_class_tokens(), tokens)
    assert tokens.shape == (3, 4, 32)
    assert not torch.allclose(tokens[0], tokens[1])
    assert not encoder.description_bank.requires_grad
    assert not encoder.class_prior.requires_grad
    assert set(dict(encoder.tcp_prompt.named_parameters())) == {
        'text_prompt.prompt_embeddings', 'down_projection.weight', 'down_projection.bias',
        'up_projection.weight', 'up_projection.bias',
    }


def test_once_at_block8_and_natural_propagation_with_gradients():
    encoder, bank, prompts, ids = original()
    inputs, outputs = {}, {}
    handles = []
    for i, layer in enumerate(encoder.transformer.encoder.layer):
        handles.append(layer.register_forward_pre_hook(lambda m, args, i=i: inputs.__setitem__(i, args[0].detach().clone())))
        handles.append(layer.register_forward_hook(lambda m, args, out, i=i: outputs.__setitem__(i, out[0].detach().clone())))
    result = encoder(prompts, ids)
    assert result.shape == (3, 16)
    torch.testing.assert_close(inputs[8][:, 1:5], encoder.aggregate_class_tokens())
    for i in range(9, 12):
        assert torch.equal(inputs[i], outputs[i - 1])
    for i in range(1, 8):
        expected = encoder.tcp_prompt.text_prompt.for_layer(i, 3, prompts.dtype, prompts.device)
        torch.testing.assert_close(inputs[i][:, 1:5], expected)
    loss = result.square().mean()
    loss.backward()
    assert encoder.tcp_prompt.down_projection.weight.grad.norm() > 0
    assert encoder.tcp_prompt.up_projection.weight.grad.norm() > 0
    assert encoder.tcp_prompt.text_prompt.prompt_embeddings.grad.norm() > 0
    assert prompts.grad.norm() > 0
    assert bank.grad is None
    assert all(p.grad is None for p in encoder.base_text_encoder.parameters())
    for handle in handles:
        handle.remove()


def test_parameter_count_and_checkpoint_isolation():
    prompt = OriginalStyleTCPPromptParameters(512, 768, 12)
    assert sum(p.numel() for n, p in prompt.named_parameters() if not n.startswith('text_prompt.')) == 461952
    assert sum(p.numel() for p in prompt.parameters()) == 486528
    encoder, _, _, _ = original()
    state = {'tcp.' + k: v for k, v in encoder.tcp_prompt.state_dict().items()}
    validate_tcp_checkpoint_state(state, encoder.tcp_prompt)
    _, multi, _, _ = _adapter()
    with pytest.raises(RuntimeError):
        validate_tcp_checkpoint_state(state, multi.tcp_prompt)
    state = {'tcp.' + k: v for k, v in multi.tcp_prompt.state_dict().items()}
    with pytest.raises(RuntimeError):
        validate_tcp_checkpoint_state(state, encoder.tcp_prompt)


@pytest.mark.parametrize('confusion', [False, True])
def test_training_classification_and_confusion_preservation(confusion):
    from test_expert_moe import expert, ConfusionSpy
    from test_coop_vpt_biomedclip import _tcp_ablation_cfg
    from trainers.CoOp.coop_vpt_biomedclip import CoOpVPT_BiomedCLIP

    trainer = object.__new__(CoOpVPT_BiomedCLIP)
    trainer.cfg = _tcp_ablation_cfg(True)
    assert trainer.cfg.TRAINER.TCP.MODE == 'multitext'
    trainer.cfg.TRAINER.TCP.MODE = 'original_style'
    trainer.check_cfg(trainer.cfg)
    trainer.tcp_mode = 'original_style'
    trainer.tcp_enabled = True
    trainer.confusion_enabled = confusion
    trainer.model = expert(confusion)
    if confusion:
        class TrainingConfusionSpy(ConfusionSpy):
            def forward(self, images, patches, text, base, scale, first):
                logits, details = super().forward(images, patches, text, base, scale, first)
                details["pair_first"] = first
                return logits, details
        trainer.model.confusion_adapter = TrainingConfusionSpy()
    calls = []
    trainer.model.text_encoder.register_forward_hook(lambda *args: calls.append(1))
    labels = torch.tensor([0, 1, 2])
    _, losses, _ = trainer._compute_training_loss(torch.randn(3, 3), labels)
    expected = losses['loss_ce']
    if confusion:
        expected = expected + trainer.cfg.TRAINER.CONFUSION_AWARE.LAMBDA_CONF * losses['loss_confuse']
    assert set(losses) == ({'loss', 'loss_ce', 'loss_confuse'} if confusion else {'loss', 'loss_ce'})
    torch.testing.assert_close(losses['loss'], expected)
    assert len(calls) == 1
    losses['loss'].backward()
    assert trainer.model.prompt_learner.ctx.grad.norm() > 0
