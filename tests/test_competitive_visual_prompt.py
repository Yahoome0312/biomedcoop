import torch
from torch import nn
from timm.models.vision_transformer import VisionTransformer

from models.competitive_visual_prompt import VisualPrototypePrompt
from models.vpt import TimmViTVisualPromptEncoder
from trainers.CoOp.coop_vpt_biomedclip import CVPCustomCLIP


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
            depth=12,
            num_heads=4,
            mlp_ratio=2,
        )
        self.head = nn.Linear(32, 16, bias=False)


def test_visual_tke_uses_frozen_mean50_prototypes_directly():
    prototypes = torch.tensor(
        [[1.0, 0.0], [0.0, 2.0], [3.0, 1.0]], requires_grad=True
    )
    visual_tke = VisualPrototypePrompt(
        prior_dim=2, bottleneck_dim=3, num_tokens=4, hidden_dim=5
    )
    inputs = []
    handle = visual_tke.down_projection.register_forward_pre_hook(
        lambda _module, args: inputs.append(args[0].detach().clone())
    )
    try:
        prompts = visual_tke(prototypes)
    finally:
        handle.remove()

    assert prompts.shape == (3, 4, 5)
    torch.testing.assert_close(inputs[0], prototypes.detach())
    prompts.sum().backward()
    assert prototypes.grad is None
    assert visual_tke.down_projection.weight.grad.norm() > 0
    assert visual_tke.up_projection.weight.grad.norm() > 0


def test_default_visual_tke_output_shape_and_parameter_count():
    visual_tke = VisualPrototypePrompt()
    prototypes = torch.randn(3, 512)

    assert visual_tke(prototypes).shape == (3, 4, 768)
    assert sum(parameter.numel() for parameter in visual_tke.parameters()) == 461952


def test_visual_tke_replaces_vpt_at_block8_once_and_then_propagates():
    torch.manual_seed(17)
    visual = _TinyTimmVisual()
    adapter = TimmViTVisualPromptEncoder(
        visual, num_tokens=4, mode="deep", dropout=0.0, prompt_depth=8
    )
    assert adapter.visual_prompt.prompt_embeddings.shape == (8, 4, 32)
    inputs = {index: [] for index in range(12)}
    outputs = {index: [] for index in range(12)}
    head_inputs = []
    handles = []
    for index, block in enumerate(visual.trunk.blocks):
        handles.append(
            block.register_forward_pre_hook(
                lambda _module, args, index=index: inputs[index].append(
                    args[0].detach().clone()
                )
            )
        )
        handles.append(
            block.register_forward_hook(
                lambda _module, _args, output, index=index: outputs[index].append(
                    output.detach().clone()
                )
            )
        )

    original_forward_head = visual.trunk.forward_head

    def record_forward_head(x, *args, **kwargs):
        head_inputs.append(x.detach().clone())
        return original_forward_head(x, *args, **kwargs)

    visual.trunk.forward_head = record_forward_head
    image = torch.randn(2, 3, 32, 32)
    cvp = torch.randn(2, 3, 4, 32, requires_grad=True)
    try:
        shared = adapter.forward_before_layer(image, 8)
        class_features, patches = adapter.forward_from_layer(
            shared, 8, prototype_prompts=cvp, return_tokens=True
        )
    finally:
        visual.trunk.forward_head = original_forward_head
        for handle in handles:
            handle.remove()

    assert class_features.shape == (2, 3, 16)
    assert patches.shape == (2, 3, 16, 32)
    assert head_inputs[0].shape == (6, 17, 32)
    normalized_final = visual.trunk.norm(outputs[11][0])
    torch.testing.assert_close(head_inputs[0][:, :1], normalized_final[:, :1])
    torch.testing.assert_close(head_inputs[0][:, 1:], normalized_final[:, 5:])
    for index in range(8):
        assert len(inputs[index]) == 1
        assert inputs[index][0].shape[:2] == (2, 21)
    for index in range(8, 12):
        assert len(inputs[index]) == 1
        assert inputs[index][0].shape[:2] == (6, 21)

    block8_input = inputs[8][0]
    expanded_shared = shared.unsqueeze(1).expand(-1, 3, -1, -1).reshape(6, 21, 32)
    torch.testing.assert_close(block8_input[:, :1], expanded_shared[:, :1])
    torch.testing.assert_close(block8_input[:, 1:5], cvp.detach().reshape(6, 4, 32))
    torch.testing.assert_close(block8_input[:, 5:], expanded_shared[:, 5:])

    for index in range(9, 12):
        current_input = inputs[index][0]
        previous_output = outputs[index - 1][0]
        torch.testing.assert_close(current_input, previous_output)

    class_features.square().mean().backward()
    assert cvp.grad.norm() > 0
    visual_prompt_grad = adapter.visual_prompt.prompt_embeddings.grad
    assert visual_prompt_grad[:8].norm() > 0
    assert torch.count_nonzero(visual_prompt_grad[8:]) == 0


def test_disabled_cvp_keeps_original_visual_forward_output():
    torch.manual_seed(23)
    adapter = TimmViTVisualPromptEncoder(
        _TinyTimmVisual(), num_tokens=4, mode="deep", dropout=0.0
    ).eval()
    image = torch.randn(2, 3, 32, 32)

    expected = adapter(image)
    shared = adapter.forward_before_layer(image, 8)
    staged_base = adapter.forward_from_layer(shared, 8)

    torch.testing.assert_close(staged_base, expected)


class _FixedPromptLearner(nn.Module):
    def __init__(self, prompts):
        super().__init__()
        self.prompts = nn.Parameter(prompts)

    def forward(self):
        return self.prompts


class _FixedTextEncoder(nn.Module):
    def __init__(self, class_prior=None):
        super().__init__()
        if class_prior is not None:
            self.register_buffer("class_prior", class_prior)

    def forward(self, prompts, _tokenized_prompts):
        return prompts


def test_custom_clip_without_cvp_uses_the_original_global_formula():
    model = CVPCustomCLIP.__new__(CVPCustomCLIP)
    nn.Module.__init__(model)
    text_features = torch.tensor([[2.0, 0.0], [0.0, 3.0]])
    model.prompt_learner = _FixedPromptLearner(text_features.clone())
    model.tokenized_prompts = None
    model.image_encoder = nn.Identity()
    model.text_encoder = _FixedTextEncoder()
    model.logit_scale = nn.Parameter(torch.tensor(0.7), requires_grad=False)
    model.dtype = torch.float32
    model.competitive_visual_prompt = None
    model.competitive_visual_insert_layer = None
    images = torch.tensor([[4.0, 1.0], [1.0, 5.0]])

    logits = model(images)
    expected = model.logit_scale.exp() * torch.nn.functional.normalize(
        images, dim=-1
    ) @ torch.nn.functional.normalize(text_features, dim=-1).t()
    torch.testing.assert_close(logits, expected)


def test_enabled_cvp_forward_uses_class_conditioned_cls_features_and_gradients():
    torch.manual_seed(31)
    model = CVPCustomCLIP.__new__(CVPCustomCLIP)
    nn.Module.__init__(model)
    model.prompt_learner = _FixedPromptLearner(torch.randn(3, 16))
    model.tokenized_prompts = None
    model.image_encoder = TimmViTVisualPromptEncoder(
        _TinyTimmVisual(),
        num_tokens=4,
        mode="deep",
        dropout=0.0,
        prompt_depth=8,
    )
    model.text_encoder = _FixedTextEncoder(torch.randn(3, 16))
    model.logit_scale = nn.Parameter(torch.tensor(0.7), requires_grad=False)
    model.dtype = torch.float32
    model.competitive_visual_prompt = VisualPrototypePrompt(
        prior_dim=16, bottleneck_dim=4, num_tokens=4, hidden_dim=32
    )
    model.competitive_visual_insert_layer = 8

    model.image_encoder.requires_grad_(False)
    model.image_encoder.visual_prompt.requires_grad_(True)
    logits, text_features, visual_features = model(
        torch.randn(2, 3, 32, 32), return_features=True
    )

    assert logits.shape == (2, 3)
    assert text_features.shape == (3, 16)
    assert visual_features.shape == (2, 3, 16)
    torch.testing.assert_close(
        logits,
        model.logit_scale.exp()
        * torch.einsum("bcd,cd->bc", visual_features, text_features),
    )
    torch.nn.functional.cross_entropy(logits, torch.tensor([0, 2])).backward()
    assert model.prompt_learner.prompts.grad.norm() > 0
    assert model.image_encoder.visual_prompt.prompt_embeddings.grad.norm() > 0
    assert model.competitive_visual_prompt.down_projection.weight.grad.norm() > 0
    assert model.competitive_visual_prompt.up_projection.weight.grad.norm() > 0
    assert all(
        parameter.grad is None
        for parameter in model.image_encoder.base_visual.parameters()
    )
    assert model.logit_scale.grad is None
