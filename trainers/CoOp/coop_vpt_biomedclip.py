"""CE-only CoOp with the project-local deep Visual Prompt Tuning adapter."""

import os.path as osp

import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint, load_pretrained_weights
from dassl.utils.torchtools import resume_from_checkpoint

from models.biomedclip_loader import load_biomedclip
from models.multitext_tcp import (
    AGGREGATION_MODES,
    CONNECTION_MODES,
    MultiTextTCPBertTextEncoder,
    build_frozen_description_bank,
    build_frozen_layer_description_bank,
)
from models.tcp import (
    TCPBertTextEncoder,
    build_frozen_class_prior,
    validate_tcp_checkpoint_state,
)
from models.text_vpt import BertTextDeepPromptEncoder
from trainers.CoOp.coop_biomedclip import CustomCLIP
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


def checkpoint_residual_scale(epoch, warmup_epochs):
    """Recover the train-time residual strength of a saved epoch."""

    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < 0:
        raise ValueError("Checkpoint epoch cannot be negative")
    if epoch == 0:
        return 0.0
    return 1.0 if warmup_epochs <= 0 else min(1.0, epoch / float(warmup_epochs))


def base_prompts_frozen(epoch, freeze_epochs):
    """Whether CoOp/VPT updates are held while the new TKE branch settles."""

    epoch = int(epoch)
    freeze_epochs = int(freeze_epochs)
    if epoch < 0 or freeze_epochs < 0:
        raise ValueError("Epoch and freeze duration must be non-negative")
    return epoch < freeze_epochs


def cosine_l2_prompt_anchor_loss(current, references, l2_weight=0.0):
    """Direction-and-scale trust region for warm-started prompt parameters."""

    if len(current) != len(references):
        raise ValueError("Current and reference prompt groups must match")
    l2_weight = float(l2_weight)
    if l2_weight < 0.0:
        raise ValueError("Prompt-anchor L2 weight must be non-negative")
    losses = []
    for parameter, reference in zip(current, references):
        parameter = parameter.float()
        reference = reference.float()
        loss = 1.0 - F.cosine_similarity(
            parameter.reshape(1, -1),
            reference.reshape(1, -1),
            dim=-1,
        ).mean()
        if l2_weight > 0.0:
            relative_l2 = (
                (parameter - reference).square().sum()
                / reference.square().sum().clamp_min(1e-12)
            )
            loss = loss + l2_weight * relative_l2
        losses.append(loss)
    if not losses:
        raise ValueError("At least one prompt group is required")
    return torch.stack(losses).sum()


def tcp_knowledge_loss(text_features, prior, mode="raw_cosine"):
    """Align raw medical semantics or only class-differential directions."""

    if mode == "centered_cosine":
        text_features = text_features.float() - text_features.float().mean(
            dim=0, keepdim=True
        )
        prior = prior.float() - prior.float().mean(dim=0, keepdim=True)
    elif mode != "raw_cosine":
        raise ValueError("Unknown TCP knowledge-loss mode: {}".format(mode))
    return 1.0 - F.cosine_similarity(text_features, prior, dim=-1).mean()


def prior_contrastive_loss(text_features, prior, temperature=0.1):
    """Match every learned class prototype against all frozen class priors."""

    if text_features.dim() != 2 or prior.dim() != 2:
        raise ValueError("Text features and priors must be rank-2 matrices")
    if text_features.shape != prior.shape:
        raise ValueError("Text features and class priors must have equal shape")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("Prior-contrastive temperature must be positive")
    learned = F.normalize(text_features.float(), dim=-1)
    fixed = F.normalize(prior.detach().float(), dim=-1)
    logits = learned @ fixed.t() / temperature
    labels = torch.arange(logits.shape[0], device=logits.device)
    return 0.5 * (
        F.cross_entropy(logits, labels)
        + F.cross_entropy(logits.t(), labels)
    )


def layer_token_alignment_loss(class_tokens, targets):
    """Align four TKE tokens with the five-group layer-8 description signal."""

    if class_tokens.shape != targets.shape or class_tokens.dim() != 3:
        raise ValueError("Class tokens and layer targets must have equal rank-3 shape")
    learned = class_tokens.float() - class_tokens.float().mean(dim=0, keepdim=True)
    fixed = targets.detach().float() - targets.detach().float().mean(
        dim=0, keepdim=True
    )
    return 1.0 - F.cosine_similarity(learned, fixed, dim=-1).mean()


def select_description_teacher_prototypes(
    image_features, description_bank, logit_scale, tau=1.5
):
    """BiomedCoOp-style robust selection over the 50 frozen descriptions."""

    if description_bank.dim() != 3:
        raise ValueError("description_bank must have [classes, descriptions, dim]")
    if image_features.shape[-1] != description_bank.shape[-1]:
        raise ValueError("Image and projected description dimensions disagree")
    logits = logit_scale * torch.einsum(
        "bd,ckd->bck", image_features, description_bank
    )
    scores = logits.max(dim=1).values.mean(dim=0)
    median = scores.median()
    mad = (scores - median).abs().median().clamp_min(1e-6)
    robust = (scores - median) / mad
    robust_std = robust.std(unbiased=False).clamp_min(1e-6)
    mask = ((robust - robust.mean()).abs() / robust_std) <= float(tau)
    if not bool(mask.any()):
        mask = torch.ones_like(mask, dtype=torch.bool)
    prototypes = F.normalize(description_bank[:, mask].mean(dim=1), dim=-1)
    return prototypes, mask


def description_distillation_loss(student_logits, teacher_logits, temperature=1.5):
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("Distillation temperature must be positive")
    return F.kl_div(
        F.log_softmax(student_logits / temperature, dim=-1),
        F.softmax(teacher_logits / temperature, dim=-1),
        reduction="batchmean",
    ) * (temperature ** 2)


def image_description_prior_loss(image_features, class_prior, labels, logit_scale):
    """Train the visual prompt against the frozen 50-description class prior."""

    if image_features.dim() != 2 or class_prior.dim() != 2:
        raise ValueError("Image features and class prior must be matrices")
    if image_features.shape[-1] != class_prior.shape[-1]:
        raise ValueError("Image and description-prior dimensions disagree")
    if labels.shape[0] != image_features.shape[0]:
        raise ValueError("Image batch and labels disagree")
    logits = (
        logit_scale
        * F.normalize(image_features.float(), dim=-1)
        @ F.normalize(class_prior.detach().float(), dim=-1).t()
    )
    return F.cross_entropy(logits, labels)


def class_balanced_cross_modal_prototype_loss(
    image_features, text_features, labels, temperature=0.1
):
    """Align batch class centroids and text prototypes in both directions.

    Every class present in the batch contributes once, independent of its
    sample count. The loss uses the same image/text features as the classifier
    and adds no inference-time module, prototype, or logits.
    """

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("Image and text features must be rank-2 matrices")
    if image_features.shape[-1] != text_features.shape[-1]:
        raise ValueError("Image and text feature dimensions disagree")
    if labels.dim() != 1 or labels.shape[0] != image_features.shape[0]:
        raise ValueError("Labels and image batch disagree")
    temperature = float(temperature)
    if temperature <= 0.0:
        raise ValueError("Cross-modal prototype temperature must be positive")
    classes = labels.unique(sorted=True)
    if classes.numel() == 0:
        raise ValueError("Cross-modal prototype loss requires a non-empty batch")
    if int(classes.min()) < 0 or int(classes.max()) >= text_features.shape[0]:
        raise ValueError("Labels are outside the text-prototype class range")

    normalized_images = F.normalize(image_features.float(), dim=-1)
    normalized_text = F.normalize(text_features.float(), dim=-1)
    centroids = torch.stack(
        [normalized_images[labels == class_id].mean(dim=0) for class_id in classes]
    )
    centroids = F.normalize(centroids, dim=-1)
    image_to_text = centroids @ normalized_text.t() / temperature
    text_to_image = normalized_text[classes] @ centroids.t() / temperature
    local_targets = torch.arange(classes.numel(), device=classes.device)
    return 0.5 * (
        F.cross_entropy(image_to_text, classes)
        + F.cross_entropy(text_to_image, local_targets)
    )


def class_balanced_hard_negative_margin_loss(
    image_features, text_features, labels, margin=0.05, temperature=0.02
):
    """Separate each image from its most-confusable wrong text prototype.

    Per-class averaging keeps the objective class balanced. This uses the same
    normalized features as the classifier and adds no inference-time branch,
    calibration, parameters, or second set of logits.
    """

    if image_features.dim() != 2 or text_features.dim() != 2:
        raise ValueError("Image and text features must be rank-2 matrices")
    if image_features.shape[-1] != text_features.shape[-1]:
        raise ValueError("Image and text feature dimensions disagree")
    if labels.dim() != 1 or labels.shape[0] != image_features.shape[0]:
        raise ValueError("Labels and image batch disagree")
    margin = float(margin)
    temperature = float(temperature)
    if margin < 0.0:
        raise ValueError("Hard-negative margin must be non-negative")
    if temperature <= 0.0:
        raise ValueError("Hard-negative temperature must be positive")
    if text_features.shape[0] < 2:
        raise ValueError("Hard-negative loss requires at least two classes")
    if int(labels.min()) < 0 or int(labels.max()) >= text_features.shape[0]:
        raise ValueError("Labels are outside the text-prototype class range")

    cosine = F.normalize(image_features.float(), dim=-1) @ F.normalize(
        text_features.float(), dim=-1
    ).t()
    positive = cosine.gather(1, labels[:, None]).squeeze(1)
    positive_mask = F.one_hot(
        labels, num_classes=text_features.shape[0]
    ).bool()
    hardest_negative = cosine.masked_fill(positive_mask, -torch.inf).max(
        dim=1
    ).values
    sample_loss = F.softplus(
        (hardest_negative - positive + margin) / temperature
    ) * temperature
    classes = labels.unique(sorted=True)
    return torch.stack(
        [sample_loss[labels == class_id].mean() for class_id in classes]
    ).mean()


class PromptParameterBundle(nn.Module):
    """Checkpoint only the trainable prompt modules, not frozen BiomedCLIP."""

    def __init__(
        self,
        prompt_learner,
        visual_prompt=None,
        text_prompt=None,
        tcp_prompt=None,
    ):
        super().__init__()
        self.prompt_learner = prompt_learner
        if visual_prompt is not None:
            self.visual_prompt = visual_prompt
        if text_prompt is not None:
            self.text_prompt = text_prompt
        if tcp_prompt is not None:
            self.tcp = tcp_prompt


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Train CoOp and VPT prompts with one AdamW and one shared learning rate."""

    def check_cfg(self, cfg):
        trainer_cfg = cfg.TRAINER.COOPVPT
        assert trainer_cfg.PREC in {"fp16", "fp32", "amp"}
        assert trainer_cfg.VPT_MODE == "deep"
        assert trainer_cfg.VPT_INIT == "uniform"
        assert trainer_cfg.OPTIM.NAME.lower() == "adamw"
        if trainer_cfg.TEXT_VPT_ENABLED:
            assert trainer_cfg.TEXT_VPT_MODE == "deep"
            assert trainer_cfg.TEXT_VPT_INIT == "normal"
            assert trainer_cfg.TEXT_VPT_N_CTX > 0
        tcp_cfg = cfg.TRAINER.TCP
        if tcp_cfg.ENABLED:
            assert not trainer_cfg.TEXT_VPT_ENABLED, (
                "External Text VPT must stay disabled; multi-text TCP owns its "
                "internal Deep Text Prompt slots"
            )
            assert tcp_cfg.NUM_TOKENS > 0
            assert tcp_cfg.BOTTLENECK_DIM > 0
            assert tcp_cfg.INSERT_LAYER >= 0
            assert tcp_cfg.PRIOR_SOURCE in {"single_template", "biomedcoop_50"}
            if tcp_cfg.PRIOR_SOURCE == "single_template":
                assert tcp_cfg.FUSION_MODE in {"replace", "gated_residual"}
                if tcp_cfg.FUSION_MODE == "replace":
                    assert float(tcp_cfg.FUSION_WEIGHT) == 1.0
                else:
                    assert 0.0 < float(tcp_cfg.FUSION_WEIGHT) < 1.0
            else:
                assert int(tcp_cfg.DESCRIPTION_COUNT) == 50
                assert int(tcp_cfg.DESCRIPTION_BATCH_SIZE) > 0
                assert tcp_cfg.PRIOR_REPRESENTATION in {
                    "projected_text",
                    "layer_cls",
                }
                assert tcp_cfg.AGGREGATION in AGGREGATION_MODES
                assert tcp_cfg.CONNECTION in CONNECTION_MODES
                assert float(tcp_cfg.CONSENSUS_TEMPERATURE) > 0.0
                if tcp_cfg.CONNECTION in {
                    "late_residual",
                    "late_norm_residual",
                    "late_centered_norm_residual",
                    "late_centered_classlayer_norm_residual",
                    "all_residual",
                    "inplace_once_norm_residual",
                    "inplace_once_centered_norm_residual",
                    "inplace_once_centered_classgate_norm_residual",
                    "inplace_deep_centered_norm_residual",
                }:
                    assert 0.0 < float(tcp_cfg.GATE_INIT) < 1.0
            assert float(tcp_cfg.KG_WEIGHT) >= 0.0
            assert tcp_cfg.KG_MODE in {"raw_cosine", "centered_cosine"}
            assert int(tcp_cfg.RESIDUAL_WARMUP_EPOCHS) >= 0
            assert float(tcp_cfg.PROMPT_ANCHOR_WEIGHT) >= 0.0
            assert float(tcp_cfg.PROMPT_ANCHOR_L2_WEIGHT) >= 0.0
            assert float(tcp_cfg.DESCRIPTION_KD_WEIGHT) >= 0.0
            assert float(tcp_cfg.DESCRIPTION_KD_TEMPERATURE) > 0.0
            assert float(tcp_cfg.DESCRIPTION_KD_TAU) > 0.0
            assert float(tcp_cfg.IMAGE_PRIOR_WEIGHT) >= 0.0
            assert float(tcp_cfg.PRIOR_CONTRASTIVE_WEIGHT) >= 0.0
            assert float(tcp_cfg.PRIOR_CONTRASTIVE_TEMPERATURE) > 0.0
            assert float(tcp_cfg.LAYER_TOKEN_ALIGNMENT_WEIGHT) >= 0.0
            assert float(tcp_cfg.CROSS_MODAL_PROTO_WEIGHT) >= 0.0
            assert float(tcp_cfg.CROSS_MODAL_PROTO_TEMPERATURE) > 0.0
            assert float(tcp_cfg.HARD_NEGATIVE_MARGIN_WEIGHT) >= 0.0
            assert float(tcp_cfg.HARD_NEGATIVE_MARGIN) >= 0.0
            assert float(tcp_cfg.HARD_NEGATIVE_TEMPERATURE) > 0.0
            assert int(tcp_cfg.BASE_PROMPT_FREEZE_EPOCHS) >= 0
            if float(tcp_cfg.DESCRIPTION_KD_WEIGHT) > 0.0:
                assert tcp_cfg.PRIOR_SOURCE == "biomedcoop_50"
            if float(tcp_cfg.IMAGE_PRIOR_WEIGHT) > 0.0:
                assert tcp_cfg.PRIOR_SOURCE == "biomedcoop_50"
            if float(tcp_cfg.PRIOR_CONTRASTIVE_WEIGHT) > 0.0:
                assert tcp_cfg.PRIOR_SOURCE == "biomedcoop_50"
            if float(tcp_cfg.LAYER_TOKEN_ALIGNMENT_WEIGHT) > 0.0:
                assert tcp_cfg.PRIOR_SOURCE == "biomedcoop_50"
                assert tcp_cfg.PRIOR_REPRESENTATION == "layer_cls"
            if float(tcp_cfg.PROMPT_ANCHOR_WEIGHT) > 0.0:
                assert tcp_cfg.INIT_BASELINE_CHECKPOINT, (
                    "Prompt anchoring requires a baseline initialization checkpoint"
                )
            if bool(tcp_cfg.EVAL_WARMSTART):
                assert tcp_cfg.INIT_BASELINE_CHECKPOINT, (
                    "Epoch-0 validation requires a baseline initialization checkpoint"
                )
            if tcp_cfg.INIT_BASELINE_CHECKPOINT:
                assert trainer_cfg.VPT_ENABLED

    def build_model(self):
        cfg = self.cfg
        trainer_cfg = cfg.TRAINER.COOPVPT
        classnames = self.dm.dataset.classnames

        print("Loading and validating BiomedCLIP")
        biomedclip_model, _ = load_biomedclip(
            vpt_enabled=trainer_cfg.VPT_ENABLED,
            vpt_mode="deep",
            vpt_num_tokens=trainer_cfg.VPT_N_CTX,
            vpt_dropout=trainer_cfg.VPT_DROPOUT,
        )
        if trainer_cfg.PREC in {"fp32", "amp"}:
            biomedclip_model.float()

        print(
            "Building CoOp{}{}{}".format(
                " + VPT-Deep" if trainer_cfg.VPT_ENABLED else "",
                " + Text-Deep-Prompt" if trainer_cfg.TEXT_VPT_ENABLED else "",
                " + TCP" if cfg.TRAINER.TCP.ENABLED else "",
            )
        )
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        self.vpt_enabled = bool(trainer_cfg.VPT_ENABLED)
        self.text_vpt_enabled = bool(trainer_cfg.TEXT_VPT_ENABLED)
        self.tcp_enabled = bool(cfg.TRAINER.TCP.ENABLED)
        self.tcp_multitext = bool(
            self.tcp_enabled
            and cfg.TRAINER.TCP.PRIOR_SOURCE == "biomedcoop_50"
        )
        self._gradient_audit_complete = False
        self._prompt_anchor = None

        text_prompt = None
        if self.text_vpt_enabled:
            self.model.text_encoder = BertTextDeepPromptEncoder(
                biomedclip_model.text,
                num_tokens=trainer_cfg.TEXT_VPT_N_CTX,
                dropout=trainer_cfg.TEXT_VPT_DROPOUT,
                init=trainer_cfg.TEXT_VPT_INIT,
            )
            text_prompt = self.model.text_encoder.text_prompt
            print(
                "Text Deep Prompt: depth={} tokens/layer={} hidden={} shape={}".format(
                    self.model.text_encoder.depth,
                    self.model.text_encoder.num_prompt_tokens,
                    self.model.text_encoder.embed_dim,
                    tuple(text_prompt.prompt_embeddings.shape),
                )
            )

        tcp_prompt = None
        if self.tcp_enabled:
            tcp_cfg = cfg.TRAINER.TCP
            if int(tcp_cfg.NUM_TOKENS) != int(self.model.prompt_learner.n_ctx):
                raise ValueError(
                    "TCP token count {} must match CoOp context count {} for replacement".format(
                        tcp_cfg.NUM_TOKENS, self.model.prompt_learner.n_ctx
                    )
                )
            if self.tcp_multitext:
                projected_bank, descriptions = build_frozen_description_bank(
                    biomedclip_model,
                    self.model.prompt_learner.tokenizer,
                    classnames,
                    BIOMEDCOOP_TEMPLATES,
                    expected_count=tcp_cfg.DESCRIPTION_COUNT,
                    batch_size=tcp_cfg.DESCRIPTION_BATCH_SIZE,
                    cache_path=tcp_cfg.DESCRIPTION_CACHE or None,
                )
                description_bank = projected_bank
                class_prior = None
                if tcp_cfg.PRIOR_REPRESENTATION == "layer_cls":
                    description_bank, layer_descriptions = (
                        build_frozen_layer_description_bank(
                            biomedclip_model,
                            self.model.prompt_learner.tokenizer,
                            classnames,
                            BIOMEDCOOP_TEMPLATES,
                            insert_layer=tcp_cfg.INSERT_LAYER,
                            expected_count=tcp_cfg.DESCRIPTION_COUNT,
                            batch_size=tcp_cfg.DESCRIPTION_BATCH_SIZE,
                            cache_path=tcp_cfg.LAYER_DESCRIPTION_CACHE or None,
                        )
                    )
                    if layer_descriptions != descriptions:
                        raise RuntimeError("Projected and layer description order mismatch")
                    class_prior = F.normalize(projected_bank.mean(dim=1), dim=-1)
                self.model.text_encoder = MultiTextTCPBertTextEncoder(
                    biomedclip_model.text,
                    description_bank=description_bank,
                    descriptions=descriptions,
                    classnames=classnames,
                    num_tokens=tcp_cfg.NUM_TOKENS,
                    bottleneck_dim=tcp_cfg.BOTTLENECK_DIM,
                    insert_layer=tcp_cfg.INSERT_LAYER,
                    aggregation=tcp_cfg.AGGREGATION,
                    connection=tcp_cfg.CONNECTION,
                    consensus_temperature=tcp_cfg.CONSENSUS_TEMPERATURE,
                    gate_init=tcp_cfg.GATE_INIT,
                    class_prior=class_prior,
                    prior_representation=tcp_cfg.PRIOR_REPRESENTATION,
                    projected_description_bank=projected_bank,
                )
            else:
                class_prior = build_frozen_class_prior(
                    biomedclip_model,
                    self.model.prompt_learner.tokenizer,
                    classnames,
                    tcp_cfg.PRIOR_TEMPLATE,
                )
                self.model.text_encoder = TCPBertTextEncoder(
                    biomedclip_model.text,
                    class_prior=class_prior,
                    classnames=classnames,
                    template=tcp_cfg.PRIOR_TEMPLATE,
                    num_tokens=tcp_cfg.NUM_TOKENS,
                    bottleneck_dim=tcp_cfg.BOTTLENECK_DIM,
                    insert_layer=tcp_cfg.INSERT_LAYER,
                    fusion_mode=tcp_cfg.FUSION_MODE,
                    fusion_weight=tcp_cfg.FUSION_WEIGHT,
                )
            tcp_prompt = self.model.text_encoder.tcp_prompt
            print(
                "TCP TKE: prior={} -> {} -> {}x{}; inject before BERT block {}; {}".format(
                    self.model.text_encoder.prior_dim,
                    self.model.text_encoder.bottleneck_dim,
                    self.model.text_encoder.num_tokens,
                    self.model.text_encoder.hidden_dim,
                    self.model.text_encoder.insert_layer + 1,
                    (
                        "aggregation={} connection={} descriptions/class={}".format(
                            self.model.text_encoder.aggregation,
                            self.model.text_encoder.connection,
                            self.model.text_encoder.description_count,
                        )
                        if self.tcp_multitext
                        else "single-template original TCP"
                    ),
                )
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.prompt_learner.ctx.requires_grad_(True)

        visual_prompt = None
        if self.vpt_enabled:
            visual_prompt = self.model.image_encoder.visual_prompt
            for parameter in visual_prompt.parameters():
                parameter.requires_grad_(True)

        if text_prompt is not None:
            for parameter in text_prompt.parameters():
                parameter.requires_grad_(True)
        if tcp_prompt is not None:
            for parameter in tcp_prompt.parameters():
                parameter.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.eval()
        self.model.prompt_learner.train()
        if visual_prompt is not None:
            visual_prompt.train()
        if text_prompt is not None:
            text_prompt.train()
        if tcp_prompt is not None:
            tcp_prompt.train()

        self.prompt_parameters = PromptParameterBundle(
            self.model.prompt_learner, visual_prompt, text_prompt, tcp_prompt
        )
        trainable_parameters = [self.model.prompt_learner.ctx]
        if visual_prompt is not None:
            trainable_parameters.extend(list(visual_prompt.parameters()))
        if text_prompt is not None:
            trainable_parameters.extend(list(text_prompt.parameters()))
        if tcp_prompt is not None:
            trainable_parameters.extend(list(tcp_prompt.parameters()))

        # One optimizer and one parameter group intentionally cover all enabled
        # prompt branches. The shared LR is supplied by TRAINER.COOPVPT.OPTIM.LR.
        self.optim = build_optimizer(trainable_parameters, trainer_cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, trainer_cfg.OPTIM)
        self.register_model(
            "prompt_parameters", self.prompt_parameters, self.optim, self.sched
        )

        self._audit_trainable_parameters()
        self.scaler = GradScaler() if trainer_cfg.PREC == "amp" else None

        device_count = torch.cuda.device_count()
        if device_count > 1:
            print("Multiple GPUs detected (n_gpus={}), using DataParallel".format(device_count))
            self.model = nn.DataParallel(self.model)

    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _audit_trainable_parameters(self):
        trainable = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        expected = {"prompt_learner.ctx"}
        if self.vpt_enabled:
            expected.add("image_encoder.visual_prompt.prompt_embeddings")
        if self.text_vpt_enabled:
            expected.add("text_encoder.text_prompt.prompt_embeddings")
        if self.tcp_enabled:
            expected.update(
                "text_encoder.tcp_prompt.{}".format(name)
                for name, _ in self._unwrapped_model().text_encoder.tcp_prompt.named_parameters()
            )
        if set(trainable) != expected:
            raise RuntimeError(
                "Unexpected trainable parameters: expected {}, got {}".format(
                    sorted(expected), sorted(trainable)
                )
            )

        if len(self.optim.param_groups) != 1:
            raise RuntimeError(
                "Expected one AdamW parameter group, got {}".format(
                    len(self.optim.param_groups)
                )
            )
        optimizer_ids = {
            id(parameter)
            for group in self.optim.param_groups
            for parameter in group["params"]
        }
        trainable_ids = {id(parameter) for parameter in trainable.values()}
        if optimizer_ids != trainable_ids:
            raise RuntimeError(
                "The single optimizer must contain every and only trainable prompt parameter"
            )

        print("Optimizer: {}".format(self.optim.__class__.__name__))
        print("Shared learning rate: {}".format(self.optim.param_groups[0]["lr"]))
        print("Trainable parameters:")
        for name, parameter in trainable.items():
            print("  {} shape={} count={:,}".format(name, tuple(parameter.shape), parameter.numel()))
        print( "Total trainable parameters: {:,}".format(sum(parameter.numel() for parameter in trainable.values())))

    def set_model_mode(self, mode="train", names=None):
        model = self._unwrapped_model()
        model.eval()
        if mode == "train":
            model.prompt_learner.train()
            if self.vpt_enabled:
                model.image_encoder.visual_prompt.train()
            if self.text_vpt_enabled:
                model.text_encoder.text_prompt.train()
            if self.tcp_enabled:
                model.text_encoder.tcp_prompt.train()
        elif mode in {"test", "eval"}:
            model.prompt_learner.eval()
            if self.vpt_enabled:
                model.image_encoder.visual_prompt.eval()
            if self.text_vpt_enabled:
                model.text_encoder.text_prompt.eval()
            if self.tcp_enabled:
                model.text_encoder.tcp_prompt.eval()
        else:
            raise KeyError(mode)

    def before_train(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        super().before_train()
        baseline_checkpoint = self.cfg.TRAINER.TCP.INIT_BASELINE_CHECKPOINT
        if self.start_epoch == 0 and baseline_checkpoint:
            self._initialize_from_baseline_checkpoint(baseline_checkpoint)
        if (
            baseline_checkpoint
            and float(self.cfg.TRAINER.TCP.PROMPT_ANCHOR_WEIGHT) > 0.0
        ):
            self._load_prompt_anchor(baseline_checkpoint)
        if (
            self.start_epoch == 0
            and baseline_checkpoint
            and bool(self.cfg.TRAINER.TCP.EVAL_WARMSTART)
        ):
            self._unwrapped_model().text_encoder.tcp_prompt.set_residual_scale(0.0)
            print("Evaluate exact zero-residual warm-start as epoch 0")
            self.evaluate_and_track_best(checkpoint_epoch=-1, record_epoch=0)

    def after_train(self):
        super().after_train()
        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            print("Peak CUDA memory allocated: {:.2f} MiB".format(peak_mib))

    def before_epoch(self):
        if not self.tcp_enabled:
            return
        warmup_epochs = int(self.cfg.TRAINER.TCP.RESIDUAL_WARMUP_EPOCHS)
        scale = (
            1.0
            if warmup_epochs <= 0
            else min(1.0, float(self.epoch + 1) / float(warmup_epochs))
        )
        self._unwrapped_model().text_encoder.tcp_prompt.set_residual_scale(scale)
        print("TCP residual scale: {:.6f}".format(scale))
        if base_prompts_frozen(
            self.epoch, self.cfg.TRAINER.TCP.BASE_PROMPT_FREEZE_EPOCHS
        ):
            print("Hold CoOp/VPT updates during TKE settling phase")

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        self.model_zero_grad()

        if self.cfg.TRAINER.COOPVPT.PREC == "amp":
            with autocast():
                (
                    output,
                    loss,
                    loss_ce,
                    loss_kg,
                    loss_anchor,
                    loss_kd,
                    loss_image_prior,
                    loss_prior_contrastive,
                    loss_layer_token_alignment,
                    loss_cross_modal_proto,
                    loss_hard_negative_margin,
                ) = self._compute_training_loss(image, label)
            self.scaler.scale(loss).backward()
            self._audit_gradients_once()
            self._mask_base_prompt_gradients()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            (
                output,
                loss,
                loss_ce,
                loss_kg,
                loss_anchor,
                loss_kd,
                loss_image_prior,
                loss_prior_contrastive,
                loss_layer_token_alignment,
                loss_cross_modal_proto,
                loss_hard_negative_margin,
            ) = self._compute_training_loss(image, label)
            self.model_backward(loss)
            self._audit_gradients_once()
            self._mask_base_prompt_gradients()
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "loss_ce": loss_ce.item(),
            "loss_kg": loss_kg.item(),
            "loss_anchor": loss_anchor.item(),
            "loss_kd": loss_kd.item(),
            "loss_image_prior": loss_image_prior.item(),
            "loss_prior_contrastive": loss_prior_contrastive.item(),
            "loss_layer_token_alignment": loss_layer_token_alignment.item(),
            "loss_cross_modal_proto": loss_cross_modal_proto.item(),
            "loss_hard_negative_margin": loss_hard_negative_margin.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "lr": self.optim.param_groups[0]["lr"],
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def _mask_base_prompt_gradients(self):
        if not self.tcp_enabled or not base_prompts_frozen(
            self.epoch, self.cfg.TRAINER.TCP.BASE_PROMPT_FREEZE_EPOCHS
        ):
            return
        model = self._unwrapped_model()
        model.prompt_learner.ctx.grad = None
        if self.vpt_enabled:
            for parameter in model.image_encoder.visual_prompt.parameters():
                parameter.grad = None
        tcp_prompt = getattr(getattr(model, "text_encoder", None), "tcp_prompt", None)
        internal_text_prompt = getattr(tcp_prompt, "text_prompt", None)
        if internal_text_prompt is not None:
            for parameter in internal_text_prompt.parameters():
                parameter.grad = None

    def _compute_training_loss(self, image, label):
        if not self.tcp_enabled:
            output = self.model(image)
            loss_ce = F.cross_entropy(output, label)
            zero = loss_ce.new_zeros(())
            return (
                output,
                loss_ce,
                loss_ce,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
                zero,
            )

        output, text_features, image_features = self.model(
            image, return_features=True
        )
        model = self._unwrapped_model()
        prior = model.text_encoder.class_prior.to(
            device=text_features.device, dtype=text_features.dtype
        )
        loss_ce = F.cross_entropy(output, label)
        loss_kg = tcp_knowledge_loss(
            text_features, prior, mode=self.cfg.TRAINER.TCP.KG_MODE
        )
        loss_anchor = self._prompt_anchor_loss()
        loss_kd = output.new_zeros(())
        loss_image_prior = output.new_zeros(())
        loss_prior_contrastive = output.new_zeros(())
        loss_layer_token_alignment = output.new_zeros(())
        loss_cross_modal_proto = output.new_zeros(())
        loss_hard_negative_margin = output.new_zeros(())
        kd_weight = float(self.cfg.TRAINER.TCP.DESCRIPTION_KD_WEIGHT)
        if kd_weight > 0.0:
            bank = model.text_encoder.projected_description_bank.to(
                device=image_features.device, dtype=image_features.dtype
            )
            with torch.no_grad():
                teacher_prototypes, _ = select_description_teacher_prototypes(
                    image_features.detach(),
                    bank,
                    model.logit_scale.exp().detach(),
                    tau=self.cfg.TRAINER.TCP.DESCRIPTION_KD_TAU,
                )
                teacher_logits = (
                    model.logit_scale.exp().detach()
                    * image_features.detach()
                    @ teacher_prototypes.t()
                )
            loss_kd = description_distillation_loss(
                output,
                teacher_logits,
                temperature=self.cfg.TRAINER.TCP.DESCRIPTION_KD_TEMPERATURE,
            )
        image_prior_weight = float(self.cfg.TRAINER.TCP.IMAGE_PRIOR_WEIGHT)
        if image_prior_weight > 0.0:
            loss_image_prior = image_description_prior_loss(
                image_features,
                prior,
                label,
                model.logit_scale.exp().detach(),
            )
        prior_contrastive_weight = float(
            self.cfg.TRAINER.TCP.PRIOR_CONTRASTIVE_WEIGHT
        )
        if prior_contrastive_weight > 0.0:
            loss_prior_contrastive = prior_contrastive_loss(
                text_features,
                prior,
                temperature=self.cfg.TRAINER.TCP.PRIOR_CONTRASTIVE_TEMPERATURE,
            )
        layer_token_weight = float(
            self.cfg.TRAINER.TCP.LAYER_TOKEN_ALIGNMENT_WEIGHT
        )
        if layer_token_weight > 0.0:
            loss_layer_token_alignment = layer_token_alignment_loss(
                model.text_encoder.last_class_tokens(),
                model.text_encoder.grouped_layer_targets(),
            )
        cross_modal_weight = float(
            self.cfg.TRAINER.TCP.CROSS_MODAL_PROTO_WEIGHT
        )
        if cross_modal_weight > 0.0:
            loss_cross_modal_proto = class_balanced_cross_modal_prototype_loss(
                image_features,
                text_features,
                label,
                temperature=self.cfg.TRAINER.TCP.CROSS_MODAL_PROTO_TEMPERATURE,
            )
        hard_negative_weight = float(
            self.cfg.TRAINER.TCP.HARD_NEGATIVE_MARGIN_WEIGHT
        )
        if hard_negative_weight > 0.0:
            loss_hard_negative_margin = class_balanced_hard_negative_margin_loss(
                image_features,
                text_features,
                label,
                margin=self.cfg.TRAINER.TCP.HARD_NEGATIVE_MARGIN,
                temperature=self.cfg.TRAINER.TCP.HARD_NEGATIVE_TEMPERATURE,
            )
        loss = (
            loss_ce
            + float(self.cfg.TRAINER.TCP.KG_WEIGHT) * loss_kg
            + float(self.cfg.TRAINER.TCP.PROMPT_ANCHOR_WEIGHT) * loss_anchor
            + kd_weight * loss_kd
            + image_prior_weight * loss_image_prior
            + prior_contrastive_weight * loss_prior_contrastive
            + layer_token_weight * loss_layer_token_alignment
            + cross_modal_weight * loss_cross_modal_proto
            + hard_negative_weight * loss_hard_negative_margin
        )
        return (
            output,
            loss,
            loss_ce,
            loss_kg,
            loss_anchor,
            loss_kd,
            loss_image_prior,
            loss_prior_contrastive,
            loss_layer_token_alignment,
            loss_cross_modal_proto,
            loss_hard_negative_margin,
        )

    def _prompt_anchor_loss(self):
        if self._prompt_anchor is None:
            return self._unwrapped_model().prompt_learner.ctx.new_zeros(())
        model = self._unwrapped_model()
        current = [
            model.prompt_learner.ctx,
            model.image_encoder.visual_prompt.prompt_embeddings,
        ]
        tcp_prompt = getattr(getattr(model, "text_encoder", None), "tcp_prompt", None)
        internal_text_prompt = getattr(tcp_prompt, "text_prompt", None)
        if internal_text_prompt is not None:
            current.append(internal_text_prompt.prompt_embeddings)
        return cosine_l2_prompt_anchor_loss(
            tuple(current),
            self._prompt_anchor,
            l2_weight=self.cfg.TRAINER.TCP.PROMPT_ANCHOR_L2_WEIGHT,
        )

    def _audit_gradients_once(self):
        if self._gradient_audit_complete or not self.tcp_enabled:
            return
        model = self._unwrapped_model()
        required_modules = {
            "CoOp": [model.prompt_learner.ctx],
            "TKE": list(model.text_encoder.tcp_prompt.parameters()),
        }
        if self.vpt_enabled:
            required_modules["VPT"] = list(model.image_encoder.visual_prompt.parameters())
        norms = {}
        for branch, parameters in required_modules.items():
            norm = sum(
                float(parameter.grad.detach().float().norm().item())
                for parameter in parameters
                if parameter.grad is not None
            )
            if norm <= 0.0:
                raise RuntimeError("Gradient audit failed: {} has zero gradient".format(branch))
            norms[branch] = norm

        trainable_ids = {
            id(parameter)
            for group in self.optim.param_groups
            for parameter in group["params"]
        }
        unexpected = [
            name
            for name, parameter in model.named_parameters()
            if id(parameter) not in trainable_ids and parameter.grad is not None
        ]
        if unexpected:
            raise RuntimeError(
                "Gradient audit failed: frozen backbone gradients found for {}".format(
                    unexpected
                )
            )
        print("TCP gradient audit passed: {}".format(norms))
        self._gradient_audit_complete = True

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def resume_model_if_exist(self, directory):
        prompt_dir = osp.join(directory, "prompt_parameters")
        if not osp.exists(osp.join(prompt_dir, "checkpoint")):
            print("No complete prompt checkpoint found, train from scratch")
            return 0
        if self.tcp_enabled:
            with open(osp.join(prompt_dir, "checkpoint"), "r") as file:
                checkpoint_name = file.readlines()[0].strip("\n")
            checkpoint = load_checkpoint(osp.join(prompt_dir, checkpoint_name))
            self._validate_tcp_state(checkpoint["state_dict"])
        return resume_from_checkpoint(
            prompt_dir, self.prompt_parameters, self.optim, self.sched
        )

    @staticmethod
    def _load_prompt_learner_state(module, state_dict):
        state_dict = dict(state_dict)
        state_dict.pop("token_prefix", None)
        state_dict.pop("token_suffix", None)
        module.load_state_dict(state_dict, strict=False)

    @staticmethod
    def _validated_baseline_prompt_tensors(bundle, state_dict):
        """Validate prompt tensors used to warm-start the active TCP connection.

        In-place TCP connections reproduce a plain CoOp+VPT checkpoint when
        their residual scale is zero. Connections with internal Deep Text
        Prompt slots instead reproduce a CoOp+VPT+TextPrompt checkpoint, so
        the baseline text prompt is required and mapped into
        ``tcp.text_prompt``. A TCP checkpoint is never accepted as a baseline.
        """

        if any(key.startswith("tcp.") for key in state_dict):
            raise RuntimeError(
                "Baseline initialization requires a non-TCP prompt checkpoint"
            )
        tcp_prompt = getattr(bundle, "tcp", None)
        internal_text_prompt = getattr(tcp_prompt, "text_prompt", None)
        needs_text_prompt = internal_text_prompt is not None
        has_text_prompt = any(key.startswith("text_prompt.") for key in state_dict)
        if needs_text_prompt and not has_text_prompt:
            raise RuntimeError(
                "This TCP connection requires a CoOp+VPT+TextPrompt baseline checkpoint"
            )
        if not needs_text_prompt and has_text_prompt:
            raise RuntimeError(
                "This in-place TCP connection requires a plain CoOp+VPT checkpoint"
            )

        source_to_target = [
            ("prompt_learner.ctx", "prompt_learner.ctx"),
            ("visual_prompt.prompt_embeddings", "visual_prompt.prompt_embeddings"),
        ]
        if needs_text_prompt:
            source_to_target.append(
                ("text_prompt.prompt_embeddings", "tcp.text_prompt.prompt_embeddings")
            )
        current = bundle.state_dict()
        for source_key, target_key in source_to_target:
            if source_key not in state_dict:
                raise RuntimeError(
                    "Baseline initialization checkpoint is missing {}".format(source_key)
                )
            if target_key not in current or tuple(current[target_key].shape) != tuple(
                state_dict[source_key].shape
            ):
                raise RuntimeError(
                    "Baseline initialization shape mismatch for {} -> {}: expected {}, got {}".format(
                        source_key,
                        target_key,
                        tuple(current[target_key].shape) if target_key in current else None,
                        tuple(state_dict[source_key].shape),
                    )
                )
            if not torch.isfinite(state_dict[source_key]).all():
                raise RuntimeError(
                    "Baseline initialization contains non-finite values for {}".format(
                        source_key
                    )
                )
        return tuple(state_dict[source] for source, _ in source_to_target)

    @staticmethod
    def _initialize_bundle_from_baseline_state(bundle, state_dict):
        """Copy the baseline prompt branches represented by the TCP connection."""

        values = CoOpVPT_BiomedCLIP._validated_baseline_prompt_tensors(
            bundle, state_dict
        )
        with torch.no_grad():
            bundle.prompt_learner.ctx.copy_(values[0])
            bundle.visual_prompt.prompt_embeddings.copy_(values[1])
            internal_text_prompt = getattr(
                getattr(bundle, "tcp", None), "text_prompt", None
            )
            if internal_text_prompt is not None:
                internal_text_prompt.prompt_embeddings.copy_(values[2])

    def _initialize_from_baseline_checkpoint(self, path):
        checkpoint = load_checkpoint(str(path))
        self._initialize_bundle_from_baseline_state(
            self.prompt_parameters, checkpoint["state_dict"]
        )
        internal_text_prompt = getattr(
            getattr(self.prompt_parameters, "tcp", None), "text_prompt", None
        )
        copied = "CoOp, Visual VPT, and Text Deep Prompt" if internal_text_prompt is not None else "CoOp and Visual VPT"
        print(
            'Initialized {} from baseline "{}" (epoch={}); '
            "TCP-specific parameters and optimizer state remain new".format(
                copied, path, checkpoint["epoch"]
            )
        )

    def _load_prompt_anchor(self, path):
        checkpoint = load_checkpoint(str(path))
        values = self._validated_baseline_prompt_tensors(
            self.prompt_parameters, checkpoint["state_dict"]
        )
        self._prompt_anchor = tuple(
            value.detach().clone().to(self.device) for value in values
        )
        print(
            'Prompt anchor loaded from "{}" with weight {}'.format(
                path, self.cfg.TRAINER.TCP.PROMPT_ANCHOR_WEIGHT
            )
        )

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-{}".format(epoch)
        prompt_dir = osp.join(directory, "prompt_parameters")
        prompt_path = osp.join(prompt_dir, model_file)

        if osp.exists(prompt_path):
            checkpoint_data = load_checkpoint(prompt_path)
            if self.tcp_enabled:
                self._validate_tcp_state(checkpoint_data["state_dict"])
            elif any(key.startswith("tcp.") for key in checkpoint_data["state_dict"]):
                raise RuntimeError("A TCP checkpoint cannot be loaded with TCP disabled")
            if self.text_vpt_enabled and not any(
                key.startswith("text_prompt.")
                for key in checkpoint_data["state_dict"]
            ):
                raise RuntimeError(
                    "The checkpoint does not contain text Deep Prompt parameters; "
                    "resume the text-Deep experiment from its own checkpoint."
                )
            print(
                'Loading prompt_parameters from "{}" (epoch={})'.format(
                    prompt_path, checkpoint_data["epoch"]
                )
            )
            self.prompt_parameters.load_state_dict(
                checkpoint_data["state_dict"], strict=True
            )
            self._restore_checkpoint_residual_scale(checkpoint_data)
            return

        # Backward-compatible explicit extraction of a text prompt from old
        # CoOp checkpoints. A VPT prompt is intentionally not fabricated from
        # an old checkpoint and remains at its configured initialization.
        old_path = osp.join(directory, "prompt_learner", model_file)
        if osp.exists(old_path):
            if self.text_vpt_enabled or self.tcp_enabled:
                raise RuntimeError(
                    "A legacy CoOp checkpoint cannot restore the enabled text adapter; "
                    "use a complete prompt_parameters checkpoint from this experiment."
                )
            checkpoint_data = load_checkpoint(old_path)
            print(
                'Loading legacy prompt_learner from "{}" (epoch={})'.format(
                    old_path, checkpoint_data["epoch"]
                )
            )
            self._load_prompt_learner_state(
                self.model.prompt_learner, checkpoint_data["state_dict"]
            )
            return

        raise FileNotFoundError(
            'Prompt checkpoint not found at "{}" or "{}"'.format(
                prompt_path, old_path
            )
        )

    def _validate_tcp_state(self, state_dict):
        if not self.tcp_enabled:
            return
        validate_tcp_checkpoint_state(
            state_dict,
            self._unwrapped_model().text_encoder.tcp_prompt,
            prefix="tcp.",
        )

    def _restore_checkpoint_residual_scale(self, checkpoint):
        if not self.tcp_enabled:
            return
        scale = checkpoint_residual_scale(
            checkpoint["epoch"],
            self.cfg.TRAINER.TCP.RESIDUAL_WARMUP_EPOCHS,
        )
        self._unwrapped_model().text_encoder.tcp_prompt.set_residual_scale(scale)
        print(
            "Restored TCP residual scale {:.6f} for checkpoint epoch {}".format(
                scale, checkpoint["epoch"]
            )
        )

    def load_prompt_checkpoint(self, path):
        """Load one prompt bundle with TCP compatibility validation."""

        checkpoint = load_checkpoint(str(path))
        self._validate_tcp_state(checkpoint["state_dict"])
        self.prompt_parameters.load_state_dict(checkpoint["state_dict"], strict=True)
        self._restore_checkpoint_residual_scale(checkpoint)
        return checkpoint
