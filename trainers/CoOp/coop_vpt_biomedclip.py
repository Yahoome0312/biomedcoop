"""CoOp/VPT trainer with the single retained LayerBasis + XProto TCP path."""

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
    MultiTextTCPBertTextEncoder,
    build_frozen_description_bank,
    build_frozen_layer_description_bank,
    validate_tcp_checkpoint_state,
)
from models.text_vpt import BertTextDeepPromptEncoder
from trainers.CoOp.coop_biomedclip import CustomCLIP
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


def checkpoint_residual_scale(epoch, warmup_epochs):
    """Recover the residual strength used when a checkpoint was saved."""

    epoch = int(epoch)
    warmup_epochs = int(warmup_epochs)
    if epoch < 0:
        raise ValueError("Checkpoint epoch cannot be negative")
    if epoch == 0:
        return 0.0
    return 1.0 if warmup_epochs <= 0 else min(1.0, epoch / float(warmup_epochs))


def cosine_l2_prompt_anchor_loss(current, references, l2_weight=0.0):
    """Direction-and-scale trust region around the TextDeep warm start."""

    if len(current) != len(references):
        raise ValueError("Current and reference prompt groups must match")
    if float(l2_weight) < 0.0:
        raise ValueError("Prompt-anchor L2 weight must be non-negative")
    losses = []
    for parameter, reference in zip(current, references):
        parameter = parameter.float()
        reference = reference.float()
        loss = 1.0 - F.cosine_similarity(
            parameter.reshape(1, -1), reference.reshape(1, -1), dim=-1
        ).mean()
        if float(l2_weight) > 0.0:
            loss = loss + float(l2_weight) * (
                (parameter - reference).square().sum()
                / reference.square().sum().clamp_min(1e-12)
            )
        losses.append(loss)
    if not losses:
        raise ValueError("At least one prompt group is required")
    return torch.stack(losses).sum()


def centered_tcp_knowledge_loss(text_features, prior):
    """Align only class-differential medical-semantic directions."""

    text_features = text_features.float() - text_features.float().mean(
        dim=0, keepdim=True
    )
    prior = prior.float() - prior.float().mean(dim=0, keepdim=True)
    return 1.0 - F.cosine_similarity(text_features, prior, dim=-1).mean()


def class_balanced_cross_modal_prototype_loss(
    image_features, text_features, labels, temperature=0.1
):
    """Bidirectionally align per-class image centroids and text prototypes."""

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
    normalized_images = F.normalize(image_features.float(), dim=-1)
    normalized_text = F.normalize(text_features.float(), dim=-1)
    centroids = F.normalize(
        torch.stack(
            [normalized_images[labels == class_id].mean(dim=0) for class_id in classes]
        ),
        dim=-1,
    )
    image_to_text = centroids @ normalized_text.t() / temperature
    text_to_image = normalized_text[classes] @ centroids.t() / temperature
    local_targets = torch.arange(classes.numel(), device=classes.device)
    return 0.5 * (
        F.cross_entropy(image_to_text, classes)
        + F.cross_entropy(text_to_image, local_targets)
    )


class PromptParameterBundle(nn.Module):
    """Checkpoint trainable prompt modules without the frozen backbone."""

    def __init__(self, prompt_learner, visual_prompt=None, text_prompt=None, tcp=None):
        super().__init__()
        self.prompt_learner = prompt_learner
        if visual_prompt is not None:
            self.visual_prompt = visual_prompt
        if text_prompt is not None:
            self.text_prompt = text_prompt
        if tcp is not None:
            self.tcp = tcp


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Train CoOp, VPT and optionally the final MT-TCP with one AdamW."""

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

        tcp = cfg.TRAINER.TCP
        if not tcp.ENABLED:
            return
        expected = {
            "NUM_TOKENS": 4,
            "BOTTLENECK_DIM": 128,
            "INSERT_LAYER": 8,
            "DESCRIPTION_COUNT": 50,
            "RESIDUAL_WARMUP_EPOCHS": 10,
        }
        for field, value in expected.items():
            if int(getattr(tcp, field)) != value:
                raise ValueError("Final TCP requires {}={}".format(field, value))
        floating = {
            "GATE_INIT": 0.05,
            "KG_WEIGHT": 4.0,
            "PROMPT_ANCHOR_WEIGHT": 4.0,
            "PROMPT_ANCHOR_L2_WEIGHT": 0.5,
            "CROSS_MODAL_PROTO_WEIGHT": 0.5,
            "CROSS_MODAL_PROTO_TEMPERATURE": 0.1,
        }
        for field, value in floating.items():
            if abs(float(getattr(tcp, field)) - value) > 1e-12:
                raise ValueError("Final TCP requires {}={}".format(field, value))
        if not bool(tcp.EVAL_WARMSTART):
            raise ValueError("Final TCP requires epoch-0 warm-start evaluation")
        if not tcp.INIT_BASELINE_CHECKPOINT:
            raise ValueError("Final TCP requires a TextDeep baseline checkpoint")
        if not trainer_cfg.VPT_ENABLED or trainer_cfg.TEXT_VPT_ENABLED:
            raise ValueError(
                "Final TCP owns its internal TextDeep prompts and requires Visual VPT"
            )

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

        self.tcp_enabled = bool(cfg.TRAINER.TCP.ENABLED)
        print(
            "Building CoOp{}{}{}".format(
                " + VPT-Deep" if trainer_cfg.VPT_ENABLED else "",
                " + Text-Deep-Prompt" if trainer_cfg.TEXT_VPT_ENABLED else "",
                " + MT-TCP-LayerBasis-XProto" if self.tcp_enabled else "",
            )
        )
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        self.vpt_enabled = bool(trainer_cfg.VPT_ENABLED)
        self.text_vpt_enabled = bool(trainer_cfg.TEXT_VPT_ENABLED)
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

        tcp_prompt = None
        if self.tcp_enabled:
            tcp = cfg.TRAINER.TCP
            if int(tcp.NUM_TOKENS) != int(self.model.prompt_learner.n_ctx):
                raise ValueError("TCP and CoOp must both use four tokens")
            projected_bank, descriptions = build_frozen_description_bank(
                biomedclip_model,
                self.model.prompt_learner.tokenizer,
                classnames,
                BIOMEDCOOP_TEMPLATES,
                expected_count=tcp.DESCRIPTION_COUNT,
                batch_size=tcp.DESCRIPTION_BATCH_SIZE,
                cache_path=tcp.DESCRIPTION_CACHE or None,
            )
            layer_bank, layer_descriptions = build_frozen_layer_description_bank(
                biomedclip_model,
                self.model.prompt_learner.tokenizer,
                classnames,
                BIOMEDCOOP_TEMPLATES,
                insert_layer=tcp.INSERT_LAYER,
                expected_count=tcp.DESCRIPTION_COUNT,
                batch_size=tcp.DESCRIPTION_BATCH_SIZE,
                cache_path=tcp.LAYER_DESCRIPTION_CACHE or None,
            )
            if layer_descriptions != descriptions:
                raise RuntimeError("Projected and layer description order mismatch")
            self.model.text_encoder = MultiTextTCPBertTextEncoder(
                biomedclip_model.text,
                layer_description_bank=layer_bank,
                projected_description_bank=projected_bank,
                descriptions=descriptions,
                classnames=classnames,
                num_tokens=tcp.NUM_TOKENS,
                bottleneck_dim=tcp.BOTTLENECK_DIM,
                insert_layer=tcp.INSERT_LAYER,
                gate_init=tcp.GATE_INIT,
            )
            tcp_prompt = self.model.text_encoder.tcp_prompt
            print(
                "Final MT-TCP: {} descriptions/class, aggregation={}, "
                "connection={}, insert block={}, gate={}".format(
                    self.model.text_encoder.description_count,
                    self.model.text_encoder.aggregation,
                    self.model.text_encoder.connection,
                    self.model.text_encoder.insert_layer,
                    self.model.text_encoder.gate_init,
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
            self.model.prompt_learner,
            visual_prompt=visual_prompt,
            text_prompt=text_prompt,
            tcp=tcp_prompt,
        )
        trainable_parameters = [self.model.prompt_learner.ctx]
        for module in (visual_prompt, text_prompt, tcp_prompt):
            if module is not None:
                trainable_parameters.extend(list(module.parameters()))
        self.optim = build_optimizer(trainable_parameters, trainer_cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, trainer_cfg.OPTIM)
        self.register_model(
            "prompt_parameters", self.prompt_parameters, self.optim, self.sched
        )

        self._audit_trainable_parameters()
        self.scaler = GradScaler() if trainer_cfg.PREC == "amp" else None
        if torch.cuda.device_count() > 1:
            print("Multiple GPUs detected, using DataParallel")
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
                for name, _ in self.model.text_encoder.tcp_prompt.named_parameters()
            )
        if set(trainable) != expected:
            raise RuntimeError(
                "Unexpected trainable parameters: expected {}, got {}".format(
                    sorted(expected), sorted(trainable)
                )
            )
        if len(self.optim.param_groups) != 1:
            raise RuntimeError("All prompt branches must share one optimizer group")
        optimizer_ids = {
            id(parameter)
            for group in self.optim.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in trainable.values()}:
            raise RuntimeError("Optimizer parameters do not match trainable prompts")
        print("Trainable prompt parameters: {:,}".format(
            sum(parameter.numel() for parameter in trainable.values())
        ))

    def set_model_mode(self, mode="train", names=None):
        model = self._unwrapped_model()
        model.eval()
        modules = [model.prompt_learner]
        if self.vpt_enabled:
            modules.append(model.image_encoder.visual_prompt)
        if self.text_vpt_enabled:
            modules.append(model.text_encoder.text_prompt)
        if self.tcp_enabled:
            modules.append(model.text_encoder.tcp_prompt)
        if mode == "train":
            for module in modules:
                module.train()
        elif mode in {"test", "eval"}:
            for module in modules:
                module.eval()
        else:
            raise KeyError(mode)

    def before_train(self):
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(self.device)
        super().before_train()
        if not self.tcp_enabled:
            return
        baseline = self.cfg.TRAINER.TCP.INIT_BASELINE_CHECKPOINT
        if self.start_epoch == 0:
            self._initialize_from_baseline_checkpoint(baseline)
        self._load_prompt_anchor(baseline)
        if self.start_epoch == 0:
            self._unwrapped_model().text_encoder.tcp_prompt.set_residual_scale(0.0)
            print("Evaluate exact TextDeep warm start as epoch 0")
            self.evaluate_and_track_best(checkpoint_epoch=-1, record_epoch=0)

    def after_train(self):
        super().after_train()
        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            print("Peak CUDA memory allocated: {:.2f} MiB".format(peak_mib))

    def before_epoch(self):
        if not self.tcp_enabled:
            return
        warmup = int(self.cfg.TRAINER.TCP.RESIDUAL_WARMUP_EPOCHS)
        scale = min(1.0, float(self.epoch + 1) / float(warmup))
        self._unwrapped_model().text_encoder.tcp_prompt.set_residual_scale(scale)
        print("TCP residual scale: {:.6f}".format(scale))

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        self.model_zero_grad()
        if self.cfg.TRAINER.COOPVPT.PREC == "amp":
            with autocast():
                output, losses = self._compute_training_loss(image, label)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optim)
            self._audit_gradients_once()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, losses = self._compute_training_loss(image, label)
            self.model_backward(losses["loss"])
            self._audit_gradients_once()
            self.optim.step()

        summary = {name: value.item() for name, value in losses.items()}
        summary.update(
            {
                "acc": compute_accuracy(output, label)[0].item(),
                "lr": self.optim.param_groups[0]["lr"],
            }
        )
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def _compute_training_loss(self, image, label):
        if not self.tcp_enabled:
            output = self.model(image)
            loss_ce = F.cross_entropy(output, label)
            return output, {"loss": loss_ce}

        output, text_features, image_features = self.model(image, return_features=True)
        model = self._unwrapped_model()
        prior = model.text_encoder.class_prior.to(
            device=text_features.device, dtype=text_features.dtype
        )
        loss_ce = F.cross_entropy(output, label)
        loss_kg = centered_tcp_knowledge_loss(text_features, prior)
        loss_anchor = self._prompt_anchor_loss()
        loss_xproto = class_balanced_cross_modal_prototype_loss(
            image_features,
            text_features,
            label,
            temperature=self.cfg.TRAINER.TCP.CROSS_MODAL_PROTO_TEMPERATURE,
        )
        loss = (
            loss_ce
            + float(self.cfg.TRAINER.TCP.KG_WEIGHT) * loss_kg
            + float(self.cfg.TRAINER.TCP.PROMPT_ANCHOR_WEIGHT) * loss_anchor
            + float(self.cfg.TRAINER.TCP.CROSS_MODAL_PROTO_WEIGHT) * loss_xproto
        )
        return output, {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_kg": loss_kg,
            "loss_anchor": loss_anchor,
            "loss_cross_modal_proto": loss_xproto,
        }

    def _prompt_anchor_loss(self):
        if self._prompt_anchor is None:
            return self._unwrapped_model().prompt_learner.ctx.new_zeros(())
        model = self._unwrapped_model()
        return cosine_l2_prompt_anchor_loss(
            (
                model.prompt_learner.ctx,
                model.image_encoder.visual_prompt.prompt_embeddings,
                model.text_encoder.tcp_prompt.text_prompt.prompt_embeddings,
            ),
            self._prompt_anchor,
            l2_weight=self.cfg.TRAINER.TCP.PROMPT_ANCHOR_L2_WEIGHT,
        )

    def _audit_gradients_once(self):
        if self._gradient_audit_complete or not self.tcp_enabled:
            return
        model = self._unwrapped_model()
        branches = {
            "CoOp": [model.prompt_learner.ctx],
            "VPT": list(model.image_encoder.visual_prompt.parameters()),
            "MT-TCP": list(model.text_encoder.tcp_prompt.parameters()),
        }
        norms = {}
        for name, parameters in branches.items():
            norm = sum(
                float(parameter.grad.detach().float().norm())
                for parameter in parameters
                if parameter.grad is not None
            )
            if norm <= 0.0:
                raise RuntimeError("Gradient audit failed for {}".format(name))
            norms[name] = norm
        print("TCP gradient audit passed: {}".format(norms))
        self._gradient_audit_complete = True

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def resume_model_if_exist(self, directory):
        prompt_dir = osp.join(directory, "prompt_parameters")
        checkpoint_index = osp.join(prompt_dir, "checkpoint")
        if not osp.exists(checkpoint_index):
            print("No complete prompt checkpoint found, train from scratch")
            return 0
        if self.tcp_enabled:
            with open(checkpoint_index, "r") as file:
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
        if any(key.startswith("tcp.") for key in state_dict):
            raise RuntimeError("Baseline must be a non-TCP TextDeep checkpoint")
        mapping = (
            ("prompt_learner.ctx", "prompt_learner.ctx"),
            ("visual_prompt.prompt_embeddings", "visual_prompt.prompt_embeddings"),
            (
                "text_prompt.prompt_embeddings",
                "tcp.text_prompt.prompt_embeddings",
            ),
        )
        current = bundle.state_dict()
        for source, target in mapping:
            if source not in state_dict:
                raise RuntimeError("Baseline checkpoint is missing {}".format(source))
            if target not in current or tuple(current[target].shape) != tuple(
                state_dict[source].shape
            ):
                raise RuntimeError("Baseline initialization shape mismatch for {}".format(source))
            if not torch.isfinite(state_dict[source]).all():
                raise RuntimeError("Baseline contains non-finite {}".format(source))
        return tuple(state_dict[source] for source, _ in mapping)

    @staticmethod
    def _initialize_bundle_from_baseline_state(bundle, state_dict):
        values = CoOpVPT_BiomedCLIP._validated_baseline_prompt_tensors(
            bundle, state_dict
        )
        with torch.no_grad():
            bundle.prompt_learner.ctx.copy_(values[0])
            bundle.visual_prompt.prompt_embeddings.copy_(values[1])
            bundle.tcp.text_prompt.prompt_embeddings.copy_(values[2])

    def _initialize_from_baseline_checkpoint(self, path):
        checkpoint = load_checkpoint(str(path))
        self._initialize_bundle_from_baseline_state(
            self.prompt_parameters, checkpoint["state_dict"]
        )
        print('Initialized CoOp, Visual VPT and TextDeep from "{}"'.format(path))

    def _load_prompt_anchor(self, path):
        checkpoint = load_checkpoint(str(path))
        values = self._validated_baseline_prompt_tensors(
            self.prompt_parameters, checkpoint["state_dict"]
        )
        self._prompt_anchor = tuple(
            value.detach().clone().to(self.device) for value in values
        )

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return
        model_file = (
            "model-best.pth.tar" if epoch is None else "model.pth.tar-{}".format(epoch)
        )
        prompt_dir = osp.join(directory, "prompt_parameters")
        prompt_path = osp.join(prompt_dir, model_file)
        if osp.exists(prompt_path):
            checkpoint = load_checkpoint(prompt_path)
            if self.tcp_enabled:
                self._validate_tcp_state(checkpoint["state_dict"])
            elif any(key.startswith("tcp.") for key in checkpoint["state_dict"]):
                raise RuntimeError("A TCP checkpoint cannot load with TCP disabled")
            if self.text_vpt_enabled and not any(
                key.startswith("text_prompt.") for key in checkpoint["state_dict"]
            ):
                raise RuntimeError("Checkpoint does not contain TextDeep prompts")
            self.prompt_parameters.load_state_dict(checkpoint["state_dict"], strict=True)
            self._restore_checkpoint_residual_scale(checkpoint)
            return

        old_path = osp.join(directory, "prompt_learner", model_file)
        if osp.exists(old_path):
            if self.text_vpt_enabled or self.tcp_enabled:
                raise RuntimeError("Legacy CoOp checkpoints cannot restore text adapters")
            checkpoint = load_checkpoint(old_path)
            self._load_prompt_learner_state(
                self.model.prompt_learner, checkpoint["state_dict"]
            )
            return
        raise FileNotFoundError(
            'Prompt checkpoint not found at "{}" or "{}"'.format(prompt_path, old_path)
        )

    def _validate_tcp_state(self, state_dict):
        if self.tcp_enabled:
            validate_tcp_checkpoint_state(
                state_dict,
                self._unwrapped_model().text_encoder.tcp_prompt,
                prefix="tcp.",
            )

    def _restore_checkpoint_residual_scale(self, checkpoint):
        if not self.tcp_enabled:
            return
        scale = checkpoint_residual_scale(
            checkpoint["epoch"], self.cfg.TRAINER.TCP.RESIDUAL_WARMUP_EPOCHS
        )
        self._unwrapped_model().text_encoder.tcp_prompt.set_residual_scale(scale)

    def load_prompt_checkpoint(self, path):
        checkpoint = load_checkpoint(str(path))
        self._validate_tcp_state(checkpoint["state_dict"])
        self.prompt_parameters.load_state_dict(checkpoint["state_dict"], strict=True)
        self._restore_checkpoint_residual_scale(checkpoint)
        return checkpoint
