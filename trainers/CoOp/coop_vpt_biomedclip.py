"""CoOp + Visual/Text VPT trainer with the shared TKE TCP."""

from __future__ import annotations

import hashlib
import json
import os.path as osp
from pathlib import Path

import torch
from torch import nn
from torch.cuda.amp import GradScaler, autocast
from torch.nn import functional as F
from tqdm import tqdm

from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_lr_scheduler, build_optimizer
from dassl.utils import load_checkpoint
from dassl.utils.torchtools import resume_from_checkpoint, save_checkpoint

from models.biomedclip_loader import load_biomedclip
from models.class_conditioned_visual_prompt import ClassConditionedVisualPrompt
from models.original_style_tcp import (
    OriginalStyleTCPBertTextEncoder,
    build_frozen_description_bank,
    validate_tcp_checkpoint_state,
)
from trainers.CoOp.coop_biomedclip import CustomCLIP
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


DESCRIPTION_COUNT = 50
PROTOCOL = "coop_vpt_tcp_tke_v1"


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2), encoding="utf-8")


def _parameter_fingerprint(named_parameters):
    digest = hashlib.sha256()
    entries = []
    for name, parameter in sorted(named_parameters, key=lambda item: item[0]):
        value = parameter.detach().float().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(value.numpy().tobytes())
        entries.append(
            {"name": name, "shape": list(value.shape), "numel": value.numel()}
        )
    return digest.hexdigest(), entries


class CVPCustomCLIP(CustomCLIP):
    """All-class visual conditioning; labels are used only by the CE loss."""

    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__(cfg, classnames, biomedclip_model)
        self.cvp_insert_layer = int(cfg.TRAINER.CVP.INSERT_LAYER)
        self.cvp_fusion_weight = float(cfg.TRAINER.CVP.FUSION_WEIGHT)

    def forward(self, image, return_text_features=False, return_features=False):
        tokens = self.cvp(self.text_encoder.class_prior)
        image = image.type(self.dtype)
        image_features = torch.stack([
            self.image_encoder.forward_with_class_prompt(
                image, prompt.unsqueeze(0).expand(image.shape[0], -1, -1),
                insert_layer=self.cvp_insert_layer,
                fusion_weight=self.cvp_fusion_weight,
            ) for prompt in tokens
        ], dim=1)
        text_features = self.text_encoder(self.prompt_learner(), self.tokenized_prompts)
        image_norm = F.normalize(image_features, dim=-1)
        text_norm = F.normalize(text_features, dim=-1)
        logits = self.logit_scale.exp() * torch.einsum("bcd,cd->bc", image_norm, text_norm)
        if return_features:
            return logits, text_norm, image_norm
        if return_text_features:
            return logits, text_norm
        return logits


class PromptParameterBundle(nn.Module):
    """Checkpoint the prompt adapters enabled for the current TCP run."""

    def __init__(self, prompt_learner, visual_prompt, tcp, cvp=None):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.visual_prompt = visual_prompt
        self.tcp = tcp
        if cvp is not None:
            self.cvp = cvp


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Joint from-scratch CoOp, VPT and shared-TKE TCP training."""

    def check_cfg(self, cfg):
        trainer_cfg = cfg.TRAINER.COOPVPT
        cvp = cfg.TRAINER.CVP
        if cvp.ENABLED and cfg.TRAINER.SEMANTIC_DISTILL.ENABLED:
            raise ValueError("CVP and Semantic Distill are independent experiments; "
                             "combined mode is not implemented.")
        if cvp.ENABLED:
            if not 0 <= float(cvp.FUSION_WEIGHT) <= 1:
                raise ValueError("CVP FUSION_WEIGHT must be in [0, 1]")
            if not cfg.TRAINER.TCP.ENABLED:
                raise ValueError("CVP requires Original TKE (TCP.ENABLED=True)")
            if int(cvp.NUM_TOKENS) != 4 or int(trainer_cfg.VPT_N_CTX) != int(cvp.NUM_TOKENS):
                raise ValueError("CVP requires four matching visual prompt slots")
            if not 1 <= int(cvp.INSERT_LAYER) < 12 or int(cvp.BOTTLENECK_DIM) < 1:
                raise ValueError("Invalid CVP insertion layer or bottleneck dimension")
        if cfg.TRAINER.SEMANTIC_DISTILL.GRAD_NORM_INTERVAL < 0:
            raise ValueError("Semantic gradient norm interval must be nonnegative")
        if cfg.TRAINER.SEMANTIC_DISTILL.TEMPERATURE <= 0 or cfg.TRAINER.SEMANTIC_DISTILL.WEIGHT < 0:
            raise ValueError("Semantic temperature must be positive and weight nonnegative")
        if trainer_cfg.PREC not in {"fp16", "fp32", "amp"}:
            raise ValueError("COOPVPT.PREC must be fp16, fp32 or amp")
        if cfg.OPTIM.NAME.lower() != "adamw":
            raise ValueError("The retained optimizer is AdamW")
        if int(cfg.TRAINER.COOP.N_CTX) != 4:
            raise ValueError("The retained CoOp setup requires four tokens")
        if int(cfg.TRAINER.TCP.INSERT_LAYER) < 1:
            raise ValueError("TCP INSERT_LAYER must be at least one")
        if not 0 <= float(cfg.TRAINER.TCP.FUSION_WEIGHT) <= 1:
            raise ValueError("TCP FUSION_WEIGHT must be in [0, 1]")

    def build_model(self):
        cfg = self.cfg
        trainer_cfg = cfg.TRAINER.COOPVPT
        classnames = self.dm.dataset.classnames

        print("Loading frozen BiomedCLIP and building from-scratch prompt adapters")
        biomedclip_model, _ = load_biomedclip(
            vpt_enabled=True,
            vpt_mode="deep",
            vpt_num_tokens=trainer_cfg.VPT_N_CTX,
            vpt_dropout=trainer_cfg.VPT_DROPOUT,
        )
        if trainer_cfg.PREC in {"fp32", "amp"}:
            biomedclip_model.float()

        model_class = CVPCustomCLIP if cfg.TRAINER.CVP.ENABLED else CustomCLIP
        self.model = model_class(cfg, classnames, biomedclip_model.eval())
        self._gradient_audit_complete = False
        self.protocol = PROTOCOL
        self._semantic_audit_complete = False
        self._semantic_step = 0

        tcp = cfg.TRAINER.TCP
        self.tcp_enabled = bool(tcp.ENABLED)
        projected_bank, _descriptions = build_frozen_description_bank(
            biomedclip_model,
            self.model.prompt_learner.tokenizer,
            classnames,
            BIOMEDCOOP_TEMPLATES,
            expected_count=DESCRIPTION_COUNT,
            batch_size=int(cfg.DATALOADER.TEST.BATCH_SIZE),
            cache_path=tcp.DESCRIPTION_CACHE or None,
        )
        self.model.text_encoder = OriginalStyleTCPBertTextEncoder(
            biomedclip_model.text,
            projected_bank,
            classnames,
            insert_layer=tcp.INSERT_LAYER,
            enabled=self.tcp_enabled,
            fusion_weight=tcp.FUSION_WEIGHT,
        )
        tcp_prompt = self.model.text_encoder.tcp_prompt
        if cfg.TRAINER.CVP.ENABLED:
            self.model.cvp = ClassConditionedVisualPrompt(
                prior_dim=self.model.text_encoder.class_prior.shape[-1],
                hidden_dim=self.model.image_encoder.visual_prompt.embed_dim,
                num_tokens=cfg.TRAINER.CVP.NUM_TOKENS,
                bottleneck_dim=cfg.TRAINER.CVP.BOTTLENECK_DIM,
            )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.prompt_learner.ctx.requires_grad_(True)
        for parameter in self.model.image_encoder.visual_prompt.parameters():
            parameter.requires_grad_(True)
        for parameter in tcp_prompt.text_prompt.parameters():
            parameter.requires_grad_(True)
        if self.tcp_enabled:
            for name, parameter in tcp_prompt.named_parameters():
                if not name.startswith("text_prompt."):
                    parameter.requires_grad_(True)
        if cfg.TRAINER.CVP.ENABLED:
            self.model.cvp.requires_grad_(True)
        if cfg.MODEL.INIT_WEIGHTS:
            raise ValueError("MODEL.INIT_WEIGHTS is forbidden in from-scratch runs")

        base_named = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        base_fingerprint, base_entries = _parameter_fingerprint(base_named)

        self.model.to(self.device)
        self.model.eval()
        for module in (
            self.model.prompt_learner,
            self.model.image_encoder.visual_prompt,
            tcp_prompt,
        ):
            module.train()
        if cfg.TRAINER.CVP.ENABLED:
            self.model.cvp.train()

        self.prompt_parameters = PromptParameterBundle(
            self.model.prompt_learner,
            visual_prompt=self.model.image_encoder.visual_prompt,
            tcp=tcp_prompt,
            cvp=getattr(self.model, "cvp", None),
        )

        trainable_parameters = [
            parameter for parameter in self.model.parameters() if parameter.requires_grad
        ]
        self.optim = build_optimizer(trainable_parameters, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model(
            "prompt_parameters", self.prompt_parameters, self.optim, self.sched
        )
        self.scaler = GradScaler() if trainer_cfg.PREC == "amp" else None
        self._audit_trainable_parameters()

        manifest = {
            "protocol": self.protocol,
            "semantic_distill": dict(self.cfg.TRAINER.SEMANTIC_DISTILL),
            "tcp_enabled": self.tcp_enabled,
            "seed": int(cfg.SEED),
            "shots": int(cfg.DATASET.NUM_SHOTS),
            "fusion_weight": float(self.cfg.TRAINER.TCP.FUSION_WEIGHT),
            "core_initialization_fingerprint": base_fingerprint,
            "core_parameters": base_entries,
            "parameter_counts": self._parameter_count_manifest(),
            **self._cvp_checkpoint_metadata(),
        }
        _json_write(Path(cfg.OUTPUT_DIR) / "initialization_manifest.json", manifest)
        if torch.cuda.device_count() > 1:
            print("Multiple GPUs detected, using DataParallel")
            self.model = nn.DataParallel(self.model)

    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

    def _cvp_checkpoint_metadata(self):
        cvp = self.cfg.TRAINER.CVP
        return dict(cvp_enabled=bool(cvp.ENABLED), cvp_insert_layer=int(cvp.INSERT_LAYER),
                    cvp_num_tokens=int(cvp.NUM_TOKENS), cvp_bottleneck_dim=int(cvp.BOTTLENECK_DIM),
                    cvp_fusion_weight=float(cvp.FUSION_WEIGHT))

    def _validate_cvp_checkpoint(self, checkpoint):
        expected = self._cvp_checkpoint_metadata()
        saved_enabled = bool(checkpoint.get("cvp_enabled", False))
        has_keys = any(key.startswith("cvp.") for key in checkpoint["state_dict"])
        if saved_enabled != has_keys or saved_enabled != expected["cvp_enabled"]:
            raise RuntimeError("Checkpoint CVP mode does not match current run or state keys")
        if saved_enabled:
            for field, value in expected.items():
                saved = checkpoint.get(field, 1.0) if field == "cvp_fusion_weight" else checkpoint.get(field)
                if saved != value:
                    raise RuntimeError("Checkpoint {} does not match current run".format(field))

    def _parameter_count_manifest(self):
        model = self._unwrapped_model()
        groups = {
            "coop": [model.prompt_learner.ctx],
            "visual_deep_prompt": list(model.image_encoder.visual_prompt.parameters()),
            "text_vpt": list(model.text_encoder.tcp_prompt.text_prompt.parameters()),
            "tcp_mechanism": [
                parameter
                for name, parameter in model.text_encoder.tcp_prompt.named_parameters()
                if not name.startswith("text_prompt.") and parameter.requires_grad
            ],
        }
        if hasattr(model, "cvp"):
            groups["cvp"] = list(model.cvp.parameters())
        counts = {
            name: sum(parameter.numel() for parameter in parameters)
            for name, parameters in groups.items()
        }
        counts["total_trainable"] = sum(counts.values())
        counts["total_frozen"] = sum(
            parameter.numel()
            for parameter in model.parameters()
            if not parameter.requires_grad
        )
        return counts

    def _audit_trainable_parameters(self):
        trainable = {
            name: parameter
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        }
        expected = {
            "prompt_learner.ctx",
            "image_encoder.visual_prompt.prompt_embeddings",
        }
        expected.update(
            "text_encoder.tcp_prompt.text_prompt.{}".format(name)
            for name, _ in self.model.text_encoder.tcp_prompt.text_prompt.named_parameters()
        )
        if self.tcp_enabled:
            expected.update(
                "text_encoder.tcp_prompt.{}".format(name)
                for name, _ in self.model.text_encoder.tcp_prompt.named_parameters()
                if not name.startswith("text_prompt.")
            )
        if hasattr(self.model, "cvp"):
            expected.update("cvp." + name for name, _ in self.model.cvp.named_parameters())
        if set(trainable) != expected:
            raise RuntimeError(
                "Unexpected trainable parameters: expected {}, got {}".format(
                    sorted(expected), sorted(trainable)
                )
            )
        optimizer_ids = {
            id(parameter)
            for group in self.optim.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in trainable.values()}:
            raise RuntimeError("Optimizer parameters do not match trainable adapters")
        print("Parameter audit: {}".format(self._parameter_count_manifest()))

    def set_model_mode(self, mode="train", names=None):
        model = self._unwrapped_model()
        model.eval()
        modules = [
            model.prompt_learner,
            model.image_encoder.visual_prompt,
            model.text_encoder.tcp_prompt,
        ]
        if hasattr(model, "cvp"):
            modules.append(model.cvp)
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

    def after_train(self):
        super().after_train()
        if self.device.type == "cuda":
            peak_mib = torch.cuda.max_memory_allocated(self.device) / (1024 ** 2)
            print("Peak CUDA memory allocated: {:.2f} MiB".format(peak_mib))

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

        if "semantic_grad_norm" in losses:
            self.write_scalar("train/semantic_grad_norm", losses["semantic_grad_norm"].item(),
                              self.epoch * self.num_batches + self.batch_idx)
        summary = {name: value.item() for name, value in losses.items()
                   if name != "semantic_grad_norm"}
        summary.update(
            acc=compute_accuracy(output, label)[0].item(),
            lr=self.optim.param_groups[0]["lr"],
        )
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def _compute_training_loss(self, image, label):
        if not self.cfg.TRAINER.SEMANTIC_DISTILL.ENABLED:
            output = self.model(image)
            loss_ce = F.cross_entropy(output, label)
            return output, {"loss": loss_ce, "loss_ce": loss_ce}
        output, text_norm, image_norm = self.model(image, return_features=True)
        temperature = self.cfg.TRAINER.SEMANTIC_DISTILL.TEMPERATURE
        student_text = text_norm.detach().float()
        text_relation = student_text @ student_text.t()
        teacher_logits = text_relation[label]
        teacher_prob = F.softmax(teacher_logits / temperature, dim=-1).detach()
        student_logits = image_norm.float() @ student_text.t()
        loss_sem = F.kl_div(F.log_softmax(student_logits / temperature, dim=-1),
                            teacher_prob, reduction="batchmean")
        loss_ce = F.cross_entropy(output, label)
        interval = self.cfg.TRAINER.SEMANTIC_DISTILL.GRAD_NORM_INTERVAL
        record_gradient = not self._semantic_audit_complete or (
            interval > 0 and self._semantic_step % interval == 0
        )
        grad_norm = None
        if record_gradient:
            model = self._unwrapped_model()
            visual_parameters = list(model.image_encoder.visual_prompt.parameters())
            semantic_grads = torch.autograd.grad(loss_sem, visual_parameters, retain_graph=True)
            grad_norm = torch.stack([g.detach().float().square().sum()
                                     for g in semantic_grads]).sum().sqrt()
        self._semantic_step += 1
        if not self._semantic_audit_complete:
            loss_sem.backward(retain_graph=True)
            forbidden = [name for name, p in model.named_parameters()
                         if not name.startswith("image_encoder.visual_prompt.") and p.grad is not None]
            if forbidden or not torch.isfinite(grad_norm) or grad_norm <= 0:
                raise RuntimeError(f"Semantic gradient audit failed: {forbidden}, norm={grad_norm}")
            self.model_zero_grad()
            audit = dict(image_features=list(image_norm.shape), text_features=list(text_norm.shape),
                         text_relation=list(text_relation.shape), teacher_logits=list(teacher_logits.shape),
                         student_logits=list(student_logits.shape), logits=list(output.shape),
                         visual_semantic_gradient_norm=float(grad_norm), forbidden_gradients=forbidden)
            _json_write(Path(self.cfg.OUTPUT_DIR) / "semantic_gradient_audit.json", audit)
            print(f"Semantic gradient audit passed: {audit}")
            self._semantic_audit_complete = True
        total = loss_ce + self.cfg.TRAINER.SEMANTIC_DISTILL.WEIGHT * loss_sem
        losses = {"loss": total, "total_loss": total, "loss_ce": loss_ce, "loss_sem": loss_sem}
        if grad_norm is not None:
            losses["semantic_grad_norm"] = grad_norm
        return output, losses

    def _audit_gradients_once(self):
        if self._gradient_audit_complete:
            return
        model = self._unwrapped_model()
        branches = {
            "CoOp": [model.prompt_learner.ctx],
            "VisualDeep": list(model.image_encoder.visual_prompt.parameters()),
            "TextVPT": list(model.text_encoder.tcp_prompt.text_prompt.parameters()),
        }
        if self.tcp_enabled:
            branches["TCP"] = [
                parameter
                for name, parameter in model.text_encoder.tcp_prompt.named_parameters()
                if not name.startswith("text_prompt.")
            ]
        if hasattr(model, "cvp"):
            branches["CVP"] = list(model.cvp.parameters())
        norms = {}
        for name, parameters in branches.items():
            norm = sum(
                float(parameter.grad.detach().float().norm())
                for parameter in parameters
                if parameter.grad is not None
            )
            if norm <= 0 or not torch.isfinite(torch.tensor(norm)):
                raise RuntimeError("Gradient audit failed for {}".format(name))
            norms[name] = norm
        frozen_with_grad = [
            name
            for name, parameter in model.named_parameters()
            if not parameter.requires_grad and parameter.grad is not None
        ]
        if frozen_with_grad:
            raise RuntimeError(
                "Frozen backbone received gradients: {}".format(frozen_with_grad)
            )
        print("Gradient audit passed: {}".format(norms))
        self._gradient_audit_complete = True

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    @torch.no_grad()
    def test(self, split=None):
        self.set_model_mode("eval")
        self.evaluator.reset()
        split = split or self.cfg.TEST.SPLIT
        data_loader = (
            self.val_loader
            if split == "val" and self.val_loader is not None
            else self.test_loader
        )
        print("Do evaluation on {} set".format(split))
        for batch in tqdm(data_loader):
            inputs, labels = self.parse_batch_test(batch)
            output = self.model(inputs)
            self.evaluator.process(output, labels)

        results = self.evaluator.evaluate()
        self.last_eval_results = results
        for key, value in results.items():
            self.write_scalar("{}/{}".format(split, key), value, self.epoch)
        best_metric = self.cfg.TEST.BEST_METRIC
        if best_metric not in results:
            raise KeyError("Validation metric {!r} is unavailable".format(best_metric))
        return results[best_metric]

    def save_model(self, epoch, directory, is_best=False, model_name=""):
        state = {
            "state_dict": self.prompt_parameters.state_dict(),
            "epoch": int(epoch + 1),
            "optimizer": self.optim.state_dict(),
            "scheduler": self.sched.state_dict() if self.sched is not None else None,
            "scaler": self.scaler.state_dict() if self.scaler is not None else None,
            "semantic_distill": dict(self.cfg.TRAINER.SEMANTIC_DISTILL),
            "tcp_enabled": self.tcp_enabled,
            "protocol": self.protocol,
            "fusion_weight": float(self.cfg.TRAINER.TCP.FUSION_WEIGHT),
            **self._cvp_checkpoint_metadata(),
        }
        save_checkpoint(
            state,
            osp.join(directory, "prompt_parameters"),
            is_best=is_best,
            model_name=model_name,
        )

    def resume_model_if_exist(self, directory):
        prompt_dir = osp.join(directory, "prompt_parameters")
        checkpoint_index = osp.join(prompt_dir, "checkpoint")
        if not osp.exists(checkpoint_index):
            print("No complete prompt checkpoint found, train from scratch")
            return 0
        with open(checkpoint_index, "r", encoding="utf-8") as stream:
            checkpoint_name = stream.readline().strip()
        checkpoint = load_checkpoint(osp.join(prompt_dir, checkpoint_name))
        self._validate_checkpoint_metadata(checkpoint)
        saved_semantic = checkpoint.get("semantic_distill", {"ENABLED": False})
        current_semantic = dict(self.cfg.TRAINER.SEMANTIC_DISTILL)
        if bool(saved_semantic["ENABLED"]) != bool(current_semantic["ENABLED"]) or (current_semantic["ENABLED"] and any(
                saved_semantic.get(key) != current_semantic[key]
                for key in ("WEIGHT", "TEMPERATURE"))):
            raise RuntimeError("Resume semantic-distillation configuration does not match checkpoint")
        start_epoch = resume_from_checkpoint(
            prompt_dir, self.prompt_parameters, self.optim, self.sched
        )
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        return start_epoch

    def _validate_checkpoint_metadata(self, checkpoint):
        self._validate_cvp_checkpoint(checkpoint)
        if checkpoint.get("protocol") != self.protocol:
            raise RuntimeError("Checkpoint training protocol does not match current run")
        if bool(checkpoint.get("tcp_enabled", True)) != self.tcp_enabled:
            raise RuntimeError("Checkpoint TCP setting does not match current run")
        if float(self.cfg.TRAINER.TCP.FUSION_WEIGHT) != 1.0 and checkpoint.get("fusion_weight", float(self.cfg.TRAINER.TCP.FUSION_WEIGHT)) != float(self.cfg.TRAINER.TCP.FUSION_WEIGHT):
            raise RuntimeError("Checkpoint fusion weight does not match current run")
        validate_tcp_checkpoint_state(
            checkpoint["state_dict"],
            self._unwrapped_model().text_encoder.tcp_prompt,
            prefix="tcp.",
        )

    def load_prompt_checkpoint(self, path):
        checkpoint = load_checkpoint(str(path))
        self._validate_checkpoint_metadata(checkpoint)
        self.prompt_parameters.load_state_dict(checkpoint["state_dict"], strict=True)
        return checkpoint

    def load_model(self, directory, epoch=None):
        if not directory:
            raise ValueError("A checkpoint directory is required")
        model_file = (
            "model-best.pth.tar"
            if epoch is None
            else "model.pth.tar-{}".format(epoch)
        )
        path = osp.join(directory, "prompt_parameters", model_file)
        if not osp.exists(path):
            raise FileNotFoundError("Prompt checkpoint not found: {}".format(path))
        return self.load_prompt_checkpoint(path)
