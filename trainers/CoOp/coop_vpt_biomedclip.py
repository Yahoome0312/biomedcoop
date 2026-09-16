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
from models.competitive_visual_prompt import VisualPrototypePrompt
from models.original_style_tcp import (
    OriginalStyleTCPBertTextEncoder,
    build_frozen_description_bank,
    validate_tcp_checkpoint_state,
)
from trainers.CoOp.coop_biomedclip import CustomCLIP
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


DESCRIPTION_COUNT = 50
BASE_PROTOCOL = "coop_vpt_no_confusion_v1"
CVP_PROTOCOL = "coop_vpt_tcp_tke_cvp_v3"


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


class PromptParameterBundle(nn.Module):
    """Checkpoint the prompt adapters enabled for the current TCP run."""

    def __init__(
        self, prompt_learner, visual_prompt, tcp, competitive_visual_prompt=None
    ):
        super().__init__()
        self.prompt_learner = prompt_learner
        self.visual_prompt = visual_prompt
        self.tcp = tcp
        if competitive_visual_prompt is not None:
            self.competitive_visual_prompt = competitive_visual_prompt


class CVPCustomCLIP(CustomCLIP):
    """Keep the Base forward untouched and add the optional CVP path."""

    def __init__(self, cfg, classnames, biomedclip_model):
        super().__init__(cfg, classnames, biomedclip_model)
        self.competitive_visual_prompt = None
        self.competitive_visual_insert_layer = None

    def enable_competitive_visual_prompt(self, visual_tke, insert_layer):
        self.competitive_visual_prompt = visual_tke
        self.competitive_visual_insert_layer = int(insert_layer)

    def forward(
        self,
        image,
        return_text_features=False,
        return_features=False,
    ):
        if self.competitive_visual_prompt is None:
            return super().forward(
                image,
                return_text_features=return_text_features,
                return_features=return_features,
            )

        prompts = self.prompt_learner()
        text_features = self.text_encoder(prompts, self.tokenized_prompts)
        normalized_text = text_features / text_features.norm(dim=-1, keepdim=True)

        shared_state = self.image_encoder.forward_before_layer(
            image.type(self.dtype), self.competitive_visual_insert_layer
        )
        class_visual_prompts = self.competitive_visual_prompt(
            self.text_encoder.class_prior
        )
        visual_prompts = class_visual_prompts.unsqueeze(0).expand(
            shared_state.shape[0], -1, -1, -1
        )
        image_features = self.image_encoder.forward_from_layer(
            shared_state,
            self.competitive_visual_insert_layer,
            prototype_prompts=visual_prompts,
        )
        normalized_images = image_features / image_features.norm(
            dim=-1, keepdim=True
        )
        logits = self.logit_scale.exp() * torch.einsum(
            "bcd,cd->bc", normalized_images, normalized_text
        )

        if return_features:
            return logits, normalized_text, normalized_images
        if return_text_features:
            return logits, normalized_text
        return logits


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Joint from-scratch CoOp, VPT and shared-TKE TCP training."""

    def check_cfg(self, cfg):
        trainer_cfg = cfg.TRAINER.COOPVPT
        if trainer_cfg.PREC not in {"fp16", "fp32", "amp"}:
            raise ValueError("COOPVPT.PREC must be fp16, fp32 or amp")
        if cfg.OPTIM.NAME.lower() != "adamw":
            raise ValueError("The retained optimizer is AdamW")
        if int(cfg.TRAINER.COOP.N_CTX) != 4:
            raise ValueError("The retained CoOp setup requires four tokens")
        if int(cfg.TRAINER.TCP.INSERT_LAYER) < 1:
            raise ValueError("TCP INSERT_LAYER must be at least one")
        cvp = cfg.TRAINER.COMPETITIVE_VISUAL_PROMPT
        if bool(cvp.ENABLED):
            if int(cvp.INSERT_LAYER) != 8:
                raise ValueError("Competitive Visual Prompt must be inserted at block 8")
            if int(cvp.NUM_TOKENS) != 4:
                raise ValueError("Competitive Visual Prompt requires four tokens")
            if int(cvp.BOTTLENECK_DIM) != 128:
                raise ValueError("Competitive Visual TKE bottleneck must be 128")

    def build_model(self):
        cfg = self.cfg
        trainer_cfg = cfg.TRAINER.COOPVPT
        cvp = cfg.TRAINER.COMPETITIVE_VISUAL_PROMPT
        classnames = self.dm.dataset.classnames

        print("Loading frozen BiomedCLIP and building from-scratch prompt adapters")
        biomedclip_model, _ = load_biomedclip(
            vpt_enabled=True,
            vpt_mode="deep",
            vpt_num_tokens=trainer_cfg.VPT_N_CTX,
            vpt_dropout=trainer_cfg.VPT_DROPOUT,
            vpt_prompt_depth=int(cvp.INSERT_LAYER) if bool(cvp.ENABLED) else None,
        )
        if trainer_cfg.PREC in {"fp32", "amp"}:
            biomedclip_model.float()

        self.model = CVPCustomCLIP(cfg, classnames, biomedclip_model.eval())
        self._gradient_audit_complete = False

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
        )
        tcp_prompt = self.model.text_encoder.tcp_prompt

        self.cvp_enabled = bool(cvp.ENABLED)
        visual_tke = None
        if self.cvp_enabled:
            visual_tke = VisualPrototypePrompt(
                prior_dim=self.model.text_encoder.prior_dim,
                bottleneck_dim=cvp.BOTTLENECK_DIM,
                num_tokens=cvp.NUM_TOKENS,
                hidden_dim=self.model.image_encoder.visual_prompt.embed_dim,
            )
            self.model.enable_competitive_visual_prompt(
                visual_tke, insert_layer=cvp.INSERT_LAYER
            )
        self.protocol = CVP_PROTOCOL if self.cvp_enabled else BASE_PROTOCOL

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
        if self.cvp_enabled:
            for parameter in visual_tke.parameters():
                parameter.requires_grad_(True)
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
        trainable_modules = [
            self.model.prompt_learner,
            self.model.image_encoder.visual_prompt,
            tcp_prompt,
        ]
        if self.cvp_enabled:
            trainable_modules.append(visual_tke)
        for module in trainable_modules:
            module.train()

        self.prompt_parameters = PromptParameterBundle(
            self.model.prompt_learner,
            visual_prompt=self.model.image_encoder.visual_prompt,
            tcp=tcp_prompt,
            competitive_visual_prompt=visual_tke,
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
            "tcp_enabled": self.tcp_enabled,
            "cvp_enabled": self.cvp_enabled,
            "cvp_metadata": self._cvp_metadata(),
            "seed": int(cfg.SEED),
            "shots": int(cfg.DATASET.NUM_SHOTS),
            "core_initialization_fingerprint": base_fingerprint,
            "core_parameters": base_entries,
            "parameter_counts": self._parameter_count_manifest(),
        }
        _json_write(Path(cfg.OUTPUT_DIR) / "initialization_manifest.json", manifest)
        if torch.cuda.device_count() > 1:
            print("Multiple GPUs detected, using DataParallel")
            self.model = nn.DataParallel(self.model)

    def _unwrapped_model(self):
        return self.model.module if isinstance(self.model, nn.DataParallel) else self.model

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
            "competitive_visual_tke": (
                list(model.competitive_visual_prompt.parameters())
                if self.cvp_enabled
                else []
            ),
        }
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
        if self.cvp_enabled:
            expected.update(
                "competitive_visual_prompt.{}".format(name)
                for name, _ in self.model.competitive_visual_prompt.named_parameters()
            )
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
        if self.cvp_enabled:
            modules.append(model.competitive_visual_prompt)
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

        summary = {name: value.item() for name, value in losses.items()}
        summary.update(
            acc=compute_accuracy(output, label)[0].item(),
            lr=self.optim.param_groups[0]["lr"],
        )
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def _compute_training_loss(self, image, label):
        output = self.model(image)
        loss_ce = F.cross_entropy(output, label)
        return output, {"loss": loss_ce, "loss_ce": loss_ce}

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
        if self.cvp_enabled:
            branches["CompetitiveVisualTKE"] = list(
                model.competitive_visual_prompt.parameters()
            )
        norms = {}
        for name, parameters in branches.items():
            norm = sum(
                float(parameter.grad.detach().float().norm())
                for parameter in parameters
                if parameter.grad is not None
            )
            if norm <= 0:
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
            "tcp_enabled": self.tcp_enabled,
            "cvp_enabled": self.cvp_enabled,
            "cvp_metadata": self._cvp_metadata(),
            "protocol": self.protocol,
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
        start_epoch = resume_from_checkpoint(
            prompt_dir, self.prompt_parameters, self.optim, self.sched
        )
        if self.scaler is not None and checkpoint.get("scaler") is not None:
            self.scaler.load_state_dict(checkpoint["scaler"])
        return start_epoch

    def _validate_checkpoint_metadata(self, checkpoint):
        if checkpoint.get("protocol") != self.protocol:
            raise RuntimeError("Checkpoint training protocol does not match current run")
        if bool(checkpoint.get("tcp_enabled", True)) != self.tcp_enabled:
            raise RuntimeError("Checkpoint TCP setting does not match current run")
        if bool(checkpoint.get("cvp_enabled", False)) != self.cvp_enabled:
            raise RuntimeError("Checkpoint CVP setting does not match current run")
        if checkpoint.get("cvp_metadata") != self._cvp_metadata():
            raise RuntimeError("Checkpoint CVP architecture does not match current run")
        validate_tcp_checkpoint_state(
            checkpoint["state_dict"],
            self._unwrapped_model().text_encoder.tcp_prompt,
            prefix="tcp.",
        )

    def _cvp_metadata(self):
        if not self.cvp_enabled:
            return None
        module = self._unwrapped_model().competitive_visual_prompt
        return {
            "insert_layer": int(
                self.cfg.TRAINER.COMPETITIVE_VISUAL_PROMPT.INSERT_LAYER
            ),
            "prior_dim": module.prior_dim,
            "bottleneck_dim": module.bottleneck_dim,
            "num_tokens": module.num_tokens,
            "hidden_dim": module.hidden_dim,
            "visual_prompt_depth": (
                self._unwrapped_model().image_encoder.visual_prompt.prompt_depth
            ),
            "input": "mean50_class_prototype",
            "connection": "single_layer_replacement_and_natural_propagation",
        }

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
        self.load_prompt_checkpoint(path)
