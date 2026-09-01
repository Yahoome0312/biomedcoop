"""From-scratch CoOp + Visual/Text Deep Prompt + MT-TCP trainer.

The optional confusion-aware path uses a fixed support-only soft probability
prior and current-image global/local evidence. There is one classifier and a
single joint forward/backward from epoch one.
"""

from __future__ import annotations

import gzip
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
from models.confusion_aware import (
    CONFUSION_VARIANTS,
    ConfusionAwareAdapter,
    bank_file,
    confusion_margin_loss,
    load_soft_confusion_bank,
)
from models.multitext_tcp import (
    MultiTextTCPBertTextEncoder,
    build_frozen_description_bank,
    build_frozen_layer_description_bank,
    validate_tcp_checkpoint_state,
)
from trainers.CoOp.coop_biomedclip import CustomCLIP
from trainers.prompt_templates import BIOMEDCOOP_TEMPLATES


DESCRIPTION_COUNT = 50


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
    """Checkpoint every trainable adapter plus the fixed Bank identity."""

    def __init__(
        self,
        prompt_learner,
        visual_prompt=None,
        tcp=None,
        confusion=None,
    ):
        super().__init__()
        self.prompt_learner = prompt_learner
        if visual_prompt is not None:
            self.visual_prompt = visual_prompt
        if tcp is not None:
            self.tcp = tcp
        if confusion is not None:
            self.confusion = confusion


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Joint from-scratch prompt training with optional confusion awareness."""

    def check_cfg(self, cfg):
        trainer_cfg = cfg.TRAINER.COOPVPT
        if trainer_cfg.PREC not in {"fp16", "fp32", "amp"}:
            raise ValueError("COOPVPT.PREC must be fp16, fp32 or amp")
        if cfg.OPTIM.NAME.lower() != "adamw":
            raise ValueError("The retained optimizer is AdamW")
        if int(cfg.TRAINER.COOP.N_CTX) != 4:
            raise ValueError("MT-TCP and CoOp must use four tokens")

        tcp = cfg.TRAINER.TCP
        expected = {
            "BOTTLENECK_DIM": 128,
            "INSERT_LAYER": 8,
        }
        for field, value in expected.items():
            if int(getattr(tcp, field)) != value:
                raise ValueError("MT-TCP requires {}={}".format(field, value))
        if abs(float(tcp.GATE_INIT) - 0.05) > 1e-12:
            raise ValueError("MT-TCP requires GATE_INIT=0.05")
        confusion = cfg.TRAINER.CONFUSION_AWARE
        if confusion.VARIANT not in CONFUSION_VARIANTS:
            raise ValueError(
                "Unknown confusion variant {!r}; choose from {}".format(
                    confusion.VARIANT, CONFUSION_VARIANTS
                )
            )
        if confusion.VARIANT != "b0" and not confusion.BANK_ROOT:
            raise ValueError("Non-b0 variants require CONFUSION_AWARE.BANK_ROOT")
        if float(confusion.PRIOR_ALPHA) < 0 or float(confusion.GAMMA) < 0:
            raise ValueError("PRIOR_ALPHA and GAMMA must be non-negative")
        if float(confusion.LAMBDA_CONF) < 0:
            raise ValueError("LAMBDA_CONF must be non-negative")

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

        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        self.variant = str(cfg.TRAINER.CONFUSION_AWARE.VARIANT)
        self._gradient_audit_complete = False
        self._train_pair_counts = torch.zeros(
            len(classnames), len(classnames), dtype=torch.long
        )
        self._train_alpha_sum = torch.zeros(2, dtype=torch.float64)
        self._train_alpha_count = 0

        tcp = cfg.TRAINER.TCP
        num_tokens = int(self.model.prompt_learner.n_ctx)
        description_batch_size = int(cfg.DATALOADER.TEST.BATCH_SIZE)
        projected_bank, descriptions = build_frozen_description_bank(
            biomedclip_model,
            self.model.prompt_learner.tokenizer,
            classnames,
            BIOMEDCOOP_TEMPLATES,
            expected_count=DESCRIPTION_COUNT,
            batch_size=description_batch_size,
            cache_path=tcp.DESCRIPTION_CACHE or None,
        )
        layer_bank, layer_descriptions = build_frozen_layer_description_bank(
            biomedclip_model,
            self.model.prompt_learner.tokenizer,
            classnames,
            BIOMEDCOOP_TEMPLATES,
            insert_layer=tcp.INSERT_LAYER,
            expected_count=DESCRIPTION_COUNT,
            batch_size=description_batch_size,
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
            num_tokens=num_tokens,
            bottleneck_dim=tcp.BOTTLENECK_DIM,
            insert_layer=tcp.INSERT_LAYER,
            gate_init=tcp.GATE_INIT,
        )

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.prompt_learner.ctx.requires_grad_(True)
        for parameter in self.model.image_encoder.visual_prompt.parameters():
            parameter.requires_grad_(True)
        for parameter in self.model.text_encoder.tcp_prompt.parameters():
            parameter.requires_grad_(True)
        if cfg.MODEL.INIT_WEIGHTS:
            raise ValueError("MODEL.INIT_WEIGHTS is forbidden in from-scratch runs")

        base_named = [
            (name, parameter)
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        base_fingerprint, base_entries = _parameter_fingerprint(base_named)

        confusion_adapter = None
        self.bank_metadata = None
        if self.variant != "b0":
            confusion_cfg = cfg.TRAINER.CONFUSION_AWARE
            path = bank_file(
                confusion_cfg.BANK_ROOT,
                cfg.DATASET.NAME,
                cfg.DATASET.NUM_SHOTS,
                cfg.SEED,
            )
            soft_prior, metadata = load_soft_confusion_bank(
                path,
                dataset_name=cfg.DATASET.NAME,
                shots=cfg.DATASET.NUM_SHOTS,
                seed=cfg.SEED,
                classnames=classnames,
                support_items=self.dm.dataset.train_x,
            )
            self.bank_metadata = metadata
            confusion_adapter = ConfusionAwareAdapter(
                self.variant,
                soft_prior,
                metadata["bank_fingerprint"],
                prior_alpha=confusion_cfg.PRIOR_ALPHA,
                gamma=confusion_cfg.GAMMA,
            )
            self.model.confusion_adapter = confusion_adapter

        self.model.to(self.device)
        self.model.eval()
        train_modules = [
            self.model.prompt_learner,
            self.model.image_encoder.visual_prompt,
            self.model.text_encoder.tcp_prompt,
        ]
        if confusion_adapter is not None:
            train_modules.append(confusion_adapter)
        for module in train_modules:
            module.train()

        self.prompt_parameters = PromptParameterBundle(
            self.model.prompt_learner,
            visual_prompt=self.model.image_encoder.visual_prompt,
            tcp=self.model.text_encoder.tcp_prompt,
            confusion=confusion_adapter,
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

        _json_write(
            Path(cfg.OUTPUT_DIR) / "initialization_manifest.json",
            {
                "protocol": "from_scratch_joint_ce",
                "variant": self.variant,
                "seed": int(cfg.SEED),
                "shots": int(cfg.DATASET.NUM_SHOTS),
                "b0_initialization_fingerprint": base_fingerprint,
                "b0_parameters": base_entries,
                "bank_fingerprint": (
                    self.bank_metadata["bank_fingerprint"]
                    if self.bank_metadata is not None
                    else None
                ),
                "parameter_counts": self._parameter_count_manifest(),
            },
        )
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
            "mt_tcp_including_text_deep": list(model.text_encoder.tcp_prompt.parameters()),
            "confusion_aware": (
                list(model.confusion_adapter.parameters())
                if model.confusion_adapter is not None
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
            "text_encoder.tcp_prompt.{}".format(name)
            for name, _ in self.model.text_encoder.tcp_prompt.named_parameters()
        )
        if self.model.confusion_adapter is not None:
            expected.update(
                "confusion_adapter.{}".format(name)
                for name, _ in self.model.confusion_adapter.named_parameters()
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
        if model.confusion_adapter is not None:
            modules.append(model.confusion_adapter)
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

    def before_epoch(self):
        self._train_pair_counts.zero_()
        self._train_alpha_sum.zero_()
        self._train_alpha_count = 0

    def after_epoch(self):
        super().after_epoch()
        if self.variant == "b0":
            return
        summary = {
            "epoch": int(self.epoch + 1),
            "variant": self.variant,
            "pair_counts": self._train_pair_counts.tolist(),
        }
        if self._train_alpha_count:
            means = self._train_alpha_sum / self._train_alpha_count
            summary.update(alpha_global=float(means[0]), alpha_local=float(means[1]))
        _json_write(
            Path(self.output_dir)
            / "confusion_analysis"
            / "train_epoch_{:03d}.json".format(self.epoch + 1),
            summary,
        )

    def forward_backward(self, batch):
        image, label = self.parse_batch_train(batch)
        self.model_zero_grad()
        if self.cfg.TRAINER.COOPVPT.PREC == "amp":
            with autocast():
                output, losses, details = self._compute_training_loss(image, label)
            self.scaler.scale(losses["loss"]).backward()
            self.scaler.unscale_(self.optim)
            self._audit_gradients_once()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output, losses, details = self._compute_training_loss(image, label)
            self.model_backward(losses["loss"])
            self._audit_gradients_once()
            self.optim.step()

        self._accumulate_training_details(details)
        summary = {name: value.item() for name, value in losses.items()}
        summary.update(
            acc=compute_accuracy(output, label)[0].item(),
            lr=self.optim.param_groups[0]["lr"],
        )
        if details and "alpha_global" in details:
            summary["alpha_global"] = details["alpha_global"].mean().item()
            summary["alpha_local"] = details["alpha_local"].mean().item()
        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return summary

    def _compute_training_loss(self, image, label):
        output, details, _base_logits = self.model(
            image, return_confusion_details=True
        )
        loss_ce = F.cross_entropy(output, label)
        if self.variant == "b0":
            return output, {"loss": loss_ce, "loss_ce": loss_ce}, None
        loss_confuse, competitor = confusion_margin_loss(
            output, label, details["pair_first"], details["pair_second"]
        )
        details["competitor"] = competitor
        loss = loss_ce + float(
            self.cfg.TRAINER.CONFUSION_AWARE.LAMBDA_CONF
        ) * loss_confuse
        return output, {
            "loss": loss,
            "loss_ce": loss_ce,
            "loss_confuse": loss_confuse,
        }, details

    def _accumulate_training_details(self, details):
        if not details:
            return
        first = details["pair_first"].detach().cpu()
        second = details["pair_second"].detach().cpu()
        for left, right in zip(first.tolist(), second.tolist()):
            self._train_pair_counts[left, right] += 1
        if "alpha_global" in details:
            self._train_alpha_sum[0] += details["alpha_global"].detach().double().sum().cpu()
            self._train_alpha_sum[1] += details["alpha_local"].detach().double().sum().cpu()
            self._train_alpha_count += int(first.numel())

    def _audit_gradients_once(self):
        if self._gradient_audit_complete:
            return
        model = self._unwrapped_model()
        branches = {
            "CoOp": [model.prompt_learner.ctx],
            "VisualDeep": list(model.image_encoder.visual_prompt.parameters()),
            "MT-TCP": list(model.text_encoder.tcp_prompt.parameters()),
        }
        if model.confusion_adapter is not None and list(model.confusion_adapter.parameters()):
            branches["ConfusionAware"] = list(model.confusion_adapter.parameters())
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
            raise RuntimeError("Frozen backbone received gradients: {}".format(frozen_with_grad))
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
        records = []
        for batch in tqdm(data_loader):
            inputs, labels = self.parse_batch_test(batch)
            output, details, _base_logits = self.model(
                inputs, return_confusion_details=True
            )
            self.evaluator.process(output, labels)
            if details:
                for index, impath in enumerate(batch["impath"]):
                    row = {
                        "image_path": str(impath),
                        "true_label": int(labels[index].item()),
                        "pair_first": int(details["pair_first"][index].item()),
                        "pair_second": int(details["pair_second"][index].item()),
                        "selected_prior": float(details["selected_prior"][index].item()),
                        "selected_score": float(details["selected_score"][index].item()),
                    }
                    if "alpha_global" in details:
                        row["alpha_global"] = float(details["alpha_global"][index].item())
                        row["alpha_local"] = float(details["alpha_local"][index].item())
                    records.append(row)

        results = self.evaluator.evaluate()
        self.last_eval_results = results
        for key, value in results.items():
            self.write_scalar("{}/{}".format(split, key), value, self.epoch)
        if records:
            tag = "{}_epoch_{:03d}".format(split, self.epoch + 1)
            path = Path(self.output_dir) / "confusion_analysis" / "{}.json.gz".format(tag)
            path.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(path, "wt", encoding="utf-8") as stream:
                json.dump(records, stream)
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
            "variant": self.variant,
            "bank_fingerprint": (
                self.bank_metadata["bank_fingerprint"]
                if self.bank_metadata is not None
                else None
            ),
            "protocol": "from_scratch_joint_ce",
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
        if checkpoint.get("protocol") != "from_scratch_joint_ce":
            raise RuntimeError("Refusing a legacy warm-start checkpoint")
        if checkpoint.get("variant") != self.variant:
            raise RuntimeError("Checkpoint variant does not match current run")
        expected_bank = (
            self.bank_metadata["bank_fingerprint"]
            if self.bank_metadata is not None
            else None
        )
        if checkpoint.get("bank_fingerprint") != expected_bank:
            raise RuntimeError("Checkpoint Bank fingerprint does not match current run")
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
        self.load_prompt_checkpoint(path)
