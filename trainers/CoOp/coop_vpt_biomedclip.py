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
from trainers.CoOp.coop_biomedclip import CustomCLIP


class PromptParameterBundle(nn.Module):
    """Checkpoint only the trainable prompt modules, not frozen BiomedCLIP."""

    def __init__(self, prompt_learner, visual_prompt=None):
        super().__init__()
        self.prompt_learner = prompt_learner
        if visual_prompt is not None:
            self.visual_prompt = visual_prompt


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Train CoOp and VPT prompts with one AdamW and one shared learning rate."""

    def check_cfg(self, cfg):
        trainer_cfg = cfg.TRAINER.COOPVPT
        assert trainer_cfg.PREC in {"fp16", "fp32", "amp"}
        assert trainer_cfg.VPT_MODE == "deep"
        assert trainer_cfg.VPT_INIT == "uniform"
        assert trainer_cfg.OPTIM.NAME.lower() == "adamw"

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
            "Building CE-only CoOp{}".format(
                " + VPT-Deep" if trainer_cfg.VPT_ENABLED else ""
            )
        )
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        self.vpt_enabled = bool(trainer_cfg.VPT_ENABLED)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.prompt_learner.ctx.requires_grad_(True)

        visual_prompt = None
        if self.vpt_enabled:
            visual_prompt = self.model.image_encoder.visual_prompt
            for parameter in visual_prompt.parameters():
                parameter.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.eval()
        self.model.prompt_learner.train()
        if visual_prompt is not None:
            visual_prompt.train()

        self.prompt_parameters = PromptParameterBundle(
            self.model.prompt_learner, visual_prompt
        )
        trainable_parameters = [self.model.prompt_learner.ctx]
        if visual_prompt is not None:
            trainable_parameters.extend(list(visual_prompt.parameters()))

        # One optimizer and one parameter group intentionally cover both prompt
        # branches. The shared LR is supplied by TRAINER.COOPVPT.OPTIM.LR.
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
        elif mode in {"test", "eval"}:
            model.prompt_learner.eval()
            if self.vpt_enabled:
                model.image_encoder.visual_prompt.eval()
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
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward(loss)
            self.optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "lr": self.optim.param_groups[0]["lr"],
        }

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr()
        return loss_summary

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def resume_model_if_exist(self, directory):
        prompt_dir = osp.join(directory, "prompt_parameters")
        if not osp.exists(osp.join(prompt_dir, "checkpoint")):
            print("No complete prompt checkpoint found, train from scratch")
            return 0
        return resume_from_checkpoint(
            prompt_dir, self.prompt_parameters, self.optim, self.sched
        )

    @staticmethod
    def _load_prompt_learner_state(module, state_dict):
        state_dict = dict(state_dict)
        state_dict.pop("token_prefix", None)
        state_dict.pop("token_suffix", None)
        module.load_state_dict(state_dict, strict=False)

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-{}".format(epoch)
        prompt_dir = osp.join(directory, "prompt_parameters")
        prompt_path = osp.join(prompt_dir, model_file)

        if osp.exists(prompt_path):
            checkpoint_data = load_checkpoint(prompt_path)
            print(
                'Loading prompt_parameters from "{}" (epoch={})'.format(
                    prompt_path, checkpoint_data["epoch"]
                )
            )
            self.prompt_parameters.load_state_dict(
                checkpoint_data["state_dict"], strict=True
            )
            return

        # Backward-compatible explicit extraction of a text prompt from old
        # CoOp checkpoints. A VPT prompt is intentionally not fabricated from
        # an old checkpoint and remains at its configured initialization.
        old_path = osp.join(directory, "prompt_learner", model_file)
        if osp.exists(old_path):
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
