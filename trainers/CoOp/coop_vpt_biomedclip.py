"""CE-only CoOp with optional shallow/deep Visual Prompt Tuning."""

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


@TRAINER_REGISTRY.register()
class CoOpVPT_BiomedCLIP(TrainerX):
    """Train only the CoOp context and, when enabled, visual prompts."""

    def check_cfg(self, cfg):
        trainer_cfg = cfg.TRAINER.COOPVPT
        assert trainer_cfg.PREC in {"fp16", "fp32", "amp"}
        assert trainer_cfg.VPT_MODE in {"shallow", "deep"}
        assert trainer_cfg.VPT_INIT == "uniform"
        assert trainer_cfg.COOP_OPTIM.NAME.lower() == "adamw"
        assert trainer_cfg.VPT_OPTIM.NAME.lower() == "adamw"

    def build_model(self):
        cfg = self.cfg
        trainer_cfg = cfg.TRAINER.COOPVPT
        classnames = self.dm.dataset.classnames

        print("Loading and validating BiomedCLIP")
        biomedclip_model, _ = load_biomedclip(
            vpt_enabled=trainer_cfg.VPT_ENABLED,
            vpt_mode=trainer_cfg.VPT_MODE,
            vpt_num_tokens=trainer_cfg.VPT_N_CTX,
            vpt_dropout=trainer_cfg.VPT_DROPOUT,
        )
        if trainer_cfg.PREC in {"fp32", "amp"}:
            biomedclip_model.float()

        print("Building CE-only CoOp{}".format(" + VPT" if trainer_cfg.VPT_ENABLED else ""))
        self.model = CustomCLIP(cfg, classnames, biomedclip_model.eval())
        self.vpt_enabled = bool(trainer_cfg.VPT_ENABLED)

        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        self.model.prompt_learner.ctx.requires_grad_(True)
        if self.vpt_enabled:
            for parameter in self.model.image_encoder.visual_prompt.parameters():
                parameter.requires_grad_(True)

        if cfg.MODEL.INIT_WEIGHTS:
            load_pretrained_weights(self.model.prompt_learner, cfg.MODEL.INIT_WEIGHTS)

        self.model.to(self.device)
        self.model.eval()
        self.model.prompt_learner.train()
        if self.vpt_enabled:
            self.model.image_encoder.visual_prompt.train()

        coop_parameters = [self.model.prompt_learner.ctx]
        self.coop_optim = build_optimizer(coop_parameters, trainer_cfg.COOP_OPTIM)
        self.coop_sched = build_lr_scheduler(self.coop_optim, trainer_cfg.COOP_OPTIM)
        self.register_model(
            "prompt_learner",
            self.model.prompt_learner,
            self.coop_optim,
            self.coop_sched,
        )

        self.vpt_optim = None
        self.vpt_sched = None
        if self.vpt_enabled:
            visual_prompt = self.model.image_encoder.visual_prompt
            self.vpt_optim = build_optimizer(
                visual_prompt.parameters(), trainer_cfg.VPT_OPTIM
            )
            self.vpt_sched = build_lr_scheduler(
                self.vpt_optim, trainer_cfg.VPT_OPTIM
            )
            self.register_model(
                "vpt_prompt", visual_prompt, self.vpt_optim, self.vpt_sched
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

        coop_ids = {id(parameter) for group in self.coop_optim.param_groups for parameter in group["params"]}
        vpt_ids = set()
        if self.vpt_optim is not None:
            vpt_ids = {id(parameter) for group in self.vpt_optim.param_groups for parameter in group["params"]}
        if coop_ids & vpt_ids:
            raise RuntimeError("CoOp and VPT optimizer parameter sets overlap")
        if coop_ids | vpt_ids != {id(parameter) for parameter in trainable.values()}:
            raise RuntimeError("Every trainable parameter must belong to exactly one optimizer")

        print("Trainable parameters:")
        for name, parameter in trainable.items():
            print("  {} shape={} count={:,}".format(name, tuple(parameter.shape), parameter.numel()))
        print("Total trainable parameters: {:,}".format(sum(p.numel() for p in trainable.values())))

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
        names = self.get_model_names()
        self.model_zero_grad(names)

        if self.cfg.TRAINER.COOPVPT.PREC == "amp":
            with autocast():
                output = self.model(image)
                loss = F.cross_entropy(output, label)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.coop_optim)
            if self.vpt_optim is not None:
                self.scaler.step(self.vpt_optim)
            self.scaler.update()
        else:
            output = self.model(image)
            loss = F.cross_entropy(output, label)
            self.model_backward(loss)
            self.coop_optim.step()
            if self.vpt_optim is not None:
                self.vpt_optim.step()

        loss_summary = {
            "loss": loss.item(),
            "acc": compute_accuracy(output, label)[0].item(),
            "lr_coop": self.coop_optim.param_groups[0]["lr"],
        }
        if self.vpt_optim is not None:
            loss_summary["lr_vpt"] = self.vpt_optim.param_groups[0]["lr"]

        if (self.batch_idx + 1) == self.num_batches:
            self.update_lr(names)
        return loss_summary

    def parse_batch_train(self, batch):
        return batch["img"].to(self.device), batch["label"].to(self.device)

    def resume_model_if_exist(self, directory):
        names = self.get_model_names()
        for name in names:
            if not osp.exists(osp.join(directory, name, "checkpoint")):
                print("No complete CoOp/VPT checkpoint found, train from scratch")
                return 0

        epochs = []
        for name in names:
            epochs.append(
                resume_from_checkpoint(
                    osp.join(directory, name),
                    self._models[name],
                    self._optims[name],
                    self._scheds[name],
                )
            )
        if len(set(epochs)) != 1:
            raise RuntimeError("Prompt checkpoint epochs do not match: {}".format(epochs))
        return epochs[0]

    def load_model(self, directory, epoch=None):
        if not directory:
            print("Note that load_model() is skipped as no pretrained model is given")
            return

        model_file = "model-best.pth.tar" if epoch is None else "model.pth.tar-{}".format(epoch)
        for name in self.get_model_names():
            model_path = osp.join(directory, name, model_file)
            if not osp.exists(model_path):
                raise FileNotFoundError('Model not found at "{}"'.format(model_path))
            checkpoint_data = load_checkpoint(model_path)
            state_dict = checkpoint_data["state_dict"]
            if name == "prompt_learner":
                state_dict.pop("token_prefix", None)
                state_dict.pop("token_suffix", None)
            print(
                'Loading {} from "{}" (epoch={})'.format(
                    name, model_path, checkpoint_data["epoch"]
                )
            )
            self._models[name].load_state_dict(
                state_dict, strict=(name == "vpt_prompt")
            )
