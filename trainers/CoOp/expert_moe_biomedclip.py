"""Train only a linear router over two historical CoOpVPT specialists."""
from pathlib import Path

import torch
from dassl.engine import TRAINER_REGISTRY, TrainerX
from dassl.metrics import compute_accuracy
from dassl.optim import build_optimizer, build_lr_scheduler
from dassl.utils import load_checkpoint

from models.expert_moe import ExpertMoE, router_loss
from trainers.CoOp.coop_vpt_biomedclip import CoOpVPT_BiomedCLIP


def load_expert(cfg, dm, device, checkpoint, tcp_enabled):
    if not checkpoint or not Path(checkpoint).is_file():
        raise FileNotFoundError(f"Historical expert checkpoint required: {checkpoint}")
    expert_cfg = cfg.clone()
    expert_cfg.defrost()
    expert_cfg.TRAINER.TCP.ENABLED = tcp_enabled
    expert_cfg.TRAINER.CONFUSION_AWARE.ENABLED = not tcp_enabled
    prefix = "TCP" if tcp_enabled else "CONF"
    for bank in ("DESCRIPTION_CACHE",):
        path = getattr(cfg.TRAINER.EXPERT_MOE, f"{prefix}_{bank}")
        if path:
            setattr(expert_cfg.TRAINER.TCP, bank, path)
    expert_cfg.freeze()
    # Reuse the original builder and strict checkpoint validator, without
    # creating another data manager, optimizer, manifest or training loop.
    builder = CoOpVPT_BiomedCLIP.__new__(CoOpVPT_BiomedCLIP)
    builder.cfg, builder.dm, builder.device = expert_cfg, dm, device
    builder.check_cfg(expert_cfg)
    builder.build_model(
        expert_checkpoint=checkpoint, rebuild_banks=cfg.TRAINER.EXPERT_MOE.REBUILD_BANKS
    )
    return builder.model


@TRAINER_REGISTRY.register()
class ExpertMoE_BiomedCLIP(TrainerX):
    def check_cfg(self, cfg):
        if not cfg.TRAINER.EXPERT_MOE.ENABLED:
            raise ValueError("Set TRAINER.EXPERT_MOE.ENABLED True")
        if cfg.TRAINER.EXPERT_MOE.ROUTER != "linear":
            raise ValueError("Only the linear router is supported")

    def build_model(self):
        cfg = self.cfg
        if self.val_loader is None:
            raise ValueError("Router selection requires the existing validation split")
        moe = cfg.TRAINER.EXPERT_MOE
        tcp = load_expert(cfg, self.dm, self.device, moe.TCP_CHECKPOINT, True)
        conf = load_expert(cfg, self.dm, self.device, moe.CONF_CHECKPOINT, False)
        self.model = ExpertMoE(tcp, conf, self.num_classes, moe.MODE).to(self.device)
        self.optim = build_optimizer(self.model.router, cfg.OPTIM)
        self.sched = build_lr_scheduler(self.optim, cfg.OPTIM)
        self.register_model("router", self.model.router, self.optim, self.sched)
        for name, expert in (("TCP", tcp), ("Confusion", conf)):
            count = sum(p.numel() for p in expert.parameters() if p.requires_grad)
            assert count == 0
            print(f"{name} Expert trainable params = {count}")
        count = sum(p.numel() for p in self.model.router.parameters())
        assert count == 6 * self.num_classes + 2
        assert {id(p) for g in self.optim.param_groups for p in g["params"]} == {
            id(p) for p in self.model.router.parameters()
        }
        print(f"Router trainable params = {count}")
        # Keep the same dataset, sampler and augmentation; include the tail batch.
        self.train_loader_x.batch_sampler.drop_last = False

    def set_model_mode(self, mode="train", names=None):
        if mode not in {"train", "eval", "test"}:
            raise KeyError(mode)
        self.model.train(mode == "train")

    def model_inference(self, image):
        return self.model(image)

    def forward_backward(self, batch):
        if self.model.mode != "linear_moe":
            raise ValueError("Single-expert modes are evaluation-only; use --eval-only")
        image = batch["img"].to(self.device)
        label = batch["label"].to(self.device)
        # Frozen experts use their loaded dtype; router and mixture use fp32.
        output = self.model(image)
        loss = router_loss(output, label)
        self.model_backward_and_update(loss)
        if self.batch_idx + 1 == self.num_batches:
            self.update_lr()
        return {"loss": loss.item(), "acc": compute_accuracy(output, label)[0].item()}

    def load_model(self, directory, epoch=None):
        if self.model.mode != "linear_moe":
            return
        filename = "model-best.pth.tar" if epoch is None else f"model.pth.tar-{epoch}"
        checkpoint = load_checkpoint(str(Path(directory) / "router" / filename))
        self.model.router.load_state_dict(checkpoint["state_dict"], strict=True)

    @torch.no_grad()
    def test(self, split=None):
        if (split or self.cfg.TEST.SPLIT) == "val" and self.val_loader is None:
            raise ValueError("Validation cannot fall back to test data")
        return super().test(split)
