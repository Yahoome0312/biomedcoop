"""Frozen specialist predictions mixed by a zero-initialized linear router."""
import torch
from torch import nn
from torch.nn import functional as F


class ExpertMoE(nn.Module):
    def __init__(self, tcp_expert, conf_expert, num_classes, mode="linear_moe"):
        super().__init__()
        if mode not in {"tcp_only", "conf_only", "linear_moe"}:
            raise ValueError(f"Unknown expert mode: {mode}")
        if {id(p) for p in tcp_expert.parameters()} & {id(p) for p in conf_expert.parameters()}:
            raise ValueError("Experts must have independent parameter objects")
        self.tcp_expert = tcp_expert.requires_grad_(False).eval()
        self.conf_expert = conf_expert.requires_grad_(False).eval()
        self.mode = mode
        self.router = nn.Linear(3 * num_classes, 2, bias=True)
        nn.init.zeros_(self.router.weight)
        nn.init.zeros_(self.router.bias)

    def train(self, mode=True):
        super().train(mode)
        self.tcp_expert.eval()
        self.conf_expert.eval()
        return self

    def forward(self, image, return_weights=False):
        with torch.no_grad():
            if self.mode == "tcp_only":
                return self.tcp_expert(image)
            if self.mode == "conf_only":
                return self.conf_expert(image)
            p_tcp = self.tcp_expert(image).float().softmax(dim=-1)
            p_conf = self.conf_expert(image).float().softmax(dim=-1)
        router_input = torch.cat([p_tcp, p_conf, (p_tcp - p_conf).abs()], dim=-1)
        weights = self.router(router_input).softmax(dim=-1)
        p_final = weights[:, :1] * p_tcp + weights[:, 1:] * p_conf
        log_probs = p_final.clamp_min(1e-8).log()
        return (log_probs, weights) if return_weights else log_probs


def router_loss(log_probs, label):
    return F.nll_loss(log_probs, label)
