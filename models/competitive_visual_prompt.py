"""Visual prompt generation from frozen Mean-50 class prototypes."""

import torch
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


class VisualPrototypePrompt(nn.Module):
    """Shared 512 -> 128 -> (4 x 768) Visual TKE, symmetric to Text TKE."""

    def __init__(self, prior_dim=512, bottleneck_dim=128, num_tokens=4, hidden_dim=768):
        super().__init__()
        self.prior_dim = int(prior_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        if min(
            self.prior_dim,
            self.bottleneck_dim,
            self.num_tokens,
            self.hidden_dim,
        ) < 1:
            raise ValueError("Visual TKE dimensions must be positive")
        self.down_projection = nn.Linear(self.prior_dim, self.bottleneck_dim)
        self.activation = QuickGELU()
        self.up_projection = nn.Linear(
            self.bottleneck_dim, self.num_tokens * self.hidden_dim
        )

    def forward(self, class_prototypes):
        """Map frozen class prototypes [C, 512] to visual tokens [C, 4, 768]."""
        if class_prototypes.ndim != 2:
            raise ValueError("Expected class prototypes [C, D]")
        if class_prototypes.shape[1] != self.prior_dim:
            raise ValueError("Class prototype dimension is incorrect")
        prototypes = class_prototypes.detach().to(
            device=self.down_projection.weight.device,
            dtype=self.down_projection.weight.dtype,
        )
        prompts = self.up_projection(
            self.activation(self.down_projection(prototypes))
        )
        return prompts.reshape(
            prototypes.shape[0],
            self.num_tokens,
            self.hidden_dim,
        )
