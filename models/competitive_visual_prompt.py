"""Visual prompts from frozen prototypes or trainable Text MLP tokens."""

import torch
from torch import nn


class QuickGELU(nn.Module):
    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


class VisualPrototypePrompt(nn.Module):
    """Shared Visual TKE; serial mode maps each text token independently."""

    def __init__(self, prior_dim=512, bottleneck_dim=128, num_tokens=4, hidden_dim=768,
                 serial=False):
        super().__init__()
        self.prior_dim = int(prior_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.serial = bool(serial)
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
            self.bottleneck_dim, self.hidden_dim if self.serial else self.num_tokens * self.hidden_dim
        )

    def forward(self, class_prototypes):
        """Map prototypes [C,D] or serial text tokens [C,4,D] to visual tokens."""
        if class_prototypes.ndim != (3 if self.serial else 2):
            raise ValueError("Expected text tokens [C, T, D] or parallel prototypes [C, D]")
        if class_prototypes.shape[-1] != self.prior_dim:
            raise ValueError("Class prototype dimension is incorrect")
        prototypes = class_prototypes if self.serial else class_prototypes.detach()
        prototypes = prototypes.to(
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
