"""Independent image MLP driven by the frozen Mean-50 class prior."""

from torch import nn

from .original_style_tcp import QuickGELU


class ClassConditionedVisualPrompt(nn.Module):
    def __init__(self, prior_dim=512, hidden_dim=768, num_tokens=4, bottleneck_dim=128):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.down_projection = nn.Linear(prior_dim, bottleneck_dim)
        self.activation = QuickGELU()
        self.up_projection = nn.Linear(bottleneck_dim, self.num_tokens * self.hidden_dim)

    def forward(self, class_prior):
        reference = self.down_projection.weight
        prior = class_prior.detach().to(device=reference.device, dtype=reference.dtype)
        return self.up_projection(self.activation(self.down_projection(prior))).reshape(
            prior.shape[0], self.num_tokens, self.hidden_dim
        )
