"""Visual Prompt Tuning for the timm ViT used by BiomedCLIP.

The implementation follows the shallow/deep prompt replacement semantics from
KMnP/vpt while preserving the original timm pooling and OpenCLIP projection.
"""

import math

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class VisualPromptParameters(nn.Module):
    """The only trainable part of the frozen visual tower."""

    def __init__(self, embed_dim, num_tokens, depth, mode="shallow", dropout=0.0, patch_size=(16, 16)):
        super().__init__()
        if mode not in {"shallow", "deep"}:
            raise ValueError("VPT mode must be 'shallow' or 'deep', got {!r}".format(mode))
        if num_tokens < 1:
            raise ValueError("VPT_N_CTX must be positive")
        if depth < 1:
            raise ValueError("The visual transformer must have at least one block")

        self.mode = mode
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        prompt_depth = 1 if mode == "shallow" else depth
        self.prompt_embeddings = nn.Parameter(
            torch.empty(prompt_depth, self.num_tokens, self.embed_dim)
        )
        self.dropout = nn.Dropout(float(dropout))

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        patch_area = math.prod(tuple(int(v) for v in patch_size))
        bound = math.sqrt(6.0 / float(3 * patch_area + self.embed_dim))
        nn.init.uniform_(self.prompt_embeddings, -bound, bound)

    def for_layer(self, layer_idx, batch_size, dtype, device):
        index = 0 if self.mode == "shallow" else layer_idx
        prompt = self.prompt_embeddings[index]
        prompt = prompt.to(device=device, dtype=dtype)
        prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(prompt)


class TimmViTVisualPromptEncoder(nn.Module):
    """Wrap an OpenCLIP TimmModel and inject prompts into its ViT trunk."""

    REQUIRED_TRUNK_ATTRIBUTES = (
        "patch_embed",
        "_pos_embed",
        "patch_drop",
        "norm_pre",
        "blocks",
        "norm",
        "forward_head",
        "num_prefix_tokens",
    )

    def __init__(self, base_visual, num_tokens, mode="shallow", dropout=0.0):
        super().__init__()
        self.base_visual = base_visual
        trunk = base_visual.trunk
        missing = [name for name in self.REQUIRED_TRUNK_ATTRIBUTES if not hasattr(trunk, name)]
        if missing:
            raise TypeError("Unsupported timm visual trunk; missing: {}".format(", ".join(missing)))

        embed_dim = getattr(trunk, "embed_dim", getattr(trunk, "num_features", None))
        if embed_dim is None:
            raise TypeError("Cannot infer visual prompt dimension from the timm trunk")
        patch_size = getattr(trunk.patch_embed, "patch_size", (16, 16))
        self.visual_prompt = VisualPromptParameters(
            embed_dim=embed_dim,
            num_tokens=num_tokens,
            depth=len(trunk.blocks),
            mode=mode,
            dropout=dropout,
            patch_size=patch_size,
        )
        self.mode = mode
        self.num_prompt_tokens = int(num_tokens)
        self.num_prefix_tokens = int(trunk.num_prefix_tokens)

    @property
    def trunk(self):
        return self.base_visual.trunk

    @property
    def head(self):
        return self.base_visual.head

    def _insert_prompt(self, x, layer_idx):
        prefix = x[:, : self.num_prefix_tokens]
        patches = x[:, self.num_prefix_tokens :]
        prompt = self.visual_prompt.for_layer(
            layer_idx, x.shape[0], x.dtype, x.device
        )
        return torch.cat((prefix, prompt, patches), dim=1)

    def _replace_prompt(self, x, layer_idx):
        prefix = x[:, : self.num_prefix_tokens]
        patches = x[:, self.num_prefix_tokens + self.num_prompt_tokens :]
        prompt = self.visual_prompt.for_layer(
            layer_idx, x.shape[0], x.dtype, x.device
        )
        return torch.cat((prefix, prompt, patches), dim=1)

    def _remove_prompt(self, x):
        return torch.cat(
            (
                x[:, : self.num_prefix_tokens],
                x[:, self.num_prefix_tokens + self.num_prompt_tokens :],
            ),
            dim=1,
        )

    @staticmethod
    def _run_block(trunk, block, x):
        if trunk.grad_checkpointing and not torch.jit.is_scripting():
            return checkpoint(block, x, use_reentrant=False)
        return block(x)

    def forward(self, image, return_tokens=False):
        trunk = self.trunk
        x = trunk.patch_embed(image)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)
        x = trunk.norm_pre(x)

        x = self._insert_prompt(x, 0)
        for layer_idx, block in enumerate(trunk.blocks):
            if self.mode == "deep" and layer_idx > 0:
                x = self._replace_prompt(x, layer_idx)
            x = self._run_block(trunk, block, x)

        x = self._remove_prompt(x)
        x = trunk.norm(x)
        patch_tokens = x[:, self.num_prefix_tokens :]
        pooled = trunk.forward_head(x)
        projected = self.head(pooled)
        if return_tokens:
            return projected, patch_tokens
        return projected
