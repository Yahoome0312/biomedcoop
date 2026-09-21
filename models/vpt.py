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

    def __init__(
        self,
        embed_dim,
        num_tokens,
        depth,
        mode="shallow",
        dropout=0.0,
        patch_size=(16, 16),
        prompt_depth=None,
        prototype_fusion=False,
        fusion_alpha=0.5,
    ):
        super().__init__()
        if mode not in {"shallow", "deep"}:
            raise ValueError("VPT mode must be 'shallow' or 'deep', got {!r}".format(mode))
        if num_tokens < 1:
            raise ValueError("VPT_N_CTX must be positive")
        if depth < 1:
            raise ValueError("The visual transformer must have at least one block")
        prompt_depth = depth if prompt_depth is None else int(prompt_depth)
        if not 1 <= prompt_depth <= depth:
            raise ValueError("Visual prompt depth must be within the transformer depth")

        self.mode = mode
        self.num_tokens = int(num_tokens)
        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.prompt_depth = prompt_depth
        parameter_depth = 1 if mode == "shallow" else prompt_depth
        self.prompt_embeddings = nn.Parameter(
            torch.empty(parameter_depth, self.num_tokens, self.embed_dim)
        )
        self.dropout = nn.Dropout(float(dropout))

        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        patch_area = math.prod(tuple(int(v) for v in patch_size))
        bound = math.sqrt(6.0 / float(3 * patch_area + self.embed_dim))
        nn.init.uniform_(self.prompt_embeddings, -bound, bound)
        self.fusion_alpha = float(fusion_alpha) if prototype_fusion else None
        if prototype_fusion:
            with torch.random.fork_rng(devices=[]):
                self.fusion_prompt = nn.Parameter(torch.empty(self.num_tokens, self.embed_dim))
                nn.init.uniform_(self.fusion_prompt, -bound, bound)

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

    def __init__(
        self,
        base_visual,
        num_tokens,
        mode="shallow",
        dropout=0.0,
        prompt_depth=None,
        prototype_fusion=False,
        fusion_alpha=0.5,
    ):
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
            prompt_depth=prompt_depth,
            prototype_fusion=prototype_fusion,
            fusion_alpha=fusion_alpha,
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

    def _replace_prompt_with_values(self, x, prompt):
        """Replace the visual prompt slots with externally generated values."""
        prompt_end = self.num_prefix_tokens + self.num_prompt_tokens
        return torch.cat(
            (x[:, : self.num_prefix_tokens], prompt, x[:, prompt_end:]), dim=1
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

    def forward_before_layer(self, image, layer_idx):
        """Run image embedding and blocks before ``layer_idx`` exactly once."""
        layer_idx = int(layer_idx)
        trunk = self.trunk
        if not 1 <= layer_idx < len(trunk.blocks):
            raise ValueError("CVP insertion layer must be inside the visual transformer")

        x = trunk.patch_embed(image)
        x = trunk._pos_embed(x)
        x = trunk.patch_drop(x)
        x = trunk.norm_pre(x)
        x = self._insert_prompt(x, 0)
        for current_idx, block in enumerate(trunk.blocks[:layer_idx]):
            if self.mode == "deep" and current_idx > 0:
                x = self._replace_prompt(x, current_idx)
            x = self._run_block(trunk, block, x)
        return x

    def forward_from_layer(
        self,
        x,
        layer_idx,
        prototype_prompts=None,
        return_tokens=False,
    ):
        """Run the visual tail with optional one-time prompt replacement."""
        layer_idx = int(layer_idx)
        trunk = self.trunk
        if not 1 <= layer_idx < len(trunk.blocks):
            raise ValueError("CVP insertion layer must be inside the visual transformer")
        if x.ndim != 3:
            raise ValueError("Expected shared visual state [B, L, D]")

        batch_size = x.shape[0]
        num_classes = None
        if prototype_prompts is not None:
            if prototype_prompts.ndim != 4:
                raise ValueError("Expected prototype prompts [B, C, T, D]")
            if prototype_prompts.shape[0] != batch_size:
                raise ValueError("Visual states and prototype prompts disagree")
            if prototype_prompts.shape[-1] != x.shape[-1]:
                raise ValueError("Prototype prompt hidden size is incorrect")
            num_classes = prototype_prompts.shape[1]
            num_prototype_tokens = prototype_prompts.shape[2]
            if num_classes < 1 or num_prototype_tokens < 1:
                raise ValueError("Prototype prompt class and token counts must be positive")
            if num_prototype_tokens != self.num_prompt_tokens:
                raise ValueError("Visual TKE tokens must match the visual prompt slots")
            x = x.unsqueeze(1).expand(-1, num_classes, -1, -1).reshape(
                batch_size * num_classes, x.shape[1], x.shape[2]
            )
            prototype_prompts = prototype_prompts.to(
                device=x.device, dtype=x.dtype
            ).reshape(batch_size * num_classes, num_prototype_tokens, x.shape[-1])

        for current_idx in range(layer_idx, len(trunk.blocks)):
            if prototype_prompts is None and self.mode == "deep":
                x = self._replace_prompt(x, current_idx)
            if current_idx == layer_idx and prototype_prompts is not None:
                alpha = self.visual_prompt.fusion_alpha
                if alpha is not None:
                    prototype_prompts = (
                        alpha * prototype_prompts
                        + (1 - alpha) * self.visual_prompt.fusion_prompt.to(dtype=x.dtype)
                    )
                x = self._replace_prompt_with_values(x, prototype_prompts)
            x = self._run_block(trunk, trunk.blocks[current_idx], x)

        x = self._remove_prompt(x)
        x = trunk.norm(x)
        patch_tokens = x[:, self.num_prefix_tokens :]
        pooled = trunk.forward_head(x)
        projected = self.head(pooled)
        if num_classes is not None:
            projected = projected.reshape(batch_size, num_classes, -1)
            patch_tokens = patch_tokens.reshape(
                batch_size, num_classes, patch_tokens.shape[1], patch_tokens.shape[2]
            )
        if return_tokens:
            return projected, patch_tokens
        return projected
