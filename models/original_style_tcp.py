"""Mean-50 projected class knowledge, shared TKE, and one-time BERT injection."""

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

from models.multitext_tcp import MultiTextTCPBertTextEncoder, QuickGELU
from models.text_vpt import TextPromptParameters


class OriginalStyleTCPPromptParameters(nn.Module):
    def __init__(self, prior_dim, hidden_dim, depth, insert_layer=8, enabled=True):
        super().__init__()
        if not 1 <= insert_layer < depth:
            raise ValueError("Original-style INSERT_LAYER must be a middle block (1 <= layer < depth)")
        self.insert_layer = int(insert_layer)
        self.enabled = bool(enabled)
        self.num_tokens = 4
        self.hidden_dim = int(hidden_dim)
        # Only pre-injection prompts are needed when TCP is enabled.
        self.text_prompt = TextPromptParameters(
            hidden_dim, insert_layer if enabled else depth, 4
        )
        self.down_projection = nn.Linear(prior_dim, prior_dim // 4)
        self.activation = QuickGELU()
        self.up_projection = nn.Linear(prior_dim // 4, 4 * hidden_dim)
        for name, value in dict(mode=1, prior_dim=prior_dim, hidden_dim=hidden_dim,
                                depth=depth, insert_layer=insert_layer, num_tokens=4).items():
            self.register_buffer("_meta_" + name, torch.tensor(value))

    @property
    def checkpoint_field_map(self):
        return {name: "_meta_" + name for name in
                ("mode", "prior_dim", "hidden_dim", "depth", "insert_layer", "num_tokens")}

    def checkpoint_metadata(self):
        return {name: int(getattr(self, key).item())
                for name, key in self.checkpoint_field_map.items()}

    def class_tokens(self, prototype):
        reference = self.down_projection.weight
        prototype = prototype.detach().to(device=reference.device, dtype=reference.dtype)
        return self.up_projection(self.activation(self.down_projection(prototype))).reshape(
            prototype.shape[0], 4, self.hidden_dim
        )

    def prompt_for_layer(self, layer_idx, class_tokens, dtype, device):
        if self.enabled and layer_idx == self.insert_layer:
            return class_tokens.to(device=device, dtype=dtype)
        return self.text_prompt.for_layer(layer_idx, class_tokens.shape[0], dtype, device)


class OriginalStyleTCPBertTextEncoder(nn.Module):
    def __init__(self, base_text_encoder, projected_description_bank, classnames,
                 insert_layer=8, enabled=True):
        super().__init__()
        transformer = base_text_encoder.transformer
        if transformer.config.model_type != "bert":
            raise TypeError("Original-style TCP requires a BERT tower")
        bank = projected_description_bank.detach().float()
        if bank.ndim != 3 or bank.shape[:2] != (len(classnames), 50):
            raise ValueError("Expected projected description bank [C, 50, D_proj]")
        self.base_text_encoder = base_text_encoder
        self.register_buffer("description_bank", bank, persistent=False)
        self.register_buffer("class_prior", F.normalize(bank.mean(dim=1), dim=-1), persistent=False)
        self.num_tokens = 4
        self.insert_layer = int(insert_layer)
        self.tcp_prompt = OriginalStyleTCPPromptParameters(
            bank.shape[-1], transformer.config.hidden_size,
            len(transformer.encoder.layer), self.insert_layer, enabled
        )

    @property
    def transformer(self):
        return self.base_text_encoder.transformer

    # Reuse only sequence/mask plumbing, never MultiText aggregation or gates.
    _attention_mask = MultiTextTCPBertTextEncoder._attention_mask
    _reserve_prompt_slots = MultiTextTCPBertTextEncoder._reserve_prompt_slots
    _extended_mask = MultiTextTCPBertTextEncoder._extended_mask
    _replace_slots = MultiTextTCPBertTextEncoder._replace_slots

    def aggregate_class_tokens(self):
        return self.tcp_prompt.class_tokens(self.class_prior)

    def forward(self, prompts, tokenized_prompts):
        reference = self.tcp_prompt.down_projection.weight
        prompts = prompts.to(dtype=reference.dtype)
        attention_mask = self._attention_mask(tokenized_prompts, prompts)
        prompts, attention_mask = self._reserve_prompt_slots(prompts, attention_mask)
        class_tokens = self.aggregate_class_tokens()
        first_prompt = self.tcp_prompt.prompt_for_layer(
            0, class_tokens, prompts.dtype, prompts.device
        )
        hidden_inputs = torch.cat((prompts[:, :1], first_prompt, prompts[:, 1:]), dim=1)
        augmented_mask = torch.cat(
            (
                attention_mask[:, :1],
                torch.ones(
                    attention_mask.shape[0],
                    self.num_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask[:, 1:],
            ),
            dim=1,
        )
        position_ids = torch.arange(
            hidden_inputs.shape[1], device=prompts.device, dtype=torch.long
        ).unsqueeze(0).expand(prompts.shape[0], -1)
        hidden_states = self.transformer.embeddings(
            inputs_embeds=hidden_inputs,
            token_type_ids=torch.zeros_like(augmented_mask),
            position_ids=position_ids,
        )

        layers = self.transformer.encoder.layer
        extended_mask = self._extended_mask(augmented_mask, hidden_states.shape[:2])
        head_mask = self.transformer.get_head_mask(None, len(layers))
        for layer_idx, layer in enumerate(layers):
            if 0 < layer_idx <= self.insert_layer or (layer_idx > 0 and not self.tcp_prompt.enabled):
                layer_prompt = self.tcp_prompt.prompt_for_layer(
                    layer_idx, class_tokens, hidden_states.dtype, hidden_states.device
                )
                hidden_states = self._replace_slots(hidden_states, layer_prompt)
            hidden_states = layer(
                hidden_states,
                attention_mask=extended_mask,
                head_mask=head_mask[layer_idx],
            )[0]

        transformer_pooler = getattr(self.transformer, "pooler", None)
        pooled_output = transformer_pooler(hidden_states) if transformer_pooler else None
        output = BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=pooled_output
        )
        pooled = self.base_text_encoder.pooler(output, augmented_mask)
        projected = self.base_text_encoder.proj(pooled)
        if getattr(self.base_text_encoder, "output_tokens", False):
            return projected, hidden_states[:, 1 + self.num_tokens :]
        return projected
