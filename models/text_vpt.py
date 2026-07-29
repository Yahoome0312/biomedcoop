"""Text-side Deep Prompt Tuning for BiomedCLIP's Hugging Face BERT tower.

The BiomedCLIP text tower is an OpenCLIP ``HFTextEncoder`` around a
Hugging Face BERT model.  This module keeps that wrapper and all of its
weights frozen, but runs the BERT embeddings and encoder layers explicitly so
that a different trainable prompt can be inserted before every encoder layer.
"""

import torch
from torch import nn

try:
    from transformers.modeling_outputs import BaseModelOutputWithPooling
except ImportError as exc:  # pragma: no cover - the project requires transformers
    raise RuntimeError("Text Deep Prompt requires transformers") from exc


class TextPromptParameters(nn.Module):
    """The trainable parameters used by text Deep Prompt Tuning."""

    def __init__(
        self,
        embed_dim,
        depth,
        num_tokens,
        dropout=0.0,
        init="normal",
    ):
        super().__init__()
        if int(depth) < 1:
            raise ValueError("The text Transformer must have at least one layer")
        if int(num_tokens) < 1:
            raise ValueError("TEXT_VPT_N_CTX must be positive")
        if init != "normal":
            raise ValueError("Text Deep Prompt currently supports init='normal' only")

        self.embed_dim = int(embed_dim)
        self.depth = int(depth)
        self.num_tokens = int(num_tokens)
        self.prompt_embeddings = nn.Parameter(
            torch.empty(self.depth, self.num_tokens, self.embed_dim)
        )
        self.dropout = nn.Dropout(float(dropout))
        nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)

    def for_layer(self, layer_idx, batch_size, dtype, device):
        if not 0 <= int(layer_idx) < self.depth:
            raise IndexError("Text prompt layer index out of range: {}".format(layer_idx))
        prompt = self.prompt_embeddings[int(layer_idx)]
        prompt = prompt.to(device=device, dtype=dtype)
        prompt = prompt.unsqueeze(0).expand(batch_size, -1, -1)
        return self.dropout(prompt)


class BertTextDeepPromptEncoder(nn.Module):
    """Run the frozen BiomedCLIP BERT tower with per-layer text prompts.

    The public forward interface matches the project-local ``TextEncoder``:
    it receives CoOp's continuous prompt embeddings and their tokenized form.
    Tensors use the Hugging Face layout ``[batch, sequence, hidden]``.
    """

    def __init__(
        self,
        base_text_encoder,
        num_tokens=4,
        dropout=0.0,
        init="normal",
    ):
        super().__init__()
        self.base_text_encoder = base_text_encoder
        transformer = base_text_encoder.transformer

        if not hasattr(transformer, "embeddings"):
            raise TypeError("Text Deep Prompt requires a Hugging Face embedding module")
        if not hasattr(transformer, "encoder") or not hasattr(transformer.encoder, "layer"):
            raise TypeError(
                "Text Deep Prompt requires a Transformer with encoder.layer blocks"
            )
        if not hasattr(transformer, "pooler"):
            raise TypeError("Text Deep Prompt requires a BERT-style pooler")

        model_type = getattr(getattr(transformer, "config", None), "model_type", None)
        if model_type != "bert":
            raise TypeError(
                "Text Deep Prompt currently supports BERT only, got {!r}".format(
                    model_type
                )
            )

        embedding_layer = transformer.embeddings.word_embeddings
        embed_dim = getattr(embedding_layer, "embedding_dim", None)
        if embed_dim is None:
            embed_dim = getattr(transformer.config, "hidden_size", None)
        if embed_dim is None:
            raise TypeError("Cannot infer text prompt embedding dimension")

        depth = len(transformer.encoder.layer)
        self.text_prompt = TextPromptParameters(
            embed_dim=embed_dim,
            depth=depth,
            num_tokens=num_tokens,
            dropout=dropout,
            init=init,
        )
        self.num_prompt_tokens = int(num_tokens)
        self.depth = int(depth)
        self.embed_dim = int(embed_dim)
        self.max_position_embeddings = int(
            getattr(transformer.config, "max_position_embeddings", 0)
        )
        if self.max_position_embeddings < 1:
            raise TypeError("Cannot infer BERT max_position_embeddings")

    @property
    def transformer(self):
        return self.base_text_encoder.transformer

    def _attention_mask(self, tokenized_prompts, prompts):
        if tokenized_prompts is None:
            raise ValueError("tokenized_prompts is required for text Deep Prompt")
        tokenized_prompts = tokenized_prompts.to(device=prompts.device)
        if tokenized_prompts.dim() != 2 or prompts.dim() != 3:
            raise ValueError(
                "Expected tokenized prompts [B,L] and embeddings [B,L,D], got {} and {}".format(
                    tuple(tokenized_prompts.shape), tuple(prompts.shape)
                )
            )
        if tokenized_prompts.shape[:2] != prompts.shape[:2]:
            raise ValueError(
                "Tokenized prompts and embeddings disagree: {} vs {}".format(
                    tuple(tokenized_prompts.shape), tuple(prompts.shape)
                )
            )

        pad_token_id = getattr(self.transformer.config, "pad_token_id", 0)
        return tokenized_prompts.ne(pad_token_id).to(dtype=torch.long)

    def _reserve_prompt_slots(self, prompts, attention_mask):
        """Reserve prompt slots without silently dropping valid text tokens."""

        batch_size, base_length, _ = prompts.shape
        available_length = self.max_position_embeddings - self.num_prompt_tokens
        if available_length < 1:
            raise ValueError(
                "Not enough BERT positions for text prompts: max_position_embeddings={} "
                "and num_prompt_tokens={}".format(
                    self.max_position_embeddings, self.num_prompt_tokens
                )
            )
        if base_length > self.max_position_embeddings:
            raise ValueError(
                "CoOp text sequence length {} exceeds BERT max_position_embeddings {}".format(
                    base_length, self.max_position_embeddings
                )
            )

        keep_length = min(base_length, available_length)
        if attention_mask[:, keep_length:].any():
            raise ValueError(
                "Adding text Deep Prompt would truncate a valid token. "
                "The current tokenizer sequence has no sufficient trailing padding."
            )
        if not attention_mask[:, 0].all():
            raise ValueError("The first text token must be the active CLS/SOS token")

        return prompts[:, :keep_length, :], attention_mask[:, :keep_length], batch_size

    def _make_extended_attention_mask(self, attention_mask, input_shape):
        transformer = self.transformer
        try:
            return transformer.get_extended_attention_mask(
                attention_mask, input_shape, device=attention_mask.device
            )
        except TypeError:
            return transformer.get_extended_attention_mask(
                attention_mask, input_shape
            )

    def _run_encoder(self, hidden_states, attention_mask):
        transformer = self.transformer
        layers = transformer.encoder.layer
        extended_attention_mask = self._make_extended_attention_mask(
            attention_mask, hidden_states.shape[:2]
        )
        head_mask = transformer.get_head_mask(None, len(layers))

        for layer_idx, layer in enumerate(layers):
            if layer_idx > 0:
                prompt = self.text_prompt.for_layer(
                    layer_idx,
                    hidden_states.shape[0],
                    hidden_states.dtype,
                    hidden_states.device,
                )
                hidden_states = torch.cat(
                    (
                        hidden_states[:, :1, :],
                        prompt,
                        hidden_states[:, 1 + self.num_prompt_tokens :, :],
                    ),
                    dim=1,
                )

            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[layer_idx],
            )
            hidden_states = layer_outputs[0]

        return hidden_states

    def forward(self, prompts, tokenized_prompts):
        if prompts.dtype != self.text_prompt.prompt_embeddings.dtype:
            prompts = prompts.to(dtype=self.text_prompt.prompt_embeddings.dtype)
        attention_mask = self._attention_mask(tokenized_prompts, prompts)
        prompts, attention_mask, batch_size = self._reserve_prompt_slots(
            prompts, attention_mask
        )

        first_prompt = self.text_prompt.for_layer(
            0, batch_size, prompts.dtype, prompts.device
        )
        hidden_inputs = torch.cat(
            (
                prompts[:, :1, :],
                first_prompt,
                prompts[:, 1:, :],
            ),
            dim=1,
        )
        augmented_mask = torch.cat(
            (
                attention_mask[:, :1],
                torch.ones(
                    batch_size,
                    self.num_prompt_tokens,
                    dtype=attention_mask.dtype,
                    device=attention_mask.device,
                ),
                attention_mask[:, 1:],
            ),
            dim=1,
        )

        position_ids = torch.arange(
            hidden_inputs.shape[1], device=hidden_inputs.device, dtype=torch.long
        ).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = torch.zeros_like(augmented_mask)
        hidden_states = self.transformer.embeddings(
            inputs_embeds=hidden_inputs,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )
        hidden_states = self._run_encoder(hidden_states, augmented_mask)

        transformer_pooler = getattr(self.transformer, "pooler", None)
        pooled_output = (
            transformer_pooler(hidden_states)
            if transformer_pooler is not None
            else None
        )
        output = BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
        )
        pooled = self.base_text_encoder.pooler(output, augmented_mask)
        projected = self.base_text_encoder.proj(pooled)

        if getattr(self.base_text_encoder, "output_tokens", False):
            tokens = hidden_states[:, 1 + self.num_prompt_tokens :, :]
            return projected, tokens
        return projected
