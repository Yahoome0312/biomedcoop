"""Textual-based Class-aware Prompt tuning for BiomedCLIP's BERT tower.

This is a project-local reimplementation of TCP's Textual Knowledge Embedding
(TKE).  It keeps the Hugging Face BERT tower, pooling, and projection frozen,
and replaces CoOp's context positions immediately before one encoder block.
"""

import hashlib
import json

import torch
from torch import nn
from torch.nn import functional as F

from models.biomedclip_loader import BIOMEDCLIP_MODEL_ID

try:
    from transformers.modeling_outputs import BaseModelOutputWithPooling
except ImportError as exc:  # pragma: no cover - transformers is a project dependency
    raise RuntimeError("TCP requires transformers") from exc


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _text_fingerprint(value):
    return _sha256_bytes(value.encode("utf-8"))


def _encode_hash(value):
    return torch.tensor(list(value.encode("ascii")), dtype=torch.uint8)


def _decode_hash(value):
    return bytes(value.detach().cpu().tolist()).decode("ascii")


class QuickGELU(nn.Module):
    """The activation used by the original TCP TKE."""

    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


class TCPPromptParameters(nn.Module):
    """Trainable TKE projections plus immutable checkpoint compatibility data."""

    _METADATA_FIELDS = (
        "category_order_fingerprint",
        "template_fingerprint",
        "model_fingerprint",
        "prior_fingerprint",
    )

    def __init__(
        self,
        prior_dim,
        hidden_dim,
        num_tokens,
        bottleneck_dim,
        metadata,
        fusion_mode="replace",
        fusion_weight=1.0,
    ):
        super().__init__()
        self.prior_dim = int(prior_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.bottleneck_dim = int(bottleneck_dim)
        self.down_projection = nn.Linear(self.prior_dim, self.bottleneck_dim)
        self.activation = QuickGELU()
        self.up_projection = nn.Linear(
            self.bottleneck_dim, self.num_tokens * self.hidden_dim
        )
        self.fusion_mode = str(fusion_mode)
        if self.fusion_mode == "gated_residual":
            initial_weight = float(fusion_weight)
            if not 0.0 < initial_weight < 1.0:
                raise ValueError("Gated residual TCP requires 0 < fusion_weight < 1")
            initial_logit = torch.logit(torch.tensor(initial_weight))
            self.fusion_logit = nn.Parameter(initial_logit)

        self.register_buffer("_meta_num_tokens", torch.tensor(self.num_tokens))
        self.register_buffer("_meta_hidden_dim", torch.tensor(self.hidden_dim))
        self.register_buffer("_meta_prior_dim", torch.tensor(self.prior_dim))
        for field in self._METADATA_FIELDS:
            self.register_buffer("_meta_{}".format(field), _encode_hash(metadata[field]))

    def forward(self, class_prior):
        prompt = self.up_projection(
            self.activation(self.down_projection(class_prior))
        )
        return prompt.reshape(-1, self.num_tokens, self.hidden_dim)

    def fusion_alpha(self):
        if self.fusion_mode == "replace":
            return self.up_projection.weight.new_ones(())
        if self.fusion_mode == "gated_residual":
            return self.fusion_logit.sigmoid()
        raise KeyError(self.fusion_mode)

    def checkpoint_metadata(self):
        result = {
            "num_tokens": int(self._meta_num_tokens.item()),
            "hidden_dim": int(self._meta_hidden_dim.item()),
            "prior_dim": int(self._meta_prior_dim.item()),
        }
        for field in self._METADATA_FIELDS:
            result[field] = _decode_hash(getattr(self, "_meta_{}".format(field)))
        return result


def build_frozen_class_prior(model, tokenizer, classnames, template):
    """Encode the single TCP knowledge template with frozen vanilla BiomedCLIP."""

    normalized_names = [name.replace("_", " ") for name in classnames]
    texts = [template.format(name) for name in normalized_names]
    tokenized = torch.cat([tokenizer(text) for text in texts])
    model_device = next(model.parameters()).device
    with torch.no_grad():
        prior = model.encode_text(tokenized.to(model_device), normalize=True)
    return prior.detach().to(dtype=torch.float32, device="cpu")


class TCPBertTextEncoder(nn.Module):
    """BERT adapter implementing TCP's class-aware intermediate replacement."""

    def __init__(
        self,
        base_text_encoder,
        class_prior,
        classnames,
        template="a photo of a {}.",
        num_tokens=4,
        bottleneck_dim=128,
        insert_layer=8,
        fusion_mode="replace",
        fusion_weight=1.0,
        model_id=BIOMEDCLIP_MODEL_ID,
    ):
        super().__init__()
        transformer = base_text_encoder.transformer
        if getattr(getattr(transformer, "config", None), "model_type", None) != "bert":
            raise TypeError("TCP currently requires a Hugging Face BERT text tower")
        if not hasattr(transformer, "encoder") or not hasattr(transformer.encoder, "layer"):
            raise TypeError("TCP requires transformer.encoder.layer blocks")
        if fusion_mode not in {"replace", "gated_residual"}:
            raise ValueError("TCP fusion mode must be replace or gated_residual")
        if fusion_mode == "replace" and float(fusion_weight) != 1.0:
            raise ValueError("Faithful replacement TCP requires fusion weight 1.0")

        layers = transformer.encoder.layer
        if not 0 <= int(insert_layer) < len(layers):
            raise ValueError(
                "TCP insert layer {} is outside BERT depth {}".format(
                    insert_layer, len(layers)
                )
            )
        embedding_layer = transformer.embeddings.word_embeddings
        hidden_dim = getattr(embedding_layer, "embedding_dim", None)
        if hidden_dim is None:
            hidden_dim = getattr(transformer.config, "hidden_size", None)
        if hidden_dim is None:
            raise TypeError("Cannot infer the BERT hidden dimension")

        class_prior = F.normalize(class_prior.detach().float(), dim=-1)
        if class_prior.dim() != 2 or class_prior.shape[0] != len(classnames):
            raise ValueError(
                "Expected one class prior per class, got {} for {} classes".format(
                    tuple(class_prior.shape), len(classnames)
                )
            )
        category_text = json.dumps(list(classnames), ensure_ascii=False, separators=(",", ":"))
        prior_bytes = class_prior.cpu().contiguous().numpy().tobytes()
        fingerprint_data = {
            "category_order_fingerprint": _text_fingerprint(category_text),
            "template_fingerprint": _text_fingerprint(template),
            "model_fingerprint": _text_fingerprint(model_id),
            "prior_fingerprint": _sha256_bytes(prior_bytes),
        }

        self.base_text_encoder = base_text_encoder
        self.register_buffer("class_prior", class_prior, persistent=False)
        self.tcp_prompt = TCPPromptParameters(
            prior_dim=class_prior.shape[1],
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            bottleneck_dim=bottleneck_dim,
            metadata=fingerprint_data,
            fusion_mode=fusion_mode,
            fusion_weight=fusion_weight,
        )
        self.classnames = tuple(classnames)
        self.template = str(template)
        self.model_id = str(model_id)
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.prior_dim = int(class_prior.shape[1])
        self.bottleneck_dim = int(bottleneck_dim)
        self.insert_layer = int(insert_layer)
        self.fusion_mode = fusion_mode
        self.fusion_weight = float(fusion_weight)

    @property
    def transformer(self):
        return self.base_text_encoder.transformer

    def metadata(self):
        metadata = self.tcp_prompt.checkpoint_metadata()
        metadata.update(
            {
                "classnames": list(self.classnames),
                "template": self.template,
                "model_id": self.model_id,
                "bottleneck_dim": self.bottleneck_dim,
                "insert_layer_zero_based": self.insert_layer,
                "fusion_mode": self.fusion_mode,
                "fusion_weight": self.fusion_weight,
            }
        )
        signature_payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        metadata["signature"] = _text_fingerprint(signature_payload)
        return metadata

    def _attention_mask(self, tokenized_prompts, prompts):
        if tokenized_prompts is None:
            raise ValueError("tokenized_prompts is required for TCP")
        tokenized_prompts = tokenized_prompts.to(prompts.device)
        if tokenized_prompts.shape[:2] != prompts.shape[:2]:
            raise ValueError(
                "Token IDs and prompt embeddings disagree: {} vs {}".format(
                    tuple(tokenized_prompts.shape), tuple(prompts.shape)
                )
            )
        pad_token_id = getattr(self.transformer.config, "pad_token_id", 0)
        return tokenized_prompts.ne(pad_token_id).to(dtype=torch.long)

    def _make_extended_attention_mask(self, attention_mask, input_shape):
        try:
            return self.transformer.get_extended_attention_mask(
                attention_mask, input_shape, device=attention_mask.device
            )
        except TypeError:
            return self.transformer.get_extended_attention_mask(
                attention_mask, input_shape
            )

    def replace_context(self, hidden_states, class_prompt):
        if hidden_states.shape[0] != class_prompt.shape[0]:
            raise ValueError("TCP prompt batch must match the number of classes")
        if hidden_states.shape[1] < 1 + self.num_tokens:
            raise ValueError("The text sequence has too few CoOp context positions")
        if class_prompt.shape[1:] != (self.num_tokens, self.hidden_dim):
            raise ValueError("Unexpected TKE output shape: {}".format(tuple(class_prompt.shape)))
        original_context = hidden_states[:, 1 : 1 + self.num_tokens, :]
        tcp_context = class_prompt.to(
            dtype=hidden_states.dtype, device=hidden_states.device
        )
        if self.fusion_mode == "replace":
            fused_context = tcp_context
        elif self.fusion_mode == "gated_residual":
            alpha = self.tcp_prompt.fusion_alpha().to(dtype=hidden_states.dtype)
            fused_context = original_context + alpha * (tcp_context - original_context)
        else:  # pragma: no cover - validated during construction
            raise KeyError(self.fusion_mode)
        return torch.cat(
            (
                hidden_states[:, :1, :],
                fused_context,
                hidden_states[:, 1 + self.num_tokens :, :],
            ),
            dim=1,
        )

    def forward(self, prompts, tokenized_prompts):
        attention_mask = self._attention_mask(tokenized_prompts, prompts)
        batch_size, sequence_length, _ = prompts.shape
        position_ids = torch.arange(
            sequence_length, dtype=torch.long, device=prompts.device
        ).unsqueeze(0).expand(batch_size, -1)
        token_type_ids = torch.zeros_like(attention_mask)
        hidden_states = self.transformer.embeddings(
            inputs_embeds=prompts,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        extended_attention_mask = self._make_extended_attention_mask(
            attention_mask, hidden_states.shape[:2]
        )
        layers = self.transformer.encoder.layer
        head_mask = self.transformer.get_head_mask(None, len(layers))
        class_prompt = self.tcp_prompt(self.class_prior)
        for layer_idx, layer in enumerate(layers):
            if layer_idx == self.insert_layer:
                hidden_states = self.replace_context(hidden_states, class_prompt)
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_attention_mask,
                head_mask=head_mask[layer_idx],
            )
            hidden_states = layer_outputs[0]

        transformer_pooler = getattr(self.transformer, "pooler", None)
        pooled_output = transformer_pooler(hidden_states) if transformer_pooler else None
        output = BaseModelOutputWithPooling(
            last_hidden_state=hidden_states,
            pooler_output=pooled_output,
        )
        pooled = self.base_text_encoder.pooler(output, attention_mask)
        projected = self.base_text_encoder.proj(pooled)
        if getattr(self.base_text_encoder, "output_tokens", False):
            return projected, hidden_states[:, 1:, :]
        return projected


def validate_tcp_checkpoint_state(state_dict, tcp_prompt, prefix="tcp."):
    """Reject missing or incompatible TCP prompt bundles before loading weights."""

    expected = tcp_prompt.checkpoint_metadata()
    fields = getattr(tcp_prompt, "checkpoint_field_map", None)
    if fields is None:
        fields = {
            "num_tokens": "_meta_num_tokens",
            "hidden_dim": "_meta_hidden_dim",
            "prior_dim": "_meta_prior_dim",
            "category_order_fingerprint": "_meta_category_order_fingerprint",
            "template_fingerprint": "_meta_template_fingerprint",
            "model_fingerprint": "_meta_model_fingerprint",
            "prior_fingerprint": "_meta_prior_fingerprint",
        }
    for field, suffix in fields.items():
        key = prefix + suffix
        if key not in state_dict:
            raise RuntimeError(
                "Checkpoint is not a complete TCP prompt bundle: missing {}".format(key)
            )
        tensor = state_dict[key]
        actual = (
            int(tensor.item())
            if tensor.numel() == 1 and not isinstance(expected[field], str)
            else _decode_hash(tensor)
        )
        if actual != expected[field]:
            raise RuntimeError(
                "TCP checkpoint {} mismatch: expected {!r}, got {!r}".format(
                    field, expected[field], actual
                )
            )
