"""Original-style Biomedical TCP: mean-50 prototype, shared TKE, one injection."""

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F
from transformers.modeling_outputs import BaseModelOutputWithPooling

from models.biomedclip_loader import BIOMEDCLIP_MODEL_ID
from models.text_vpt import TextPromptParameters


DESCRIPTION_COUNT = 50
NUM_TOKENS = 4


class QuickGELU(nn.Module):
    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _text_fingerprint(value):
    return _sha256_bytes(value.encode("utf-8"))


def description_source_fingerprint(classnames, descriptions):
    payload = [
        {"classname": classname, "descriptions": list(descriptions[classname])}
        for classname in classnames
    ]
    return _text_fingerprint(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def _ordered_description_set(classnames, description_map, expected_count):
    normalized_names = [name.replace("_", " ") for name in classnames]
    ordered = {}
    flattened = []
    for classname in normalized_names:
        if classname not in description_map:
            raise KeyError("Missing BiomedCoOp descriptions for {!r}".format(classname))
        values = tuple(str(value).strip() for value in description_map[classname])
        if len(values) != int(expected_count):
            raise ValueError(
                "Expected {} descriptions for {!r}, got {}".format(
                    expected_count, classname, len(values)
                )
            )
        if any(not value for value in values):
            raise ValueError("Empty description found for {!r}".format(classname))
        ordered[classname] = values
        flattened.extend(values)
    return normalized_names, ordered, flattened


def _load_cached_bank(cache_path, metadata, expected_shape, label):
    payload = torch.load(cache_path, map_location="cpu", weights_only=True)
    if payload.get("metadata") != metadata:
        raise RuntimeError("{} cache metadata mismatch: {}".format(label, cache_path))
    bank = payload.get("bank")
    if not torch.is_tensor(bank) or tuple(bank.shape[:2]) != tuple(expected_shape):
        raise RuntimeError("{} cache shape mismatch: {}".format(label, cache_path))
    bank = bank.detach().float().cpu()
    fingerprint = _sha256_bytes(bank.contiguous().numpy().tobytes())
    if payload.get("bank_fingerprint") != fingerprint:
        raise RuntimeError("{} cache fingerprint mismatch: {}".format(label, cache_path))
    if not torch.isfinite(bank).all():
        raise RuntimeError("{} cache contains non-finite values: {}".format(label, cache_path))
    norms = bank.norm(dim=-1)
    if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
        raise RuntimeError("{} cache is not normalized: {}".format(label, cache_path))
    return bank


def _save_cached_bank(cache_path, metadata, bank):
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_name("{}.tmp.{}".format(cache_path.name, os.getpid()))
    torch.save(
        {
            "metadata": metadata,
            "bank": bank,
            "bank_fingerprint": _sha256_bytes(bank.contiguous().numpy().tobytes()),
        },
        temporary,
    )
    os.replace(temporary, cache_path)


def build_frozen_description_bank(
    model,
    tokenizer,
    classnames,
    description_map,
    expected_count=DESCRIPTION_COUNT,
    batch_size=32,
    cache_path=None,
    model_id=BIOMEDCLIP_MODEL_ID,
):
    """Encode each fixed description independently in projected BiomedCLIP space."""

    normalized_names, ordered, flattened = _ordered_description_set(
        classnames, description_map, expected_count
    )
    metadata = {
        "schema_version": 1,
        "model_id": str(model_id),
        "classnames": list(normalized_names),
        "description_count": int(expected_count),
        "description_fingerprint": description_source_fingerprint(
            normalized_names, ordered
        ),
    }
    cache_path = Path(cache_path) if cache_path else None
    if cache_path is not None and cache_path.exists():
        return (
            _load_cached_bank(
                cache_path,
                metadata,
                (len(normalized_names), int(expected_count)),
                "Description-bank",
            ),
            ordered,
        )

    tokenized = torch.cat([tokenizer(text) for text in flattened])
    device = next(model.parameters()).device
    features = []
    with torch.no_grad():
        for start in range(0, tokenized.shape[0], int(batch_size)):
            batch = tokenized[start : start + int(batch_size)].to(device)
            features.append(model.encode_text(batch, normalize=True).float().cpu())
    bank = F.normalize(
        torch.cat(features).reshape(
            len(normalized_names), int(expected_count), -1
        ),
        dim=-1,
    )
    if cache_path is not None:
        _save_cached_bank(cache_path, metadata, bank)
    return bank, ordered


def validate_tcp_checkpoint_state(state_dict, tcp_prompt, prefix="tcp."):
    """Validate the shared-TKE TCP architecture metadata in a checkpoint."""
    expected = tcp_prompt.checkpoint_metadata()
    for field, suffix in tcp_prompt.checkpoint_field_map.items():
        key = prefix + suffix
        if key not in state_dict:
            raise RuntimeError(
                "Checkpoint is not a complete TCP prompt bundle: missing {}".format(key)
            )
        tensor = state_dict[key]
        if not torch.is_tensor(tensor) or tensor.numel() != 1:
            raise RuntimeError("TCP checkpoint {} has an invalid metadata value".format(field))
        actual = int(tensor.item())
        if actual != expected[field]:
            raise RuntimeError(
                "TCP checkpoint {} mismatch: expected {!r}, got {!r}".format(
                    field, expected[field], actual
                )
            )


class OriginalStyleTCPPromptParameters(nn.Module):
    """Shared TKE and the pre-injection ordinary Text Deep Prompt parameters."""

    _META_FIELDS = ("mode", "prior_dim", "hidden_dim", "depth", "insert_layer", "num_tokens")

    def __init__(self, prior_dim, hidden_dim, depth, insert_layer=8, enabled=True,
                 fusion=False, fusion_alpha=0.5):
        super().__init__()
        prior_dim = int(prior_dim)
        hidden_dim = int(hidden_dim)
        depth = int(depth)
        insert_layer = int(insert_layer)
        if prior_dim < 4 or prior_dim % 4:
            raise ValueError("Original-style TKE requires D_proj divisible by four")
        if not 1 <= insert_layer < depth:
            raise ValueError(
                "Original-style INSERT_LAYER must be a middle block (1 <= layer < depth)"
            )
        self.prior_dim = prior_dim
        self.insert_layer = insert_layer
        self.enabled = bool(enabled)
        self.num_tokens = NUM_TOKENS
        self.hidden_dim = hidden_dim
        self.depth = depth
        self.text_prompt = TextPromptParameters(
            hidden_dim,
            insert_layer if self.enabled else depth,
            self.num_tokens,
        )
        self.down_projection = nn.Linear(prior_dim, prior_dim // 4)
        self.activation = QuickGELU()
        self.up_projection = nn.Linear(prior_dim // 4, self.num_tokens * hidden_dim)
        self.fusion = bool(fusion) and self.enabled
        self.fusion_alpha = float(fusion_alpha)
        if self.fusion:
            # Keep existing parameter initialization and the training RNG unchanged.
            with torch.random.fork_rng(devices=[]):
                self.fusion_prompt = nn.Parameter(torch.empty(self.num_tokens, hidden_dim))
                nn.init.normal_(self.fusion_prompt, std=0.02)
        for name, value in {
            "mode": 1,
            "prior_dim": prior_dim,
            "hidden_dim": hidden_dim,
            "depth": depth,
            "insert_layer": insert_layer,
            "num_tokens": self.num_tokens,
        }.items():
            self.register_buffer("_meta_" + name, torch.tensor(value))

    @property
    def checkpoint_field_map(self):
        return {
            name: "_meta_" + name
            for name in self._META_FIELDS
        }

    def checkpoint_metadata(self):
        return {
            name: int(getattr(self, key).item())
            for name, key in self.checkpoint_field_map.items()
        }

    def class_tokens(self, prototype):
        reference = self.down_projection.weight
        prototype = prototype.detach().to(
            device=reference.device, dtype=reference.dtype
        )
        return self.up_projection(
            self.activation(self.down_projection(prototype))
        ).reshape(prototype.shape[0], self.num_tokens, self.hidden_dim)

    def prompt_for_layer(self, layer_idx, class_tokens, dtype, device):
        if self.enabled and int(layer_idx) == self.insert_layer:
            if self.fusion:
                class_tokens = (
                    self.fusion_alpha * class_tokens
                    + (1 - self.fusion_alpha) * self.fusion_prompt
                )
            return class_tokens.to(device=device, dtype=dtype)
        return self.text_prompt.for_layer(
            layer_idx, class_tokens.shape[0], dtype, device
        )


class OriginalStyleTCPBertTextEncoder(nn.Module):
    """Frozen BERT tower with one TCP replacement or fusion at INSERT_LAYER."""

    def __init__(
        self,
        base_text_encoder,
        projected_description_bank,
        classnames,
        insert_layer=8,
        enabled=True,
        fusion=False,
        fusion_alpha=0.5,
    ):
        super().__init__()
        transformer = base_text_encoder.transformer
        if getattr(getattr(transformer, "config", None), "model_type", None) != "bert":
            raise TypeError("Original-style TCP requires a BERT tower")
        bank = projected_description_bank.detach().float()
        if bank.ndim != 3 or bank.shape[0] != len(classnames):
            raise ValueError("Expected projected description bank [C, 50, D_proj]")
        if bank.shape[1] != DESCRIPTION_COUNT:
            raise ValueError(
                "Expected {} descriptions per class, got {}".format(
                    DESCRIPTION_COUNT, bank.shape[1]
                )
            )
        if not torch.isfinite(bank).all():
            raise ValueError("Projected description bank contains non-finite values")
        bank = F.normalize(bank, dim=-1)
        self.base_text_encoder = base_text_encoder
        self.register_buffer("description_bank", bank, persistent=False)
        self.register_buffer(
            "class_prior", F.normalize(bank.mean(dim=1), dim=-1), persistent=False
        )
        self.classnames = tuple(name.replace("_", " ") for name in classnames)
        self.num_tokens = NUM_TOKENS
        self.prior_dim = int(bank.shape[-1])
        self.hidden_dim = int(transformer.config.hidden_size)
        self.depth = len(transformer.encoder.layer)
        self.insert_layer = int(insert_layer)
        self.description_count = DESCRIPTION_COUNT
        self.tcp_prompt = OriginalStyleTCPPromptParameters(
            self.prior_dim,
            self.hidden_dim,
            self.depth,
            self.insert_layer,
            enabled,
            fusion,
            fusion_alpha,
        )

    @property
    def transformer(self):
        return self.base_text_encoder.transformer

    def aggregate_class_tokens(self):
        """Return [C, 4, hidden_dim] tokens from each class's mean-50 prototype."""
        return self.tcp_prompt.class_tokens(self.class_prior)

    def metadata(self):
        metadata = self.tcp_prompt.checkpoint_metadata()
        metadata.update(
            {
                "classnames": list(self.classnames),
                "description_source": "BIOMEDCOOP_TEMPLATES",
                "description_count": self.description_count,
                "aggregation": "mean50_class_prototype",
                "connection": "weighted_deep_prompt_fusion" if self.tcp_prompt.fusion else "single_layer_replacement",
                "fusion_alpha": self.tcp_prompt.fusion_alpha if self.tcp_prompt.fusion else None,
                "insert_layer_zero_based": self.insert_layer,
            }
        )
        return metadata

    def _attention_mask(self, tokenized_prompts, prompts):
        if tokenized_prompts is None:
            raise ValueError("tokenized_prompts is required for Original-style TCP")
        tokenized_prompts = tokenized_prompts.to(device=prompts.device)
        if tokenized_prompts.dim() != 2 or prompts.dim() != 3:
            raise ValueError("Expected token IDs [B,L] and prompt embeddings [B,L,D]")
        if tokenized_prompts.shape[:2] != prompts.shape[:2]:
            raise ValueError("Token IDs and prompt embeddings disagree")
        pad_token_id = getattr(self.transformer.config, "pad_token_id", 0)
        return tokenized_prompts.ne(pad_token_id).to(dtype=torch.long)

    def _reserve_prompt_slots(self, prompts, attention_mask):
        maximum = int(self.transformer.config.max_position_embeddings)
        available_length = maximum - self.num_tokens
        if available_length < 1:
            raise ValueError("Not enough BERT positions for TCP prompt slots")
        if prompts.shape[1] > maximum:
            raise ValueError("CoOp text sequence exceeds BERT max_position_embeddings")
        keep_length = min(prompts.shape[1], available_length)
        if attention_mask[:, keep_length:].any():
            raise ValueError("Adding TCP prompt slots would truncate a valid token")
        if not attention_mask[:, 0].all():
            raise ValueError("The first text token must be the active CLS/SOS token")
        return prompts[:, :keep_length, :], attention_mask[:, :keep_length]

    def _extended_mask(self, attention_mask, input_shape):
        try:
            return self.transformer.get_extended_attention_mask(
                attention_mask, input_shape, device=attention_mask.device
            )
        except TypeError:
            return self.transformer.get_extended_attention_mask(
                attention_mask, input_shape
            )

    def _replace_slots(self, hidden_states, values):
        end = 1 + self.num_tokens
        if hidden_states.shape[1] < end:
            raise ValueError("Text sequence has too few prompt slots")
        return torch.cat((hidden_states[:, :1], values, hidden_states[:, end:]), dim=1)

    def forward(self, prompts, tokenized_prompts, class_tokens=None):
        reference = self.tcp_prompt.down_projection.weight
        prompts = prompts.to(dtype=reference.dtype)
        attention_mask = self._attention_mask(tokenized_prompts, prompts)
        prompts, attention_mask = self._reserve_prompt_slots(prompts, attention_mask)
        if class_tokens is None:
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
            # Blocks before INSERT_LAYER keep ordinary deep prompts.  At the
            # insertion block the four slots are replaced once by TKE output;
            # later blocks consume those hidden states without replacement.
            if layer_idx > 0 and (
                not self.tcp_prompt.enabled or layer_idx <= self.insert_layer
            ):
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
