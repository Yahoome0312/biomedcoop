"""Final multi-description TCP adapter retained after the TCP ablation study.

The production path is fixed to the reported LayerBasis + XProto model:
50 BiomedCoOp descriptions are encoded at the input of BERT block 8, grouped
into five blocks of ten, and converted into four layer-aligned class tokens.
Those tokens enter the internal four-token Deep Text Prompt through a centered,
norm-matched residual before blocks 8--11.  There is one model and one logit
tensor at training and inference time.
"""

import hashlib
import json
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.biomedclip_loader import BIOMEDCLIP_MODEL_ID
from models.text_vpt import TextPromptParameters

try:
    from transformers.modeling_outputs import BaseModelOutputWithPooling
except ImportError as exc:  # pragma: no cover - project dependency
    raise RuntimeError("Multi-text TCP requires transformers") from exc


FINAL_AGGREGATION = "grouped10_layer_residual"
FINAL_CONNECTION = "late_centered_norm_residual"


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _text_fingerprint(value):
    return _sha256_bytes(value.encode("utf-8"))


def _encode_hash(value):
    return torch.tensor(list(value.encode("ascii")), dtype=torch.uint8)


def _decode_hash(value):
    return bytes(value.detach().cpu().tolist()).decode("ascii")


class QuickGELU(nn.Module):
    def forward(self, value):
        return value * torch.sigmoid(1.702 * value)


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


def _load_cached_bank(cache_path, metadata, expected_shape, label, normalized=False):
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
    if normalized:
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
    expected_count=50,
    batch_size=32,
    cache_path=None,
    model_id=BIOMEDCLIP_MODEL_ID,
):
    """Encode the fixed descriptions in BiomedCLIP's projected text space."""

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
                normalized=True,
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
    bank = F.normalize(torch.cat(features).reshape(
        len(normalized_names), int(expected_count), -1
    ), dim=-1)
    if cache_path is not None:
        _save_cached_bank(cache_path, metadata, bank)
    return bank, ordered


def build_frozen_layer_description_bank(
    model,
    tokenizer,
    classnames,
    description_map,
    insert_layer=8,
    expected_count=50,
    batch_size=32,
    cache_path=None,
    model_id=BIOMEDCLIP_MODEL_ID,
):
    """Encode descriptions in the frozen BERT space before block 8."""

    normalized_names, ordered, flattened = _ordered_description_set(
        classnames, description_map, expected_count
    )
    transformer = model.text.transformer
    depth = len(transformer.encoder.layer)
    if not 0 <= int(insert_layer) < depth:
        raise ValueError("insert_layer is outside the frozen BERT depth")
    metadata = {
        "schema_version": 2,
        "representation": "bert_cls_before_block",
        "insert_layer_zero_based": int(insert_layer),
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
                "Layer-description",
                normalized=True,
            ),
            ordered,
        )

    tokenized = torch.cat([tokenizer(text) for text in flattened])
    device = next(model.parameters()).device
    pad_token_id = getattr(transformer.config, "pad_token_id", 0)
    features = []
    with torch.no_grad():
        for start in range(0, tokenized.shape[0], int(batch_size)):
            input_ids = tokenized[start : start + int(batch_size)].to(device)
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=input_ids.ne(pad_token_id).long(),
                output_hidden_states=True,
                return_dict=True,
            )
            features.append(outputs.hidden_states[int(insert_layer)][:, 0].float().cpu())
    bank = F.normalize(torch.cat(features), dim=-1).reshape(
        len(normalized_names), int(expected_count), -1
    )
    if cache_path is not None:
        _save_cached_bank(cache_path, metadata, bank)
    return bank, ordered


class MultiTextTCPPromptParameters(nn.Module):
    """The retained LayerBasis TKE and centered Deep Text Prompt residual."""

    _HASH_FIELDS = (
        "category_order_fingerprint",
        "template_fingerprint",
        "model_fingerprint",
        "prior_fingerprint",
        "description_fingerprint",
        "aggregation_fingerprint",
        "connection_fingerprint",
    )

    def __init__(
        self,
        hidden_dim,
        num_tokens,
        bottleneck_dim,
        depth,
        description_count,
        insert_layer,
        gate_init,
        metadata,
    ):
        super().__init__()
        if int(description_count) != 50:
            raise ValueError("Final LayerBasis TCP requires exactly 50 descriptions")
        if int(num_tokens) != 4:
            raise ValueError("Final LayerBasis TCP requires exactly four text tokens")
        if not 0 <= int(insert_layer) < int(depth):
            raise ValueError("insert_layer is outside the text-transformer depth")
        if not 0.0 < float(gate_init) < 1.0:
            raise ValueError("Residual gate initialization must satisfy 0 < gate_init < 1")

        self.prior_dim = int(hidden_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.bottleneck_dim = int(bottleneck_dim)
        self.depth = int(depth)
        self.description_count = int(description_count)
        self.insert_layer = int(insert_layer)
        self.residual_scale = 1.0

        self.text_prompt = TextPromptParameters(
            embed_dim=self.hidden_dim,
            depth=self.depth,
            num_tokens=self.num_tokens,
            dropout=0.0,
            init="normal",
        )
        self.down_projection = nn.Linear(self.hidden_dim, self.bottleneck_dim)
        self.activation = QuickGELU()
        self.up_projection = nn.Linear(
            self.bottleneck_dim, self.num_tokens * self.hidden_dim
        )
        nn.init.zeros_(self.up_projection.weight)
        nn.init.zeros_(self.up_projection.bias)
        self.layer_residual_logit = nn.Parameter(torch.logit(torch.tensor(0.1)))
        initial_logit = torch.logit(torch.tensor(float(gate_init)))
        self.gate_logits = nn.Parameter(
            initial_logit.repeat(self.depth - self.insert_layer)
        )

        self.register_buffer("_meta_num_tokens", torch.tensor(self.num_tokens))
        self.register_buffer("_meta_hidden_dim", torch.tensor(self.hidden_dim))
        self.register_buffer("_meta_prior_dim", torch.tensor(self.prior_dim))
        self.register_buffer("_meta_depth", torch.tensor(self.depth))
        self.register_buffer(
            "_meta_description_count", torch.tensor(self.description_count)
        )
        for field in self._HASH_FIELDS:
            self.register_buffer("_meta_{}".format(field), _encode_hash(metadata[field]))

    @property
    def checkpoint_field_map(self):
        fields = {
            "num_tokens": "_meta_num_tokens",
            "hidden_dim": "_meta_hidden_dim",
            "prior_dim": "_meta_prior_dim",
            "depth": "_meta_depth",
            "description_count": "_meta_description_count",
        }
        fields.update({field: "_meta_{}".format(field) for field in self._HASH_FIELDS})
        return fields

    def checkpoint_metadata(self):
        result = {
            "num_tokens": int(self._meta_num_tokens.item()),
            "hidden_dim": int(self._meta_hidden_dim.item()),
            "prior_dim": int(self._meta_prior_dim.item()),
            "depth": int(self._meta_depth.item()),
            "description_count": int(self._meta_description_count.item()),
        }
        for field in self._HASH_FIELDS:
            result[field] = _decode_hash(getattr(self, "_meta_{}".format(field)))
        return result

    def aggregate_descriptions(self, description_bank):
        if tuple(description_bank.shape[1:]) != (
            self.description_count,
            self.hidden_dim,
        ):
            raise ValueError("Unexpected layer-description bank shape")
        reference = self.down_projection.weight
        bank = description_bank.to(dtype=reference.dtype, device=reference.device)
        groups = bank.reshape(bank.shape[0], 5, 10, self.hidden_dim).mean(dim=2)
        layer_basis = F.normalize(
            groups[:, : self.num_tokens] + groups.mean(dim=1, keepdim=True), dim=-1
        )
        prototype = F.normalize(bank.mean(dim=1), dim=-1)
        learned_delta = self.up_projection(
            self.activation(self.down_projection(prototype))
        ).reshape(bank.shape[0], self.num_tokens, self.hidden_dim)
        return layer_basis + self.layer_residual_logit.sigmoid() * learned_delta

    def prompt_for_layer(self, layer_idx, class_tokens, dtype, device):
        shared = self.text_prompt.for_layer(
            layer_idx, class_tokens.shape[0], dtype, device
        )
        if int(layer_idx) < self.insert_layer:
            return shared
        centered = class_tokens.float() - class_tokens.float().mean(dim=0, keepdim=True)
        reference_norm = shared.detach().float().norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        matched = (F.normalize(centered, dim=-1) * reference_norm).to(dtype=dtype)
        alpha = self.gate_logits[int(layer_idx) - self.insert_layer].sigmoid()
        return shared + (alpha * self.residual_scale).to(dtype=dtype) * matched

    def set_residual_scale(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("residual_scale must satisfy 0 <= value <= 1")
        self.residual_scale = value


class MultiTextTCPBertTextEncoder(nn.Module):
    """Frozen BERT tower with the final internal TextDeep + LayerBasis adapter."""

    def __init__(
        self,
        base_text_encoder,
        layer_description_bank,
        projected_description_bank,
        descriptions,
        classnames,
        num_tokens=4,
        bottleneck_dim=128,
        insert_layer=8,
        gate_init=0.05,
        model_id=BIOMEDCLIP_MODEL_ID,
    ):
        super().__init__()
        transformer = base_text_encoder.transformer
        if getattr(getattr(transformer, "config", None), "model_type", None) != "bert":
            raise TypeError("Multi-text TCP requires a Hugging Face BERT tower")
        hidden_dim = getattr(
            transformer.embeddings.word_embeddings, "embedding_dim", None
        ) or transformer.config.hidden_size
        if layer_description_bank.dim() != 3:
            raise ValueError("Layer-description bank must be rank 3")
        if layer_description_bank.shape[0] != len(classnames):
            raise ValueError("Expected one layer-description bank per class")
        if layer_description_bank.shape[-1] != hidden_dim:
            raise ValueError("Layer-description vectors must match BERT hidden size")
        if projected_description_bank.shape[:2] != layer_description_bank.shape[:2]:
            raise ValueError("Projected and layer description banks disagree")

        normalized_names = tuple(name.replace("_", " ") for name in classnames)
        layer_bank = F.normalize(layer_description_bank.detach().float(), dim=-1)
        raw_projected_bank = projected_description_bank.detach().float()
        # Preserve the exact operation order used by the reported checkpoints:
        # the class mean is formed before the per-description bank is normalized,
        # then normalized once by the trainer and once by the adapter.
        class_prior = F.normalize(raw_projected_bank.mean(dim=1), dim=-1)
        class_prior = F.normalize(class_prior, dim=-1)
        projected_bank = F.normalize(raw_projected_bank, dim=-1)
        metadata = {
            "category_order_fingerprint": _text_fingerprint(
                json.dumps(list(normalized_names), ensure_ascii=False, separators=(",", ":"))
            ),
            "template_fingerprint": _text_fingerprint("BIOMEDCOOP_TEMPLATES:50"),
            "model_fingerprint": _text_fingerprint(model_id),
            "prior_fingerprint": _sha256_bytes(
                class_prior.cpu().contiguous().numpy().tobytes()
            ),
            "description_fingerprint": description_source_fingerprint(
                normalized_names, descriptions
            ),
            "aggregation_fingerprint": _text_fingerprint(FINAL_AGGREGATION),
            "connection_fingerprint": _text_fingerprint(
                "{}|prior=layer_cls|layer={}".format(FINAL_CONNECTION, insert_layer)
            ),
        }

        self.base_text_encoder = base_text_encoder
        self.register_buffer("description_bank", layer_bank, persistent=False)
        self.register_buffer("projected_description_bank", projected_bank, persistent=False)
        self.register_buffer("class_prior", class_prior, persistent=False)
        self.tcp_prompt = MultiTextTCPPromptParameters(
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            bottleneck_dim=bottleneck_dim,
            depth=len(transformer.encoder.layer),
            description_count=layer_bank.shape[1],
            insert_layer=insert_layer,
            gate_init=gate_init,
            metadata=metadata,
        )
        self.classnames = normalized_names
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.prior_dim = int(hidden_dim)
        self.bottleneck_dim = int(bottleneck_dim)
        self.insert_layer = int(insert_layer)
        self.depth = len(transformer.encoder.layer)
        self.description_count = int(layer_bank.shape[1])
        self.aggregation = FINAL_AGGREGATION
        self.connection = FINAL_CONNECTION
        self.gate_init = float(gate_init)
        self.model_id = str(model_id)

    @property
    def transformer(self):
        return self.base_text_encoder.transformer

    def metadata(self):
        metadata = self.tcp_prompt.checkpoint_metadata()
        metadata.update(
            {
                "classnames": list(self.classnames),
                "description_source": "BIOMEDCOOP_TEMPLATES",
                "aggregation": self.aggregation,
                "connection": self.connection,
                "bottleneck_dim": self.bottleneck_dim,
                "insert_layer_zero_based": self.insert_layer,
                "gate_init": self.gate_init,
                "model_id": self.model_id,
                "prior_representation": "layer_cls",
            }
        )
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        metadata["signature"] = _text_fingerprint(payload)
        return metadata

    def aggregate_class_tokens(self):
        return self.tcp_prompt.aggregate_descriptions(self.description_bank)

    def _attention_mask(self, tokenized_prompts, prompts):
        tokenized_prompts = tokenized_prompts.to(prompts.device)
        if tokenized_prompts.shape[:2] != prompts.shape[:2]:
            raise ValueError("Token IDs and prompt embeddings disagree")
        pad_token_id = getattr(self.transformer.config, "pad_token_id", 0)
        return tokenized_prompts.ne(pad_token_id).long()

    def _reserve_prompt_slots(self, prompts, attention_mask):
        maximum = int(self.transformer.config.max_position_embeddings)
        keep_length = min(prompts.shape[1], maximum - self.num_tokens)
        if attention_mask[:, keep_length:].any():
            raise ValueError("Adding text prompt slots would truncate a valid token")
        return prompts[:, :keep_length], attention_mask[:, :keep_length]

    def _extended_mask(self, attention_mask, input_shape):
        return self.transformer.get_extended_attention_mask(
            attention_mask, input_shape, device=attention_mask.device
        )

    def _replace_slots(self, hidden_states, values):
        end = 1 + self.num_tokens
        if hidden_states.shape[1] < end:
            raise ValueError("Text sequence has too few prompt slots")
        return torch.cat((hidden_states[:, :1], values, hidden_states[:, end:]), dim=1)

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
            if layer_idx > 0:
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


def validate_tcp_checkpoint_state(state_dict, tcp_prompt, prefix="tcp."):
    """Reject incomplete or incompatible final TCP prompt bundles."""

    expected = tcp_prompt.checkpoint_metadata()
    for field, suffix in tcp_prompt.checkpoint_field_map.items():
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
