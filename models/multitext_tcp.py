"""Multi-description TCP with an internal Deep Text Prompt connection.

This module keeps BiomedCLIP frozen and turns the fixed BiomedCoOp
description set into class-specific text-prompt residuals.  It is deliberately
single-branch: no second model or output-logit ensemble is involved.
"""

import hashlib
import json
import math
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from models.biomedclip_loader import BIOMEDCLIP_MODEL_ID
from models.text_vpt import TextPromptParameters
from models.tcp import QuickGELU, _decode_hash, _encode_hash, _text_fingerprint

try:
    from transformers.modeling_outputs import BaseModelOutputWithPooling
except ImportError as exc:  # pragma: no cover
    raise RuntimeError("Multi-text TCP requires transformers") from exc


AGGREGATION_MODES = (
    "feature_mean",
    "tke_mean",
    "consensus_weighted",
    "set_attention",
    "cosine_set_attention",
    "grouped10_cosine_attention",
    "grouped10_layer_residual",
    "grouped10_layer_projected_hybrid",
    "grouped10_layer_projected_residual",
    "layer_cosine_set_hybrid",
    "layer_cosine_set_hybrid_light",
    "layer_cosine_set_residual",
)
CONNECTION_MODES = (
    "late_residual",
    "late_norm_residual",
    "late_centered_norm_residual",
    "late_centered_classlayer_norm_residual",
    "late_replace",
    "all_residual",
    "original_coop_replace",
    "inplace_once_norm_residual",
    "inplace_once_centered_norm_residual",
    "inplace_once_centered_classgate_norm_residual",
    "inplace_deep_centered_norm_residual",
    "inplace_deep_ramped_centered_norm_residual",
    "inplace_deep_balanced_ramp_centered_norm_residual",
    "inplace_deep_terminal_boost_centered_norm_residual",
    "inplace_deep_terminal_peak_centered_norm_residual",
)


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def description_source_fingerprint(classnames, descriptions):
    payload = [
        {"classname": classname, "descriptions": list(descriptions[classname])}
        for classname in classnames
    ]
    return _text_fingerprint(
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    )


def build_frozen_description_bank(
    model,
    tokenizer,
    classnames,
    description_map,
    expected_count=50,
    batch_size=64,
    cache_path=None,
    model_id=BIOMEDCLIP_MODEL_ID,
):
    """Encode exactly one fixed description set per class with vanilla BiomedCLIP."""

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
        if len(set(values)) != len(values):
            raise ValueError("Duplicate description found for {!r}".format(classname))
        ordered[classname] = values
        flattened.extend(values)

    cache_metadata = {
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
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover - compatibility with older torch
            payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") != cache_metadata:
            raise RuntimeError(
                "Description-bank cache metadata mismatch: {}".format(cache_path)
            )
        bank = payload.get("bank")
        expected_shape = (len(normalized_names), int(expected_count))
        if not torch.is_tensor(bank) or tuple(bank.shape[:2]) != expected_shape:
            raise RuntimeError(
                "Description-bank cache shape mismatch: {}".format(cache_path)
            )
        bank = bank.detach().float().cpu()
        fingerprint = _sha256_bytes(bank.contiguous().numpy().tobytes())
        if payload.get("bank_fingerprint") != fingerprint:
            raise RuntimeError(
                "Description-bank cache fingerprint mismatch: {}".format(cache_path)
            )
        if not torch.isfinite(bank).all():
            raise RuntimeError(
                "Description-bank cache contains non-finite values: {}".format(
                    cache_path
                )
            )
        norms = bank.norm(dim=-1)
        if not torch.allclose(norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5):
            raise RuntimeError(
                "Description-bank cache is not normalized: {}".format(cache_path)
            )
        return bank, ordered

    tokenized = torch.cat([tokenizer(text) for text in flattened])
    model_device = next(model.parameters()).device
    features = []
    with torch.no_grad():
        for start in range(0, tokenized.shape[0], int(batch_size)):
            batch = tokenized[start : start + int(batch_size)].to(model_device)
            encoded = model.encode_text(batch, normalize=True)
            features.append(encoded.detach().float().cpu())
    bank = torch.cat(features, dim=0).reshape(
        len(normalized_names), int(expected_count), -1
    )
    bank = F.normalize(bank, dim=-1)
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name(
            "{}.tmp.{}".format(cache_path.name, os.getpid())
        )
        torch.save(
            {
                "metadata": cache_metadata,
                "bank": bank,
                "bank_fingerprint": _sha256_bytes(
                    bank.contiguous().numpy().tobytes()
                ),
            },
            temporary,
        )
        os.replace(temporary, cache_path)
    return bank, ordered


def build_frozen_layer_description_bank(
    model,
    tokenizer,
    classnames,
    description_map,
    insert_layer,
    expected_count=50,
    batch_size=64,
    cache_path=None,
    model_id=BIOMEDCLIP_MODEL_ID,
):
    """Encode descriptions in the frozen BERT space used at TCP injection."""

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
        if any(not value for value in values) or len(set(values)) != len(values):
            raise ValueError("Descriptions must be non-empty and unique per class")
        ordered[classname] = values
        flattened.extend(values)

    transformer = model.text.transformer
    depth = len(transformer.encoder.layer)
    if not 0 <= int(insert_layer) < depth:
        raise ValueError("insert_layer is outside the frozen BERT depth")
    cache_metadata = {
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
        try:
            payload = torch.load(cache_path, map_location="cpu", weights_only=True)
        except TypeError:  # pragma: no cover
            payload = torch.load(cache_path, map_location="cpu")
        if payload.get("metadata") != cache_metadata:
            raise RuntimeError(
                "Layer-description cache metadata mismatch: {}".format(cache_path)
            )
        bank = payload.get("bank")
        expected_shape = (len(normalized_names), int(expected_count))
        if not torch.is_tensor(bank) or tuple(bank.shape[:2]) != expected_shape:
            raise RuntimeError(
                "Layer-description cache shape mismatch: {}".format(cache_path)
            )
        bank = bank.detach().float().cpu()
        fingerprint = _sha256_bytes(bank.contiguous().numpy().tobytes())
        if payload.get("bank_fingerprint") != fingerprint:
            raise RuntimeError(
                "Layer-description cache fingerprint mismatch: {}".format(cache_path)
            )
        norms = bank.norm(dim=-1)
        if not torch.isfinite(bank).all() or not torch.allclose(
            norms, torch.ones_like(norms), atol=1e-5, rtol=1e-5
        ):
            raise RuntimeError("Layer-description cache is invalid: {}".format(cache_path))
        return bank, ordered

    tokenized = torch.cat([tokenizer(text) for text in flattened])
    model_device = next(model.parameters()).device
    pad_token_id = getattr(transformer.config, "pad_token_id", 0)
    features = []
    with torch.no_grad():
        for start in range(0, tokenized.shape[0], int(batch_size)):
            input_ids = tokenized[start : start + int(batch_size)].to(model_device)
            attention_mask = input_ids.ne(pad_token_id).long()
            outputs = transformer(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
            )
            # hidden_states[0] is embeddings; index L is the state after the
            # first L blocks, exactly the input space of zero-based block L.
            features.append(outputs.hidden_states[int(insert_layer)][:, 0].float().cpu())
    bank = F.normalize(torch.cat(features, dim=0), dim=-1).reshape(
        len(normalized_names), int(expected_count), -1
    )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = cache_path.with_name("{}.tmp.{}".format(cache_path.name, os.getpid()))
        torch.save(
            {
                "metadata": cache_metadata,
                "bank": bank,
                "bank_fingerprint": _sha256_bytes(
                    bank.contiguous().numpy().tobytes()
                ),
            },
            temporary,
        )
        os.replace(temporary, cache_path)
    return bank, ordered


class MultiTextTCPPromptParameters(nn.Module):
    """Trainable description aggregator, Deep Text Prompt, and internal gates."""

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
        prior_dim,
        hidden_dim,
        num_tokens,
        bottleneck_dim,
        depth,
        description_count,
        aggregation,
        connection,
        insert_layer,
        consensus_temperature,
        gate_init,
        metadata,
        num_classes,
    ):
        super().__init__()
        if aggregation not in AGGREGATION_MODES:
            raise ValueError("Unknown multi-text aggregation: {}".format(aggregation))
        if connection not in CONNECTION_MODES:
            raise ValueError("Unknown TCP/TextPrompt connection: {}".format(connection))
        if not 0 <= int(insert_layer) < int(depth):
            raise ValueError("insert_layer is outside the text-transformer depth")
        if float(consensus_temperature) <= 0.0:
            raise ValueError("consensus_temperature must be positive")
        if (
            aggregation in {
                "grouped10_cosine_attention",
                "grouped10_layer_residual",
                "grouped10_layer_projected_hybrid",
                "grouped10_layer_projected_residual",
                "layer_cosine_set_hybrid",
                "layer_cosine_set_hybrid_light",
                "layer_cosine_set_residual",
            }
            and int(description_count) % 10 != 0
        ):
            raise ValueError(
                "Grouped-10 aggregation requires 10 descriptions per group"
            )
        if aggregation in {
            "grouped10_layer_residual",
            "grouped10_layer_projected_hybrid",
            "grouped10_layer_projected_residual",
            "layer_cosine_set_hybrid",
            "layer_cosine_set_hybrid_light",
            "layer_cosine_set_residual",
        }:
            if int(description_count) != 50:
                raise ValueError(
                    "Grouped layer aggregation requires exactly 50 descriptions"
                )
        if aggregation == "grouped10_layer_residual":
            if int(prior_dim) != int(hidden_dim):
                raise ValueError(
                    "grouped10_layer_residual requires layer-aligned description vectors"
                )
        if connection in {
            "late_residual",
            "late_norm_residual",
            "late_centered_norm_residual",
            "late_centered_classlayer_norm_residual",
            "all_residual",
            "inplace_once_norm_residual",
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        } and not 0.0 < float(gate_init) < 1.0:
            raise ValueError("Residual gate initialization must satisfy 0 < gate_init < 1")

        self.prior_dim = int(prior_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_tokens = int(num_tokens)
        self.bottleneck_dim = int(bottleneck_dim)
        self.depth = int(depth)
        self.description_count = int(description_count)
        self.num_classes = int(num_classes)
        self.aggregation = str(aggregation)
        self.connection = str(connection)
        self.insert_layer = int(insert_layer)
        self.consensus_temperature = float(consensus_temperature)
        self.residual_scale = 1.0

        # Extra Deep Text Prompt slots make the model differ from CoOp even
        # when the class residual is zero. The in-place connection omits those
        # slots so residual_scale=0 is an exact CoOp text path.
        self.text_prompt = None
        if self.connection not in {
            "inplace_once_norm_residual",
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        }:
            self.text_prompt = TextPromptParameters(
                embed_dim=self.hidden_dim,
                depth=self.depth,
                num_tokens=self.num_tokens,
                dropout=0.0,
                init="normal",
            )

        if self.aggregation in {
            "set_attention",
            "cosine_set_attention",
            "grouped10_cosine_attention",
            "layer_cosine_set_hybrid",
            "layer_cosine_set_hybrid_light",
            "layer_cosine_set_residual",
        }:
            self.set_queries = nn.Parameter(
                torch.empty(self.num_tokens, self.bottleneck_dim)
            )
            self.key_projection = nn.Linear(self.prior_dim, self.bottleneck_dim)
            self.value_projection = nn.Linear(self.prior_dim, self.bottleneck_dim)
            self.output_projection = nn.Linear(self.bottleneck_dim, self.hidden_dim)
            nn.init.normal_(self.set_queries, mean=0.0, std=0.02)
            if self.aggregation in {
                "layer_cosine_set_hybrid",
                "layer_cosine_set_hybrid_light",
            }:
                # Start primarily from the frozen PubMedBERT layer basis and
                # let the learned all-description set residual enter gently.
                self.layer_cosine_mix_logit = nn.Parameter(
                    torch.logit(
                        torch.tensor(
                            0.05
                            if self.aggregation == "layer_cosine_set_hybrid_light"
                            else 0.1
                        )
                    )
                )
            elif self.aggregation == "layer_cosine_set_residual":
                # Exact frozen-layer start: the set branch has no effect until
                # its output map receives supervised gradients. This preserves
                # the stable layer-basis trajectory without random token noise.
                nn.init.zeros_(self.output_projection.weight)
                nn.init.zeros_(self.output_projection.bias)
                self.layer_cosine_residual_logit = nn.Parameter(
                    torch.logit(torch.tensor(0.5))
                )
        else:
            self.down_projection = nn.Linear(self.prior_dim, self.bottleneck_dim)
            self.up_projection = nn.Linear(
                self.bottleneck_dim, self.num_tokens * self.hidden_dim
            )
            if self.aggregation == "grouped10_layer_residual":
                # Start from the frozen, layer-aligned semantic basis exactly.
                # The MLP is a trainable residual rather than a random
                # replacement direction, so epoch-one injection is meaningful
                # even before the TKE projection has learned the BERT space.
                nn.init.zeros_(self.up_projection.weight)
                nn.init.zeros_(self.up_projection.bias)
                self.layer_residual_logit = nn.Parameter(
                    torch.logit(torch.tensor(0.1))
                )
            elif self.aggregation == "grouped10_layer_projected_hybrid":
                # The frozen layer basis supplies a stable injection direction;
                # the ordinary projected-text TKE supplies the image-aligned
                # correction. Unit-normalizing the correction makes this gate
                # an interpretable direction mixture rather than a magnitude
                # race between the two representations.
                self.layer_projected_hybrid_logit = nn.Parameter(
                    torch.logit(torch.tensor(0.25))
                )
            elif self.aggregation == "grouped10_layer_projected_residual":
                # Exact frozen-basis start with a stronger learnable residual
                # path. Unlike the normalized hybrid, the projected correction
                # begins at zero and can only enter after receiving training
                # signal from CE and knowledge consistency.
                nn.init.zeros_(self.up_projection.weight)
                nn.init.zeros_(self.up_projection.bias)
                self.layer_projected_residual_logit = nn.Parameter(
                    torch.logit(torch.tensor(0.5))
                )
        self.activation = QuickGELU()

        conditioned_layers = 0
        if self.connection in {
            "late_residual",
            "late_norm_residual",
            "late_centered_norm_residual",
            "late_centered_classlayer_norm_residual",
        }:
            conditioned_layers = self.depth - self.insert_layer
        elif self.connection == "all_residual":
            conditioned_layers = self.depth
        if conditioned_layers:
            initial_logit = torch.logit(torch.tensor(float(gate_init)))
            gate_shape = (
                (conditioned_layers, self.num_classes)
                if self.connection == "late_centered_classlayer_norm_residual"
                else (conditioned_layers,)
            )
            self.gate_logits = nn.Parameter(initial_logit.repeat(*gate_shape))
        elif self.connection in {
            "inplace_once_norm_residual",
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        }:
            gate_count = (
                self.num_classes
                if self.connection
                == "inplace_once_centered_classgate_norm_residual"
                else (
                    self.depth - self.insert_layer
                    if self.connection in {
                        "inplace_deep_centered_norm_residual",
                        "inplace_deep_ramped_centered_norm_residual",
                        "inplace_deep_balanced_ramp_centered_norm_residual",
                        "inplace_deep_terminal_boost_centered_norm_residual",
                        "inplace_deep_terminal_peak_centered_norm_residual",
                    }
                    else 1
                )
            )
            if self.connection == "inplace_deep_ramped_centered_norm_residual":
                # Earlier residuals pass through more frozen BERT blocks and
                # therefore have a larger effective influence. Increase gates
                # toward the output to balance that depth asymmetry. With the
                # standard four conditioned blocks and gate_init=.15 this is
                # exactly [.05, .10, .15, .20].
                denominator = float(max(gate_count - 1, 1))
                values = float(gate_init) * (
                    torch.arange(1, gate_count + 1, dtype=torch.float32)
                    / denominator
                )
                if torch.any(values >= 1.0):
                    raise ValueError("Ramped residual gates must remain below one")
                self.gate_logits = nn.Parameter(torch.logit(values))
            elif self.connection == "inplace_deep_balanced_ramp_centered_norm_residual":
                # Preserve the total strength of the uniform initialization
                # while shifting capacity toward later blocks. For four blocks
                # and gate_init=.15 this is [.075, .125, .175, .225], whose
                # mean remains exactly .15.
                factors = torch.linspace(
                    0.5, 1.5, steps=gate_count, dtype=torch.float32
                )
                values = float(gate_init) * factors
                if torch.any(values >= 1.0):
                    raise ValueError(
                        "Balanced ramp residual gates must remain below one"
                    )
                self.gate_logits = nn.Parameter(torch.logit(values))
            elif self.connection == "inplace_deep_terminal_boost_centered_norm_residual":
                # Keep the early-to-late ramp but reserve extra capacity for
                # the final conditioned block.  The production PubMedBERT
                # path has four conditioned blocks, giving exactly
                # [.05, .10, .15, .25] for gate_init=.15.  Constructing the
                # profile dynamically keeps the adapter valid for compatible
                # BERT towers with a different depth as well.
                denominator = float(max(gate_count - 1, 1))
                factors = torch.arange(
                    1, gate_count + 1, dtype=torch.float32
                ) / denominator
                factors[-1] = 5.0 / 3.0
                values = float(gate_init) * factors
                if torch.any(values >= 1.0):
                    raise ValueError(
                        "Terminal-boost residual gates must remain below one"
                    )
                self.gate_logits = nn.Parameter(torch.logit(values))
            elif self.connection == "inplace_deep_terminal_peak_centered_norm_residual":
                # A separately fingerprinted refinement of terminal-boost:
                # preserve the first three production gates and raise only
                # the last gate from .25 to .30.  Keeping a distinct
                # connection name prevents old .25 checkpoints from being
                # interpreted under the new initialization semantics.
                denominator = float(max(gate_count - 1, 1))
                factors = torch.arange(
                    1, gate_count + 1, dtype=torch.float32
                ) / denominator
                factors[-1] = 2.0
                values = float(gate_init) * factors
                if torch.any(values >= 1.0):
                    raise ValueError(
                        "Terminal-peak residual gates must remain below one"
                    )
                self.gate_logits = nn.Parameter(torch.logit(values))
            else:
                initial_logit = torch.logit(torch.tensor(float(gate_init)))
                self.gate_logits = nn.Parameter(initial_logit.repeat(gate_count))

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

    def _mlp_tke(self, values):
        output = self.up_projection(
            self.activation(self.down_projection(values))
        )
        return output.reshape(*values.shape[:-1], self.num_tokens, self.hidden_dim)

    def _consensus_weights(self, bank):
        normalized = F.normalize(bank, dim=-1)
        similarity = torch.matmul(normalized, normalized.transpose(-1, -2))
        count = similarity.shape[-1]
        score = (similarity.sum(dim=-1) - 1.0) / float(max(count - 1, 1))
        return F.softmax(score / self.consensus_temperature, dim=-1)

    def aggregate_descriptions(self, description_bank, layer_basis=None):
        if description_bank.dim() != 3:
            raise ValueError("description_bank must have shape [classes, descriptions, dim]")
        if description_bank.shape[1:] != (self.description_count, self.prior_dim):
            raise ValueError(
                "Unexpected description bank shape: {}".format(
                    tuple(description_bank.shape)
                )
            )
        reference = next(self.parameters())
        bank = description_bank.to(dtype=reference.dtype, device=reference.device)
        if self.aggregation == "feature_mean":
            prototype = F.normalize(bank.mean(dim=1), dim=-1)
            return self._mlp_tke(prototype)
        if self.aggregation == "grouped10_layer_residual":
            groups = bank.reshape(
                bank.shape[0], 5, 10, self.prior_dim
            ).mean(dim=2)
            shared = groups.mean(dim=1, keepdim=True)
            layer_basis = F.normalize(
                groups[:, : self.num_tokens] + shared, dim=-1
            )
            prototype = F.normalize(bank.mean(dim=1), dim=-1)
            learned_delta = self._mlp_tke(prototype)
            delta_scale = self.layer_residual_logit.sigmoid().to(
                dtype=learned_delta.dtype
            )
            return layer_basis + delta_scale * learned_delta
        if self.aggregation == "grouped10_layer_projected_hybrid":
            if layer_basis is None:
                raise ValueError(
                    "grouped10_layer_projected_hybrid requires a frozen layer basis"
                )
            expected = (
                bank.shape[0],
                self.num_tokens,
                self.hidden_dim,
            )
            if tuple(layer_basis.shape) != expected:
                raise ValueError(
                    "Unexpected layer basis shape: {} (expected {})".format(
                        tuple(layer_basis.shape), expected
                    )
                )
            prototype = F.normalize(bank.mean(dim=1), dim=-1)
            projected_tokens = self._mlp_tke(prototype)
            projected_tokens = F.normalize(projected_tokens.float(), dim=-1).to(
                dtype=projected_tokens.dtype
            )
            mix = self.layer_projected_hybrid_logit.sigmoid().to(
                dtype=projected_tokens.dtype
            )
            return layer_basis.to(
                dtype=projected_tokens.dtype, device=projected_tokens.device
            ) + mix * projected_tokens
        if self.aggregation == "grouped10_layer_projected_residual":
            if layer_basis is None:
                raise ValueError(
                    "grouped10_layer_projected_residual requires a frozen layer basis"
                )
            expected = (
                bank.shape[0],
                self.num_tokens,
                self.hidden_dim,
            )
            if tuple(layer_basis.shape) != expected:
                raise ValueError(
                    "Unexpected layer basis shape: {} (expected {})".format(
                        tuple(layer_basis.shape), expected
                    )
                )
            prototype = F.normalize(bank.mean(dim=1), dim=-1)
            projected_delta = self._mlp_tke(prototype)
            mix = self.layer_projected_residual_logit.sigmoid().to(
                dtype=projected_delta.dtype
            )
            return layer_basis.to(
                dtype=projected_delta.dtype, device=projected_delta.device
            ) + mix * projected_delta
        if self.aggregation == "tke_mean":
            return self._mlp_tke(bank).mean(dim=1)
        if self.aggregation == "consensus_weighted":
            weights = self._consensus_weights(bank)
            tokens = self._mlp_tke(bank)
            return (weights[..., None, None] * tokens).sum(dim=1)
        if self.aggregation in {
            "set_attention",
            "cosine_set_attention",
            "grouped10_cosine_attention",
            "layer_cosine_set_hybrid",
            "layer_cosine_set_hybrid_light",
            "layer_cosine_set_residual",
        }:
            if self.aggregation == "grouped10_cosine_attention":
                # BiomedCoOp descriptions are stored in ordered blocks of ten.
                # Average each block first, then let four shared semantic
                # queries select among the five class-level facets. This tests
                # the requested ten-description averaging without discarding
                # all between-block information.
                bank = F.normalize(
                    bank.reshape(
                        bank.shape[0],
                        self.description_count // 10,
                        10,
                        self.prior_dim,
                    ).mean(dim=2),
                    dim=-1,
                )
            keys = self.key_projection(bank)
            values = self.activation(self.value_projection(bank))
            if self.aggregation in {
                "cosine_set_attention",
                "grouped10_cosine_attention",
                "layer_cosine_set_hybrid",
                "layer_cosine_set_hybrid_light",
                "layer_cosine_set_residual",
            }:
                # The frozen description vectors and small prompt-style query
                # initialization make ordinary scaled dot products nearly
                # zero, which empirically collapses all four queries to the
                # uniform 50-description mean. Unit-normalized query/key
                # directions with dimension-derived scaling preserve the
                # standard unit-variance attention-logit regime without a
                # shot-specific or tuned temperature.
                queries = F.normalize(self.set_queries.float(), dim=-1)
                normalized_keys = F.normalize(keys.float(), dim=-1)
                score = torch.einsum(
                    "nd,ckd->cnk", queries, normalized_keys
                ) * math.sqrt(float(self.bottleneck_dim))
            else:
                score = torch.einsum("nd,ckd->cnk", self.set_queries, keys)
                score = score / math.sqrt(float(self.bottleneck_dim))
            attention = F.softmax(score, dim=-1)
            pooled = torch.einsum("cnk,ckd->cnd", attention, values)
            projected_tokens = self.output_projection(pooled)
            if self.aggregation in {
                "layer_cosine_set_hybrid",
                "layer_cosine_set_hybrid_light",
                "layer_cosine_set_residual",
            }:
                if layer_basis is None:
                    raise ValueError(
                        "layer_cosine_set_hybrid requires a frozen layer basis"
                    )
                expected = (
                    bank.shape[0],
                    self.num_tokens,
                    self.hidden_dim,
                )
                if tuple(layer_basis.shape) != expected:
                    raise ValueError(
                        "Unexpected layer basis shape: {} (expected {})".format(
                            tuple(layer_basis.shape), expected
                        )
                    )
                basis = layer_basis.to(
                    dtype=projected_tokens.dtype,
                    device=projected_tokens.device,
                )
                if self.aggregation == "layer_cosine_set_residual":
                    mix = self.layer_cosine_residual_logit.sigmoid().to(
                        dtype=projected_tokens.dtype
                    )
                    return basis + mix * projected_tokens
                mix = self.layer_cosine_mix_logit.sigmoid().to(
                    dtype=projected_tokens.dtype
                )
                return basis + mix * F.normalize(
                    projected_tokens.float(), dim=-1
                ).to(dtype=projected_tokens.dtype)
            return projected_tokens
        raise KeyError(self.aggregation)  # pragma: no cover

    def prompt_for_layer(self, layer_idx, class_tokens, dtype, device):
        if self.text_prompt is None:
            raise RuntimeError(
                "The in-place TCP connection does not create separate text-prompt slots"
            )
        shared = self.text_prompt.for_layer(
            layer_idx, class_tokens.shape[0], dtype, device
        )
        class_tokens = class_tokens.to(dtype=dtype, device=device)
        if self.connection == "late_residual":
            if layer_idx < self.insert_layer:
                return shared
            alpha = self.gate_logits[layer_idx - self.insert_layer].sigmoid()
            return shared + (alpha * self.residual_scale).to(dtype=dtype) * class_tokens
        if self.connection == "late_norm_residual":
            if layer_idx < self.insert_layer:
                return shared
            # TKE output magnitude is unconstrained and can otherwise dwarf the
            # shared prompt even with a small gate. Match every class-token
            # direction to the corresponding shared token norm before fusion.
            # Detaching only the reference norm prevents either branch from
            # gaming the scale while retaining directional TKE gradients.
            reference_norm = shared.detach().float().norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            matched_tokens = F.normalize(class_tokens.float(), dim=-1)
            matched_tokens = (matched_tokens * reference_norm).to(dtype=dtype)
            alpha = self.gate_logits[layer_idx - self.insert_layer].sigmoid()
            return shared + (alpha * self.residual_scale).to(dtype=dtype) * matched_tokens
        if self.connection == "late_centered_norm_residual":
            if layer_idx < self.insert_layer:
                return shared
            # The description bank contains a large medical-language component
            # shared by all classes. Remove it so TKE contributes only
            # class-differential directions; shared semantics remain in CoOp.
            centered_tokens = class_tokens.float() - class_tokens.float().mean(
                dim=0, keepdim=True
            )
            reference_norm = shared.detach().float().norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            matched_tokens = F.normalize(centered_tokens, dim=-1)
            matched_tokens = (matched_tokens * reference_norm).to(dtype=dtype)
            alpha = self.gate_logits[layer_idx - self.insert_layer].sigmoid()
            return shared + (alpha * self.residual_scale).to(dtype=dtype) * matched_tokens
        if self.connection == "late_centered_classlayer_norm_residual":
            if layer_idx < self.insert_layer:
                return shared
            centered_tokens = class_tokens.float() - class_tokens.float().mean(
                dim=0, keepdim=True
            )
            reference_norm = shared.detach().float().norm(
                dim=-1, keepdim=True
            ).clamp_min(1e-6)
            matched_tokens = F.normalize(centered_tokens, dim=-1)
            matched_tokens = (matched_tokens * reference_norm).to(dtype=dtype)
            alpha = self.gate_logits[layer_idx - self.insert_layer].sigmoid()
            if alpha.numel() != class_tokens.shape[0]:
                raise RuntimeError("Class-layer gate count does not match classes")
            alpha = alpha.reshape(-1, 1, 1)
            return shared + (alpha * self.residual_scale).to(
                dtype=dtype
            ) * matched_tokens
        if self.connection == "late_replace":
            return shared if layer_idx < self.insert_layer else class_tokens
        if self.connection == "all_residual":
            alpha = self.gate_logits[layer_idx].sigmoid()
            return shared + (alpha * self.residual_scale).to(dtype=dtype) * class_tokens
        if self.connection == "original_coop_replace":
            return shared
        raise KeyError(self.connection)  # pragma: no cover

    def inplace_context(self, layer_idx, hidden_context, class_tokens):
        """Add one norm-controlled class residual to existing CoOp slots."""

        if self.connection not in {
            "inplace_once_norm_residual",
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        }:
            raise RuntimeError("inplace_context is only valid for the in-place connection")
        if self.connection in {
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        }:
            if not self.insert_layer <= int(layer_idx) < self.depth:
                return hidden_context
        elif int(layer_idx) != self.insert_layer:
            return hidden_context
        if hidden_context.shape != class_tokens.shape:
            raise ValueError(
                "In-place CoOp slots and TKE tokens disagree: {} vs {}".format(
                    tuple(hidden_context.shape), tuple(class_tokens.shape)
                )
            )
        reference_norm = hidden_context.detach().float().norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-6)
        residual_tokens = class_tokens.float()
        if self.connection in {
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        }:
            residual_tokens = residual_tokens - residual_tokens.mean(
                dim=0, keepdim=True
            )
        matched_tokens = F.normalize(residual_tokens, dim=-1) * reference_norm
        if self.connection == "inplace_once_centered_classgate_norm_residual":
            if self.gate_logits.numel() != hidden_context.shape[0]:
                raise RuntimeError("Class-gate count does not match prompt classes")
            alpha = self.gate_logits.sigmoid().reshape(-1, 1, 1)
        else:
            gate_index = (
                int(layer_idx) - self.insert_layer
                if self.connection in {
                    "inplace_deep_centered_norm_residual",
                    "inplace_deep_ramped_centered_norm_residual",
                    "inplace_deep_balanced_ramp_centered_norm_residual",
                    "inplace_deep_terminal_boost_centered_norm_residual",
                    "inplace_deep_terminal_peak_centered_norm_residual",
                }
                else 0
            )
            alpha = self.gate_logits[gate_index].sigmoid()
        alpha = alpha * self.residual_scale
        return hidden_context + alpha.to(hidden_context.dtype) * matched_tokens.to(
            hidden_context.dtype
        )

    def set_residual_scale(self, value):
        value = float(value)
        if not 0.0 <= value <= 1.0:
            raise ValueError("residual_scale must satisfy 0 <= value <= 1")
        self.residual_scale = value


class MultiTextTCPBertTextEncoder(nn.Module):
    """BERT text adapter combining 50-description TKE and Deep Text Prompt."""

    def __init__(
        self,
        base_text_encoder,
        description_bank,
        descriptions,
        classnames,
        num_tokens=4,
        bottleneck_dim=128,
        insert_layer=8,
        aggregation="feature_mean",
        connection="late_residual",
        consensus_temperature=0.07,
        gate_init=0.1,
        model_id=BIOMEDCLIP_MODEL_ID,
        class_prior=None,
        prior_representation="projected_text",
        projected_description_bank=None,
    ):
        super().__init__()
        transformer = base_text_encoder.transformer
        if getattr(getattr(transformer, "config", None), "model_type", None) != "bert":
            raise TypeError("Multi-text TCP requires a Hugging Face BERT tower")
        layers = transformer.encoder.layer
        hidden_dim = getattr(
            transformer.embeddings.word_embeddings, "embedding_dim", None
        )
        if hidden_dim is None:
            hidden_dim = transformer.config.hidden_size
        if description_bank.dim() != 3 or description_bank.shape[0] != len(classnames):
            raise ValueError("Expected one description bank per class")
        if int(num_tokens) < 1:
            raise ValueError("num_tokens must be positive")

        normalized_names = tuple(name.replace("_", " ") for name in classnames)
        if (
            aggregation in {
                "grouped10_layer_residual",
                "grouped10_layer_projected_hybrid",
                "grouped10_layer_projected_residual",
                "layer_cosine_set_hybrid",
                "layer_cosine_set_hybrid_light",
                "layer_cosine_set_residual",
            }
            and prior_representation != "layer_cls"
        ):
            raise ValueError(
                "Grouped layer aggregation requires prior_representation='layer_cls'"
            )
        bank = F.normalize(description_bank.detach().float(), dim=-1)
        if class_prior is None:
            class_prior = F.normalize(bank.mean(dim=1), dim=-1)
        else:
            class_prior = F.normalize(class_prior.detach().float(), dim=-1)
            if class_prior.shape[0] != len(normalized_names):
                raise ValueError("Expected one projected KG prior per class")
        if projected_description_bank is None:
            projected_description_bank = bank
        else:
            projected_description_bank = F.normalize(
                projected_description_bank.detach().float(), dim=-1
            )
            if projected_description_bank.shape[:2] != bank.shape[:2]:
                raise ValueError("Projected and injection description banks disagree")
        category_text = json.dumps(
            list(normalized_names), ensure_ascii=False, separators=(",", ":")
        )
        description_fp = description_source_fingerprint(
            normalized_names, descriptions
        )
        metadata = {
            "category_order_fingerprint": _text_fingerprint(category_text),
            "template_fingerprint": _text_fingerprint("BIOMEDCOOP_TEMPLATES:50"),
            "model_fingerprint": _text_fingerprint(model_id),
            "prior_fingerprint": _sha256_bytes(
                class_prior.cpu().contiguous().numpy().tobytes()
            ),
            "description_fingerprint": description_fp,
            "aggregation_fingerprint": _text_fingerprint(aggregation),
            "connection_fingerprint": _text_fingerprint(
                connection
                if prior_representation == "projected_text"
                else "{}|prior={}|layer={}".format(
                    connection, prior_representation, insert_layer
                )
            ),
        }

        self.base_text_encoder = base_text_encoder
        self.register_buffer("description_bank", bank, persistent=False)
        self.register_buffer("class_prior", class_prior, persistent=False)
        self.register_buffer(
            "projected_description_bank",
            projected_description_bank,
            persistent=False,
        )
        prompt_bank = (
            projected_description_bank
            if aggregation in {
                "grouped10_layer_projected_hybrid",
                "grouped10_layer_projected_residual",
                "layer_cosine_set_hybrid",
                "layer_cosine_set_hybrid_light",
                "layer_cosine_set_residual",
            }
            else bank
        )
        self.tcp_prompt = MultiTextTCPPromptParameters(
            prior_dim=prompt_bank.shape[-1],
            hidden_dim=hidden_dim,
            num_tokens=num_tokens,
            bottleneck_dim=bottleneck_dim,
            depth=len(layers),
            description_count=bank.shape[1],
            aggregation=aggregation,
            connection=connection,
            insert_layer=insert_layer,
            consensus_temperature=consensus_temperature,
            gate_init=gate_init,
            metadata=metadata,
            num_classes=len(normalized_names),
        )
        self.classnames = normalized_names
        self.descriptions = descriptions
        self.num_tokens = int(num_tokens)
        self.hidden_dim = int(hidden_dim)
        self.prior_dim = int(bank.shape[-1])
        self.bottleneck_dim = int(bottleneck_dim)
        self.insert_layer = int(insert_layer)
        self.depth = len(layers)
        self.description_count = int(bank.shape[1])
        self.aggregation = aggregation
        self.connection = connection
        self.consensus_temperature = float(consensus_temperature)
        self.gate_init = float(gate_init)
        self.model_id = str(model_id)
        self.prior_representation = str(prior_representation)
        self._last_class_tokens = None

    @property
    def transformer(self):
        return self.base_text_encoder.transformer

    def metadata(self):
        metadata = self.tcp_prompt.checkpoint_metadata()
        metadata.update(
            {
                "classnames": list(self.classnames),
                "description_source": "BIOMEDCOOP_TEMPLATES",
                "description_count": self.description_count,
                "aggregation": self.aggregation,
                "connection": self.connection,
                "bottleneck_dim": self.bottleneck_dim,
                "insert_layer_zero_based": self.insert_layer,
                "consensus_temperature": self.consensus_temperature,
                "gate_init": self.gate_init,
                "model_id": self.model_id,
                "prior_representation": self.prior_representation,
            }
        )
        payload = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
        metadata["signature"] = _text_fingerprint(payload)
        return metadata

    def grouped_layer_targets(self):
        """Build four layer-aligned targets from all five ten-text groups."""

        if self.prior_representation != "layer_cls":
            raise RuntimeError(
                "Layer-token targets require prior_representation='layer_cls'"
            )
        if self.description_count % 10 != 0 or self.description_count // 10 != 5:
            raise RuntimeError("Layer-token targets require exactly five groups of ten")
        groups = self.description_bank.reshape(
            self.description_bank.shape[0], 5, 10, self.prior_dim
        ).mean(dim=2)
        # Four token-specific facets plus the mean of all five facets. The
        # shared mean incorporates the fifth group without creating a fifth
        # prompt position or dropping any descriptions.
        shared = groups.mean(dim=1, keepdim=True)
        return F.normalize(groups[:, : self.num_tokens] + shared, dim=-1)

    def last_class_tokens(self):
        if self._last_class_tokens is None:
            raise RuntimeError("Class tokens are unavailable before a text forward pass")
        return self._last_class_tokens

    def aggregate_class_tokens(self):
        """Aggregate the fixed description banks into four TKE tokens."""

        if self.aggregation in {
            "grouped10_layer_projected_hybrid",
            "grouped10_layer_projected_residual",
            "layer_cosine_set_hybrid",
            "layer_cosine_set_hybrid_light",
            "layer_cosine_set_residual",
        }:
            return self.tcp_prompt.aggregate_descriptions(
                self.projected_description_bank,
                layer_basis=self.grouped_layer_targets(),
            )
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
            raise ValueError("Adding multi-text prompt slots would truncate a valid token")
        return prompts[:, :keep_length], attention_mask[:, :keep_length]

    def _extended_mask(self, attention_mask, input_shape):
        try:
            return self.transformer.get_extended_attention_mask(
                attention_mask, input_shape, device=attention_mask.device
            )
        except TypeError:
            return self.transformer.get_extended_attention_mask(
                attention_mask, input_shape
            )

    def _replace_slots(self, hidden_states, start, values):
        end = start + self.num_tokens
        if hidden_states.shape[1] < end:
            raise ValueError("Text sequence has too few prompt slots")
        return torch.cat(
            (hidden_states[:, :start], values, hidden_states[:, end:]), dim=1
        )

    def forward(self, prompts, tokenized_prompts):
        reference = next(self.tcp_prompt.parameters())
        dtype = reference.dtype
        prompts = prompts.to(dtype=dtype)
        attention_mask = self._attention_mask(tokenized_prompts, prompts)
        class_tokens = self.aggregate_class_tokens()
        self._last_class_tokens = class_tokens
        if self.connection in {
            "inplace_once_norm_residual",
            "inplace_once_centered_norm_residual",
            "inplace_once_centered_classgate_norm_residual",
            "inplace_deep_centered_norm_residual",
            "inplace_deep_ramped_centered_norm_residual",
            "inplace_deep_balanced_ramp_centered_norm_residual",
            "inplace_deep_terminal_boost_centered_norm_residual",
            "inplace_deep_terminal_peak_centered_norm_residual",
        }:
            return self._forward_inplace(prompts, attention_mask, class_tokens)

        prompts, attention_mask = self._reserve_prompt_slots(prompts, attention_mask)
        first_prompt = self.tcp_prompt.prompt_for_layer(
            0, class_tokens, prompts.dtype, prompts.device
        )
        hidden_inputs = torch.cat(
            (prompts[:, :1], first_prompt, prompts[:, 1:]), dim=1
        )
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
                    layer_idx,
                    class_tokens,
                    hidden_states.dtype,
                    hidden_states.device,
                )
                hidden_states = self._replace_slots(hidden_states, 1, layer_prompt)
            if (
                self.connection == "original_coop_replace"
                and layer_idx == self.insert_layer
            ):
                coop_start = 1 + self.num_tokens
                hidden_states = self._replace_slots(
                    hidden_states,
                    coop_start,
                    class_tokens.to(hidden_states.dtype),
                )
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_mask,
                head_mask=head_mask[layer_idx],
            )
            hidden_states = layer_outputs[0]

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

    def _forward_inplace(self, prompts, attention_mask, class_tokens):
        """Run BERT without adding slots and alter the four CoOp positions once."""

        position_ids = torch.arange(
            prompts.shape[1], device=prompts.device, dtype=torch.long
        ).unsqueeze(0).expand(prompts.shape[0], -1)
        hidden_states = self.transformer.embeddings(
            inputs_embeds=prompts,
            token_type_ids=torch.zeros_like(attention_mask),
            position_ids=position_ids,
        )
        layers = self.transformer.encoder.layer
        extended_mask = self._extended_mask(attention_mask, hidden_states.shape[:2])
        head_mask = self.transformer.get_head_mask(None, len(layers))
        for layer_idx, layer in enumerate(layers):
            context = hidden_states[:, 1 : 1 + self.num_tokens]
            updated_context = self.tcp_prompt.inplace_context(
                layer_idx, context, class_tokens
            )
            if updated_context is not context:
                hidden_states = self._replace_slots(
                    hidden_states, 1, updated_context
                )
            layer_outputs = layer(
                hidden_states,
                attention_mask=extended_mask,
                head_mask=head_mask[layer_idx],
            )
            hidden_states = layer_outputs[0]

        transformer_pooler = getattr(self.transformer, "pooler", None)
        pooled_output = transformer_pooler(hidden_states) if transformer_pooler else None
        output = BaseModelOutputWithPooling(
            last_hidden_state=hidden_states, pooler_output=pooled_output
        )
        pooled = self.base_text_encoder.pooler(output, attention_mask)
        projected = self.base_text_encoder.proj(pooled)
        if getattr(self.base_text_encoder, "output_tokens", False):
            return projected, hidden_states
        return projected
