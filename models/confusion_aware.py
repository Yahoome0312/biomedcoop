"""Online confusion-aware components using current-image predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _encode_ascii(value: str) -> torch.Tensor:
    return torch.tensor(list(value.encode("ascii")), dtype=torch.uint8)


def decode_ascii(value: torch.Tensor) -> str:
    return bytes(value.detach().cpu().tolist()).decode("ascii")


def _normalized_classname(value):
    return " ".join(str(value).replace("_", " ").strip().casefold().split())


def build_frozen_pair_description_bank(
    model, tokenizer, classnames, description_file, batch_size=32
):
    """Load and encode every directed LLM class-pair description.

    The JSON object must map each class name to every other class name, whose
    value is a non-empty list of descriptions. Class count and description
    count are inferred from the active dataset rather than fixed here.
    """

    description_file = Path(description_file)
    if not description_file.is_file():
        raise FileNotFoundError(
            "LLM confusion-pair description file is missing: {}".format(
                description_file
            )
        )
    with description_file.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError("LLM pair description root must be a JSON object")

    normalized_names = [_normalized_classname(name) for name in classnames]
    if len(normalized_names) < 2 or len(set(normalized_names)) != len(normalized_names):
        raise ValueError("Dataset class names must contain at least two unique classes")

    normalized_payload = {}
    for outer_name, targets in payload.items():
        outer = _normalized_classname(outer_name)
        if outer in normalized_payload:
            raise ValueError("Duplicate normalized source class {!r}".format(outer))
        if not isinstance(targets, dict):
            raise ValueError("Descriptions for {!r} must be a JSON object".format(outer_name))
        normalized_targets = {}
        for target_name, descriptions in targets.items():
            target = _normalized_classname(target_name)
            if target in normalized_targets:
                raise ValueError(
                    "Duplicate normalized target class {!r} for {!r}".format(
                        target, outer
                    )
                )
            if (
                not isinstance(descriptions, list)
                or not descriptions
                or any(not isinstance(text, str) or not text.strip() for text in descriptions)
            ):
                raise ValueError(
                    "Pair ({!r}, {!r}) must contain non-empty description strings".format(
                        outer_name, target_name
                    )
                )
            normalized_targets[target] = [text.strip() for text in descriptions]
        normalized_payload[outer] = normalized_targets

    expected_classes = set(normalized_names)
    if set(normalized_payload) != expected_classes:
        raise ValueError(
            "LLM pair source classes do not match dataset classes: expected {}, got {}".format(
                sorted(expected_classes), sorted(normalized_payload)
            )
        )

    ordered_pairs = []
    flattened = []
    canonical_descriptions = []
    for first_index, first_name in enumerate(normalized_names):
        targets = normalized_payload[first_name]
        expected_targets = expected_classes - {first_name}
        if set(targets) != expected_targets:
            raise ValueError(
                "LLM targets for {!r} do not match the other dataset classes".format(
                    classnames[first_index]
                )
            )
        for second_index, second_name in enumerate(normalized_names):
            if first_index == second_index:
                continue
            descriptions = targets[second_name]
            start = len(flattened)
            flattened.extend(descriptions)
            ordered_pairs.append((first_index, second_index, start, len(flattened)))
            canonical_descriptions.append(
                {
                    "first": first_name,
                    "second": second_name,
                    "descriptions": descriptions,
                }
            )

    tokenized = torch.cat([tokenizer(text) for text in flattened])
    device = next(model.parameters()).device
    encoded = []
    with torch.no_grad():
        for start in range(0, tokenized.shape[0], int(batch_size)):
            batch = tokenized[start : start + int(batch_size)].to(device)
            encoded.append(model.encode_text(batch, normalize=True).float().cpu())
    features = torch.cat(encoded)
    pair_bank = torch.zeros(
        len(normalized_names),
        len(normalized_names),
        features.shape[-1],
        dtype=torch.float32,
    )
    for first_index, second_index, start, end in ordered_pairs:
        pair_bank[first_index, second_index] = F.normalize(
            features[start:end].mean(dim=0), dim=0
        )

    canonical = json.dumps(
        {
            "class_order": normalized_names,
            "pairs": canonical_descriptions,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    metadata = {
        "source_file": str(description_file.resolve()),
        "class_order": [str(name) for name in classnames],
        "pair_count": len(ordered_pairs),
        "description_count": len(flattened),
        "description_fingerprint": _sha256_bytes(canonical),
        "feature_fingerprint": _sha256_bytes(
            pair_bank.contiguous().numpy().tobytes()
        ),
    }
    return pair_bank, metadata


def select_confusion_pairs(logits, first):
    """Select the hardest negative for each explicit class anchor."""

    if logits.dim() != 2:
        raise ValueError("Classification logits must be rank 2")
    classes = logits.shape[1]
    if classes < 2:
        raise ValueError("Confusion selection requires at least two classes")
    first = torch.as_tensor(first, dtype=torch.long, device=logits.device)
    if first.dim() != 1 or first.shape[0] != logits.shape[0]:
        raise ValueError("Pair anchors and logits disagree")
    if first.numel() and (first.min() < 0 or first.max() >= classes):
        raise ValueError("Pair anchor is outside the class range")
    probabilities = logits.detach().float().softmax(dim=-1)
    scores = probabilities
    scores = scores.scatter(1, first.unsqueeze(1), float("-inf"))
    second = scores.argmax(dim=-1)
    return first, second, probabilities, scores


def confusion_margin_loss(logits, labels, competitor):
    labels = labels.long()
    competitor = competitor.long()
    if labels.shape != competitor.shape or labels.shape[0] != logits.shape[0]:
        raise ValueError("Labels, competitors and logits disagree")
    if labels.eq(competitor).any():
        raise ValueError("Margin competitor must differ from the true label")
    row = torch.arange(logits.shape[0], device=logits.device)
    loss = F.softplus(logits[row, competitor] - logits[row, labels]).mean()
    return loss


class ConfusionAwareAdapter(nn.Module):
    """Fixed full semantic/global/local branch with no second classifier."""

    def __init__(
        self,
        pair_description_bank,
        pair_feature_fingerprint,
        gamma=0.2,
        feature_dim=512,
        patch_dim=768,
    ):
        super().__init__()
        self.gamma = float(gamma)
        if self.gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.feature_dim = int(feature_dim)
        self.patch_dim = int(patch_dim)
        classes = pair_description_bank.shape[0]
        if tuple(pair_description_bank.shape) != (
            classes,
            classes,
            self.feature_dim,
        ):
            raise ValueError("LLM pair-description Bank shape is invalid")
        self.register_buffer(
            "pair_description_bank",
            pair_description_bank.detach().float().clone(),
            persistent=False,
        )
        self.register_buffer(
            "_pair_feature_fingerprint",
            _encode_ascii(str(pair_feature_fingerprint)),
        )

        self.semantic_projector = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
        )
        self.global_gate = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
            nn.Sigmoid(),
        )
        self.global_norm = nn.LayerNorm(feature_dim)
        self.query_projection = nn.Linear(feature_dim, patch_dim)
        self.patch_projection = nn.Linear(patch_dim, feature_dim)
        nn.init.xavier_uniform_(self.query_projection.weight)
        nn.init.zeros_(self.query_projection.bias)
        nn.init.normal_(self.patch_projection.weight, std=1e-3)
        nn.init.zeros_(self.patch_projection.bias)
        self.global_local_gate = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, 256),
            nn.GELU(),
            nn.Linear(256, 2),
        )
        self.final_fusion = nn.Sequential(
            nn.LayerNorm(2 * feature_dim),
            nn.Linear(2 * feature_dim, feature_dim),
            nn.GELU(),
            nn.Linear(feature_dim, feature_dim),
        )

    @property
    def pair_feature_fingerprint(self):
        return decode_ascii(self._pair_feature_fingerprint)

    @property
    def needs_patch_tokens(self):
        return True

    def forward(
        self,
        global_features,
        patch_tokens,
        text_features,
        base_logits,
        logit_scale,
        first,
    ):
        first, second, probabilities, scores = select_confusion_pairs(
            base_logits, first
        )
        details = {
            "pair_first": first,
            "pair_second": second,
            "base_probabilities": probabilities,
            "selected_score": scores.gather(1, second.unsqueeze(1)).squeeze(1),
        }
        semantic_input = self.pair_description_bank[first, second].to(
            dtype=global_features.dtype
        )
        semantic = self.semantic_projector(semantic_input)
        details["llm_pair_feature_norm"] = semantic_input.float().norm(dim=-1)
        details["semantic_norm"] = semantic.float().norm(dim=-1)

        gate = self.global_gate(torch.cat((global_features, semantic), dim=-1))
        global_confusion = self.global_norm(gate * global_features)
        if patch_tokens is None:
            raise RuntimeError("Full confusion branch requires patch tokens")
        query = self.query_projection(semantic)
        attention = torch.softmax(
            torch.einsum("bd,bnd->bn", query, patch_tokens)
            / (self.patch_dim ** 0.5),
            dim=-1,
        )
        local_value = torch.einsum("bn,bnd->bd", attention, patch_tokens)
        local_confusion = self.patch_projection(local_value)
        details["local_attention_max"] = attention.max(dim=-1).values

        weights = torch.softmax(
            self.global_local_gate(torch.cat((global_features, semantic), dim=-1)),
            dim=-1,
        )
        visual_confusion = (
            weights[:, :1] * global_confusion
            + weights[:, 1:] * local_confusion
        )
        details["alpha_global"] = weights[:, 0]
        details["alpha_local"] = weights[:, 1]
        confusion = self.final_fusion(
            torch.cat((semantic, visual_confusion), dim=-1)
        )

        final_features = F.normalize(
            F.normalize(global_features, dim=-1)
            + self.gamma * F.normalize(confusion, dim=-1),
            dim=-1,
        )
        logits = logit_scale * final_features @ F.normalize(text_features, dim=-1).t()
        details["final_features"] = final_features
        return logits, details
