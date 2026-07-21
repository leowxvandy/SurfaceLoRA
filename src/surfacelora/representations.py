"""Attention-mask-aware prompt representations and aligned local caches."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def final_nonpadding_indices(attention_mask: torch.Tensor) -> torch.Tensor:
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, sequence]")
    mask = attention_mask.to(dtype=torch.bool)
    if not torch.all(mask.any(dim=1)):
        raise ValueError("Every sequence must contain a non-padding token")
    positions = torch.arange(mask.shape[1], device=mask.device).expand_as(mask)
    return positions.masked_fill(~mask, -1).max(dim=1).values


def lasttok(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden-state and attention-mask shapes are incompatible")
    rows = torch.arange(hidden.shape[0], device=hidden.device)
    indices = final_nonpadding_indices(attention_mask.to(hidden.device))
    return hidden[rows, indices]


def meanpool(hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    if hidden.ndim != 3 or hidden.shape[:2] != attention_mask.shape:
        raise ValueError("Hidden-state and attention-mask shapes are incompatible")
    mask = attention_mask.to(device=hidden.device, dtype=hidden.dtype).unsqueeze(-1)
    return (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


def extract_representation(
    hidden_states: Sequence[torch.Tensor],
    attention_mask: torch.Tensor,
    representation: str,
    *,
    layer: int = -1,
) -> torch.Tensor:
    if representation not in {"lasttok", "meanpool"}:
        raise ValueError("representation must be 'lasttok' or 'meanpool'")
    # Transformers returns the embedding output followed by one state per block.
    # Public layer indices refer to transformer blocks: 0 is the first block and
    # -1 is the final block (L-1), never the embedding output.
    block_count = len(hidden_states) - 1
    block_index = layer if layer >= 0 else block_count + layer
    if block_index < 0 or block_index >= block_count:
        raise ValueError(f"Transformer-block index {layer} is unavailable")
    hidden = hidden_states[block_index + 1]
    return lasttok(hidden, attention_mask) if representation == "lasttok" else meanpool(hidden, attention_mask)


@dataclass(frozen=True)
class CacheMetadata:
    split: str
    representation: str
    layer: int
    checkpoint_identifier: str
    tokenizer_identifier: str
    prompt_identifier: str
    maximum_sequence_length: int
    padding_side: str
    truncation_side: str
    label_order: tuple[str, ...]
    format_version: int = 1


@dataclass
class RepresentationCache:
    vectors: np.ndarray
    labels: np.ndarray
    example_indices: np.ndarray
    row_keys: np.ndarray
    metadata: CacheMetadata
    alignment_digest: str | None = None

    def digest(self) -> str:
        value = hashlib.sha256()
        vectors = np.ascontiguousarray(self.vectors)
        value.update(str(vectors.shape).encode("utf-8"))
        value.update(str(vectors.dtype).encode("utf-8"))
        value.update(vectors.view(np.uint8))
        value.update(json.dumps(asdict(self.metadata), sort_keys=True).encode("utf-8"))
        for index, key, label in zip(self.example_indices, self.row_keys, self.labels):
            value.update(f"{int(index)}\t{str(key)}\t{int(label)}\n".encode("utf-8"))
        return value.hexdigest()

    def validate(self) -> None:
        if self.vectors.ndim != 2:
            raise ValueError("vectors must have shape [examples, dimensions]")
        count = self.vectors.shape[0]
        if any(len(values) != count for values in (self.labels, self.example_indices, self.row_keys)):
            raise ValueError("Vectors, labels, indices, and row keys must align")
        indices = [int(value) for value in self.example_indices]
        keys = [str(value) for value in self.row_keys]
        if indices != list(range(count)):
            raise ValueError("Example indices must be contiguous and ordered")
        if len(set(keys)) != count:
            raise ValueError("Duplicate row keys in representation cache")
        if not set(int(value) for value in self.labels) <= set(range(len(self.metadata.label_order))):
            raise ValueError("Cache contains a label outside label_order")
        required_identity = (
            self.metadata.split,
            self.metadata.representation,
            self.metadata.checkpoint_identifier,
            self.metadata.tokenizer_identifier,
            self.metadata.prompt_identifier,
        )
        if any(not str(value).strip() for value in required_identity):
            raise ValueError("Representation-cache identity fields must be non-empty")
        if self.alignment_digest is not None and self.alignment_digest != self.digest():
            raise ValueError("Representation-cache alignment digest does not match")

    def save(self, path: str | Path) -> None:
        self.validate()
        self.alignment_digest = self.digest()
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            destination,
            vectors=self.vectors,
            labels=self.labels,
            example_indices=self.example_indices,
            row_keys=self.row_keys.astype(str),
            metadata=np.asarray(json.dumps(asdict(self.metadata))),
            alignment_digest=np.asarray(self.alignment_digest),
        )

    @classmethod
    def load(cls, path: str | Path) -> "RepresentationCache":
        with np.load(Path(path), allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            metadata["label_order"] = tuple(metadata["label_order"])
            cache = cls(
                vectors=np.asarray(data["vectors"]),
                labels=np.asarray(data["labels"]),
                example_indices=np.asarray(data["example_indices"]),
                row_keys=np.asarray(data["row_keys"]).astype(str),
                metadata=CacheMetadata(**metadata),
                alignment_digest=str(data["alignment_digest"].item()),
            )
        cache.validate()
        return cache


def validate_cache_collection(
    caches: Sequence[RepresentationCache], expected_splits: Sequence[str]
) -> None:
    if [cache.metadata.split for cache in caches] != list(expected_splits):
        raise ValueError(f"Expected caches in this order: {list(expected_splits)}")
    identity = (
        "representation",
        "layer",
        "checkpoint_identifier",
        "tokenizer_identifier",
        "prompt_identifier",
        "maximum_sequence_length",
        "padding_side",
        "truncation_side",
        "label_order",
    )
    for cache in caches:
        cache.validate()
    for field in identity:
        if len({getattr(cache.metadata, field) for cache in caches}) != 1:
            raise ValueError(f"Cache metadata mismatch for {field}")
    key_sets = [set(str(value) for value in cache.row_keys) for cache in caches]
    if any(
        key_sets[left] & key_sets[right]
        for left in range(len(key_sets))
        for right in range(left + 1, len(key_sets))
    ):
        raise ValueError("Row keys must be disjoint across splits")


def validate_cache_pair(caches: Sequence[RepresentationCache]) -> None:
    validate_cache_collection(caches, ("train", "validation"))


def validate_cache_triplet(caches: Sequence[RepresentationCache]) -> None:
    validate_cache_collection(caches, ("train", "validation", "test"))
