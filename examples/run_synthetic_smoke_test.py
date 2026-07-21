#!/usr/bin/env python3
"""Run a small synthetic interface test; this does not reproduce paper results."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import torch

from surfacelora.checkpoint_selection import select_checkpoint
from surfacelora.data import balanced_sample, make_runtime_splits, patient_group_split
from surfacelora.grl import gradient_reverse
from surfacelora.metrics import leakage_gap, rouge_scores
from surfacelora.probes import evaluate_linear_validation, evaluate_probes
from surfacelora.representations import (
    CacheMetadata,
    RepresentationCache,
    lasttok,
    meanpool,
    validate_cache_triplet,
)
from surfacelora.utils import load_config


LABELS = ("WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER")


def make_cache(split: str, count_per_class: int, offset: int, rng: np.random.RandomState):
    labels = np.repeat(np.arange(5), count_per_class)
    vectors = np.eye(5, dtype=np.float32)[labels] + rng.normal(0, 0.05, (len(labels), 5)).astype(np.float32)
    return RepresentationCache(
        vectors=vectors,
        labels=labels,
        example_indices=np.arange(len(labels)),
        row_keys=np.asarray([f"synthetic-{offset + index}" for index in range(len(labels))]),
        metadata=CacheMetadata(
            split=split,
            representation="lasttok",
            layer=-1,
            checkpoint_identifier="synthetic-checkpoint",
            tokenizer_identifier="synthetic-tokenizer",
            prompt_identifier="synthetic-prompt-v1",
            maximum_sequence_length=16,
            padding_side="left",
            truncation_side="left",
            label_order=LABELS,
        ),
    )


def main() -> None:
    config_path = Path(__file__).resolve().parents[1] / "configs" / "main.yaml"
    assert load_config(config_path)["representation"]["target"] == "lasttok"
    rows = [
        {"_source_index": group * 5 + label_id, "_group": f"group-{group}", "source": "synthetic source", "target": "synthetic target", "sensitive_label": label}
        for group in range(20)
        for label_id, label in enumerate(LABELS)
    ]
    splits = patient_group_split(rows, seed=42, test_fraction=0.2, validation_fraction_of_remainder=0.25)
    runtime = make_runtime_splits(splits, salt="synthetic-only")
    for values in runtime.values():
        assert len(balanced_sample(values, per_class=1, label_order=LABELS, seed=42)) == 5

    hidden = torch.arange(2 * 4 * 3, dtype=torch.float32).reshape(2, 4, 3).requires_grad_()
    mask = torch.tensor([[0, 1, 1, 1], [1, 1, 0, 0]])
    assert lasttok(hidden, mask).shape == (2, 3)
    assert meanpool(hidden, mask).shape == (2, 3)
    gradient_reverse(lasttok(hidden, mask), 0.5).sum().backward()

    rng = np.random.RandomState(42)
    caches = [make_cache("train", 10, 0, rng), make_cache("validation", 5, 100, rng), make_cache("test", 5, 200, rng)]
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "train.npz"
        caches[0].save(path)
        caches[0] = RepresentationCache.load(path)
    validate_cache_triplet(caches)
    metrics = evaluate_probes(
        *caches,
        config={
            "seed": 42,
            "selection": {"chance_accuracy": 0.20},
            "probes": {"linear_max_iter": 100, "mlp_hidden_dim": 16, "mlp_max_epochs": 4, "mlp_patience": 2, "mlp_batch_size": 16},
        },
        device="cpu",
    )
    assert set(metrics).isdisjoint({"labels", "predictions", "row_keys", "texts"})
    validation_record = evaluate_linear_validation(
        caches[0],
        caches[1],
        config={"seed": 42, "selection": {"chance_accuracy": 0.20}, "probes": {"linear_max_iter": 100}},
        rouge_l=32.0,
        step=200,
    )
    assert set(validation_record).isdisjoint({"test_accuracy", "labels", "predictions"})
    assert leakage_gap(0.20) == 0.0
    assert rouge_scores(["brief course improved"], ["brief course improved"])["rougeL"] == 100.0
    selected = select_checkpoint([validation_record])
    assert selected["checkpoint_identifier"] == "synthetic-checkpoint"
    print("Synthetic SurfaceLoRA smoke test: PASS")


if __name__ == "__main__":
    main()
