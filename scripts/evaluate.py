#!/usr/bin/env python3
"""Fit independent post-hoc probes on aligned frozen representation caches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from surfacelora.probes import evaluate_linear_validation, evaluate_probes
from surfacelora.representations import (
    RepresentationCache,
    validate_cache_pair,
    validate_cache_triplet,
)
from surfacelora.utils import load_config, resolve_device, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--representations", required=True, help="Directory containing aligned .npz caches")
    parser.add_argument("--output-dir", default="outputs/evaluation")
    parser.add_argument("--split", required=True, choices=("validation", "test"))
    parser.add_argument("--rouge-l", type=float, help="Aggregate validation ROUGE-L for a selection record")
    parser.add_argument("--step", type=int, default=0, help="Checkpoint optimizer step recorded in validation mode")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    root = Path(args.representations)
    train = RepresentationCache.load(root / "train.npz")
    validation = RepresentationCache.load(root / "validation.npz")
    if args.split == "validation":
        if args.rouge_l is None:
            raise ValueError("--rouge-l is required for a validation selection record")
        validate_cache_pair((train, validation))
        metrics = evaluate_linear_validation(
            train,
            validation,
            config=config,
            rouge_l=args.rouge_l,
            step=args.step,
        )
        filename = "validation_metrics.json"
    else:
        if args.rouge_l is not None:
            raise ValueError("--rouge-l is only accepted in validation mode")
        test = RepresentationCache.load(root / "test.npz")
        validate_cache_triplet((train, validation, test))
        metrics = evaluate_probes(
            train,
            validation,
            test,
            config=config,
            device=resolve_device(args.device),
        )
        filename = "aggregate_test_probe_metrics.json"
    write_json(metrics, Path(args.output_dir) / filename)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
