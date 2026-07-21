#!/usr/bin/env python3
"""Prepare patient-disjoint runtime splits and balanced audit subsets."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from surfacelora.data import balanced_sample, make_runtime_splits, patient_group_split, prepare_records
from surfacelora.utils import load_config, read_jsonl, write_json, write_jsonl


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to the YAML configuration")
    parser.add_argument("--input-path", help="Override data.input_path")
    parser.add_argument("--output-dir", help="Override data.output_dir")
    parser.add_argument("--seed", type=int, help="Override the configured random seed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed = int(args.seed if args.seed is not None else config.get("seed", 42))
    data = config["data"]
    input_path = Path(args.input_path or data["input_path"])
    output_dir = Path(args.output_dir or data["output_dir"])
    salt = os.environ.get("SURFACELORA_ALIGNMENT_SALT")
    if not salt:
        raise RuntimeError("Set SURFACELORA_ALIGNMENT_SALT to a private local value")

    cleaned, removal_counts = prepare_records(read_jsonl(input_path), fields=data["fields"])
    split_rows = patient_group_split(cleaned, seed=seed)
    runtime = make_runtime_splits(split_rows, salt=salt)
    balanced_counts = data["balanced_per_class"]
    label_order = tuple(data["label_order"])

    for split in ("train", "validation", "test"):
        write_jsonl(runtime[split], output_dir / f"{split}.jsonl")
        balanced = balanced_sample(
            runtime[split],
            per_class=int(balanced_counts[split]),
            label_order=label_order,
            seed=seed,
        )
        for index, row in enumerate(balanced):
            row["example_index"] = index
        name = "test_balanced.jsonl" if split == "test" else f"bal_{split}.jsonl"
        write_jsonl(balanced, output_dir / name)

    report = {
        "seed": seed,
        "removal_counts": removal_counts,
        "split_sizes": {name: len(rows) for name, rows in runtime.items()},
        "balanced_per_class": {name: int(value) for name, value in balanced_counts.items()},
        "contains_protected_runtime_data": True,
    }
    write_json(report, output_dir / "preparation_report.json")
    print(f"Prepared local runtime data in {output_dir}")


if __name__ == "__main__":
    main()
