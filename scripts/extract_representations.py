#!/usr/bin/env python3
"""Extract aligned prompt representations from balanced local audit splits."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from surfacelora.prompting import PROMPT_IDENTIFIER, render_prompt
from surfacelora.representations import CacheMetadata, RepresentationCache, extract_representation
from surfacelora.utils import load_config, read_jsonl, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--representation", required=True, choices=("lasttok", "meanpool"))
    parser.add_argument("--adapter", required=True, help="Trained LoRA adapter directory")
    parser.add_argument("--checkpoint-id", required=True, help="Stable non-path identifier stored with the cache")
    parser.add_argument("--output-dir", default="outputs/representations")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=("train", "validation", "test"),
        default=("train", "validation"),
        help="Defaults to train/validation; extract test only after checkpoint selection",
    )
    return parser.parse_args()


def load_model(config: dict, adapter: str | None, device: str):
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the 'llm' dependencies for representation extraction") from exc
    name = str(config["model"]["name_or_path"])
    local_only = bool(config["model"].get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_only, use_fast=True)
    tokenizer.padding_side = str(config["tokenization"].get("padding_side", "left"))
    tokenizer.truncation_side = str(config["tokenization"].get("truncation_side", "left"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = torch.bfloat16 if device.startswith("cuda") and config["model"].get("dtype") == "bfloat16" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name,
        local_files_only=local_only,
        torch_dtype=dtype,
    )
    if adapter:
        try:
            from peft import PeftModel
        except ImportError as exc:
            raise RuntimeError("Install PEFT to load a LoRA adapter") from exc
        model = PeftModel.from_pretrained(model, adapter)
    model.to(device).eval()
    return model, tokenizer


@torch.inference_mode()
def extract_split(model, tokenizer, rows, *, config, representation, device, batch_size):
    vectors: list[np.ndarray] = []
    max_length = int(config["tokenization"]["max_prompt_length"])
    layer = int(config["representation"].get("layer", -1))
    for start in range(0, len(rows), batch_size):
        batch_rows = rows[start : start + batch_size]
        prompts = [render_prompt(tokenizer, str(row["source"])) for row in batch_rows]
        encoded = tokenizer(
            prompts,
            add_special_tokens=False,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(device)
        output = model(**encoded, output_hidden_states=True, use_cache=False)
        values = extract_representation(
            output.hidden_states,
            encoded["attention_mask"],
            representation,
            layer=layer,
        )
        vectors.append(values.float().cpu().numpy())
    return np.concatenate(vectors, axis=0)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    device = resolve_device(args.device or str(config.get("runtime", {}).get("device", "auto")))
    model, tokenizer = load_model(config, args.adapter, device)
    prepared = Path(config["data"]["output_dir"])
    files = {
        "train": prepared / "bal_train.jsonl",
        "validation": prepared / "bal_validation.jsonl",
        "test": prepared / "test_balanced.jsonl",
    }
    label_order = tuple(config["data"]["label_order"])
    label_to_id = {label: index for index, label in enumerate(label_order)}
    checkpoint_identifier = str(args.checkpoint_id)
    batch_size = int(args.batch_size or config.get("generation", {}).get("batch_size", 8))
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if "/" in checkpoint_identifier or "\\" in checkpoint_identifier:
        raise ValueError("checkpoint-id must be a stable identifier, not a filesystem path")
    output_dir = Path(args.output_dir)

    for split in args.splits:
        path = files[split]
        rows = read_jsonl(path)
        if not rows:
            raise ValueError(f"The {split} audit split is empty")
        expected = list(range(len(rows)))
        if [int(row["example_index"]) for row in rows] != expected:
            raise ValueError(f"The {split} audit rows are not contiguously ordered")
        cache = RepresentationCache(
            vectors=extract_split(
                model,
                tokenizer,
                rows,
                config=config,
                representation=args.representation,
                device=device,
                batch_size=batch_size,
            ),
            labels=np.asarray([label_to_id[str(row["sensitive_label"])] for row in rows], dtype=np.int64),
            example_indices=np.asarray(expected, dtype=np.int64),
            row_keys=np.asarray([str(row["row_key"]) for row in rows]),
            metadata=CacheMetadata(
                split=split,
                representation=args.representation,
                layer=int(config["representation"].get("layer", -1)),
                checkpoint_identifier=checkpoint_identifier,
                tokenizer_identifier=str(getattr(tokenizer, "name_or_path", config["model"]["name_or_path"])),
                prompt_identifier=PROMPT_IDENTIFIER,
                maximum_sequence_length=int(config["tokenization"]["max_prompt_length"]),
                padding_side=tokenizer.padding_side,
                truncation_side=tokenizer.truncation_side,
                label_order=label_order,
            ),
        )
        cache.save(output_dir / f"{split}.npz")
        print(f"Saved {split} aggregate representation cache with {len(rows)} rows")


if __name__ == "__main__":
    main()
