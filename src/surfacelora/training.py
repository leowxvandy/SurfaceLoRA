"""Baseline LoRA and paired-branch SurfaceLoRA training."""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from .grl import gradient_reverse
from .modeling import assert_trainable_isolation, build_discriminator
from .prompting import PROMPT_IDENTIFIER, render_prompt
from .representations import extract_representation
from .utils import read_jsonl, resolve_device, set_seed


def encode_sft_example(
    tokenizer: Any,
    prompt: str,
    target: str,
    *,
    max_length: int,
) -> dict[str, list[int]]:
    prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
    target_ids = tokenizer(target, add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        target_ids = target_ids + [tokenizer.eos_token_id]
    input_ids = prompt_ids + target_ids
    labels = [-100] * len(prompt_ids) + target_ids
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids, labels = input_ids[overflow:], labels[overflow:]
    if not any(label != -100 for label in labels):
        raise ValueError("SFT truncation removed all target tokens")
    return {"input_ids": input_ids, "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, rows: Sequence[Mapping[str, Any]], tokenizer: Any, max_length: int) -> None:
        self.rows, self.tokenizer, self.max_length = list(rows), tokenizer, int(max_length)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, list[int]]:
        row = self.rows[index]
        return encode_sft_example(
            self.tokenizer,
            render_prompt(self.tokenizer, str(row["source"])),
            str(row["target"]),
            max_length=self.max_length,
        )


class AuditDataset(Dataset):
    def __init__(
        self,
        rows: Sequence[Mapping[str, Any]],
        tokenizer: Any,
        *,
        label_order: Sequence[str],
        max_length: int,
    ) -> None:
        label_to_id = {label: index for index, label in enumerate(label_order)}
        self.items = []
        for row in rows:
            label = str(row["sensitive_label"])
            if label not in label_to_id:
                raise ValueError(f"Unexpected sensitive label: {label}")
            prompt = render_prompt(tokenizer, str(row["source"]))
            ids = tokenizer(prompt, add_special_tokens=False)["input_ids"][-max_length:]
            self.items.append({"input_ids": ids, "sensitive_label": label_to_id[label]})

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, Any]:
        return self.items[index]


def left_pad_sft(batch: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, attention_mask, labels = [], [], []
    for row in batch:
        padding = width - len(row["input_ids"])
        input_ids.append([pad_token_id] * padding + list(row["input_ids"]))
        attention_mask.append([0] * padding + [1] * len(row["input_ids"]))
        labels.append([-100] * padding + list(row["labels"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
    }


def left_pad_audit(batch: Sequence[Mapping[str, Any]], pad_token_id: int) -> dict[str, torch.Tensor]:
    width = max(len(row["input_ids"]) for row in batch)
    input_ids, attention_mask = [], []
    for row in batch:
        padding = width - len(row["input_ids"])
        input_ids.append([pad_token_id] * padding + list(row["input_ids"]))
        attention_mask.append([0] * padding + [1] * len(row["input_ids"]))
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "sensitive_label": torch.tensor(
            [int(row["sensitive_label"]) for row in batch], dtype=torch.long
        ),
    }


def _infinite(loader: DataLoader):
    while True:
        yield from loader


def load_lora_model(config: Mapping[str, Any], device: str) -> tuple[Any, Any]:
    try:
        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as exc:
        raise RuntimeError("Install the 'llm' dependencies for model training") from exc
    model_config = config["model"]
    name = str(model_config["name_or_path"])
    local_only = bool(model_config.get("local_files_only", False))
    tokenizer = AutoTokenizer.from_pretrained(name, local_files_only=local_only, use_fast=True)
    tokenizer.padding_side = str(config["tokenization"].get("padding_side", "left"))
    tokenizer.truncation_side = str(config["tokenization"].get("truncation_side", "left"))
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if device.startswith("cuda"):
        allow_tf32 = bool(config.get("runtime", {}).get("allow_tf32", True))
        torch.backends.cuda.matmul.allow_tf32 = allow_tf32
        torch.backends.cudnn.allow_tf32 = allow_tf32
    dtype = torch.bfloat16 if device.startswith("cuda") and model_config.get("dtype") == "bfloat16" else torch.float32
    load_kwargs = {"local_files_only": local_only, "torch_dtype": dtype}
    try:
        model = AutoModelForCausalLM.from_pretrained(
            name, attn_implementation="flash_attention_2", **load_kwargs
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(name, **load_kwargs)
    lora = config["lora"]
    model = get_peft_model(
        model,
        LoraConfig(
            r=int(lora.get("rank", 16)),
            lora_alpha=int(lora.get("alpha", 32)),
            lora_dropout=float(lora.get("dropout", 0.05)),
            target_modules=list(lora.get("target_modules", ["q_proj", "v_proj"])),
            bias="none",
            task_type="CAUSAL_LM",
        ),
    ).to(device)
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    return model, tokenizer


def _training_settings(config: Mapping[str, Any]) -> tuple[int, int, int]:
    settings = config["training"]
    batch_size = int(settings.get("per_device_batch_size", 4))
    accumulation = int(settings.get("gradient_accumulation_steps", 32))
    max_steps = int(settings.get("max_steps", 2000))
    if min(batch_size, accumulation, max_steps) <= 0:
        raise ValueError("Batch size, accumulation, and max_steps must be positive")
    return batch_size, accumulation, max_steps


def train_baseline(config: Mapping[str, Any], output_dir: str | Path) -> dict[str, Any]:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    model, tokenizer = load_lora_model(config, device)
    rows = read_jsonl(Path(config["data"]["output_dir"]) / "train.jsonl")
    batch_size, accumulation, max_steps = _training_settings(config)
    if len(rows) < batch_size:
        raise ValueError("The utility training split is smaller than one full batch")
    loader = DataLoader(
        SFTDataset(rows, tokenizer, int(config["tokenization"]["max_sequence_length"])),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda batch: left_pad_sft(batch, tokenizer.pad_token_id),
    )
    batches = _infinite(loader)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters or any("lora_" not in name.lower() for name, parameter in model.named_parameters() if parameter.requires_grad):
        raise ValueError("Baseline training must update LoRA parameters only")
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings.get("learning_rate", 2e-4)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
    )
    model.train()
    for _ in range(max_steps):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            batch = {key: value.to(device) for key, value in next(batches).items()}
            (model(**batch, use_cache=False).loss / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(settings.get("max_grad_norm", 1.0)))
        optimizer.step()
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination / "adapter_final")
    tokenizer.save_pretrained(destination / "adapter_final")
    report = {"method": "lora_baseline", "optimizer_steps": max_steps, "seed": seed}
    (destination / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def _assert_globally_balanced(rows: Sequence[Mapping[str, Any]], label_order: Sequence[str]) -> dict[str, int]:
    counts = Counter(str(row["sensitive_label"]) for row in rows)
    if set(counts) != set(label_order) or len(set(counts.values())) != 1:
        raise ValueError("Adversarial rows must be globally class balanced")
    return {label: counts[label] for label in label_order}


def train_surfacelora(
    config: Mapping[str, Any],
    *,
    lambda_adv: float,
    output_dir: str | Path,
) -> dict[str, Any]:
    if lambda_adv < 0:
        raise ValueError("lambda_adv must be non-negative")
    seed = int(config.get("seed", 42))
    set_seed(seed)
    device = resolve_device(str(config.get("runtime", {}).get("device", "auto")))
    language_model, tokenizer = load_lora_model(config, device)
    prepared = Path(config["data"]["output_dir"])
    utility_rows = read_jsonl(prepared / "train.jsonl")
    adversarial_rows = read_jsonl(prepared / "bal_train.jsonl")
    label_order = tuple(config["data"]["label_order"])
    balanced_counts = _assert_globally_balanced(adversarial_rows, label_order)
    batch_size, accumulation, max_steps = _training_settings(config)
    if len(utility_rows) < batch_size or len(adversarial_rows) < batch_size:
        raise ValueError("Each SurfaceLoRA branch must contain at least one full batch")
    utility_loader = DataLoader(
        SFTDataset(utility_rows, tokenizer, int(config["tokenization"]["max_sequence_length"])),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed),
        collate_fn=lambda batch: left_pad_sft(batch, tokenizer.pad_token_id),
    )
    adversarial_loader = DataLoader(
        AuditDataset(
            adversarial_rows,
            tokenizer,
            label_order=label_order,
            max_length=int(config["tokenization"]["max_prompt_length"]),
        ),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        generator=torch.Generator().manual_seed(seed + 1),
        collate_fn=lambda batch: left_pad_audit(batch, tokenizer.pad_token_id),
    )
    discriminator_config = config.get("discriminator", {})
    architecture = str(discriminator_config.get("architecture", "linear"))
    representation = str(config.get("representation", {}).get("target", "lasttok"))
    layer = int(config.get("representation", {}).get("layer", -1))
    if representation == "meanpool" and architecture != "mlp":
        raise ValueError("Paper-aligned meanpool-targeted training requires an MLP discriminator")
    discriminator = build_discriminator(
        architecture,
        int(language_model.config.hidden_size),
        len(label_order),
        hidden_dim=int(discriminator_config.get("hidden_dim", 128)),
        dropout=float(discriminator_config.get("dropout", 0.3)),
    ).to(device=device, dtype=torch.float32)
    assert_trainable_isolation(language_model, discriminator)
    parameters = [
        parameter
        for parameter in (*language_model.parameters(), *discriminator.parameters())
        if parameter.requires_grad
    ]
    settings = config["training"]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(settings.get("learning_rate", 2e-4)),
        weight_decay=float(settings.get("weight_decay", 0.01)),
    )
    utility_batches, adversarial_batches = _infinite(utility_loader), _infinite(adversarial_loader)
    save_every = int(settings.get("save_every_steps", 200))
    language_model.train()
    discriminator.train()
    for optimizer_step in range(1, max_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        for _ in range(accumulation):
            utility = {key: value.to(device) for key, value in next(utility_batches).items()}
            adversarial = {key: value.to(device) for key, value in next(adversarial_batches).items()}
            utility_loss = language_model(**utility, use_cache=False).loss
            output = language_model(
                input_ids=adversarial["input_ids"],
                attention_mask=adversarial["attention_mask"],
                output_hidden_states=True,
                use_cache=False,
            )
            exported = extract_representation(
                output.hidden_states,
                adversarial["attention_mask"],
                representation,
                layer=layer,
            )
            logits = discriminator(gradient_reverse(exported.float(), lambda_adv))
            adversarial_loss = F.cross_entropy(logits, adversarial["sensitive_label"])
            ((utility_loss + adversarial_loss) / accumulation).backward()
        torch.nn.utils.clip_grad_norm_(parameters, float(settings.get("max_grad_norm", 1.0)))
        optimizer.step()
        if save_every > 0 and optimizer_step % save_every == 0:
            checkpoint = Path(output_dir) / f"step_{optimizer_step}"
            checkpoint.mkdir(parents=True, exist_ok=True)
            language_model.save_pretrained(checkpoint / "adapter")
            torch.save(discriminator.state_dict(), checkpoint / "training_discriminator.pt")
            metadata = {
                "method": "surfacelora",
                "optimizer_step": optimizer_step,
                "paired_micro_steps": optimizer_step * accumulation,
                "lambda_adv": float(lambda_adv),
                "representation": representation,
                "transformer_block": layer,
                "utility_adversarial_ratio": [1, 1],
                "gradient_accumulation_steps": accumulation,
                "prompt_identifier": PROMPT_IDENTIFIER,
                "maximum_prompt_length": int(config["tokenization"]["max_prompt_length"]),
                "maximum_sft_sequence_length": int(config["tokenization"]["max_sequence_length"]),
                "seed": seed,
            }
            (checkpoint / "checkpoint_metadata.json").write_text(
                json.dumps(metadata, indent=2), encoding="utf-8"
            )
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    language_model.save_pretrained(destination / "adapter_final")
    tokenizer.save_pretrained(destination / "adapter_final")
    torch.save(discriminator.state_dict(), destination / "training_discriminator_final.pt")
    report = {
        "method": "surfacelora",
        "optimizer_steps": max_steps,
        "paired_micro_steps": max_steps * accumulation,
        "gradient_accumulation_steps": accumulation,
        "effective_examples_per_branch_per_optimizer_step": batch_size * accumulation,
        "utility_adversarial_ratio": [1, 1],
        "lambda_adv": float(lambda_adv),
        "representation": representation,
        "layer": layer,
        "discriminator_architecture": architecture,
        "balanced_adversarial_counts": balanced_counts,
        "trained_components": ["lora_adapters", "training_time_discriminator"],
    }
    (destination / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


@torch.inference_mode()
def generate_summaries(
    model: Any,
    tokenizer: Any,
    sources: Sequence[str],
    *,
    max_prompt_length: int = 1024,
    max_new_tokens: int = 256,
    batch_size: int = 8,
    device: str = "cuda",
) -> list[str]:
    tokenizer.padding_side = "left"
    tokenizer.truncation_side = "left"
    summaries = []
    for start in range(0, len(sources), batch_size):
        prompts = [render_prompt(tokenizer, source) for source in sources[start : start + batch_size]]
        encoded = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_prompt_length,
            add_special_tokens=False,
        ).to(device)
        input_width = encoded["input_ids"].shape[1]
        generated = model.generate(
            **encoded,
            do_sample=False,
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )
        summaries.extend(
            tokenizer.decode(row[input_width:], skip_special_tokens=True).strip()
            for row in generated
        )
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    return summaries
