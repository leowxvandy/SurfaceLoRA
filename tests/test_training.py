import numpy as np
import pytest
import torch
from torch import nn

from surfacelora.data import balanced_sample, clean_source, patient_group_split, verify_patient_disjoint
from surfacelora.modeling import TrainingDiscriminator, assert_trainable_isolation
from surfacelora.probes import MLPProbe, probe_optimizer
from surfacelora.training import (
    _assert_globally_balanced,
    encode_sft_example,
    generate_summaries,
    left_pad_sft,
)


class TinyTokenizer:
    eos_token_id = 99

    def __call__(self, text, add_special_tokens=False):
        del add_special_tokens
        return {"input_ids": [ord(character) % 50 for character in text]}


class TinyLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.frozen_backbone = nn.Linear(2, 2)
        self.lora_adapter = nn.Linear(2, 2, bias=False)
        for parameter in self.frozen_backbone.parameters():
            parameter.requires_grad = False


class TinyBatch(dict):
    def to(self, device):
        del device
        return self


class GenerationTokenizer:
    pad_token_id = 0
    eos_token_id = 9

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        del tokenize, add_generation_prompt
        return messages[-1]["content"]

    def __call__(self, prompts, **kwargs):
        del prompts, kwargs
        return TinyBatch(
            input_ids=torch.tensor([[0, 1, 2], [3, 4, 5]]),
            attention_mask=torch.tensor([[0, 1, 1], [1, 1, 1]]),
        )

    def decode(self, token_ids, skip_special_tokens=True):
        del skip_special_tokens
        return " ".join(str(int(value)) for value in token_ids)


class GenerationModel:
    def generate(self, input_ids, attention_mask, **kwargs):
        del attention_mask, kwargs
        continuation = torch.tensor([[7, 8], [6, 8]])
        return torch.cat((input_ids, continuation), dim=1)


def test_prompt_tokens_are_loss_masked_after_left_truncation():
    encoded = encode_sft_example(TinyTokenizer(), "abcdef", "xy", max_length=5)
    assert encoded["input_ids"][-3:] == [20, 21, 99]
    assert encoded["labels"][-3:] == [20, 21, 99]
    assert encoded["labels"][:2] == [-100, -100]
    batch = left_pad_sft([encoded, {"input_ids": [1], "labels": [1]}], pad_token_id=0)
    assert batch["labels"][1, :-1].tolist() == [-100] * 4


def test_generation_decodes_only_tokens_after_padded_input_width():
    summaries = generate_summaries(
        GenerationModel(),
        GenerationTokenizer(),
        ["synthetic one", "synthetic two"],
        device="cpu",
    )
    assert summaries == ["7 8", "6 8"]


def test_only_lora_and_training_discriminator_are_trainable():
    model = TinyLanguageModel()
    discriminator = TrainingDiscriminator(2, 5)
    assert_trainable_isolation(model, discriminator)
    trainable = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    assert trainable == {"lora_adapter.weight"}


def test_posthoc_probe_optimizer_is_isolated_from_language_model():
    model = TinyLanguageModel()
    probe = MLPProbe(2, 5)
    optimizer = probe_optimizer(probe)
    optimized = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized == {id(parameter) for parameter in probe.parameters()}
    assert optimized.isdisjoint({id(parameter) for parameter in model.parameters()})


def test_patient_split_and_balancing_are_deterministic():
    rows = [
        {"_group": f"g-{group}", "sensitive_label": label, "value": group}
        for group in range(20)
        for label in ("A", "B")
    ]
    split_one = patient_group_split(rows, seed=7, test_fraction=0.2, validation_fraction_of_remainder=0.25)
    split_two = patient_group_split(rows, seed=7, test_fraction=0.2, validation_fraction_of_remainder=0.25)
    verify_patient_disjoint(split_one)
    assert split_one == split_two
    balanced_one = balanced_sample(rows, per_class=4, label_order=("A", "B"), seed=9)
    balanced_two = balanced_sample(rows, per_class=4, label_order=("A", "B"), seed=9)
    assert balanced_one == balanced_two
    assert _assert_globally_balanced(balanced_one, ("A", "B")) == {"A": 4, "B": 4}
    with pytest.raises(ValueError, match="Insufficient"):
        balanced_sample(rows, per_class=21, label_order=("A", "B"), seed=9)


def test_report_header_cleaning_does_not_truncate_narrative_word():
    text = "The labs were reassuring and treatment continued.\nIMAGING:\nprivate report tail"
    assert clean_source(text) == "The labs were reassuring and treatment continued."
