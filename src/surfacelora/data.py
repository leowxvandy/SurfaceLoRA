"""Local preprocessing, patient-group splitting, and balanced audit subsets."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

RACE_LABELS = ("WHITE", "BLACK", "HISPANIC", "ASIAN", "OTHER")
_UNKNOWN = ("UNKNOWN", "DECLINED", "UNABLE")
_REPORT_BLOCK = re.compile(
    r"^[ \t]*(?:IMAGING|MICROBIOLOGY|(?:DISCHARGE\s+)?LABS|FINAL REPORT|RESULTS|DATA)"
    r"(?:[ \t]*:|[ \t]*$)",
    re.IGNORECASE | re.MULTILINE,
)
_STRONG_INSTRUCTION = re.compile(
    r"(?:\bDISCHARGE INSTRUCTIONS\b|\bFOLLOWUP INSTRUCTIONS\b|"
    r"\bDear\s+(?:Ms\.|Mr\.|Mrs\.)|\bSincerely\b|\bIt was a pleasure\b|"
    r"\bPlease call\b|\bReturn to the emergency room\b|\bcall 911\b|"
    r"\bYour\s+[^\n]{0,80}\s+Team\b)",
    re.IGNORECASE,
)
_SECOND_PERSON = re.compile(
    r"\b(?:You were admitted|You presented|We treated you)\b", re.IGNORECASE
)
_ACTION = re.compile(
    r"\b(?:please|appointment|emergency|return|call|take)\b", re.IGNORECASE
)


def _missing(value: Any) -> bool:
    return value is None or str(value).strip().upper() in {"", "NAN", "NONE", "<NA>", "NAT"}


def map_race(value: Any) -> str | None:
    """Map an EHR-recorded administrative value into the five audit classes."""
    if _missing(value):
        return None
    normalized = str(value).strip().upper()
    if any(marker in normalized for marker in _UNKNOWN):
        return None
    if "WHITE" in normalized or "PORTUGUESE" in normalized:
        return "WHITE"
    if "BLACK" in normalized or "AFRICAN" in normalized:
        return "BLACK"
    if "HISPANIC" in normalized or "LATINO" in normalized:
        return "HISPANIC"
    if "ASIAN" in normalized:
        return "ASIAN"
    return "OTHER"


def clean_source(text: Any, max_characters: int = 4000) -> str:
    if not isinstance(text, str):
        return ""
    match = _REPORT_BLOCK.search(text)
    value = text[: match.start()] if match else text
    return re.sub(r"\s+", " ", value[:max_characters]).strip()


def clean_target(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text).strip()


def resembles_instructions(text: str) -> bool:
    return bool(_STRONG_INSTRUCTION.search(text) or (_SECOND_PERSON.search(text) and _ACTION.search(text)))


def prepare_records(
    rows: Sequence[Mapping[str, Any]],
    *,
    fields: Mapping[str, str],
    min_source_characters: int = 200,
    min_target_characters: int = 80,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for index, row in enumerate(rows):
        group = row.get(fields["group"])
        race = map_race(row.get(fields["sensitive_label"]))
        source = clean_source(row.get(fields["source"]))
        target = clean_target(row.get(fields["target"]))
        if _missing(group):
            counts["missing_group"] += 1
        elif race is None:
            counts["unknown_label"] += 1
        elif len(source) < min_source_characters:
            counts["source_too_short"] += 1
        elif len(target) < min_target_characters:
            counts["target_too_short"] += 1
        elif resembles_instructions(target):
            counts["instruction_like_target"] += 1
        else:
            kept.append(
                {
                    "_source_index": index,
                    "_group": str(group),
                    "source": source,
                    "target": target,
                    "sensitive_label": race,
                }
            )
            counts["kept"] += 1
    return kept, dict(counts)


def patient_group_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    seed: int = 42,
    test_fraction: float = 0.10,
    validation_fraction_of_remainder: float = 0.10,
) -> dict[str, list[dict[str, Any]]]:
    from sklearn.model_selection import GroupShuffleSplit

    if len({str(row["_group"]) for row in rows}) < 3:
        raise ValueError("At least three patient groups are required")
    indices = np.arange(len(rows))
    groups = np.asarray([str(row["_group"]) for row in rows])
    train_pool, test = next(
        GroupShuffleSplit(n_splits=1, test_size=test_fraction, random_state=seed).split(
            indices, groups=groups
        )
    )
    train_relative, validation_relative = next(
        GroupShuffleSplit(
            n_splits=1,
            test_size=validation_fraction_of_remainder,
            random_state=seed,
        ).split(train_pool, groups=groups[train_pool])
    )
    assignments = {
        "train": set(int(value) for value in train_pool[train_relative]),
        "validation": set(int(value) for value in train_pool[validation_relative]),
        "test": set(int(value) for value in test),
    }
    output = {
        name: [dict(rows[index]) for index in range(len(rows)) if index in selected]
        for name, selected in assignments.items()
    }
    verify_patient_disjoint(output)
    return output


def verify_patient_disjoint(splits: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    groups = {name: {str(row["_group"]) for row in rows} for name, rows in splits.items()}
    names = sorted(groups)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if groups[left] & groups[right]:
                raise ValueError(f"Patient-group overlap between {left} and {right}")


def balanced_sample(
    rows: Sequence[Mapping[str, Any]],
    *,
    per_class: int,
    label_order: Sequence[str] = RACE_LABELS,
    seed: int = 42,
) -> list[dict[str, Any]]:
    if per_class <= 0:
        raise ValueError("per_class must be positive")
    groups = {label: [] for label in label_order}
    for row in rows:
        label = str(row["sensitive_label"])
        if label in groups:
            groups[label].append(dict(row))
    insufficient = {label: len(values) for label, values in groups.items() if len(values) < per_class}
    if insufficient:
        raise ValueError(f"Insufficient examples for balanced audit subset: {insufficient}")
    selected: list[dict[str, Any]] = []
    for label in label_order:
        indices = np.random.RandomState(seed).choice(len(groups[label]), per_class, replace=False)
        selected.extend(groups[label][int(index)] for index in indices)
    order = np.random.RandomState(seed).permutation(len(selected))
    return [selected[int(index)] for index in order]


def _local_key(value: Any, salt: str) -> str:
    return hashlib.sha256(f"{salt}\0{value}".encode("utf-8")).hexdigest()


def make_runtime_splits(
    splits: Mapping[str, Sequence[Mapping[str, Any]]], *, salt: str
) -> dict[str, list[dict[str, Any]]]:
    output: dict[str, list[dict[str, Any]]] = {}
    for split, rows in splits.items():
        output[split] = []
        for example_index, row in enumerate(rows):
            output[split].append(
                {
                    "example_index": example_index,
                    "row_key": _local_key(row["_source_index"], salt),
                    "group_key": _local_key(row["_group"], salt),
                    "split": split,
                    "source": row["source"],
                    "target": row["target"],
                    "sensitive_label": row["sensitive_label"],
                }
            )
    return output
