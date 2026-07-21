"""Aggregate utility and representation-leakage metrics."""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Sequence

CHANCE_ACCURACY = 0.20
_WORD = re.compile(r"[A-Za-z0-9]+")


def leakage_gap(accuracy: float, chance_accuracy: float = CHANCE_ACCURACY) -> float:
    return abs(float(accuracy) - float(chance_accuracy))


def _tokens(text: str) -> list[str]:
    return _WORD.findall((text or "").lower())


def _f1(overlap: int, predicted: int, expected: int) -> float:
    if overlap == 0 or predicted == 0 or expected == 0:
        return 0.0
    precision = overlap / predicted
    recall = overlap / expected
    return 2.0 * precision * recall / (precision + recall)


def _rouge_n(prediction: str, reference: str, n: int) -> float:
    predicted_tokens, reference_tokens = _tokens(prediction), _tokens(reference)
    predicted = Counter(
        tuple(predicted_tokens[index : index + n])
        for index in range(max(0, len(predicted_tokens) - n + 1))
    )
    reference_counts = Counter(
        tuple(reference_tokens[index : index + n])
        for index in range(max(0, len(reference_tokens) - n + 1))
    )
    overlap = sum(min(count, reference_counts[item]) for item, count in predicted.items())
    return _f1(overlap, sum(predicted.values()), sum(reference_counts.values()))


def _lcs(left: Sequence[str], right: Sequence[str]) -> int:
    previous = [0] * (len(right) + 1)
    for left_value in left:
        current = [0]
        for index, right_value in enumerate(right, start=1):
            current.append(
                previous[index - 1] + 1
                if left_value == right_value
                else max(previous[index], current[-1])
            )
        previous = current
    return previous[-1]


def _rouge_l(prediction: str, reference: str) -> float:
    predicted, expected = _tokens(prediction), _tokens(reference)
    return _f1(_lcs(predicted, expected), len(predicted), len(expected))


def rouge_scores(predictions: Sequence[str], references: Sequence[str]) -> dict[str, float]:
    if not predictions or len(predictions) != len(references):
        raise ValueError("Predictions and references must be non-empty and aligned")
    count = len(predictions)
    return {
        "rouge1": 100.0 * sum(_rouge_n(p, r, 1) for p, r in zip(predictions, references)) / count,
        "rouge2": 100.0 * sum(_rouge_n(p, r, 2) for p, r in zip(predictions, references)) / count,
        "rougeL": 100.0 * sum(_rouge_l(p, r) for p, r in zip(predictions, references)) / count,
        "n": count,
    }

