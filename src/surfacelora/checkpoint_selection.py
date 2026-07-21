"""Validation-only, representation-specific checkpoint selection."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

from .metrics import leakage_gap

REQUIRED = {
    "checkpoint_identifier",
    "split",
    "representation",
    "layer",
    "lr_probe_accuracy",
    "rougeL",
}


def _normalize(record: Mapping[str, Any], chance_accuracy: float) -> dict[str, Any]:
    missing = REQUIRED - record.keys()
    if missing:
        raise ValueError(f"Checkpoint record is missing: {sorted(missing)}")
    if str(record["split"]).lower() not in {"validation", "val", "bal_validation", "bal_val"}:
        raise ValueError("Checkpoint selection accepts validation records only")
    lowered = {str(key).lower() for key in record}
    if any("test" in key for key in lowered):
        raise ValueError("Test metrics must not be present during checkpoint selection")
    forbidden_parts = (
        "label",
        "prediction",
        "row_key",
        "group_key",
        "subject",
        "patient",
        "admission",
        "encounter",
        "example",
        "text",
    )
    if any(any(part in key for part in forbidden_parts) for key in lowered):
        raise ValueError("Checkpoint selection accepts aggregate metrics only")
    if any(isinstance(value, (dict, list, tuple, set)) for value in record.values()):
        raise ValueError("Checkpoint selection accepts scalar aggregate metrics only")
    output = dict(record)
    output["checkpoint_identifier"] = str(record["checkpoint_identifier"])
    output["representation"] = str(record["representation"])
    output["layer"] = int(record["layer"])
    output["lr_probe_accuracy"] = float(record["lr_probe_accuracy"])
    output["rougeL"] = float(record["rougeL"])
    if not 0.0 <= output["lr_probe_accuracy"] <= 1.0:
        raise ValueError("lr_probe_accuracy must be between zero and one")
    if not 0.0 <= output["rougeL"] <= 100.0:
        raise ValueError("rougeL must use the 0-to-100 scale")
    output["step"] = int(record.get("step", 0))
    # Decimal accuracies read from JSON/CSV can produce asymmetric binary-float
    # artifacts around chance (for example, 0.16 and 0.24). Quantizing makes
    # deterministic tie-breaking reflect the reported metric values.
    output["lr_leakage_gap"] = round(
        leakage_gap(output["lr_probe_accuracy"], chance_accuracy), 12
    )
    return output


def _pareto(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front = []
    for candidate in records:
        dominated = any(
            other is not candidate
            and other["lr_leakage_gap"] <= candidate["lr_leakage_gap"]
            and other["rougeL"] >= candidate["rougeL"]
            and (
                other["lr_leakage_gap"] < candidate["lr_leakage_gap"]
                or other["rougeL"] > candidate["rougeL"]
            )
            for other in records
        )
        if not dominated:
            front.append(candidate)
    return sorted(
        front,
        key=lambda row: (
            row["lr_leakage_gap"],
            -row["rougeL"],
            row["step"],
            str(row["checkpoint_identifier"]),
        ),
    )


def select_checkpoint(
    records: Iterable[Mapping[str, Any]],
    *,
    leakage_budget: float = 0.025,
    chance_accuracy: float = 0.20,
) -> dict[str, Any]:
    if leakage_budget < 0 or not 0.0 <= chance_accuracy <= 1.0:
        raise ValueError("Invalid leakage budget or chance accuracy")
    normalized = [_normalize(record, chance_accuracy) for record in records]
    if not normalized:
        raise ValueError("No validation metrics were provided")
    if len({row["representation"] for row in normalized}) != 1:
        raise ValueError("Select checkpoints for one representation at a time")
    if len({row["layer"] for row in normalized}) != 1:
        raise ValueError("Select checkpoints for one transformer layer at a time")
    feasible = [row for row in normalized if row["lr_leakage_gap"] <= leakage_budget]
    if feasible:
        selected = min(
            feasible,
            key=lambda row: (
                -row["rougeL"],
                row["lr_leakage_gap"],
                row["step"],
                str(row["checkpoint_identifier"]),
            ),
        )
        rule = "highest_validation_rougeL_within_lr_budget"
    else:
        selected = _pareto(normalized)[0]
        rule = "deterministic_validation_pareto_fallback"
    return {
        **selected,
        "selection_rule": rule,
        "leakage_budget": float(leakage_budget),
        "chance_accuracy": float(chance_accuracy),
        "used_test_metrics": False,
        "used_mlp_for_selection": False,
    }


def load_metric_records(directory: str | Path) -> list[dict[str, Any]]:
    root = Path(directory)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        records.extend(value if isinstance(value, list) else [value])
    for path in sorted(root.rglob("*.csv")):
        with path.open("r", encoding="utf-8", newline="") as handle:
            records.extend(dict(row) for row in csv.DictReader(handle))
    return records
