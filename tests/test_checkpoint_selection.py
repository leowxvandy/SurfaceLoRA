import pytest

from surfacelora.checkpoint_selection import select_checkpoint


def row(checkpoint, accuracy, rouge, step):
    return {
        "checkpoint_identifier": checkpoint,
        "split": "validation",
        "representation": "lasttok",
        "layer": -1,
        "lr_probe_accuracy": accuracy,
        "rougeL": rouge,
        "step": step,
    }


def test_selects_highest_validation_rouge_within_budget():
    selected = select_checkpoint(
        [row("a", 0.22, 0.31, 100), row("b", 0.21, 0.34, 200), row("c", 0.30, 0.40, 300)]
    )
    assert selected["checkpoint_identifier"] == "b"
    assert selected["used_test_metrics"] is False
    assert selected["used_mlp_for_selection"] is False


def test_deterministic_pareto_fallback_prefers_lowest_gap_then_tiebreaks():
    records = [row("later", 0.24, 0.30, 200), row("early", 0.16, 0.30, 100)]
    selected = select_checkpoint(records, leakage_budget=0.01)
    assert selected["checkpoint_identifier"] == "early"
    assert selected["selection_rule"] == "deterministic_validation_pareto_fallback"


@pytest.mark.parametrize(
    "extra",
    [
        {"test_accuracy": 0.9},
        {"labels": [0, 1]},
        {"predictions": [0, 1]},
        {"row_keys": ["x"]},
    ],
)
def test_rejects_test_or_row_level_fields(extra):
    with pytest.raises(ValueError):
        select_checkpoint([{**row("a", 0.2, 0.3, 1), **extra}])


def test_rejects_test_split_and_mixed_representations():
    with pytest.raises(ValueError, match="validation"):
        select_checkpoint([{**row("a", 0.2, 0.3, 1), "split": "test"}])
    with pytest.raises(ValueError, match="one representation"):
        select_checkpoint([row("a", 0.2, 0.3, 1), {**row("b", 0.2, 0.3, 2), "representation": "meanpool"}])
    with pytest.raises(ValueError, match="one transformer layer"):
        select_checkpoint([row("a", 0.2, 0.3, 1), {**row("b", 0.2, 0.3, 2), "layer": -2}])
