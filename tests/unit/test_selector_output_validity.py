"""Selector validity is an all-offered endpoint, independent of fallback report quality."""

from __future__ import annotations

from shapeflow_p1.evaluation.runner import selector_normalization_metrics


def _record(
    *,
    stage: str = "single",
    duplicate_count: int = 0,
    semantic_conflict_count: int = 0,
    rejected_reason: str | None = None,
) -> dict:
    return {
        "node": "H",
        "stage": stage,
        "selector_attempted": True,
        "normalization": {
            "raw_count": 1 + duplicate_count,
            "unique_count": 1,
            "duplicate_count": duplicate_count,
            "semantic_conflict_count": semantic_conflict_count,
            "rejected_reason": rejected_reason,
        },
        "failure": (
            {"reason": "CONTRACT", "detail": "bad selector output"}
            if rejected_reason is not None
            else None
        ),
    }


def test_out_of_set_id_is_a_hard_count_even_when_fallback_can_recover():
    record = _record(rejected_reason="out_of_set_label")
    record["fell_back"] = True
    summary = selector_normalization_metrics([record], node="H")
    assert summary["selector_attempt_count"] == 1
    assert summary["invalid_id_count"] == 1
    assert summary["strict_valid_count"] == 0
    assert summary["no_repair_adverse"] is True


def test_duplicate_is_repair_only_not_an_invalid_id():
    summary = selector_normalization_metrics(
        [_record(duplicate_count=1)], node="H")
    assert summary["repaired_attempt_count"] == 1
    assert summary["repair_rate"] == 1.0
    assert summary["invalid_id_count"] == 0
    assert summary["strict_valid_count"] == 0


def test_missing_normalization_trace_cannot_masquerade_as_strict_valid():
    record = _record()
    record["normalization"] = None
    summary = selector_normalization_metrics([record], node="H")
    assert summary["status"] == "INVALID_NORMALIZATION_TRACE"
    assert summary["normalization_missing_count"] == 1
    assert summary["strict_valid_count"] == 0
    assert summary["no_repair_adverse"] is True


def test_forged_derived_flag_is_rejected_and_stage_counts_remain_separate():
    local = _record(stage="local")
    local["normalization"]["strict_valid"] = True
    global_record = _record(stage="global")
    summary = selector_normalization_metrics([local, global_record], node="H")
    assert summary["normalization_invalid_count"] == 1
    assert summary["strict_valid_count"] == 1
    assert summary["by_stage"]["local"]["status"] == "INVALID_NORMALIZATION_TRACE"
    assert summary["by_stage"]["global"]["strict_valid_rate"] == 1.0
