"""SHORT_PROSE traces distinguish policy-valid budgeting from raw-contract adherence."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace

import pytest

from shapeflow_p1.campaign.evaluate import _prose_control_records
from shapeflow_p1.campaign.graph_driver import TrajectoryRecorder
from shapeflow_p1.evaluation.runner import (
    ArmOutput,
    EvaluationError,
    prose_control_normalization_metrics,
    score_task,
)
from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer
from shapeflow_p1.strategies.prose import _render_page_prose
from shapeflow_p1.odr.vendor_hooks import (
    RunBinding,
    _emit_selection_records,
    bind_run,
)

TOK = WhitespaceTokenizer()


def _record(text: str, *, budget: int = 16) -> dict:
    rendered, _body, normalization = _render_page_prose(
        text,
        title="Frozen title",
        url="https://source.example/page",
        source_number=1,
        tokenizer=TOK,
        budget=budget,
    )
    return {
        "node": "H",
        "candidate_view_sha256": "a" * 64,
        "offered_span_ids": ["span-1"],
        "offered_source_occurrence_ids": ["occ-1"],
        "completion_token_cap": 4,
        "selector_attempted": True,
        "attempt_status": "OUTPUT_OBSERVED",
        "normalization_trace_status": "OK",
        "publication_status": "PUBLISHED",
        "batch_accepted": True,
        "work_incomplete": False,
        "rendered_tokens": TOK.count(rendered),
        "normalization": normalization,
    }


def test_budget_prefix_is_policy_valid_but_raw_contract_nonadherent() -> None:
    summary = prose_control_normalization_metrics(
        [_record("one two three four five six seven eight nine ten", budget=13)],
        expected_nodes=("H",),
    )

    assert summary["status"] == "OK"
    assert summary["truncated_attempt_count"] == 1
    assert summary["semantic_repair_count"] == 0
    assert summary["published_policy_adverse"] is False
    assert summary["raw_contract_adverse"] is True


def test_invalid_raw_suffix_outside_independent_budget_is_not_semantic_repair() -> None:
    record = _record(
        "one two three four five six seven eight\nURL: injected",
        # Source-definition injection begins a new line, matching the actual publisher format.
        budget=13,
    )
    normalization = record["normalization"]

    assert normalization["published_policy_valid"] is True
    assert normalization["invalid_raw_suffix_outside_policy_output"] is True
    assert normalization["semantic_repair_applied"] is False
    summary = prose_control_normalization_metrics(
        [record], expected_nodes=("H",)
    )
    assert summary["published_policy_adverse"] is False
    assert summary["raw_contract_adverse"] is True


def test_tampered_derived_flag_invalidates_the_trace() -> None:
    record = _record("one two", budget=20)
    tampered = deepcopy(record)
    tampered["normalization"]["raw_contract_adherent"] = False

    summary = prose_control_normalization_metrics(
        [tampered], expected_nodes=("H",)
    )
    assert summary["status"] == "INVALID_NORMALIZATION_TRACE"
    assert summary["trace_complete"] is False


def test_call_failure_is_explicit_not_a_missing_normalization_trace() -> None:
    record = {
        "node": "C",
        "candidate_view_sha256": "b" * 64,
        "offered_span_ids": ["span-1"],
        "offered_source_occurrence_ids": ["occ-1"],
        "completion_token_cap": 4,
        "selector_attempted": True,
        "attempt_status": "CALL_FAILED",
        "normalization_trace_status": "EXPLICIT_NO_OUTPUT",
        "publication_status": "FALLBACK_DISCARDED",
        "batch_accepted": False,
        "work_incomplete": True,
        "normalization": None,
    }

    summary = prose_control_normalization_metrics(
        [record], expected_nodes=("C",)
    )
    assert summary["status"] == "OK"
    assert summary["trace_complete"] is True
    assert summary["call_failed_count"] == 1
    assert summary["published_policy_adverse"] is True
    assert summary["raw_contract_adverse"] is True


def test_one_joint_node_cannot_substitute_for_the_other() -> None:
    summary = prose_control_normalization_metrics(
        [_record("one two", budget=20)],
        expected_nodes=("H", "C"),
    )

    assert summary["status"] == "INVALID_NORMALIZATION_TRACE"
    assert summary["by_node"]["C"]["status"] == "NOT_APPLICABLE_NO_PROSE_ATTEMPT"
    assert summary["missing_expected_nodes"] == ["C"]
    assert summary["attempt_count"] == 1


def test_a_node_the_graph_never_presented_is_inapplicable_not_corrupt() -> None:
    """Total absence and partial absence are different facts.

    Every zero-attempt node used to be counted as a trace error, which made the
    NOT_APPLICABLE_NO_PROSE_ATTEMPT status unreachable and reported a structurally
    inapplicable arm as a *corrupt* trace. The refusal is right either way -- the downstream
    integrity gate still declines to establish the contrast -- but the stated reason was
    wrong, and a wrong reason is what someone reads when deciding whether to trust the run.
    """
    summary = prose_control_normalization_metrics([], expected_nodes=("H", "C"))

    assert summary["status"] == "NOT_APPLICABLE_NO_PROSE_ATTEMPT"
    assert summary["normalization_trace_error_count"] == 0
    assert summary["attempt_count"] == 0
    assert summary["missing_expected_nodes"] == ["C", "H"]
    assert summary["by_node"]["H"]["status"] == "NOT_APPLICABLE_NO_PROSE_ATTEMPT"


def test_an_unexpected_node_is_still_a_corrupt_trace_even_with_no_attempts() -> None:
    """Inapplicability must not become a way to launder a node that should not be there."""
    summary = prose_control_normalization_metrics(
        [_record("one two", budget=20)], expected_nodes=("C",)
    )

    assert summary["status"] == "INVALID_NORMALIZATION_TRACE"
    assert summary["unexpected_nodes"] == ["H"]


def test_frozen_event_records_flow_into_the_score_row() -> None:
    record = _record("one two", budget=20)
    frozen = _prose_control_records([
        {
            "kind": "PROSE_CONTROL_OUTPUT",
            "event_index": 7,
            "node": "H",
            "checkpoint": "c" * 64,
            "control_records": [record],
        }
    ])
    score = score_task(
        truth_body={
            "content_sha256": "d" * 64,
            "required_facets": [],
            "atomic_evidence": [],
            "contradiction_pairs": [],
        },
        outputs=[
            ArmOutput(
                task_id="task",
                arm_id="H_PROSE_CONTROL",
                variant_id="H00-PROSE",
                replicate_id="0",
                final_report="",
                frozen=True,
                prose_control_records=frozen,
                prose_control_expected_nodes=("H",),
            )
        ],
        atom_texts={},
        judge_relation=lambda _claim, _fact: "unrelated",
        citation_supports=lambda _citation, _claim: False,
        judge_policy_sha256="e" * 64,
        claim_scope="FORMATIVE",
    )

    summary = score.per_arm["H_PROSE_CONTROL:0"][
        "prose_control_normalization"
    ]
    assert summary["status"] == "OK"
    assert summary["attempt_count"] == 1
    assert frozen[0]["checkpoint_hash"] == "c" * 64
    assert frozen[0]["event_index"] == 7


def test_event_node_tampering_fails_before_scoring() -> None:
    record = {**_record("one two", budget=20), "node": "C"}
    with pytest.raises(EvaluationError, match="changes its node"):
        _prose_control_records([
            {
                "kind": "PROSE_CONTROL_OUTPUT",
                "node": "H",
                "checkpoint": "c" * 64,
                "control_records": [record],
            }
        ])


def test_failed_atomic_prose_batch_emits_attempts_not_synthetic_selector_records() -> None:
    record = {
        **_record("one two", budget=20),
        "publication_status": "FALLBACK_DISCARDED",
        "batch_accepted": False,
    }
    strategy = SimpleNamespace(
        last_outcomes=(),
        last_control_records=(record,),
        last_rendered_token_counts=(),
        last_published_token_counts=(),
    )
    recorder = TrajectoryRecorder(treatment_node="H")
    binding = RunBinding(
        task_id="task",
        researcher_id="researcher",
        attempt_id="attempt",
        task_ctx=SimpleNamespace(),
        on_event=recorder.record,
    )
    with bind_run(binding):
        _emit_selection_records(
            strategy,
            node="H",
            checkpoint="c" * 64,
            fell_back=True,
            batch_failure=SimpleNamespace(reason="PROSE_CONTROL_ERROR", detail="bad"),
        )

    assert [event["kind"] for event in recorder.events] == [
        "PROSE_CONTROL_OUTPUT"
    ]
    assert recorder.events[0]["batch_failure"]["reason"] == "PROSE_CONTROL_ERROR"
    assert recorder.events[0]["position"] == "TREATMENT"
    assert recorder.direct_node_records == []
