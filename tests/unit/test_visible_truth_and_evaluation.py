"""The C_VISIBLE denominator, and the rules that keep truth out of the treatment path.

The projection decides what the compressor *could* have kept. Getting its denominator wrong is
the most likely way to manufacture a false C_VISIBLE failure: penalise the reducer for an atom
that an upstream page summary already dropped, and the number reads as "the compressor lost it".
"""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

from shapeflow_p1.evaluation.runner import (
    ArmOutput,
    EvaluationError,
    TaskScore,
    judge_policy_digest,
    score_direct_node_records,
    score_task,
    treatment_import_violations,
    write_scores,
)
from shapeflow_p1.evaluation.visible_truth_projection import (
    AMBIGUOUS,
    EXPLICITLY_VISIBLE,
    NOT_VISIBLE,
    project_visible_truth,
    visible_recall,
)


def _spans():
    return [
        {"visible_span_id": "v1", "kind": "TOOL_EVIDENCE"},
        {"visible_span_id": "v2", "kind": "TOOL_EVIDENCE"},
        {"visible_span_id": "v3", "kind": "MODEL_DERIVED_CONTEXT"},
    ]


def _texts():
    return {
        "v1": "The Harbour Authority reported 4821 units in 2025.",
        "v2": "Unrelated background about weather patterns.",
        "v3": "I should check whether the Meadow Board reported 9310 units in 2025.",
    }


# --- the denominator -------------------------------------------------------------------------


def test_an_atom_present_in_tool_output_is_explicitly_visible():
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a1", "Harbour Authority reported 4821 units in 2025")],
    )
    assert projection.projected_atoms[0].projection_status == EXPLICITLY_VISIBLE
    assert projection.projected_atoms[0].supporting_visible_span_ids == ("v1",)


def test_an_atom_absent_from_the_compressor_input_is_not_visible():
    """The reducer must not be charged for what an upstream summary already dropped."""
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a2", "Citadel Registry recorded 7788 filings in 2024")],
    )
    assert projection.projected_atoms[0].projection_status == NOT_VISIBLE
    assert projection.explicitly_visible_ids == ()


def test_model_reasoning_can_never_make_an_atom_visible():
    """The model asserting a fact is not the source saying it.

    v3 mentions the Meadow Board figure verbatim, but it is the researcher thinking aloud. If
    that counted, the compressor would be credited for information nothing in its input
    established -- which is the C_REGISTRY treatment, and a different experiment.
    """
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a3", "Meadow Board reported 9310 units in 2025")],
    )
    atom = projection.projected_atoms[0]
    assert atom.projection_status != EXPLICITLY_VISIBLE
    assert atom.supporting_visible_span_ids == (), "a model-derived span supported an atom"
    assert projection.explicitly_visible_ids == ()
    assert projection.notes["model_derived_spans"] == 1


def test_a_partial_match_is_ambiguous_rather_than_visible():
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a4", "Harbour Authority reported 9999 units in 2031")],
    )
    assert projection.projected_atoms[0].projection_status == AMBIGUOUS


def test_recall_counts_only_the_explicitly_visible_atoms():
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a1", "Harbour Authority reported 4821 units in 2025"),
               ("a2", "Citadel Registry recorded 7788 filings in 2024")],
    )
    assert visible_recall(projection, retained_atom_ids=["a1"]) == 1.0
    assert visible_recall(projection, retained_atom_ids=[]) == 0.0


def test_recall_with_nothing_visible_is_not_a_zero():
    """An empty denominator has no answer; reporting 0.0 records a failure nothing could avoid."""
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a2", "Citadel Registry recorded 7788 filings in 2024")],
    )
    assert visible_recall(projection, retained_atom_ids=[]) is None


def test_the_projection_is_content_addressed():
    projection = project_visible_truth(
        checkpoint_hash="ck1", visible_view_hash="vh1", spans=_spans(), span_texts=_texts(),
        atoms=[("a1", "Harbour Authority reported 4821 units in 2025")],
    )
    body = projection.to_json()
    assert len(body["content_sha256"]) == 64
    assert body["audit_status"] == "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT"


# --- the evaluation runner ----------------------------------------------------------------------


_TRUTH = {
    "task_id": "T1",
    "required_facets": ["f1"],
    "atomic_evidence": [
        {"atom_id": "a1", "facet_id": "f1", "weight": 1.0, "critical": False,
         "known_unresolved": False, "supporting_span_ids": ["s1"]},
    ],
    "contradiction_pairs": [],
    "content_sha256": "d" * 64,
}
_ATOM_TEXTS = {"a1": "The Harbour Authority reported 4821 units in 2025."}


def _output(arm="P0", frozen=True, report=None):
    return ArmOutput(
        task_id="T1", arm_id=arm, variant_id=arm, replicate_id="0",
        final_report=report if report is not None else (
            "The Harbour Authority reported 4821 units in 2025 [1].\n\n[1] https://a.example/doc"
        ),
        frozen=frozen,
    )


def _entails(claim: str, atom: str) -> str:
    return "entail" if "4821" in claim and "4821" in atom else "unrelated"


def test_scoring_refuses_an_output_that_is_not_frozen():
    with pytest.raises(EvaluationError, match="not frozen"):
        score_task(
            truth_body=_TRUTH, outputs=[_output(frozen=False)], atom_texts=_ATOM_TEXTS,
            judge_relation=_entails, citation_supports=lambda claim, link: True,
            judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
        )


def test_every_arm_is_scored_against_one_packet_and_one_policy():
    score = score_task(
        truth_body=_TRUTH,
        outputs=[_output("P0"), _output("H02"), _output("C01")],
        atom_texts=_ATOM_TEXTS, judge_relation=_entails,
        citation_supports=lambda claim, link: True,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
    )
    assert set(score.per_arm) == {"P0:0", "H02:0", "C01:0"}
    assert {v["claim_scope"] for v in score.per_arm.values()} == {"FORMATIVE_ONLY"}
    assert score.truth_packet_sha256 == "d" * 64
    assert score.judge_policy_sha256 == "p" * 64


def test_direct_node_recall_uses_published_ids_not_final_report():
    truth = {
        **_TRUTH,
        "atomic_evidence": [{
            **_TRUTH["atomic_evidence"][0],
            "supporting_span_ids": ["s1"],
        }],
    }
    metrics = score_direct_node_records(truth, [{
        "node": "H", "checkpoint_hash": "ck",
        "offered_span_ids": ["s1", "s2"],
        "selected_span_ids": ["s2"],
        "published_span_ids": ["s2"],
        "offered_source_occurrence_ids": ["o1"],
        "chunker": "markdown_structure_v1",
        "contract": "P1_ID", "aggregation": "stable_union_v1",
    }], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"], "s2": ["o1"]}},
    })
    assert metrics["status"] == "OK"
    assert metrics["exact_truth_recall"] == 0.0
    assert metrics["published_truth_atom_ids"] == []


def test_direct_node_metrics_measure_selected_token_precision_and_efficiency():
    metrics = score_direct_node_records(_TRUTH, [{
        "node": "H",
        "checkpoint_hash": "ck",
        "stage": "single",
        "chunker": "markdown_structure_v1",
        "contract": "P1_ID",
        "aggregation": "stable_union_v1",
        "offered_span_ids": ["s1", "s2"],
        "selected_span_ids": ["s1", "s2"],
        "published_span_ids": ["s1", "s2"],
        "offered_source_occurrence_ids": ["o1"],
        "offered_span_token_counts": [["s1", 10], ["s2", 30]],
        "offered_evidence_tokens": 40,
        "staged_rendered_tokens": 50,
        "published_rendered_tokens": 50,
    }], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"], "s2": ["o1"]},
        },
    })

    h = metrics["by_node"]["H"]
    assert h["token_trace_complete"] is True
    assert h["selected_token_precision"] == pytest.approx(0.25)
    assert h["weighted_truth_per_100_rendered_tokens"] == pytest.approx(2.0)
    assert h["materialization_ratio"] == pytest.approx(1.25)


def test_c_visible_token_metrics_never_count_context_as_evidence():
    from shapeflow_p1.campaign.evaluate import _evidence_context_token_partition

    offered_evidence, offered_context, published_evidence, published_context = (
        _evidence_context_token_partition(
            span_kind_by_id={
                "tool": "TOOL_EVIDENCE",
                "model": "MODEL_DERIVED_CONTEXT",
                "user": "USER_CONTEXT",
            },
            span_token_counts={"tool": 10, "model": 30, "user": 20},
            published_span_ids={"tool", "model"},
        )
    )
    assert offered_evidence == 10
    assert offered_context == 50
    assert published_evidence == 10
    assert published_context == 30


def test_direct_node_token_trace_rejects_totals_that_do_not_match_span_counts():
    metrics = score_direct_node_records(_TRUTH, [{
        "node": "H",
        "checkpoint_hash": "ck",
        "stage": "single",
        "chunker": "markdown_structure_v1",
        "offered_span_ids": ["s1"],
        "selected_span_ids": ["s1"],
        "published_span_ids": ["s1"],
        "offered_source_occurrence_ids": ["o1"],
        "offered_span_token_counts": [["s1", 10]],
        "offered_evidence_tokens": 999,
        "staged_rendered_tokens": 12,
        "published_rendered_tokens": 12,
    }], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]},
        },
    })
    assert metrics["status"] == "INVALID_DIRECT_TRACE"
    assert any("per-span sum" in error for error in metrics["errors"])


def test_direct_node_trace_rejects_ids_outside_the_offered_set():
    metrics = score_direct_node_records(_TRUTH, [{
        "node": "H", "checkpoint_hash": "ck", "offered_span_ids": ["s1"],
        "selected_span_ids": ["forged"], "published_span_ids": ["forged"],
        "chunker": "markdown_structure_v1",
    }], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
    })
    assert metrics["status"] == "INVALID_DIRECT_TRACE"


def test_direct_denominator_is_unavailable_without_cross_chunker_support_index():
    metrics = score_direct_node_records(_TRUTH, [{
        "node": "H", "checkpoint_hash": "ck",
        "chunker": "fixed_token_v1", "offered_span_ids": ["other-id"],
        "selected_span_ids": [], "published_span_ids": [],
        "offered_source_occurrence_ids": ["o1"],
    }])
    assert metrics["direct_denominator_status"].startswith(
        "DIRECT_DENOMINATOR_UNAVAILABLE")
    assert metrics["exact_truth_recall"] is None
    assert not metrics["selector_guards_complete"]


def test_bad_chunker_lowers_candidate_coverage_instead_of_erasing_the_denominator():
    """A fact in the source occurrence remains owed even when no candidate chunk contains it."""
    metrics = score_direct_node_records(_TRUTH, [{
        "node": "H",
        "checkpoint_hash": "ck",
        "chunker": "bad_split_v1",
        "offered_span_ids": ["left-half", "right-half"],
        "selected_span_ids": [],
        "published_span_ids": [],
        "offered_source_occurrence_ids": ["o1"],
    }], atom_support_index={
        "chunkers": {"bad_split_v1": {"a1": []}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "bad_split_v1": {
                "left-half": ["o1"],
                "right-half": ["o1"],
            },
        },
    })
    h = metrics["by_node"]["H"]
    assert h["status"] == "OK"
    assert h["prechunk_truth_atom_ids"] == ["a1"]
    assert h["candidate_covered_truth_atom_ids"] == []
    assert h["candidate_coverage"] == 0.0
    assert h["total_published_prechunk_recall"] == 0.0


def test_incomplete_support_index_cannot_make_a_truth_atom_disappear():
    truth = {
        **_TRUTH,
        "atomic_evidence": [
            _TRUTH["atomic_evidence"][0],
            {
                "atom_id": "a2",
                "facet_id": "f1",
                "weight": 1.0,
                "critical": False,
                "known_unresolved": False,
                "supporting_span_ids": ["s2"],
            },
        ],
    }
    metrics = score_direct_node_records(truth, [{
        "node": "H",
        "chunker": "markdown_structure_v1",
        "offered_span_ids": ["s1"],
        "selected_span_ids": ["s1"],
        "published_span_ids": ["s1"],
        "offered_source_occurrence_ids": ["o1"],
    }], atom_support_index={
        # a2 is absent from both atom-index maps: this is a corrupt denominator, not 100% recall.
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]},
        },
    })
    assert metrics["by_node"]["H"]["status"] == "DIRECT_DENOMINATOR_UNAVAILABLE"
    assert metrics["by_node"]["H"]["candidate_coverage"] is None


def test_contradiction_roles_must_be_opposite_on_the_two_distinct_atoms():
    truth = {
        **_TRUTH,
        "atomic_evidence": [
            {
                **_TRUTH["atomic_evidence"][0],
                "atom_id": "a1",
                "supporting_span_ids": ["s1"],
            },
            {
                **_TRUTH["atomic_evidence"][0],
                "atom_id": "a2",
                "supporting_span_ids": ["s2"],
            },
        ],
        "contradiction_pairs": [{"atom_id_a": "a1", "atom_id_b": "a2"}],
    }
    support = {
        "chunkers": {
            "markdown_structure_v1": {"a1": ["s1"], "a2": ["s2"]},
        },
        "prechunk_atom_occurrence_ids": {"a1": ["o1"], "a2": ["o2"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"], "s2": ["o2"]},
        },
    }
    base = {
        "node": "H",
        "chunker": "markdown_structure_v1",
        "contract": "P1_TYPED",
        "offered_span_ids": ["s1", "s2"],
        "selected_span_ids": ["s1", "s2"],
        "published_span_ids": ["s1", "s2"],
        "offered_source_occurrence_ids": ["o1", "o2"],
    }
    same_atom = score_direct_node_records(truth, [{
        **base,
        "published_relations": [
            {"span_id": "s1", "facet_id": "f1", "role": "support"},
            {"span_id": "s1", "facet_id": "f1", "role": "contradict"},
        ],
    }], atom_support_index=support)
    opposite_atoms = score_direct_node_records(truth, [{
        **base,
        "published_relations": [
            {"span_id": "s1", "facet_id": "f1", "role": "support"},
            {"span_id": "s2", "facet_id": "f1", "role": "contradict"},
        ],
    }], atom_support_index=support)
    assert same_atom["by_node"]["H"]["contradiction_pair_recall"] == 1.0
    assert same_atom["by_node"]["H"]["contradiction_role_pair_recall"] == 0.0
    assert opposite_atoms["by_node"]["H"]["contradiction_role_pair_recall"] == 1.0


def test_hierarchical_final_recall_uses_global_not_union_of_local_publications():
    support = {
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]}},
    }
    metrics = score_direct_node_records(_TRUTH, [
        {
            "node": "H", "checkpoint_hash": "ck", "stage": "local",
            "chunker": "markdown_structure_v1",
            "offered_span_ids": ["s1"], "selected_span_ids": ["s1"],
            "published_span_ids": ["s1"],
            "offered_source_occurrence_ids": ["o1"],
        },
        {
            "node": "H", "checkpoint_hash": "ck", "stage": "global",
            "chunker": "markdown_structure_v1",
            "offered_span_ids": ["s1"], "selected_span_ids": [],
            "published_span_ids": [],
            "offered_source_occurrence_ids": ["o1"],
        },
    ], atom_support_index=support)
    assert metrics["stage_metrics"]["map"]["truth_recall"] == 1.0
    assert metrics["stage_metrics"]["reduce"]["truth_recall"] == 0.0
    assert metrics["exact_truth_recall"] == 0.0


def test_c_checkpoint_does_not_destroy_the_h_denominator_in_a_combined_arm():
    metrics = score_direct_node_records(_TRUTH, [
        {
            "node": "H",
            "checkpoint_hash": "h",
            "chunker": "markdown_structure_v1",
            "offered_span_ids": ["s1"],
            "selected_span_ids": ["s1"],
            "published_span_ids": ["s1"],
            "offered_source_occurrence_ids": ["o1"],
        },
        {
            "node": "C_VISIBLE",
            "checkpoint_hash": "c",
            "chunker": "markdown_structure_v1",
            "offered_span_ids": ["v1"],
            "selected_span_ids": ["v1"],
            "published_span_ids": ["v1"],
        },
    ], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]}},
    })
    assert metrics["h_checkpoint_count"] == 1
    assert metrics["c_checkpoint_count"] == 1
    assert metrics["exact_truth_recall"] == 1.0


def test_query_guard_is_checked_at_the_final_hierarchical_stage():
    truth = {
        **_TRUTH,
        "known_gaps": [{
            "query_attempt_id": "q1",
            "query_text": "missing evidence",
            "status": "TIMEOUT",
        }],
    }
    metrics = score_direct_node_records(truth, [
        {
            "node": "H",
            "stage": "local",
            "chunker": "markdown_structure_v1",
            "offered_span_ids": ["s1"],
            "selected_span_ids": ["s1"],
            "published_span_ids": ["s1"],
            "offered_query_attempt_ids": ["q1"],
            "published_gaps": [{"query_attempt_ids": ["q1"]}],
            "offered_source_occurrence_ids": ["o1"],
        },
        {
            "node": "H",
            "stage": "global",
            "chunker": "markdown_structure_v1",
            "offered_span_ids": ["s1"],
            "selected_span_ids": ["s1"],
            "published_span_ids": ["s1"],
            "offered_source_occurrence_ids": ["o1"],
        },
    ], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]}},
    })
    assert not metrics["query_guard_complete"]
    assert not metrics["selector_guards_complete"]


def test_direct_trace_rejects_published_query_outside_offered_set():
    metrics = score_direct_node_records(_TRUTH, [{
        "node": "H",
        "chunker": "markdown_structure_v1",
        "offered_span_ids": ["s1"],
        "selected_span_ids": ["s1"],
        "published_span_ids": ["s1"],
        "offered_query_attempt_ids": ["q1"],
        "published_gaps": [{"query_attempt_ids": ["forged"]}],
        "offered_source_occurrence_ids": ["o1"],
    }], atom_support_index={
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]}},
    })
    assert metrics["status"] == "INVALID_DIRECT_TRACE"
    assert any("query ids outside offered" in error for error in metrics["errors"])


def test_fallback_has_separate_strict_and_assisted_itt_views():
    output = ArmOutput(
        task_id="T1", arm_id="H02", variant_id="H02", replicate_id="0",
        final_report=_output().final_report, frozen=True, fell_back=True,
    )
    score = score_task(
        truth_body=_TRUTH, outputs=[output], atom_texts=_ATOM_TEXTS,
        judge_relation=_entails, citation_supports=lambda claim, link: True,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
    )
    row = score.per_arm["H02:0"]
    assert row["quality_views"]["strict"]["qualified_report"] == 0
    assert row["quality_views"]["fallback_assisted"]["weighted_required_atom_recall"] == 1.0


def test_final_report_scores_grounded_absence_and_operational_gap_separately():
    truth = {
        **_TRUTH,
        "atomic_evidence": [{
            **_TRUTH["atomic_evidence"][0],
            "atom_id": "a-negative",
            "known_unresolved": True,
        }],
        "negative_evidence": [{
            "atom_id": "a-negative",
            "facet_id": "f1",
            "query_attempt_id": "q-negative",
        }],
        "known_gaps": [{
            "query_attempt_id": "q-gap",
            "query_text": "Project Zephyr permit database",
            "status": "TIMEOUT",
        }],
    }
    report = (
        "No public filing exists for Project Zephyr in 2025 [1].\n"
        "The evidence search for the Project Zephyr permit database remained unresolved "
        "because its frozen query attempt ended with status TIMEOUT.\n\n"
        "[1] https://a.example/doc"
    )

    def relation(claim: str, fact: str) -> str:
        if "No public filing" in claim and "No public filing" in fact:
            return "entail"
        if "remained unresolved" in claim and "remained unresolved" in fact:
            return "entail"
        return "unrelated"

    score = score_task(
        truth_body=truth,
        outputs=[_output(report=report)],
        atom_texts={"a-negative": "No public filing exists for Project Zephyr in 2025."},
        judge_relation=relation,
        citation_supports=lambda claim, link: True,
        judge_policy_sha256="p" * 64,
        claim_scope="FORMATIVE_ONLY",
    )
    row = score.per_arm["P0:0"]
    assert row["grounded_negative_recall"] == 1.0
    assert row["unresolved_gap_reporting_recall"] == 1.0
    assert row["negative_gap_report"] == {
        "grounded_negative_atom_ids": ["a-negative"],
        "reported_grounded_negative_atom_ids": ["a-negative"],
        "known_gap_query_attempt_ids": ["q-gap"],
        "reported_unresolved_query_attempt_ids": ["q-gap"],
    }


def test_judge_policy_digest_binds_prompt_sampling_retry_and_failure_policy():
    config = {
        "relation_prompt_sha256": "a" * 64,
        "prompts": {"report_version": "relation_v1"},
        "model": {"temperature": 0.0, "top_p": 1.0, "seed": 7},
        "retry": {"max_retries": 3},
        "blinding": {"hide_arm": True},
        "failure_policy": {"on_unavailable": "FAIL_CLOSED"},
    }
    base = judge_policy_digest(
        config, requested_model="judge-a", returned_model="judge-a",
        system_fingerprint="fp",
    )
    for section, key, value in (
        (None, "relation_prompt_sha256", "b" * 64),
        ("model", "seed", 8),
        ("retry", "max_retries", 4),
        ("blinding", "hide_arm", False),
        ("failure_policy", "on_unavailable", "IMPUTE"),
    ):
        changed = json.loads(json.dumps(config))
        if section is None:
            changed[key] = value
        else:
            changed[section][key] = value
        assert judge_policy_digest(
            changed, requested_model="judge-a", returned_model="judge-a",
            system_fingerprint="fp",
        ) != base


def test_an_unavailable_judgment_is_never_imputed():
    """A missing score stays missing, and the sample is not dropped either."""
    from shapeflow_p1.bench.grading.judge_client import JudgeUnavailable

    def refuses(claim: str, atom: str) -> str:
        raise JudgeUnavailable("provider down")

    score = score_task(
        truth_body=_TRUTH, outputs=[_output("P0"), _output("H02")], atom_texts=_ATOM_TEXTS,
        judge_relation=refuses, citation_supports=lambda claim, link: True,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
    )
    assert set(score.per_arm) == {"P0:0", "H02:0"}
    assert sorted(score.unavailable) == ["H02:0", "P0:0"]
    assert all(item["reason"].startswith("JUDGE_UNAVAILABLE") for item in score.human_queue)
    # Missing judgments remain in the ITT denominator. They are not silently dropped or
    # presented as observed zeroes: the uncertainty is an explicit [worst,best] sensitivity.
    assert score.per_arm["H02:0"]["evaluation_status"] == "JUDGE_UNAVAILABLE"
    assert score.per_arm["H02:0"]["quality_views"]["strict"] is None
    assert score.per_arm["H02:0"]["quality_views"]["worst_case"]["qualified_report"] == 0
    assert score.per_arm["H02:0"]["quality_views"]["best_case"]["qualified_report"] == 1


def test_scores_are_write_once(tmp_path):
    score = score_task(
        truth_body=_TRUTH, outputs=[_output("P0")], atom_texts=_ATOM_TEXTS,
        judge_relation=_entails, citation_supports=lambda claim, link: True,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
    )
    path = tmp_path / "T1.json"
    digest = write_scores(score, path)
    assert write_scores(score, path) == digest      # idempotent for identical content

    other = TaskScore(task_id="T1", truth_packet_sha256="e" * 64,
                      judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY")
    with pytest.raises(EvaluationError, match="new\\s+version"):
        write_scores(other, path)
    assert json.loads(path.read_text())["truth_packet_sha256"] == "d" * 64


def test_score_creation_is_exclusive_under_concurrent_writers(tmp_path):
    first = TaskScore(
        task_id="T-race", truth_packet_sha256="a" * 64,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY")
    second = TaskScore(
        task_id="T-race", truth_packet_sha256="b" * 64,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY")
    path = tmp_path / "T-race.json"
    barrier = threading.Barrier(2)

    def write(score):
        barrier.wait()
        try:
            write_scores(score, path)
            return "WROTE"
        except EvaluationError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, (first, second)))

    assert sorted(outcomes) == ["REFUSED", "WROTE"]
    assert json.loads(path.read_text(encoding="utf-8"))[
        "truth_packet_sha256"] in {"a" * 64, "b" * 64}
    assert path.stat().st_mode & 0o222 == 0


def test_existing_score_is_rehashed_before_idempotent_return(tmp_path):
    score = TaskScore(
        task_id="T-tamper", truth_packet_sha256="a" * 64,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY")
    path = tmp_path / "T-tamper.json"
    write_scores(score, path)
    os.chmod(path, 0o640)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["truth_packet_sha256"] = "b" * 64
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(EvaluationError, match="was edited"):
        write_scores(score, path)


def test_the_treatment_path_does_not_import_the_evaluator(tmp_path):
    """Truth reachable from a selector would be an oracle, whatever the directory permissions."""
    import importlib
    import pkgutil
    import sys

    for name in ("shapeflow_p1.strategies", "shapeflow_p1.p1", "shapeflow_p1.odr"):
        package = importlib.import_module(name)
        for module in pkgutil.walk_packages(package.__path__, name + "."):
            importlib.import_module(module.name)

    treatment_modules = {
        name for name in sys.modules
        if name.startswith(("shapeflow_p1.strategies", "shapeflow_p1.p1", "shapeflow_p1.odr"))
    }
    imported_by_treatment = set()
    for name in treatment_modules:
        module = sys.modules[name]
        for attr in vars(module).values():
            candidate = getattr(attr, "__module__", None)
            if candidate:
                imported_by_treatment.add(candidate)
    assert treatment_import_violations(imported_by_treatment) == []
