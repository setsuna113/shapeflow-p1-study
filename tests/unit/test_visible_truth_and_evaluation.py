"""The C_VISIBLE denominator, and the rules that keep truth out of the treatment path.

The projection decides what the compressor *could* have kept. Getting its denominator wrong is
the most likely way to manufacture a false C_VISIBLE failure: penalise the reducer for an atom
that an upstream page summary already dropped, and the number reads as "the compressor lost it".
"""

from __future__ import annotations

import json

import pytest

from shapeflow_p1.evaluation.runner import (
    ArmOutput,
    EvaluationError,
    TaskScore,
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
            judge_relation=_entails, citation_supports=lambda c, l: True,
            judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
        )


def test_every_arm_is_scored_against_one_packet_and_one_policy():
    score = score_task(
        truth_body=_TRUTH,
        outputs=[_output("P0"), _output("H02"), _output("C01")],
        atom_texts=_ATOM_TEXTS, judge_relation=_entails,
        citation_supports=lambda c, l: True,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
    )
    assert set(score.per_arm) == {"P0:0", "H02:0", "C01:0"}
    assert {v["claim_scope"] for v in score.per_arm.values()} == {"FORMATIVE_ONLY"}
    assert score.truth_packet_sha256 == "d" * 64
    assert score.judge_policy_sha256 == "p" * 64


def test_an_unavailable_judgment_is_never_imputed():
    """A missing score stays missing, and the sample is not dropped either."""
    from shapeflow_p1.evaluation.judge_client import JudgeUnavailable

    def refuses(claim: str, atom: str) -> str:
        raise JudgeUnavailable("provider down")

    score = score_task(
        truth_body=_TRUTH, outputs=[_output("P0"), _output("H02")], atom_texts=_ATOM_TEXTS,
        judge_relation=refuses, citation_supports=lambda c, l: True,
        judge_policy_sha256="p" * 64, claim_scope="FORMATIVE_ONLY",
    )
    assert score.per_arm == {}
    assert sorted(score.unavailable) == ["H02:0", "P0:0"]
    assert all(item["reason"].startswith("JUDGE_UNAVAILABLE") for item in score.human_queue)


def test_scores_are_write_once(tmp_path):
    score = score_task(
        truth_body=_TRUTH, outputs=[_output("P0")], atom_texts=_ATOM_TEXTS,
        judge_relation=_entails, citation_supports=lambda c, l: True,
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
