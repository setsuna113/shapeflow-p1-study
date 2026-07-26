"""Counterexamples for the preregistered weighted and negative/gap direct gates."""

from __future__ import annotations

import pytest

from shapeflow_p1.evaluation.runner import score_direct_node_records
from shapeflow_p1.evaluation.visible_truth_projection import (
    EXPLICITLY_VISIBLE,
    ProjectedAtom,
    VisibleTruthProjection,
    visible_recall,
)


def _h_fixture(
    weights: list[float],
    *,
    published_indexes: set[int],
    critical_indexes: frozenset[int] = frozenset(),
) -> tuple[dict, list[dict], dict]:
    atoms = []
    chunker_index: dict[str, list[str]] = {}
    occurrence_index: dict[str, list[str]] = {}
    span_occurrences: dict[str, list[str]] = {}
    spans: list[str] = []
    occurrences: list[str] = []
    for index, weight in enumerate(weights):
        atom_id = f"a{index}"
        span_id = f"s{index}"
        occurrence_id = f"o{index}"
        atoms.append({
            "atom_id": atom_id,
            "facet_id": "f",
            "weight": weight,
            "critical": index in critical_indexes,
            "supporting_span_ids": [span_id],
        })
        chunker_index[atom_id] = [span_id]
        occurrence_index[atom_id] = [occurrence_id]
        span_occurrences[span_id] = [occurrence_id]
        spans.append(span_id)
        occurrences.append(occurrence_id)
    published = [f"s{index}" for index in sorted(published_indexes)]
    truth = {
        "task_id": "weighted",
        "required_facets": ["f"],
        "atomic_evidence": atoms,
        "contradiction_pairs": [],
    }
    records = [{
        "node": "H",
        "checkpoint_hash": "ck",
        "chunker": "markdown_structure_v1",
        "contract": "P1_ID",
        "aggregation": "stable_union_v1",
        "offered_span_ids": spans,
        "selected_span_ids": published,
        "published_span_ids": published,
        "offered_source_occurrence_ids": occurrences,
    }]
    support = {
        "chunkers": {"markdown_structure_v1": chunker_index},
        "prechunk_atom_occurrence_ids": occurrence_index,
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": span_occurrences,
        },
    }
    return truth, records, support


def test_high_weight_miss_cannot_pass_via_unweighted_ninety_percent_recall():
    # Ten low-value atoms survive and one high-value atom is lost. Set recall exceeds the
    # configured 0.90 threshold, but the preregistered weighted raw-evidence gate must fail.
    truth, records, support = _h_fixture(
        [1.0] * 10 + [90.0],
        published_indexes=set(range(10)),
        critical_indexes=frozenset({0}),
    )
    metrics = score_direct_node_records(
        truth, records, atom_support_index=support)["by_node"]["H"]

    assert metrics["selector_conditional_recall"] == pytest.approx(10 / 11)
    assert metrics["selector_conditional_recall"] > 0.90
    assert metrics["weighted_evidence_recall"] == pytest.approx(0.10)
    assert metrics["eligible_denominators"]["weighted_evidence_recall"] == 100.0
    # The critical guard stays an independent, unweighted 100% requirement.
    assert metrics["critical_truth_recall"] == 1.0


def test_c_visible_weighted_recall_uses_only_projected_visible_atoms():
    projection = VisibleTruthProjection(
        checkpoint_hash="c",
        visible_compressor_view_hash="v",
        projected_atoms=tuple(
            ProjectedAtom(f"a{index}", (f"s{index}",), EXPLICITLY_VISIBLE)
            for index in range(11)
        ),
    )
    retained = {f"a{index}" for index in range(10)}
    weights = {f"a{index}": 1.0 for index in range(10)}
    weights["a10"] = 90.0
    # An atom outside C_VISIBLE's reference projection cannot enter its denominator.
    weights["raw-registry-only"] = 10_000.0

    assert visible_recall(projection, retained) == pytest.approx(10 / 11)
    assert visible_recall(projection, retained, weights=weights) == pytest.approx(0.10)


def test_c_visible_weighted_recall_refuses_a_missing_frozen_weight():
    projection = VisibleTruthProjection(
        checkpoint_hash="c",
        visible_compressor_view_hash="v",
        projected_atoms=(
            ProjectedAtom("a", ("s",), EXPLICITLY_VISIBLE),
        ),
    )
    with pytest.raises(ValueError, match="lack frozen weights"):
        visible_recall(projection, {"a"}, weights={})


def test_missing_frozen_truth_weight_invalidates_the_direct_measurement():
    truth, records, support = _h_fixture([1.0], published_indexes={0})
    del truth["atomic_evidence"][0]["weight"]

    metrics = score_direct_node_records(truth, records, atom_support_index=support)
    h = metrics["by_node"]["H"]

    assert metrics["status"] == "INVALID_DIRECT_TRACE"
    assert h["status"] == "INVALID_DIRECT_TRACE"
    assert h["weighted_evidence_recall"] is None
    assert h["eligible_denominators"]["weighted_evidence_recall"] is None
    assert any("no frozen weight" in error for error in h["errors"])


def test_zero_positive_weight_is_explicitly_not_applicable():
    truth, records, support = _h_fixture([0.0], published_indexes=set())
    h = score_direct_node_records(
        truth, records, atom_support_index=support)["by_node"]["H"]

    assert h["status"] == "OK"
    assert h["eligible_denominators"]["weighted_evidence_recall"] == 0.0
    assert h["weighted_evidence_recall"] is None


def test_negative_gap_gate_pools_opportunities_instead_of_macro_averaging_categories():
    truth, records, support = _h_fixture([8.0], published_indexes={0})
    truth["negative_evidence"] = [{
        "atom_id": "a0",
        "facet_id": "f",
        "query_attempt_id": "q-negative",
    }]
    truth["known_gaps"] = [
        {
            "query_attempt_id": f"q-gap-{index}",
            "query_text": "unresolved",
            "status": "TIMEOUT",
        }
        for index in range(9)
    ]
    records[0]["offered_query_attempt_ids"] = [
        "q-negative", *(f"q-gap-{index}" for index in range(9)),
    ]
    records[0]["published_query_attempt_ids"] = []

    h = score_direct_node_records(
        truth, records, atom_support_index=support)["by_node"]["H"]

    assert h["grounded_negative_atom_recall"] == 1.0
    assert h["negative_query_trace_recall"] == 0.0
    assert h["unresolved_gap_recall"] == 0.0
    assert h["negative_gap_recall"] == pytest.approx(8.0 / 18.0)
    assert h["negative_gap_recall"] != pytest.approx((1.0 + 0.0 + 0.0) / 3.0)
    assert h["eligible_denominators"]["negative_gap_recall"] == 18.0
