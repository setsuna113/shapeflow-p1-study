"""Trajectory metrics and the 2x2 factorial contrasts."""

from __future__ import annotations

import numpy as np

from shapeflow_p1.analysis.factorial import ArmOutcomes, estimate_contrasts
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.evaluation.trajectory_eval import (
    Trajectory,
    facet_followup_rate,
    query_redundancy,
    retrieval_failure_rate,
    summarize_frozen_events,
    summarize_trajectory,
)
from shapeflow_p1.hashing import sha256_hex


def _traj(**over):
    base = dict(
        waves=2, conduct_research_calls=3,
        queries=("eiffel tower height", "eiffel tower height meters", "paris population"),
        required_facets=("height", "location"), followed_up_facets=("height",),
        close_reasons=("RESEARCH_COMPLETE",), failed_retrievals=1, total_retrievals=5,
    )
    base.update(over)
    return Trajectory(**base)


def test_query_redundancy_detects_near_duplicates():
    # first two queries are near-duplicates; third is distinct -> ~1/3 redundant
    r = query_redundancy(_traj().queries, threshold=0.5)
    assert r > 0


def test_query_redundancy_zero_for_distinct():
    assert query_redundancy(("cats", "dogs", "quantum physics")) == 0.0


def test_facet_followup_rate():
    assert facet_followup_rate(_traj()) == 0.5  # 1 of 2 facets followed up
    assert facet_followup_rate(_traj(required_facets=())) == 1.0  # NA -> 1.0


def test_retrieval_failure_rate():
    assert retrieval_failure_rate(_traj()) == 0.2
    assert retrieval_failure_rate(_traj(total_retrievals=0, failed_retrievals=0)) == 0.0


def test_summarize_has_all_metrics():
    s = summarize_trajectory(_traj())
    for k in ("waves", "fan_out", "query_redundancy", "facet_followup_rate",
              "retrieval_failure_rate", "premature_completes"):
        assert k in s


def _event(index, kind, *, position="PRE_TREATMENT", **payload):
    body = {
        "event_index": index,
        "kind": kind,
        "position": position,
        **payload,
    }
    body["event_sha256"] = sha256_hex(canonical_json(body))
    return body


def test_frozen_event_summary_measures_divergence_instead_of_rejecting_it():
    events = [
        _event(
            0,
            "SEARCH_QUERY",
            query="battery safety evidence",
            result_count=2,
            source_occurrence_ids=["O1", "O2"],
        ),
        _event(
            1,
            "PAGE_BATCH_REDUCED",
            position="TREATMENT",
            checkpoint="H1",
            siblings=1,
        ),
        _event(
            2,
            "SEARCH_QUERY",
            position="POST_TREATMENT",
            query="battery safety evidence",
            result_count=0,
            source_occurrence_ids=[],
        ),
        _event(
            3,
            "MODEL_TOOL_DECISION",
            position="POST_TREATMENT",
            tool_names=["ConductResearch", "think_tool"],
        ),
        _event(
            4,
            "CLOSE_REDUCED",
            position="POST_TREATMENT",
            close_reason="RESEARCH_COMPLETE",
        ),
    ]
    metrics = summarize_frozen_events(events)
    assert metrics["status"] == "OK"
    assert metrics["query_count"] == 2
    assert metrics["query_redundancy"] == 0.5
    assert metrics["retrieval_empty_rate"] == 0.5
    assert metrics["post_treatment_query_count"] == 1
    assert metrics["conduct_research_calls"] == 1
    assert metrics["close_reason"] == "RESEARCH_COMPLETE"


def test_frozen_event_summary_rejects_tampering_not_natural_arm_difference():
    events = [
        _event(
            0,
            "SEARCH_QUERY",
            query="x",
            result_count=1,
            source_occurrence_ids=["O1"],
        )
    ]
    events[0]["result_count"] = 999
    with np.testing.assert_raises_regex(ValueError, "does not verify"):
        summarize_frozen_events(events)


def test_frozen_event_summary_counts_production_checkpoint_incidents_once():
    """NODE_SELECTION and the reducer event describe one incident, not two."""
    events = [
        _event(
            0,
            "NODE_SELECTION",
            position="TREATMENT",
            checkpoint="H1",
            direct_node_record={
                "node": "H",
                "checkpoint_hash": "H1",
                "fell_back": True,
                "failure": {"reason": "INVALID_OUTPUT"},
            },
        ),
        _event(
            1,
            "PAGE_BATCH_REDUCED",
            position="POST_TREATMENT",
            checkpoint="H1",
            fell_back=True,
            failure="INVALID_OUTPUT",
            siblings=2,
        ),
        _event(
            2,
            "NODE_SELECTION",
            position="POST_TREATMENT",
            checkpoint="C1",
            direct_node_record={
                "node": "C_VISIBLE",
                "checkpoint_hash": "C1",
                "fell_back": False,
                "failure": None,
            },
        ),
        _event(
            3,
            "CLOSE_REDUCED",
            position="POST_TREATMENT",
            checkpoint="C1",
            close_reason="RESEARCH_COMPLETE",
        ),
    ]

    metrics = summarize_frozen_events(events)

    assert metrics["h_checkpoint_count"] == 1
    assert metrics["c_checkpoint_count"] == 1
    assert metrics["fallback_count"] == 1
    assert metrics["failure_count"] == 1


def test_frozen_event_summary_rejects_unattributable_failure_incident():
    events = [
        _event(
            0,
            "PAGE_BATCH_REDUCED",
            position="TREATMENT",
            failure="INVALID_OUTPUT",
            siblings=1,
        )
    ]
    with np.testing.assert_raises_regex(ValueError, "node-scoped checkpoint"):
        summarize_frozen_events(events)


# --- factorial ------------------------------------------------------------------------


def _outcomes(interaction_effect=0.0):
    # Construct arms where H saves 0.1, C saves 0.1, and interaction is as specified.
    out = []
    rng = np.random.default_rng(0)
    for i in range(24):
        p0 = 1.0 + rng.normal(0, 0.01)
        h = p0 - 0.1
        c = p0 - 0.1
        hc = p0 - 0.2 - interaction_effect  # extra saving beyond additive = interaction
        out.append(ArmOutcomes(f"t{i}", f"cl{i % 6}", p0=p0, h=h, c=c, hc=hc))
    return out


def test_main_effects_recovered():
    con = estimate_contrasts(_outcomes(0.0), n_boot=500, seed=1)
    assert con.h_simple.point < -0.05   # H reduces the endpoint by ~0.1
    assert con.c_simple.point < -0.05
    assert con.joint.point < -0.15      # joint ~ -0.2


def test_interaction_detected_when_present():
    # No extra interaction -> interaction term ~ 0.
    flat = estimate_contrasts(_outcomes(0.0), n_boot=500, seed=1)
    assert abs(flat.interaction.point) < 0.03
    # Super-additive combination -> negative interaction term.
    synergy = estimate_contrasts(_outcomes(0.1), n_boot=500, seed=1)
    assert synergy.interaction.point < -0.05
