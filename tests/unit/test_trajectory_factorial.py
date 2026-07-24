"""Trajectory metrics and the 2x2 factorial contrasts."""

from __future__ import annotations

import numpy as np

from shapeflow_p1.analysis.factorial import ArmOutcomes, estimate_contrasts
from shapeflow_p1.evaluation.trajectory_eval import (
    Trajectory,
    facet_followup_rate,
    query_redundancy,
    retrieval_failure_rate,
    summarize_trajectory,
)


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
