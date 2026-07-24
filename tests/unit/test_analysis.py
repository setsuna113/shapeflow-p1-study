"""Analysis: cluster bootstrap reproducibility, work-saving, verdict tree, corrections."""

from __future__ import annotations

import numpy as np
import pytest

from shapeflow_p1.analysis.bootstrap import (
    cluster_bootstrap_ci,
    paired_log_ratio_saving,
    seed_from,
)
from shapeflow_p1.analysis.decision import (
    NodeEvidence,
    Verdict,
    decide,
    holm_reject,
    intersection_union_pass,
)


# --- bootstrap ------------------------------------------------------------------------


def test_seed_is_deterministic():
    assert seed_from("protocol", "holdout", "H02") == seed_from("protocol", "holdout", "H02")
    assert seed_from("a") != seed_from("b")


def test_cluster_bootstrap_is_reproducible():
    vals = list(range(20))
    clusters = [f"c{i % 5}" for i in range(20)]
    a = cluster_bootstrap_ci(vals, clusters, np.mean, n_boot=500, seed=42, side="two")
    b = cluster_bootstrap_ci(vals, clusters, np.mean, n_boot=500, seed=42, side="two")
    assert (a.point, a.lower, a.upper) == (b.point, b.lower, b.upper)


def test_cluster_bootstrap_requires_two_clusters():
    with pytest.raises(ValueError, match="2 clusters"):
        cluster_bootstrap_ci([1.0, 2.0], ["c", "c"], np.mean)


def test_work_saving_positive_when_p1_cheaper():
    # P1 uses ~70% of P0 work across clusters -> ~30% saving, LCB should be well above 0.
    rng = np.random.default_rng(0)
    p0 = rng.uniform(8, 12, size=40)
    p1 = p0 * rng.uniform(0.65, 0.75, size=40)
    clusters = [f"c{i % 8}" for i in range(40)]
    ci = paired_log_ratio_saving(p1, p0, clusters, n_boot=1000, seed=7)
    assert ci.point > 0.2
    assert ci.lower > 0.0  # 95% lower bound clears zero


def test_work_saving_rejects_nonpositive():
    with pytest.raises(ValueError, match="strictly positive"):
        paired_log_ratio_saving([1.0, 0.0], [2.0, 2.0], ["a", "b"])


# --- verdict tree ---------------------------------------------------------------------


def _ev(**over):
    base = dict(
        structural_pass=True, deterministic_harm=False, quality_guards_pass=True,
        isolated_saving_lcb=0.2, saving_ucb=0.3, operational_available=True,
        operational_saving_lcb=0.15, operational_quality_pass=True, e2e_speedup_lcb=1.0,
    )
    base.update(over)
    return NodeEvidence(**base)


def test_keep_when_both_layers_pass():
    assert decide(_ev()) is Verdict.KEEP


def test_thesis_grade_needs_speedup():
    assert decide(_ev(e2e_speedup_lcb=1.6)) is Verdict.THESIS_GRADE


def test_structural_failure_dominates():
    assert decide(_ev(structural_pass=False, quality_guards_pass=True)) is Verdict.KILL_STRUCTURAL


def test_harm_dominates_even_with_savings():
    assert decide(_ev(deterministic_harm=True)) is Verdict.KILL_HARM


def test_isolated_pass_operational_fail_is_mechanism_only():
    v = decide(_ev(operational_saving_lcb=0.0, operational_quality_pass=False))
    assert v is Verdict.MECHANISM_ONLY


def test_underpowered_operational_is_mechanism_only():
    v = decide(_ev(operational_available=False))
    assert v is Verdict.MECHANISM_ONLY


def test_no_headroom_when_quality_fine_but_saving_bounded_low():
    v = decide(_ev(isolated_saving_lcb=-0.05, saving_ucb=0.05, operational_saving_lcb=0.0,
                   operational_quality_pass=False))
    assert v is Verdict.KILL_NO_HEADROOM


def test_not_established_when_nothing_proven():
    # Quality guards fail (so not no-headroom), savings inconclusive, no conditional rule.
    v = decide(_ev(quality_guards_pass=False, isolated_saving_lcb=0.0, saving_ucb=0.5,
                   operational_saving_lcb=0.0, operational_quality_pass=False))
    assert v is Verdict.NOT_ESTABLISHED


def test_conditional_when_subset_keeps_with_coverage():
    v = decide(_ev(operational_saving_lcb=0.0, operational_quality_pass=False,
                   conditional_rule_keeps=True, coverage_lcb=0.35))
    assert v is Verdict.CONDITIONAL


def test_conditional_needs_coverage():
    v = decide(_ev(operational_saving_lcb=0.0, operational_quality_pass=False,
                   conditional_rule_keeps=True, coverage_lcb=0.25))
    assert v is not Verdict.CONDITIONAL  # coverage below 0.30


# --- guard combiners ------------------------------------------------------------------


def test_intersection_union_all_must_pass():
    margins = {"recall": -0.05, "citation": -0.03}
    assert intersection_union_pass({"recall": -0.02, "citation": -0.01}, margins)
    # one guard below margin -> whole family fails
    assert not intersection_union_pass({"recall": -0.06, "citation": -0.01}, margins)


def test_holm_step_down():
    # Classic Holm behavior: smallest p compared to alpha/m, and a failure stops the chain.
    pvals = {"a": 0.001, "b": 0.04, "c": 0.20}
    rej = holm_reject(pvals, alpha=0.05)
    assert rej["a"] is True          # 0.001 <= 0.05/3
    assert rej["b"] is False         # 0.04 > 0.05/2 -> retained
    assert rej["c"] is False         # chain stopped


def test_holm_rejects_all_when_tiny():
    rej = holm_reject({"a": 0.001, "b": 0.002, "c": 0.003}, alpha=0.05)
    assert all(rej.values())
