"""Balanced randomization and the eligibility CART."""

from __future__ import annotations

from collections import Counter

from shapeflow_p1.analysis.heterogeneity import (
    Sample,
    coverage_estimands,
    fit_eligibility_tree,
    rule_stability,
)
from shapeflow_p1.experiment.randomization import (
    ab_ba_order,
    arm_order_for_block,
    derive_seed,
    seeded_permutation,
    williams_square,
)

# --- randomization --------------------------------------------------------------------


def test_williams_rows_are_permutations():
    sq = williams_square(4)
    for row in sq:
        assert sorted(row) == [0, 1, 2, 3]


def test_williams_first_order_carryover_is_balanced_even_n():
    # For even n each ordered adjacent pair appears equally often across the square.
    sq = williams_square(4)
    pairs = Counter()
    for row in sq:
        for a, b in zip(row, row[1:], strict=False):
            pairs[(a, b)] += 1
    counts = set(pairs.values())
    assert len(counts) == 1  # perfectly balanced


def test_derive_seed_deterministic():
    assert derive_seed("proto", "block", "3") == derive_seed("proto", "block", "3")
    assert derive_seed("a") != derive_seed("b")


def test_seeded_permutation_reproducible():
    items = list(range(10))
    assert seeded_permutation(items, 123) == seeded_permutation(items, 123)
    # a different seed generally reorders
    assert seeded_permutation(items, 1) != seeded_permutation(items, 2)


def test_ab_ba_alternates_by_replicate():
    o0 = ab_ba_order(("A", "B"), replicate=0, seed=0)
    o1 = ab_ba_order(("A", "B"), replicate=1, seed=0)
    assert o0 != o1
    assert set(o0) == {"A", "B"}


def test_arm_order_maps_labels_and_interleaves():
    arms = ["P0", "H", "C", "HC"]
    order = arm_order_for_block(arms, block_index=0, seed=0)
    assert sorted(order) == sorted(arms)


# --- eligibility CART -----------------------------------------------------------------


def _samples_where_big_evidence_helps():
    # Two boundaries per task; P1 succeeds when candidate tokens are large.
    samples = []
    for t in range(20):
        big = t % 2 == 0
        for _b in range(2):
            samples.append(Sample(
                task_id=f"t{t}",
                features={"candidate_tokens": 5000 if big else 200, "source_count": 3},
                training_success=big,
            ))
    return samples


def test_cart_learns_the_separating_feature():
    rule = fit_eligibility_tree(
        _samples_where_big_evidence_helps(), ["candidate_tokens", "source_count"],
        min_tasks_per_leaf=4,
    )
    assert rule.root_split_feature == "candidate_tokens"
    assert rule.predict({"candidate_tokens": 5000, "source_count": 3}) is True
    assert rule.predict({"candidate_tokens": 100, "source_count": 3}) is False


def test_cart_respects_min_tasks_per_leaf():
    # With a min that both children cannot satisfy, no split is made (a single leaf).
    rule = fit_eligibility_tree(
        _samples_where_big_evidence_helps(), ["candidate_tokens"], min_tasks_per_leaf=50,
    )
    assert rule.root_split_feature is None  # could not split with >=50 tasks/side


def test_cart_refuses_zero_gain_split_for_constant_labels():
    samples = [
        Sample(
            task_id=f"t{index}",
            features={"candidate_tokens": float(index), "source_count": float(index % 3)},
            training_success=True,
        )
        for index in range(24)
    ]
    rule = fit_eligibility_tree(
        samples,
        ["candidate_tokens", "source_count"],
        min_tasks_per_leaf=8,
    )
    assert rule.root_split_feature is None
    assert rule.predict({"candidate_tokens": 0.0, "source_count": 0.0}) is True


def test_rule_stability_high_for_clean_signal():
    stab = rule_stability(
        _samples_where_big_evidence_helps(), ["candidate_tokens", "source_count"],
        n_boot=50, seed=1, min_tasks_per_leaf=4,
    )
    assert stab >= 0.70  # a clean signal reproduces


def test_coverage_estimands_two_ways():
    rule = fit_eligibility_tree(
        _samples_where_big_evidence_helps(), ["candidate_tokens"], min_tasks_per_leaf=4,
    )
    cov = coverage_estimands(rule, _samples_where_big_evidence_helps())
    # half the tasks have big evidence -> ~0.5 both ways
    assert 0.4 <= cov["invocation_coverage"] <= 0.6
    assert 0.4 <= cov["task_exposure_coverage"] <= 0.6


def test_readable_and_hash_stable():
    rule = fit_eligibility_tree(
        _samples_where_big_evidence_helps(), ["candidate_tokens"], min_tasks_per_leaf=4,
    )
    assert rule.to_readable()
    assert rule.rule_hash == rule.rule_hash
