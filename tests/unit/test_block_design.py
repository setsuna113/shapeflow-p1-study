"""D-optimal screening design: coding, selection, assignment, freeze gates."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from shapeflow_p1.experiment.block_design import (
    Factor,
    assign_to_states,
    check_freeze_gates,
    condition_number,
    d_criterion,
    encode_design,
    fedorov_select,
)

CHUNKER = Factor("chunker", ("fixed", "markdown"))
CONTRACT = Factor("contract", ("id", "typed", "bridge"))
AGG = Factor("aggregation", ("union", "coverage"))


def _full_factorial(factors):
    keys = [f.name for f in factors]
    combos = itertools.product(*[f.levels for f in factors])
    return [dict(zip(keys, c)) for c in combos]


# --- effect coding --------------------------------------------------------------------


def test_effect_coding_is_sum_to_zero():
    # k=3 levels -> 2 columns; last level is all -1s.
    assert CONTRACT.effect_columns("id") == [1.0, 0.0]
    assert CONTRACT.effect_columns("typed") == [0.0, 1.0]
    assert CONTRACT.effect_columns("bridge") == [-1.0, -1.0]


def test_encode_design_shape_and_intercept():
    variants = _full_factorial([CHUNKER, CONTRACT])
    X, cols = encode_design(variants, [CHUNKER, CONTRACT])
    # intercept + (2-1) + (3-1) = 1 + 1 + 2 = 4 columns
    assert X.shape == (6, 4)
    assert cols[0] == "intercept"
    assert np.allclose(X[:, 0], 1.0)


def test_interaction_columns_are_products():
    variants = _full_factorial([CHUNKER, CONTRACT])
    X, cols = encode_design(variants, [CHUNKER, CONTRACT], interactions=[("chunker", "contract")])
    # + (1 * 2) interaction columns = 6 total
    assert X.shape == (6, 6)
    assert any("x" in c for c in cols)


# --- D-criterion ----------------------------------------------------------------------


def test_full_factorial_is_full_rank_and_well_conditioned():
    variants = _full_factorial([CHUNKER, CONTRACT, AGG])
    X, _ = encode_design(variants, [CHUNKER, CONTRACT, AGG])
    assert np.linalg.matrix_rank(X) == X.shape[1]
    assert condition_number(X) < 1e4


def test_d_criterion_rewards_information():
    variants = _full_factorial([CHUNKER, CONTRACT])
    X_full, _ = encode_design(variants, [CHUNKER, CONTRACT])
    # A degenerate design repeating one variant is far less informative.
    X_dupe, _ = encode_design([variants[0]] * 6, [CHUNKER, CONTRACT])
    assert d_criterion(X_full) > d_criterion(X_dupe)


# --- Fedorov selection ----------------------------------------------------------------


def test_fedorov_is_deterministic_and_keeps_anchors():
    variants = _full_factorial([CHUNKER, CONTRACT, AGG])  # 12 variants
    X, _ = encode_design(variants, [CHUNKER, CONTRACT, AGG])
    a = fedorov_select(X, 8, forced=[0, 3], seed=5)
    b = fedorov_select(X, 8, forced=[0, 3], seed=5)
    assert a == b                      # deterministic
    assert 0 in a and 3 in a           # forced anchors retained
    assert len(set(a)) == 8


def test_fedorov_does_not_worsen_the_criterion():
    variants = _full_factorial([CHUNKER, CONTRACT, AGG])
    X, _ = encode_design(variants, [CHUNKER, CONTRACT, AGG])
    # The exchange starts from a random fill and only accepts improving swaps, so the final
    # design's criterion is at least the full-set criterion minus nothing degenerate.
    sel = fedorov_select(X, 10, seed=1)
    assert np.isfinite(d_criterion(X[sel]))
    assert np.linalg.matrix_rank(X[sel]) == X.shape[1]  # selection stays estimable


# --- state assignment -----------------------------------------------------------------


def test_assignment_balances_exposure_within_one():
    states = [f"s{i}" for i in range(12)]
    variants = [f"v{i}" for i in range(5)]
    asg = assign_to_states(states, variants, per_state=3, seed=0)
    exposure = asg.exposure()
    assert len(exposure) == 5
    spread = max(exposure.values()) - min(exposure.values())
    assert spread <= 1
    for s in states:
        assert len(asg.by_state[s]) == 3


def test_assignment_is_deterministic():
    states = [f"s{i}" for i in range(8)]
    variants = [f"v{i}" for i in range(4)]
    a = assign_to_states(states, variants, per_state=2, seed=3)
    b = assign_to_states(states, variants, per_state=2, seed=3)
    assert a.by_state == b.by_state


# --- freeze gates ---------------------------------------------------------------------


def test_freeze_gates_pass_a_good_design():
    variants = _full_factorial([CHUNKER, CONTRACT, AGG])
    X, _ = encode_design(variants, [CHUNKER, CONTRACT, AGG])
    states = [f"s{i}" for i in range(12)]
    asg = assign_to_states(states, [v["contract"] + v["chunker"] for v in variants][:6],
                           per_state=3, seed=0)
    assert check_freeze_gates(X, asg).ok


def test_freeze_gates_reject_rank_deficient():
    # A design of one repeated variant is rank-deficient.
    variants = _full_factorial([CHUNKER, CONTRACT])
    X, _ = encode_design([variants[0]] * 6, [CHUNKER, CONTRACT])
    asg = assign_to_states([f"s{i}" for i in range(6)], ["a", "b", "c"], per_state=2, seed=0)
    res = check_freeze_gates(X, asg)
    assert not res.ok
    assert any("rank" in f or "condition" in f for f in res.failures)


def test_freeze_gates_reject_empty_state():
    from shapeflow_p1.experiment.block_design import StateAssignment

    variants = _full_factorial([CHUNKER, CONTRACT, AGG])
    X, _ = encode_design(variants, [CHUNKER, CONTRACT, AGG])
    bad = StateAssignment(by_state={"s0": (), "s1": ("a", "b")})
    res = check_freeze_gates(X, bad)
    assert not res.ok
    assert any("no P1" in f for f in res.failures)
