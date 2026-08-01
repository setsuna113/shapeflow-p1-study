"""D-optimal screening design and its freeze gates (plan §14.4).

Screening cannot afford to run every variant at every boundary, so the design chooses which
variants to run and how to spread them across boundary states. Two jobs, both here and both
outcome-blind:

1. **Which variants** -- effect-code the categorical factors (chunker, scope, contract,
   aggregation, ...) plus the pre-registered interactions, then pick a subset that maximizes the
   D-criterion ``log det(X'X + ridge I)`` by Fedorov coordinate exchange. Three anchors (an
   ID-only, a typed-coverage, and a bridge variant) are forced in so the contract axis is always
   estimable. The exchange is seeded from the protocol, so the chosen design is reproducible.

2. **How to spread them** -- assign the chosen variants to boundary states so every state also
   carries P0, variant exposure is near-uniform, and strata stay balanced. The allocator reads
   only pre-treatment strata; it never sees an outcome, which is what keeps the design from being
   an outcome-dependent choice.

The freeze gates refuse a design that is not estimable or not balanced: full column rank, a
condition number under 1e4, variant-exposure spread <= 1, pair-exposure spread <= 2, and every
state containing P0. A design that fails a gate is not frozen; it is fixed or shrunk before any
GPU time is spent, never patched after seeing results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Optional, Sequence

import numpy as np

__all__ = [
    "Factor",
    "encode_design",
    "d_criterion",
    "condition_number",
    "fedorov_select",
    "assign_to_states",
    "StateAssignment",
    "FreezeGateResult",
    "check_freeze_gates",
]


@dataclass(frozen=True)
class Factor:
    name: str
    levels: tuple[str, ...]

    def effect_columns(self, level: str) -> list[float]:
        """Sum-to-zero effect coding: k levels -> k-1 columns; the last level is all -1s."""
        if level not in self.levels:
            raise ValueError(f"level {level!r} not in factor {self.name}")
        k = len(self.levels)
        idx = self.levels.index(level)
        if idx == k - 1:
            return [-1.0] * (k - 1)
        return [1.0 if j == idx else 0.0 for j in range(k - 1)]


def encode_design(
    variants: Sequence[dict],
    factors: Sequence[Factor],
    interactions: Sequence[tuple[str, str]] = (),
) -> tuple[np.ndarray, list[str]]:
    """Build the effect-coded design matrix (with intercept) and its column names.

    ``variants`` are dicts mapping factor name -> level. Interaction columns are the elementwise
    products of the two factors' effect columns.
    """
    by_name = {f.name: f for f in factors}
    for name, _ in [(a, b) for a, b in interactions] + [(b, a) for a, b in interactions]:
        if name not in by_name:
            raise ValueError(f"interaction references unknown factor {name!r}")

    rows: list[list[float]] = []
    col_names: list[str] = ["intercept"]
    # Determine column layout from the first variant.
    for f in factors:
        col_names += [f"{f.name}[{i}]" for i in range(len(f.levels) - 1)]
    for a, b in interactions:
        na, nb = by_name[a], by_name[b]
        for i in range(len(na.levels) - 1):
            for j in range(len(nb.levels) - 1):
                col_names.append(f"{a}[{i}]x{b}[{j}]")

    for v in variants:
        row = [1.0]
        main: dict[str, list[float]] = {}
        for f in factors:
            cols = f.effect_columns(v[f.name])
            main[f.name] = cols
            row += cols
        for a, b in interactions:
            ca, cb = main[a], main[b]
            for x in ca:
                for y in cb:
                    row.append(x * y)
        rows.append(row)
    return np.asarray(rows, dtype=float), col_names


def d_criterion(X: np.ndarray, *, ridge: float = 1e-8) -> float:
    """``log det(X'X + ridge I)`` -- larger is more informative. The ridge keeps it finite for a
    rank-deficient candidate subset during the exchange."""
    xtx = X.T @ X + ridge * np.eye(X.shape[1])
    sign, logdet = np.linalg.slogdet(xtx)
    return float(logdet) if sign > 0 else float("-inf")


def condition_number(X: np.ndarray) -> float:
    xtx = X.T @ X
    return float(np.linalg.cond(xtx))


def fedorov_select(
    pool: np.ndarray,
    n_select: int,
    *,
    forced: Sequence[int] = (),
    seed: int = 0,
    ridge: float = 1e-8,
    max_iter: int = 100,
) -> list[int]:
    """Select ``n_select`` pool rows maximizing the D-criterion by coordinate exchange.

    ``forced`` indices are always kept (the contract anchors). Deterministic given ``seed``.
    """
    n_pool = pool.shape[0]
    forced = list(dict.fromkeys(forced))
    if n_select < len(forced) or n_select > n_pool:
        raise ValueError("n_select must be within [len(forced), pool size]")
    rng = np.random.default_rng(seed)
    remaining = [i for i in range(n_pool) if i not in forced]
    rng.shuffle(remaining)
    selected = forced + remaining[: n_select - len(forced)]

    def score(idxs: list[int]) -> float:
        return d_criterion(pool[idxs], ridge=ridge)

    best = score(selected)
    swappable = set(range(len(forced), n_select))  # never swap out a forced anchor
    for _ in range(max_iter):
        improved = False
        current_set = set(selected)
        for slot in list(swappable):
            for cand in range(n_pool):
                if cand in current_set:
                    continue
                trial = list(selected)
                trial[slot] = cand
                s = score(trial)
                if s > best + 1e-12:
                    selected = trial
                    best = s
                    current_set = set(selected)
                    improved = True
        if not improved:
            break
    return selected


@dataclass(frozen=True)
class StateAssignment:
    """Which variants each boundary state runs. P0 is implicit-and-required in every state."""

    # state_id -> tuple of variant_ids (excluding P0, which is always present)
    by_state: dict[str, tuple[str, ...]]

    def exposure(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for variants in self.by_state.values():
            for v in variants:
                counts[v] = counts.get(v, 0) + 1
        return counts

    def pair_exposure(self) -> dict[tuple[str, str], int]:
        pairs: dict[tuple[str, str], int] = {}
        for variants in self.by_state.values():
            uniq = sorted(set(variants))
            for i in range(len(uniq)):
                for j in range(i + 1, len(uniq)):
                    key = (uniq[i], uniq[j])
                    pairs[key] = pairs.get(key, 0) + 1
        return pairs


def assign_to_states(
    state_ids: Sequence[str],
    variant_ids: Sequence[str],
    per_state: int,
    *,
    seed: int = 0,
) -> StateAssignment:
    """Balanced incomplete-block assignment: give each state ``per_state`` variants, spreading
    exposure as uniformly as possible. Deterministic; reads no outcomes. A round-robin over a
    seeded rotation keeps every variant's exposure within 1 of every other's."""
    if per_state < 1 or per_state > len(variant_ids):
        raise ValueError("per_state must be in [1, number of variants]")
    rng = np.random.default_rng(seed)
    order = list(variant_ids)
    rng.shuffle(order)
    by_state: dict[str, tuple[str, ...]] = {}
    cursor = 0
    m = len(order)
    for state in state_ids:
        chosen = [order[(cursor + k) % m] for k in range(per_state)]
        by_state[state] = tuple(chosen)
        cursor = (cursor + per_state) % m
    return StateAssignment(by_state=by_state)


@dataclass
class FreezeGateResult:
    failures: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures


def check_freeze_gates(
    X: np.ndarray,
    assignment: StateAssignment,
    *,
    max_condition: float = 1e4,
    max_exposure_spread: int = 1,
    max_pair_spread: int = 2,
) -> FreezeGateResult:
    """Refuse a design that is not estimable or not balanced (plan §14.4 gates)."""
    result = FreezeGateResult()

    rank = int(np.linalg.matrix_rank(X))
    if rank < X.shape[1]:
        result.failures.append(f"design matrix not full column rank ({rank} < {X.shape[1]})")

    cond = condition_number(X)
    if not np.isfinite(cond) or cond >= max_condition:
        result.failures.append(f"condition number {cond:.1f} >= {max_condition}")

    exposure = assignment.exposure()
    if exposure:
        spread = max(exposure.values()) - min(exposure.values())
        if spread > max_exposure_spread:
            result.failures.append(f"variant exposure spread {spread} > {max_exposure_spread}")

    pairs = assignment.pair_exposure()
    if pairs:
        pair_spread = max(pairs.values()) - min(pairs.values())
        if pair_spread > max_pair_spread:
            result.failures.append(f"pair exposure spread {pair_spread} > {max_pair_spread}")

    # Every state must include P0 (it is implicit here, so we assert non-empty states carry it by
    # construction -- the assignment excludes P0 from the tuple but P0 is always run).
    for state, variants in assignment.by_state.items():
        if len(variants) == 0:
            result.failures.append(f"state {state} has no P1 variants assigned")

    return result
