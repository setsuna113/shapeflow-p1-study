"""Supervisor-trajectory metrics (plan §13.4).

P1 changes the researcher's path, so trajectory is measured -- but the metrics are bounded and
interpretable, and none of them require P1 to reproduce P0's text or its exact path. They describe
*how* the search unfolded (waves, fan-out, query redundancy, facet follow-up, retrieval failure),
so a shift shows up as a described change rather than a penalty for diverging.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

__all__ = ["Trajectory", "query_redundancy", "facet_followup_rate", "retrieval_failure_rate",
           "summarize_trajectory"]

_WORD = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Trajectory:
    waves: int
    conduct_research_calls: int
    queries: tuple[str, ...]
    required_facets: tuple[str, ...]
    followed_up_facets: tuple[str, ...]
    close_reasons: tuple[str, ...]
    failed_retrievals: int
    total_retrievals: int


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b) if (a | b) else 0.0


def query_redundancy(queries: tuple[str, ...], *, threshold: float = 0.8) -> float:
    """Fraction of queries that are near-duplicates (Jaccard over word sets >= threshold) of an
    earlier query. High redundancy means the searcher circled rather than broadened."""
    if len(queries) < 2:
        return 0.0
    token_sets = [set(_WORD.findall(q.lower())) for q in queries]
    redundant = 0
    for i in range(1, len(token_sets)):
        if any(_jaccard(token_sets[i], token_sets[j]) >= threshold for j in range(i)):
            redundant += 1
    return redundant / len(queries)


def facet_followup_rate(traj: Trajectory) -> float:
    """Fraction of required facets that received a follow-up. NA (returns 1.0) when there are no
    required facets."""
    if not traj.required_facets:
        return 1.0
    followed = set(traj.followed_up_facets)
    hit = sum(1 for f in traj.required_facets if f in followed)
    return hit / len(traj.required_facets)


def retrieval_failure_rate(traj: Trajectory) -> float:
    if traj.total_retrievals == 0:
        return 0.0
    return traj.failed_retrievals / traj.total_retrievals


def summarize_trajectory(traj: Trajectory) -> dict[str, float]:
    return {
        "waves": float(traj.waves),
        "fan_out": float(traj.conduct_research_calls),
        "query_count": float(len(traj.queries)),
        "query_redundancy": query_redundancy(traj.queries),
        "facet_followup_rate": facet_followup_rate(traj),
        "retrieval_failure_rate": retrieval_failure_rate(traj),
        "premature_completes": float(
            sum(1 for r in traj.close_reasons if r == "RESEARCH_COMPLETE")
            if facet_followup_rate(traj) < 1.0 else 0
        ),
    }
