"""Supervisor-trajectory metrics (plan §13.4).

P1 changes the researcher's path, so trajectory is measured -- but the metrics are bounded and
interpretable, and none of them require P1 to reproduce P0's text or its exact path. They describe
*how* the search unfolded (waves, fan-out, query redundancy, facet follow-up, retrieval failure),
so a shift shows up as a described change rather than a penalty for diverging.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = ["Trajectory", "query_redundancy", "facet_followup_rate", "retrieval_failure_rate",
           "summarize_trajectory", "summarize_frozen_events"]

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


def summarize_frozen_events(events: Sequence[Mapping]) -> dict:
    """Recompute bounded end-to-end trajectory outcomes from the frozen event stream.

    This deliberately does not ask P1 to reproduce P0's path.  Counts, redundancy, retrieval
    failures, fan-out and close behaviour are outcomes of the treatment.  The only equality
    checks here are internal integrity checks on one arm's own event indices and hashes.

    Metrics prefixed ``post_treatment_`` are descriptive for a treated arm only: P0 has no
    treatment boundary, so those fields are ``None`` and must not be used as a P0 contrast.
    The total-trajectory metrics are comparable across all arms.
    """
    if isinstance(events, str | bytes) or not isinstance(events, Sequence):
        raise ValueError("trajectory events must be an ordered sequence")

    normalized: list[dict] = []
    for expected_index, raw in enumerate(events):
        if not isinstance(raw, Mapping):
            raise ValueError(f"trajectory event {expected_index} is not an object")
        event = dict(raw)
        if event.get("event_index") != expected_index:
            raise ValueError(
                f"trajectory event index {event.get('event_index')!r} != {expected_index}"
            )
        position = str(event.get("position") or "")
        if position not in {"PRE_TREATMENT", "TREATMENT", "POST_TREATMENT"}:
            raise ValueError(
                f"trajectory event {expected_index} has invalid position {position!r}"
            )
        recorded_sha = str(event.pop("event_sha256", "") or "")
        if len(recorded_sha) != 64 or recorded_sha != sha256_hex(canonical_json(event)):
            raise ValueError(f"trajectory event {expected_index} does not verify")
        event["event_sha256"] = recorded_sha
        normalized.append(event)

    queries = [event for event in normalized if event.get("kind") == "SEARCH_QUERY"]
    query_texts = tuple(str(event.get("query") or "") for event in queries)
    result_counts: list[int] = []
    occurrence_ids: set[str] = set()
    for index, event in enumerate(queries):
        raw_count = event.get("result_count")
        if (
            not isinstance(raw_count, int)
            or isinstance(raw_count, bool)
            or raw_count < 0
        ):
            raise ValueError(f"SEARCH_QUERY event {index} has invalid result_count")
        result_counts.append(raw_count)
        ids = event.get("source_occurrence_ids")
        if not isinstance(ids, list) or any(
            not isinstance(value, str) or not value for value in ids
        ):
            raise ValueError(
                f"SEARCH_QUERY event {index} lacks valid source occurrence lineage"
            )
        occurrence_ids.update(ids)

    h_reductions = [
        event for event in normalized if event.get("kind") == "PAGE_BATCH_REDUCED"
    ]
    close_events = [
        event for event in normalized if str(event.get("kind") or "").startswith("CLOSE_")
    ]
    decisions = [
        event for event in normalized if event.get("kind") == "MODEL_TOOL_DECISION"
    ]
    tool_names = [
        str(name)
        for event in decisions
        for name in (event.get("tool_names") or ())
        if str(name)
    ]
    first_treatment = next(
        (
            int(event["event_index"])
            for event in normalized
            if event.get("position") == "TREATMENT"
        ),
        None,
    )
    post_queries = [
        event for event in queries if event.get("position") == "POST_TREATMENT"
    ]
    close_reason = next(
        (
            str(event.get("close_reason") or "")
            for event in reversed(close_events)
            if event.get("close_reason")
        ),
        "",
    )

    # The production hook emits one NODE_SELECTION event per selector stage and then one
    # PAGE_BATCH_REDUCED/CLOSE_* event for the same checkpoint.  Counting event rows would
    # therefore count a single checkpoint (and its fallback/failure) two or more times.  Treat
    # the checkpoint identity as the unit, while retaining compatibility with the explicit
    # *_CHECKPOINT kinds used by older frozen probes.
    def _node_and_checkpoint(event: Mapping) -> tuple[str, str]:
        kind = str(event.get("kind") or "")
        direct = event.get("direct_node_record")
        direct = direct if isinstance(direct, Mapping) else {}
        node = str(direct.get("node") or event.get("node") or "").upper()
        if node.startswith("H"):
            node = "H"
        elif node.startswith("C"):
            node = "C"
        if not node:
            if kind.startswith("PAGE_") or kind in {"H_CHECKPOINT", "NODE_SELECTION_H"}:
                node = "H"
            elif kind.startswith("CLOSE_") or kind in {
                "C_CHECKPOINT", "NODE_SELECTION_C",
            }:
                node = "C"
        checkpoint = str(
            event.get("checkpoint")
            or event.get("checkpoint_hash")
            or direct.get("checkpoint_hash")
            or direct.get("checkpoint")
            or ""
        )
        return node, checkpoint

    checkpoint_kinds = {
        "NODE_SELECTION", "NODE_SELECTION_H", "NODE_SELECTION_C",
        "PAGE_BATCH_REDUCED", "H_CHECKPOINT", "C_CHECKPOINT",
        "CLOSE_REDUCED", "CLOSE_FAILED", "CLOSE_DEFERRED_TO_VENDOR",
        "CLOSE_CANCELLED",
    }
    checkpoint_units: set[tuple[str, str]] = set()
    fallback_units: set[tuple[str, str]] = set()
    failure_units: set[tuple[str, str]] = set()
    for event in normalized:
        kind = str(event.get("kind") or "")
        if kind not in checkpoint_kinds:
            continue
        node, checkpoint = _node_and_checkpoint(event)
        if node in {"H", "C"} and checkpoint:
            checkpoint_units.add((node, checkpoint))
        direct = event.get("direct_node_record")
        direct = direct if isinstance(direct, Mapping) else {}
        fell_back = bool(event.get("fell_back") or direct.get("fell_back"))
        failure = event.get("failure") or direct.get("failure")
        if fell_back or failure:
            if node not in {"H", "C"} or not checkpoint:
                raise ValueError(
                    "fallback/failure event lacks a node-scoped checkpoint identity"
                )
            unit = (node, checkpoint)
            if fell_back:
                fallback_units.add(unit)
            if failure:
                failure_units.add(unit)

    query_count = len(queries)
    return {
        "schema_version": "frozen_trajectory_metrics_v1",
        "status": "OK",
        "event_count": float(len(normalized)),
        "query_count": float(query_count),
        "unique_query_count": float(len(set(query_texts))),
        "query_redundancy": query_redundancy(query_texts),
        "retrieved_result_count": float(sum(result_counts)),
        "unique_source_occurrence_count": float(len(occurrence_ids)),
        "retrieval_empty_rate": (
            sum(count == 0 for count in result_counts) / query_count
            if query_count else 0.0
        ),
        "research_rounds": float(len(h_reductions)),
        "tool_calls_observed": float(
            sum(int(event.get("siblings") or 0) for event in h_reductions)
        ),
        "model_tool_decision_count": float(len(decisions)),
        "conduct_research_calls": float(tool_names.count("ConductResearch")),
        "think_calls": float(tool_names.count("think_tool")),
        "research_complete_calls": float(tool_names.count("ResearchComplete")),
        "h_checkpoint_count": float(sum(node == "H" for node, _ in checkpoint_units)),
        "c_checkpoint_count": float(sum(node == "C" for node, _ in checkpoint_units)),
        "fallback_count": float(len(fallback_units)),
        "failure_count": float(len(failure_units)),
        "first_treatment_event_index": first_treatment,
        "post_treatment_query_count": (
            float(len(post_queries)) if first_treatment is not None else None
        ),
        "post_treatment_retrieved_result_count": (
            float(sum(
                int(event.get("result_count") or 0) for event in post_queries
            ))
            if first_treatment is not None else None
        ),
        "close_reason": close_reason,
        "trajectory_sha256": sha256_hex(canonical_json(normalized)),
    }
