"""Turning a resolved selection into the ordered, budgeted evidence to render.

Three aggregators (plan §9.4), all deterministic:

- ``stable_union_v1`` -- dedup by span id and present in a stable order. A control.
- ``coverage_budget_v1`` -- enforce the token budget (over *rendered* tokens, not id count,
  so a coarse chunker can't smuggle more text for the same nominal budget), keep a minimum of
  distinct sources, and never split a declared contradiction pair -- if one side of a
  support/contradict pair is kept, the other is kept too, so the aggregator can't quietly
  one-side a disagreement.
- ``global_rerank_v1`` -- reorder a hierarchical shortlist; still emits ids only.

The token budget is checked against an injected coster that estimates the renderer's real
output, so selection and rendering agree on what "budget" means. Preflight later enforces the
*true* rendered token count as the binding gate.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from .contracts import ParsedBridge, ParsedGap, ParsedItem, ParsedSelection

__all__ = [
    "AggregatedItem",
    "AggregatedEvidence",
    "Coster",
    "stable_union_v1",
    "coverage_budget_v1",
    "global_rerank_v1",
    "AggregationError",
]

# Estimates rendered tokens for a list of items given the span registry.
Coster = Callable[[list["AggregatedItem"], dict], int]


class AggregationError(ValueError):
    pass


@dataclass(frozen=True)
class AggregatedItem:
    span_id: str
    role: Optional[str]
    facet_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class AggregatedEvidence:
    items: tuple[AggregatedItem, ...]
    gaps: tuple[ParsedGap, ...] = ()
    bridges: tuple[ParsedBridge, ...] = ()
    dropped_for_budget: tuple[str, ...] = ()


def _source_key(span: dict) -> str:
    """A stable per-source key: raw-source spans group by snapshot, visible spans by message."""
    return span.get("content_hash") or span.get("message_id") or span.get("span_id", "")


def _order_key(span: dict) -> tuple:
    return (
        span.get("namespace", ""),
        _source_key(span),
        span.get("char_start", span.get("byte_start", 0)),
        span.get("span_id") or span.get("visible_span_id", ""),
    )


def _require(registry: dict, span_id: str) -> dict:
    span = registry.get(span_id)
    if span is None:
        raise AggregationError(f"selected span {span_id[:12]} is not in the candidate registry")
    return span


def stable_union_v1(sel: ParsedSelection, registry: dict) -> AggregatedEvidence:
    """Dedup by span id, present in stable source/offset order. Byte-identical content already
    shares a span id, so this also merges duplicates while occurrences are preserved on the
    span records themselves."""
    seen: dict[str, ParsedItem] = {}
    for item in sel.items:
        _require(registry, item.span_id)
        seen.setdefault(item.span_id, item)
    ordered = sorted(seen.values(), key=lambda it: _order_key(registry[it.span_id]))
    items = tuple(AggregatedItem(it.span_id, it.role, it.facet_ids) for it in ordered)
    return AggregatedEvidence(items=items, gaps=sel.gaps, bridges=sel.bridges)


# Priority for budgeted inclusion: evidence bearing on a claim before background.
_ROLE_RANK = {"contradict": 0, "support": 1, None: 2, "background": 3}


def coverage_budget_v1(
    sel: ParsedSelection,
    registry: dict,
    *,
    token_budget: int,
    coster: Coster,
    min_sources: int = 1,
) -> AggregatedEvidence:
    """Greedily include selections under a rendered-token budget, keeping contradiction pairs
    intact and a minimum number of distinct sources."""
    for item in sel.items:
        _require(registry, item.span_id)

    dedup: dict[str, ParsedItem] = {}
    for item in sel.items:
        dedup.setdefault(item.span_id, item)
    items = list(dedup.values())

    # Facets the selector marked as a live disagreement (both a support and a contradict span).
    # Items on such a facet are force-kept as a unit, so budget pressure can never one-side a
    # conflict; if that overflows the budget, preflight catches it and the sample falls to P0.
    facet_roles: dict[str, set[str]] = {}
    for item in items:
        for facet in item.facet_ids:
            if item.role:
                facet_roles.setdefault(facet, set()).add(item.role)
    contradiction_facets = {f for f, roles in facet_roles.items() if {"support", "contradict"} <= roles}

    # Deterministic consideration order: role priority, then source, then span id.
    items.sort(key=lambda it: (_ROLE_RANK.get(it.role, 2), _order_key(registry[it.span_id])))

    kept: list[AggregatedItem] = []
    kept_sources: set[str] = set()
    dropped: list[str] = []

    def would_fit(candidate: list[AggregatedItem]) -> bool:
        return coster(candidate, registry) <= token_budget

    for item in items:
        agg = AggregatedItem(item.span_id, item.role, item.facet_ids)
        trial = kept + [agg]
        source = _source_key(registry[item.span_id])
        in_conflict = any(f in contradiction_facets for f in item.facet_ids)
        # Force-admit conflict spans (keep both sides) and up to the source-diversity minimum,
        # even against a tight budget, so neither a disagreement nor breadth is silently lost.
        force = in_conflict or (len(kept_sources) < min_sources and source not in kept_sources)
        if would_fit(trial) or force:
            kept.append(agg)
            kept_sources.add(source)
        else:
            dropped.append(item.span_id)

    ordered = sorted(kept, key=lambda it: _order_key(registry[it.span_id]))
    return AggregatedEvidence(
        items=tuple(ordered), gaps=sel.gaps, bridges=sel.bridges,
        dropped_for_budget=tuple(dropped),
    )


def global_rerank_v1(
    sel: ParsedSelection,
    registry: dict,
    *,
    token_budget: int,
    coster: Coster,
    scores: Optional[dict[str, float]] = None,
) -> AggregatedEvidence:
    """Rerank a hierarchical shortlist by an optional score (desc), then apply the same
    budgeted inclusion. Still emits ids only; its own LLM work is accounted separately by the
    caller, never hidden here."""
    scores = scores or {}
    reordered = sorted(
        sel.items,
        key=lambda it: (-scores.get(it.span_id, 0.0), _order_key(registry[it.span_id])),
    )
    reranked = ParsedSelection(
        contract=sel.contract, items=tuple(reordered), gaps=sel.gaps, bridges=sel.bridges
    )
    return coverage_budget_v1(reranked, registry, token_budget=token_budget, coster=coster)
