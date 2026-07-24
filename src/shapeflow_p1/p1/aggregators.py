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

# Estimates rendered tokens for a whole evidence object given the span registry. It takes the
# full object, not just the items, because gaps and bridges are rendered too -- costing items
# alone understates the P1 budget by exactly the output that distinguishes the contracts.
Coster = Callable[["AggregatedEvidence", dict], int]


class AggregationError(ValueError):
    pass


@dataclass(frozen=True)
class AggregatedItem:
    """A published span, carrying exactly the annotations the selector gave it.

    ``relations`` is the authoritative (facet, role) list; ``role``/``facet_ids`` are the
    convenience views preflight compares against the selection. An aggregator may drop an item;
    it may never edit these, and preflight enforces that.
    """

    span_id: str
    role: Optional[str] = None
    facet_ids: tuple[str, ...] = ()
    relations: tuple[tuple[str, Optional[str]], ...] = ()

    @classmethod
    def from_parsed(cls, item) -> "AggregatedItem":
        return cls(span_id=item.span_id, role=item.role,
                   facet_ids=item.facet_ids, relations=item.relations)


@dataclass(frozen=True)
class AggregatedEvidence:
    items: tuple[AggregatedItem, ...]
    gaps: tuple[ParsedGap, ...] = ()
    bridges: tuple[ParsedBridge, ...] = ()
    dropped_for_budget: tuple[str, ...] = ()
    # Bridges whose bound evidence did not survive the budget. A bridge is connective text
    # *about* specific spans; keeping it after those spans are gone would leave an unsourced
    # assertion in the output and a citation pointing at nothing.
    dropped_bridges: tuple[str, ...] = ()


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
    span records themselves.

    Adjacent same-source spans are *not* fused into a synthetic span here. A merged span would
    have no entry in the candidate registry, no recorded ``text_sha256`` and therefore nothing
    for preflight to reconstruct -- it would defeat the integrity check it passes through. The
    duplicated-header cost that fusing was meant to avoid is instead removed at render time,
    where contiguous runs from one source share a single header (see ``renderer.render``).
    Doing it there also keeps it uniform across all three aggregators, so it cannot become a
    confound between them.
    """
    seen: dict[str, ParsedItem] = {}
    for item in sel.items:
        _require(registry, item.span_id)
        seen.setdefault(item.span_id, item)
    ordered = sorted(seen.values(), key=lambda it: _order_key(registry[it.span_id]))
    items = tuple(AggregatedItem.from_parsed(it) for it in ordered)
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
        # Cost the whole object: the gaps and bridges are rendered alongside the items, so a
        # budget check that ignores them lets a TYPED or BRIDGE arm overshoot by exactly the
        # part of its output that is not an item.
        trial = AggregatedEvidence(
            items=tuple(candidate), gaps=sel.gaps,
            bridges=_bridges_supported_by(sel.bridges, {it.span_id for it in candidate}),
        )
        return coster(trial, registry) <= token_budget

    for item in items:
        agg = AggregatedItem.from_parsed(item)
        source = _source_key(registry[item.span_id])
        in_conflict = any(f in contradiction_facets for f in item.facet_ids)
        # Force-admit conflict spans (keep both sides) and up to the source-diversity minimum,
        # even against a tight budget, so neither a disagreement nor breadth is silently lost.
        force = in_conflict or (len(kept_sources) < min_sources and source not in kept_sources)
        if would_fit(kept + [agg]) or force:
            kept.append(agg)
            kept_sources.add(source)
        else:
            dropped.append(item.span_id)

    ordered = sorted(kept, key=lambda it: _order_key(registry[it.span_id]))
    kept_ids = {it.span_id for it in ordered}
    bridges = _bridges_supported_by(sel.bridges, kept_ids)
    dropped_bridges = tuple(
        b.text for b in sel.bridges if b not in bridges
    )
    return AggregatedEvidence(
        items=tuple(ordered), gaps=sel.gaps, bridges=bridges,
        dropped_for_budget=tuple(dropped), dropped_bridges=dropped_bridges,
    )


def _bridges_supported_by(
    bridges: tuple[ParsedBridge, ...], kept_ids: set[str]
) -> tuple[ParsedBridge, ...]:
    """Keep only bridges whose every bound span survived.

    A bridge names specific spans. Once any of them is dropped for budget, the bridge is an
    assertion whose support is no longer in the output and whose citation resolves to nothing --
    the renderer would have to label a span that is not there. Dropping the bridge is the honest
    resolution; it is recorded in ``dropped_bridges`` rather than vanishing.
    """
    return tuple(b for b in bridges if set(b.evidence_span_ids) <= kept_ids)


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
