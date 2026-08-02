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

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from .contracts import ParsedBridge, ParsedGap, ParsedItem, ParsedSelection

__all__ = [
    "AggregatedItem",
    "AggregatedEvidence",
    "Coster",
    "stable_union_v1",
    "coverage_budget_v1",
    "budget_pack_v1",
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
    """A published span and the (facet, role) relations the selector gave it.

    ``relations`` is the ONLY stored annotation. It previously sat alongside ``role`` and
    ``facet_ids`` as three fields describing one fact, and they diverged exactly as two
    representations of one fact eventually do: preflight compared role/facet_ids while the
    renderer preferred relations, so an item with `role="support"` and
    `relations=(("f1","contradict"),)` passed preflight and published "contradict".

    ``role`` and ``facet_ids`` remain as derived properties for the aggregators' ordering, but
    nothing can set them independently.
    """

    span_id: str
    relations: tuple[tuple[str, Optional[str]], ...] = ()

    @property
    def role(self) -> Optional[str]:
        """The single role, when the span plays exactly one. None if it plays several.

        Callers that make a *decision* from this must handle None: a span supporting one facet
        and contradicting another legitimately has no single role, and reading None as "no
        role" is what made the contradiction guard fail on the very evidence it protects.
        """
        roles = {r for _, r in self.relations if r}
        return roles.pop() if len(roles) == 1 else None

    @property
    def facet_ids(self) -> tuple[str, ...]:
        from .contracts import OVERALL_FACET

        return tuple(f for f, _ in self.relations if f != OVERALL_FACET)

    def roles_on(self, facet: str) -> set[str]:
        return {r for f, r in self.relations if f == facet and r}

    @classmethod
    def from_parsed(cls, item) -> "AggregatedItem":
        return cls(span_id=item.span_id, relations=item.relations)


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
    if span.get("namespace") == "VISIBLE_MESSAGE":
        # The compressor saw one byte stream.  message_id does not encode its order and may be
        # random, so it cannot be the primary key for C_VISIBLE publication.
        return (
            "VISIBLE_MESSAGE",
            span.get("visible_compressor_view_hash", ""),
            span.get("byte_start", 0),
            span.get("byte_end", 0),
            span.get("visible_span_id") or span.get("span_id", ""),
        )
    return (
        span.get("namespace", ""),
        _source_key(span),
        span.get("char_start", span.get("byte_start", 0)),
        span.get("char_end", span.get("byte_end", 0)),
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
    _selection_rank: Optional[dict[str, int]] = None,
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
    # Read from relations, not from a single role. A span can support one facet and contradict
    # another, and collapsing that to one value both loses the conflict and misranks the item.
    facet_roles: dict[str, set[str]] = {}
    for item in items:
        for facet, role in item.relations:
            if role:
                facet_roles.setdefault(facet, set()).add(role)
    contradiction_facets = {f for f, roles in facet_roles.items() if {"support", "contradict"} <= roles}

    # Deterministic consideration order: strongest role the span plays, then source, then id.
    def rank(it) -> int:
        roles = {r for _, r in it.relations if r}
        return min((_ROLE_RANK.get(r, 2) for r in roles), default=2)

    if _selection_rank is None:
        items.sort(key=lambda it: (rank(it), _order_key(registry[it.span_id])))
    else:
        # The global selector's output order is its reranking signal. Canonical source order is
        # restored for rendering below, but budget admission must honor the ranking or
        # global_rerank_v1 is just coverage_budget_v1 under another label.
        items.sort(key=lambda it: (
            _selection_rank.get(it.span_id, len(_selection_rank)),
            rank(it),
            _order_key(registry[it.span_id]),
        ))

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
        in_conflict = any(f in contradiction_facets for f, _ in item.relations)
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


def budget_pack_v1(
    sel: ParsedSelection,
    registry: dict,
    *,
    token_budget: int,
    coster: Coster,
    min_sources: int = 1,
    _selection_rank: Optional[dict[str, int]] = None,
) -> AggregatedEvidence:
    """Pack under a rendered-token budget that is never exceeded, and prove it before returning.

    `coverage_budget_v1` force-admits contradiction pairs and the source-diversity minimum even
    when they overflow, on the reasoning that preflight will catch it and the sample will fall
    back to P0. Under a per-page budget that was a rare escape hatch. Under one whole-batch
    budget it is the common case -- and it is the measured mechanism behind Freeze-1's 100%
    H-boundary preflight rejection -- so this aggregator removes it.

    The replacement is **pair-atomic admission with rollback**: a contradiction facet's spans
    form one unit that is admitted or dropped together. That is legal because `preflight` rejects
    a *one-sided* contradiction, not a *dropped* one -- dropping both sides costs nothing, while
    keeping both over budget costs the entire batch. `min_sources` has no preflight check at all,
    so it can only ever be a preference, and here it is expressed as consideration order rather
    than as a forced admission.

    **The guarantee, stated exactly.** The returned object is bit-for-bit the last object
    `coster` measured, and that measurement was at or under `token_budget`. Nothing more is
    claimed: the cost function is neither monotone nor submodular, because a contiguous run
    shares one `SOURCE:` header (so adding a span can make the render *cheaper*) and a bridge is
    re-admitted only once all its spans survive (so adding a span can cost far more than its own
    rows). Greedy therefore has no approximation ratio here, and the distance to an optimal pack
    is *measured* against an exhaustive packer rather than asserted away.

    At most 64 exact costings run per call, because the selector schema caps `selected_ids` at
    that -- independent of how many candidates the view offered.
    """
    for item in sel.items:
        _require(registry, item.span_id)

    dedup: dict[str, ParsedItem] = {}
    for item in sel.items:
        dedup.setdefault(item.span_id, item)
    items = list(dedup.values())
    if not items:
        return AggregatedEvidence(items=(), gaps=sel.gaps, bridges=sel.bridges)

    # Facets the selector marked as a live disagreement. Read from relations rather than from a
    # single role: a span may support one facet and contradict another, and collapsing that to
    # one value both loses the conflict and misranks the item.
    facet_roles: dict[str, set[str]] = {}
    for item in items:
        for facet, role in item.relations:
            if role:
                facet_roles.setdefault(facet, set()).add(role)
    contradiction_facets = {
        f for f, roles in facet_roles.items() if {"support", "contradict"} <= roles
    }

    def rank(it) -> int:
        roles = {r for _, r in it.relations if r}
        return min((_ROLE_RANK.get(r, 2) for r in roles), default=2)

    def sort_key(span_id: str) -> tuple:
        it = dedup[span_id]
        selection_rank = (
            _selection_rank.get(span_id, len(_selection_rank))
            if _selection_rank is not None else 0
        )
        return (selection_rank, rank(it), _order_key(registry[span_id]))

    units = _atomic_units(items, contradiction_facets)
    units.sort(key=lambda unit: min(sort_key(span_id) for span_id in unit))
    units = _promote_new_sources(units, registry=registry, min_sources=min_sources)

    kept: list[str] = []
    kept_ids: set[str] = set()

    def cost_of(span_ids: list[str]) -> int:
        ordered = sorted(span_ids, key=lambda sid: _order_key(registry[sid]))
        trial = AggregatedEvidence(
            items=tuple(AggregatedItem.from_parsed(dedup[sid]) for sid in ordered),
            gaps=sel.gaps,
            bridges=_bridges_supported_by(sel.bridges, set(span_ids)),
        )
        return coster(trial, registry)

    for unit in units:
        trial = kept + [sid for sid in unit if sid not in kept_ids]
        if not trial or len(trial) == len(kept):
            continue
        if cost_of(trial) <= token_budget:
            kept = trial
            kept_ids = set(trial)

    ordered_ids = sorted(kept_ids, key=lambda sid: _order_key(registry[sid]))
    bridges = _bridges_supported_by(sel.bridges, kept_ids)
    # Exactly the ledger `preflight._provenance_errors` demands: everything the selector chose
    # that did not survive. Computed from the two sets rather than accumulated in the loop, so a
    # unit that was tried and rolled back cannot be double-counted or missed.
    dropped = tuple(sid for sid in dedup if sid not in kept_ids)

    packed = AggregatedEvidence(
        items=tuple(AggregatedItem.from_parsed(dedup[sid]) for sid in ordered_ids),
        gaps=sel.gaps, bridges=bridges,
        dropped_for_budget=dropped,
        dropped_bridges=tuple(b.text for b in sel.bridges if b not in bridges),
    )
    final = coster(packed, registry)
    if final > token_budget:
        # Cheap proof that the loop's invariant held on the object actually returned, rather
        # than on some trial that resembled it. Reaching this means the ordering or the bridge
        # set diverged between costing and assembly, which is a defect in this function -- not a
        # budget the selector overshot -- so it must not be reported as a P1 budget failure.
        raise AggregationError(
            f"budget_pack_v1 assembled {final} rendered tokens against a {token_budget} budget; "
            "the packed object differs from the trial that was costed")
    return packed


def _atomic_units(
    items: Sequence[ParsedItem], contradiction_facets: set[str]
) -> list[list[str]]:
    """Group span ids into sets that must be admitted or dropped together.

    A contradiction facet's spans are one unit, and units sharing a span merge -- a span can sit
    on two live disagreements, and admitting it for one while dropping it for the other would
    one-side the second. Everything else is its own unit.
    """
    parent: dict[str, str] = {item.span_id: item.span_id for item in items}

    def find(node: str) -> str:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    by_facet: dict[str, list[str]] = {}
    for item in items:
        for facet, _role in item.relations:
            if facet in contradiction_facets:
                by_facet.setdefault(facet, []).append(item.span_id)
    for members in by_facet.values():
        for other in members[1:]:
            union(members[0], other)

    grouped: dict[str, list[str]] = {}
    for item in items:
        grouped.setdefault(find(item.span_id), []).append(item.span_id)
    return list(grouped.values())


def _promote_new_sources(
    units: list[list[str]], *, registry: dict, min_sources: int
) -> list[list[str]]:
    """Pull the best-ranked unit of each new source forward, up to ``min_sources`` sources.

    Breadth expressed as consideration order, not as a forced admission: a promoted unit that
    does not fit is still dropped. At ``min_sources=1`` -- the only value any registered variant
    uses -- the top-ranked unit is already the first new source, so this returns the input
    unchanged and costs nothing.
    """
    if min_sources <= 1:
        return units
    promoted: list[list[str]] = []
    remainder: list[list[str]] = []
    seen: set[str] = set()
    for unit in units:
        sources = {_source_key(registry[span_id]) for span_id in unit}
        fresh = sources - seen
        if fresh and len(seen) < min_sources:
            seen |= fresh
            promoted.append(unit)
        else:
            remainder.append(unit)
    return promoted + remainder


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
    if scores is None:
        # Structured output has no score field. Its stable list order is therefore the only
        # ranking the global model emitted; discarding it made this aggregator behaviorally
        # identical to coverage_budget_v1.
        rank_by_id = {item.span_id: i for i, item in enumerate(sel.items)}
    else:
        ranked = sorted(
            sel.items,
            key=lambda it: (-scores.get(it.span_id, 0.0), _order_key(registry[it.span_id])),
        )
        rank_by_id = {item.span_id: i for i, item in enumerate(ranked)}
    reordered = sorted(sel.items, key=lambda it: rank_by_id[it.span_id])
    reranked = ParsedSelection(
        contract=sel.contract, items=tuple(reordered), gaps=sel.gaps, bridges=sel.bridges
    )
    return coverage_budget_v1(
        reranked, registry, token_budget=token_budget, coster=coster,
        _selection_rank=rank_by_id,
    )
