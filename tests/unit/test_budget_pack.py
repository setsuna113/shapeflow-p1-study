"""`budget_pack_v1` never returns an object it did not cost at or under budget.

`coverage_budget_v1` force-admits contradiction pairs and the source-diversity minimum even when
they overflow, on the reasoning that preflight will catch it and the sample falls back to P0.
Under one whole-batch budget that stops being an escape hatch and becomes the failure mode: it is
the measured mechanism behind Freeze-1's 100% H-boundary preflight rejection.

So the property under test is feasibility, and the cases that matter are the ones where a naive
greedy would violate it -- a contradiction pair that does not fit, a cost function that is not
monotone because contiguous spans share a `SOURCE:` header, and a bridge whose re-admission costs
more than the span that completed it.

The costers here are stand-ins with the *shape* of the real one, not models of it. The real
`view.coster()` is the renderer, and `test_p1.py` already pins `view.cost` against the rendered
token count; what these need to be is adversarial, which a real render is not on demand.
"""

from __future__ import annotations

import pytest

from shapeflow.p1.aggregators import (
    AggregatedEvidence,
    AggregationError,
    budget_pack_v1,
    coverage_budget_v1,
)
from shapeflow.p1.contracts import ParsedBridge, ParsedItem, ParsedSelection


def _span(span_id: str, *, source: str = "s1", start: int = 0) -> dict:
    return {
        "span_id": span_id,
        "namespace": "RAW_SOURCE",
        "content_hash": source,
        "char_start": start,
        "char_end": start + 10,
    }


def _registry(*specs: tuple[str, str, int]) -> dict:
    return {sid: _span(sid, source=source, start=start) for sid, source, start in specs}


def _selection(*items: ParsedItem, bridges: tuple[ParsedBridge, ...] = ()) -> ParsedSelection:
    return ParsedSelection(contract="P1_TYPED", items=tuple(items), bridges=bridges)


def _item(span_id: str, *relations: tuple[str, str]) -> ParsedItem:
    return ParsedItem(span_id=span_id, relations=tuple(relations))


def _flat_coster(per_item: int = 10):
    """Each item costs the same; the whole object costs their sum."""
    return lambda evidence, _registry: per_item * len(evidence.items)


def test_a_contradiction_pair_that_does_not_fit_is_dropped_whole_not_forced_in():
    """The case that produced Freeze-1's H rejections, and the reason for the rollback.

    Preflight rejects a *one-sided* contradiction, not a dropped one. So dropping both sides is
    legal and free, while keeping both over budget costs the entire batch.
    """
    registry = _registry(("a", "s1", 0), ("b", "s1", 20), ("c", "s2", 0))
    selection = _selection(
        _item("a", ("f1", "support")),
        _item("b", ("f1", "contradict")),
        _item("c", ("f2", "support")),
    )

    packed = budget_pack_v1(selection, registry, token_budget=15, coster=_flat_coster(10))

    assert {it.span_id for it in packed.items} == {"c"}, "the pair must not be half-admitted"
    assert set(packed.dropped_for_budget) == {"a", "b"}
    assert _flat_coster(10)(packed, registry) <= 15

    # And the aggregator it replaces does exactly what this exists to stop.
    forced = coverage_budget_v1(selection, registry, token_budget=15, coster=_flat_coster(10))
    assert _flat_coster(10)(forced, registry) > 15, (
        "coverage_budget_v1 is expected to overshoot here; if it no longer does, this "
        "aggregator's reason for existing has changed")


def test_a_contradiction_pair_that_fits_is_kept_whole():
    registry = _registry(("a", "s1", 0), ("b", "s1", 20))
    selection = _selection(_item("a", ("f1", "support")), _item("b", ("f1", "contradict")))

    packed = budget_pack_v1(selection, registry, token_budget=25, coster=_flat_coster(10))

    assert {it.span_id for it in packed.items} == {"a", "b"}
    assert packed.dropped_for_budget == ()


def test_spans_on_two_live_disagreements_merge_into_one_unit():
    """`b` contradicts on f1 and supports on f2, both live. Admitting it for one facet while
    dropping it for the other would one-side the second."""
    registry = _registry(("a", "s1", 0), ("b", "s1", 20), ("c", "s1", 40), ("d", "s2", 0))
    selection = _selection(
        _item("a", ("f1", "support")),
        _item("b", ("f1", "contradict"), ("f2", "support")),
        _item("c", ("f2", "contradict")),
        _item("d", ("f3", "support")),
    )

    packed = budget_pack_v1(selection, registry, token_budget=25, coster=_flat_coster(10))
    kept = {it.span_id for it in packed.items}

    assert kept in ({"d"}, set()), f"the a/b/c unit costs 30 and cannot fit 25; got {kept}"
    assert not ({"a", "b", "c"} & kept) or {"a", "b", "c"} <= kept


def test_a_shared_source_header_making_a_span_cheaper_cannot_break_feasibility():
    """Non-monotone cost: adding a span that bridges two runs *removes* a header line.

    Greedy has no approximation ratio under this, and none is claimed. What must survive is
    feasibility, which the final re-cost proves on the object actually returned.
    """
    registry = _registry(("a", "s1", 0), ("b", "s1", 10), ("c", "s1", 20))

    def header_sharing_coster(evidence, _registry) -> int:
        # 10 per item, plus 12 per contiguous run -- so filling a gap merges two runs into one.
        starts = sorted(registry[it.span_id]["char_start"] for it in evidence.items)
        runs = 1 if starts else 0
        for previous, current in zip(starts, starts[1:]):
            if current != previous + 10:
                runs += 1
        return 10 * len(starts) + 12 * runs

    selection = _selection(_item("a"), _item("c"), _item("b"))
    packed = budget_pack_v1(selection, registry, token_budget=42,
                            coster=header_sharing_coster)

    assert header_sharing_coster(packed, registry) <= 42
    assert set(packed.dropped_for_budget) == {"a", "b", "c"} - {
        it.span_id for it in packed.items}


def test_a_bridge_readmitted_by_the_last_span_cannot_push_the_result_over():
    """A bridge counts only once every span it cites survives, so one span can cost far more
    than its own rows. The admission test prices the whole object, bridges included."""
    registry = _registry(("a", "s1", 0), ("b", "s1", 20))
    bridge = ParsedBridge(text="joins a and b", evidence_span_ids=("a", "b"))

    def bridge_coster(evidence, _registry) -> int:
        return 10 * len(evidence.items) + 100 * len(evidence.bridges)

    selection = _selection(_item("a"), _item("b"), bridges=(bridge,))
    packed = budget_pack_v1(selection, registry, token_budget=50, coster=bridge_coster)

    assert bridge_coster(packed, registry) <= 50
    assert packed.bridges == (), "the bridge costs 100 and cannot be in a 50-token object"


def test_the_dropped_ledger_is_exactly_what_preflight_demands():
    """`preflight._provenance_errors` requires
    ``dropped_for_budget == set(selected_span_ids) - published_ids`` exactly."""
    registry = _registry(("a", "s1", 0), ("b", "s1", 20), ("c", "s2", 0))
    selection = _selection(_item("a"), _item("b"), _item("c"))

    packed = budget_pack_v1(selection, registry, token_budget=15, coster=_flat_coster(10))
    published = {it.span_id for it in packed.items}

    assert set(packed.dropped_for_budget) == set(selection.selected_span_ids) - published


def test_a_duplicate_selection_is_deduped_before_the_ledger_is_computed():
    registry = _registry(("a", "s1", 0))
    selection = _selection(_item("a"), _item("a"))

    packed = budget_pack_v1(selection, registry, token_budget=10, coster=_flat_coster(10))

    assert [it.span_id for it in packed.items] == ["a"]
    assert packed.dropped_for_budget == ()


def test_an_empty_selection_packs_to_an_empty_object():
    packed = budget_pack_v1(_selection(), {}, token_budget=512, coster=_flat_coster(10))
    assert packed.items == ()


def test_nothing_fits_yields_an_empty_pack_rather_than_an_overshoot():
    """An empty publication is a recorded outcome; an over-budget one fails the whole batch."""
    registry = _registry(("a", "s1", 0))
    packed = budget_pack_v1(_selection(_item("a")), registry, token_budget=5,
                            coster=_flat_coster(10))

    assert packed.items == ()
    assert packed.dropped_for_budget == ("a",)


def test_the_result_is_returned_in_canonical_order_not_ranked_order():
    """Ranking governs admission only; presentation stays in source/offset order, as every
    other aggregator does. Otherwise the rendered bytes would encode the selector's ranking."""
    registry = _registry(("a", "s1", 0), ("b", "s1", 20), ("c", "s1", 40))
    selection = _selection(_item("c"), _item("a"), _item("b"))

    packed = budget_pack_v1(selection, registry, token_budget=100, coster=_flat_coster(10),
                            _selection_rank={"c": 0, "a": 1, "b": 2})

    assert [it.span_id for it in packed.items] == ["a", "b", "c"]


def test_the_selection_rank_decides_which_span_survives_a_tight_budget():
    """Without this the shootout would compare packers, not selectors: every candidate would be
    admitted in document order regardless of what it ranked first."""
    registry = _registry(("a", "s1", 0), ("b", "s1", 20))
    selection = _selection(_item("a"), _item("b"))

    first = budget_pack_v1(selection, registry, token_budget=10, coster=_flat_coster(10),
                           _selection_rank={"b": 0, "a": 1})
    assert [it.span_id for it in first.items] == ["b"]

    second = budget_pack_v1(selection, registry, token_budget=10, coster=_flat_coster(10),
                            _selection_rank={"a": 0, "b": 1})
    assert [it.span_id for it in second.items] == ["a"]


def test_a_coster_that_disagrees_with_itself_is_refused_not_published():
    """The final re-cost is a proof about the object returned, not about a trial resembling it.

    A coster whose answer depends on call order would let the loop believe it stayed under
    budget while the assembled object did not. That is a defect in the packer's own assembly,
    so it must surface as an aggregation error rather than as a P1 budget failure the selector
    gets blamed for.
    """
    registry = _registry(("a", "s1", 0))
    calls = {"n": 0}

    def drifting(evidence: AggregatedEvidence, _registry) -> int:
        calls["n"] += 1
        return 1 if calls["n"] == 1 else 9_999

    with pytest.raises(AggregationError, match="differs from the trial"):
        budget_pack_v1(_selection(_item("a")), registry, token_budget=10, coster=drifting)


def test_at_most_one_costing_per_unit_plus_the_final_proof():
    """The exact coster is the expensive call. Bounding it by the selection size -- capped at 64
    by the output schema -- is what keeps it independent of the ~880 candidates a whole-batch
    view offers."""
    registry = _registry(*[(f"s{n}", "s1", n * 20) for n in range(20)])
    selection = _selection(*[_item(f"s{n}") for n in range(20)])
    calls = {"n": 0}

    def counting(evidence, reg):
        calls["n"] += 1
        return 10 * len(evidence.items)

    budget_pack_v1(selection, registry, token_budget=100, coster=counting)

    assert calls["n"] <= 20 + 1, f"{calls['n']} costings for 20 selected spans"
