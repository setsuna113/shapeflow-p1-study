"""Semantic normalization of selector output, and the provenance of everything rendered.

Every test here is a reproduction of a defect that passed the previous suite while changing
what the experiment concludes. They are grouped by the property they defend:

1. A repeated label must not let the *order* of a model's output decide the result.
2. Per-span annotations must survive rendering; a reader has to know which role belongs to
   which span.
3. Nothing reaches a prompt unless it is reconstructible from frozen bytes.
4. Preflight must verify what is actually published, not a RenderResult handed to it.
"""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer, paragraph_sentence_v1
from shapeflow_p1.evidence.identity import CandidateSet, build_evidence_span
from shapeflow_p1.p1.aggregators import AggregatedEvidence, AggregatedItem, stable_union_v1
from shapeflow_p1.p1.contracts import SelectionContractError, parse_selection
from shapeflow_p1.p1.preflight import PreflightConfig, preflight
from shapeflow_p1.p1.renderer import make_coster, render

TOK = WhitespaceTokenizer()
SOURCE = "Cats are feline animals here. Dogs are canine animals here. Birds can surely fly here."
CH = "c" * 64
OCC = "occ1"


def _fixture():
    chunks = paragraph_sentence_v1(SOURCE, tokenizer=TOK, max_tokens=6)
    spans = [
        build_evidence_span(c, SOURCE, content_hash=CH, source_occurrence_ids=[OCC],
                            chunker_version="v1")
        for c in chunks
    ]
    registry = {s["span_id"]: s for s in spans}
    span_ids = [s["span_id"] for s in spans]
    cs = CandidateSet.build(span_ids, ["q_a", "q_b"])
    return registry, span_ids, cs


def _text_of(span: dict) -> str:
    return SOURCE[span["char_start"]:span["char_end"]]


def _run_preflight(sel, agg, registry, cs, *, budget=10_000, **kw):
    return preflight(
        selection=sel, aggregated=agg, registry=registry,
        snapshot_texts={CH: SOURCE}, known_occurrence_ids={OCC}, tokenizer=TOK,
        label_for=cs.label_for,
        config=PreflightConfig(selected_token_budget=budget), **kw
    )


# --- 1. duplicate labels may not make the outcome order-dependent ---------------------


@pytest.mark.parametrize("order", [("support", "contradict"), ("contradict", "support")])
def test_same_span_with_conflicting_roles_is_rejected(order):
    """The same span cannot be both support and contradict for one facet.

    Under first-wins dedup the survivor was whichever the model happened to emit first, so
    swapping two lines of model output flipped the recorded role -- and preflight passed both
    times. Worse, the pair looks like a preserved contradiction to the contradiction guard,
    manufacturing a false positive for the property the study most wants to measure.
    """
    registry, _, cs = _fixture()
    raw = {"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": order[0], "facet_ids": ["f"]},
        {"span_id": "E1", "role": order[1], "facet_ids": ["f"]},
    ]}
    with pytest.raises(SelectionContractError, match="conflicting roles"):
        parse_selection(raw, cs)


def test_exact_duplicate_selection_is_normalized_not_rejected():
    """An exactly repeated line is harmless redundancy; rejecting it would only bias to P0."""
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["f"]},
        {"span_id": "E1", "role": "support", "facet_ids": ["f"]},
    ]}, cs)
    assert sel.selected_span_ids == (span_ids[0],)
    assert sel.normalization.raw_count == 2
    assert sel.normalization.unique_count == 1
    assert sel.normalization.duplicate_count == 1
    assert sel.normalization.semantic_conflict_count == 0


def test_same_span_same_role_different_facets_merges_by_a_stated_rule():
    """Two facets on one span+role is a coherent claim, so it merges -- deterministically.

    The merge is union in first-seen order, which is the auditable rule; first-wins silently
    discarded the second facet and the coverage metric then scored a facet the selector had
    actually addressed as missed.
    """
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["diet"]},
        {"span_id": "E1", "role": "support", "facet_ids": ["origin"]},
    ]}, cs)
    assert len(sel.items) == 1
    assert sel.items[0].facet_ids == ("diet", "origin")
    assert sel.items[0].relations == (("diet", "support"), ("origin", "support"))
    assert sel.normalization.duplicate_count == 0  # two distinct facets, nothing repaired


def test_p1_id_exact_duplicates_are_deduped_stably():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E3", "E1", "E1", "E3"]}, cs)
    assert sel.selected_span_ids == (span_ids[2], span_ids[0])  # first-seen order preserved
    assert sel.normalization.raw_count == 4
    assert sel.normalization.duplicate_count == 2


def test_repeated_gap_facet_unions_its_query_attempts():
    registry, _, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED",
                           "selections": [{"span_id": "E1", "role": "support"}],
                           "gaps": [
                               {"facet_id": "origin", "query_attempt_ids": ["Q1"]},
                               {"facet_id": "origin", "query_attempt_ids": ["Q2", "Q1"]},
                           ]}, cs)
    assert len(sel.gaps) == 1
    assert sel.gaps[0].query_attempt_ids == ("q_a", "q_b")


def test_normalization_counts_are_recorded_for_the_strict_valid_rate():
    """The campaign reports how often output needed repair, and a strict no-repair sensitivity.

    Normalizing silently would let a variant that emits malformed output look identical to one
    that does not, hiding a real quality difference between contracts.
    """
    registry, _, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E1"]}, cs)
    assert sel.normalization.was_repaired is True
    clean = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    assert clean.normalization.was_repaired is False


# --- 2. per-span annotations must survive rendering ------------------------------------


def test_render_keeps_role_and_facet_attached_to_their_own_span():
    """Merging a run's annotations destroys the mapping the downstream reader needs.

    Collapsing to `[E1,E2] (support,contradict) facets:f1,f2` leaves no way to tell which role
    or facet belongs to which span, nor which body text is which label -- and a contradiction
    the selector correctly marked becomes unreadable.
    """
    registry, span_ids, cs = _fixture()
    ev = AggregatedEvidence(items=(
        AggregatedItem(span_ids[0], "support", ("f1",)),
        AggregatedItem(span_ids[1], "contradict", ("f2",)),
    ))
    out = render(ev, registry, source_text_for=_text_of, label_for=cs.label_for,
                 tokenizer=TOK).text
    # One shared SOURCE header for the run...
    assert out.count("SOURCE:") == 1
    # ...but each span keeps its own label line with its own role and facet.
    assert "[E1] (support) facets: f1" in out
    assert "[E2] (contradict) facets: f2" in out
    assert "(support,contradict)" not in out
    # And each label line immediately precedes its own bytes.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines[lines.index("[E1] (support) facets: f1") + 1].startswith("Cats are feline")
    assert lines[lines.index("[E2] (contradict) facets: f2") + 1].startswith("Dogs are canine")


def test_render_keeps_each_spans_own_breadcrumb():
    registry, span_ids, cs = _fixture()
    from shapeflow_p1.hashing import sha256_hex
    # Breadcrumbs are addressed ranges, so each resolves to real snapshot bytes.
    registry[span_ids[0]]["heading_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode())}]
    registry[span_ids[1]]["heading_refs"] = [{
        "content_hash": CH, "char_start": 30, "char_end": 34,
        "text_sha256": sha256_hex(SOURCE[30:34].encode())}]
    ev = AggregatedEvidence(items=(
        AggregatedItem(span_ids[0], None, ()), AggregatedItem(span_ids[1], None, ()),
    ))
    res = _run_preflight(
        parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs), ev, registry, cs)
    assert res.ok, res.errors
    assert f"under: {SOURCE[0:4]}" in res.rendered.text
    assert f"under: {SOURCE[30:34]}" in res.rendered.text


# --- 3. nothing reaches a prompt unless it reconstructs from frozen bytes ---------------


def test_injected_unhashed_context_is_rejected():
    """A free-string context field is an unbound channel straight into the prompt.

    With `context` as plain text, tampering left the span id unchanged, RAW_SOURCE
    reconstruction passing and preflight ok -- while the injected string was rendered
    downstream. Context must address frozen bytes like any other evidence.
    """
    registry, span_ids, cs = _fixture()
    registry[span_ids[0]]["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": "0" * 64,        # does not match SOURCE[0:4]
    }]
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    res = _run_preflight(sel, agg, registry, cs)
    assert not res.ok
    assert any("context" in e for e in res.errors)


def test_faithful_context_ref_reconstructs_and_renders():
    registry, span_ids, cs = _fixture()
    from shapeflow_p1.hashing import sha256_hex
    registry[span_ids[0]]["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode()),
    }]
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    assert _run_preflight(sel, agg, registry, cs).ok


def test_out_of_bounds_context_ref_is_rejected():
    registry, span_ids, cs = _fixture()
    registry[span_ids[0]]["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 99_999,
        "text_sha256": "0" * 64,
    }]
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    assert not _run_preflight(sel, agg, registry, cs).ok


# --- 4. preflight verifies what is published -------------------------------------------


def test_preflight_recomputes_the_render_itself():
    """Preflight must not accept a caller's word for what the render cost.

    Taking a RenderResult as input means the budget gate checks a number the caller supplied,
    so a caller that renders one thing and reports another passes. Preflight renders the
    aggregated evidence itself, in canonical order, and gates on that.
    """
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2", "E3"]}, cs)
    agg = stable_union_v1(sel, registry)
    res = _run_preflight(sel, agg, registry, cs, budget=1)
    assert not res.ok
    assert any("exceeds selected_token_budget" in e for e in res.errors)
    # And it reports the render it actually verified, so the caller publishes that exact text.
    ok = _run_preflight(sel, agg, registry, cs, budget=10_000)
    assert ok.ok and ok.rendered is not None
    # Every span's exact frozen bytes appear, resolved by preflight itself.
    for sid in span_ids:
        assert _text_of(registry[sid]) in ok.rendered.text


def test_preflight_rejects_evidence_the_selector_never_chose():
    """Aggregated output must be a subset of the resolved selection.

    An aggregator bug (or a strategy assembling evidence by hand) could otherwise publish a
    span the model never selected, and every downstream metric would attribute it to the
    selector.
    """
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    smuggled = AggregatedEvidence(items=(
        AggregatedItem(span_ids[0], None, ()), AggregatedItem(span_ids[1], None, ()),
    ))
    res = _run_preflight(sel, smuggled, registry, cs)
    assert not res.ok
    assert any("not in the resolved selection" in e for e in res.errors)


def test_preflight_checks_contradiction_against_published_roles():
    """The contradiction guard must read what was published, not what was requested.

    Checking the pre-aggregation selection lets an aggregator drop one side while preflight
    still sees both in the selection and passes.
    """
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["diet"]},
        {"span_id": "E2", "role": "contradict", "facet_ids": ["diet"]},
    ]}, cs)
    one_sided = AggregatedEvidence(items=(AggregatedItem(span_ids[0], "support", ("diet",)),))
    res = _run_preflight(sel, one_sided, registry, cs)
    assert not res.ok
    assert any("one-sided" in e for e in res.errors)


def test_costing_uses_canonical_order_so_the_estimate_is_the_published_cost():
    """The coster and the renderer must agree on order, or the budget gate is off by the
    difference between them."""
    registry, span_ids, cs = _fixture()
    coster = make_coster(source_text_for=_text_of, label_for=cs.label_for, tokenizer=TOK)
    forward = AggregatedEvidence(items=tuple(AggregatedItem(s, None, ()) for s in span_ids))
    reverse = AggregatedEvidence(
        items=tuple(AggregatedItem(s, None, ()) for s in reversed(span_ids))
    )
    assert coster(forward, registry) == coster(reverse, registry)
