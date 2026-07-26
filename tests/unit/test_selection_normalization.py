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
from shapeflow_p1.p1.contracts import (
    SelectionContractError,
    canonical_normalization_document,
    parse_selection,
)
from shapeflow_p1.p1.preflight import PreflightConfig, preflight
from shapeflow_p1.p1.view import CandidateViewRecord, ViewConstructionError

TOK = WhitespaceTokenizer()
SOURCE = "Cats are feline animals here. Dogs are canine animals here. Birds can surely fly here."
CH = "c" * 64
OCC = "occ1"


def _spans():
    return [
        build_evidence_span(c, SOURCE, content_hash=CH, source_occurrence_ids=[OCC],
                            chunker_version="v1")
        for c in paragraph_sentence_v1(SOURCE, tokenizer=TOK, max_tokens=6)
    ]


def _fixture(spans=None):
    spans = spans if spans is not None else _spans()
    view = _view(spans)
    return view, [c.span_id for c in view.candidates], view.candidate_set


def _text_of(span: dict) -> str:
    return SOURCE[span["char_start"]:span["char_end"]]


def _view(spans, **kw):
    """Build the offered view. Several checks that preflight used to make now happen HERE.

    That is the point of the move: a candidate the selector must not see has to be rejected
    before the model call, not after -- once the model has read it, no downstream check can
    undo what it read.
    """
    kw.setdefault("namespace", "RAW_SOURCE")
    kw.setdefault("snapshot_texts", {CH: SOURCE})
    kw.setdefault("query_attempts", [("q_a", "cats"), ("q_b", "dogs")])
    kw.setdefault("token_budget", 10_000)
    kw.setdefault("contract", "P1_ID")
    kw.setdefault("topic", "feline animals")
    return CandidateViewRecord.build(spans=spans, tokenizer=TOK, **kw)


def _run_preflight(sel, agg, view, *, budget=10_000, **kw):
    return preflight(
        selection=sel, aggregated=agg, view=view, known_occurrence_ids={OCC},
        config=PreflightConfig(selected_token_budget=budget, **kw),
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
    view, _, cs = _fixture()
    raw = {"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": order[0], "facet_ids": ["f"]},
        {"span_id": "E1", "role": order[1], "facet_ids": ["f"]},
    ]}
    with pytest.raises(SelectionContractError, match="conflicting roles") as caught:
        parse_selection(raw, cs)
    assert caught.value.normalization.semantic_conflict_count == 1
    assert caught.value.normalization.rejected_reason == "semantic_conflict"


def test_exact_duplicate_selection_is_normalized_not_rejected():
    """An exactly repeated line is harmless redundancy; rejecting it would only bias to P0."""
    view, span_ids, cs = _fixture()
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
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["diet"]},
        {"span_id": "E1", "role": "support", "facet_ids": ["origin"]},
    ]}, cs)
    assert len(sel.items) == 1
    assert sel.items[0].facet_ids == ("diet", "origin")
    assert sel.items[0].relations == (("diet", "support"), ("origin", "support"))
    assert sel.normalization.duplicate_count == 0  # two distinct facets, nothing repaired


def test_p1_id_exact_duplicates_are_deduped_stably():
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E3", "E1", "E1", "E3"]}, cs)
    assert sel.selected_span_ids == (span_ids[2], span_ids[0])  # first-seen order preserved
    assert sel.normalization.raw_count == 4
    assert sel.normalization.duplicate_count == 2


def test_out_of_set_id_keeps_the_all_offered_invalid_attempt_record():
    _, _, cs = _fixture()
    with pytest.raises(SelectionContractError, match="not in the offered") as caught:
        parse_selection(
            {"contract": "P1_ID", "selected_ids": ["E1", "E99"]}, cs)
    record = caught.value.normalization
    assert record.raw_count == 2
    assert record.rejected_reason == "out_of_set_label"
    assert record.duplicate_count == 0


def test_normalization_wire_document_recomputes_derived_flags():
    document = canonical_normalization_document({
        "raw_count": 2,
        "unique_count": 1,
        "duplicate_count": 1,
        "semantic_conflict_count": 0,
        "rejected_reason": None,
    })
    assert document["was_repaired"] is True
    assert document["was_rejected"] is False
    assert document["strict_valid"] is False
    with pytest.raises(ValueError, match="not closed"):
        canonical_normalization_document({
            **document,
            "strict_valid": True,
        })


def test_repeated_gap_facet_unions_its_query_attempts():
    view, _, cs = _fixture()
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
    view, _, cs = _fixture()
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
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["f1"]},
        {"span_id": "E2", "role": "contradict", "facet_ids": ["f2"]},
    ]}, cs)
    out = _run_preflight(sel, stable_union_v1(sel, view.registry), view).rendered.text
    # One shared SOURCE header for the run...
    assert out.count("SOURCE:") == 1
    # ...but each span keeps its own label line with its own relation.
    first_label = f"[{view.publication_handle_for(span_ids[0])}] (support:f1)"
    second_label = f"[{view.publication_handle_for(span_ids[1])}] (contradict:f2)"
    assert first_label in out
    assert second_label in out
    # And each label line immediately precedes its own bytes.
    lines = [ln for ln in out.splitlines() if ln.strip()]
    assert lines[lines.index(first_label) + 1].startswith("Cats are feline")
    assert lines[lines.index(second_label) + 1].startswith("Dogs are canine")


def test_render_keeps_each_spans_own_breadcrumb():
    from shapeflow_p1.hashing import sha256_hex

    spans = _spans()
    # Breadcrumbs are addressed ranges into the span's own source, so each resolves to real
    # snapshot bytes and is re-derived by the view rather than trusted.
    spans[0]["heading_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode())}]
    spans[1]["heading_refs"] = [{
        "content_hash": CH, "char_start": 30, "char_end": 34,
        "text_sha256": sha256_hex(SOURCE[30:34].encode())}]
    view, span_ids, cs = _fixture(spans)
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    res = _run_preflight(sel, stable_union_v1(sel, view.registry), view)
    assert res.ok, res.errors
    assert f"under: {SOURCE[0:4]}" in res.rendered.text
    assert f"under: {SOURCE[30:34]}" in res.rendered.text


# --- 3. nothing reaches a prompt unless it reconstructs from frozen bytes ---------------


def test_injected_unhashed_context_is_rejected():
    """A free-string context field is an unbound channel straight into the prompt.

    With `context` as plain text, tampering left the span id unchanged, RAW_SOURCE
    reconstruction passing and preflight ok -- while the injected string was rendered
    downstream. Context must address frozen bytes like any other evidence.

    The rejection now happens when the VIEW is built, which is strictly earlier: context is
    rendered into the selector prompt, so catching it at publish time would already be one
    model call too late.
    """
    spans = _spans()
    spans[0]["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": "0" * 64,        # does not match SOURCE[0:4]
    }]
    with pytest.raises(ViewConstructionError, match="hash mismatch"):
        _view(spans)


def test_faithful_context_ref_reconstructs_and_renders():
    from shapeflow_p1.hashing import sha256_hex

    spans = _spans()
    spans[0]["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode()),
    }]
    view, span_ids, cs = _fixture(spans)
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    res = _run_preflight(sel, stable_union_v1(sel, view.registry), view)
    assert res.ok, res.errors
    assert f"ctx| {SOURCE[0:4]}" in res.rendered.text


def test_out_of_bounds_context_ref_is_rejected():
    spans = _spans()
    spans[0]["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 99_999,
        "text_sha256": "0" * 64,
    }]
    with pytest.raises(ViewConstructionError, match="out of bounds"):
        _view(spans)


# --- 4. preflight verifies what is published -------------------------------------------


def test_preflight_recomputes_the_render_itself():
    """Preflight must not accept a caller's word for what the render cost.

    Taking a RenderResult as input means the budget gate checks a number the caller supplied,
    so a caller that renders one thing and reports another passes. Preflight renders the
    aggregated evidence itself, in canonical order, and gates on that.
    """
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2", "E3"]}, cs)
    agg = stable_union_v1(sel, view.registry)
    res = _run_preflight(sel, agg, view, budget=1)
    assert not res.ok
    assert any("exceeds selected_token_budget" in e for e in res.errors)
    # And it reports the render it actually verified, so the caller publishes that exact text.
    ok = _run_preflight(sel, agg, view, budget=10_000)
    assert ok.ok and ok.rendered is not None
    # Every span's exact frozen bytes appear, resolved by the view rather than by a caller.
    for candidate in view.candidates:
        assert candidate.text in ok.rendered.text


def test_preflight_rejects_evidence_the_selector_never_chose():
    """Aggregated output must be a subset of the resolved selection.

    An aggregator bug (or a strategy assembling evidence by hand) could otherwise publish a
    span the model never selected, and every downstream metric would attribute it to the
    selector.
    """
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    smuggled = AggregatedEvidence(items=(
        AggregatedItem(span_ids[0]), AggregatedItem(span_ids[1]),
    ))
    res = _run_preflight(sel, smuggled, view)
    assert not res.ok
    assert any("not in the resolved selection" in e for e in res.errors)


def test_preflight_checks_contradiction_against_published_roles():
    """The contradiction guard must read what was published, not what was requested.

    Checking the pre-aggregation selection lets an aggregator drop one side while preflight
    still sees both in the selection and passes.
    """
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["diet"]},
        {"span_id": "E2", "role": "contradict", "facet_ids": ["diet"]},
    ]}, cs)
    one_sided = AggregatedEvidence(items=(AggregatedItem(span_ids[0], (("diet", "support"),)),))
    res = _run_preflight(sel, one_sided, view)
    assert not res.ok
    assert any("one-sided" in e for e in res.errors)


def test_costing_uses_canonical_order_so_the_estimate_is_the_published_cost():
    """The cost and the render must agree on order, or the budget gate is off by the
    difference between them. Both now come from the view, so they cannot diverge."""
    view, span_ids, cs = _fixture()
    forward = AggregatedEvidence(items=tuple(AggregatedItem(s) for s in span_ids))
    reverse = AggregatedEvidence(items=tuple(AggregatedItem(s) for s in reversed(span_ids)))
    assert view.cost(forward) == view.cost(reverse)
    assert view.cost(forward) == view.render(forward).token_count
