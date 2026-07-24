"""Publication integrity: what reaches the model, and what the model's choice may become.

These defend one property, from both ends:

    Everything the selector sees, and everything published from what it chose, must be
    derivable from frozen bytes and from the selector's own normalized output.

Each test is a reproduction of a hole that passed the previous suite. The recurring shape is a
*callable or field the caller controls* sitting on the path between frozen storage and the
prompt -- a `source_text_for` that can return anything, a `heading_path` bound to nothing, an
aggregator free to rewrite the annotations it was given. Hashing the span body alone does not
close any of them, because none of them goes through the span body.
"""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import (
    WhitespaceTokenizer,
    markdown_structure_v1,
    paragraph_sentence_v1,
)
from shapeflow_p1.evidence.identity import CandidateSet, build_evidence_span
from shapeflow_p1.hashing import sha256_hex
from shapeflow_p1.p1.aggregators import AggregatedEvidence, AggregatedItem, stable_union_v1
from shapeflow_p1.p1.contracts import (
    OVERALL_FACET,
    ParsedBridge,
    ParsedGap,
    SelectionContractError,
    parse_selection,
)
from shapeflow_p1.p1.preflight import (
    PreflightConfig,
    candidate_view_sha,
    preflight,
)

TOK = WhitespaceTokenizer()
SOURCE = "Cats are feline animals here. Dogs are canine animals here. Birds can surely fly here."
CH = "c" * 64
OCC = "occ1"


def _fixture():
    spans = [
        build_evidence_span(c, SOURCE, content_hash=CH, source_occurrence_ids=[OCC],
                            chunker_version="v1")
        for c in paragraph_sentence_v1(SOURCE, tokenizer=TOK, max_tokens=6)
    ]
    registry = {s["span_id"]: s for s in spans}
    span_ids = [s["span_id"] for s in spans]
    return registry, span_ids, CandidateSet.build(span_ids, ["q_a", "q_b"])


def _pf(sel, agg, registry, cs, **kw):
    return preflight(
        selection=sel, aggregated=agg, registry=registry,
        snapshot_texts={CH: SOURCE}, known_occurrence_ids={OCC}, tokenizer=TOK,
        label_for=cs.label_for, config=PreflightConfig(selected_token_budget=10_000), **kw
    )


# --- 1. preflight reconstructs the body; it does not ask for it ------------------------


def test_preflight_does_not_accept_a_body_supplier():
    """`source_text_for` was a caller-controlled callable on the publication path.

    Preflight rendered the evidence itself -- but pulled every span's *body* from a callable
    the caller supplied, so returning "INJECTED BODY" produced ok=True with the injected string
    in the verified render. Hashing the span body cannot catch this: the hash is checked against
    the snapshot, and then a different string is rendered.

    Preflight now resolves bodies from `snapshot_texts` / `visible_views` by namespace, so the
    bytes it verifies and the bytes it renders are the same bytes.
    """
    import inspect

    sig = inspect.signature(preflight)
    assert "source_text_for" not in sig.parameters, (
        "preflight must reconstruct bodies itself, not accept a supplier for them"
    )


def test_preflight_renders_the_frozen_bytes():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    res = _pf(sel, stable_union_v1(sel, registry), registry, cs)
    assert res.ok
    assert "Cats are feline animals here." in res.rendered.text


# --- 2. context and heading are addressed, and pinned to the offered set ---------------


def test_context_ref_repointed_to_other_legal_bytes_is_rejected():
    """Fixing the hash after repointing is not tampering the hash check can see.

    A context_ref moved to a different, *real* passage with a correctly recomputed hash passed
    every check and published text the chunker never associated with that span. Per-field
    hashing cannot detect this; only a digest over the whole offered candidate view can, since
    the view is what the selector was actually shown.
    """
    registry, span_ids, cs = _fixture()
    span = registry[span_ids[0]]
    span["context_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode()),
    }]
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    view_sha = candidate_view_sha(registry, [span_ids[0]], namespace="RAW_SOURCE")

    # Repoint to other legitimate bytes and fix the hash -- per-field integrity still holds.
    span["context_refs"] = [{
        "content_hash": CH, "char_start": 30, "char_end": 58,
        "text_sha256": sha256_hex(SOURCE[30:58].encode()),
    }]
    res = _pf(sel, agg, registry, cs, candidate_view_sha=view_sha)
    assert not res.ok
    assert any("candidate view" in e for e in res.errors)


def test_heading_path_is_addressed_not_free_text():
    """`heading_path` was an unbound list of strings rendered straight into the prompt."""
    registry, span_ids, cs = _fixture()
    span = registry[span_ids[0]]
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    view_sha = candidate_view_sha(registry, [span_ids[0]], namespace="RAW_SOURCE")
    span["heading_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode()),
    }]
    res = _pf(sel, agg, registry, cs, candidate_view_sha=view_sha)
    assert not res.ok


def test_candidate_view_sha_covers_the_whole_offered_set():
    """Every offered candidate is pinned, not only the chosen ones.

    An unselected candidate still changes what the model chose from -- a forged one can steer
    the selection it never appears in. Verification therefore happens over the whole offered
    set, before the selector call.
    """
    registry, span_ids, cs = _fixture()
    base = candidate_view_sha(registry, span_ids, namespace="RAW_SOURCE")
    # Tamper with a candidate the selection will NOT include.
    registry[span_ids[2]]["heading_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode()),
    }]
    assert candidate_view_sha(registry, span_ids, namespace="RAW_SOURCE") != base


def test_candidate_view_sha_covers_order_labels_namespace_and_versions():
    registry, span_ids, cs = _fixture()
    base = candidate_view_sha(registry, span_ids, namespace="RAW_SOURCE")
    # Reordering changes the labels the model saw, so it changes the view.
    assert candidate_view_sha(registry, list(reversed(span_ids)), namespace="RAW_SOURCE") != base
    # Namespace is part of the view: the same span ids offered as a different namespace is a
    # different experiment (C_VISIBLE vs C_REGISTRY).
    assert candidate_view_sha(registry, span_ids, namespace="VISIBLE_MESSAGE") != base


# --- 3. the real chunker -> span -> candidate chain --------------------------------------


def test_table_context_survives_the_whole_real_chain():
    """chunk.context_ranges -> span.context_refs -> Candidate.context was broken in the middle.

    `Candidate.from_span` still read the old free-text `context` key, which the schema no longer
    has, so real table rows reached the selector with no header at all. The previous test used a
    hand-built span carrying that dead field, which hid the break.
    """
    from shapeflow_p1.p1.selectors import Candidate

    table = "| Name | Value |\n|------|-------|\n| a | 1 |\n"
    row = next(c for c in markdown_structure_v1(table, tokenizer=TOK, max_tokens=100)
               if c.kind == "table_row")
    assert row.context_ranges, "chunker must emit the header range"
    span = build_evidence_span(row, table, content_hash="d" * 64,
                               source_occurrence_ids=["o"], chunker_version="v1")
    assert span["context_refs"], "span must carry the header as an addressed ref"
    cand = Candidate.from_span(span, row.text(table), snapshot_texts={"d" * 64: table})
    assert cand.context, "the header must reach the selector"
    assert any("Name" in c and "Value" in c for c in cand.context)


# --- 4. the aggregator may drop, never rewrite or invent -------------------------------


def test_aggregator_may_not_rewrite_role_or_facet():
    """Publication carried annotations preflight never compared to the selection.

    An aggregator (or a hand-assembling strategy) could turn `support/efficacy` into
    `background/safety`, and every downstream metric would attribute the rewrite to the model.
    """
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["f1"]},
    ]}, cs)
    forged = AggregatedEvidence(items=(AggregatedItem(span_ids[0], "background", ("f2",)),))
    res = _pf(sel, forged, registry, cs)
    assert not res.ok
    assert any("annotation" in e for e in res.errors)


def test_aggregator_may_not_invent_a_gap_or_bridge():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    with_gap = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0], None, ()),),
        gaps=(ParsedGap(facet_id="invented", query_attempt_ids=("q_a",)),),
    )
    assert not _pf(sel, with_gap, registry, cs).ok
    with_bridge = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0], None, ()),),
        bridges=(ParsedBridge(text="Invented.", evidence_span_ids=(span_ids[0],)),),
    )
    assert not _pf(sel, with_bridge, registry, cs).ok


def test_p1_id_output_may_not_acquire_roles_facets_gaps_or_bridges():
    """P1_ID isolates the pointer mechanism; anything else in its output is a different arm."""
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    typed = AggregatedEvidence(items=(AggregatedItem(span_ids[0], "support", ("f",)),))
    res = _pf(sel, typed, registry, cs)
    assert not res.ok


def test_dropped_for_budget_must_equal_the_real_difference():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    lying = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0], None, ()),),
        dropped_for_budget=(),          # E2 vanished but nothing was recorded
    )
    res = _pf(sel, lying, registry, cs)
    assert not res.ok
    assert any("dropped_for_budget" in e for e in res.errors)


# --- 5. role conflicts are scoped to a facet -------------------------------------------


def test_one_span_may_support_one_facet_and_contradict_another():
    """A blanket "same span, two roles -> reject" penalises exactly the hard tasks.

    A source can perfectly well support an efficacy claim and undercut a safety claim. Rejecting
    the whole sample there fails P1 on the conflict-rich tasks the study cares most about.
    """
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["efficacy"]},
        {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
    ]}, cs)
    assert len(sel.items) == 1
    assert set(sel.items[0].relations) == {("efficacy", "support"), ("safety", "contradict")}


def test_one_span_may_not_take_two_roles_on_the_same_facet():
    registry, span_ids, cs = _fixture()
    with pytest.raises(SelectionContractError, match="conflicting roles on facet"):
        parse_selection({"contract": "P1_TYPED", "selections": [
            {"span_id": "E1", "role": "support", "facet_ids": ["safety"]},
            {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
        ]}, cs)


def test_facetless_relations_use_an_explicit_sentinel_not_a_missing_key():
    """A role claimed with no facet is an overall relation, and conflicts with another one."""
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support"},
    ]}, cs)
    assert sel.items[0].relations == ((OVERALL_FACET, "support"),)
    with pytest.raises(SelectionContractError, match="conflicting roles on facet"):
        parse_selection({"contract": "P1_TYPED", "selections": [
            {"span_id": "E1", "role": "support"},
            {"span_id": "E1", "role": "contradict"},
        ]}, cs)


def test_render_shows_each_facet_role_relation():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["efficacy"]},
        {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
    ]}, cs)
    res = _pf(sel, stable_union_v1(sel, registry), registry, cs)
    assert res.ok
    assert "support:efficacy" in res.rendered.text
    assert "contradict:safety" in res.rendered.text


# --- 6. the normalization record is the strict-valid rate ------------------------------


def test_intra_field_duplicates_are_counted():
    """Repair inside a field was invisible: facet_ids=["f","f"] reported duplicate_count=0."""
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED",
                           "selections": [{"span_id": "E1", "role": "support",
                                           "facet_ids": ["f", "f"]}],
                           "gaps": [{"facet_id": "g", "query_attempt_ids": ["Q1", "Q1"]}]}, cs)
    assert sel.items[0].facet_ids == ("f",)
    assert sel.normalization.duplicate_count >= 2
    assert sel.normalization.was_repaired is True


def test_bridge_evidence_duplicates_are_counted():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_BRIDGE",
                           "selections": [{"span_id": "E1", "role": "support"}],
                           "bridges": [{"text": "x", "evidence_ids": ["E1", "E1"]}]}, cs)
    assert sel.normalization.duplicate_count >= 1


def test_a_rejected_parse_still_reports_what_it_saw():
    """A conflict raised and took its counters with it, so failures never reached the rate.

    The strict-valid rate needs the denominator: how often output was malformed, not only how
    often we managed to repair it.
    """
    registry, span_ids, cs = _fixture()
    try:
        parse_selection({"contract": "P1_TYPED", "selections": [
            {"span_id": "E1", "role": "support", "facet_ids": ["safety"]},
            {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
        ]}, cs)
    except SelectionContractError as e:
        assert e.normalization is not None
        assert e.normalization.semantic_conflict_count == 1
        assert e.normalization.raw_count == 2
    else:
        pytest.fail("expected a conflict")


# --- 7. an integrity failure never raises out of preflight ------------------------------


def test_missing_context_snapshot_fails_closed_instead_of_raising():
    """Preflight recorded the error and then rendered anyway, raising KeyError.

    The adapter's whole-batch fallback needs a decision object; an exception escaping preflight
    means the batch's failure path is whatever the caller's `except` happens to be.
    """
    registry, span_ids, cs = _fixture()
    registry[span_ids[0]]["context_refs"] = [{
        "content_hash": "z" * 64, "char_start": 0, "char_end": 3, "text_sha256": "0" * 64,
    }]
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    res = _pf(sel, stable_union_v1(sel, registry), registry, cs)
    assert not res.ok
    assert res.rendered is None


def test_missing_body_snapshot_fails_closed():
    registry, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    res = preflight(
        selection=sel, aggregated=stable_union_v1(sel, registry), registry=registry,
        snapshot_texts={}, known_occurrence_ids={OCC}, tokenizer=TOK,
        label_for=cs.label_for, config=PreflightConfig(selected_token_budget=10_000),
    )
    assert not res.ok
    assert res.rendered is None


# --- 8. the renderer's grouping behaviour is frozen -------------------------------------


def test_renderer_grouping_version_is_pinned_in_config():
    """Sharing a SOURCE header changes P1's materialized-token delta against P0.

    Declaring it "enters the protocol hash" in a comment does nothing; it has to be a config
    value that config_sha actually digests.
    """
    from pathlib import Path

    import yaml

    from shapeflow_p1.p1.renderer import RENDERER_GROUPING_VERSION

    configs = Path(__file__).resolve().parents[2] / "configs"
    data = yaml.safe_load((configs / "variants.yaml").read_text(encoding="utf-8"))
    assert data["renderer"]["grouping_version"] == RENDERER_GROUPING_VERSION
