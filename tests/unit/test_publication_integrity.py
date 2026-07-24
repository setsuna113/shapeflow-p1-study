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


def _view(spans=None, **kw):
    kw.setdefault("namespace", "RAW_SOURCE")
    kw.setdefault("snapshot_texts", {CH: SOURCE})
    kw.setdefault("query_attempts", [("q_a", "cats"), ("q_b", "dogs")])
    kw.setdefault("token_budget", 10_000)
    kw.setdefault("contract", "P1_ID")
    kw.setdefault("topic", "feline animals")
    return CandidateViewRecord.build(
        spans=spans if spans is not None else _spans(), tokenizer=TOK, **kw)


def _fixture(spans=None):
    view = _view(spans)
    return view, [c.span_id for c in view.candidates], view.candidate_set


def _pf(sel, agg, view, **kw):
    return preflight(selection=sel, aggregated=agg, view=view, known_occurrence_ids={OCC},
                     config=PreflightConfig(selected_token_budget=10_000, **kw))


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
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    res = _pf(sel, stable_union_v1(sel, view.registry), view)
    assert res.ok
    assert "Cats are feline animals here." in res.rendered.text


# --- 2. context and heading are addressed, and pinned to the offered set ---------------


def test_context_ref_repointed_to_other_legal_bytes_is_rejected():
    """Fixing the hash after repointing is not tampering the hash check can see.

    A context_ref moved to a different, *real* passage with a correctly recomputed hash passed
    every per-field check. The view stops it a step earlier and more simply: context must
    address the same source as its span, and the view resolves it once, so a later mutation of
    the registry cannot agree with what the selector was shown (see drift_errors).
    """
    spans = _spans()
    other = "d" * 64
    spans[0]["context_refs"] = [{
        "content_hash": other, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex("Dogs".encode()),
    }]
    with pytest.raises(ViewConstructionError, match="same source"):
        _view(spans, snapshot_texts={CH: SOURCE, other: "Dogs are canine."})


def test_a_registry_mutated_after_the_view_was_built_is_caught():
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, view.registry)
    view.registry[span_ids[0]]["char_end"] = view.registry[span_ids[0]]["char_start"] + 3
    res = _pf(sel, agg, view)
    assert not res.ok
    assert any("drifted" in e for e in res.errors)


def test_heading_path_is_addressed_not_free_text():
    """`heading_path` was an unbound list of strings rendered straight into the prompt."""
    spans = _spans()
    spans[0]["heading_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4, "text_sha256": "0" * 64,
    }]
    with pytest.raises(ViewConstructionError, match="hash mismatch"):
        _view(spans)


def test_the_view_digest_covers_every_offered_candidate_and_the_versions():
    """An unselected candidate still shapes what the model picked, so it is inside the digest.

    The digest also closes over the prompt bytes and the prompt/renderer versions, and needs no
    arguments from the caller: `candidate_view_sha` previously required the renderer version to
    be passed by hand or a clean publication failed, which is not a usable protocol.
    """
    base = _view().view_sha256
    # A candidate the selection will not include still changes the view.
    spans = _spans()
    spans[2]["heading_refs"] = [{
        "content_hash": CH, "char_start": 0, "char_end": 4,
        "text_sha256": sha256_hex(SOURCE[0:4].encode()),
    }]
    assert _view(spans).view_sha256 != base
    # Order allocates the labels, so it is part of the view.
    assert _view(list(reversed(_spans()))).view_sha256 != base
    # And the versions the digest claims to cover are actually in it.
    view = _view()
    assert view.prompt_bundle_version and view.renderer_grouping_version
    assert view.prompt_sha256 == sha256_hex(view.prompt_bytes)


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
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["f1"]},
    ]}, cs)
    forged = AggregatedEvidence(items=(AggregatedItem(span_ids[0], (("f2", "contradict"),)),))
    res = _pf(sel, forged, view)
    assert not res.ok
    assert any("annotation" in e for e in res.errors)


def test_aggregator_may_not_invent_a_gap_or_bridge():
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    with_gap = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0]),),
        gaps=(ParsedGap(facet_id="invented", query_attempt_ids=("q_a",)),),
    )
    assert not _pf(sel, with_gap, view).ok
    with_bridge = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0]),),
        bridges=(ParsedBridge(text="Invented.", evidence_span_ids=(span_ids[0],)),),
    )
    assert not _pf(sel, with_bridge, view).ok


def test_p1_id_output_may_not_acquire_roles_facets_gaps_or_bridges():
    """P1_ID isolates the pointer mechanism; anything else in its output is a different arm."""
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    typed = AggregatedEvidence(items=(AggregatedItem(span_ids[0], (("f", "support"),)),))
    res = _pf(sel, typed, view)
    assert not res.ok


def test_dropped_for_budget_must_equal_the_real_difference():
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    lying = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0]),),
        dropped_for_budget=(),          # E2 vanished but nothing was recorded
    )
    res = _pf(sel, lying, view)
    assert not res.ok
    assert any("dropped_for_budget" in e for e in res.errors)


# --- 5. role conflicts are scoped to a facet -------------------------------------------


def test_one_span_may_support_one_facet_and_contradict_another():
    """A blanket "same span, two roles -> reject" penalises exactly the hard tasks.

    A source can perfectly well support an efficacy claim and undercut a safety claim. Rejecting
    the whole sample there fails P1 on the conflict-rich tasks the study cares most about.
    """
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["efficacy"]},
        {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
    ]}, cs)
    assert len(sel.items) == 1
    assert set(sel.items[0].relations) == {("efficacy", "support"), ("safety", "contradict")}


def test_one_span_may_not_take_two_roles_on_the_same_facet():
    view, span_ids, cs = _fixture()
    with pytest.raises(SelectionContractError, match="conflicting roles on facet"):
        parse_selection({"contract": "P1_TYPED", "selections": [
            {"span_id": "E1", "role": "support", "facet_ids": ["safety"]},
            {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
        ]}, cs)


def test_facetless_relations_use_an_explicit_sentinel_not_a_missing_key():
    """A role claimed with no facet is an overall relation, and conflicts with another one."""
    view, span_ids, cs = _fixture()
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
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED", "selections": [
        {"span_id": "E1", "role": "support", "facet_ids": ["efficacy"]},
        {"span_id": "E1", "role": "contradict", "facet_ids": ["safety"]},
    ]}, cs)
    res = _pf(sel, stable_union_v1(sel, view.registry), view)
    assert res.ok, res.errors
    assert "support:efficacy" in res.rendered.text
    assert "contradict:safety" in res.rendered.text


# --- 6. the normalization record is the strict-valid rate ------------------------------


def test_intra_field_duplicates_are_counted():
    """Repair inside a field was invisible: facet_ids=["f","f"] reported duplicate_count=0."""
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_TYPED",
                           "selections": [{"span_id": "E1", "role": "support",
                                           "facet_ids": ["f", "f"]}],
                           "gaps": [{"facet_id": "g", "query_attempt_ids": ["Q1", "Q1"]}]}, cs)
    assert sel.items[0].facet_ids == ("f",)
    assert sel.normalization.duplicate_count >= 2
    assert sel.normalization.was_repaired is True


def test_bridge_evidence_duplicates_are_counted():
    view, span_ids, cs = _fixture()
    sel = parse_selection({"contract": "P1_BRIDGE",
                           "selections": [{"span_id": "E1", "role": "support"}],
                           "bridges": [{"text": "x", "evidence_ids": ["E1", "E1"]}]}, cs)
    assert sel.normalization.duplicate_count >= 1


def test_a_rejected_parse_still_reports_what_it_saw():
    """A conflict raised and took its counters with it, so failures never reached the rate.

    The strict-valid rate needs the denominator: how often output was malformed, not only how
    often we managed to repair it.
    """
    view, span_ids, cs = _fixture()
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


def test_an_unresolvable_candidate_is_refused_before_the_model_sees_it():
    """Preflight recorded the error and then rendered anyway, raising KeyError.

    Both halves are now stronger: an unresolvable body or context makes the VIEW refuse to
    build, so the candidate never reaches a prompt; and preflight itself never raises, so the
    adapter's whole-batch fallback always has a decision object rather than an exception.
    """
    spans = _spans()
    spans[0]["context_refs"] = [{
        "content_hash": "z" * 64, "char_start": 0, "char_end": 3, "text_sha256": "0" * 64,
    }]
    with pytest.raises(ViewConstructionError):
        _view(spans)
    # A body that cannot be resolved is likewise refused, not rendered as an empty string.
    with pytest.raises(ViewConstructionError, match="cannot resolve"):
        _view(snapshot_texts={})


def test_preflight_returns_a_decision_rather_than_raising():
    from shapeflow_p1.p1.view import guard_publication

    def boom():
        raise RuntimeError("structural failure inside rendering")

    errors = guard_publication(boom)
    assert errors and "structural failure" in errors[0]


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
