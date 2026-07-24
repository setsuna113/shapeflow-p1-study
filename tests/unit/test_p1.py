"""P1 treatment path: contracts, aggregators, renderer, preflight, selectors."""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer, paragraph_sentence_v1
from shapeflow_p1.evidence.identity import CandidateSet, build_evidence_span
from shapeflow_p1.odr.hooks import TaskContext
from shapeflow_p1.p1.aggregators import (
    AggregatedEvidence,
    AggregatedItem,
    coverage_budget_v1,
    global_rerank_v1,
    stable_union_v1,
)
from shapeflow_p1.p1.contracts import (
    ParsedBridge,
    ParsedGap,
    SelectionContractError,
    parse_selection,
)
from shapeflow_p1.p1.preflight import (
    PreflightConfig,
    bridge_novelty_errors,
    preflight,
)
from shapeflow_p1.p1.view import CandidateViewRecord
from shapeflow_p1.p1.selectors import (
    Candidate,
    CpuLexicalSelector,
    LlmSelector,
    SelectorInput,
    partition_by_scope,
)

TOK = WhitespaceTokenizer()
SOURCE = "Cats are feline animals here. Dogs are canine animals here. Birds can surely fly here."
CH = "c" * 64
OCC = "occ1"


def _view(spans=None, **kw):
    """The offered candidate view -- the single source for bodies, labels, cost and render."""
    if spans is None:
        spans = [
            build_evidence_span(c, SOURCE, content_hash=CH, source_occurrence_ids=[OCC],
                                chunker_version="v1")
            for c in paragraph_sentence_v1(SOURCE, tokenizer=TOK, max_tokens=6)
        ]
    kw.setdefault("namespace", "RAW_SOURCE")
    kw.setdefault("snapshot_texts", {CH: SOURCE})
    kw.setdefault("query_attempts", [("q_a", "cats")])
    kw.setdefault("token_budget", 10_000)
    kw.setdefault("contract", "P1_ID")
    kw.setdefault("topic", "feline animals")
    return CandidateViewRecord.build(spans=spans, tokenizer=TOK, **kw)


def _fixture():
    view = _view()
    return (list(view.registry.values()), view.registry,
            [c.span_id for c in view.candidates], view.candidate_set, view.coster())


TASK = TaskContext(task_id="t", protocol_sha="p", variant_id="H02", seed=1,
                   research_topic="feline animals", max_content_length=50000)


# --- contracts ------------------------------------------------------------------------


def test_parse_p1_id_resolves_labels():
    _, _, span_ids, cs, _ = _fixture()
    parsed = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    assert parsed.selected_span_ids == (span_ids[0], span_ids[1])
    assert all(item.role is None for item in parsed.items)


def test_parse_rejects_out_of_set_label():
    _, _, _, cs, _ = _fixture()
    with pytest.raises(SelectionContractError, match="not in the offered"):
        parse_selection({"contract": "P1_ID", "selected_ids": ["E99"]}, cs)


def test_parse_bridge_requires_binding_and_text():
    _, _, _, cs, _ = _fixture()
    with pytest.raises(SelectionContractError):
        parse_selection(
            {"contract": "P1_BRIDGE", "selections": [{"span_id": "E1", "role": "support"}],
             "bridges": [{"text": "  ", "evidence_ids": ["E1"]}]}, cs)


# --- aggregators ----------------------------------------------------------------------


def test_stable_union_dedups_and_orders():
    _, registry, span_ids, cs, _ = _fixture()
    sel = parse_selection(
        {"contract": "P1_ID", "selected_ids": ["E3", "E1", "E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    # deduped E1; ordered by source/offset so E1 (earlier) precedes E3.
    ordered_ids = [it.span_id for it in agg.items]
    assert ordered_ids == [span_ids[0], span_ids[2]]


def test_coverage_budget_respects_rendered_token_budget():
    _, registry, span_ids, cs, coster = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2", "E3"]}, cs)
    everything = AggregatedEvidence(
        items=tuple(AggregatedItem(s) for s in span_ids)
    )
    full = coster(everything, registry)
    tight = full // 2
    agg = coverage_budget_v1(sel, registry, token_budget=tight, coster=coster, min_sources=1)
    assert agg.dropped_for_budget  # something was dropped under the tight budget
    # what remains renders within budget
    assert _view().render(agg).token_count <= tight


def test_coverage_budget_keeps_both_sides_of_contradiction():
    _, registry, span_ids, cs, coster = _fixture()
    # E1 support, E2 contradict, on the same facet. Even at a punishing budget, both survive.
    sel = parse_selection(
        {"contract": "P1_TYPED", "selections": [
            {"span_id": "E1", "role": "support", "facet_ids": ["diet"]},
            {"span_id": "E2", "role": "contradict", "facet_ids": ["diet"]},
        ]}, cs)
    agg = coverage_budget_v1(sel, registry, token_budget=1, coster=coster, min_sources=1)
    kept = {it.span_id for it in agg.items}
    assert span_ids[0] in kept and span_ids[1] in kept


# --- renderer -------------------------------------------------------------------------


def test_renderer_deterministic_and_coster_matches():
    _, registry, span_ids, cs, coster = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    agg = stable_union_v1(sel, registry)
    view = _view()
    r1 = view.render(agg)
    r2 = view.render(agg)
    assert r1 == r2
    # the coster's estimate equals the real render token count for the same evidence
    assert view.cost(agg) == r1.token_count
    # exact span bytes appear verbatim -- the renderer never rewrites content
    assert "feline" in r1.text


def test_coster_charges_for_gaps_and_bridges():
    """The budget must cover everything the renderer emits, not just the items.

    Gap and bridge lines are exactly the output that distinguishes P1_TYPED and P1_BRIDGE from
    P1_ID. Costing items alone lets those arms overshoot their token budget by the very thing
    that makes them different, which biases the work comparison in their favour.
    """
    _, registry, span_ids, cs, coster = _fixture()
    items = (AggregatedItem(span_ids[0], (("diet", "support"),)),)
    bare = AggregatedEvidence(items=items)
    with_extras = AggregatedEvidence(
        items=items,
        gaps=(ParsedGap(facet_id="origin", query_attempt_ids=("q1",)),),
        bridges=(ParsedBridge(text="These agree on diet.", evidence_span_ids=(span_ids[0],)),),
    )
    assert coster(with_extras, registry) > coster(bare, registry)
    # And the estimate is the truth, not an approximation of it.
    assert coster(with_extras, registry) == _view().render(with_extras).token_count


def test_gap_line_distinguishes_a_failed_search_from_a_real_absence():
    """A timed-out lookup is not evidence that the world contains nothing.

    Rendering every gap as "no evidence found" turns an infrastructure failure into a factual
    claim of absence -- and the gap-honesty guard then scores that claim as correct.
    """
    _, registry, span_ids, cs, _ = _fixture()
    ev = AggregatedEvidence(
        items=(AggregatedItem(span_ids[0]),),
        gaps=(ParsedGap(facet_id="origin", query_attempt_ids=("q_timeout",)),),
    )
    failed = _view(query_status={"q_timeout": "TIMEOUT"}).render(ev)
    assert "no evidence found" not in failed.text
    assert "did not complete" in failed.text

    answered = _view(query_status={"q_timeout": "EMPTY"}).render(ev)
    assert "no evidence found" in answered.text


def test_a_facet_is_never_both_answered_and_declared_a_gap():
    """A facet is either answered or explicitly unanswered -- claiming both is incoherent.

    Left unchecked, a selector can bank the coverage credit for a facet *and* the gap-honesty
    credit for the same facet, which the quality guards score as two separate goods.
    """
    _, _, span_ids, _, _ = _fixture()
    cs = CandidateSet.build(span_ids, ["q_attempt_1"])
    with pytest.raises(SelectionContractError, match="both selected-for and declared a gap"):
        parse_selection({
            "contract": "P1_TYPED",
            "selections": [{"span_id": "E1", "role": "support", "facet_ids": ["diet"]}],
            "gaps": [{"facet_id": "diet", "query_attempt_ids": ["Q1"]}],
        }, cs)
    # The same selection with distinct facets is fine.
    ok = parse_selection({
        "contract": "P1_TYPED",
        "selections": [{"span_id": "E1", "role": "support", "facet_ids": ["diet"]}],
        "gaps": [{"facet_id": "origin", "query_attempt_ids": ["Q1"]}],
    }, cs)
    assert ok.gaps[0].query_attempt_ids == ("q_attempt_1",)


def test_adjacent_same_source_spans_share_one_header():
    """Two abutting sentences from one page render under one header, not two.

    The saving is real prompt tokens, and doing it at render time keeps every span's id, label
    and recorded hash intact -- a fused synthetic span would have no registry entry and nothing
    for preflight to reconstruct.
    """
    _, registry, span_ids, cs, _ = _fixture()
    ev = AggregatedEvidence(items=tuple(AggregatedItem(s) for s in span_ids[:2]))
    out = _view().render(ev)
    # One shared SOURCE header, but each span keeps its own label line -- see
    # test_render_keeps_role_and_facet_attached_to_their_own_span for why merging is wrong.
    assert out.text.count("SOURCE:") == 1
    assert "[E1]" in out.text and "[E2]" in out.text
    # Both spans' bytes are still present verbatim.
    assert "feline" in out.text and "canine" in out.text


# --- preflight ------------------------------------------------------------------------


def _clean_preflight(sel, agg, registry, cs, budget=10_000, view=None, **kw):
    return preflight(
        selection=sel, aggregated=agg, view=view if view is not None else _view(),
        known_occurrence_ids={OCC},
        config=PreflightConfig(selected_token_budget=budget, **kw),
    )


def test_preflight_passes_clean_selection():
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    agg = stable_union_v1(sel, registry)
    assert _clean_preflight(sel, agg, registry, cs).ok


def test_preflight_flags_tampered_offsets():
    """A corrupted recorded hash is caught even though the view resolved the body by offset.

    The view reads bodies from the snapshot, so a bad `text_sha256` no longer changes what is
    rendered -- but it still means the span record and the bytes disagree, which is corruption
    and must stop the publication rather than be quietly rendered around.
    """
    view = _view()
    cs = view.candidate_set
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, view.registry)
    view.registry[agg.items[0].span_id]["text_sha256"] = "0" * 64
    assert not _clean_preflight(sel, agg, view.registry, cs, view=view).ok


def test_preflight_flags_dangling_citation():
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    # Occurrence set does not contain OCC -> lineage closure fails.
    res = preflight(selection=sel, aggregated=agg, view=_view(),
                    known_occurrence_ids=set(),
                    config=PreflightConfig(selected_token_budget=10_000))
    assert not res.ok


def test_preflight_flags_over_budget():
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2", "E3"]}, cs)
    agg = stable_union_v1(sel, registry)
    res = _clean_preflight(sel, agg, registry, cs, budget=1)  # impossibly small
    assert not res.ok


def test_preflight_rejects_a_span_outside_the_variants_namespace():
    """C_VISIBLE publishing a RAW_SOURCE span is C_REGISTRY provenance wearing a C_VISIBLE label.

    The candidate set makes this unreachable in the normal path, but preflight is the last gate
    before publish and must not depend on an upstream promise: a bug in candidate construction
    would otherwise produce a compressor-only claim that was never compressor-only.
    """
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)   # RAW_SOURCE spans
    res = _clean_preflight(sel, agg, registry, cs, expected_namespace="VISIBLE_MESSAGE")
    assert not res.ok
    assert any("may only publish 'VISIBLE_MESSAGE'" in e for e in res.errors)


def test_preflight_flags_a_bridge_citing_a_dropped_span():
    """A bridge whose support was dropped for budget cites something not in the output.

    The aggregator drops such bridges itself, but preflight has to catch a hand-assembled or
    buggy one: otherwise the renderer is asked to label a span that is not published, and the
    downstream reader sees an assertion whose citation resolves to nothing.
    """
    _, registry, span_ids, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    tampered = AggregatedEvidence(
        items=agg.items,
        bridges=(ParsedBridge(text="Both agree.", evidence_span_ids=(span_ids[2],)),),
    )
    res = _clean_preflight(sel, tampered, registry, cs)
    assert not res.ok
    assert any("not in the published output" in e for e in res.errors)


def test_aggregator_drops_a_bridge_whose_support_lost_the_budget():
    _, registry, span_ids, cs, coster = _fixture()
    sel = parse_selection({
        "contract": "P1_BRIDGE",
        "selections": [{"span_id": "E1", "role": "support"},
                       {"span_id": "E3", "role": "background"}],
        "bridges": [{"text": "E3 qualifies E1.", "evidence_ids": ["E1", "E3"]}],
    }, cs)
    tiny = coster(AggregatedEvidence(items=(AggregatedItem(span_ids[0]),)), registry)
    agg = coverage_budget_v1(sel, registry, token_budget=tiny, coster=coster, min_sources=1)
    if span_ids[2] not in {it.span_id for it in agg.items}:
        assert agg.bridges == ()
        assert agg.dropped_bridges  # recorded, not silently vanished


# --- selectors ------------------------------------------------------------------------


def test_cpu_lexical_selector_is_deterministic_and_relevant():
    _, _, span_ids, _, _ = _fixture()
    inp = SelectorInput(
        topic="feline animals",
        candidates=[Candidate(span_ids[0], "Cats are feline animals here."),
                    Candidate(span_ids[1], "Dogs are canine animals here."),
                    Candidate(span_ids[2], "Birds can surely fly here.")],
        query_attempts=[], token_budget=1000, contract="P1_ID",
    )
    sel = CpuLexicalSelector(max_selected=1)
    out1 = sel.select(TASK, inp)
    out2 = sel.select(TASK, inp)
    assert out1 == out2  # deterministic
    # The cat sentence is most relevant to "feline animals".
    assert out1["selected_ids"] == ["E1"]


def test_llm_selector_renders_prompt_and_passes_through():
    seen = {}

    def fake_model(prompt: str) -> dict:
        seen["prompt"] = prompt
        return {"contract": "P1_ID", "selected_ids": ["E1"]}

    inp = SelectorInput(topic="feline animals",
                        candidates=[Candidate("sid1", "Cats are feline.")],
                        query_attempts=[], token_budget=500, contract="P1_ID")
    out = LlmSelector(fake_model).select(TASK, inp)
    assert out == {"contract": "P1_ID", "selected_ids": ["E1"]}
    assert "feline animals" in seen["prompt"]
    assert "[E1] Cats are feline." in seen["prompt"]


def test_selector_prompt_carries_chunker_metadata():
    """A table row is unreadable without the header that names its columns.

    The chunker computes both the heading breadcrumb and the table header; dropping them before
    the prompt made the selector guess, and the study would then have charged that guessing to
    the contract under test rather than to the missing context.
    """
    seen = {}

    def fake_model(prompt: str) -> dict:
        seen["prompt"] = prompt
        return {"contract": "P1_ID", "selected_ids": []}

    inp = SelectorInput(
        topic="revenue",
        candidates=[Candidate("sid1", "| Q3 | 41.2 |",
                              heading_path=("Financials", "Quarterly"),
                              context=("| Quarter | Revenue (M) |",))],
        query_attempts=[], token_budget=500, contract="P1_ID",
    )
    LlmSelector(fake_model).select(TASK, inp)
    assert "under: Financials > Quarterly" in seen["prompt"]
    assert "Revenue (M)" in seen["prompt"]


def test_candidate_from_span_picks_up_metadata_in_both_namespaces():
    from shapeflow_p1.hashing import sha256_hex
    snap = {"h" * 64: "Top matter"}
    raw = {"span_id": "a" * 64, "namespace": "RAW_SOURCE",
           "heading_refs": [{"content_hash": "h" * 64, "char_start": 0, "char_end": 3,
                             "text_sha256": sha256_hex(b"Top")}]}
    assert Candidate.from_span(raw, "txt", snapshot_texts=snap).heading_path == ("Top",)
    visible = {"visible_span_id": "b" * 64, "namespace": "VISIBLE_MESSAGE"}
    assert Candidate.from_span(visible, "txt", snapshot_texts={}).span_id == "b" * 64


# --- bridge novelty -------------------------------------------------------------------


def test_bridge_may_not_introduce_a_number_or_entity_its_spans_lack():
    """A bridge that adds a figure is unsourced generation wearing a citation."""
    cited = ["Revenue grew in the third quarter.", "Costs were flat."]
    assert bridge_novelty_errors("Revenue grew while costs were flat.", cited) == []
    assert any("42" in e for e in bridge_novelty_errors(
        "Revenue grew 42 percent while costs were flat.", cited))
    assert any("Acme" in e for e in bridge_novelty_errors(
        "Acme grew while costs were flat.", cited))


def test_bridge_novelty_allows_ordinary_connective_capitalization():
    cited = ["Revenue grew.", "Costs were flat."]
    assert bridge_novelty_errors("Both sources agree. However, costs were flat.", cited) == []


def test_partition_by_scope():
    groups = partition_by_scope("per_page", {"pageB": [("s2", "b")], "pageA": [("s1", "a")]})
    # deterministic order by source key
    assert groups == [[("s1", "a")], [("s2", "b")]]
    with pytest.raises(ValueError):
        partition_by_scope("nonsense", {})
