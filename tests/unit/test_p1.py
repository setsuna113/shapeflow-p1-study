"""P1 treatment path: contracts, aggregators, renderer, preflight, selectors."""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer, paragraph_sentence_v1
from shapeflow_p1.evidence.identity import CandidateSet, build_evidence_span
from shapeflow_p1.odr.hooks import TaskContext
from shapeflow_p1.p1.aggregators import (
    coverage_budget_v1,
    global_rerank_v1,
    stable_union_v1,
)
from shapeflow_p1.p1.contracts import SelectionContractError, parse_selection
from shapeflow_p1.p1.preflight import PreflightConfig, preflight
from shapeflow_p1.p1.renderer import make_coster, render
from shapeflow_p1.p1.selectors import (
    CpuLexicalSelector,
    LlmSelector,
    SelectorInput,
    partition_by_scope,
)

TOK = WhitespaceTokenizer()
SOURCE = "Cats are feline animals here. Dogs are canine animals here. Birds can surely fly here."
CH = "c" * 64
OCC = "occ1"


def _fixture():
    """Build spans/registry/candidate-set/texts from SOURCE."""
    chunks = paragraph_sentence_v1(SOURCE, tokenizer=TOK, max_tokens=6)
    spans = [
        build_evidence_span(c, SOURCE, content_hash=CH, source_occurrence_ids=[OCC],
                            chunker_version="v1")
        for c in chunks
    ]
    registry = {s["span_id"]: s for s in spans}
    span_ids = [s["span_id"] for s in spans]
    cs = CandidateSet.build(span_ids)
    coster = make_coster(
        source_text_for=lambda sp: SOURCE[sp["char_start"]:sp["char_end"]],
        label_for=cs.label_for,
        tokenizer=TOK,
    )
    return spans, registry, span_ids, cs, coster


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
    full = coster([type("I", (), {"span_id": s, "role": None, "facet_ids": ()})() for s in span_ids], registry)
    tight = full // 2
    agg = coverage_budget_v1(sel, registry, token_budget=tight, coster=coster, min_sources=1)
    assert agg.dropped_for_budget  # something was dropped under the tight budget
    # what remains renders within budget
    r = render(agg, registry, source_text_for=lambda sp: SOURCE[sp["char_start"]:sp["char_end"]],
               label_for=cs.label_for, tokenizer=TOK)
    assert r.token_count <= tight


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
    kw = dict(source_text_for=lambda sp: SOURCE[sp["char_start"]:sp["char_end"]],
              label_for=cs.label_for, tokenizer=TOK)
    r1 = render(agg, registry, **kw)
    r2 = render(agg, registry, **kw)
    assert r1 == r2
    # the coster's estimate equals the real render token count for the same items
    from shapeflow_p1.p1.aggregators import AggregatedItem
    items = [AggregatedItem(it.span_id, it.role, it.facet_ids) for it in agg.items]
    assert coster(items, registry) == r1.token_count
    # exact span bytes appear verbatim -- the renderer never rewrites content
    assert "feline" in r1.text


# --- preflight ------------------------------------------------------------------------


def _clean_preflight(sel, agg, registry, cs, budget=10_000):
    r = render(agg, registry, source_text_for=lambda sp: SOURCE[sp["char_start"]:sp["char_end"]],
               label_for=cs.label_for, tokenizer=TOK)
    return preflight(
        selection=sel, aggregated=agg, rendered=r, registry=registry,
        snapshot_texts={CH: SOURCE}, known_occurrence_ids={OCC}, tokenizer=TOK,
        config=PreflightConfig(selected_token_budget=budget),
    )


def test_preflight_passes_clean_selection():
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2"]}, cs)
    agg = stable_union_v1(sel, registry)
    assert _clean_preflight(sel, agg, registry, cs).ok


def test_preflight_flags_tampered_offsets():
    spans, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    # Corrupt the recorded hash of the selected span.
    registry[agg.items[0].span_id]["text_sha256"] = "0" * 64
    assert not _clean_preflight(sel, agg, registry, cs).ok


def test_preflight_flags_dangling_citation():
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1"]}, cs)
    agg = stable_union_v1(sel, registry)
    r = render(agg, registry, source_text_for=lambda sp: SOURCE[sp["char_start"]:sp["char_end"]],
               label_for=cs.label_for, tokenizer=TOK)
    # Occurrence set does not contain OCC -> lineage closure fails.
    res = preflight(selection=sel, aggregated=agg, rendered=r, registry=registry,
                    snapshot_texts={CH: SOURCE}, known_occurrence_ids=set(), tokenizer=TOK,
                    config=PreflightConfig(selected_token_budget=10_000))
    assert not res.ok


def test_preflight_flags_over_budget():
    _, registry, _, cs, _ = _fixture()
    sel = parse_selection({"contract": "P1_ID", "selected_ids": ["E1", "E2", "E3"]}, cs)
    agg = stable_union_v1(sel, registry)
    res = _clean_preflight(sel, agg, registry, cs, budget=1)  # impossibly small
    assert not res.ok


# --- selectors ------------------------------------------------------------------------


def test_cpu_lexical_selector_is_deterministic_and_relevant():
    _, _, span_ids, _, _ = _fixture()
    inp = SelectorInput(
        topic="feline animals",
        candidates=[(span_ids[0], "Cats are feline animals here."),
                    (span_ids[1], "Dogs are canine animals here."),
                    (span_ids[2], "Birds can surely fly here.")],
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
                        candidates=[("sid1", "Cats are feline.")],
                        query_attempts=[], token_budget=500, contract="P1_ID")
    out = LlmSelector(fake_model).select(TASK, inp)
    assert out == {"contract": "P1_ID", "selected_ids": ["E1"]}
    assert "feline animals" in seen["prompt"]
    assert "[E1] Cats are feline." in seen["prompt"]


def test_partition_by_scope():
    groups = partition_by_scope("per_page", {"pageB": [("s2", "b")], "pageA": [("s1", "a")]})
    # deterministic order by source key
    assert groups == [[("s1", "a")], [("s2", "b")]]
    with pytest.raises(ValueError):
        partition_by_scope("nonsense", {})
