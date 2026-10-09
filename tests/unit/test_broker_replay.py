"""Replayed page batches become offers that price exactly what the host would send."""

from __future__ import annotations

from shapeflow.broker.replay import (
    ReplayBatch,
    ReplayPage,
    build_h_offer,
    render_selection,
)
from shapeflow.contracts.offer import Form
from shapeflow.evidence.chunkers import WhitespaceTokenizer
from shapeflow.hashing import sha256_hex
from shapeflow.runtime.request_tags import OpClass

TOK = WhitespaceTokenizer()


def _batch(n=2):
    pages = tuple(
        ReplayPage(page_id=f"p{i}", title=f"Title {i}",
                   text=f"First paragraph of page {i}.\n\nSecond paragraph, page {i}.")
        for i in range(n))
    return ReplayBatch(batch_id="B1", topic="What happened?", pages=pages)


def test_p0_is_one_summary_per_page_and_p1_one_selector_over_all_pages():
    built = build_h_offer(_batch(3), tokenizer=TOK, submitted_at_ns=7)
    offer = built.offer
    assert len(offer.p0.requests) == 3
    assert {r.op_class for r in offer.p0.requests} == {OpClass.PAGE_P0_SUMMARY}
    assert offer.p1.selector.op_class is OpClass.PAGE_P1_SELECTOR_BATCH
    assert offer.p1.render.candidate_count == len(built.candidates) == 6
    assert len(built.bodies[Form.P0]) == 3 and len(built.bodies[Form.P1]) == 1


def test_every_descriptor_prices_the_exact_bytes_its_body_sends():
    built = build_h_offer(_batch(2), tokenizer=TOK, submitted_at_ns=0)
    pairs = list(zip(built.offer.p0.requests, built.bodies[Form.P0], strict=True))
    pairs.append((built.offer.p1.selector, built.bodies[Form.P1][0]))
    for descriptor, body in pairs:
        assert descriptor.prompt_sha256 == sha256_hex(body.prompt.encode("utf-8"))
        assert descriptor.est_uncached_prefill == TOK.count(body.prompt)
        assert descriptor.est_decode == descriptor.max_tokens == body.max_tokens
        assert descriptor.est_cached_prefill == 0


def test_the_selector_prompt_lists_every_candidate_label():
    built = build_h_offer(_batch(2), tokenizer=TOK, submitted_at_ns=0)
    selector = built.bodies[Form.P1][0].prompt
    for label, _ in built.candidates:
        assert f"[{label}]" in selector
    assert '"contract":"P1_ID"' in selector


def test_a_batch_with_nothing_chunkable_offers_only_p0():
    batch = ReplayBatch(batch_id="B2", topic="t",
                        pages=(ReplayPage(page_id="p", title="T", text="   "),))
    built = build_h_offer(batch, tokenizer=TOK, submitted_at_ns=0)
    assert built.offer.p1 is None and Form.P1 not in built.bodies


def test_render_accepts_offered_ids_and_rejects_invented_ones():
    candidates = (("E0", "alpha"), ("E1", "beta"), ("E2", "gamma"))
    ok = render_selection('{"contract":"P1_ID","selected_ids":["E2","E0"]}', candidates)
    assert ok.ok and ok.selected == ("E2", "E0") and ok.rendered_chars == len("alpha\n\ngamma")
    bad = render_selection('{"selected_ids":["E0","E9"]}', candidates)
    assert not bad.ok and bad.invalid == ("E9",)
    assert not render_selection("no json here", candidates).ok
