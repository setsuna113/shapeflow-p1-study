"""The prompt view must cover every page, or it is not covering the batch.

Phase 0 measured two independent selector barriers: a whole-batch prompt exceeds the 32,000-token
context ceiling on 61.9% of batches, and the rendered output exceeds 512 tokens on most
selections. `budget_pack_v1` handles the output side. This handles the input side, and the
property that matters is not "it fits" -- almost anything fits if you throw enough away -- but
that what survives still represents every page.

Filling the window by score alone would concentrate on one or two documents, because query-term
density clusters. A prompt built that way covers the batch on paper and, in the results, is
indistinguishable from a batch that never retrieved the missing pages.
"""

from __future__ import annotations

import pytest

from shapeflow.p1.prompt_pack import (
    PromptBudget,
    PromptPackUnsatisfiable,
    prompt_pack_v1,
)


def _batch(pages: int, spans_per_page: int, tokens: int = 10, *, dense_page: int = -1):
    """A batch of `pages` sources. `dense_page` gets all the query terms."""
    spans, texts, counts = [], [], []
    for page in range(pages):
        for slot in range(spans_per_page):
            spans.append({"span_id": f"p{page}s{slot}", "content_hash": f"src{page}"})
            texts.append("alpha beta gamma" if page == dense_page else "filler words here")
            counts.append(tokens)
    return spans, texts, counts


def test_every_page_gets_a_span_before_any_page_gets_a_second():
    """The floor, and the whole reason this stage exists.

    Page 2 carries every query term, so a score-ordered fill would spend the budget there and
    drop four pages entirely.
    """
    spans, texts, counts = _batch(pages=5, spans_per_page=4, tokens=10, dense_page=2)

    # Room for 6 spans out of 20. A greedy-by-score packer would take 4 from page 2 and 2 more.
    result = prompt_pack_v1(spans=spans, texts=texts, token_counts=counts,
                            budget=60, topic="alpha beta gamma")

    covered = {key for key, count in result.per_source_admitted if count}
    assert covered == {f"src{p}" for p in range(5)}, (
        f"only {sorted(covered)} survived; a page missing from the prompt is a page the "
        "selector cannot choose from, and is indistinguishable from one never retrieved")
    assert result.admitted_tokens <= 60


def test_a_batch_whose_floor_does_not_fit_is_refused_not_truncated():
    """Long pages can make one-span-per-page unaffordable. That is an outcome, not a prompt.

    Silently dropping pages would produce a request that does not cover the batch while
    reporting that it does -- explicitly barred by Amendment 1.
    """
    spans, texts, counts = _batch(pages=10, spans_per_page=1, tokens=500)

    with pytest.raises(PromptPackUnsatisfiable, match="covering this batch"):
        prompt_pack_v1(spans=spans, texts=texts, token_counts=counts, budget=1000, topic="q")


def test_a_view_that_already_fits_is_returned_whole():
    """No pruning when none is needed, so CPU-FULL and CPU-PROMPTVIEW coincide on small batches
    and the contrast between them isolates pruning rather than including a fixed offset."""
    spans, texts, counts = _batch(pages=3, spans_per_page=2, tokens=10)

    result = prompt_pack_v1(spans=spans, texts=texts, token_counts=counts,
                            budget=10_000, topic="q")

    assert len(result.span_ids) == 6
    assert result.dropped_span_ids == ()
    assert not result.pruned


def test_the_admitted_view_keeps_the_offered_order():
    """The candidate list's order is part of what the prompt shows. Emitting it in score order
    would make the prompt encode this stage's ranking, which is the selector's job."""
    spans, texts, counts = _batch(pages=3, spans_per_page=3, tokens=10, dense_page=2)

    result = prompt_pack_v1(spans=spans, texts=texts, token_counts=counts,
                            budget=100, topic="alpha beta gamma")

    offered = [str(s["span_id"]) for s in spans]
    assert list(result.span_ids) == [s for s in offered if s in set(result.span_ids)]


def test_the_result_is_deterministic():
    """Non-generative and reproducible: the same batch must yield the same view every time, or
    a resumed shootout compares two different prompts inside one arm."""
    spans, texts, counts = _batch(pages=4, spans_per_page=5, tokens=10, dense_page=1)
    kwargs = dict(spans=spans, texts=texts, token_counts=counts, budget=120, topic="alpha beta")

    first = prompt_pack_v1(**kwargs)
    second = prompt_pack_v1(**kwargs)

    assert first.span_ids == second.span_ids
    assert first.dropped_span_ids == second.dropped_span_ids


def test_the_accounting_records_what_was_dropped_not_just_how_much():
    """CPU-FULL minus CPU-PROMPTVIEW is the price of this pruning. Attributing that difference
    needs the identities of the removed spans, not a count."""
    spans, texts, counts = _batch(pages=4, spans_per_page=4, tokens=10)

    result = prompt_pack_v1(spans=spans, texts=texts, token_counts=counts,
                            budget=80, topic="q")

    assert len(result.span_ids) + len(result.dropped_span_ids) == 16
    assert set(result.span_ids).isdisjoint(result.dropped_span_ids)
    accounting = result.accounting()
    assert accounting["sources_offered"] == 4
    assert accounting["sources_covered"] == 4
    assert accounting["admitted_tokens"] <= 80


def test_remaining_room_is_spread_across_pages_rather_than_given_to_the_densest():
    """After the floor, the fill is round-robin. A per-page greedy fill would hand the whole
    remainder to whichever page is densest in query terms, which is the concentration the floor
    exists to prevent -- just one layer later."""
    spans, texts, counts = _batch(pages=4, spans_per_page=4, tokens=10, dense_page=0)

    # Floor costs 40; 40 left, which is 4 more spans across 4 pages.
    result = prompt_pack_v1(spans=spans, texts=texts, token_counts=counts,
                            budget=80, topic="alpha beta gamma")

    per_source = dict(result.per_source_admitted)
    assert max(per_source.values()) - min(per_source.values()) <= 1, (
        f"fill concentrated on one page: {per_source}")


def test_the_candidate_budget_is_derived_from_the_window_not_chosen():
    """Every term moves independently; a separately-chosen candidate budget is how a legal count
    becomes an illegal request."""
    budget = PromptBudget(max_model_len=32768, completion_cap=768, overhead_tokens=1024)

    assert budget.candidates == 32768 - 768 - 1024 - 256
    assert PromptBudget(max_model_len=40960, completion_cap=768,
                        overhead_tokens=1024).candidates > budget.candidates


def test_a_budget_with_no_room_is_refused():
    spans, texts, counts = _batch(pages=2, spans_per_page=1, tokens=10)

    with pytest.raises(PromptPackUnsatisfiable, match="no room"):
        prompt_pack_v1(spans=spans, texts=texts, token_counts=counts, budget=0, topic="q")


def test_parallel_inputs_are_required_to_be_parallel():
    spans, texts, counts = _batch(pages=2, spans_per_page=2, tokens=10)

    with pytest.raises(ValueError, match="parallel"):
        prompt_pack_v1(spans=spans, texts=texts[:-1], token_counts=counts,
                       budget=100, topic="q")
