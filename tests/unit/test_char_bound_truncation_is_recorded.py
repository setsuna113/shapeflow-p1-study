"""A page cut by the character bound must be recorded, not silently clipped.

Regression gate 3 forbids silently dropping long pages. Callers gate their truncation ledger on
``SharedContent.reason`` being set, and for a long time the character bound set ``truncated=True``
with an empty reason -- so a page clipped by characters was truncated and recorded nowhere.

That was survivable on the Week-1 corpus, where pages came from a search vendor as short extracts
and the character bound almost never bit. It is not survivable on a full-document corpus, where
documents run to thousands of words and the character bound is the common case: the capacity gate
would report no long-page truncation while quietly clipping a large share of what the agent read.
"""

from __future__ import annotations

from shapeflow.evidence.shared_view import (
    CHAR_BOUND_REASON,
    OVERFLOW_REASON,
    SharedContentBudget,
    apply_shared_budget,
)


class _WhitespaceTokenizer:
    """One token per whitespace-separated word, with exact character offsets."""

    def encode_offsets(self, text: str) -> list[tuple[int, int]]:
        offsets, i = [], 0
        for word in text.split(" "):
            if word:
                start = text.index(word, i)
                offsets.append((start, start + len(word)))
                i = start + len(word)
        return offsets


def test_a_short_page_is_not_marked_truncated():
    budget = SharedContentBudget(max_chars=1000, max_tokens=1000)
    out = apply_shared_budget("one two three", budget, _WhitespaceTokenizer())
    assert not out.truncated and out.reason == "" and not out.char_truncated


def test_the_character_bound_now_names_itself():
    """The defect: this used to come back truncated with an empty reason."""
    budget = SharedContentBudget(max_chars=10, max_tokens=1000)
    out = apply_shared_budget("x" * 50, budget, _WhitespaceTokenizer())
    assert out.truncated
    assert out.reason == CHAR_BOUND_REASON, (
        "a character-bound truncation carried no reason, and every caller gates its truncation "
        "ledger on the reason being set -- so the clip was invisible")
    assert out.char_truncated


def test_the_character_bound_names_itself_without_a_tokenizer_too():
    """The no-engine path is the one the offline and acquisition callers take."""
    budget = SharedContentBudget(max_chars=10, max_tokens=1000)
    out = apply_shared_budget("y" * 50, budget, None)
    assert out.truncated and out.reason == CHAR_BOUND_REASON and out.char_truncated


def test_the_token_bound_still_names_itself():
    budget = SharedContentBudget(max_chars=10_000, max_tokens=3)
    out = apply_shared_budget(" ".join(["w"] * 50), budget, _WhitespaceTokenizer())
    assert out.truncated and out.reason == OVERFLOW_REASON
    assert not out.char_truncated, "the character bound did not bite here"


def test_when_both_bounds_bite_the_understatement_is_visible():
    """The token bound decided the length, so it owns ``reason`` -- but ``original_tokens`` was
    counted on text the character bound had already clipped, so it understates the page. Without
    the flag, an understated measurement is indistinguishable from an exact one."""
    text = " ".join(["w"] * 500)          # 999 chars
    budget = SharedContentBudget(max_chars=100, max_tokens=3)
    out = apply_shared_budget(text, budget, _WhitespaceTokenizer())
    assert out.reason == OVERFLOW_REASON
    assert out.char_truncated, (
        "both bounds bit; without this flag the recorded original_tokens looks like the true "
        "page size when it is only the size of what survived the character clip")
    assert out.original_tokens < 500


def test_every_truncating_path_sets_a_reason():
    """No path may report truncation without naming which bound did it.

    Stated as a property rather than three separate cases, so a fourth return path added later
    cannot reintroduce the silent one.
    """
    tokenizer = _WhitespaceTokenizer()
    cases = [
        ("x" * 50, SharedContentBudget(max_chars=10, max_tokens=1000), tokenizer),
        ("y" * 50, SharedContentBudget(max_chars=10, max_tokens=1000), None),
        (" ".join(["w"] * 50), SharedContentBudget(max_chars=10_000, max_tokens=3), tokenizer),
        (" ".join(["w"] * 500), SharedContentBudget(max_chars=100, max_tokens=3), tokenizer),
    ]
    for text, budget, tok in cases:
        out = apply_shared_budget(text, budget, tok)
        assert out.truncated, "fixture no longer truncates; the case has stopped testing anything"
        assert out.reason, f"truncated with no reason: chars={budget.max_chars} tokens={budget.max_tokens}"
