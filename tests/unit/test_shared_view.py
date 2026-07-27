"""One truncation, in the units the engine counts, applied to what both arms read.

`odr.max_content_length` is 50,000 characters -- a vendor default written for a model with a
very large window. The engine here has 32,768 tokens, and across the frozen corpus the densest
pages run at 1.08 chars per token, so a legal character count can be a 46,202-token prompt.

What that cost was not a truncated summary but a refused request: vLLM rejects when
prompt + max_tokens exceeds the window, and vendor catches the rejection and returns the raw
page as though it were a summary. P1 never reaches that path, so the failure fell on P0 alone,
on the largest pages -- the stratum P1 is supposed to win.
"""

from __future__ import annotations

import pytest

from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer
from shapeflow_p1.evidence.shared_view import (
    OVERFLOW_REASON,
    SharedContentBudget,
    apply_shared_budget,
)


def test_the_budget_is_derived_from_the_window_not_chosen():
    """A separately-chosen number is how the bound drifts out of agreement with the window."""
    budget = SharedContentBudget.derive(
        max_chars=50_000, max_model_len=32_768, completion_cap=8_192,
        prompt_overhead_tokens=1_024,
    )
    assert budget.max_tokens == 32_768 - 8_192 - 1_024
    assert budget.max_chars == 50_000


def test_a_window_too_small_for_its_own_completion_cap_is_refused():
    """Better to fail here than to emit a negative budget and truncate every page to nothing."""
    with pytest.raises(ValueError, match="no room for page content"):
        SharedContentBudget.derive(
            max_chars=50_000, max_model_len=8_192, completion_cap=8_192,
            prompt_overhead_tokens=1_024,
        )


def test_short_content_is_untouched_and_not_marked_truncated():
    budget = SharedContentBudget(max_chars=1_000, max_tokens=100)
    result = apply_shared_budget("one two three", budget, WhitespaceTokenizer())
    assert result.text == "one two three"
    assert result.truncated is False
    assert result.reason == ""


def test_the_character_bound_still_applies_and_is_vendors_own():
    budget = SharedContentBudget(max_chars=10, max_tokens=1_000)
    result = apply_shared_budget("abcdefghijklmnop", budget, WhitespaceTokenizer())
    assert result.text == "abcdefghij"
    assert result.truncated is True
    # A plain character clip is not an overflow: it is the rule vendor already applied.
    assert result.reason == ""


def test_token_dense_content_within_the_character_bound_is_still_cut():
    """The case the character bound cannot see, and the one that refused the request."""
    budget = SharedContentBudget(max_chars=10_000, max_tokens=5)
    text = " ".join(f"w{i}" for i in range(50))
    result = apply_shared_budget(text, budget, WhitespaceTokenizer())
    assert result.reason == OVERFLOW_REASON
    assert result.truncated is True
    assert result.original_tokens == 50
    assert result.kept_tokens == 5
    assert len(WhitespaceTokenizer().encode_offsets(result.text)) == 5


def test_the_cut_lands_on_a_token_boundary():
    """The surviving text is the decoding of the tokens the engine would have read.

    A byte prefix that splits a token would hand the model a fragment neither arm's accounting
    describes, and the kept-token count would be a claim about something that was never sent.
    """
    budget = SharedContentBudget(max_chars=10_000, max_tokens=3)
    result = apply_shared_budget("alpha beta gamma delta epsilon", budget, WhitespaceTokenizer())
    assert result.text == "alpha beta gamma"


def test_without_a_tokenizer_only_the_character_bound_is_claimed():
    """Off the run host there is no engine, so nothing may claim the result fits a window."""
    budget = SharedContentBudget(max_chars=5, max_tokens=1)
    result = apply_shared_budget("abcdefgh", budget, None)
    assert result.text == "abcde"
    assert result.reason == ""
    assert result.kept_tokens == 0


def test_both_arms_receive_identical_bytes_from_one_application():
    """The asymmetry this exists to prevent.

    Bounding the page in two places -- our runner for P1, vendor's utils for P0 -- agreed only
    because both sliced the same string by the same integer, and could not express a token bound
    at all. Applying the rule once means a page that loses its tail loses it for both arms.
    """
    budget = SharedContentBudget(max_chars=10_000, max_tokens=4)
    text = " ".join(f"token{i}" for i in range(40))
    p0_view = apply_shared_budget(text, budget, WhitespaceTokenizer()).text
    p1_view = apply_shared_budget(text, budget, WhitespaceTokenizer()).text
    assert p0_view == p1_view
    # And the trimmed page is genuinely shorter than what either arm would have read unbounded.
    assert len(p0_view) < len(text)


def test_the_frozen_configuration_leaves_real_room_for_page_content():
    """The numbers this study actually runs on, asserted rather than assumed."""
    from pathlib import Path

    import yaml

    root = Path(__file__).resolve().parents[2]
    stack = yaml.safe_load((root / "configs" / "stack.yaml").read_text(encoding="utf-8"))
    week1 = yaml.safe_load((root / "configs" / "week1.yaml").read_text(encoding="utf-8"))
    budget = SharedContentBudget.derive(
        max_chars=int(week1["odr"]["max_content_length"]),
        max_model_len=int(stack["engine"]["max_model_len"]),
        completion_cap=int(week1["odr"]["summarization_model_max_tokens"]),
        prompt_overhead_tokens=int(week1["odr"]["summarization_prompt_overhead_tokens"]),
    )
    # p95 of the frozen corpus is 14,293 content tokens; the bound must clear it comfortably or
    # the overflow rule would be trimming the common case rather than the extreme one.
    assert budget.max_tokens > 14_293
    # And the window must be within what the model declares (max_position_embeddings 40960).
    assert int(stack["engine"]["max_model_len"]) <= 40_960
