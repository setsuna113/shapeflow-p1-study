"""The one truncation both arms see, measured in the units the engine actually counts.

``odr.max_content_length`` is a character budget inherited from a vendor default written for a
hosted model with a very large context window. The engine serving this study has 32,768 tokens,
and characters are not tokens: across the frozen corpus the densest pages run at 1.08 chars per
token, so a 50,000-character page can be 46,202 tokens -- more than the whole window.

What that cost was not a truncated summary but a *rejected request*. vLLM refuses when
``prompt + max_tokens`` exceeds the window, and vendor catches the refusal and returns the raw
page as though it were a summary. P1 never reaches that code path, because its page hook returns
from ``defer_page_batch`` before vendor's summariser is awaited -- so the failure fell on P0
alone, on the largest pages, which is exactly the high-evidence stratum P1 is supposed to win.
An apparatus that quietly handicaps the baseline where the treatment is meant to look best is
not measuring the treatment.

So the visible content is bounded in tokens as well as characters, and both bounds are applied
*here*, once, to the bytes both arms read. A rule applied in two places is a rule that will
eventually be applied differently in two places: the previous arrangement had our runner
truncating for P1 and vendor's ``utils.py`` truncating for P0, agreeing only because both
happened to slice the same string by the same integer.

Truncation is on an exact token boundary and is recorded, never silent. A page that loses its
tail is a page whose evidence both arms were denied equally, and the count of them belongs in
the write-up rather than in a log nobody reads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from .chunkers import Tokenizer

__all__ = [
    "SharedContentBudget",
    "SharedContent",
    "OVERFLOW_REASON",
    "apply_shared_budget",
]

#: Recorded on a page whose tail was removed so the request could fit the engine's window.
#: Named rather than described so it can be counted, and so a reader can tell this apart from a
#: page that was simply short.
OVERFLOW_REASON = "SHARED_CONTEXT_OVERFLOW_TRUNCATION"


@dataclass(frozen=True)
class SharedContentBudget:
    """How much of a page either arm may see.

    ``max_tokens`` is derived, not chosen: it is what remains of the engine's context window
    after the largest completion the page summariser may request, plus the prompt scaffolding
    vendor wraps around the content. Deriving it is what keeps it correct when the window or the
    completion cap moves; choosing it is how the two drift apart until a request is refused.
    """

    max_chars: int
    max_tokens: int

    @classmethod
    def derive(
        cls,
        *,
        max_chars: int,
        max_model_len: int,
        completion_cap: int,
        prompt_overhead_tokens: int,
    ) -> "SharedContentBudget":
        budget = max_model_len - completion_cap - prompt_overhead_tokens
        if budget < 1:
            raise ValueError(
                f"no room for page content: a {max_model_len}-token window cannot hold a "
                f"{completion_cap}-token completion plus {prompt_overhead_tokens} tokens of "
                "prompt scaffolding"
            )
        return cls(max_chars=int(max_chars), max_tokens=int(budget))


@dataclass(frozen=True)
class SharedContent:
    text: str
    truncated: bool
    reason: str = ""
    original_tokens: int = 0
    kept_tokens: int = 0


def apply_shared_budget(
    text: str,
    budget: SharedContentBudget,
    tokenizer: Optional[Tokenizer] = None,
) -> SharedContent:
    """Bound one page's visible content, in characters and then in tokens.

    Characters first, because that is vendor's own rule and the overwhelming majority of pages
    never reach the token bound. The token bound then catches what the character bound cannot
    see: content whose bytes are dense enough that a legal character count is an illegal token
    count.

    ``tokenizer`` may be None only where no engine is involved -- the character bound alone is
    then applied, and nothing claims the result fits a window.
    """
    clipped = text[: budget.max_chars]
    if tokenizer is None:
        return SharedContent(text=clipped, truncated=len(clipped) < len(text))

    offsets = tokenizer.encode_offsets(clipped)
    if len(offsets) <= budget.max_tokens:
        return SharedContent(
            text=clipped,
            truncated=len(clipped) < len(text),
            original_tokens=len(offsets),
            kept_tokens=len(offsets),
        )
    # Cut on the boundary of the last token that fits, so the surviving text is exactly the
    # decoding of the tokens the engine would have read -- not a byte prefix that splits one.
    cut = offsets[budget.max_tokens - 1][1]
    return SharedContent(
        text=clipped[:cut],
        truncated=True,
        reason=OVERFLOW_REASON,
        original_tokens=len(offsets),
        kept_tokens=budget.max_tokens,
    )
