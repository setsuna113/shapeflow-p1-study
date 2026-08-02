"""Admit candidates into a prompt that fits the engine, deterministically and without a model.

There are two independent budgets at the H boundary and they fail at opposite ends of the same
request. The *output* budget is the frozen 512 rendered tokens, enforced after the selector has
chosen. The *input* budget is the engine's context window, and it binds before the selector is
called at all: measured over 1,632 reconstructed gather batches, a whole-batch prompt has a
median of 37,094 tokens against a 32,000 ceiling and 61.9% of batches do not fit.

`budget_pack_v1` solves the output side. This solves the input side, and the two are deliberately
separate objects: one runs on what the selector chose, the other on what the selector is allowed
to see. Fusing them would put the model's ranking inside the decision about what to show the
model.

**What this is not.** It is not a selector and it is not generative. It ranks by the same lexical
signal the CPU control uses, it never calls a model, and its output is a subset of the offered
candidates in their original order -- so the arm that reads it is still choosing from frozen
bytes with intact provenance. It also never sees a relevance label, a P0 summary or any
evaluator material; it is treatment-side by construction.

**Per-page coverage is the load-bearing property.** Filling the window with whichever spans score
highest would silently drop whole pages -- and since the highest-scoring spans cluster in one or
two documents, it would drop most of them. That is indistinguishable downstream from a batch that
never retrieved those pages. So every page with content is guaranteed at least one span before
any page gets a second, and a view that cannot afford one span per page is refused rather than
truncated: `HC_MECHANISM_v1` requires a request covering every page in the batch, and a prompt
missing the batch's tail is not that.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "PROMPT_PACK_VERSION",
    "PromptPackResult",
    "PromptPackUnsatisfiable",
    "prompt_pack_v1",
]

#: Hashed into the protocol binding alongside the prompt bundle and renderer versions. Changing
#: how candidates are admitted changes what every LLM arm was allowed to see, which is a
#: different treatment rather than a refactor.
PROMPT_PACK_VERSION = "prompt_pack_page_floor_then_diversity_v1"

_TOKEN = re.compile(r"[a-z0-9]+")


class PromptPackUnsatisfiable(RuntimeError):
    """The window cannot hold one span per page, so no covering prompt exists for this batch."""


@dataclass(frozen=True)
class PromptPackResult:
    """The admitted view, and the full accounting of what it cost to fit.

    `dropped_span_ids` is kept in full rather than as a count: the difference between
    `CPU-FULL` and `CPU-PROMPTVIEW` is the price of this pruning, and attributing that
    difference needs to know which spans were removed, not how many.
    """

    span_ids: tuple[str, ...]
    dropped_span_ids: tuple[str, ...]
    #: source key -> spans admitted. Every source present in the offer must appear here.
    per_source_admitted: tuple[tuple[str, int], ...]
    per_source_offered: tuple[tuple[str, int], ...]
    admitted_tokens: int
    offered_tokens: int
    budget: int
    version: str = PROMPT_PACK_VERSION

    @property
    def pruned(self) -> bool:
        return bool(self.dropped_span_ids)

    @property
    def sources_covered(self) -> int:
        return sum(1 for _key, count in self.per_source_admitted if count)

    def accounting(self) -> dict:
        return {
            "version": self.version,
            "budget": self.budget,
            "admitted_tokens": self.admitted_tokens,
            "offered_tokens": self.offered_tokens,
            "admitted_spans": len(self.span_ids),
            "dropped_spans": len(self.dropped_span_ids),
            "sources_offered": len(self.per_source_offered),
            "sources_covered": self.sources_covered,
            "pruned": self.pruned,
        }


def _source_key(span: dict) -> str:
    return span.get("content_hash") or span.get("message_id") or span.get("span_id", "")


def _relevance(text: str, query_terms: Counter) -> float:
    """Query-term overlap, the same signal `CpuLexicalAsyncSelector` ranks by.

    Deliberately the same and deliberately weak: this stage decides what the model may *see*, so
    a cleverer score here would move selection work out of the arm under test and into a fixture
    every arm shares.
    """
    if not query_terms:
        return 0.0
    counts = Counter(_TOKEN.findall(text.lower()))
    return float(sum(query_terms[term] * counts[term] for term in query_terms))


def prompt_pack_v1(
    *,
    spans: Sequence[dict],
    texts: Sequence[str],
    token_counts: Sequence[int],
    budget: int,
    topic: str = "",
    min_per_source: int = 1,
) -> PromptPackResult:
    """Admit as much of the batch as the window holds, one span per page first.

    ``budget`` is the room left for candidate material after the caller has already subtracted
    the instructions, the schema, the completion ceiling and its safety margin. Computing it here
    would put two different subtractions in two places.

    Raises `PromptPackUnsatisfiable` when the floor alone does not fit. That is a real outcome for
    a batch of very long pages and must be recorded as one -- dropping pages to make it fit would
    produce a prompt that does not cover the batch while reporting that it does.
    """
    if budget <= 0:
        raise PromptPackUnsatisfiable(f"no room for candidates: budget is {budget}")
    if not (len(spans) == len(texts) == len(token_counts)):
        raise ValueError("spans, texts and token_counts must be parallel")

    query_terms = Counter(_TOKEN.findall(topic.lower()))
    offered_tokens = sum(token_counts)

    by_source: dict[str, list[int]] = {}
    for index, span in enumerate(spans):
        by_source.setdefault(_source_key(span), []).append(index)

    # Within a page, best-scoring first, ties by original order so the choice is reproducible.
    for indices in by_source.values():
        indices.sort(key=lambda i: (-_relevance(texts[i], query_terms), i))

    admitted: set[int] = set()
    spent = 0

    # Floor: `min_per_source` spans of every page, before any page gets more. Pages are taken in
    # their offered order so that a batch's tail is not systematically the part that misses out.
    for key in sorted(by_source, key=lambda k: min(by_source[k])):
        for index in by_source[key][:min_per_source]:
            spent += token_counts[index]
            admitted.add(index)
    if spent > budget:
        raise PromptPackUnsatisfiable(
            f"one span per page costs {spent} tokens against a {budget}-token budget over "
            f"{len(by_source)} pages; no prompt covering this batch fits the window")

    # Fill: round-robin across pages by descending score, so the remaining room is spread rather
    # than consumed by whichever page happens to be densest in query terms.
    rounds = max((len(v) for v in by_source.values()), default=0)
    for depth in range(min_per_source, rounds):
        for key in sorted(by_source, key=lambda k: min(by_source[k])):
            indices = by_source[key]
            if depth >= len(indices):
                continue
            index = indices[depth]
            if index in admitted:
                continue
            if spent + token_counts[index] > budget:
                continue
            spent += token_counts[index]
            admitted.add(index)

    kept = sorted(admitted)
    dropped = [i for i in range(len(spans)) if i not in admitted]
    admitted_by_source = Counter(_source_key(spans[i]) for i in kept)
    offered_by_source = Counter(_source_key(span) for span in spans)

    return PromptPackResult(
        # Original offered order, not score order: the candidate view's own ordering is part of
        # what the selector prompt shows, and reordering it here would make the prompt encode
        # this stage's ranking.
        span_ids=tuple(str(spans[i].get("span_id") or "") for i in kept),
        dropped_span_ids=tuple(str(spans[i].get("span_id") or "") for i in dropped),
        per_source_admitted=tuple(sorted(admitted_by_source.items())),
        per_source_offered=tuple(sorted(offered_by_source.items())),
        admitted_tokens=spent,
        offered_tokens=offered_tokens,
        budget=budget,
    )


@dataclass
class PromptBudget:
    """What is left for candidate material once the fixed parts of the request are paid for.

    Derived, never chosen. The window, the completion ceiling and the instruction block all move
    independently, and a separately-chosen candidate budget is how a legal count becomes an
    illegal request.
    """

    max_model_len: int
    completion_cap: int
    #: The rendered instruction/topic/schema scaffolding around the candidate list.
    overhead_tokens: int
    #: Slack against tokenizer disagreement between this count and the engine's own.
    safety_margin: int = 256

    @property
    def candidates(self) -> int:
        return (self.max_model_len - self.completion_cap
                - self.overhead_tokens - self.safety_margin)
