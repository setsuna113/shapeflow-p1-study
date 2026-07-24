"""Selectors: the deterministic CPU control and the LLM selector, plus scope partitioning.

Two selectors, for two different jobs:

- ``CpuLexicalSelector`` -- deterministic lexical selection (no model). It is a *control*, not
  a candidate: comparing it to the LLM selector separates "does an LLM add value" from "does
  pointing at spans instead of prose add value". It emits the P1_ID contract.
- ``LlmSelector`` -- runs the same local target model P0 uses, via an injected model-call
  callable so tests drive it deterministically and it never reaches for a network of its own.

Scope decides how candidates are partitioned across selector calls (plan §9.3): ``per_page``
(one call per source), ``per_tool_call`` (one call per search result set), and ``hierarchical``
(a per-page shortlist then a cross-source reducer, whose second pass is accounted separately by
the caller, never hidden in the aggregator).
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional, Protocol

from ..evidence.identity import CandidateSet
from ..odr.hooks import TaskContext
from .prompts import render_selector_prompt

__all__ = [
    "SelectorInput",
    "Selector",
    "CpuLexicalSelector",
    "LlmSelector",
    "partition_by_scope",
]

_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Candidate:
    """One offered evidence span, as the selector will see it.

    ``heading_path`` and ``context`` are the chunker's metadata (enclosing headings; a table's
    header row). They are carried here so they actually reach the prompt: dropping them made a
    selected table row unreadable and then charged the resulting bad selection to the contract
    under test rather than to the missing context.
    """

    span_id: str
    text: str
    heading_path: tuple[str, ...] = ()
    context: tuple[str, ...] = ()

    @classmethod
    def from_span(cls, span: dict, text: str, *, snapshot_texts: dict[str, str]) -> "Candidate":
        """Resolve a span's addressed metadata into the strings the prompt will show.

        Reads ``heading_refs``/``context_refs`` -- the addressed, re-hashable form. It used to
        read a free-text ``context`` key the schema no longer has, so the real chain
        (chunk.context_ranges -> span.context_refs -> Candidate.context) silently produced an
        empty tuple and every table row reached the selector with no header at all.
        """
        def resolve(refs: list[dict]) -> tuple[str, ...]:
            out = []
            for ref in refs or ():
                snapshot = snapshot_texts.get(ref["content_hash"])
                if snapshot is None:
                    raise KeyError(
                        f"candidate metadata addresses snapshot {ref['content_hash'][:12]}, "
                        "which was not supplied; a candidate must never be offered with "
                        "unresolvable metadata"
                    )
                out.append(snapshot[ref["char_start"]:ref["char_end"]])
            return tuple(out)

        return cls(
            span_id=span.get("span_id") or span["visible_span_id"],
            text=text,
            heading_path=resolve(span.get("heading_refs")),
            context=resolve(span.get("context_refs")),
        )


@dataclass(frozen=True)
class SelectorInput:
    """What a selector is given for one call. ``query_attempts`` is (attempt_id, text)."""

    topic: str
    candidates: list[Candidate]
    query_attempts: list[tuple[str, str]]
    token_budget: int
    contract: str


class Selector(Protocol):
    def select(self, task_ctx: TaskContext, inp: SelectorInput) -> dict:
        """Return a raw selector-output dict (to be parsed by contracts.parse_selection)."""


class CpuLexicalSelector:
    """Deterministic BM25-ish lexical relevance to the topic, no model. Emits P1_ID."""

    def __init__(self, *, max_selected: int = 8) -> None:
        self._max = max_selected

    def select(self, task_ctx: TaskContext, inp: SelectorInput) -> dict:
        q_terms = Counter(_TOKEN.findall(inp.topic.lower()))
        scored: list[tuple[float, str]] = []
        for cand in inp.candidates:
            tf = Counter(_TOKEN.findall(cand.text.lower()))
            score = float(sum(q_terms[t] * tf[t] for t in q_terms))
            if score > 0:
                scored.append((score, cand.span_id))
        # Deterministic: score desc, then span_id asc.
        scored.sort(key=lambda pair: (-pair[0], pair[1]))
        # Build a CandidateSet local to this call to convert ids -> labels for output.
        cs = CandidateSet.build([c.span_id for c in inp.candidates])
        selected = [cs.label_for(sid) for _, sid in scored[: self._max]]
        return {"contract": "P1_ID", "selected_ids": selected}


# model_call(prompt, *, contract) -> raw selector-output dict (already JSON-parsed).
ModelCall = Callable[[str], dict]


class LlmSelector:
    """Runs the local target model via an injected callable. The callable is responsible for
    the actual (proxied, telemetered) request and for returning parsed JSON; this class only
    renders the prompt and passes the result through.

    NOT YET a bounded decode. The contract's ``maxItems`` is a post-hoc validator: an
    over-long selection is rejected *after* its tokens were generated and charged to the P1
    arm. A real ceiling needs both halves of the model call, which land with the telemetry
    proxy (Block 5):

    - a completion ``max_tokens`` derived from ``token_budget``, so runaway output is cut at
      the engine rather than measured and discarded;
    - schema-constrained generation (guided decoding against
      ``schemas/selector_output.schema.json``), so malformed output is unrepresentable rather
      than repaired afterwards -- repair is free for us and not for the GPU.

    Until then, P1's measured decode cost includes work that a deployed P1 would not do, which
    biases *against* P1. That direction is the safe one, but it is not zero.
    """

    def __init__(self, model_call: ModelCall) -> None:
        self._model_call = model_call

    def select(self, task_ctx: TaskContext, inp: SelectorInput) -> dict:
        prompt = render_selector_prompt(
            topic=inp.topic,
            candidates=[
                (_label(i), c.text, c.heading_path, c.context)
                for i, c in enumerate(inp.candidates, 1)
            ],
            query_attempts=[(f"Q{j}", text) for j, (_, text) in enumerate(inp.query_attempts, 1)],
            budget=inp.token_budget,
            contract=inp.contract,
        )
        return self._model_call(prompt)


def _label(i: int) -> str:
    return f"E{i}"


def partition_by_scope(
    scope: str, candidates_by_source: dict[str, list[tuple[str, str]]]
) -> list[list[tuple[str, str]]]:
    """Partition candidates into selector-call groups.

    ``candidates_by_source`` maps a source key (content_hash for a page, tool_call_id for a
    search call) to its ordered candidates.
    - ``per_page`` / ``per_tool_call``: one group per source key (the key's meaning is set by
      how the caller built the dict).
    - ``hierarchical``: same per-source groups for the shortlist pass; the caller runs the
      cross-source reducer over the union afterwards and accounts that second pass separately.
    """
    if scope in {"per_page", "per_tool_call", "hierarchical"}:
        return [candidates_by_source[k] for k in sorted(candidates_by_source)]
    raise ValueError(f"unknown scope {scope!r}")
