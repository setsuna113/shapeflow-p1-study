"""Offers from replayed page batches, for driving a live host without the agent.

Nothing in the repository turns a running agent's tool batches into Offers: the P1 study ran
one form per cell. A live host is therefore driven by replaying page batches. Each batch stands
in for one assistant turn's sibling tool calls -- the H boundary's unit -- and becomes one Offer
with both futures priced and neither started:

- **P0**: one summary request per page;
- **P1**: one selector request over the batch's chunked candidates, rendered by the production
  ``render_selector_prompt`` (contract ``P1_ID``), plus a CPU render of the selected ids.

Token estimates are counted by the engine's own tokenizer on the exact prompt bytes that will be
sent. ``est_uncached_prefill`` is that count; ``est_cached_prefill`` is 0 because nothing here
predicts prefix-cache hits; ``est_decode`` is the request's ``max_tokens`` -- the plan's own
upper bound, not a prediction.

The P0 prompt is a stand-in: vendor's summarization prompt lives in the pinned ODR checkout,
which this tree does not carry. Every record names it by ``P0_PROMPT_VERSION``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts.offer import (
    Boundary,
    Form,
    Offer,
    P0Plan,
    P1Plan,
    RenderDescriptor,
    RequestDescriptor,
)
from ..evidence.chunkers import Tokenizer, markdown_structure_v1
from ..hashing import sha256_hex
from ..p1.prompts import PROMPT_BUNDLE_VERSION, render_selector_prompt
from ..runtime.request_tags import OpClass
from .hosts.proxy import RequestBody

__all__ = [
    "P0_PROMPT_VERSION", "CHAT_FORMAT", "NOMINAL_RENDER_CPU_MS", "ReplayPage", "ReplayBatch",
    "BuiltOffer", "build_h_offer", "chat_prompt", "render_selection", "RenderOutcome",
]

P0_PROMPT_VERSION = "replay_p0_summary_standin_v1"
#: Qwen3 ChatML with thinking disabled: what the chat template emits for one user turn and
#: ``enable_thinking=False``. The prompt is sent pre-rendered to ``/v1/completions`` so the
#: bytes counted are the bytes prefilled.
CHAT_FORMAT = "qwen3_chatml_user_nothink_v1"
#: No CPU cost model is calibrated. ``plan_weight`` does not read the render estimate.
NOMINAL_RENDER_CPU_MS = 1.0

_P0_TEMPLATE = """Summarize the web page below for a researcher working on this topic:
{topic}

Keep the facts, names, dates and numbers that bear on the topic. Write at most {words} words.

PAGE TITLE: {title}
PAGE CONTENT:
{content}
"""


def chat_prompt(user: str) -> str:
    return (f"<|im_start|>user\n{user}<|im_end|>\n"
            "<|im_start|>assistant\n<think>\n\n</think>\n\n")


@dataclass(frozen=True)
class ReplayPage:
    page_id: str
    title: str
    text: str


@dataclass(frozen=True)
class ReplayBatch:
    """One assistant turn's sibling page batch, in join order."""

    batch_id: str
    topic: str
    pages: tuple[ReplayPage, ...]


@dataclass(frozen=True)
class BuiltOffer:
    offer: Offer
    bodies: Mapping[Form, tuple[RequestBody, ...]]
    #: ``(label, text)`` in candidate order, for the CPU render of a P1 selection.
    candidates: tuple[tuple[str, str], ...]


def _descriptor(op: OpClass, prompt: str, max_tokens: int, tokenizer: Tokenizer):
    return RequestDescriptor(
        op_class=op,
        prompt_sha256=sha256_hex(prompt.encode("utf-8")),
        max_tokens=max_tokens,
        est_uncached_prefill=tokenizer.count(prompt),
        est_cached_prefill=0,
        est_decode=max_tokens,
    )


def build_h_offer(
    batch: ReplayBatch,
    *,
    tokenizer: Tokenizer,
    submitted_at_ns: int,
    p0_max_tokens: int = 8192,
    selector_max_tokens: int = 768,
    chunk_max_tokens: int = 320,
    selector_budget: int = 512,
) -> BuiltOffer:
    """Defaults are configs/week1.yaml's: vendor's ``summarization_model_max_tokens``,
    ``selector_max_completion_tokens``, ``selected_token_budget`` and the corpus chunker at
    320 tokens."""
    if not batch.pages:
        raise ValueError(f"{batch.batch_id}: an H offer needs at least one page")

    p0_prompts = [
        chat_prompt(_P0_TEMPLATE.format(topic=batch.topic, words=p0_max_tokens // 2,
                                        title=page.title, content=page.text))
        for page in batch.pages
    ]
    p0 = P0Plan(requests=tuple(
        _descriptor(OpClass.PAGE_P0_SUMMARY, p, p0_max_tokens, tokenizer) for p in p0_prompts))
    bodies: dict[Form, tuple[RequestBody, ...]] = {
        Form.P0: tuple(RequestBody(prompt=p, max_tokens=p0_max_tokens) for p in p0_prompts),
    }

    candidates: list[tuple[str, str]] = []
    rows: list[tuple] = []
    for page in batch.pages:
        for chunk in markdown_structure_v1(page.text, tokenizer=tokenizer,
                                           max_tokens=chunk_max_tokens):
            text = chunk.text(page.text).strip()
            if not text:
                continue
            label = f"E{len(candidates)}"  # the LlmSelector's label format
            candidates.append((label, text))
            rows.append((label, text, (page.title, *chunk.heading_path), ()))

    p1: P1Plan | None = None
    if candidates:
        selector = chat_prompt(render_selector_prompt(
            topic=batch.topic, candidates=rows, query_attempts=[("Q0", batch.topic)],
            budget=selector_budget, contract="P1_ID"))
        p1 = P1Plan(
            selector=_descriptor(OpClass.PAGE_P1_SELECTOR_BATCH, selector, selector_max_tokens,
                                 tokenizer),
            render=RenderDescriptor(est_cpu_ms=NOMINAL_RENDER_CPU_MS,
                                    candidate_count=len(candidates)),
        )
        bodies[Form.P1] = (RequestBody(prompt=selector, max_tokens=selector_max_tokens),)

    offer = Offer(item_id=batch.batch_id, boundary=Boundary.H, unit_id=f"turn-{batch.batch_id}",
                  p0=p0, p1=p1, submitted_at_ns=submitted_at_ns)
    return BuiltOffer(offer=offer, bodies=bodies, candidates=tuple(candidates))


@dataclass(frozen=True)
class RenderOutcome:
    """The CPU half of P1. A failed render is the case where C2 would republish the batch as
    P0; this replay records it and does not republish."""

    ok: bool
    selected: tuple[str, ...]
    invalid: tuple[str, ...]
    rendered_chars: int
    cpu_ns: int
    error: str = ""


def render_selection(output: str | None,
                     candidates: Sequence[tuple[str, str]]) -> RenderOutcome:
    started = time.perf_counter_ns()

    def done(ok, selected=(), invalid=(), chars=0, error=""):
        return RenderOutcome(ok=ok, selected=tuple(selected), invalid=tuple(invalid),
                             rendered_chars=chars, cpu_ns=time.perf_counter_ns() - started,
                             error=error)

    text = output or ""
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return done(False, error="no JSON object in selector output")
    try:
        obj = json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        return done(False, error=f"unparseable selector output: {exc}")
    ids = obj.get("selected_ids") if isinstance(obj, dict) else None
    if not isinstance(ids, list) or not all(isinstance(i, str) for i in ids):
        return done(False, error="selected_ids is not a list of strings")
    known = dict(candidates)
    invalid = [i for i in ids if i not in known]
    if invalid:
        return done(False, selected=ids, invalid=invalid,
                    error="selector named ids outside the offered set")
    chosen = set(ids)
    rendered = "\n\n".join(t for label, t in candidates if label in chosen)
    return done(True, selected=ids, chars=len(rendered))


#: Recorded with every run, so a selector record names the exact template bundle behind it.
SELECTOR_PROMPT_VERSION = PROMPT_BUNDLE_VERSION
