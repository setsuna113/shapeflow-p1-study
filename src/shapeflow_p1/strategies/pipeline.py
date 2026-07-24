"""The selection pipeline every P1 arm shares.

One implementation, because the arms must differ *only* in the dimensions the study varies --
chunker, scope, contract, aggregator, namespace. If each arm assembled its own pipeline, a
difference in how two of them budgeted or rendered would be indistinguishable from the
mechanism under test.

The order is fixed and each step's output is the next one's only input:

    frozen bytes -> chunker -> candidate view -> selector -> parse -> aggregate -> preflight

Preflight owns the render, so what is verified is what is published. A failure anywhere returns
a :class:`SelectionFailure` rather than raising, because the caller's job is to fall the whole
batch back to P0, and a decision object is what that needs.

Work accounting is not optional and not conditional on success. A selector call that produced
an unusable selection still spent its tokens; recording them only on the happy path would make
P1 look cheaper exactly when it went wrong.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..evidence.chunkers import Tokenizer, fixed_token_v1, markdown_structure_v1, paragraph_sentence_v1
from ..evidence.identity import build_evidence_span, build_visible_message_span
from ..p1.aggregators import coverage_budget_v1, global_rerank_v1, stable_union_v1
from ..p1.contracts import SelectionContractError, parse_selection
from ..p1.preflight import PreflightConfig, preflight
from ..p1.view import CandidateViewRecord, ViewConstructionError

__all__ = [
    "SelectionOutcome",
    "SelectionFailure",
    "WorkRecord",
    "CHUNKERS",
    "AGGREGATORS",
    "run_selection",
]

CHUNKERS: dict[str, Callable] = {
    "fixed_token_v1": lambda text, tok, **kw: fixed_token_v1(text, tokenizer=tok, **kw),
    "paragraph_sentence_v1": lambda text, tok, **kw: paragraph_sentence_v1(text, tokenizer=tok, **kw),
    "markdown_structure_v1": lambda text, tok, **kw: markdown_structure_v1(text, tokenizer=tok, **kw),
}

AGGREGATORS = {"stable_union_v1", "coverage_budget_v1", "global_rerank_v1", "mmr_stable_union",
               "token_matched"}


@dataclass
class WorkRecord:
    """Everything an arm spent, whether or not it produced output."""

    selector_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cpu_seconds: float = 0.0
    retries: int = 0

    def add(self, other: "WorkRecord") -> None:
        self.selector_calls += other.selector_calls
        self.prompt_tokens += other.prompt_tokens
        self.completion_tokens += other.completion_tokens
        self.cpu_seconds += other.cpu_seconds
        self.retries += other.retries


@dataclass(frozen=True)
class SelectionFailure:
    reason: str
    detail: str = ""


@dataclass
class SelectionOutcome:
    """What one selector call produced, plus what it cost either way."""

    text: Optional[str] = None
    work: WorkRecord = field(default_factory=WorkRecord)
    failure: Optional[SelectionFailure] = None
    view_sha256: str = ""
    selected: int = 0
    offered: int = 0
    dropped_for_budget: int = 0
    normalization: Optional[dict] = None

    @property
    def ok(self) -> bool:
        return self.failure is None and self.text is not None


def _aggregate(name: str, selection, registry, *, token_budget: int, coster):
    if name in ("stable_union_v1", "mmr_stable_union"):
        return stable_union_v1(selection, registry)
    if name == "coverage_budget_v1":
        return coverage_budget_v1(selection, registry, token_budget=token_budget,
                                  coster=coster, min_sources=1)
    if name == "global_rerank_v1":
        return global_rerank_v1(selection, registry, token_budget=token_budget, coster=coster)
    raise ValueError(f"unknown aggregator {name!r}")


async def run_selection(
    *,
    spans: Sequence[dict],
    namespace: str,
    snapshot_texts: dict[str, str],
    visible_views: dict[str, bytes],
    topic: str,
    contract: str,
    aggregation: str,
    token_budget: int,
    tokenizer: Tokenizer,
    query_attempts: Sequence[tuple[str, str]],
    selector,
    task_ctx,
    known_occurrence_ids: set[str],
    query_status: Optional[dict[str, str]] = None,
    source_meta: Optional[dict] = None,
) -> SelectionOutcome:
    """Offer, select, publish -- or fail with everything it cost recorded."""
    work = WorkRecord()
    if not spans:
        return SelectionOutcome(text="", work=work, offered=0)

    try:
        view = CandidateViewRecord.build(
            spans=list(spans), tokenizer=tokenizer, namespace=namespace, topic=topic,
            contract=contract, token_budget=token_budget, query_attempts=list(query_attempts),
            snapshot_texts=snapshot_texts, visible_views=visible_views,
            source_meta=source_meta, query_status=query_status,
        )
    except ViewConstructionError as e:
        return SelectionOutcome(
            work=work, failure=SelectionFailure("VIEW_CONSTRUCTION", str(e)))

    try:
        raw, call_work = await selector.select(task_ctx=task_ctx, view=view)
    except asyncio.CancelledError:
        raise
    except Exception as e:  # noqa: BLE001
        return SelectionOutcome(
            work=work, view_sha256=view.view_sha256,
            failure=SelectionFailure("SELECTOR_ERROR", f"{type(e).__name__}: {e}"))
    work.add(call_work)

    try:
        selection = parse_selection(raw, view.candidate_set)
    except SelectionContractError as e:
        # The tokens were still spent. Recording them only on success would make P1 look
        # cheaper exactly when it failed.
        return SelectionOutcome(
            work=work, view_sha256=view.view_sha256,
            normalization=getattr(e.normalization, "__dict__", None),
            failure=SelectionFailure("CONTRACT", str(e)))

    try:
        aggregated = _aggregate(aggregation, selection, view.registry,
                                token_budget=token_budget, coster=view.coster())
    except Exception as e:  # noqa: BLE001
        return SelectionOutcome(work=work, view_sha256=view.view_sha256,
                                failure=SelectionFailure("AGGREGATION", str(e)))

    result = preflight(
        selection=selection, aggregated=aggregated, view=view,
        known_occurrence_ids=known_occurrence_ids,
        config=PreflightConfig(selected_token_budget=token_budget,
                               expected_namespace=namespace),
    )
    if not result.ok:
        return SelectionOutcome(
            work=work, view_sha256=view.view_sha256,
            failure=SelectionFailure("PREFLIGHT", "; ".join(result.errors[:3])))

    return SelectionOutcome(
        text=result.rendered.text, work=work, view_sha256=view.view_sha256,
        selected=len(aggregated.items), offered=len(view.candidates),
        dropped_for_budget=len(aggregated.dropped_for_budget),
        normalization=selection.normalization.__dict__,
    )


def spans_from_page(text: str, *, content_hash: str, occurrence_id: str, chunker: str,
                    tokenizer: Tokenizer, max_tokens: int = 320) -> list[dict]:
    """Chunk one page's frozen bytes into RAW_SOURCE spans."""
    fn = CHUNKERS[chunker]
    kw = {"window": max_tokens, "overlap": 0} if chunker == "fixed_token_v1" \
        else {"max_tokens": max_tokens}
    return [
        build_evidence_span(c, text, content_hash=content_hash,
                            source_occurrence_ids=[occurrence_id], chunker_version=chunker)
        for c in fn(text, tokenizer, **kw)
    ]


def spans_from_visible_view(view_bytes: bytes, *, view_hash: str, messages: Sequence[dict],
                            tokenizer: Tokenizer, max_tokens: int = 320) -> list[dict]:
    """Chunk the exact compressor-visible bytes into VISIBLE_MESSAGE spans.

    ``messages`` gives each message's ``(message_id, role, byte_start, byte_end, kind)`` inside
    the view, so a span never straddles two messages and every span's kind is the message's own.
    AI reasoning is ``MODEL_DERIVED_CONTEXT`` and can never become TOOL_EVIDENCE -- that is the
    line between C_VISIBLE reading what the compressor read and C_VISIBLE inventing provenance.
    """
    spans: list[dict] = []
    for msg in messages:
        segment = view_bytes[msg["byte_start"]:msg["byte_end"]]
        try:
            text = segment.decode("utf-8")
        except UnicodeDecodeError:
            continue
        for chunk in paragraph_sentence_v1(text, tokenizer=tokenizer, max_tokens=max_tokens):
            start = msg["byte_start"] + len(text[:chunk.char_start].encode("utf-8"))
            end = msg["byte_start"] + len(text[:chunk.char_end].encode("utf-8"))
            spans.append(build_visible_message_span(
                message_id=msg["message_id"], message_role=msg["role"],
                byte_start=start, byte_end=end, message_bytes=view_bytes,
                kind=msg["kind"], visible_compressor_view_hash=view_hash,
                source_occurrence_ids=list(msg.get("occurrence_ids") or [])
                if msg["kind"] == "TOOL_EVIDENCE" else None,
                token_len=chunk.token_len,
            ))
    return spans
