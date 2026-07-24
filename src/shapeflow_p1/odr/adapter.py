"""The langchain boundary: freezing live messages, and the whole-batch publish protocol.

Everything the patch injects into vendor ODR is implemented here, so the patch itself stays a
handful of dispatch lines and the logic is testable without a graph.

Two things this module exists to get right.

**The batch is the publish unit.** `utils.tavily_search` sees one tool call's results;
sibling calls only meet at `asyncio.gather` in `deep_researcher.researcher_tools`. So when a
strategy is bound, `tavily_search` returns a :class:`DeferredPageBatch` -- acquisition, URL
dedup, first-occurrence ordering and the pinned truncation all done, nothing summarised,
nothing formatted, nothing published. `reduce_tool_batch` then runs after the gather, over the
whole assistant turn, and produces every `ToolMessage` at once for a single `Command`. Half a
batch is never observable, and a P1 failure takes the entire batch to P0 rather than leaving a
`[P1(A), P0(B)]` hybrid, which is not an arm and would be scored as one.

**A frozen message must rebuild into the same prompt.** C_VISIBLE's claim is that the selector
saw exactly what P0's compressor saw, and what a compressor sees is the rendered message list.
So :func:`freeze_message` keeps content blocks, tool calls, invalid tool calls, additional
kwargs, response and usage metadata, artifact and status; :func:`thaw_message` rebuilds them;
and the round-trip is asserted on rendered bytes, not on field equality.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..canonical import canonical_json
from ..hashing import sha256_hex
from .checkpoints import (
    FrozenMessage,
    FrozenToolCall,
    HCheckpoint,
    SamplingEnvelope,
    VendorVisibleResult,
)
from .hooks import ToolObservation

__all__ = [
    "DeferredPageBatch",
    "PageFailure",
    "BatchOutcome",
    "freeze_message",
    "thaw_message",
    "freeze_messages",
    "build_h_checkpoint",
    "reduce_tool_batch",
    "SHAPEFLOW_DEFERRED",
]

# Marker key on the object vendor's gather hands back. Vendor treats observations as opaque
# until it wraps them in ToolMessage, so a sentinel object passes through untouched.
SHAPEFLOW_DEFERRED = "__shapeflow_deferred_page_batch__"


@dataclass
class DeferredPageBatch:
    """One search tool call's vendor-visible results, acquired but not yet transformed.

    Carries exactly what vendor's step 2 produced -- URL-deduped, first-occurrence ordered,
    each `raw_content` already truncated with the pinned `max_content_length` -- plus the
    callable that would have produced vendor's string. Keeping that callable is what makes the
    whole-batch P0 fallback reproduce vendor byte-for-byte instead of approximating it.
    """

    tool_call_id: str
    tool_name: str
    results: tuple[dict, ...]
    render_vendor: Callable[[], Any]     # async () -> str; vendor steps 3-7, unchanged
    max_content_length: int

    def __post_init__(self) -> None:
        setattr(self, SHAPEFLOW_DEFERRED, True)

    def visible_results(self) -> tuple[VendorVisibleResult, ...]:
        return tuple(
            VendorVisibleResult(
                vendor_visible_order=i,
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                raw_content_id=(
                    sha256_hex(r["raw_content"].encode("utf-8"))
                    if r.get("raw_content") else None
                ),
            )
            for i, r in enumerate(self.results)
        )


def is_deferred(observation: Any) -> bool:
    return getattr(observation, SHAPEFLOW_DEFERRED, False) is True


@dataclass(frozen=True)
class PageFailure:
    """Why a P1 batch could not be published, and what it had already cost."""

    reason: str
    detail: str = ""
    # Work already spent before the failure. A P1 that fell back to P0 is not free, and the
    # ledger must carry both halves.
    spent_prompt_tokens: int = 0
    spent_completion_tokens: int = 0


@dataclass(frozen=True)
class BatchOutcome:
    """The result of reducing one assistant turn's sibling batch."""

    observations: tuple[Any, ...]          # in the original zip(observations, tool_calls) order
    checkpoint_digest: str
    fell_back: bool = False
    failure: Optional[PageFailure] = None


# --- freezing -------------------------------------------------------------------------


def _canon(value: Any) -> str:
    return canonical_json(value if value is not None else {}).decode("utf-8") \
        if isinstance(canonical_json(value if value is not None else {}), bytes) \
        else canonical_json(value if value is not None else {})


def _role_of(message: Any) -> str:
    mapping = {"ai": "ai", "human": "human", "system": "system", "tool": "tool"}
    return mapping.get(getattr(message, "type", ""), getattr(message, "type", "unknown"))


def freeze_message(message: Any) -> FrozenMessage:
    """Freeze a langchain message losslessly.

    Every field kept here is a field the rebuilt clone would otherwise lack -- and since the
    compressor reads the *rendered* message list, a missing field makes the two prompts differ
    while every hash we compute still agrees. That is the exact failure C_VISIBLE cannot
    tolerate, so the round-trip is asserted on rendered bytes.
    """
    tool_calls = tuple(
        FrozenToolCall(
            id=tc.get("id") or "",
            name=tc.get("name") or "",
            args_canonical=_canon(tc.get("args")),
        )
        for tc in (getattr(message, "tool_calls", None) or ())
    )
    artifact = getattr(message, "artifact", None)
    return FrozenMessage(
        role=_role_of(message),
        content=getattr(message, "content", ""),
        tool_calls=tool_calls,
        name=getattr(message, "name", None),
        tool_call_id=getattr(message, "tool_call_id", None),
        message_id=getattr(message, "id", None),
        additional_kwargs_canonical=_canon(getattr(message, "additional_kwargs", None)),
        response_metadata_canonical=_canon(getattr(message, "response_metadata", None)),
        usage_metadata_canonical=_canon(getattr(message, "usage_metadata", None)),
        artifact_canonical=_canon(artifact) if artifact is not None else None,
        invalid_tool_calls_canonical=_canon(getattr(message, "invalid_tool_calls", None) or []),
        status=getattr(message, "status", None),
    )


def freeze_messages(messages: Sequence[Any]) -> tuple[FrozenMessage, ...]:
    return tuple(freeze_message(m) for m in messages)


def thaw_message(frozen: FrozenMessage) -> Any:
    """Rebuild a langchain message from its frozen form.

    Imported lazily so this module stays importable (and unit-testable) without langchain --
    the dev environment runs the pure-Python tests, the run host runs the graph.
    """
    import json

    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

    def load(text: str) -> Any:
        return json.loads(text) if text else {}

    common = dict(
        content=frozen.content,
        additional_kwargs=load(frozen.additional_kwargs_canonical),
        response_metadata=load(frozen.response_metadata_canonical),
    )
    if frozen.message_id:
        common["id"] = frozen.message_id
    if frozen.name:
        common["name"] = frozen.name

    if frozen.role == "ai":
        usage = load(frozen.usage_metadata_canonical)
        message = AIMessage(
            **common,
            tool_calls=[
                {"id": tc.id, "name": tc.name, "args": load(tc.args_canonical), "type": "tool_call"}
                for tc in frozen.tool_calls
            ],
            invalid_tool_calls=load(frozen.invalid_tool_calls_canonical) or [],
        )
        if usage:
            message.usage_metadata = usage
        return message
    if frozen.role == "tool":
        kwargs = dict(common, tool_call_id=frozen.tool_call_id or "")
        if frozen.artifact_canonical is not None:
            kwargs["artifact"] = load(frozen.artifact_canonical)
        if frozen.status:
            kwargs["status"] = frozen.status
        return ToolMessage(**kwargs)
    if frozen.role == "system":
        return SystemMessage(**common)
    return HumanMessage(**common)


# --- the whole-batch reduce ------------------------------------------------------------


def build_h_checkpoint(
    *,
    task_id: str,
    researcher_id: str,
    assistant_turn_index: int,
    assistant_message: Any,
    tool_calls: Sequence[dict],
    observations: Sequence[Any],
    researcher_state_hash: str,
    sampling: SamplingEnvelope,
) -> HCheckpoint:
    """Freeze the entire assistant turn: every sibling, deferred or not, in pinned order.

    Non-search siblings are carried verbatim rather than dropped. A turn that mixed a search
    with a `think_tool` call is a different turn from one that did not, and a checkpoint that
    forgot the think call would let two different worlds fork from the same digest.
    """
    search_sets: list[tuple[str, tuple[VendorVisibleResult, ...]]] = []
    non_search: list[tuple[str, str]] = []
    for observation, tool_call in zip(observations, tool_calls):
        call_id = tool_call.get("id") or ""
        if is_deferred(observation):
            search_sets.append((call_id, observation.visible_results()))
        else:
            non_search.append((call_id, sha256_hex(str(observation).encode("utf-8"))))
    return HCheckpoint(
        task_id=task_id,
        researcher_id=researcher_id,
        assistant_turn_index=assistant_turn_index,
        assistant_message=freeze_message(assistant_message),
        sibling_tool_calls=tuple(
            FrozenToolCall(id=tc.get("id") or "", name=tc.get("name") or "",
                           args_canonical=_canon(tc.get("args")))
            for tc in tool_calls
        ),
        search_result_sets=tuple(search_sets),
        non_search_outputs=tuple(non_search),
        researcher_state_hash=researcher_state_hash,
        sampling=sampling,
    )


async def reduce_tool_batch(
    *,
    strategy,
    task_ctx,
    checkpoint: HCheckpoint,
    observations: Sequence[Any],
    tool_calls: Sequence[dict],
    component_trial: bool,
    store_checkpoint: Optional[Callable[[HCheckpoint], str]] = None,
) -> BatchOutcome:
    """Transform the whole batch, or fall back to vendor for the whole batch.

    Nothing is published from here: the caller receives a full observation list and issues one
    ``Command``. Staging happens in memory precisely so a failure part-way through cannot leave
    the graph holding some P1 outputs and some P0 ones.

    ``component_trial`` selects the failure policy. In a component trial a P1 failure is the
    measurement -- falling back would hide the failure rate being measured -- so it is recorded
    and the sample fails. End to end, the arm under test is "P1 with its fallback", so the
    whole batch re-runs vendor's path from the same checkpoint.
    """
    if store_checkpoint is not None:
        store_checkpoint(checkpoint)

    deferred_positions = [i for i, o in enumerate(observations) if is_deferred(o)]
    if not deferred_positions:
        return BatchOutcome(observations=tuple(observations),
                            checkpoint_digest=checkpoint.digest)

    try:
        produced: Sequence[ToolObservation] = await strategy.transform_tool_batch(
            task_ctx=task_ctx, checkpoint=checkpoint
        )
    except asyncio.CancelledError:
        # The run is being torn down. Converting this into "P1 failed, use P0" would fabricate
        # a P0 result for work that was abandoned -- and charge the arm for it.
        raise
    except Exception as e:  # noqa: BLE001
        failure = PageFailure(reason="STRATEGY_ERROR", detail=f"{type(e).__name__}: {e}")
        return await _resolve_failure(
            failure, observations, deferred_positions, checkpoint, component_trial
        )

    by_call = {obs.tool_call_id: obs for obs in produced}
    missing = [
        tool_calls[i].get("id") for i in deferred_positions
        if tool_calls[i].get("id") not in by_call
    ]
    if missing:
        failure = PageFailure(
            reason="INCOMPLETE_BATCH",
            detail=f"strategy returned no observation for {missing}; a partial batch is not "
                   "publishable, so the whole batch falls back",
        )
        return await _resolve_failure(
            failure, observations, deferred_positions, checkpoint, component_trial
        )

    staged = list(observations)
    for i in deferred_positions:
        staged[i] = by_call[tool_calls[i]["id"]].content
    return BatchOutcome(observations=tuple(staged), checkpoint_digest=checkpoint.digest)


async def _resolve_failure(
    failure: PageFailure,
    observations: Sequence[Any],
    deferred_positions: list[int],
    checkpoint: HCheckpoint,
    component_trial: bool,
) -> BatchOutcome:
    if component_trial:
        # Record and fail. The component trial exists to measure how often P1 cannot produce a
        # publishable batch; silently substituting P0 would erase that number.
        return BatchOutcome(
            observations=tuple(observations), checkpoint_digest=checkpoint.digest,
            fell_back=False, failure=failure,
        )
    staged = list(observations)
    for i in deferred_positions:
        staged[i] = await observations[i].render_vendor()
    return BatchOutcome(
        observations=tuple(staged), checkpoint_digest=checkpoint.digest,
        fell_back=True, failure=failure,
    )
