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
from ..p1.handle_codec import HandleDomainError
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
        def visible_raw(result: dict) -> Optional[str]:
            raw = result.get("raw_content")
            if not raw:
                return None
            # Vendor hashes/feeds only this prefix downstream. Addressing the untruncated page
            # makes the checkpoint name bytes P0 never saw and cannot be resolved by the runner's
            # same-visible-byte registry for pages longer than max_content_length.
            return str(raw)[: self.max_content_length]

        return tuple(
            VendorVisibleResult(
                vendor_visible_order=i,
                url=r.get("url", ""),
                title=r.get("title", ""),
                snippet=r.get("content", ""),
                raw_content_id=(
                    sha256_hex(visible_raw(r).encode("utf-8"))
                    if visible_raw(r) is not None else None
                ),
                source_occurrence_id=r.get("_shapeflow_occurrence_id"),
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
    # Search-call provenance for the exact bytes staged in ``observations``.  This remains
    # sidecar state until record_tool_batch_publication verifies the graph's complete
    # ToolMessage batch; a staged-but-discarded P1 output must never reach C_VISIBLE.
    publication_provenance: tuple[tuple[str, tuple[str, ...]], ...] = ()


# --- freezing -------------------------------------------------------------------------


def _canon(value: Any) -> str:
    return canonical_json(value if value is not None else {}).decode("utf-8") \
        if isinstance(canonical_json(value if value is not None else {}), bytes) \
        else canonical_json(value if value is not None else {})


def _role_of(message: Any) -> str:
    mapping = {"ai": "ai", "human": "human", "system": "system", "tool": "tool"}
    return mapping.get(getattr(message, "type", ""), getattr(message, "type", "unknown"))


def freeze_message(
    message: Any,
    *,
    published_provenance: Optional[dict] = None,
) -> FrozenMessage:
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
    if published_provenance is not None:
        if _role_of(message) != "tool":
            raise ValueError("published tool provenance cannot be attached to a non-tool message")
        content_sha = sha256_hex(canonical_json(getattr(message, "content", "")))
        expected_sha = str(published_provenance.get("content_sha256") or "")
        if content_sha != expected_sha:
            raise ValueError(
                "published tool content changed before the C checkpoint: "
                f"{content_sha!r} != {expected_sha!r}"
            )
        expected_name = str(published_provenance.get("name") or "")
        actual_name = str(getattr(message, "name", None) or "")
        if actual_name != expected_name:
            raise ValueError(
                "published tool name changed before the C checkpoint: "
                f"{actual_name!r} != {expected_name!r}"
            )
        occurrence_ids = tuple(dict.fromkeys(
            str(value)
            for value in (published_provenance.get("source_occurrence_ids") or ())
            if str(value)
        ))
        if occurrence_ids:
            if artifact is None:
                artifact = {}
            if not isinstance(artifact, dict):
                raise ValueError(
                    "a published tool message has a non-object artifact; capture-time "
                    "source provenance cannot be merged safely"
                )
            existing_ids = tuple(dict.fromkeys(
                str(value)
                for value in (artifact.get("source_occurrence_ids") or ())
                if str(value)
            ))
            if existing_ids and existing_ids != occurrence_ids:
                raise ValueError(
                    "graph-visible artifact and capture-time sidecar disagree about source "
                    "occurrences"
                )
            artifact = {
                **artifact,
                "source_occurrence_ids": list(occurrence_ids),
                "provenance_capture": "whole_batch_publish_v1",
                "published_content_sha256": content_sha,
            }
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


def freeze_messages(
    messages: Sequence[Any],
    *,
    tool_provenance: Optional[dict[str, dict]] = None,
) -> tuple[FrozenMessage, ...]:
    """Freeze messages, optionally enriching only the checkpoint clone with provenance.

    ``tool_provenance`` is a run-local sidecar committed at the graph's atomic publication
    point.  The live ToolMessage is never mutated, preserving hooks-off/explicit-P0 prompt
    parity.  Missing or empty provenance stays missing/empty and therefore remains
    TOOL_UNATTRIBUTED_CONTEXT downstream.
    """
    provenance = tool_provenance or {}
    frozen: list[FrozenMessage] = []
    for message in messages:
        entry = None
        if _role_of(message) == "tool":
            call_id = str(getattr(message, "tool_call_id", None) or "")
            entry = provenance.get(call_id)
        frozen.append(freeze_message(message, published_provenance=entry))
    return tuple(frozen)


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
    # `is not None`, not truthiness. An empty string is a value the vendor really set, and
    # dropping it here rebuilds the message with `None` instead -- which re-freezes to a
    # different FrozenMessage and therefore a different checkpoint digest. The C fork proves
    # it started from the planned boundary by re-deriving that digest, so a field that does
    # not survive freeze -> thaw -> freeze silently makes every fork from such a message fail.
    if frozen.message_id is not None:
        common["id"] = frozen.message_id
    if frozen.name is not None:
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
        # `or ""` would turn None into "", which re-freezes to a different value and moves the
        # digest. langchain requires tool_call_id on a real ToolMessage, so None here means
        # this did not come from one -- same unreachable-state class as `status` below.
        if frozen.tool_call_id is None:
            raise ValueError(
                "tool message has no tool_call_id; a frozen ToolMessage always carries one, "
                "so this state did not come from a real message"
            )
        kwargs = dict(common, tool_call_id=frozen.tool_call_id)
        if frozen.artifact_canonical is not None:
            kwargs["artifact"] = load(frozen.artifact_canonical)
        # ToolMessage.status is a Literal["success", "error"] that defaults to "success", so a
        # real frozen tool message always carries one of those two. Anything else -- None, "",
        # a typo -- cannot have come from freezing a live message, and quietly letting the
        # default fill it in would produce a state that re-freezes to *different* bytes and
        # therefore a different checkpoint digest. Refuse instead: a fork proves it started
        # from the planned boundary by re-deriving that digest, so a silent substitution here
        # is indistinguishable from a tampered checkpoint.
        if frozen.status not in ("success", "error"):
            raise ValueError(
                f"tool message has status {frozen.status!r}; a frozen ToolMessage carries "
                "'success' or 'error', so this state did not come from a real message"
            )
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
    researcher_coordinate: Optional[tuple[int, int, str]] = None,
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
            # The tool call's id is assigned HERE, not inside tavily_search: a tool cannot see
            # its own call id, and the zip order is where the mapping actually exists.
            observation.tool_call_id = call_id
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
        researcher_coordinate=researcher_coordinate,
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
    except HandleDomainError:
        # The frozen publication-handle domain cannot name this batch's spans. That is a defect
        # in frozen protocol, not a property of the treatment, and the fallback is not an honest
        # reading of it: a cell recorded as "P1 tried and fell back" would be indistinguishable
        # from one where P1 was never reachable at all. That indistinguishability is what let an
        # entirely inert P1 look like 146 completed canary cells, so this stops the cell instead.
        raise
    except Exception as e:  # noqa: BLE001
        failure = PageFailure(reason="STRATEGY_ERROR", detail=f"{type(e).__name__}: {e}")
        return await _resolve_failure(
            failure, observations, deferred_positions, checkpoint, component_trial
        )

    expected_call_ids = [
        str(tool_calls[i].get("id") or "") for i in deferred_positions
    ]
    produced_call_ids = [str(obs.tool_call_id or "") for obs in produced]
    if produced_call_ids != expected_call_ids or len(set(produced_call_ids)) != len(
        produced_call_ids
    ):
        failure = PageFailure(
            reason="INCOMPLETE_BATCH",
            detail=(
                "strategy output ids/order do not equal the deferred search calls "
                f"(expected {expected_call_ids}, got {produced_call_ids}); a partial, duplicate, "
                "extra, or reordered batch is not publishable, so the whole batch falls back"
            ),
        )
        return await _resolve_failure(
            failure, observations, deferred_positions, checkpoint, component_trial
        )

    by_call = {str(obs.tool_call_id or ""): obs for obs in produced}
    offered_occurrences = {
        str(call_id): {
            str(result.source_occurrence_id)
            for result in results
            if result.source_occurrence_id
        }
        for call_id, results in checkpoint.search_result_sets
    }
    publication_provenance: list[tuple[str, tuple[str, ...]]] = []
    for call_id in expected_call_ids:
        claimed = tuple(dict.fromkeys(
            str(value) for value in by_call[call_id].source_occurrence_ids if str(value)
        ))
        unknown = sorted(set(claimed) - offered_occurrences.get(call_id, set()))
        if unknown:
            failure = PageFailure(
                reason="PROVENANCE_OUTSIDE_CHECKPOINT",
                detail=(
                    f"strategy attributed tool call {call_id!r} to occurrences {unknown} that "
                    "were not in that call's frozen H checkpoint"
                ),
            )
            return await _resolve_failure(
                failure, observations, deferred_positions, checkpoint, component_trial
            )
        publication_provenance.append((call_id, claimed))

    staged = list(observations)
    for i in deferred_positions:
        staged[i] = by_call[str(tool_calls[i].get("id") or "")].content
    return BatchOutcome(
        observations=tuple(staged),
        checkpoint_digest=checkpoint.digest,
        publication_provenance=tuple(publication_provenance),
    )


async def _resolve_failure(
    failure: PageFailure,
    observations: Sequence[Any],
    deferred_positions: list[int],
    checkpoint: HCheckpoint,
    component_trial: bool,
) -> BatchOutcome:
    if component_trial:
        # Record and fail. The caller must terminate the component sample before vendor wraps
        # these opaque deferred objects in ToolMessage. Returning the original observations is
        # only a transport for the failure metadata; it is never a publishable result.
        return BatchOutcome(
            observations=tuple(observations), checkpoint_digest=checkpoint.digest,
            fell_back=False, failure=failure,
        )
    # Vendor executes sibling tool calls concurrently. A P1 failure must not quietly turn the
    # fallback into a sequential P0 implementation: that changes the critical path and makes the
    # failure penalty an artefact of the adapter. Gather all vendor closures first, then stage the
    # complete batch. If one closure fails, nothing is published.
    rendered = await asyncio.gather(
        *(observations[i].render_vendor() for i in deferred_positions)
    )
    staged = list(observations)
    for i, value in zip(deferred_positions, rendered):
        staged[i] = value
    return BatchOutcome(
        observations=tuple(staged), checkpoint_digest=checkpoint.digest,
        fell_back=True, failure=failure,
        # E2E fallback publishes vendor's rendering of every result in each deferred call.
        # Use the frozen checkpoint, not a URL reverse lookup and not the failed P1 selection.
        publication_provenance=tuple(
            (
                str(call_id),
                tuple(dict.fromkeys(
                    str(result.source_occurrence_id)
                    for result in results
                    if result.source_occurrence_id
                )),
            )
            for call_id, results in getattr(checkpoint, "search_result_sets", ())
        ),
    )
