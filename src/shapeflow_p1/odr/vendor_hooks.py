"""What the vendor patch calls. Keeping this here keeps the patch to a dozen dispatch lines.

The smaller the patch, the smaller the surface on which "hooks off is byte-for-byte vendor"
can fail. Every branch below returns ``None`` when no strategy is bound, and the patch's only
job is to fall through to vendor's untouched code in that case.

Vendor's own concurrency is preserved on both sides. Hooks-off runs vendor's inner
``asyncio.gather`` over summarization tasks exactly as written; the explicit-P0 strategy runs
the *same closure*, so it reproduces vendor's timing rather than approximating it with a
sequential loop. An arm whose summaries ran one at a time would have a different critical path
from the baseline it is compared against, and that difference would be read as a treatment
effect.
"""

from __future__ import annotations

import asyncio
import contextvars
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .adapter import DeferredPageBatch, build_h_checkpoint, is_deferred, reduce_tool_batch
from .checkpoints import SamplingEnvelope
from .hooks import current_strategies

__all__ = [
    "RunBinding",
    "bind_run",
    "current_run",
    "page_hook_active",
    "defer_page_batch",
    "reduce_published_batch",
    "close_hook_active",
    "run_close_strategy",
]


@dataclass(frozen=True)
class RunBinding:
    """Per-run identity and policy the patch threads through, without a mutable global.

    A module-level "current run" would be shared by every researcher the supervisor runs
    concurrently; a ContextVar is copied into each task, so an arm sees only what its own run
    bound.
    """

    task_id: str
    researcher_id: str
    attempt_id: str
    task_ctx: Any                       # p1 TaskContext
    component_trial: bool = False
    sampling: Optional[SamplingEnvelope] = None
    store_checkpoint: Optional[Callable[[Any], str]] = None
    on_event: Optional[Callable[[str, dict], None]] = None


_RUN: contextvars.ContextVar[Optional[RunBinding]] = contextvars.ContextVar(
    "shapeflow_run_binding", default=None
)


def bind_run(binding: Optional[RunBinding]):
    """Context manager binding the run for the duration of the block, resetting in finally."""
    import contextlib

    @contextlib.contextmanager
    def _bound():
        token = _RUN.set(binding)
        try:
            yield binding
        finally:
            _RUN.reset(token)

    return _bound()


def current_run() -> Optional[RunBinding]:
    return _RUN.get()


def _emit(kind: str, payload: dict) -> None:
    run = current_run()
    if run is not None and run.on_event is not None:
        run.on_event(kind, payload)


# --- WEBPAGE_P1 -----------------------------------------------------------------------


def page_hook_active() -> bool:
    """True only when both a strategy and a run are bound. Either alone is a misconfiguration
    rather than a reason to take a half-instrumented path."""
    return current_strategies() is not None and current_run() is not None


def defer_page_batch(
    *, tool_call_id: str, tool_name: str, unique_results: dict,
    render_vendor: Callable[[], Any], max_content_length: int,
) -> DeferredPageBatch:
    """Package one search call's vendor-visible results without transforming them.

    Everything vendor's step 2 produced is kept in its order: URL-deduped, first-occurrence
    ordered. ``render_vendor`` is vendor's own steps 3-7 as a closure, so the whole-batch P0
    fallback reproduces vendor's bytes instead of re-implementing them.
    """
    batch = DeferredPageBatch(
        tool_call_id=tool_call_id,
        tool_name=tool_name,
        results=tuple(unique_results.values()),
        render_vendor=render_vendor,
        max_content_length=max_content_length,
    )
    _emit("PAGE_BATCH_DEFERRED", {"tool_call_id": tool_call_id, "results": len(batch.results)})
    return batch


async def reduce_published_batch(
    *, assistant_message: Any, tool_calls: Sequence[dict], observations: Sequence[Any],
    researcher_state_hash: str, assistant_turn_index: int,
) -> Sequence[Any]:
    """Turn the whole sibling batch into the observations vendor will publish.

    Called after vendor's ``asyncio.gather`` and before its ``ToolMessage`` list is built, which
    is the only point where the complete assistant turn exists and nothing has been published.
    Returns a full observation list so the caller issues exactly one ``Command``.
    """
    run = current_run()
    bundle = current_strategies()
    if run is None or bundle is None or not any(is_deferred(o) for o in observations):
        return observations

    checkpoint = build_h_checkpoint(
        task_id=run.task_id,
        researcher_id=run.researcher_id,
        assistant_turn_index=assistant_turn_index,
        assistant_message=assistant_message,
        tool_calls=tool_calls,
        observations=observations,
        researcher_state_hash=researcher_state_hash,
        sampling=run.sampling or SamplingEnvelope(model="", temperature=0.0, top_p=1.0,
                                                 max_tokens=0),
    )
    outcome = await reduce_tool_batch(
        strategy=bundle.page,
        task_ctx=run.task_ctx,
        checkpoint=checkpoint,
        observations=observations,
        tool_calls=tool_calls,
        component_trial=run.component_trial,
        store_checkpoint=run.store_checkpoint,
    )
    _emit("PAGE_BATCH_REDUCED", {
        "checkpoint": outcome.checkpoint_digest,
        "fell_back": outcome.fell_back,
        "failure": outcome.failure.reason if outcome.failure else None,
        "detail": outcome.failure.detail if outcome.failure else None,
        "siblings": len(observations),
    })
    if outcome.failure is not None and not outcome.fell_back:
        # Component trial: the failure IS the measurement. Raising here would be caught by
        # vendor's per-tool handler and turned into an error string for one sibling, which is
        # exactly the partial-batch state that must never exist -- so the batch is published
        # with vendor's own outputs and the failure is recorded out of band.
        return observations
    return outcome.observations


# --- RESEARCHER_CLOSE -------------------------------------------------------------------


def close_hook_active() -> bool:
    return page_hook_active()


async def run_close_strategy(*, researcher_messages: Sequence[Any], close_reason: str,
                             evidence_span_ids: Sequence[str] = (),
                             query_attempt_ids: Sequence[str] = ()) -> Optional[dict]:
    """Clone, checkpoint and hand the close boundary to the strategy.

    Reached independently of the page path: a researcher can close having published no batch at
    all (``ResearchComplete`` on the first turn, or a no-tool exit), and those are precisely the
    cheap runs a close hook hanging off the H path would skip.

    Returns vendor's own two fields so the patch can ``return`` it directly, or None when no
    strategy is bound.
    """
    from .adapter import freeze_messages
    from .checkpoints import CCheckpoint, EvidenceManifest

    run = current_run()
    bundle = current_strategies()
    if run is None or bundle is None:
        return None

    # Cloned BEFORE vendor's in-place append. compress_research mutates the list it was given,
    # so a clone taken afterwards would carry the compression instruction -- and, through the
    # shared list, poison every later fork from this same state.
    frozen = freeze_messages(researcher_messages)
    checkpoint = CCheckpoint(
        task_id=run.task_id,
        researcher_id=run.researcher_id,
        researcher_messages=frozen,
        evidence_manifest=EvidenceManifest(span_ids=tuple(evidence_span_ids)),
        query_attempt_ids=tuple(query_attempt_ids),
        close_reason=close_reason,
        sampling=run.sampling or SamplingEnvelope(model="", temperature=0.0, top_p=1.0,
                                                 max_tokens=0),
    )
    if run.store_checkpoint is not None:
        run.store_checkpoint(checkpoint)

    try:
        handoff = await bundle.close.close_researcher(
            task_ctx=run.task_ctx, checkpoint=checkpoint
        )
    except asyncio.CancelledError:
        _emit("CLOSE_CANCELLED", {"checkpoint": checkpoint.digest,
                                  "close_reason": close_reason})
        raise
    except Exception as e:  # noqa: BLE001
        _emit("CLOSE_FAILED", {"checkpoint": checkpoint.digest,
                               "close_reason": close_reason,
                               "detail": f"{type(e).__name__}: {e}"})
        if run.component_trial:
            return None          # measured as a failure; vendor path publishes
        return None              # end-to-end: fall back to vendor compression
    _emit("CLOSE_REDUCED", {"checkpoint": checkpoint.digest, "close_reason": close_reason})
    return {
        "compressed_research": handoff.compressed_research,
        "raw_notes": list(handoff.raw_notes),
    }
