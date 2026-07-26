"""The strategy contract the patch dispatches to, and the context that selects it.

The patched vendor does not know about P0 or P1. At each boundary it captures the frozen
checkpoint, reads the strategy bound to the current execution context, and asks it to
produce the observation(s) or the handoff. Everything here is langchain-free: strategies
consume the frozen checkpoints from ``checkpoints`` and return small serializable results
that the adapter turns back into langchain messages.

The active strategy is held in a :class:`contextvars.ContextVar`, not a module global. The
supervisor runs researchers concurrently (``asyncio.gather`` over ``researcher_subgraph``),
so two researchers on two arms can be in flight at once; a module global would let one
arm's strategy leak into another's boundary. A ContextVar is copied into each task, so each
researcher sees exactly the strategy its run bound.
"""

from __future__ import annotations

import contextlib
import contextvars
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from .checkpoints import CCheckpoint, HCheckpoint

__all__ = [
    "TaskContext",
    "ToolObservation",
    "ResearcherHandoff",
    "PageTransformStrategy",
    "ResearchCloseStrategy",
    "StrategyBundle",
    "bind_strategies",
    "strategies_bound",
    "current_strategies",
]


@dataclass(frozen=True)
class TaskContext:
    """The decision-time context a strategy is allowed to see.

    Deliberately narrow: only bytes already visible to P0 (the question/topic, the vendor
    config knobs). It carries NO acquisition spec, truth packet, or gold facet -- those are
    evaluator-only and their absence here is a structural guarantee, not a convention.
    """

    task_id: str
    protocol_sha: str
    variant_id: str
    seed: int
    # Bytes the researcher/supervisor already put in front of P0.
    research_topic: str
    # Vendor knobs both arms share.
    max_content_length: int
    selected_token_budget: Optional[int] = None


@dataclass(frozen=True)
class ToolObservation:
    """One tool message a page-transform strategy produces. The adapter renders this into a
    langchain ``ToolMessage`` in the pinned join order.

    ``source_occurrence_ids`` is capture-time sidecar provenance for the bytes in ``content``.
    It is deliberately not placed on the graph-visible ToolMessage: doing that would change
    explicit-P0 prompts and destroy parity.  The whole-batch publication hook commits it only
    after the exact ToolMessage content/order has been verified, and the close hook injects it
    only into the frozen C checkpoint.
    """

    tool_call_id: str
    name: str
    content: str
    source_occurrence_ids: tuple[str, ...] = ()


@dataclass(frozen=True)
class ResearcherHandoff:
    """What a close strategy hands back to the supervisor, matching the two fields vendor
    ``compress_research`` returns so P0 and P1 are drop-in comparable."""

    compressed_research: str
    raw_notes: tuple[str, ...]


@runtime_checkable
class PageTransformStrategy(Protocol):
    """Turns a captured WEBPAGE boundary into the tool observations to publish.

    P0's implementation reproduces vendor ``summarize_webpage`` byte-for-byte; P1's runs
    the selector path. Both must emit one observation per sibling tool call, in the batch's
    pinned order, because the publish unit is the whole batch.

    Async because the vendor path is async and a strategy makes model calls. A sync strategy
    would either block the event loop -- changing the batching and timing this study measures --
    or force the adapter to thread-hop, which changes them differently."""

    async def transform_tool_batch(
        self, *, task_ctx: TaskContext, checkpoint: HCheckpoint
    ) -> Sequence[ToolObservation]: ...


@runtime_checkable
class ResearchCloseStrategy(Protocol):
    """Turns a captured RESEARCHER_CLOSE boundary into the handoff.

    P0 reproduces vendor ``compress_research``; P1 runs the close selector. The strategy
    receives the close reason so it can (for analysis) distinguish the three exit paths,
    but every path enters the same strategy.

    RESEARCHER_CLOSE is its own boundary, not a consequence of the page batch: a researcher can
    close having published no page batch at all (``ResearchComplete`` on the first turn, or a
    no-tool exit), so a close hook reached only through the H path would silently skip those
    runs -- and they are exactly the cheap ones."""

    async def close_researcher(
        self, *, task_ctx: TaskContext, checkpoint: CCheckpoint
    ) -> ResearcherHandoff: ...


@dataclass(frozen=True)
class StrategyBundle:
    """The pair of strategies plus the variant they implement, bound for one run."""

    variant_id: str
    page: PageTransformStrategy
    close: ResearchCloseStrategy


_ACTIVE: contextvars.ContextVar[Optional[StrategyBundle]] = contextvars.ContextVar(
    "shapeflow_active_strategies", default=None
)


def bind_strategies(bundle: StrategyBundle) -> contextvars.Token:
    """Bind ``bundle`` for the current context. Returns a token to restore the previous
    binding, so nested/concurrent runs don't clobber each other.

    Prefer :func:`strategies_bound`, which cannot leak the binding.
    """
    return _ACTIVE.set(bundle)


@contextlib.contextmanager
def strategies_bound(bundle: Optional[StrategyBundle]):
    """Bind for the duration of the block and restore in ``finally``.

    The reset has to be unconditional. If an arm raises -- or is cancelled -- between bind and
    reset, a manual token reset is skipped and the binding survives into whatever runs next in
    that context. The supervisor runs researchers concurrently, so the next thing is very often
    a *different arm*, which would then execute P1's strategy while recording P0's variant id:
    a mislabelled observation, not a crash, and nothing downstream could detect it.
    """
    token = _ACTIVE.set(bundle)
    try:
        yield bundle
    finally:
        _ACTIVE.reset(token)


def current_strategies() -> Optional[StrategyBundle]:
    """The strategies bound in the current context, or None (vendor default path)."""
    return _ACTIVE.get()
