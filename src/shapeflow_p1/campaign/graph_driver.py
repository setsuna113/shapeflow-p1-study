"""Running one cell of the experiment on the real, patched Open Deep Research graph.

A *cell* is one (task, arm, seed). Driving it means four things, and each is arranged so the
thing that could quietly invalidate the measurement is impossible rather than merely discouraged.

**The world does not move.** ``open_deep_research.utils.tavily_search_async`` is rebound, in
this process, to a deterministic query over the task's frozen pool. Rebinding here rather than
in the patch keeps the patched bytes -- and therefore both actual-graph parity gates --
untouched. There is no code path from the frozen backend to a live search: the replacement
never constructs an HTTP client, so a miss returns an empty result set rather than silently
reaching the network.

**Every model call is attributed.** The cell is registered with the provider and
``OPENAI_BASE_URL`` points at that cell's path, so vendor's own ``init_chat_model`` calls carry
the cell identity without ODR having to know anything about it. The four model roles are given
distinct aliases, which the provider rewrites to the one served model -- so P0 and P1 issue
byte-identical upstream requests while remaining separable in the work ledger.

**The arm is bound for exactly its own execution.** ``strategies_bound`` and ``bind_run`` are
context managers whose reset is in a ``finally``. Researchers run concurrently; a binding that
survived an exception would execute one arm's strategy while recording another arm's id, which
is a mislabelled observation rather than a crash.

**Environment mutation is scoped.** ``OPENAI_BASE_URL`` and friends are restored afterwards, so
a crashed cell cannot leave the next one pointing at its URL.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..acquire.frozen_search import FrozenTaskCorpusBackend
from ..acquire.snapshot_store import SnapshotStore
from ..acquire.source_pool import SourcePool
from ..odr.checkpoints import SamplingEnvelope
from ..odr.hooks import StrategyBundle, TaskContext, strategies_bound
from ..odr.vendor_hooks import RunBinding, bind_run
from .settings import Settings

__all__ = [
    "CellSpec",
    "CellResult",
    "frozen_search_async",
    "install_frozen_search",
    "odr_config",
    "run_cell",
]


@dataclass(frozen=True)
class CellSpec:
    """One unit of execution: a task, an arm, a seed, and the identity they are recorded under."""

    run_id: str
    task_id: str
    arm_id: str
    page_variant: str
    close_variant: str
    replicate_id: str
    seed: int
    work_key: str
    question: str
    cell_token: str
    component_trial: bool = False

    @property
    def variant_id(self) -> str:
        """The composed arm id. H+C is the two halves, named as such, not a third thing."""
        if self.page_variant == "P0" and self.close_variant == "P0":
            return "P0"
        if self.close_variant == "P0":
            return self.page_variant
        if self.page_variant == "P0":
            return self.close_variant
        return f"{self.page_variant}+{self.close_variant}"


@dataclass
class CellResult:
    final_report: str = ""
    notes: tuple[str, ...] = ()
    raw_notes: tuple[str, ...] = ()
    events: list = field(default_factory=list)
    checkpoints: list = field(default_factory=list)
    error: Optional[str] = None

    @property
    def ok(self) -> bool:
        return self.error is None


def frozen_search_async(backend: FrozenTaskCorpusBackend, *, max_results_default: int = 5):
    """A drop-in replacement for vendor's ``tavily_search_async`` over the frozen pool.

    Returns vendor's exact shape -- a list of ``{"query", "results"}`` with Tavily's field names
    -- so everything downstream, including vendor's own URL dedup and truncation, behaves as it
    does against the live API. A query with no match returns an empty result set: that is a real
    state of the frozen world (plan §8.2), and inventing a result to avoid it would be
    fabricating evidence.
    """

    async def search(queries, max_results=max_results_default, topic="general",
                     include_raw_content=True, config=None):
        payload = []
        for query in queries:
            records = backend.search(str(query), max_results=max_results)
            payload.append({
                "query": query,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "content": r.content,
                        "score": r.score,
                        "raw_content": r.raw_content if include_raw_content else None,
                    }
                    for r in records
                ],
            })
        return payload

    return search


@contextlib.contextmanager
def install_frozen_search(pool: SourcePool, snapshots: SnapshotStore, *, max_results: int):
    """Rebind vendor's search for the duration of one cell, and put it back afterwards.

    Restoring in ``finally`` matters: a cell that raised while the live function was replaced
    would leave the next cell searching the previous task's corpus, and the results would look
    entirely plausible.
    """
    import open_deep_research.utils as vendor_utils

    backend = FrozenTaskCorpusBackend(pool, snapshots)
    original = vendor_utils.tavily_search_async
    vendor_utils.tavily_search_async = frozen_search_async(
        backend, max_results_default=max_results)
    try:
        yield backend
    finally:
        vendor_utils.tavily_search_async = original


@contextlib.contextmanager
def _environment(overrides: dict[str, str]):
    """Set environment variables for one cell and restore them, including deletions."""
    previous = {key: os.environ.get(key) for key in overrides}
    os.environ.update(overrides)
    try:
        yield
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


def odr_config(settings: Settings, *, cell: CellSpec) -> dict:
    """Vendor's configurable block. Identical for every arm except the model aliases.

    The aliases are the only difference, and the provider rewrites them all to one served model
    -- so what actually reaches the engine is the same for P0 and P1, and the separation exists
    only in the ledger.
    """
    odr = settings.get("week1", "odr")
    return {"configurable": {
        "search_api": str(odr["search_api"]),
        "allow_clarification": False,
        "max_react_tool_calls": int(odr["max_react_tool_calls"]),
        "max_content_length": int(odr["max_content_length"]),
        "max_structured_output_retries": int(odr["max_structured_output_retries"]),
        "max_researcher_iterations": int(odr["max_researcher_iterations"]),
        "max_concurrent_research_units": int(odr["max_concurrent_research_units"]),
        "research_model": "openai:qwen-research",
        "research_model_max_tokens": int(odr["research_model_max_tokens"]),
        "summarization_model": "openai:qwen-summarize",
        "summarization_model_max_tokens": int(odr["summarization_model_max_tokens"]),
        "compression_model": "openai:qwen-compress",
        "compression_model_max_tokens": int(odr["compression_model_max_tokens"]),
        "final_report_model": "openai:qwen-final",
        "final_report_model_max_tokens": int(odr["final_report_model_max_tokens"]),
        "mcp_config": None,
    }}


async def run_cell(
    settings: Settings,
    cell: CellSpec,
    *,
    pool: SourcePool,
    snapshots: SnapshotStore,
    bundle: StrategyBundle,
    provider_base_url: str,
    runner_token: str,
    store_checkpoint: Optional[Callable[[Any], str]] = None,
    graph: Any = None,
) -> CellResult:
    """Run one cell on the real graph and return what it produced.

    ``graph`` is injected only so tests can drive a smaller compiled graph; production passes
    None and the module imports vendor's own compiled ``deep_researcher``.
    """
    from langchain_core.messages import HumanMessage

    if graph is None:
        import open_deep_research.deep_researcher as vendor_graph

        graph = vendor_graph.deep_researcher

    events: list = []
    checkpoints: list = []

    def on_event(kind: str, payload: dict) -> None:
        events.append({"kind": kind, **payload})

    def capture(checkpoint) -> str:
        digest = checkpoint.digest
        checkpoints.append({"kind": type(checkpoint).__name__, "digest": digest})
        if store_checkpoint is not None:
            store_checkpoint(checkpoint)
        return digest

    sampling = settings.get("stack", "sampling")
    envelope = SamplingEnvelope(
        model=str(settings.get("week1", "provider", "served_model")),
        temperature=float(sampling["temperature"]),
        top_p=float(sampling["top_p"]),
        max_tokens=int(settings.get("week1", "odr", "research_model_max_tokens")),
        seed=cell.seed,
    )
    task_ctx = TaskContext(
        task_id=cell.task_id,
        protocol_sha=settings.shas["week1"],
        variant_id=cell.variant_id,
        seed=cell.seed,
        research_topic=cell.question,
        max_content_length=int(settings.get("week1", "odr", "max_content_length")),
        selected_token_budget=int(settings.get("week1", "measurement",
                                               "selected_token_budget")),
    )
    binding = RunBinding(
        task_id=cell.task_id,
        researcher_id=f"{cell.task_id}:{cell.arm_id}:{cell.replicate_id}",
        attempt_id=cell.work_key,
        task_ctx=task_ctx,
        component_trial=cell.component_trial,
        sampling=envelope,
        store_checkpoint=capture,
        on_event=on_event,
    )

    max_results = int(settings.get("acquisition", "frozen_corpus", "top_k"))
    result = CellResult(events=events, checkpoints=checkpoints)
    env = {
        "OPENAI_BASE_URL": f"{provider_base_url.rstrip('/')}/v1/cell/{cell.cell_token}",
        "OPENAI_API_KEY": runner_token,
        # Vendor reads TAVILY_API_KEY at import of its search path. The frozen backend never
        # uses it; a recognisable placeholder makes a leak into a log obviously not a key.
        "TAVILY_API_KEY": "@SHAPEFLOW_FROZEN_CORPUS@",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    with _environment(env), install_frozen_search(pool, snapshots, max_results=max_results), \
            strategies_bound(bundle), bind_run(binding):
        try:
            state = await graph.ainvoke(
                {"messages": [HumanMessage(content=cell.question)]},
                odr_config(settings, cell=cell),
            )
        except Exception as e:  # noqa: BLE001 - recorded, never converted into a P0 result
            result.error = f"{type(e).__name__}: {e}"
            return result

    result.final_report = str(state.get("final_report", "") or "")
    result.notes = tuple(str(n) for n in (state.get("notes") or ()))
    result.raw_notes = tuple(str(n) for n in (state.get("raw_notes") or ()))
    return result


def summarize_events(events: Sequence[dict]) -> dict:
    """Counts the canary and the ledger both need, derived from the run's own events."""
    kinds: dict[str, int] = {}
    for event in events:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1
    return {
        "page_batches_deferred": kinds.get("PAGE_BATCH_DEFERRED", 0),
        "page_batches_reduced": kinds.get("PAGE_BATCH_REDUCED", 0),
        "page_fallbacks": sum(1 for e in events
                              if e["kind"] == "PAGE_BATCH_REDUCED" and e.get("fell_back")),
        "close_reduced": kinds.get("CLOSE_REDUCED", 0),
        "close_failed": kinds.get("CLOSE_FAILED", 0),
        "close_deferred_to_vendor": kinds.get("CLOSE_DEFERRED_TO_VENDOR", 0),
        "close_cancelled": kinds.get("CLOSE_CANCELLED", 0),
    }
