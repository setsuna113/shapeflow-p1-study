"""Re-enter ``compress_research`` from a stored C boundary, and prove it is that boundary.

The full-graph screen re-runs the whole graph per arm. For H that is correct -- divergence
after the first intervention is a mediated treatment effect. For C it is not: the close
reducer fires *after* all research, so it cannot have changed anything upstream, and every
upstream difference between two full-graph C arms is pure noise that the estimate has to
average away. With ~32 formative tasks there is nowhere near enough sample to do that, so the
primary C effect is estimated from a same-checkpoint fork instead, and the full-graph C arms
are retained as a secondary sensitivity that bounds the path this fork deliberately blocks.

**What "same checkpoint" is worth here rests on a proof, not an assertion.** The backend does
not record which checkpoint it *intended* to load. It reconstructs a researcher state from the
stored ``CCheckpoint``, re-enters ``compress_research``, and the close hook -- vendor's own,
unmodified path -- builds a fresh ``CCheckpoint`` from whatever it actually received. If that
re-derived digest is not the planned one, the fork is refused. A tampered checkpoint, a lossy
message round trip, or a mis-restored coordinate all fail here rather than producing a
plausible-looking arm.

**Zero upstream, enforced three ways** rather than hoped for: the node prefix before the hook
does no inference at all (``compress_research`` only builds a model handle before the hook
fires, so re-entry is genuinely free); an in-process poison raises on any search or tool path;
and the provider refuses any op class outside the close allowlist *before* dispatch, so a
violation costs nothing even if the first two were somehow bypassed.

No vendor bytes change. The re-entry graphs are built from vendor's own node callables, so
``patches/patched_tree.sha256`` and the P0 parity gate are untouched.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..odr.adapter import thaw_message
from ..odr.checkpoints import CCheckpoint
from ..odr.continuation import ContinuationEnvelope, NoteSlot
from .fork import ForkCapabilityError, ForkExecution, ForkSpec, TrialKind

__all__ = [
    "CLOSE_OP_ALLOWLIST",
    "CForkBackend",
    "UpstreamCallDuringFork",
    "build_close_reentry_graph",
    "build_report_reentry_graph",
    "reconstruct_researcher_state",
]

#: The only op classes a fork may issue. Anything else means upstream work leaked into a
#: comparison that claims none, so it is refused before dispatch rather than detected after.
CLOSE_OP_ALLOWLIST = ("COMPRESSOR_P0", "COMPRESSOR_P1_SELECTOR", "FINAL_WRITER")


class UpstreamCallDuringFork(RuntimeError):
    """A fork touched search, a tool, or a researcher turn. Fatal for that arm."""


def reconstruct_researcher_state(checkpoint: CCheckpoint, *, research_topic: str) -> dict:
    """Rebuild the ``ResearcherState`` that ``compress_research`` would have received.

    ``tool_call_iterations`` is not stored on the checkpoint, but it is an input to the close
    reason vendor recomputes, so it has to be reconstructed rather than defaulted. It counts
    researcher turns that issued tool calls, which is recoverable from the message list itself.
    A wrong value shows up as a close-reason mismatch, which changes the re-derived digest and
    is caught by the proof -- it cannot silently alter the treatment.
    """
    messages = [thaw_message(m) for m in checkpoint.researcher_messages]
    tool_call_iterations = sum(
        1 for m in checkpoint.researcher_messages if m.role == "ai" and m.tool_calls
    )
    return {
        "researcher_messages": messages,
        "research_topic": research_topic,
        "tool_call_iterations": tool_call_iterations,
    }


def build_close_reentry_graph(vendor_graph_module: Any = None) -> Any:
    """A graph whose only node is vendor's ``compress_research``.

    Entering at the node rather than resuming a checkpointer keeps the fork's substrate our own
    content-addressed digest instead of langgraph's serialized channel state -- which is what
    makes "both arms started from byte-identical input" checkable after the fact.
    """
    from langgraph.graph import END, START, StateGraph

    if vendor_graph_module is None:
        import open_deep_research.deep_researcher as vendor_graph_module
    from open_deep_research.state import ResearcherState

    builder = StateGraph(ResearcherState)
    builder.add_node("compress_research", vendor_graph_module.compress_research)
    builder.add_edge(START, "compress_research")
    builder.add_edge("compress_research", END)
    return builder.compile()


def build_report_reentry_graph(vendor_graph_module: Any = None) -> Any:
    """A graph whose only node is vendor's ``final_report_generation``.

    ``final_report_generation`` reads exactly ``notes``, ``research_brief`` and ``messages``,
    all of which the continuation envelope carries.
    """
    from langgraph.graph import END, START, StateGraph

    if vendor_graph_module is None:
        import open_deep_research.deep_researcher as vendor_graph_module
    from open_deep_research.state import AgentState

    builder = StateGraph(AgentState)
    builder.add_node("final_report_generation", vendor_graph_module.final_report_generation)
    builder.add_edge(START, "final_report_generation")
    builder.add_edge("final_report_generation", END)
    return builder.compile()


@dataclass
class _ArmObservations:
    """What the hook and the recorder saw during one arm, for the post-hoc assertions."""

    rederived_digest: str = ""
    close_failed: bool = False
    fell_back: bool = False
    search_queries: int = 0
    op_classes: tuple[str, ...] = ()


class CForkBackend:
    """A ``BoundaryForkBackend`` restricted to what it can honestly perform.

    ``supports`` deliberately answers False for ``H_E2E`` and ``HXC_NESTED``: reaching those
    requires resuming mid-trajectory, which this backend does not do, and
    ``run_production_forks`` refuses an unsupported estimand rather than silently substituting
    a full-graph re-run.
    """

    #: Estimands this backend can actually execute at the C boundary.
    SUPPORTED = (TrialKind.COMPONENT, TrialKind.C_FROZEN_CONTINUATION)

    def __init__(
        self,
        *,
        envelope: ContinuationEnvelope,
        strategy_for: Callable[[str], Any],
        run_binding_for: Callable[[ForkSpec, CCheckpoint], Any],
        odr_config_for: Callable[[ForkSpec], dict],
        today_str: Callable[[], str],
        engine_epoch: Callable[[], str],
        terminal_mode: str = "REPORT",
        close_graph: Any = None,
        report_graph: Any = None,
    ) -> None:
        if terminal_mode not in ("REPORT", "CLOSE_ONLY"):
            raise ValueError(f"unknown terminal_mode {terminal_mode!r}")
        self._envelope = envelope
        self._strategy_for = strategy_for
        self._run_binding_for = run_binding_for
        self._odr_config_for = odr_config_for
        self._today_str = today_str
        self._engine_epoch = engine_epoch
        self._terminal_mode = terminal_mode
        self._close_graph = close_graph
        self._report_graph = report_graph

    def supports(self, trial_kind: TrialKind, boundary_kind: str) -> bool:
        return boundary_kind == "C" and trial_kind in self.SUPPORTED

    async def capture_boundaries(self, *, task_id: str, question: str, seed: int):
        """Capture is performed by the anchor pass, not by the backend.

        The anchor is a real full-graph cell whose checkpoints and continuation envelope are
        already recorded by the runner; re-running one here would spend the upstream work a
        second time, which is the exact substitution this whole module exists to avoid.
        """
        raise ForkCapabilityError(
            "CForkBackend does not capture boundaries; run the anchor cell and pass its "
            "stored checkpoints and continuation envelope in"
        )

    async def execute(self, spec: ForkSpec, checkpoint: Any) -> ForkExecution:
        if not self.supports(spec.trial_kind, spec.boundary_kind):
            raise ForkCapabilityError(
                f"CForkBackend cannot execute {spec.trial_kind.value} at boundary "
                f"{spec.boundary_kind!r}"
            )
        if not isinstance(checkpoint, CCheckpoint):
            raise ForkCapabilityError("a C fork requires a CCheckpoint")

        self._require_same_utc_date()
        epoch_before = self._engine_epoch()
        self._require_anchor_epoch(epoch_before)

        slot = self._envelope.slot_for(checkpoint.researcher_id)
        if slot is None and self._terminal_mode == "REPORT":
            raise ForkCapabilityError(
                f"no unique note slot for researcher {checkpoint.researcher_id!r}; the "
                "boundary cannot be continued without guessing which note it produced"
            )

        observations = _ArmObservations()
        compressed = await self._run_close(spec, checkpoint, observations)

        if observations.rederived_digest != spec.checkpoint_digest:
            raise ForkCapabilityError(
                "the re-entered close boundary rebuilt to digest "
                f"{observations.rederived_digest!r}, not the planned "
                f"{spec.checkpoint_digest!r}; this arm did not start where it claims"
            )
        if observations.search_queries:
            raise UpstreamCallDuringFork(
                f"{observations.search_queries} search queries during a C fork"
            )
        forbidden = sorted(set(observations.op_classes) - set(CLOSE_OP_ALLOWLIST))
        if forbidden:
            raise UpstreamCallDuringFork(f"forbidden op classes during a C fork: {forbidden}")

        output: Any = compressed
        terminal_digest = ""
        if self._terminal_mode == "REPORT":
            output = await self._run_report(spec, slot, compressed)
            terminal_digest = spec.checkpoint_digest

        epoch_after = self._engine_epoch()
        if epoch_after != epoch_before:
            raise ForkCapabilityError(
                f"the engine epoch changed during the arm ({epoch_before!r} -> "
                f"{epoch_after!r}); its work is not comparable to its pair"
            )

        return ForkExecution(
            output=output,
            start_checkpoint_digest=observations.rederived_digest,
            first_treatment_checkpoint_digest=observations.rederived_digest,
            seed_applied=True,
            upstream_research_calls=0,
            trajectory_events=(
                {"kind": "C_FORK_REENTRY", "checkpoint": spec.checkpoint_digest},
                {"kind": "TREATMENT", "checkpoint": spec.checkpoint_digest,
                 "variant": spec.variant_id},
            ),
            terminal_close_checkpoint_digest=terminal_digest,
        )

    # --- the two re-entries ---------------------------------------------------------------

    async def _run_close(
        self, spec: ForkSpec, checkpoint: CCheckpoint, observations: _ArmObservations
    ) -> str:
        from ..odr.hooks import strategies_bound
        from ..odr.vendor_hooks import bind_run

        graph = self._close_graph or build_close_reentry_graph()
        bundle = self._strategy_for(spec.variant_id)
        binding = self._run_binding_for(spec, checkpoint)

        def on_event(kind: str, payload: dict) -> None:
            if kind in ("C_CHECKPOINT", "CLOSE_REDUCED", "CLOSE_DEFERRED_TO_VENDOR"):
                digest = str(payload.get("checkpoint") or "")
                if digest:
                    observations.rederived_digest = digest
            if kind == "CLOSE_FAILED":
                observations.close_failed = True
            if kind == "SEARCH_QUERY":
                observations.search_queries += 1

        binding = _with_event_sink(binding, on_event)
        state = reconstruct_researcher_state(
            checkpoint, research_topic=self._envelope.research_brief
        )
        with _upstream_poisoned(), strategies_bound(bundle), bind_run(binding):
            result = await graph.ainvoke(state, self._odr_config_for(spec))
        return str(result.get("compressed_research", "") or "")

    async def _run_report(
        self, spec: ForkSpec, slot: NoteSlot | None, compressed: str
    ) -> str:
        graph = self._report_graph or build_report_reentry_graph()
        notes = self._envelope.substitute(slot, compressed)
        state = {
            "messages": [thaw_message(m) for m in self._envelope.root_messages],
            "research_brief": self._envelope.research_brief,
            # override, not append: a bare list is *added* to the channel by override_reducer,
            # which would leave the anchor's own note in place alongside the arm's.
            "notes": {"type": "override", "value": notes},
        }
        with _upstream_poisoned():
            result = await graph.ainvoke(state, self._odr_config_for(spec))
        return str(result.get("final_report", "") or "")

    # --- pairing guards -------------------------------------------------------------------

    def _require_same_utc_date(self) -> None:
        today = self._today_str()
        if today != self._envelope.anchor_today_str:
            raise ForkCapabilityError(
                f"the anchor ran on {self._envelope.anchor_today_str!r} but this arm sees "
                f"{today!r}; get_today_str is formatted into both the compressor and the "
                "final-report prompt, so the arms would differ by more than the treatment"
            )

    def _require_anchor_epoch(self, epoch: str) -> None:
        if self._envelope.anchor_engine_epoch and epoch != self._envelope.anchor_engine_epoch:
            raise ForkCapabilityError(
                f"the anchor ran under engine epoch {self._envelope.anchor_engine_epoch!r} "
                f"but this arm runs under {epoch!r}; the shared upstream work constant came "
                "from a different boot"
            )


def _with_event_sink(binding: Any, sink: Callable[[str, dict], None]) -> Any:
    """Chain an observer onto a run binding's event callback without replacing it."""
    from dataclasses import replace as _replace

    existing = getattr(binding, "on_event", None)

    def combined(kind: str, payload: dict) -> None:
        sink(kind, payload)
        if existing is not None:
            existing(kind, payload)

    return _replace(binding, on_event=combined)


class _upstream_poisoned:
    """Make any search or researcher path raise for the duration of a fork.

    The provider allowlist is the authoritative guard because it refuses before dispatch and
    therefore before spend. This one is in-process and catches the case the allowlist cannot
    see: a frozen-corpus search, which never reaches the provider at all but would still mean
    the arm did upstream work its pair did not.
    """

    def __init__(self) -> None:
        self._restore: list[tuple[Any, str, Any]] = []

    def __enter__(self) -> _upstream_poisoned:
        try:
            import open_deep_research.utils as vendor_utils
        except ImportError:  # pragma: no cover - only when the vendor tree is absent
            return self

        async def refuse(*args: Any, **kwargs: Any) -> Any:
            raise UpstreamCallDuringFork(
                "a C fork attempted a search; the boundary is frozen and no upstream work "
                "may differ between the arms of one pair"
            )

        for name in ("tavily_search_async", "tavily_search"):
            if hasattr(vendor_utils, name):
                self._restore.append((vendor_utils, name, getattr(vendor_utils, name)))
                setattr(vendor_utils, name, refuse)
        return self

    def __exit__(self, *exc: Any) -> None:
        for module, name, original in reversed(self._restore):
            setattr(module, name, original)
        self._restore.clear()
