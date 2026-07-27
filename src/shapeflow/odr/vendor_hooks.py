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
from dataclasses import dataclass, replace
from typing import Any, Callable, Optional, Sequence

from ..canonical import canonical_json
from ..hashing import sha256_hex
from .adapter import DeferredPageBatch, build_h_checkpoint, is_deferred, reduce_tool_batch
from .checkpoints import SamplingEnvelope
from .hooks import current_strategies


class _UseVendorCompression(Exception):
    """A close strategy declining, so vendor's own compression runs."""


class ComponentTrialFailure(RuntimeError):
    """A component treatment failed before publishing its atomic boundary output.

    Raised only from the whole-batch/whole-close hook, outside vendor's per-tool exception
    handlers. It terminates the cell so a deferred Python object or a vendor fallback can never
    be recorded as a successful P1 component observation.
    """

    def __init__(self, *, node: str, checkpoint: str, reason: str, detail: str = "") -> None:
        super().__init__(f"{node} component failed at {checkpoint}: {reason}: {detail}")
        self.node = node
        self.checkpoint = checkpoint
        self.reason = reason
        self.detail = detail


__all__ = [
    "ComponentTrialFailure",
    "RunBinding",
    "bind_run",
    "current_run",
    "child_researcher_binding_active",
    "invoke_child_researcher",
    "refine_child_research_tasks",
    "current_published_tool_provenance",
    "page_hook_active",
    "defer_page_batch",
    "reduce_published_batch",
    "record_tool_batch_publication",
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
    # ``None`` is the direct researcher-subgraph/root coordinate. ConductResearch children
    # receive (supervisor research iteration, allowed-call ordinal, exact tool-call id).
    researcher_coordinate: Optional[tuple[int, int, str]] = None


_RUN: contextvars.ContextVar[Optional[RunBinding]] = contextvars.ContextVar(
    "shapeflow_run_binding", default=None
)

# Written by the whole-batch reducer and consumed only after the patched graph has successfully
# constructed every ToolMessage in the batch.  ContextVar isolation matters here: the supervisor
# can run multiple researchers concurrently, and a process-global "pending publish" slot would
# let one researcher's Command certify another researcher's batch.
_PENDING_PUBLICATION: contextvars.ContextVar[Optional[dict]] = contextvars.ContextVar(
    "shapeflow_pending_publication", default=None
)

# Committed only at record_tool_batch_publication, after the exact graph-visible batch has been
# checked. Values include the content/name identity as well as occurrence ids so a reused
# tool_call_id cannot silently attach old provenance to different bytes.
_PUBLISHED_TOOL_PROVENANCE: contextvars.ContextVar[
    Optional[dict[str, dict]]
] = contextvars.ContextVar("shapeflow_published_tool_provenance", default=None)

# Parent-run registry of ConductResearch structural slots. Each nested child gets a fresh dict.
# Within one graph invocation, LangGraph copies ContextVars between node tasks; retaining the
# same dict object is intentional so later supervisor nodes observe earlier structural slots.
_CHILD_COORDINATES: contextvars.ContextVar[
    Optional[dict[tuple[str, int, int], str]]
] = contextvars.ContextVar("shapeflow_child_coordinates", default=None)


def bind_run(binding: Optional[RunBinding]):
    """Bind one run/child and all of its sidecars, resetting every slot in ``finally``."""
    import contextlib

    @contextlib.contextmanager
    def _bound():
        run_token = _RUN.set(binding)
        pending_token = _PENDING_PUBLICATION.set(None)
        provenance_token = _PUBLISHED_TOOL_PROVENANCE.set({})
        child_token = _CHILD_COORDINATES.set({})
        try:
            yield binding
        finally:
            # Reverse acquisition order. A cancellation or strategy exception cannot leak a
            # pending batch, a prior child's provenance, or a structural coordinate into the
            # next arm/researcher that reuses this asyncio context.
            _CHILD_COORDINATES.reset(child_token)
            _PUBLISHED_TOOL_PROVENANCE.reset(provenance_token)
            _PENDING_PUBLICATION.reset(pending_token)
            _RUN.reset(run_token)

    return _bound()


def current_run() -> Optional[RunBinding]:
    return _RUN.get()


def current_published_tool_provenance() -> dict[str, dict]:
    """A defensive copy of this run/child's committed publication sidecar."""
    return {
        str(call_id): {
            **entry,
            "source_occurrence_ids": tuple(
                entry.get("source_occurrence_ids") or ()
            ),
        }
        for call_id, entry in (_PUBLISHED_TOOL_PROVENANCE.get() or {}).items()
    }


def child_researcher_binding_active() -> bool:
    """Whether ConductResearch must refine the cell-level binding for a child."""
    return current_run() is not None


def invoke_child_researcher(
    *,
    researcher_subgraph: Any,
    input_state: dict,
    config: Any,
    supervisor_research_iteration: int,
    allowed_tool_call_ordinal: int,
    tool_call_id: str,
):
    """Return one child invocation coroutine under a collision-free nested run binding.

    This is intentionally a regular function: the patched supervisor calls it while building
    the ordered task list, so structural-slot conflicts are checked deterministically before
    ``asyncio.gather`` starts the children. The returned coroutine performs the nested
    ``bind_run`` and resets it in ``finally``.
    """
    parent = current_run()
    if parent is None:
        raise RuntimeError("child researcher binding requested without a parent run")
    iteration = int(supervisor_research_iteration)
    ordinal = int(allowed_tool_call_ordinal)
    call_id = str(tool_call_id or "")
    if iteration < 0 or ordinal < 0 or not call_id:
        raise RuntimeError(
            "ConductResearch child coordinate requires non-negative iteration/ordinal and "
            "a non-empty tool-call id"
        )
    registry = _CHILD_COORDINATES.get()
    if registry is None:
        raise RuntimeError("child coordinate registry is not bound to this run")
    slot = (parent.researcher_id, iteration, ordinal)
    previous = registry.get(slot)
    if previous is not None and previous != call_id:
        raise RuntimeError(
            "ConductResearch structural slot was reused by a different tool-call id: "
            f"{slot!r} held {previous!r}, now {call_id!r}"
        )
    registry[slot] = call_id

    coordinate = (iteration, ordinal, call_id)
    child = replace(
        parent,
        researcher_id=(
            f"{parent.researcher_id}/child-"
            f"{iteration}-{ordinal}-{sha256_hex(call_id.encode('utf-8'))[:16]}"
        ),
        researcher_coordinate=coordinate,
    )

    async def _invoke():
        with bind_run(child):
            return await researcher_subgraph.ainvoke(input_state, config)

    return _invoke()


def refine_child_research_tasks(
    *,
    original_tasks: Sequence[Any],
    researcher_subgraph: Any,
    allowed_tool_calls: Sequence[dict],
    config: Any,
    supervisor_research_iteration: int,
) -> list[Any]:
    """Replace vendor-created, unstarted coroutines with structurally bound child coroutines.

    The patch leaves vendor's hooks-off comprehension byte-for-byte in place. In a bound run,
    this function closes those unawaited coroutines before constructing replacements; no child
    graph byte executes under the coarse cell-level binding.
    """
    from langchain_core.messages import HumanMessage

    def discard(awaitable: Any) -> None:
        close = getattr(awaitable, "close", None)
        if callable(close):
            close()
            return
        cancel = getattr(awaitable, "cancel", None)
        if callable(cancel):
            cancel()

    for awaitable in original_tasks:
        discard(awaitable)

    refined: list[Any] = []
    try:
        for ordinal, tool_call in enumerate(allowed_tool_calls):
            refined.append(invoke_child_researcher(
                researcher_subgraph=researcher_subgraph,
                input_state={
                    "researcher_messages": [
                        HumanMessage(content=tool_call["args"]["research_topic"])
                    ],
                    "research_topic": tool_call["args"]["research_topic"],
                },
                config=config,
                supervisor_research_iteration=supervisor_research_iteration,
                allowed_tool_call_ordinal=ordinal,
                tool_call_id=tool_call["id"],
            ))
    except BaseException:
        for awaitable in refined:
            discard(awaitable)
        raise
    return refined


def _emit(kind: str, payload: dict) -> None:
    run = current_run()
    if run is not None and run.on_event is not None:
        run.on_event(kind, payload)


def _failure_document(failure: Any) -> Optional[dict]:
    if failure is None:
        return None
    return {
        "reason": str(getattr(failure, "reason", type(failure).__name__)),
        "detail": str(getattr(failure, "detail", "") or ""),
    }


def _emit_selection_records(
    strategy: Any,
    *,
    node: str,
    checkpoint: str,
    fell_back: bool = False,
    batch_failure: Any = None,
) -> None:
    """Publish direct reducer facts instead of reconstructing them from final prose.

    Page strategies expose one outcome per local/global selector stage; close strategies expose
    one outcome.  A batch-level integrity failure can occur after all selector stages succeeded
    (for example, a missing sibling observation), so it is attached to each affected record and
    a synthetic record is emitted when the strategy produced no stage outcome at all.
    """
    if node == "H":
        outcomes = list(getattr(strategy, "last_outcomes", ()) or ())
    else:
        value = getattr(strategy, "last_outcome", None)
        outcomes = [value] if value is not None else []

    failure_doc = _failure_document(batch_failure)
    rendered_token_counts = tuple(
        int(value)
        for value in (getattr(strategy, "last_rendered_token_counts", ()) or ())
    )
    published_token_counts = tuple(
        int(value)
        for value in (getattr(strategy, "last_published_token_counts", ()) or ())
    )
    control_records = list(
        getattr(strategy, "last_control_records", ()) or ()
    )
    if not outcomes and (
        control_records or (failure_doc is None and rendered_token_counts)
    ):
        _emit("PROSE_CONTROL_OUTPUT", {
            "checkpoint": checkpoint,
            "node": node,
            "rendered_token_counts": list(rendered_token_counts),
            "max_rendered_tokens": max(rendered_token_counts, default=0),
            "total_rendered_tokens": sum(rendered_token_counts),
            # H's selector/materializer budget applies before the common outer
            # ``Selected evidence`` tool-message wrapper.  Keep both quantities so the
            # mechanism control is checked against the same inner budget while the actual
            # downstream context bytes remain auditable.
            "published_token_counts": list(published_token_counts),
            "max_published_tokens": max(published_token_counts, default=0),
            "total_published_tokens": sum(published_token_counts),
            "control_records": control_records,
            "fell_back": bool(fell_back),
            "batch_failure": failure_doc,
        })
        return
    if not outcomes and failure_doc is not None:
        _emit("NODE_SELECTION", {
            "checkpoint": checkpoint,
            "direct_node_record": {
                "node": node,
                "checkpoint_hash": checkpoint,
                "offered_span_ids": [],
                "offered_source_occurrence_ids": [],
                "selected_span_ids": [],
                "published_span_ids": [],
                "selector_attempted": False,
                "normalization": None,
                "fell_back": fell_back,
                "failure": failure_doc,
                "contract": "",
                "aggregation": "",
                "stage": "batch",
            },
        })
        return

    for selected in outcomes:
        own_failure = _failure_document(getattr(selected, "failure", None))
        accepted = not fell_back and batch_failure is None and own_failure is None
        staged_span_ids = list(
            getattr(selected, "staged_span_ids", ()) or
            getattr(selected, "published_span_ids", ()) or ()
        )
        staged_relations = list(
            getattr(selected, "staged_relations", ()) or
            getattr(selected, "published_relations", ()) or ()
        )
        staged_gaps = list(
            getattr(selected, "staged_gaps", ()) or
            getattr(selected, "published_gaps", ()) or ()
        )
        staged_query_attempt_ids = list(
            getattr(selected, "staged_query_attempt_ids", ()) or
            getattr(selected, "published_query_attempt_ids", ()) or ()
        )
        _emit("NODE_SELECTION", {
            "checkpoint": checkpoint,
            "direct_node_record": {
                "node": node,
                "checkpoint_hash": str(
                    getattr(selected, "checkpoint_hash", "") or checkpoint
                ),
                "candidate_view_sha256": str(
                    getattr(selected, "view_sha256", "") or ""
                ),
                "offered_span_ids": list(
                    getattr(selected, "offered_span_ids", ()) or ()
                ),
                "offered_span_token_counts": [
                    [str(span_id), int(token_count)]
                    for span_id, token_count in (
                        getattr(selected, "offered_span_token_counts", ()) or ()
                    )
                ],
                "publication_handle_map": [
                    [str(handle), str(span_id)]
                    for handle, span_id in (
                        getattr(selected, "publication_handle_map", ()) or ()
                    )
                ],
                "publication_handle_token_counts": [
                    [str(handle), int(token_count)]
                    for handle, token_count in (
                        getattr(
                            selected,
                            "publication_handle_token_counts",
                            (),
                        ) or ()
                    )
                ],
                "publication_map_sha256": str(
                    getattr(selected, "publication_map_sha256", "") or ""
                ),
                "offered_material_tokens": int(
                    getattr(
                        selected,
                        "offered_material_tokens",
                        getattr(selected, "offered_evidence_tokens", 0),
                    ) or 0
                ),
                "offered_evidence_tokens": int(
                    getattr(selected, "offered_evidence_tokens", 0) or 0
                ),
                "offered_context_tokens": int(
                    getattr(selected, "offered_context_tokens", 0) or 0
                ),
                "staged_rendered_tokens": int(
                    getattr(selected, "staged_rendered_tokens", 0) or 0
                ),
                "published_rendered_tokens": (
                    int(getattr(selected, "published_rendered_tokens", 0) or 0)
                    if accepted else 0
                ),
                "offered_source_occurrence_ids": list(
                    getattr(selected, "offered_source_occurrence_ids", ()) or ()
                ),
                "selected_span_ids": list(
                    getattr(selected, "selected_span_ids", ()) or ()
                ),
                "staged_span_ids": staged_span_ids,
                # "Published" means it survived BOTH strategy preflight and whole-batch
                # acceptance.  On fallback the P1 bytes were staged then discarded; keeping
                # their IDs here would credit an output the graph never saw.
                "published_span_ids": list(
                    getattr(selected, "published_span_ids", ()) or ()
                ) if accepted else [],
                "selected_relations": list(
                    getattr(selected, "selected_relations", ()) or ()
                ),
                "staged_relations": staged_relations,
                "published_relations": list(
                    getattr(selected, "published_relations", ()) or ()
                ) if accepted else [],
                "selected_gaps": list(
                    getattr(selected, "selected_gaps", ()) or ()
                ),
                "staged_gaps": staged_gaps,
                "published_gaps": list(
                    getattr(selected, "published_gaps", ()) or ()
                ) if accepted else [],
                "offered_query_attempt_ids": list(
                    getattr(selected, "offered_query_attempt_ids", ()) or ()
                ),
                "selected_query_attempt_ids": list(
                    getattr(selected, "selected_query_attempt_ids", ()) or ()
                ),
                "staged_query_attempt_ids": staged_query_attempt_ids,
                "published_query_attempt_ids": list(
                    getattr(selected, "published_query_attempt_ids", ()) or ()
                ) if accepted else [],
                # Keep parser validity independent of publication. A bad selector completion
                # that fell back to excellent P0 prose is still an invalid P1 attempt.
                "selector_attempted": bool(
                    getattr(selected, "selector_attempted", False)
                ),
                "normalization": getattr(selected, "normalization", None),
                "fell_back": bool(fell_back),
                "failure": own_failure or failure_doc,
                "contract": str(getattr(selected, "contract", "") or ""),
                "aggregation": str(getattr(selected, "aggregation", "") or ""),
                "chunker": str(getattr(selected, "chunker", "") or ""),
                "tokenizer_sha256": str(
                    getattr(selected, "tokenizer_sha256", "") or ""
                ),
                "stage": str(getattr(selected, "stage", "") or ""),
            },
        })


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
        researcher_coordinate=run.researcher_coordinate,
    )
    # The P0 strategy replays vendor's own closures, so it needs the deferred objects rather
    # than only the checkpoint's hashes. Handing them over here keeps the strategy protocol
    # free of a vendor-specific parameter.
    page = bundle.page
    if hasattr(page, "_deferred"):
        page._deferred = {
            tool_calls[i]["id"]: observations[i]
            for i in range(len(observations)) if is_deferred(observations[i])
        }

    outcome = await reduce_tool_batch(
        strategy=page,
        task_ctx=run.task_ctx,
        checkpoint=checkpoint,
        observations=observations,
        tool_calls=tool_calls,
        component_trial=run.component_trial,
        store_checkpoint=run.store_checkpoint,
    )
    _emit_selection_records(
        page,
        node="H",
        checkpoint=outcome.checkpoint_digest,
        fell_back=outcome.fell_back,
        batch_failure=outcome.failure,
    )
    _emit("PAGE_BATCH_REDUCED", {
        "checkpoint": outcome.checkpoint_digest,
        "fell_back": outcome.fell_back,
        "failure": outcome.failure.reason if outcome.failure else None,
        "detail": outcome.failure.detail if outcome.failure else None,
        "siblings": len(observations),
    })
    if outcome.failure is not None and not outcome.fell_back:
        # We are after vendor's asyncio.gather, not inside a per-tool handler. Raising here aborts
        # the whole cell before ToolMessage construction, which is the only honest component
        # outcome. Returning `observations` would publish DeferredPageBatch.__repr__ strings.
        raise ComponentTrialFailure(
            node="H",
            checkpoint=outcome.checkpoint_digest,
            reason=outcome.failure.reason,
            detail=outcome.failure.detail,
        )
    # This is not yet a publication event.  It becomes one only after the patched graph has
    # constructed every ToolMessage and calls record_tool_batch_publication immediately before
    # returning its single Command.
    _PENDING_PUBLICATION.set({
        "checkpoint": outcome.checkpoint_digest,
        "sibling_count": len(observations),
        "tool_call_ids": [str(call.get("id") or "") for call in tool_calls],
        "observation_sha256s": [
            sha256_hex(canonical_json(value)) for value in outcome.observations
        ],
        "publication_provenance": {
            str(call_id): list(occurrence_ids)
            for call_id, occurrence_ids in outcome.publication_provenance
        },
        "fell_back": bool(outcome.fell_back),
    })
    return outcome.observations


def record_tool_batch_publication(
    *, tool_calls: Sequence[dict], tool_outputs: Sequence[Any]
) -> None:
    """Certify the exact full batch at the graph's single publication point.

    Calling this before ToolMessage construction would only prove that the reducer returned a
    list.  The patch calls it after construction and immediately before one of the two native
    ``Command(update={"researcher_messages": tool_outputs})`` returns.  A missing sibling,
    changed order, or second/partial call fails closed and emits no success event.
    """
    pending = _PENDING_PUBLICATION.get()
    if pending is None:
        return
    _PENDING_PUBLICATION.set(None)
    call_ids = [str(call.get("id") or "") for call in tool_calls]
    call_names = [str(call.get("name") or "") for call in tool_calls]
    output_ids = [str(getattr(output, "tool_call_id", "") or "") for output in tool_outputs]
    output_names = [str(getattr(output, "name", None) or "") for output in tool_outputs]
    output_content_sha256s = [
        sha256_hex(canonical_json(getattr(output, "content", None)))
        for output in tool_outputs
    ]
    if (
        len(tool_outputs) != pending["sibling_count"]
        or call_ids != pending["tool_call_ids"]
        or output_ids != call_ids
        or output_names != call_names
        or output_content_sha256s != pending["observation_sha256s"]
        or len(set(call_ids)) != len(call_ids)
    ):
        _emit("TOOL_BATCH_PUBLICATION_REJECTED", {
            "checkpoint": pending["checkpoint"],
            "expected_sibling_count": pending["sibling_count"],
            "actual_sibling_count": len(tool_outputs),
            "expected_tool_call_ids": pending["tool_call_ids"],
            "actual_tool_call_ids": output_ids,
            "expected_tool_names": call_names,
            "actual_tool_names": output_names,
            "expected_content_sha256s": pending["observation_sha256s"],
            "actual_content_sha256s": output_content_sha256s,
        })
        raise RuntimeError(
            "atomic publication integrity failed: ToolMessage batch differs from reducer batch"
        )

    provenance_by_call = {
        str(call_id): tuple(dict.fromkeys(
            str(value) for value in occurrence_ids if str(value)
        ))
        for call_id, occurrence_ids in (
            pending.get("publication_provenance") or {}
        ).items()
    }
    if any(call_id not in call_ids for call_id in provenance_by_call):
        raise RuntimeError(
            "atomic publication integrity failed: provenance names a call outside the batch"
        )
    committed = _PUBLISHED_TOOL_PROVENANCE.get()
    if committed is None:
        raise RuntimeError(
            "atomic publication integrity failed: no run-scoped provenance sidecar is bound"
        )
    staged_entries = {
        call_id: {
            "name": output_names[index],
            "content_sha256": output_content_sha256s[index],
            "source_occurrence_ids": provenance_by_call.get(call_id, ()),
            "checkpoint": pending["checkpoint"],
        }
        for index, call_id in enumerate(call_ids)
    }
    conflicts = {
        call_id: (committed[call_id], entry)
        for call_id, entry in staged_entries.items()
        if call_id in committed and committed[call_id] != entry
    }
    if conflicts:
        _emit("TOOL_PROVENANCE_CONFLICT", {
            "checkpoint": pending["checkpoint"],
            "tool_call_ids": sorted(conflicts),
        })
        raise RuntimeError(
            "published tool_call_id was reused with different content, name, checkpoint, "
            "or source provenance"
        )
    # LangGraph executes the close node in a copied Context. Mutating this run-local object is
    # what makes the already-bound sidecar visible there; ContextVar.set() here would update only
    # the researcher_tools task and the C checkpoint would see an empty mapping.
    committed.update(staged_entries)

    _emit("TOOL_BATCH_PUBLISHED", {
        **pending,
        "atomic_publish": True,
        "tool_output_sha256s": [
            sha256_hex(canonical_json({
                "content": getattr(output, "content", None),
                "name": getattr(output, "name", None),
                "tool_call_id": getattr(output, "tool_call_id", None),
            }))
            for output in tool_outputs
        ],
        "published_source_occurrence_ids_by_call": {
            call_id: list(staged_entries[call_id]["source_occurrence_ids"])
            for call_id in call_ids
        },
        "publish_calls": 1,
    })


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
    try:
        frozen = freeze_messages(
            researcher_messages,
            tool_provenance=current_published_tool_provenance(),
        )
    except ValueError as exc:
        # This is an integrity failure, not a selector failure. Falling through to vendor would
        # turn a mismatched/reused provenance sidecar into a successful arm while C_VISIBLE's
        # evidence boundary was no longer knowable.
        _emit("CLOSE_PROVENANCE_REJECTED", {
            "close_reason": close_reason,
            "detail": str(exc),
        })
        raise RuntimeError(
            f"capture-time tool provenance does not match the close messages: {exc}"
        ) from exc
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
    except _UseVendorCompression:
        # The explicit-P0 close strategy declines rather than re-implementing
        # compress_research: that function has retry-on-token-limit, an in-place append and a
        # specific raw_notes construction, and a second implementation would drift into
        # looking like a P1 effect.
        _emit("CLOSE_DEFERRED_TO_VENDOR", {"checkpoint": checkpoint.digest})
        return None
    except Exception as e:  # noqa: BLE001
        failure = getattr(e, "failure", None)
        failure_reason = str(
            getattr(failure, "reason", None) or type(e).__name__
        )
        _emit_selection_records(
            bundle.close,
            node="C",
            checkpoint=checkpoint.digest,
            fell_back=not run.component_trial,
            batch_failure=failure or e,
        )
        _emit("CLOSE_FAILED", {"checkpoint": checkpoint.digest,
                               "close_reason": close_reason,
                               "reason": failure_reason,
                               "detail": f"{type(e).__name__}: {e}"})
        if run.component_trial:
            # Component trials estimate the close reducer itself. Falling through to vendor
            # compression would turn a failed C treatment into an apparent successful P0 result.
            raise ComponentTrialFailure(
                node="C",
                checkpoint=checkpoint.digest,
                reason="STRATEGY_ERROR",
                detail=f"{type(e).__name__}: {e}",
            ) from e
        return None              # end-to-end: fall back to vendor compression
    _emit_selection_records(
        bundle.close,
        node="C",
        checkpoint=checkpoint.digest,
    )
    _emit("CLOSE_REDUCED", {"checkpoint": checkpoint.digest, "close_reason": close_reason})
    return {
        "compressed_research": handoff.compressed_research,
        "raw_notes": list(handoff.raw_notes),
    }
