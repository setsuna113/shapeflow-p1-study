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
import contextvars
import os
import re
import threading
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Sequence

from ..acquire.frozen_search import FrozenTaskCorpusBackend
from ..acquire.snapshot_store import SnapshotStore
from ..acquire.source_pool import SourcePool
from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..odr.checkpoints import SamplingEnvelope
from ..odr.hooks import StrategyBundle, TaskContext, strategies_bound
from ..odr.vendor_hooks import RunBinding, bind_run
from ..p1.contracts import P1_CONTRACTS, canonical_normalization_document
from .settings import Settings

__all__ = [
    "CellSpec",
    "CellResult",
    "SeedDeliveryError",
    "TrajectoryRecorder",
    "first_divergence",
    "frozen_search_async",
    "install_frozen_search",
    "odr_config",
    "run_cell",
]


class SeedDeliveryError(RuntimeError):
    """A seed was recorded as paired but could not be attached to every ODR request."""


_ACTIVE_ODR_SEED: contextvars.ContextVar[Optional[int]] = contextvars.ContextVar(
    "shapeflow_active_odr_seed", default=None
)
_SEED_PATCH_LOCK = threading.Lock()
_SEED_PATCH_STATE: dict[str, Optional[int]] = {"active": None}


def _selector_normalization_trace(record: dict) -> dict:
    """Canonicalize selector validity without allowing missing telemetry to look clean."""
    raw = record.get("normalization")
    has_explicit_attempt = "selector_attempted" in record
    attempt_value = record.get("selector_attempted")
    attempt_error = ""
    if has_explicit_attempt and type(attempt_value) is not bool:
        attempt_error = "selector_attempted must be a boolean"
        attempted = bool(
            raw is not None
            or record.get("candidate_view_sha256")
            or (
                str(record.get("contract") or "") in P1_CONTRACTS
                and record.get("offered_span_ids")
            )
        )
    elif has_explicit_attempt:
        attempted = attempt_value
    else:
        # Compatibility inference is deliberately conservative. New production records always
        # carry the explicit bit; an older P1 record with an offered view is an attempt whose
        # absent normalization must be MISSING, never silently strict-valid.
        attempted = bool(
            raw is not None
            or record.get("candidate_view_sha256")
            or (
                str(record.get("contract") or "") in P1_CONTRACTS
                and record.get("offered_span_ids")
            )
        )

    provenance = "EXPLICIT" if has_explicit_attempt else "INFERRED_LEGACY"
    if attempt_error:
        return {
            "selector_attempted": attempted,
            "selector_attempt_provenance": provenance,
            "normalization": None,
            "normalization_trace_status": "INVALID",
            "normalization_trace_error": attempt_error,
        }
    if not attempted:
        if raw is not None:
            return {
                "selector_attempted": False,
                "selector_attempt_provenance": provenance,
                "normalization": None,
                "normalization_trace_status": "INVALID",
                "normalization_trace_error":
                    "normalization was present for a non-attempted selector",
            }
        return {
            "selector_attempted": False,
            "selector_attempt_provenance": provenance,
            "normalization": None,
            "normalization_trace_status": "NOT_APPLICABLE",
            "normalization_trace_error": "",
        }
    if raw is None:
        return {
            "selector_attempted": True,
            "selector_attempt_provenance": provenance,
            "normalization": None,
            "normalization_trace_status": "MISSING",
            "normalization_trace_error":
                "selector attempt has no normalization trace",
        }
    try:
        normalized = canonical_normalization_document(raw)
    except ValueError as exc:
        return {
            "selector_attempted": True,
            "selector_attempt_provenance": provenance,
            "normalization": None,
            "normalization_trace_status": "INVALID",
            "normalization_trace_error": str(exc),
        }
    return {
        "selector_attempted": True,
        "selector_attempt_provenance": provenance,
        "normalization": normalized,
        "normalization_trace_status": "OK",
        "normalization_trace_error": "",
    }


def _materialization_trace(record: dict) -> dict:
    """Copy the renderer-token trace without coercing malformed values into validity.

    The hook is the producer and :func:`score_direct_node_records` is the independent
    consumer.  The recorder used to whitelist an older field set and silently discarded the
    four token measurements in between, so every live run reached evaluation as
    ``DIRECT_TRACE_UNAVAILABLE`` despite a correct hook event.  This boundary now preserves
    the exact values and records a fail-closed validation status; it never reconstructs them
    from a summary event.
    """
    token_fields = (
        "offered_span_token_counts",
        "offered_evidence_tokens",
        "staged_rendered_tokens",
        "published_rendered_tokens",
    )
    semantic_token_fields = (
        "offered_material_tokens",
        "offered_context_tokens",
    )
    present = [field for field in token_fields if field in record]
    if not present:
        return {
            "materialization_trace_status": "MISSING",
            "materialization_trace_errors": [
                "direct node record has no materialization token trace"
            ],
        }

    copied = {field: record[field] for field in present}
    copied.update({
        field: record[field] for field in semantic_token_fields if field in record
    })
    errors: list[str] = []
    if len(present) != len(token_fields):
        errors.append(
            "partial materialization token trace: missing "
            + ", ".join(field for field in token_fields if field not in record)
        )

    counts = record.get("offered_span_token_counts")
    normalized_counts = counts
    seen: set[str] = set()
    total = 0
    if "offered_span_token_counts" in record:
        if not isinstance(counts, (list, tuple)):
            errors.append("offered_span_token_counts is not a sequence")
        else:
            normalized_counts = []
            for item in counts:
                if (
                    not isinstance(item, (list, tuple))
                    or len(item) != 2
                    or not isinstance(item[0], str)
                    or not item[0]
                    or isinstance(item[1], bool)
                    or not isinstance(item[1], int)
                    or item[1] < 0
                    or item[0] in seen
                ):
                    errors.append(
                        "offered_span_token_counts has an invalid or duplicate entry"
                    )
                    # Preserve the malformed value for the evaluator rather than repairing it.
                    normalized_counts.append(item)
                    continue
                seen.add(item[0])
                total += item[1]
                normalized_counts.append([item[0], item[1]])
            copied["offered_span_token_counts"] = normalized_counts
            offered = list(record.get("offered_span_ids") or ())
            if len(offered) != len(set(offered)) or seen != set(map(str, offered)):
                errors.append(
                    "offered_span_token_counts does not cover exactly the offered span ids"
                )

    scalar_values: dict[str, int] = {}
    for field in (*token_fields[1:], *semantic_token_fields):
        if field not in record:
            continue
        value = record[field]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            errors.append(f"{field} is not a non-negative integer")
        else:
            scalar_values[field] = value
    material = scalar_values.get("offered_material_tokens")
    evidence = scalar_values.get("offered_evidence_tokens")
    context = scalar_values.get("offered_context_tokens")
    if isinstance(counts, (list, tuple)):
        if material is not None:
            if material != total:
                errors.append(
                    "offered_material_tokens does not equal the per-span sum"
                )
            if evidence is not None and context is not None and evidence + context != material:
                errors.append(
                    "offered evidence plus context tokens do not equal offered material"
                )
        elif evidence is not None and evidence != total:
            # Backward compatibility for pre-origin-semantics traces, where this wire field
            # meant all offered material.  New traces always carry offered_material_tokens.
            errors.append("offered_evidence_tokens does not equal the per-span sum")
    staged = scalar_values.get("staged_rendered_tokens")
    published = scalar_values.get("published_rendered_tokens")
    if staged is not None and published is not None and published > staged:
        errors.append("published_rendered_tokens exceeds staged_rendered_tokens")
    if (
        published is not None
        and (record.get("fell_back") or record.get("failure"))
        and published != 0
    ):
        errors.append("failed/fallback record claims published rendered tokens")
    if (
        published is not None
        and record.get("published_span_ids")
        and not record.get("fell_back")
        and not record.get("failure")
        and published == 0
    ):
        errors.append("published evidence has no rendered-token count")

    tokenizer = record.get("tokenizer_sha256")
    if (
        not isinstance(tokenizer, str)
        or len(tokenizer) != 64
        or any(char not in "0123456789abcdef" for char in tokenizer.lower())
    ):
        errors.append("tokenizer_sha256 is not a 64-character hex digest")
    copied["tokenizer_sha256"] = tokenizer
    copied["materialization_trace_status"] = (
        "OK" if not errors else ("PARTIAL" if len(present) < len(token_fields) else "INVALID")
    )
    copied["materialization_trace_errors"] = errors
    return copied


def _publication_handle_trace(record: dict) -> dict:
    """Closed-validate the exact publication handle mapping emitted by the strategy."""
    raw_mapping = record.get("publication_handle_map")
    raw_counts = record.get("publication_handle_token_counts")
    recorded_sha = str(record.get("publication_map_sha256") or "")
    if raw_mapping is None and raw_counts is None and not recorded_sha:
        return {
            "publication_handle_map": [],
            "publication_handle_token_counts": [],
            "publication_map_sha256": "",
            "publication_handle_trace_status": "MISSING",
            "publication_handle_trace_errors": [
                "direct node record has no publication handle mapping"
            ],
        }

    errors: list[str] = []
    mapping: list[list] = []
    counts: list[list] = []
    if not isinstance(raw_mapping, (list, tuple)):
        errors.append("publication_handle_map is not a sequence")
    else:
        for item in raw_mapping:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not all(isinstance(value, str) and value for value in item)
            ):
                errors.append("publication_handle_map has an invalid entry")
                continue
            mapping.append([str(item[0]), str(item[1])])
    if not isinstance(raw_counts, (list, tuple)):
        errors.append("publication_handle_token_counts is not a sequence")
    else:
        for item in raw_counts:
            if (
                not isinstance(item, (list, tuple))
                or len(item) != 2
                or not isinstance(item[0], str)
                or not item[0]
                or isinstance(item[1], bool)
                or not isinstance(item[1], int)
                or item[1] < 0
            ):
                errors.append(
                    "publication_handle_token_counts has an invalid entry"
                )
                continue
            counts.append([str(item[0]), int(item[1])])

    handles = [item[0] for item in mapping]
    span_ids = [item[1] for item in mapping]
    count_handles = [item[0] for item in counts]
    if len(handles) != len(set(handles)):
        errors.append("publication handles are not unique")
    if len(span_ids) != len(set(span_ids)):
        errors.append("one offered span has multiple publication handles")
    if handles != count_handles:
        errors.append("publication handle token counts do not cover mapping order exactly")
    offered = list(map(str, record.get("offered_span_ids") or ()))
    if span_ids != offered:
        errors.append("publication handle mapping does not cover offered order exactly")
    if str(record.get("node") or "").upper().startswith("H"):
        # One grammar, owned by the codec the producer encodes with. A regex written here
        # instead is how the validator came to accept three coordinate groups while the
        # producer emitted four.
        from ..p1 import handle_codec

        invalid_handles = [
            handle for handle in handles if not handle_codec.validate(handle)
        ]
        if invalid_handles:
            errors.append("H publication handles are not structural ordinals")
        if any(
            token_count > handle_codec.MAX_PUBLICATION_HANDLE_TOKENS
            for _, token_count in counts
        ):
            errors.append("H publication handle exceeds exact-token cap")
    actual_sha = sha256_hex(canonical_json({
        "handle_to_span": mapping,
        "handle_token_counts": counts,
    }))
    if recorded_sha != actual_sha:
        errors.append("publication_map_sha256 does not bind the recorded mapping")
    return {
        "publication_handle_map": mapping,
        "publication_handle_token_counts": counts,
        "publication_map_sha256": recorded_sha,
        "publication_handle_trace_status": "OK" if not errors else "INVALID",
        "publication_handle_trace_errors": errors,
    }


@dataclass
class TrajectoryRecorder:
    """Ordered, hashable trajectory telemetry relative to the first treatment.

    H's downstream searches and close behaviour are outcomes, not parity failures.  The
    recorder therefore labels the first actual H/C intervention and keeps everything after
    it.  A query that happened before the first H reducer stays in ``PRE_TREATMENT``; a query
    triggered by its published context is ``POST_TREATMENT``.
    """

    treatment_node: Optional[str]
    events: list[dict] = field(default_factory=list)
    direct_node_records: list[dict] = field(default_factory=list)
    first_treatment_checkpoint_digest: str = ""

    def record(self, kind: str, payload: Optional[dict] = None) -> dict:
        payload = dict(payload or {})
        record = payload.get("direct_node_record")
        if kind == "NODE_SELECTION" and record is None:
            record = payload
        checkpoint = str(
            payload.get("checkpoint")
            or payload.get("checkpoint_hash")
            or (
                record.get("checkpoint_hash")
                if isinstance(record, dict)
                else ""
            )
            or ""
        )
        boundary = "H" if kind in {
            "PAGE_BATCH_DEFERRED", "PAGE_BATCH_REDUCED", "H_CHECKPOINT", "NODE_SELECTION_H"
        } else "C" if kind in {
            "CLOSE_REDUCED", "CLOSE_FAILED", "CLOSE_DEFERRED_TO_VENDOR",
            "CLOSE_CANCELLED", "C_CHECKPOINT", "NODE_SELECTION_C",
        } else (
            "H"
            if kind == "PROSE_CONTROL_OUTPUT"
            and str(payload.get("node") or "").upper().startswith("H")
            else "C"
            if kind == "PROSE_CONTROL_OUTPUT"
            and str(payload.get("node") or "").upper().startswith("C")
            else (
                "H" if isinstance(record, dict)
                and str(record.get("node") or "").upper().startswith("H")
                else "C" if isinstance(record, dict)
                and str(record.get("node") or "").upper().startswith("C")
                else ""
            )
        )
        is_treatment = bool(
            self.treatment_node
            and boundary == self.treatment_node
            and checkpoint
            and not self.first_treatment_checkpoint_digest
            and kind not in {"PAGE_BATCH_DEFERRED", "H_CHECKPOINT", "C_CHECKPOINT"}
        )
        if is_treatment:
            self.first_treatment_checkpoint_digest = checkpoint
            position = "TREATMENT"
        elif self.first_treatment_checkpoint_digest:
            position = "POST_TREATMENT"
        else:
            position = "PRE_TREATMENT"
        event = {
            "event_index": len(self.events),
            "kind": kind,
            "position": position,
            **payload,
        }
        event["event_sha256"] = sha256_hex(canonical_json(event))
        self.events.append(event)

        if isinstance(record, dict):
            selector_trace = _selector_normalization_trace(record)
            materialization_trace = _materialization_trace(record)
            publication_handle_trace = _publication_handle_trace(record)
            normalized = {
                "node": str(record.get("node") or boundary),
                "checkpoint_hash": str(
                    record.get("checkpoint_hash") or record.get("checkpoint") or checkpoint),
                "candidate_view_sha256": str(
                    record.get("candidate_view_sha256") or ""),
                "offered_span_ids": list(record.get("offered_span_ids") or ()),
                "offered_source_occurrence_ids": list(
                    record.get("offered_source_occurrence_ids") or ()),
                "selected_span_ids": list(record.get("selected_span_ids") or ()),
                "staged_span_ids": list(record.get("staged_span_ids") or ()),
                "published_span_ids": list(record.get("published_span_ids") or ()),
                "selected_relations": list(record.get("selected_relations") or ()),
                "staged_relations": list(record.get("staged_relations") or ()),
                "published_relations": list(record.get("published_relations") or ()),
                "selected_gaps": list(record.get("selected_gaps") or ()),
                "staged_gaps": list(record.get("staged_gaps") or ()),
                "published_gaps": list(record.get("published_gaps") or ()),
                "offered_query_attempt_ids": list(
                    record.get("offered_query_attempt_ids") or ()),
                "selected_query_attempt_ids": list(
                    record.get("selected_query_attempt_ids") or ()),
                "staged_query_attempt_ids": list(
                    record.get("staged_query_attempt_ids") or ()),
                "published_query_attempt_ids": list(
                    record.get("published_query_attempt_ids") or ()),
                **selector_trace,
                "fell_back": bool(record.get("fell_back", False)),
                "failure": record.get("failure"),
                "contract": str(record.get("contract") or ""),
                "aggregation": str(record.get("aggregation") or ""),
                "chunker": str(record.get("chunker") or ""),
                **materialization_trace,
                **publication_handle_trace,
                "stage": str(record.get("stage") or ""),
            }
            normalized["record_sha256"] = sha256_hex(canonical_json(normalized))
            self.direct_node_records.append(normalized)
        return event

    def summary(self) -> dict:
        queries = [e for e in self.events if e["kind"] == "SEARCH_QUERY"]
        h_events = [e for e in self.events if e["kind"] == "PAGE_BATCH_REDUCED"]
        close_events = [e for e in self.events if e["kind"].startswith("CLOSE_")]
        tools = sum(int(e.get("siblings") or 0) for e in h_events)
        close_reason = next(
            (str(e.get("close_reason")) for e in reversed(close_events)
             if e.get("close_reason")), "")
        body = {
            "first_treatment_checkpoint_digest":
                self.first_treatment_checkpoint_digest,
            "query_count": len(queries),
            "unique_query_count": len({str(e.get("query") or "") for e in queries}),
            "search_result_count": sum(int(e.get("result_count") or 0) for e in queries),
            "research_rounds": len(h_events),
            "tool_calls_observed": tools,
            "h_checkpoint_count": len({
                str(e.get("checkpoint")) for e in h_events if e.get("checkpoint")
            }),
            "close_checkpoint_count": len({
                str(e.get("checkpoint")) for e in close_events if e.get("checkpoint")
            }),
            "close_reason": close_reason,
            "fallback_count": sum(1 for e in self.events if e.get("fell_back")),
            "failure_count": sum(1 for e in self.events if e.get("failure")),
            "trajectory_sha256": sha256_hex(canonical_json(self.events)),
        }
        return body


def first_divergence(left: Sequence[dict], right: Sequence[dict]) -> Optional[dict]:
    """First ordered trajectory mismatch, including whether it precedes treatment."""
    width = max(len(left), len(right))
    for index in range(width):
        lhs = left[index] if index < len(left) else None
        rhs = right[index] if index < len(right) else None
        # Event digests include the index and are stable because no wall clock enters events.
        if canonical_json(lhs) != canonical_json(rhs):
            positions = {
                str(e.get("position")) for e in (lhs, rhs) if isinstance(e, dict)
            }
            return {
                "event_index": index,
                "left": lhs,
                "right": rhs,
                "before_treatment": bool(positions and positions <= {"PRE_TREATMENT"}),
            }
    return None


def _trajectory_callback(recorder: TrajectoryRecorder):
    """A LangChain callback that records model tool decisions the H hook cannot see.

    H captures batches containing a deferred search. Think-only, ResearchComplete and no-tool
    turns never enter that hook, but they are trajectory outcomes. The callback records only
    control-relevant tool decisions; page-summary completion order is intentionally excluded
    because those calls run concurrently and their callback order is not a stable trajectory.
    """
    from langchain_core.callbacks import BaseCallbackHandler

    relevant = {"tavily_search", "think_tool", "ResearchComplete", "ConductResearch"}

    class _Handler(BaseCallbackHandler):
        def on_llm_end(self, response, **kwargs) -> None:
            names: list[str] = []
            for generation_list in getattr(response, "generations", ()) or ():
                for generation in generation_list or ():
                    message = getattr(generation, "message", None)
                    for call in getattr(message, "tool_calls", ()) or ():
                        name = str(call.get("name") or "")
                        if name in relevant:
                            names.append(name)
            if names:
                metadata = kwargs.get("metadata") or {}
                recorder.record("MODEL_TOOL_DECISION", {
                    "tool_names": names,
                    "tool_call_count": len(names),
                    "graph_node": str(metadata.get("langgraph_node") or ""),
                })

    return _Handler()


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
    execution_binding_sha256: str
    protocol_document_sha256: str
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
    direct_node_records: list = field(default_factory=list)
    trajectory_summary: dict = field(default_factory=dict)
    first_treatment_checkpoint_digest: str = ""
    seed_applied: bool = False
    error: Optional[str] = None
    #: The root/supervisor state a C fork needs to write a report, captured mid-run because
    #: ``final_report_generation`` overrides ``notes`` to [] on its way out -- so by the time
    #: the graph returns, the note vector every fork has to substitute into is already gone.
    #: ``None`` when this cell is not usable as a fork anchor; never a partial envelope.
    continuation: Optional[dict] = None
    #: Digest of the stored envelope, when a continuation store was supplied.
    continuation_digest: str = ""

    @property
    def ok(self) -> bool:
        return self.error is None


def frozen_search_async(
    backend: FrozenTaskCorpusBackend,
    *,
    max_results_default: int = 5,
    on_query: Optional[Callable[[dict], None]] = None,
):
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
            if on_query is not None:
                on_query({
                    "query": str(query),
                    "result_count": len(records),
                    "max_results": int(max_results),
                    "topic": str(topic),
                    # Final citation scoring is restricted to sources this arm actually
                    # retrieved, not every URL that happened to exist elsewhere in the task
                    # pool.  Keep occurrence identities rather than trusting a report URL.
                    "source_occurrence_ids": [
                        str(record.occurrence_id) for record in records
                    ],
                })
            payload.append({
                "query": query,
                "results": [
                    {
                        "url": r.url,
                        "title": r.title,
                        "content": r.content,
                        "score": r.score,
                        "raw_content": r.raw_content if include_raw_content else None,
                        # The vendor ignores unknown result keys.  The H checkpoint adapter
                        # consumes this one so duplicate page bytes from two URL occurrences
                        # retain the citation lineage of the occurrence actually offered.
                        "_shapeflow_occurrence_id": r.occurrence_id,
                    }
                    for r in records
                ],
            })
        return payload

    return search


@contextlib.contextmanager
def install_frozen_search(
    pool: SourcePool,
    snapshots: SnapshotStore,
    *,
    max_results: int,
    on_query: Optional[Callable[[dict], None]] = None,
    content_budget=None,
    tokenizer=None,
):
    """Rebind vendor's search for the duration of one cell, and put it back afterwards.

    Restoring in ``finally`` matters: a cell that raised while the live function was replaced
    would leave the next cell searching the previous task's corpus, and the results would look
    entirely plausible.
    """
    import open_deep_research.utils as vendor_utils

    backend = FrozenTaskCorpusBackend(
        pool, snapshots, content_budget=content_budget, tokenizer=tokenizer)
    original = vendor_utils.tavily_search_async
    vendor_utils.tavily_search_async = frozen_search_async(
        backend, max_results_default=max_results, on_query=on_query)
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


@contextlib.contextmanager
def _summarize_timeout_applied(seconds: str):
    """Bind vendor's summarization timeout for one cell, then put it back.

    Vendor reads ``SHAPEFLOW_SUMMARIZE_TIMEOUT_S`` through the patch and falls back to its own
    60.0 when unset, so an unconfigured graph stays byte-for-byte vendor. It is bound on the one
    path every arm takes, because only P0 can reach vendor's summarizer at all -- P1 returns
    from ``defer_page_batch`` first -- and a value that reached one arm and not the other would
    be exactly the asymmetry this exists to remove.

    Scoped rather than assigned. An earlier version set the variable directly and leaked it into
    the rest of the process: once any test had run a cell, every later test in the session
    inherited a 300s ceiling instead of vendor's 60s, and the suite ran until it was killed.
    """
    previous = os.environ.get("SHAPEFLOW_SUMMARIZE_TIMEOUT_S")
    os.environ["SHAPEFLOW_SUMMARIZE_TIMEOUT_S"] = seconds
    try:
        yield
    finally:
        if previous is None:
            os.environ.pop("SHAPEFLOW_SUMMARIZE_TIMEOUT_S", None)
        else:
            os.environ["SHAPEFLOW_SUMMARIZE_TIMEOUT_S"] = previous


@contextlib.contextmanager
def _odr_seed_applied(seed: int):
    """Attach ``seed`` to both ODR model construction paths, scoped to one cell.

    ODR uses a module-level configurable model for researcher/supervisor/compressor/final
    calls, but creates the page summarizer with a separate ``init_chat_model`` call.  Merely
    storing a seed in :class:`SamplingEnvelope` reaches neither.  This binding updates the
    former's default model parameters and wraps the latter's constructor, then restores both
    in ``finally``.  If the pinned runtime no longer exposes either hook, execution stops
    before the graph call instead of claiming a paired seed that was never sent.
    """
    import langchain.chat_models.base as langchain_chat_models
    import open_deep_research.deep_researcher as vendor_graph
    import open_deep_research.utils as vendor_utils

    model = getattr(vendor_graph, "configurable_model", None)
    fields = getattr(model, "_configurable_fields", None)
    if not isinstance(fields, list):
        raise SeedDeliveryError(
            "pinned ODR configurable_model is not the expected configurable model; cannot "
            "prove seed delivery"
        )
    active = _ACTIVE_ODR_SEED.get()
    if active is not None and active != seed:
        raise SeedDeliveryError(
            f"nested ODR runs requested different seeds ({active} and {seed})"
        )
    with _SEED_PATCH_LOCK:
        process_active = _SEED_PATCH_STATE["active"]
        if process_active is not None:
            raise SeedDeliveryError(
                f"two cells tried to patch the process-wide ODR model concurrently "
                f"({process_active} and {seed}); paired runs require one graph cell per process"
            )
        _SEED_PATCH_STATE["active"] = seed
    token = _ACTIVE_ODR_SEED.set(seed)
    original_init = vendor_utils.init_chat_model
    original_helper = langchain_chat_models._init_chat_model_helper

    def _inject(kwargs):
        requested = kwargs.get("seed")
        if requested is not None and int(requested) != seed:
            raise SeedDeliveryError(
                f"ODR model requested seed {requested}, active cell requires {seed}"
            )
        kwargs["seed"] = seed
        return kwargs

    def seeded_init_chat_model(*args, **kwargs):
        _inject(kwargs)
        return original_init(*args, **kwargs)

    def seeded_helper(*args, **kwargs):
        _inject(kwargs)
        return original_helper(*args, **kwargs)

    # ConfigurableModel eventually calls this helper after its per-role model alias has been
    # resolved.  Patching the helper rather than putting ``seed`` into an otherwise model-less
    # default config avoids instantiating ChatOpenAI before ``model`` exists.
    langchain_chat_models._init_chat_model_helper = seeded_helper
    vendor_utils.init_chat_model = seeded_init_chat_model
    try:
        yield
    finally:
        vendor_utils.init_chat_model = original_init
        langchain_chat_models._init_chat_model_helper = original_helper
        _ACTIVE_ODR_SEED.reset(token)
        with _SEED_PATCH_LOCK:
            _SEED_PATCH_STATE["active"] = None


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
        # Consumed by the scoped configurable-model binding above.  Keeping it in the
        # RunnableConfig also makes the requested seed visible to graph-level traces.
        "seed": int(cell.seed),
        "mcp_config": None,
    }}


async def _stream_graph(graph, initial_state: dict, config: dict):
    """Drive the graph and return ``(final_state, supervisor_update, pre_report_values)``.

    Streaming rather than ``ainvoke`` because two things a C fork needs exist only *during* the
    run. ``final_report_generation`` returns ``{"notes": {"type": "override", "value": []}}``,
    so the note vector every fork substitutes into is empty by the time the graph returns; and
    the supervisor's own ``supervisor_messages`` -- which carry the ``ConductResearch``
    tool-call ids that identify *which* note each child produced -- are likewise only visible
    at the node update that produced them.

    The final state is reconstructed from the last ``values`` chunk, which is exactly what
    ``ainvoke`` returns for this graph, so every existing caller sees no behavioural change.
    """
    final_state: dict = {}
    supervisor_update: dict = {}
    pre_report_values: dict = {}
    stream = graph.astream(initial_state, config, stream_mode=["updates", "values"])
    async for mode, chunk in stream:
        if mode == "updates":
            for node, update in (chunk or {}).items():
                if node == "research_supervisor" and isinstance(update, dict):
                    supervisor_update = dict(update)
                if node == "final_report_generation":
                    # The values chunk *before* this node ran still holds the note vector.
                    pre_report_values = dict(final_state)
        elif mode == "values" and isinstance(chunk, dict):
            final_state = dict(chunk)
    if not pre_report_values:
        # No report node ran (an early exit, or a graph that ends at research). The last
        # observed state is the closest thing to a pre-report snapshot.
        pre_report_values = dict(final_state)
    return final_state, supervisor_update, pre_report_values


def _continuation_from(
    supervisor_update: dict, pre_report_values: dict, *, task_id: str, seed: int
) -> Optional[dict]:
    """Build the fork continuation, or return None rather than a partial one.

    Every field here is held byte-identical across the arms of a boundary, so a missing or
    inconsistent one is a reason to refuse the whole anchor -- before any arm is offered, which
    leaves the ITT denominator untouched. Guessing a note vector would silently attribute one
    child's output to another child's slot.
    """
    from ..odr.adapter import freeze_messages
    from ..odr.continuation import note_slots_from_supervisor_messages

    notes = supervisor_update.get("notes")
    if notes is None:
        notes = pre_report_values.get("notes")
    supervisor_messages = (
        supervisor_update.get("supervisor_messages")
        or pre_report_values.get("supervisor_messages")
    )
    research_brief = (
        supervisor_update.get("research_brief")
        or pre_report_values.get("research_brief")
    )
    root_messages = pre_report_values.get("messages")
    if not notes or not supervisor_messages or not research_brief or not root_messages:
        return None
    try:
        slots = note_slots_from_supervisor_messages(
            supervisor_messages, [str(n) for n in notes]
        )
    except ValueError:
        # The note vector could not be attributed to the tool calls that produced it. That is
        # exactly the ambiguity a fork must not paper over.
        return None
    if not any(slot.tool_call_id for slot in slots):
        return None
    return {
        "task_id": task_id,
        "seed": seed,
        "research_brief": str(research_brief),
        "root_messages": freeze_messages(root_messages),
        "notes": slots,
    }


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
    store_continuation: Optional[Callable[[dict], str]] = None,
    graph: Any = None,
    content_budget=None,
    tokenizer=None,
) -> CellResult:
    """Run one cell on the real graph and return what it produced.

    ``graph`` is injected only so tests can drive a smaller compiled graph; production passes
    None and the module imports vendor's own compiled ``deep_researcher``.

    ``content_budget`` bounds what either arm may see of a page, in characters and in tokens.
    It is threaded down to the frozen backend so vendor and the P1 hook read the same bytes; a
    page bounded in one place and not the other is how P0 came to be handed requests the engine
    refused outright.
    """
    from langchain_core.messages import HumanMessage

    for label, value in (
        ("execution_binding_sha256", cell.execution_binding_sha256),
        ("protocol_document_sha256", cell.protocol_document_sha256),
    ):
        if len(value) != 64 or any(ch not in "0123456789abcdef" for ch in value):
            raise ValueError(f"{label} must be a lowercase SHA-256 digest")

    summarize_timeout = str(
        float(settings.get("week1", "odr", "summarization_timeout_seconds")))

    if graph is None:
        import open_deep_research.deep_researcher as vendor_graph

        graph = vendor_graph.deep_researcher

    checkpoints: list = []
    treatment_node = "H" if cell.page_variant != "P0" \
        else "C" if cell.close_variant != "P0" else None
    recorder = TrajectoryRecorder(treatment_node=treatment_node)
    events = recorder.events

    def on_event(kind: str, payload: dict) -> None:
        recorder.record(kind, payload)

    def capture(checkpoint) -> str:
        digest = checkpoint.digest
        boundary = "H" if type(checkpoint).__name__ == "HCheckpoint" else "C"
        checkpoints.append({"kind": type(checkpoint).__name__, "digest": digest})
        recorder.record(f"{boundary}_CHECKPOINT", {
            "checkpoint": digest,
            "boundary": boundary,
        })
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
        # TaskContext.protocol_sha is consumed as the fork namespace.  It therefore carries
        # the complete approved execution identity, not merely the week1-config digest.
        protocol_sha=cell.execution_binding_sha256,
        variant_id=cell.variant_id,
        seed=cell.seed,
        research_topic=cell.question,
        max_content_length=int(settings.get("week1", "odr", "max_content_length")),
        selected_token_budget=int(settings.get("week1", "measurement",
                                               "selected_token_budget")),
    )
    binding = RunBinding(
        task_id=cell.task_id,
        # This is the logical researcher coordinate, not the treatment assignment.  Putting
        # arm_id in it made otherwise byte-identical pre-treatment H/C checkpoints hash
        # differently by construction, so the coupled-seed diagnostic could never observe a
        # match and a future exact-checkpoint component replay could never share the boundary.
        # Attempt/work attribution remains arm-scoped through work_key and the cell token.
        researcher_id=f"{cell.task_id}:{cell.replicate_id}",
        attempt_id=cell.work_key,
        task_ctx=task_ctx,
        component_trial=cell.component_trial,
        sampling=envelope,
        store_checkpoint=capture,
        on_event=on_event,
    )

    max_results = int(settings.get("acquisition", "frozen_corpus", "top_k"))
    result = CellResult(
        events=events,
        checkpoints=checkpoints,
        direct_node_records=recorder.direct_node_records,
    )
    env = {
        "OPENAI_BASE_URL": f"{provider_base_url.rstrip('/')}/v1/cell/{cell.cell_token}",
        "OPENAI_API_KEY": runner_token,
        # Vendor reads TAVILY_API_KEY at import of its search path. The frozen backend never
        # uses it; a recognisable placeholder makes a leak into a log obviously not a key.
        "TAVILY_API_KEY": "@SHAPEFLOW_FROZEN_CORPUS@",
        "PYTHONHASHSEED": "0",
        "TZ": "UTC",
    }
    with _environment(env), _summarize_timeout_applied(summarize_timeout), \
            _odr_seed_applied(cell.seed), install_frozen_search(
            pool, snapshots, max_results=max_results,
            on_query=lambda payload: recorder.record("SEARCH_QUERY", payload),
            content_budget=content_budget, tokenizer=tokenizer), \
            strategies_bound(bundle), bind_run(binding):
        result.seed_applied = True
        try:
            config = odr_config(settings, cell=cell)
            config["callbacks"] = [_trajectory_callback(recorder)]
            state, supervisor_update, pre_report_values = await _stream_graph(
                graph, {"messages": [HumanMessage(content=cell.question)]}, config
            )
        except Exception as e:  # noqa: BLE001 - recorded, never converted into a P0 result
            result.error = f"{type(e).__name__}: {e}"
            result.trajectory_summary = recorder.summary()
            result.first_treatment_checkpoint_digest = \
                recorder.first_treatment_checkpoint_digest
            return result

    result.continuation = _continuation_from(
        supervisor_update, pre_report_values, task_id=cell.task_id, seed=cell.seed
    )
    if result.continuation is not None and store_continuation is not None:
        result.continuation_digest = store_continuation(result.continuation)

    result.final_report = str(state.get("final_report", "") or "")
    result.notes = tuple(str(n) for n in (state.get("notes") or ()))
    result.raw_notes = tuple(str(n) for n in (state.get("raw_notes") or ()))
    recorder.record("GRAPH_TERMINAL", {
        "final_report_sha256": sha256_hex(result.final_report.encode("utf-8")),
        "notes_count": len(result.notes),
    })
    result.trajectory_summary = recorder.summary()
    result.first_treatment_checkpoint_digest = recorder.first_treatment_checkpoint_digest
    return result


def summarize_events(events: Sequence[dict]) -> dict:
    """Counts the canary and the ledger both need, derived from the run's own events."""
    kinds: dict[str, int] = {}
    for event in events:
        kinds[event["kind"]] = kinds.get(event["kind"], 0) + 1
    queries = [e for e in events if e["kind"] == "SEARCH_QUERY"]
    materialized_token_counts = [
        int(record.get("published_rendered_tokens") or 0)
        for event in events if event["kind"] == "NODE_SELECTION"
        for record in [event.get("direct_node_record") or {}]
        if not record.get("failure") and not record.get("fell_back")
    ]
    prose_token_counts = [
        int(value)
        for event in events if event["kind"] == "PROSE_CONTROL_OUTPUT"
        for value in (event.get("rendered_token_counts") or ())
    ]
    prose_published_token_counts = [
        int(value)
        for event in events if event["kind"] == "PROSE_CONTROL_OUTPUT"
        for value in (event.get("published_token_counts") or ())
    ]
    all_rendered_token_counts = materialized_token_counts + prose_token_counts
    first_treatment = next(
        (str(e.get("checkpoint") or "") for e in events
         if e.get("position") == "TREATMENT"), "")
    return {
        "page_batches_deferred": kinds.get("PAGE_BATCH_DEFERRED", 0),
        "page_batches_reduced": kinds.get("PAGE_BATCH_REDUCED", 0),
        "page_fallbacks": sum(1 for e in events
                              if e["kind"] == "PAGE_BATCH_REDUCED" and e.get("fell_back")),
        "close_reduced": kinds.get("CLOSE_REDUCED", 0),
        "close_failed": kinds.get("CLOSE_FAILED", 0),
        "close_deferred_to_vendor": kinds.get("CLOSE_DEFERRED_TO_VENDOR", 0),
        "close_cancelled": kinds.get("CLOSE_CANCELLED", 0),
        "search_queries": len(queries),
        "unique_search_queries": len({
            str(e.get("query") or "") for e in queries
        }),
        "search_results": sum(int(e.get("result_count") or 0) for e in queries),
        "research_rounds": kinds.get("PAGE_BATCH_REDUCED", 0),
        "rendered_output_count": len(all_rendered_token_counts),
        "max_rendered_tokens": max(all_rendered_token_counts, default=0),
        "total_rendered_tokens": sum(all_rendered_token_counts),
        "max_published_control_tokens": max(
            prose_published_token_counts, default=0
        ),
        "total_published_control_tokens": sum(prose_published_token_counts),
        "first_treatment_checkpoint_digest": first_treatment,
    }
