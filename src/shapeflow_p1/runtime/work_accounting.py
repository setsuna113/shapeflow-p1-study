"""Model-work accounting, and the non-overlap invariant that makes it meaningful.

The primary isolated work metric is deliberately simple and observable: the summed
upstream service intervals (dispatch -> response-end) of the treatment-model calls. It is NOT
an unobservable per-request "engine-busy" figure. For that sum to be a fair measure of work,
the isolated mode must actually run one upstream request at a time -- so the gateway asserts
the treatment intervals never overlap, and if they do, the task's work metric is declared
invalid rather than quietly summing overlapping time.

Two costs are kept out of the work total by construction:

- **Gateway queue wait** (ingress -> dispatch): reported separately, never added to work, so a
  sibling request waiting its turn doesn't inflate the measured service time.
- **Judge cost**: judge op classes are excluded from treatment work entirely and reported as
  API cost.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Optional, Sequence

from .request_tags import JUDGE_OPS, TREATMENT_OPS, OpClass

__all__ = [
    "RequestEvent",
    "OverlapError",
    "assert_non_overlapping",
    "isolated_service_work",
    "queue_wait_total",
    "work_by_op",
    "token_work",
    "WorkSummary",
    "WorkExtraction",
    "extract_request_events",
    "summarize_work_extraction",
]


class OverlapError(RuntimeError):
    """Two treatment intervals overlapped in isolated mode -- the work metric is invalid."""


@dataclass(frozen=True)
class RequestEvent:
    op_class: OpClass
    task_id: str
    arm_id: str
    upstream_dispatch_ts: float
    upstream_response_end_ts: float
    proxy_ingress_ts: Optional[float] = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: Optional[int] = None
    retry_ordinal: int = 0
    terminal_status: str = "COMMITTED"
    call_id: str = ""
    work_key: str = ""
    run_id: str = ""
    variant_id: str = ""
    replicate_id: str = ""
    layer: str = ""

    @property
    def service_seconds(self) -> float:
        return self.upstream_response_end_ts - self.upstream_dispatch_ts


@dataclass(frozen=True)
class WorkSummary:
    service_seconds: float
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    count: int


@dataclass(frozen=True)
class WorkExtraction:
    """Durable inference telemetry for one or more work keys.

    A response-complete attempt becomes a RequestEvent. A sent attempt that ended
    FAILED_UNKNOWN cannot honestly be assigned a response-end timestamp, so it is counted
    separately and keeps its worst-case budget charge. Malformed/missing telemetry is explicit;
    callers must never turn it into zero work.
    """

    events: tuple[RequestEvent, ...]
    unknown_after_send: int = 0
    unavailable_attempt_ids: tuple[str, ...] = ()

    @property
    def complete(self) -> bool:
        return self.unknown_after_send == 0 and not self.unavailable_attempt_ids


def _treatment(events: Iterable[RequestEvent]) -> list[RequestEvent]:
    return [e for e in events if e.op_class in TREATMENT_OPS]


def assert_non_overlapping(events: Iterable[RequestEvent]) -> None:
    """Assert the treatment intervals form a non-overlapping sequence (isolated-mode gateway
    invariant). Raises :class:`OverlapError` on the first overlap."""
    intervals = sorted(
        ((e.upstream_dispatch_ts, e.upstream_response_end_ts) for e in _treatment(events)),
        key=lambda iv: iv[0],
    )
    for (s0, e0), (s1, e1) in zip(intervals, intervals[1:]):
        if s1 < e0:
            raise OverlapError(
                f"treatment intervals overlap: [{s0},{e0}] and [{s1},{e1}]; "
                "isolated work metric is invalid for this task"
            )


def isolated_service_work(events: Iterable[RequestEvent], *, verify: bool = True) -> float:
    """Sum treatment service seconds. With ``verify`` (the default), first assert non-overlap;
    a summed overlap would over- or under-count work depending on scheduling, so it is refused."""
    events = list(events)
    if verify:
        assert_non_overlapping(events)
    return sum(e.service_seconds for e in _treatment(events))


def queue_wait_total(events: Iterable[RequestEvent]) -> float:
    """Total gateway queue wait (ingress -> dispatch), reported separately from work."""
    total = 0.0
    for e in events:
        if e.proxy_ingress_ts is not None:
            total += max(0.0, e.upstream_dispatch_ts - e.proxy_ingress_ts)
    return total


def work_by_op(events: Iterable[RequestEvent]) -> dict[OpClass, WorkSummary]:
    """Per-op-class aggregation (all ops, including judge ops, so callers can report each)."""
    acc: dict[OpClass, list] = {}
    for e in events:
        s = acc.setdefault(e.op_class, [0.0, 0, 0, 0, 0])
        s[0] += e.service_seconds
        s[1] += e.prompt_tokens
        s[2] += e.completion_tokens
        s[3] += e.cached_prompt_tokens or 0
        s[4] += 1
    return {
        op: WorkSummary(service_seconds=v[0], prompt_tokens=v[1], completion_tokens=v[2],
                        cached_prompt_tokens=v[3], count=v[4])
        for op, v in acc.items()
    }


def token_work(events: Iterable[RequestEvent], *, treatment_only: bool = True) -> dict[str, int]:
    """Total prompt/completion/cached tokens. Treatment ops only by default (judge excluded)."""
    prompt = completion = cached = 0
    for e in events:
        if treatment_only and e.op_class not in TREATMENT_OPS:
            continue
        prompt += e.prompt_tokens
        completion += e.completion_tokens
        cached += e.cached_prompt_tokens or 0
    return {"prompt_tokens": prompt, "completion_tokens": completion, "cached_prompt_tokens": cached}


def extract_request_events(ledger: Any, *, work_keys: Optional[Sequence[str]] = None) -> WorkExtraction:
    """Read immutable vLLM request intervals from the provider ledger.

    This is the production bridge that the previous in-memory-only accounting lacked. Coordinates
    come from the telemetry captured at dispatch/response time; the external-call columns provide
    the state-machine identity and make a missing or corrupt telemetry row auditable.
    """

    sql = (
        "SELECT c.call_id, c.op_class, c.work_key, a.attempt_id, a.attempt_ordinal,"
        " a.state, a.dispatched_at, a.response_end_at, a.proxy_ingress_at,"
        " a.usage_json, a.telemetry_json"
        " FROM external_calls c JOIN external_call_attempts a USING(call_id)"
        " WHERE c.provider='vllm'"
    )
    params: list[str] = []
    if work_keys is not None:
        keys = [str(k) for k in work_keys]
        if not keys:
            return WorkExtraction(events=())
        sql += " AND c.work_key IN (" + ",".join("?" for _ in keys) + ")"
        params.extend(keys)
    sql += " ORDER BY a.dispatched_at, a.attempt_id"

    with ledger.lock:
        rows = ledger.raw_connection.execute(sql, params).fetchall()

    events: list[RequestEvent] = []
    unknown = 0
    unavailable: list[str] = []
    for row in rows:
        attempt_id = str(row["attempt_id"])
        dispatch = row["dispatched_at"]
        end = row["response_end_at"]
        if dispatch is None or end is None:
            if row["state"] == "FAILED_UNKNOWN" and dispatch is not None:
                unknown += 1
            else:
                unavailable.append(attempt_id)
            continue
        try:
            telemetry = json.loads(row["telemetry_json"] or "{}")
            usage = json.loads(row["usage_json"] or "{}")
            op = OpClass(str(row["op_class"]))
        except (json.JSONDecodeError, TypeError, ValueError):
            unavailable.append(attempt_id)
            continue
        prompt = int(telemetry.get("prompt_tokens", usage.get("prompt_tokens", 0)) or 0)
        completion = int(
            telemetry.get("completion_tokens", usage.get("completion_tokens", 0)) or 0
        )
        cached = telemetry.get("cached_prompt_tokens")
        if cached is None:
            details = usage.get("prompt_tokens_details") or {}
            cached = details.get("cached_tokens") if isinstance(details, dict) else None
        events.append(
            RequestEvent(
                op_class=op,
                task_id=str(telemetry.get("task_id", "")),
                arm_id=str(telemetry.get("arm_id", "")),
                upstream_dispatch_ts=float(dispatch),
                upstream_response_end_ts=float(end),
                proxy_ingress_ts=(
                    float(row["proxy_ingress_at"]) if row["proxy_ingress_at"] is not None else None
                ),
                prompt_tokens=prompt,
                completion_tokens=completion,
                cached_prompt_tokens=int(cached) if cached is not None else None,
                retry_ordinal=int(row["attempt_ordinal"]),
                terminal_status=str(row["state"]),
                call_id=str(row["call_id"]),
                work_key=str(row["work_key"] or ""),
                run_id=str(telemetry.get("run_id", "")),
                variant_id=str(telemetry.get("variant_id", "")),
                replicate_id=str(telemetry.get("replicate_id", "")),
                layer=str(telemetry.get("layer", "")),
            )
        )
    return WorkExtraction(
        events=tuple(events),
        unknown_after_send=unknown,
        unavailable_attempt_ids=tuple(unavailable),
    )


def summarize_work_extraction(
    extraction: WorkExtraction, *, require_isolated: bool
) -> dict[str, Any]:
    """Render one audit-safe work object for a cell/result record."""

    events = list(extraction.events)
    overlap_error = ""
    if require_isolated:
        try:
            assert_non_overlapping(events)
        except OverlapError as exc:
            overlap_error = str(exc)
    by_op = work_by_op(events)
    return {
        "telemetry_complete": extraction.complete,
        "unknown_after_send": extraction.unknown_after_send,
        "unavailable_attempt_ids": list(extraction.unavailable_attempt_ids),
        "overlap_valid": not overlap_error,
        "overlap_error": overlap_error,
        "service_seconds": (
            isolated_service_work(events, verify=False) if not overlap_error else None
        ),
        "queue_wait_seconds": queue_wait_total(events),
        "tokens": token_work(events),
        "by_op": {
            op.value: {
                "service_seconds": summary.service_seconds,
                "prompt_tokens": summary.prompt_tokens,
                "completion_tokens": summary.completion_tokens,
                "cached_prompt_tokens": summary.cached_prompt_tokens,
                "count": summary.count,
            }
            for op, summary in sorted(by_op.items(), key=lambda item: item[0].value)
        },
    }
