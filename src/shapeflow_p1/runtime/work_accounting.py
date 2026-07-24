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

from dataclasses import dataclass
from typing import Iterable, Optional

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
