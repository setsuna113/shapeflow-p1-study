"""The telemetry proxy that sits between ODR and vLLM.

Every model request flows through here so it can be tagged (op class + task/arm/variant/node ids),
timed at the dispatch->response boundary, and recorded as a RequestEvent. In the causal isolation
mode it also enforces the single-upstream-in-flight invariant: an exclusive gate serializes upstream
calls and records their intervals, so the work metric's non-overlap assumption is *made true*, not
merely assumed. Sibling requests wait at the gate, and that wait is recorded as queue time -- kept
out of the service-work total.

The upstream call is injected, so the tagging, gating, and event recording are tested without a
vLLM; the host wires the real httpx forwarder to 127.0.0.1.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Awaitable, Callable, Optional

from .request_tags import OpClass
from .work_accounting import RequestEvent

__all__ = ["ProxyRequest", "UpstreamResult", "TelemetryProxy", "InflightGate", "OverlapViolation"]


@dataclass(frozen=True)
class ProxyRequest:
    op_class: OpClass
    task_id: str
    arm_id: str
    body: dict


@dataclass(frozen=True)
class UpstreamResult:
    response: dict
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: Optional[int] = None
    finish_reason: str = "stop"


class OverlapViolation(RuntimeError):
    """Two upstream calls overlapped while the causal gate should have serialized them."""


class InflightGate:
    """Serializes upstream calls in causal mode and asserts they never overlap.

    In operational mode (``exclusive=False``) it is a no-op, so continuous batching runs freely.
    """

    def __init__(self, *, exclusive: bool, clock: Callable[[], float]) -> None:
        self._exclusive = exclusive
        self._clock = clock
        self._lock = asyncio.Lock()
        self._last_end: Optional[float] = None

    async def __aenter__(self) -> "InflightGate":
        if self._exclusive:
            await self._lock.acquire()
        return self

    async def __aexit__(self, *exc) -> None:
        if self._exclusive:
            self._lock.release()

    def record_interval(self, dispatch_ts: float, response_end_ts: float) -> None:
        """Assert this interval does not overlap the previous one (causal mode only)."""
        if self._exclusive and self._last_end is not None and dispatch_ts < self._last_end - 1e-9:
            raise OverlapViolation(
                f"upstream interval [{dispatch_ts},{response_end_ts}] overlaps prior end {self._last_end}"
            )
        self._last_end = response_end_ts


Upstream = Callable[[dict], Awaitable[UpstreamResult]]
EventSink = Callable[[RequestEvent], None]


class TelemetryProxy:
    def __init__(
        self,
        upstream: Upstream,
        *,
        gate: InflightGate,
        event_sink: EventSink,
        clock: Callable[[], float],
    ) -> None:
        self._upstream = upstream
        self._gate = gate
        self._sink = event_sink
        self._clock = clock

    async def handle(self, request: ProxyRequest) -> dict:
        ingress_ts = self._clock()
        async with self._gate:
            dispatch_ts = self._clock()
            result = await self._upstream(request.body)
            response_end_ts = self._clock()
            self._gate.record_interval(dispatch_ts, response_end_ts)

        self._sink(RequestEvent(
            op_class=request.op_class,
            task_id=request.task_id,
            arm_id=request.arm_id,
            upstream_dispatch_ts=dispatch_ts,
            upstream_response_end_ts=response_end_ts,
            proxy_ingress_ts=ingress_ts,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            cached_prompt_tokens=result.cached_prompt_tokens,
            terminal_status="COMMITTED",
        ))
        return result.response
