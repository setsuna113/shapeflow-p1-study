"""Telemetry proxy tagging/gating and NVML energy integration."""

from __future__ import annotations

import asyncio

import pytest

from shapeflow.runtime.nvml_sampler import PowerSample, integrate_energy_joules
from shapeflow.runtime.openai_proxy import (
    InflightGate,
    OverlapViolation,
    ProxyRequest,
    TelemetryProxy,
    UpstreamResult,
)
from shapeflow.runtime.request_tags import OpClass
from shapeflow.runtime.work_accounting import RequestEvent, isolated_service_work


class _Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t

    def advance(self, dt):
        self.t += dt


async def test_proxy_records_tagged_event():
    clock = _Clock()
    events: list[RequestEvent] = []

    async def upstream(body):
        clock.advance(1.0)  # 1s of service
        return UpstreamResult(response={"ok": True}, prompt_tokens=50, completion_tokens=10)

    gate = InflightGate(exclusive=True, clock=clock)
    proxy = TelemetryProxy(upstream, gate=gate, event_sink=events.append, clock=clock)
    resp = await proxy.handle(ProxyRequest(OpClass.RESEARCHER_REACT, "t1", "H", {"model": "q"}))

    assert resp == {"ok": True}
    assert len(events) == 1
    ev = events[0]
    assert ev.op_class is OpClass.RESEARCHER_REACT
    assert ev.service_seconds == pytest.approx(1.0)
    assert ev.prompt_tokens == 50


async def test_causal_gate_serializes_and_work_is_non_overlapping():
    clock = _Clock()
    events: list[RequestEvent] = []

    async def upstream(body):
        # simulate work; because the gate is exclusive, two of these cannot interleave
        clock.advance(2.0)
        return UpstreamResult(response={}, prompt_tokens=1, completion_tokens=1)

    gate = InflightGate(exclusive=True, clock=clock)
    proxy = TelemetryProxy(upstream, gate=gate, event_sink=events.append, clock=clock)

    await asyncio.gather(
        proxy.handle(ProxyRequest(OpClass.RESEARCHER_REACT, "t", "H", {})),
        proxy.handle(ProxyRequest(OpClass.RESEARCHER_REACT, "t", "H", {})),
    )
    # Both recorded; their intervals do not overlap, so the work metric is valid.
    assert len(events) == 2
    assert isolated_service_work(events) == pytest.approx(4.0)


async def test_gate_records_detect_overlap_directly():
    clock = _Clock()
    gate = InflightGate(exclusive=True, clock=clock)
    gate.record_interval(0.0, 2.0)
    with pytest.raises(OverlapViolation):
        gate.record_interval(1.0, 3.0)  # starts before the prior interval ended


# --- energy ---------------------------------------------------------------------------


def test_integrate_energy_constant_power():
    # 100 W held for 2 s = 200 J.
    samples = [PowerSample(0.0, 100.0), PowerSample(1.0, 100.0), PowerSample(2.0, 100.0)]
    assert integrate_energy_joules(samples) == pytest.approx(200.0)


def test_integrate_energy_trapezoid():
    # ramp 0->100 W over 1 s = 50 J.
    samples = [PowerSample(0.0, 0.0), PowerSample(1.0, 100.0)]
    assert integrate_energy_joules(samples) == pytest.approx(50.0)


def test_integrate_energy_needs_two_samples():
    assert integrate_energy_joules([PowerSample(0.0, 100.0)]) == 0.0


def test_integrate_energy_rejects_unordered_after_sort_is_fine():
    # Given out of order, it sorts; a genuinely negative dt cannot occur post-sort.
    samples = [PowerSample(2.0, 100.0), PowerSample(0.0, 100.0), PowerSample(1.0, 100.0)]
    assert integrate_energy_joules(samples) == pytest.approx(200.0)
