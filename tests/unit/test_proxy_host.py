"""ProxyHost: the obligations a host outside the engine has to meet on its own.

The shared C1-C10 obligations run against this host in the conformance suite. These tests cover
what only a proxy can get wrong: reading the engine's real metrics format, telling an engine
restart and an aged scrape apart from a current tick, and putting P0's siblings on the wire as
independent requests rather than a batch.
"""

from __future__ import annotations

import threading

import pytest
from fixtures.fake_vllm import FakeVllm, bodies_for

from shapeflow.broker.driver import run_tick
from shapeflow.broker.hosts.proxy import EngineUnobservable, parse_prometheus
from shapeflow.broker.reference import EngineGatedBroker, GreedyJointBroker
from shapeflow.contracts.offer import (
    Boundary,
    ContractViolation,
    Decision,
    Form,
    Offer,
    P0Plan,
    P1Plan,
    RenderDescriptor,
    RequestDescriptor,
    StaleSnapshot,
)
from shapeflow.runtime.request_tags import OpClass


def _rd(op, *, uncached=100, decode=200):
    return RequestDescriptor(op_class=op, prompt_sha256="a" * 64, max_tokens=decode,
                             est_uncached_prefill=uncached, est_cached_prefill=0,
                             est_decode=decode)


def h_offer(item_id="H1", *, pages=3, selector_cost=50):
    return Offer(
        item_id=item_id, boundary=Boundary.H, unit_id=f"turn-{item_id}", submitted_at_ns=0,
        p0=P0Plan(requests=tuple(_rd(OpClass.PAGE_P0_SUMMARY) for _ in range(pages))),
        p1=P1Plan(selector=_rd(OpClass.PAGE_P1_SELECTOR_BATCH, uncached=selector_cost, decode=60),
                  render=RenderDescriptor(est_cpu_ms=1.0, candidate_count=9)),
    )


def _decision(snapshot, **forms):
    return Decision(admitted=tuple(forms), form=dict(forms), deferred=(),
                    snapshot_version=snapshot.version, solve_ns=0)


class _Clock:
    def __init__(self):
        self.now = 10_000_000_000

    def __call__(self):
        return self.now


def _host(engine, offers=(), **kwargs):
    host = engine.host(**kwargs)
    for offer in offers:
        host.submit(offer, bodies_for(offer))
    return host


# --- reading the engine ----------------------------------------------------------------------


def test_metrics_are_summed_over_label_sets_and_names_match_exactly():
    text = (
        '# HELP x\nvllm:num_requests_waiting{engine="0",model_name="a b"} 2.0\n'
        'vllm:num_requests_waiting{engine="1",model_name="a b"} 3.0\n'
        'vllm:num_requests_waiting_by_reason{engine="0",reason="capacity"} 7.0\n'
        "process_start_time_seconds 1.5e+09\n"
    )
    values = parse_prometheus(text, {"vllm:num_requests_waiting", "process_start_time_seconds"})
    assert values == {"vllm:num_requests_waiting": 5.0, "process_start_time_seconds": 1.5e9}


def test_a_snapshot_carries_the_engine_gauges_as_registered_features():
    engine = FakeVllm(running=3, waiting=2, kv=0.25)
    snapshot = _host(engine).take_snapshot()
    assert dict(snapshot.features) == {
        "engine.num_requests_running": 3.0, "engine.num_requests_waiting": 2.0,
        "engine.kv_cache_usage_perc": 0.25, "local.inflight_total": 0,
    }
    assert snapshot.version.epoch == f"vllm-start-{engine.start_time!r}"


def test_metrics_without_the_gauges_produce_no_snapshot():
    host = FakeVllm().host()
    host._scrape = lambda: "process_start_time_seconds 1.0\n"
    with pytest.raises(EngineUnobservable):
        host.take_snapshot()


# --- staleness: nothing is sent --------------------------------------------------------------


def test_an_engine_restart_makes_the_earlier_snapshot_stale_and_sends_nothing():
    engine = FakeVllm()
    host = _host(engine, [h_offer()])
    before = host.take_snapshot()
    engine.start_time += 60.0  # the engine restarted between the scrape and the commit
    after = host.take_snapshot()
    assert before.version.epoch != after.version.epoch
    with pytest.raises(StaleSnapshot):
        host.materialize(_decision(before, H1=Form.P0), before)
    assert host.dispatches == [] and engine.requests == []
    assert [o.item_id for o in host.pending()] == ["H1"]


def test_a_snapshot_older_than_the_ceiling_is_stale_and_sends_nothing():
    engine, clock = FakeVllm(), _Clock()
    host = _host(engine, [h_offer()], clock_ns=clock, max_snapshot_age_ns=1_000_000)
    snapshot = host.take_snapshot()
    clock.now += 2_000_000
    with pytest.raises(StaleSnapshot):
        host.materialize(_decision(snapshot, H1=Form.P1), snapshot)
    assert host.dispatches == [] and engine.requests == []
    assert host.gate_log[-1]["why"] == "too old" and host.gate_log[-1]["age_ns"] == 2_000_000


def test_the_driver_retries_a_stale_tick_and_commits_on_a_fresh_snapshot():
    engine, clock = FakeVllm(), _Clock()
    host = _host(engine, [h_offer()], clock_ns=clock, max_snapshot_age_ns=1_000_000)
    original = host.take_snapshot
    calls = []

    def aging_snapshot():
        snapshot = original()
        calls.append(snapshot)
        if len(calls) == 1:
            clock.now += 5_000_000  # first attempt: the scrape is too old by commit time
        return snapshot

    host.take_snapshot = aging_snapshot
    outcome = run_tick(host, GreedyJointBroker(capacity_tokens=100_000))
    assert outcome.committed and outcome.stale_retries == 1
    assert host.drain(timeout_s=5)
    assert host.dispatch_count("H1") == 1  # P1 was cheaper; sent once, after the retry


# --- materialization -------------------------------------------------------------------------


def test_p0_sends_one_independent_request_per_page_and_p1_nothing():
    engine = FakeVllm()
    host = _host(engine, [h_offer(pages=3)])
    snapshot = host.take_snapshot()
    result = host.materialize(_decision(snapshot, H1=Form.P0), snapshot)
    assert host.drain(timeout_s=5)
    assert result.dispatched == {"H1": 3}
    assert sorted(r["prompt"] for r in engine.requests) == ["H1 P0 0", "H1 P0 1", "H1 P0 2"]
    assert all(isinstance(r["prompt"], str) for r in engine.requests), "one prompt per request"
    assert host.pending() == ()


def test_p0_siblings_do_not_wait_for_each_other():
    """Sibling 0 is held inside the engine; siblings 1 and 2 must still complete."""
    engine = FakeVllm()
    release = threading.Event()
    engine.gate = lambda body: release.wait(5) if body["prompt"].endswith(" 0") else None
    host = _host(engine, [h_offer(pages=3)])
    snapshot = host.take_snapshot()
    host.materialize(_decision(snapshot, H1=Form.P0), snapshot)
    try:
        for _ in range(200):
            if len(host.records) == 2:
                break
            threading.Event().wait(0.01)
        assert sorted(r.ordinal for r in host.records) == [1, 2]
        assert host.inflight() == 1
    finally:
        release.set()
    assert host.drain(timeout_s=5)


def test_a_transport_failure_is_recorded_on_the_request_and_does_not_uncommit():
    engine = FakeVllm()
    engine.fail = lambda body: body["prompt"].endswith(" 1")
    host = _host(engine, [h_offer(pages=3)])
    snapshot = host.take_snapshot()
    result = host.materialize(_decision(snapshot, H1=Form.P0), snapshot)
    assert host.drain(timeout_s=5)
    assert result.materialized == ("H1",)
    errors = {r.ordinal: r.error for r in host.records}
    assert errors[1].startswith("ConnectionError") and errors[0] is None and errors[2] is None


def test_bodies_must_match_the_priced_descriptors():
    host = FakeVllm().host()
    offer = h_offer(pages=3)
    bodies = bodies_for(offer)
    bodies[Form.P0] = bodies[Form.P0][:2]
    with pytest.raises(ContractViolation, match="2 bodies for 3 descriptors"):
        host.submit(offer, bodies)
    assert host.pending() == ()


# --- the engine-reading policy ---------------------------------------------------------------


def test_the_engine_gate_defers_everything_while_the_engine_queue_is_long():
    host = _host(FakeVllm(waiting=9), [h_offer("H1"), h_offer("H2")])
    outcome = run_tick(host, EngineGatedBroker(GreedyJointBroker(capacity_tokens=100_000),
                                               max_waiting=4))
    assert outcome.committed and outcome.decision.admitted == ()
    assert set(outcome.decision.deferred) == {"H1", "H2"}
    assert host.dispatches == [] and len(host.pending()) == 2


def test_the_engine_gate_open_decides_as_the_inner_broker():
    offers = [h_offer("H1"), h_offer("H2")]
    gated = run_tick(_host(FakeVllm(waiting=1), offers),
                     EngineGatedBroker(GreedyJointBroker(capacity_tokens=100_000), max_waiting=4))
    plain = run_tick(_host(FakeVllm(waiting=1), offers), GreedyJointBroker(capacity_tokens=100_000))
    assert gated.decision.form == plain.decision.form
    assert gated.decision.admitted == plain.decision.admitted
