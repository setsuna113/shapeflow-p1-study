"""The broker side of a brokered agent: AgentHost, DeadlineBroker, the service and its client.

The agent side waits on these, so the properties that matter are the ones an agent depends on:
a decision releases it exactly once, a stale snapshot releases nobody, a withdrawn item is never
decided, and an offer the policy would defer forever is still decided once it is overdue.
"""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request

import pytest
from fixtures.fake_vllm import FakeVllm, agent_host_with

from shapeflow.broker.client import BROKER, FAIL_CLOSED, BrokerClient
from shapeflow.broker.codec import offer_digest, offer_from_content
from shapeflow.broker.driver import run_tick
from shapeflow.broker.hosts.agent import DECIDED, PENDING, WITHDRAWN
from shapeflow.broker.reference import DeadlineBroker, EngineGatedBroker, GreedyJointBroker
from shapeflow.broker.service import BrokerService
from shapeflow.contracts.offer import (
    Boundary,
    ContractViolation,
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


def h_offer(item_id="H1", *, pages=3, deadline_ns=None, submitted_at_ns=0, with_p1=True,
            uncached=100):
    return Offer(
        item_id=item_id, boundary=Boundary.H, unit_id=f"turn-{item_id}",
        submitted_at_ns=submitted_at_ns, deadline_ns=deadline_ns,
        p0=P0Plan(requests=tuple(_rd(OpClass.PAGE_P0_SUMMARY, uncached=uncached)
                                 for _ in range(pages))),
        p1=P1Plan(selector=_rd(OpClass.PAGE_P1_SELECTOR_BATCH, uncached=50, decode=60),
                  render=RenderDescriptor(est_cpu_ms=1.0, candidate_count=9))
        if with_p1 else None,
    )


# --- codec -----------------------------------------------------------------------------------


@pytest.mark.parametrize("offer", [h_offer(), h_offer(with_p1=False, deadline_ns=123)])
def test_an_offer_survives_the_wire_unchanged(offer):
    wire = json.loads(json.dumps(offer.content()))
    back = offer_from_content(wire)
    assert back == offer and offer_digest(back) == offer_digest(offer)


def test_a_field_the_codec_does_not_carry_is_refused():
    wire = json.loads(json.dumps(h_offer().content()))
    wire["p1"]["render"]["est_cpu_ms"] = 1.5
    wire["surprise"] = 1
    with pytest.raises(ContractViolation, match="round-trip"):
        offer_from_content(wire)


# --- AgentHost -------------------------------------------------------------------------------


def test_submit_is_idempotent_and_refuses_a_different_plan_under_the_same_id():
    host = agent_host_with([h_offer()])
    assert host.submit(h_offer())["state"] == PENDING
    with pytest.raises(ContractViolation, match="different content"):
        host.submit(h_offer(pages=2))


def test_a_decision_releases_the_waiting_agent_with_its_ticket():
    host = agent_host_with([h_offer()])
    released = {}
    waiter = threading.Thread(target=lambda: released.update(host.wait("H1", timeout_s=5)))
    waiter.start()
    outcome = run_tick(host, GreedyJointBroker(capacity_tokens=100_000))
    waiter.join(5)
    assert outcome.committed and released["state"] == DECIDED
    ticket = released["ticket"]
    assert ticket["form"] == "P1" and ticket["requests"] == 1
    assert ticket["decision_digest"] == outcome.decision.digest
    assert host.pending() == ()


def test_a_stale_snapshot_releases_nobody():
    engine = FakeVllm()
    host = agent_host_with([h_offer()], engine)
    before = host.take_snapshot()
    engine.start_time += 60.0
    host.take_snapshot()
    decision = GreedyJointBroker(capacity_tokens=100_000).decide(host.pending(), before, 0)
    with pytest.raises(StaleSnapshot):
        host.materialize(decision, before)
    assert host.status("H1")["state"] == PENDING and host.dispatches == []


def test_withdraw_before_a_decision_takes_the_item_out_of_every_later_tick():
    host = agent_host_with([h_offer("H1"), h_offer("H2")])
    assert host.withdraw("H1", "AGENT_WAIT_TIMEOUT")["state"] == WITHDRAWN
    outcome = run_tick(host, GreedyJointBroker(capacity_tokens=100_000))
    assert outcome.decision.admitted == ("H2",)
    assert host.forms_of("H1") == set()


def test_withdraw_after_a_decision_leaves_the_decision_standing():
    host = agent_host_with([h_offer()])
    run_tick(host, GreedyJointBroker(capacity_tokens=100_000))
    status = host.withdraw("H1", "AGENT_WAIT_TIMEOUT")
    assert status["state"] == DECIDED and status["withdrawn"] is None


def test_materializing_a_withdrawn_item_is_refused():
    host = agent_host_with([h_offer()])
    snapshot = host.take_snapshot()
    decision = GreedyJointBroker(capacity_tokens=100_000).decide(host.pending(), snapshot, 0)
    host.withdraw("H1", "x")
    with pytest.raises(ContractViolation, match="not pending"):
        host.materialize(decision, snapshot)


# --- DeadlineBroker --------------------------------------------------------------------------


def test_an_overdue_offer_is_admitted_even_when_the_gate_is_shut_and_it_does_not_fit():
    host = agent_host_with(
        [h_offer("H1", deadline_ns=1, uncached=50_000), h_offer("H2", deadline_ns=None)],
        FakeVllm(waiting=9))
    broker = DeadlineBroker(EngineGatedBroker(GreedyJointBroker(capacity_tokens=1000),
                                              max_waiting=4))
    outcome = run_tick(host, broker)
    assert outcome.decision.admitted == ("H1",)
    assert outcome.decision.deferred == ("H2",)
    assert "deadline-forced 1" in outcome.decision.reason
    assert "engine gate shut" in outcome.decision.reason


def test_a_deadline_forced_form_is_the_cheaper_one_and_deterministic():
    offers = [h_offer("H1", deadline_ns=1), h_offer("H2", deadline_ns=1, with_p1=False)]
    digests = set()
    for _ in range(3):
        outcome = run_tick(agent_host_with(offers), DeadlineBroker())
        digests.add(outcome.decision.digest)
        assert outcome.decision.form == {"H1": Form.P1, "H2": Form.P0}
    assert len(digests) == 1


# --- service and client ----------------------------------------------------------------------


@pytest.fixture()
def service_factory(tmp_path):
    started = []

    def make(engine=None, broker=None, tick_ms=20.0):
        service = BrokerService(agent_host_with([], engine), broker or GreedyJointBroker(
            capacity_tokens=100_000), tick_ms=tick_ms, log_dir=tmp_path / "broker")
        port = service.start()
        started.append(service)
        return service, f"http://127.0.0.1:{port}"

    yield make
    for service in started:
        service.stop()


async def test_the_client_gets_the_broker_decision(service_factory):
    service, url = service_factory()
    now = time.time_ns()
    resolution = await BrokerClient(url).resolve(h_offer(deadline_ns=now + 10**9))
    assert resolution.source == BROKER and resolution.form is Form.P1
    assert resolution.ticket["form"] == "P1"


async def test_a_wait_past_the_deadline_withdraws_and_runs_p0_fail_closed(service_factory):
    # A gate that never opens and no deadline policy: the agent must give up on its own.
    service, url = service_factory(
        FakeVllm(waiting=9),
        EngineGatedBroker(GreedyJointBroker(capacity_tokens=100_000), max_waiting=4))
    client = BrokerClient(url, grace_ns=200_000_000, poll_ms=100)
    resolution = await client.resolve(h_offer(deadline_ns=time.time_ns() + 100_000_000))
    assert resolution.source == FAIL_CLOSED and resolution.reason == "AGENT_WAIT_TIMEOUT"
    assert resolution.form is Form.P0
    assert service.host.status("H1")["state"] == WITHDRAWN


async def test_the_deadline_policy_decides_before_the_agent_gives_up(service_factory):
    service, url = service_factory(
        FakeVllm(waiting=9),
        DeadlineBroker(EngineGatedBroker(GreedyJointBroker(capacity_tokens=100_000),
                                         max_waiting=4)))
    client = BrokerClient(url, grace_ns=2_000_000_000, poll_ms=100)
    resolution = await client.resolve(h_offer(deadline_ns=time.time_ns() + 100_000_000))
    assert resolution.source == BROKER
    assert "deadline-forced" in resolution.ticket["reason"]


def test_an_unknown_item_is_a_404_and_a_withdraw_waits_for_the_tick(service_factory):
    service, url = service_factory(tick_ms=10_000.0)
    with pytest.raises(urllib.error.HTTPError) as err:
        urllib.request.urlopen(url + "/v1/offers/nope?wait_ms=0", timeout=5)  # noqa: S310
    assert err.value.code == 404

    service.host.submit(h_offer())
    done = threading.Event()
    with service._tick_lock:
        worker = threading.Thread(target=lambda: (service.withdraw("H1", "x"), done.set()))
        worker.start()
        assert not done.wait(0.3), "a withdrawal landed inside a tick"
    worker.join(5)
    assert done.is_set() and service.host.status("H1")["state"] == WITHDRAWN


def test_the_service_refuses_a_non_loopback_bind():
    service = BrokerService(agent_host_with([]), GreedyJointBroker())
    with pytest.raises(ValueError, match="loopback"):
        service.start(bind_host="0.0.0.0")  # noqa: S104 - the refusal under test
