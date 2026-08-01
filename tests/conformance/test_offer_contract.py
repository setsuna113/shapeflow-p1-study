"""The offer contract's ten obligations, checked against every host.

Each test names the obligation it enforces (C1..C10 in `OFFER_INTERFACE_v1.md`). The document is
the contract; these are the assertions that make it binding.

The last test in this file is the one that matters most: it runs the suite's own logic against a
deliberately broken broker and requires it to FAIL. A conformance suite nothing can fail proves
only that it was written after the implementation.
"""

from __future__ import annotations

import time

import pytest
from conformance.chaos_host import BrokenBroker, ChaosHost

from shapeflow.broker.driver import DEFAULT_SOLVE_BUDGET_NS, run_tick
from shapeflow.broker.hosts.simulated import SimulatedHost
from shapeflow.broker.reference import ReferenceBroker
from shapeflow.contracts.observability import ClosedFeatureMap, UnobservableFeature
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
    SchedulerSnapshot,
)
from shapeflow.runtime.request_tags import OpClass


def _rd(op, *, uncached=100, cached=0, decode=200):
    return RequestDescriptor(
        op_class=op, prompt_sha256="a" * 64, max_tokens=512,
        est_uncached_prefill=uncached, est_cached_prefill=cached, est_decode=decode,
    )


def h_offer(item_id="H1", *, pages=4, with_p1=True, selector_cost=600):
    return Offer(
        item_id=item_id, boundary=Boundary.H, unit_id=f"turn-{item_id}", submitted_at_ns=0,
        p0=P0Plan(requests=tuple(_rd(OpClass.PAGE_P0_SUMMARY) for _ in range(pages))),
        p1=P1Plan(
            selector=_rd(OpClass.PAGE_P1_SELECTOR_LOCAL, uncached=selector_cost, decode=60),
            render=RenderDescriptor(est_cpu_ms=3.0, candidate_count=48),
        ) if with_p1 else None,
    )


def c_offer(item_id="C1"):
    return Offer(
        item_id=item_id, boundary=Boundary.C, unit_id=f"close-{item_id}", submitted_at_ns=0,
        p0=P0Plan(requests=(_rd(OpClass.COMPRESSOR_P0, uncached=4000, decode=1200),)),
        p1=P1Plan(
            selector=_rd(OpClass.COMPRESSOR_P1_SELECTOR, uncached=4200, decode=80),
            render=RenderDescriptor(est_cpu_ms=5.0, candidate_count=120),
        ),
    )


@pytest.fixture(params=["simulated", "chaos-benign"])
def host_factory(request):
    """Every obligation is checked on both a cooperative and an adversarial host.

    The adversarial host is configured benignly here -- its hostile knobs are exercised by the
    tests that target them -- so that the shared obligations are genuinely checked twice against
    different implementations rather than once against one.
    """
    if request.param == "simulated":
        return lambda offers: SimulatedHost(offers=list(offers))
    return lambda offers: ChaosHost(
        offers=list(offers), gang_schedule=False, reverse_dispatch=False,
        materialize_partially=False,
    )


# --- C1: no form mixing within an item -------------------------------------------------------


def test_c1_an_item_is_published_under_exactly_one_form(host_factory):
    host = host_factory([h_offer(), c_offer()])
    run_tick(host, ReferenceBroker(capacity_tokens=100_000))
    for item_id in ("H1", "C1"):
        forms = host.forms_of(item_id)
        assert len(forms) == 1, f"{item_id} was published under {forms}; a hybrid batch is barred"


def test_c1_a_mixed_form_decision_is_unconstructible():
    """The type refuses it, so no host can be asked to produce one."""
    with pytest.raises(ContractViolation, match="not admitted"):
        Decision(admitted=("H1",), form={"H1": Form.P0, "H2": Form.P1}, deferred=(),
                 snapshot_version=SimulatedHost().take_snapshot().version, solve_ns=0)


# --- C2: whole-batch fallback ----------------------------------------------------------------


def test_c2_degradation_takes_every_item_to_p0_not_some_of_them():
    """Exhausted retries degrade the entire item set to P0, never part of it.

    Adversarial-host only: making materialization fail is a hostile act, and the cooperative
    host has no knob for it. Parameterizing this over both hosts would have silently tested
    nothing on the cooperative one.
    """
    host = ChaosHost(offers=[h_offer(), h_offer("H2")], stale_until_attempt=2,
                     gang_schedule=False, reverse_dispatch=False)
    outcome = run_tick(host, ReferenceBroker(capacity_tokens=100_000), max_retries=1)
    assert outcome.fail_closed
    assert outcome.committed, "the degradation itself should have committed"
    assert set(outcome.decision.form.values()) == {Form.P0}, "degradation must be P0 for all"
    assert set(outcome.decision.admitted) == {"H1", "H2"}, "no item may vanish from the ITT set"
    for item_id in ("H1", "H2"):
        assert host.forms_of(item_id) == {Form.P0}


def test_c2_when_even_the_degradation_cannot_commit_the_work_stays_pending():
    """An engine that never accepts anything must not consume the work.

    Found by the adversarial host: the driver originally let this case raise, which would have
    made the caller's loop responsible for telling an unreachable engine apart from a broker
    bug, and would have dropped the offers on the floor between the two.
    """
    host = ChaosHost(offers=[h_offer(), h_offer("H2")], stale_until_attempt=99,
                     gang_schedule=False, reverse_dispatch=False)
    outcome = run_tick(host, ReferenceBroker(capacity_tokens=100_000), max_retries=1)
    assert not outcome.committed
    assert outcome.fail_closed and "remains pending" in outcome.reason
    assert host.dispatches == [], "nothing may be materialized when every attempt was stale"
    assert {o.item_id for o in host.pending()} == {"H1", "H2"}, (
        "the offers were consumed despite never running; they would leave the ITT denominator")


# --- C3: no form is started before a decision ------------------------------------------------


def test_c3_reading_pending_offers_starts_nothing(host_factory):
    host = host_factory([h_offer()])
    assert host.pending()
    host.take_snapshot()
    assert host.dispatches == [], "an offer was materialized before any decision was made"


# --- C4: atomicity ---------------------------------------------------------------------------


def test_c4_partial_materialization_is_visible_and_rejected():
    """A host that commits only part of a decision must be detectable, not silently tolerated."""
    host = ChaosHost(offers=[h_offer(), h_offer("H2")], materialize_partially=True,
                     gang_schedule=False, reverse_dispatch=False)
    outcome = run_tick(host, ReferenceBroker(capacity_tokens=100_000))
    left = host.uncommitted(outcome.decision)
    assert left, "this host was configured to materialize partially; the test double is broken"
    # The obligation is on the host, and the driver's result reports exactly what was committed,
    # so the discrepancy is observable rather than hidden inside a success.
    assert set(outcome.result.materialized) != set(outcome.decision.admitted)


# --- C5: no gang scheduling ------------------------------------------------------------------


def test_c5_p0_sub_requests_are_independently_scheduled(host_factory):
    host = host_factory([h_offer(pages=4)])
    run_tick(host, ReferenceBroker(capacity_tokens=100_000, allow_p1=False))
    requests = host.requests_for("H1")
    assert len(requests) == 4, "a P0 plan of 4 pages must produce 4 sub-requests"
    assert not any(d.gang_scheduled for d in requests), (
        "sub-requests were gang-scheduled; native continuous batching must schedule them "
        "independently or the measured throughput is an artifact of the harness"
    )


def test_c5_a_host_that_gangs_is_detected():
    """The check must be able to fail: a ganging host is caught rather than passing quietly."""
    host = ChaosHost(offers=[h_offer(pages=4)], gang_schedule=True, reverse_dispatch=False)
    run_tick(host, ReferenceBroker(capacity_tokens=100_000, allow_p1=False))
    assert any(d.gang_scheduled for d in host.requests_for("H1"))


# --- C6/C7: staleness, retry, fail closed ----------------------------------------------------


def test_c6_a_stale_snapshot_retries_the_whole_item_with_nothing_partial():
    host = ChaosHost(offers=[h_offer()], stale_until_attempt=1,
                     gang_schedule=False, reverse_dispatch=False)
    outcome = run_tick(host, ReferenceBroker(capacity_tokens=100_000), max_retries=3)
    assert outcome.stale_retries == 1
    assert outcome.committed, "the retry should have succeeded on the second attempt"
    assert host.dispatch_count("H1") == 1, "the stale attempt must have materialized nothing"


def test_c6_an_epoch_change_is_not_mistaken_for_a_current_tick():
    """Version identity includes the epoch, so a restarted engine cannot alias a live tick."""
    host = SimulatedHost(offers=[h_offer()])
    before = host.take_snapshot().version
    host.restart_engine()
    after = host.take_snapshot().version
    assert before.tick == after.tick, "the fixture should reproduce the tick counter reset"
    assert before != after, "same tick in a new epoch compared equal; staleness would never fire"


def test_c7_exhausted_retries_degrade_fail_closed_and_say_why():
    host = ChaosHost(offers=[h_offer()], stale_until_attempt=2,
                     gang_schedule=False, reverse_dispatch=False)
    outcome = run_tick(host, ReferenceBroker(capacity_tokens=100_000), max_retries=1)
    assert outcome.fail_closed
    assert outcome.decision.form["H1"] is Form.P0
    assert "stale" in outcome.reason, "a fail-closed degradation must record its reason"


# --- C8: the solve budget --------------------------------------------------------------------


class _SlowBroker:
    def __init__(self, sleep_ns: int) -> None:
        self.sleep_ns = sleep_ns

    def decide(self, offers, snapshot: SchedulerSnapshot, budget_ns: int) -> Decision:
        end = time.perf_counter_ns() + self.sleep_ns
        while time.perf_counter_ns() < end:
            pass
        ids = tuple(o.item_id for o in offers)
        return Decision(admitted=ids, form={i: Form.P1 for i in ids}, deferred=(),
                        snapshot_version=snapshot.version, solve_ns=0, reason="slow")


def test_c8_an_overrun_decision_is_discarded_not_applied():
    host = SimulatedHost(offers=[h_offer()])
    outcome = run_tick(host, _SlowBroker(sleep_ns=8_000_000), budget_ns=DEFAULT_SOLVE_BUDGET_NS)
    assert outcome.budget_overrun
    # The slow broker asked for P1; the discarded decision must not be the one applied.
    assert outcome.decision.form["H1"] is Form.P0
    assert outcome.decision.fail_closed


def test_c8_the_reference_broker_fits_the_budget_at_p99():
    offers = [h_offer(f"H{i}") for i in range(64)]
    snapshot = SimulatedHost(offers=offers).take_snapshot()
    broker = ReferenceBroker(capacity_tokens=50_000)
    samples = []
    for _ in range(200):
        start = time.perf_counter_ns()
        broker.decide(offers, snapshot, DEFAULT_SOLVE_BUDGET_NS)
        samples.append(time.perf_counter_ns() - start)
    samples.sort()
    p99 = samples[int(0.99 * len(samples)) - 1]
    assert p99 <= DEFAULT_SOLVE_BUDGET_NS, (
        f"p99 solve {p99}ns exceeds the {DEFAULT_SOLVE_BUDGET_NS}ns budget over 64 offers")


# --- C9: observability -----------------------------------------------------------------------


def test_c9_a_decision_cannot_read_an_unregistered_feature():
    with pytest.raises(UnobservableFeature, match="observability contract"):
        ClosedFeatureMap({"engine.how_long_will_this_take_ms": 12})


def test_c9_the_registry_excludes_the_offers_own_outcome():
    from shapeflow.contracts.observability import OBSERVABLE

    for banned in ("offer.realized_service_ns", "offer.evidence_lost", "offer.future_quality"):
        assert banned not in OBSERVABLE


# --- C10: determinism ------------------------------------------------------------------------


def test_c10_identical_inputs_give_a_byte_identical_decision(host_factory):
    offers = [h_offer(f"H{i}") for i in range(8)] + [c_offer()]
    digests = set()
    for _ in range(5):
        host = host_factory(offers)
        outcome = run_tick(host, ReferenceBroker(capacity_tokens=6000))
        digests.add(outcome.decision.digest)
    assert len(digests) == 1, f"the same input produced {len(digests)} different decisions"


def test_c10_the_digest_ignores_how_long_the_solver_took():
    snapshot = SimulatedHost().take_snapshot()
    kwargs = dict(admitted=("H1",), form={"H1": Form.P0}, deferred=(),
                  snapshot_version=snapshot.version)
    assert Decision(solve_ns=1, **kwargs).digest == Decision(solve_ns=999_999, **kwargs).digest


# --- the driver refuses decisions that name work nobody offered ------------------------------


def test_a_decision_naming_an_unoffered_item_is_refused():
    host = SimulatedHost(offers=[h_offer()])
    with pytest.raises(ContractViolation, match="not offered"):
        run_tick(host, BrokenBroker(violation="unoffered_item"))


def test_a_decision_choosing_a_form_that_was_never_offered_is_refused():
    host = SimulatedHost(offers=[h_offer(with_p1=False)])
    with pytest.raises(ContractViolation, match="offered no P1 plan"):
        run_tick(host, BrokenBroker(violation="p1_never_offered"))


# --- the suite must be able to fail ----------------------------------------------------------


def test_the_suite_is_not_vacuous():
    """A broken broker must be caught by the checks above, not slip through them.

    Without this, every other test in the file could be passing because the contract was written
    to describe whatever the implementation already did.
    """
    caught = []
    for violation in ("unoffered_item", "p1_never_offered"):
        host = SimulatedHost(offers=[h_offer(with_p1=False)])
        try:
            run_tick(host, BrokenBroker(violation=violation))
        except ContractViolation:
            caught.append(violation)
    assert len(caught) == 2, (
        f"only {caught} were caught; a conformance suite that cannot fail is not a suite")
