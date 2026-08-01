"""A host that tries to break every obligation the offer contract states.

The conformance suite's real risk is not that an implementation fails it -- it is that the
contract was written to describe the implementation, so everything passes and nothing is
constrained. S2 would then discover, against a real engine, that the contract said nothing.

So the positive obligations are checked against the cooperative host, and the negative ones are
checked here, against a host that actively misbehaves:

- the snapshot version moves between ``take_snapshot`` and ``materialize``, always;
- it will materialize *part* of a decision if the driver lets it;
- it returns snapshots from a previous engine epoch;
- it reorders and delays sub-requests, and will gang-schedule siblings if permitted.

Deliberately in the test tree rather than under ``src/``: a host designed to violate the contract
must not be importable from production code by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from shapeflow.broker.hosts.simulated import Dispatch, SimulatedHost
from shapeflow.contracts.offer import (
    Decision,
    Form,
    MaterializeResult,
    Offer,
    SchedulerSnapshot,
    SnapshotVersion,
    StaleSnapshot,
)

__all__ = ["ChaosHost", "PartialMaterialization"]


class PartialMaterialization(AssertionError):
    """The driver allowed a decision to be materialized in pieces."""


@dataclass
class ChaosHost(SimulatedHost):
    """Adversarial host. Every knob defaults to hostile."""

    #: Report the snapshot stale on every materialize until this many attempts have been made.
    stale_until_attempt: int = 0
    #: Hand back a snapshot whose epoch is one the engine has already left.
    serve_stale_epoch: bool = False
    #: Materialize only the first admitted item, leaving the rest uncommitted.
    materialize_partially: bool = False
    #: Record siblings as gang-scheduled, so a driver that permits it is visible in the log.
    gang_schedule: bool = True
    #: Emit sub-requests in reverse order.
    reverse_dispatch: bool = True

    _attempts: int = field(default=0, init=False)

    def take_snapshot(self) -> SchedulerSnapshot:
        snapshot = super().take_snapshot()
        if self.serve_stale_epoch:
            # A snapshot from an epoch the engine has left. If version identity ignored the
            # epoch, this would compare equal to a current tick and staleness detection would
            # be silently disabled -- passing perfectly while checking nothing.
            return SchedulerSnapshot(
                version=SnapshotVersion(epoch="an-epoch-ago", tick=snapshot.version.tick),
                taken_at_ns=snapshot.taken_at_ns,
                features=snapshot.features,
            )
        return snapshot

    def materialize(self, decision: Decision, snapshot: SchedulerSnapshot) -> MaterializeResult:
        self._attempts += 1
        if self._attempts <= self.stale_until_attempt:
            raise StaleSnapshot(snapshot.version)

        admitted = list(decision.admitted)
        if self.materialize_partially and len(admitted) > 1:
            # If the driver tolerates this, a caller can never know whether an item ran.
            admitted = admitted[:1]

        by_id = {o.item_id: o for o in self.offers}
        dispatched: dict[str, int] = {}
        for item_id in admitted:
            offer, form = by_id[item_id], decision.form[item_id]
            plan = offer.plan_for(form)
            requests = list(plan.requests) if form is Form.P0 else [plan.selector]
            ordered = list(reversed(requests)) if self.reverse_dispatch else requests
            for ordinal, request in enumerate(ordered):
                self.dispatches.append(Dispatch(
                    item_id=item_id, form=form, ordinal=ordinal,
                    op_class=request.op_class.value,
                    gang_scheduled=self.gang_schedule and len(requests) > 1,
                ))
            dispatched[item_id] = len(requests)

        self.committed.append(decision)
        self.offers = [o for o in self.offers if o.item_id not in set(admitted)]
        return MaterializeResult(materialized=tuple(admitted), dispatched=dispatched)

    def uncommitted(self, decision: Decision) -> Sequence[str]:
        """Admitted items this host declined to materialize."""
        done = {d.item_id for d in self.dispatches}
        return tuple(i for i in decision.admitted if i not in done)


@dataclass
class BrokenBroker:
    """A broker that violates the contract on purpose, to prove the suite can fail.

    Its two faults are the two most plausible real ones: assigning a form to an item that was
    never admitted, and choosing P1 for an item that offered no P1 plan. Both produce a
    well-formed-looking decision, which is exactly why they need a check rather than a review.
    """

    violation: str = "form_without_admission"

    def decide(self, offers: Sequence[Offer], snapshot: SchedulerSnapshot, budget_ns: int):
        ids = tuple(o.item_id for o in offers)
        if self.violation == "p1_never_offered":
            # Object construction is legal here; the driver has to be the thing that refuses.
            return Decision(
                admitted=ids, form={i: Form.P1 for i in ids}, deferred=(),
                snapshot_version=snapshot.version, solve_ns=0, reason="broken: forces P1")
        if self.violation == "unoffered_item":
            return Decision(
                admitted=("ghost",), form={"ghost": Form.P0}, deferred=(),
                snapshot_version=snapshot.version, solve_ns=0, reason="broken: invents work")
        raise AssertionError(f"unknown violation {self.violation!r}")
