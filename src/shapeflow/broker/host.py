"""The two protocols a broker deployment must satisfy.

:class:`BrokerHost` is the only thing that varies between the simulation, a proxy in front of the
engine, and a scheduler-resident extension. :class:`Broker` is the policy under study.

Keeping them separate is what makes the comparison in Freeze-1 §7 meaningful: swapping the policy
must not change the mechanism, and swapping the host must not change the policy.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

from ..contracts.offer import Decision, MaterializeResult, Offer, SchedulerSnapshot

__all__ = ["BrokerHost", "Broker"]


@runtime_checkable
class BrokerHost(Protocol):
    """Where offers come from, what the engine looks like, and how work is committed."""

    def take_snapshot(self) -> SchedulerSnapshot:
        """Capture engine state for this tick, with a version and a real timestamp."""

    def pending(self) -> Sequence[Offer]:
        """Offers awaiting a decision. No form has been started for any of them."""

    def materialize(self, decision: Decision, snapshot: SchedulerSnapshot) -> MaterializeResult:
        """Commit the decision: materialize only the chosen form for each admitted item.

        Must raise :class:`~shapeflow.contracts.offer.StaleSnapshot` **having materialized
        nothing** if the snapshot is no longer current. Partial materialization is the one
        outcome the contract does not permit: an item half-committed belongs to neither arm and
        cannot be accounted in either.

        When a P0 form is materialized, its N sub-requests must enter the native queue as
        independently schedulable work. The host may not gang-schedule them, block one on
        another, or require them to start together.
        """


@runtime_checkable
class Broker(Protocol):
    """The policy: given offers and engine state, decide admission and form together."""

    def decide(
        self, offers: Sequence[Offer], snapshot: SchedulerSnapshot, budget_ns: int
    ) -> Decision:
        """Return the tick's decision within ``budget_ns``.

        May read only features registered in the observability contract. Must be deterministic in
        ``(offers, snapshot, seed)``. Overrunning the budget is handled by the driver, which
        discards the decision -- so an implementation should return its best answer rather than
        an exception when time runs short.
        """
