"""A host with no engine: scripted snapshots and an in-memory commit log.

This is B5's substrate, so it has to be honest about the two things B5 claims to measure --
which form each item got, and how many sub-requests that produced. It therefore records
dispatches individually rather than per item, which is what makes "P0's N sub-requests are
independent" checkable rather than assumed.

It is also the host the conformance suite uses to prove the *positive* obligations. The
adversarial host proves the negative ones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ...contracts.observability import ClosedFeatureMap
from ...contracts.offer import (
    Decision,
    Form,
    MaterializeResult,
    Offer,
    SchedulerSnapshot,
    SnapshotVersion,
    StaleSnapshot,
)
from ...contracts.versions import assert_contract

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["SimulatedHost", "Dispatch"]


@dataclass(frozen=True)
class Dispatch:
    """One materialized sub-request. Recorded per request, never per item."""

    item_id: str
    form: Form
    ordinal: int
    op_class: str
    #: Independently schedulable. The simulated engine never forces siblings to start together,
    #: so a driver that relied on gang scheduling would produce a different log here.
    gang_scheduled: bool = False


@dataclass
class SimulatedHost:
    """Deterministic host with an explicit queue and a scripted tick counter."""

    offers: list[Offer] = field(default_factory=list)
    epoch: str = "sim-epoch-0"
    tick: int = 0
    now_ns: int = 0
    features: dict = field(default_factory=lambda: {
        "engine.num_requests_running": 0,
        "engine.num_requests_waiting": 0,
        "engine.kv_cache_usage_perc": 0.0,
        "local.inflight_total": 0,
    })
    dispatches: list[Dispatch] = field(default_factory=list)
    committed: list[Decision] = field(default_factory=list)
    #: Ticks at which materialization must report the snapshot as stale.
    stale_at_ticks: set = field(default_factory=set)

    def take_snapshot(self) -> SchedulerSnapshot:
        self.tick += 1
        self.now_ns += 1000
        return SchedulerSnapshot(
            version=SnapshotVersion(epoch=self.epoch, tick=self.tick),
            taken_at_ns=self.now_ns,
            features=ClosedFeatureMap(dict(self.features)),
        )

    def pending(self) -> Sequence[Offer]:
        return tuple(self.offers)

    def materialize(self, decision: Decision, snapshot: SchedulerSnapshot) -> MaterializeResult:
        if snapshot.version.tick in self.stale_at_ticks:
            # Nothing is written before this check, so "materialized nothing" is structural
            # rather than a matter of having remembered to roll back.
            raise StaleSnapshot(snapshot.version)

        by_id = {o.item_id: o for o in self.offers}
        dispatched: dict[str, int] = {}
        for item_id in decision.admitted:
            offer, form = by_id[item_id], decision.form[item_id]
            plan = offer.plan_for(form)
            requests = plan.requests if form is Form.P0 else (plan.selector,)
            for ordinal, request in enumerate(requests):
                self.dispatches.append(Dispatch(
                    item_id=item_id, form=form, ordinal=ordinal,
                    op_class=request.op_class.value,
                ))
            dispatched[item_id] = len(requests)

        self.committed.append(decision)
        self.offers = [o for o in self.offers if o.item_id not in set(decision.admitted)]
        return MaterializeResult(
            materialized=tuple(decision.admitted),
            dispatched=dispatched,
            degraded_to_p0=tuple(i for i in decision.admitted
                                 if decision.fail_closed and decision.form[i] is Form.P0),
        )

    # --- inspection helpers used by the conformance suite -------------------------------------

    def forms_of(self, item_id: str) -> set[Form]:
        """Every form this item was ever dispatched under. More than one is a contract breach."""
        return {d.form for d in self.dispatches if d.item_id == item_id}

    def dispatch_count(self, item_id: str) -> int:
        return sum(1 for d in self.dispatches if d.item_id == item_id)

    def requests_for(self, item_id: str) -> list[Dispatch]:
        return [d for d in self.dispatches if d.item_id == item_id]

    def restart_engine(self, new_epoch: Optional[str] = None) -> None:
        """Reset the tick counter under a new epoch, the way a real restart does."""
        self.epoch = new_epoch or f"{self.epoch}-restarted"
        self.tick = 0


assert_contract(CONTRACT, CONTRACT_SHA256)
