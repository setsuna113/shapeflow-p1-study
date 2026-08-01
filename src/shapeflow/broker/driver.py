"""The six steps, implemented once.

Every host reuses this verbatim. If a host needed its own copy of this logic, the contract would
be a description of three separate systems that happen to resemble each other, and the B5
simulation would stop being evidence about the real one.

The ordering below is the contract's ordering and is load-bearing at three points:

- the budget is checked **after** the solver returns and the decision is discarded on overrun,
  because a late decision was computed against a snapshot the engine has already moved past;
- staleness retries the **whole item**, never part of it;
- retry exhaustion degrades to P0-only rather than dropping the work, because a dropped item
  would silently leave the ITT denominator.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional, Sequence

from ..contracts.offer import (
    ContractViolation,
    Decision,
    Form,
    MaterializeResult,
    Offer,
    SchedulerSnapshot,
    StaleSnapshot,
)
from ..contracts.versions import assert_contract
from .host import Broker, BrokerHost

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["TickOutcome", "run_tick", "DEFAULT_SOLVE_BUDGET_NS"]

#: Freeze-1 §1. The solver budget, not the whole critical path.
DEFAULT_SOLVE_BUDGET_NS = 2_000_000


@dataclass(frozen=True)
class TickOutcome:
    """What one tick did, including the ways it declined to do anything."""

    decision: Optional[Decision]
    result: Optional[MaterializeResult]
    attempts: int
    stale_retries: int
    budget_overrun: bool = False
    fail_closed: bool = False
    reason: str = ""

    @property
    def committed(self) -> bool:
        return self.result is not None


def _p0_only(offers: Sequence[Offer], snapshot: SchedulerSnapshot, reason: str) -> Decision:
    """The fail-closed decision: admit everything offered, all of it as P0.

    P0 is always offered, so this is always constructible. Deferring the work instead would be
    the other plausible degradation and is wrong: the offers still exist, and an item that
    vanished from both the admitted and the deferred set would leave the ITT denominator
    without anyone noticing.
    """
    ids = tuple(o.item_id for o in offers)
    return Decision(
        admitted=ids,
        form={i: Form.P0 for i in ids},
        deferred=(),
        snapshot_version=snapshot.version,
        solve_ns=0,
        fail_closed=True,
        reason=reason,
    )


def run_tick(
    host: BrokerHost,
    broker: Broker,
    *,
    budget_ns: int = DEFAULT_SOLVE_BUDGET_NS,
    max_retries: int = 2,
    clock_ns: Callable[[], int] = time.perf_counter_ns,
) -> TickOutcome:
    """Run one scheduler tick end to end.

    Returns rather than raises for every outcome the contract anticipates -- an empty queue, a
    budget overrun, exhausted retries. A tick that raised would make the caller's loop responsible
    for distinguishing "nothing to do" from "something went wrong", and those must stay distinct
    in the ledger.
    """
    stale_retries = 0
    attempts = 0

    while True:
        # 1. Offers exist without any form started. Reading them starts nothing.
        offers = list(host.pending())
        if not offers:
            return TickOutcome(decision=None, result=None, attempts=attempts,
                               stale_retries=stale_retries, reason="no offers pending")

        # 2. Versioned snapshot of engine state.
        snapshot = host.take_snapshot()

        # 3. One atomic choice over admission and form.
        attempts += 1
        started = clock_ns()
        decision = broker.decide(offers, snapshot, budget_ns)
        solve_ns = clock_ns() - started

        if solve_ns > budget_ns:
            # Discarded, not applied. The engine has moved on from the snapshot this was
            # computed against, so applying it would be acting on state that no longer exists.
            decision = _p0_only(
                offers, snapshot,
                f"solver overran budget: {solve_ns}ns > {budget_ns}ns; decision discarded")
            overrun = True
        else:
            overrun = False
            _assert_decision_is_answerable(decision, offers)

        # 4/5. Materialize only the chosen form. P0's sub-requests enter the native queue as
        # independent work; the host is contractually forbidden from ganging them.
        try:
            result = host.materialize(decision, snapshot)
        except StaleSnapshot:
            # 6. Nothing was materialized. Retry the whole item set, never part of it.
            stale_retries += 1
            if stale_retries > max_retries:
                fail = _p0_only(
                    offers, snapshot,
                    f"snapshot went stale {stale_retries} times; degrading to P0-only")
                try:
                    result = host.materialize(fail, host.take_snapshot())
                except StaleSnapshot:
                    # Even the degradation could not commit. Report the tick as uncommitted and
                    # leave the offers pending rather than raising: the work has not been done,
                    # and an exception here would make the caller's loop responsible for telling
                    # "the engine is unreachable" apart from "the broker has a bug". The offers
                    # were never consumed, so nothing has left the ITT denominator.
                    return TickOutcome(
                        decision=fail, result=None, attempts=attempts,
                        stale_retries=stale_retries, budget_overrun=overrun, fail_closed=True,
                        reason=(f"{fail.reason}; the degradation could not be materialized "
                                "either, so the work remains pending"))
                return TickOutcome(decision=fail, result=result, attempts=attempts,
                                   stale_retries=stale_retries, budget_overrun=overrun,
                                   fail_closed=True, reason=fail.reason)
            continue

        return TickOutcome(decision=decision, result=result, attempts=attempts,
                           stale_retries=stale_retries, budget_overrun=overrun,
                           fail_closed=decision.fail_closed, reason=decision.reason)


def _assert_decision_is_answerable(decision: Decision, offers: Sequence[Offer]) -> None:
    """Refuse a decision that names work nobody offered, or a form nobody offered for it.

    Checked here rather than trusted, because both mistakes produce a decision that looks
    perfectly well formed. Admitting an unknown id would materialize work with no offer behind
    it; choosing P1 for an item that offered only P0 would materialize a plan that was never
    priced, and its cost would land in the ledger attributed to a plan that does not exist.
    """
    by_id = {o.item_id: o for o in offers}
    named = set(decision.admitted) | set(decision.deferred)
    unknown = sorted(named - set(by_id))
    if unknown:
        raise ContractViolation(f"decision names items that were not offered: {unknown}")
    for item_id, form in decision.form.items():
        if form is Form.P1 and by_id[item_id].p1 is None:
            raise ContractViolation(
                f"item {item_id} was admitted as P1 but offered no P1 plan")


assert_contract(CONTRACT, CONTRACT_SHA256)
