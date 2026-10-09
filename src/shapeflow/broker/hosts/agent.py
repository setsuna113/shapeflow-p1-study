"""A host whose offers come from live agents that wait, suspended, for the decision.

Each offer stands for an agent paused at a handoff point -- a page batch (H) or a researcher
close (C) -- with both futures priced and neither started. ``materialize`` does not send
anything itself: it commits the decision and *releases* the waiting agent, which then runs only
the chosen form through its own model path. That keeps the dispatch native. A P0 page batch is
vendor's own ``asyncio.gather`` of N summaries through a provider that does not serialize them,
so the siblings enter vLLM's queue independently (C5) without this host pretending to be the
thing that sent them.

An item moves PENDING -> DECIDED, or PENDING -> WITHDRAWN, and a decided item is acknowledged
once its agent has run it. Withdrawal is how an agent that stopped waiting gets a definite
answer: if the item is still pending it is withdrawn and the agent runs P0 fail-closed; if it was
decided first, the decision stands and the agent runs that. Either way the item is run exactly
once.

Snapshots and the staleness gate are :mod:`.engine_state`'s, so this host and ``ProxyHost``
judge staleness identically. There is no ``local.inflight_total`` here: the agents dispatch,
not the host, so an in-flight count kept here would be a guess presented as an exact feature.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from ...contracts.offer import (
    ContractViolation,
    Decision,
    Form,
    MaterializeResult,
    Offer,
    SchedulerSnapshot,
)
from ...contracts.versions import assert_contract
from ..codec import offer_digest
from .engine_state import DEFAULT_MAX_SNAPSHOT_AGE_NS, EngineSnapshotGate, SnapshotRecord
from .simulated import Dispatch

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["AgentHost", "PENDING", "DECIDED", "WITHDRAWN", "UnknownItem"]

PENDING, DECIDED, WITHDRAWN = "PENDING", "DECIDED", "WITHDRAWN"


class UnknownItem(KeyError):
    """No offer with this id was ever submitted to this host."""


@dataclass
class _Item:
    offer: Offer
    digest: str
    received_ns: int
    state: str = PENDING
    ticket: dict | None = None
    withdrawn: dict | None = None
    acks: list = field(default_factory=list)


class AgentHost:
    """``BrokerHost`` for offers submitted by agents that block on the outcome."""

    def __init__(
        self,
        *,
        scrape: Callable[[], str],
        clock_ns: Callable[[], int] = time.time_ns,
        max_snapshot_age_ns: int = DEFAULT_MAX_SNAPSHOT_AGE_NS,
    ) -> None:
        self._scrape = scrape
        self._clock_ns = clock_ns
        self._gate = EngineSnapshotGate(clock_ns=clock_ns,
                                        max_snapshot_age_ns=max_snapshot_age_ns)
        self._lock = threading.Lock()
        self._changed = threading.Condition(self._lock)
        self._items: dict[str, _Item] = {}
        self.dispatches: list[Dispatch] = []
        self.committed: list[Decision] = []

    # --- agent-facing -------------------------------------------------------------------------

    def submit(self, offer: Offer) -> dict:
        """Queue an offer. Idempotent for identical content; a different offer under a known id
        is refused, since the agent would be waiting on a decision about a plan it never priced."""
        digest = offer_digest(offer)
        with self._lock:
            known = self._items.get(offer.item_id)
            if known is not None:
                if known.digest != digest:
                    raise ContractViolation(
                        f"{offer.item_id} was already submitted with different content")
                return self._status(known)
            item = _Item(offer=offer, digest=digest, received_ns=self._clock_ns())
            self._items[offer.item_id] = item
            return self._status(item)

    def wait(self, item_id: str, timeout_s: float) -> dict:
        """Block until the item leaves PENDING or ``timeout_s`` passes; return its status."""
        deadline = time.monotonic() + max(0.0, timeout_s)
        with self._lock:
            item = self._get(item_id)
            while item.state == PENDING:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._changed.wait(remaining)
            return self._status(item)

    def withdraw(self, item_id: str, reason: str) -> dict:
        """Withdraw a still-pending item. A decided item stays decided: the decision wins."""
        with self._lock:
            item = self._get(item_id)
            if item.state == PENDING:
                item.state = WITHDRAWN
                item.withdrawn = {"reason": reason, "at_ns": self._clock_ns()}
                self._changed.notify_all()
            return self._status(item)

    def ack(self, item_id: str, outcome: dict) -> dict:
        """Record what the agent actually ran. Kept as evidence, never consulted by a decision."""
        with self._lock:
            item = self._get(item_id)
            item.acks.append({**dict(outcome), "at_ns": self._clock_ns()})
            return self._status(item)

    def status(self, item_id: str) -> dict:
        with self._lock:
            return self._status(self._get(item_id))

    # --- BrokerHost ---------------------------------------------------------------------------

    def pending(self) -> Sequence[Offer]:
        with self._lock:
            items = [i for i in self._items.values() if i.state == PENDING]
        items.sort(key=lambda i: (i.offer.submitted_at_ns, i.offer.item_id))
        return tuple(i.offer for i in items)

    def take_snapshot(self) -> SchedulerSnapshot:
        observation = self._gate.observe(self._scrape)
        with self._lock:
            return self._gate.stamp(observation)

    def materialize(self, decision: Decision, snapshot: SchedulerSnapshot) -> MaterializeResult:
        with self._lock:
            # The gate runs before anything is written: a stale snapshot releases no agent and
            # every offer stays pending for the driver's whole-item retry.
            gate = self._gate.check(snapshot, decision.admitted)
            items = []
            for item_id in decision.admitted:
                item = self._items.get(item_id)
                if item is None or item.state != PENDING:
                    raise ContractViolation(
                        f"{item_id} was admitted but is not pending; a withdrawn item must never "
                        "be decided, or its agent would run P0 while the broker records P1")
                items.append(item)

            # --- commit: every admitted item is decided, and its agent released, together ---
            self._gate.passed(gate)
            self.committed.append(decision)
            committed_ns = self._clock_ns()
            dispatched: dict[str, int] = {}
            for item in items:
                form = decision.form[item.offer.item_id]
                plan = item.offer.plan_for(form)
                descriptors = plan.requests if form is Form.P0 else (plan.selector,)
                for ordinal, descriptor in enumerate(descriptors):
                    self.dispatches.append(Dispatch(
                        item_id=item.offer.item_id, form=form, ordinal=ordinal,
                        op_class=descriptor.op_class.value))
                dispatched[item.offer.item_id] = len(descriptors)
                item.state = DECIDED
                item.ticket = {
                    "form": form.value,
                    "decision_digest": decision.digest,
                    "snapshot": snapshot.version.content(),
                    "snapshot_taken_at_ns": snapshot.taken_at_ns,
                    "committed_ns": committed_ns,
                    "fail_closed": decision.fail_closed,
                    "reason": decision.reason,
                    "requests": len(descriptors),
                }
            self._changed.notify_all()

        return MaterializeResult(
            materialized=tuple(decision.admitted),
            dispatched=dispatched,
            degraded_to_p0=tuple(i for i in decision.admitted
                                 if decision.fail_closed and decision.form[i] is Form.P0),
        )

    # --- inspection ---------------------------------------------------------------------------

    @property
    def max_snapshot_age_ns(self) -> int:
        return self._gate.max_snapshot_age_ns

    @property
    def snapshots(self) -> list[SnapshotRecord]:
        return self._gate.snapshots

    @property
    def gate_log(self) -> list[dict]:
        return self._gate.gate_log

    def items(self) -> list[dict]:
        with self._lock:
            return [self._status(i) for i in self._items.values()]

    def forms_of(self, item_id: str) -> set[Form]:
        return {d.form for d in self.dispatches if d.item_id == item_id}

    def dispatch_count(self, item_id: str) -> int:
        return sum(1 for d in self.dispatches if d.item_id == item_id)

    def requests_for(self, item_id: str) -> list[Dispatch]:
        return [d for d in self.dispatches if d.item_id == item_id]

    # --- internals ----------------------------------------------------------------------------

    def _get(self, item_id: str) -> _Item:
        item = self._items.get(item_id)
        if item is None:
            raise UnknownItem(item_id)
        return item

    @staticmethod
    def _status(item: _Item) -> dict:
        return {
            "item_id": item.offer.item_id,
            "state": item.state,
            "digest": item.digest,
            "received_ns": item.received_ns,
            "ticket": dict(item.ticket) if item.ticket else None,
            "withdrawn": dict(item.withdrawn) if item.withdrawn else None,
            "acks": [dict(a) for a in item.acks],
        }


assert_contract(CONTRACT, CONTRACT_SHA256)
