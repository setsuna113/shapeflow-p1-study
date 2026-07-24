"""Budget as admission control.

The rule the plan is adamant about: **reserve worst-case before dispatch, settle actual
after.** You never call an external API and then check whether you could afford it. So
the only way to spend is to first win a reservation, atomically, against every resource
the call touches (requests, credits, tokens, dollars, GPU-seconds, disk). If any single
cap would be exceeded, the whole group is refused and nothing is dispatched -- the work
item goes ``BLOCKED_BUDGET`` instead.

A reservation is worst-case on purpose. After the response comes back with real usage we
:meth:`settle` at the actual amount and the surplus is released. Two failure shapes get
special, deliberately pessimistic handling:

- :meth:`release` -- the call provably never went out (reservation refused downstream, or
  a pre-send error). The reservation is returned in full.
- :meth:`keep_worst_case` -- the call was sent and then contact was lost (timeout after
  send). We cannot know whether it was billed, so we settle at the *reserved* worst-case
  rather than assume it was free. Under-counting a possible charge is the one error this
  module refuses to make.

The accounting invariant, preserved by every operation:
``reserved_total >= 0`` and ``settled_total >= 0`` and
``reserved_total + settled_total <= cap``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..hashing import derive_id
from .ledger import Ledger

__all__ = ["Budget", "BudgetExceeded", "ReservationGroup"]


class BudgetExceeded(RuntimeError):
    """A reservation was refused because a cap would be exceeded. Carries the offending
    resource so the caller can log a precise incident and block the right work item."""

    def __init__(self, resource: str, requested: float, available: float) -> None:
        super().__init__(
            f"budget for {resource!r} exhausted: requested {requested}, available {available}"
        )
        self.resource = resource
        self.requested = requested
        self.available = available


@dataclass(frozen=True)
class ReservationGroup:
    group_id: str
    amounts: Mapping[str, float]


class Budget:
    """Reservation accounting layered on the ledger's SQLite connection."""

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger

    def _conn(self):
        return self._ledger.raw_connection

    def _now(self) -> float:
        return self._ledger.now()

    def ensure_account(self, resource: str, cap: float) -> None:
        """Register (or update) a cap. Called once per resource at launch from the frozen
        budget config; the effective cap may only be tightened, never raised, elsewhere."""
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                cur.execute(
                    "INSERT INTO budget_accounts(resource, cap, reserved_total, settled_total,"
                    " updated_at) VALUES (?,?,0,0,?)"
                    " ON CONFLICT(resource) DO UPDATE SET cap=excluded.cap, updated_at=excluded.updated_at",
                    (resource, cap, self._now()),
                )
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise

    def available(self, resource: str) -> float:
        with self._ledger.lock:
            row = self._conn().execute(
                "SELECT cap, reserved_total, settled_total FROM budget_accounts WHERE resource=?",
                (resource,),
            ).fetchone()
        if row is None:
            return 0.0
        return row["cap"] - row["reserved_total"] - row["settled_total"]

    def reserve(
        self,
        amounts: Mapping[str, float],
        *,
        work_key: Optional[str] = None,
        external_call_id: Optional[str] = None,
    ) -> ReservationGroup:
        """Atomically reserve worst-case ``amounts`` across resources, all-or-nothing.

        Raises :class:`BudgetExceeded` (rolling back every partial reservation) if any cap
        would be crossed. On success the resources are debited and a group handle returned.
        """
        if not amounts:
            raise ValueError("reserve requires at least one resource")
        group_id = external_call_id or derive_id(
            "reservation_group",
            {"work_key": work_key or "-", "amounts": dict(sorted(amounts.items()))},
        )
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                for resource, amount in amounts.items():
                    if amount < 0:
                        raise ValueError(f"negative reservation for {resource}")
                    acct = cur.execute(
                        "SELECT cap, reserved_total, settled_total FROM budget_accounts"
                        " WHERE resource=?",
                        (resource,),
                    ).fetchone()
                    if acct is None:
                        raise BudgetExceeded(resource, amount, 0.0)
                    avail = acct["cap"] - acct["reserved_total"] - acct["settled_total"]
                    # Strict admission: the reservation must fit entirely within headroom.
                    if amount > avail:
                        raise BudgetExceeded(resource, amount, avail)
                    cur.execute(
                        "UPDATE budget_accounts SET reserved_total=reserved_total+?, updated_at=?"
                        " WHERE resource=?",
                        (amount, self._now(), resource),
                    )
                    reservation_id = derive_id(
                        "reservation", {"group": group_id, "resource": resource}
                    )
                    cur.execute(
                        "INSERT OR REPLACE INTO budget_reservations(reservation_id, resource,"
                        " amount, state, settled_amount, work_key, external_call_id, created_at,"
                        " updated_at) VALUES (?,?,?,'RESERVED',NULL,?,?,?,?)",
                        (
                            reservation_id,
                            resource,
                            amount,
                            work_key,
                            external_call_id,
                            self._now(),
                            self._now(),
                        ),
                    )
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise
        return ReservationGroup(group_id=group_id, amounts=dict(amounts))

    def settle(self, group: ReservationGroup, actuals: Mapping[str, float]) -> None:
        """Settle each reservation at its actual usage; release the surplus. A resource
        missing from ``actuals`` settles at 0 (full release)."""
        self._resolve(group, lambda resource, reserved: max(0.0, actuals.get(resource, 0.0)))

    def keep_worst_case(self, group: ReservationGroup) -> None:
        """Settle every reservation at its full reserved amount -- used when a call was
        sent and its outcome is unknown, so a possible charge is never under-counted."""
        self._resolve(group, lambda resource, reserved: reserved)

    def release(self, group: ReservationGroup) -> None:
        """Return every reservation in full -- the call provably never went out."""
        self._resolve(group, lambda resource, reserved: 0.0)

    def _resolve(self, group: ReservationGroup, actual_fn) -> None:
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                rows = cur.execute(
                    "SELECT reservation_id, resource, amount, state FROM budget_reservations"
                    " WHERE external_call_id=? OR reservation_id IN ("
                    "  SELECT reservation_id FROM budget_reservations WHERE reservation_id=?)",
                    (group.group_id, group.group_id),
                ).fetchall()
                # Fall back to deriving the per-resource ids when no external_call_id link.
                if not rows:
                    rows = []
                    for resource in group.amounts:
                        rid = derive_id("reservation", {"group": group.group_id, "resource": resource})
                        r = cur.execute(
                            "SELECT reservation_id, resource, amount, state FROM"
                            " budget_reservations WHERE reservation_id=?",
                            (rid,),
                        ).fetchone()
                        if r is not None:
                            rows.append(r)
                for r in rows:
                    if r["state"] != "RESERVED":
                        continue  # already resolved; idempotent
                    reserved = r["amount"]
                    actual = min(reserved, actual_fn(r["resource"], reserved))
                    cur.execute(
                        "UPDATE budget_accounts SET reserved_total=reserved_total-?,"
                        " settled_total=settled_total+?, updated_at=? WHERE resource=?",
                        (reserved, actual, self._now(), r["resource"]),
                    )
                    cur.execute(
                        "UPDATE budget_reservations SET state='SETTLED', settled_amount=?,"
                        " updated_at=? WHERE reservation_id=?",
                        (actual, self._now(), r["reservation_id"]),
                    )
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise
