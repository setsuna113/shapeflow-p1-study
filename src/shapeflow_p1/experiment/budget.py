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

__all__ = [
    "Budget",
    "BudgetExceeded",
    "BudgetCapRaised",
    "BudgetUnderReserved",
    "ReservationGroup",
]


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


class BudgetCapRaised(RuntimeError):
    """Someone tried to widen an authorized ceiling. Protocol §3.5: the effective budget
    may only be tightened, and any increase mints a new protocol SHA and a new approval.

    Raised from ``ensure_account``, which every provider start calls once per resource -- so a
    config carrying raised caps against a ledger that has not had them applied does not merely
    refuse a spend, it stops the provider from starting at all, three times, until the
    supervisor gives up. That happened here. The message therefore names the remedy, because
    the failure surfaces as a dead service rather than as a rejected request.
    """

    def __init__(self, resource: str, current: float, requested: float) -> None:
        super().__init__(
            f"cap for {resource!r} may not be raised from {current} to {requested}: a wider "
            "budget needs a new protocol version and a new approval, not an overwrite. "
            "Apply it deliberately first -- freeze an approval over the new config, then run "
            "`shapeflow-p1 authorize-budget-raise --reason ...` as the provider identity, "
            "which records the old and new ceilings against that approval. The ledger and the "
            "config must agree before the provider can start."
        )
        self.resource = resource
        self.current = current
        self.requested = requested


class BudgetUnderReserved(RuntimeError):
    """Actual usage exceeded the worst case that was reserved for it.

    This is an admission-control failure, not an accounting one: the real amount is still
    charged. It is raised so the caller records an incident, because a reservation that is
    not an upper bound means dispatch decisions were made against a number that was too
    small.
    """

    def __init__(self, resource: str, reserved: float, actual: float) -> None:
        super().__init__(
            f"{resource!r} settled at {actual} against a worst-case reservation of "
            f"{reserved}: the reservation was not an upper bound"
        )
        self.resource = resource
        self.reserved = reserved
        self.actual = actual


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
        """Register a cap, or tighten an existing one. Called once per resource at launch
        from the frozen budget config.

        A cap may only ever move down. Raising one mid-campaign would let a round spend
        past the amount its approval was granted against, and doing it by silently
        overwriting the row -- which is what this used to do -- leaves no evidence that the
        authorized ceiling ever changed.
        """
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                row = cur.execute(
                    "SELECT cap FROM budget_accounts WHERE resource=?", (resource,)
                ).fetchone()
                if row is None:
                    cur.execute(
                        "INSERT INTO budget_accounts(resource, cap, reserved_total,"
                        " settled_total, updated_at) VALUES (?,?,0,0,?)",
                        (resource, cap, self._now()),
                    )
                elif cap < row["cap"]:
                    cur.execute(
                        "UPDATE budget_accounts SET cap=?, updated_at=? WHERE resource=?",
                        (cap, self._now(), resource),
                    )
                elif cap > row["cap"]:
                    raise BudgetCapRaised(resource, row["cap"], cap)
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise

    def authorize_cap_raise(
        self, resource: str, cap: float, *, authorization: str, reason: str
    ) -> dict:
        """Raise one cap deliberately, leaving evidence that the ceiling moved.

        ``ensure_account`` refuses to raise, and that refusal is right: a cap that drifted
        upward as a side effect of loading a config would let a round spend past what its
        approval was granted against, with nothing in the ledger showing it had changed.

        A raise is nonetheless sometimes the correct answer -- a ceiling set before anything was
        measured can turn out to be unable to buy the artifact the campaign exists to produce.
        So it is available, but only as an explicit act that names its authorization and records
        an incident carrying the old and new values. The distinction being preserved is not
        "caps never rise", it is "a cap never rises without a trace".

        Refuses to *lower* here: that is ``ensure_account``'s job, and accepting both directions
        through one call is how an audited raise becomes an ordinary write.
        """
        if not authorization or not reason:
            raise ValueError("a cap raise must record its authorization and its reason")
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                row = cur.execute(
                    "SELECT cap, reserved_total, settled_total FROM budget_accounts"
                    " WHERE resource=?", (resource,),
                ).fetchone()
                # These raise out to the handler below, which is what rolls back. Rolling back
                # here as well left no transaction for it to undo and turned a clear refusal
                # into "cannot rollback - no transaction is active".
                if row is None:
                    raise BudgetCapRaised(resource, 0.0, cap)
                previous = float(row["cap"])
                if cap < previous:
                    raise ValueError(
                        f"{resource} cap {cap} is below the current {previous}; tightening goes "
                        "through ensure_account, not through an authorized raise"
                    )
                if cap == previous:
                    cur.execute("COMMIT;")
                    return {"resource": resource, "previous_cap": previous, "cap": cap,
                            "changed": False}
                cur.execute(
                    "UPDATE budget_accounts SET cap=?, updated_at=? WHERE resource=?",
                    (cap, self._now(), resource),
                )
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise
        self._ledger.record_incident(
            severity="WARNING", kind="budget_cap_raised",
            detail=(
                f"{resource}: {previous} -> {cap}; spent={row['settled_total']}; "
                f"authorization={authorization}; reason={reason}"
            ),
        )
        return {"resource": resource, "previous_cap": previous, "cap": cap, "changed": True,
                "settled": float(row["settled_total"])}

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
        attempt_id: Optional[str] = None,
    ) -> ReservationGroup:
        """Atomically reserve worst-case ``amounts`` across resources, all-or-nothing.

        Raises :class:`BudgetExceeded` (rolling back every partial reservation) if any cap
        would be crossed. On success the resources are debited and a group handle returned.
        """
        with self._ledger.transaction() as cur:
            return self.reserve_in(
                cur, amounts, work_key=work_key,
                external_call_id=external_call_id, attempt_id=attempt_id,
            )

    def reserve_in(
        self,
        cur,
        amounts: Mapping[str, float],
        *,
        work_key: Optional[str] = None,
        external_call_id: Optional[str] = None,
        attempt_id: Optional[str] = None,
    ) -> ReservationGroup:
        """:meth:`reserve` inside a transaction the caller already opened.

        The group is keyed on the **attempt**, not the logical call. Two dispatches of the
        same call are two charges and must hold two reservations; keying on the call is how
        a retry used to overwrite the first reservation row and lose the record of what it
        had already settled.
        """
        if not amounts:
            raise ValueError("reserve requires at least one resource")
        group_id = attempt_id or external_call_id
        if group_id is None:
            # An anonymous reservation still needs a distinct identity: two identical
            # reservations are two claims on the cap, and collapsing them onto one derived
            # id is how a reservation used to overwrite its predecessor.
            seq = cur.execute(
                "SELECT COUNT(*) c FROM budget_reservations"
            ).fetchone()["c"]
            group_id = derive_id(
                "reservation_group",
                {
                    "work_key": work_key or "-",
                    "amounts": dict(sorted(amounts.items())),
                    "seq": int(seq),
                },
            )
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
            # Plain INSERT: a duplicate reservation id means the same dispatch was
            # reserved twice, which is a bug we want to hear about rather than paper over.
            cur.execute(
                "INSERT INTO budget_reservations(reservation_id, resource,"
                " amount, state, settled_amount, work_key, external_call_id, attempt_id,"
                " created_at, updated_at) VALUES (?,?,?,'RESERVED',NULL,?,?,?,?,?)",
                (
                    reservation_id,
                    resource,
                    amount,
                    work_key,
                    external_call_id,
                    attempt_id,
                    self._now(),
                    self._now(),
                ),
            )
        return ReservationGroup(group_id=group_id, amounts=dict(amounts))

    def settle(self, group: ReservationGroup, actuals: Mapping[str, float]) -> list:
        """Settle each reservation at its actual usage; release the surplus. A resource
        missing from ``actuals`` settles at 0 (full release)."""
        with self._ledger.transaction() as cur:
            return self.settle_in(cur, group, actuals)

    def settle_in(self, cur, group: ReservationGroup, actuals: Mapping[str, float]) -> list:
        return self._resolve_in(
            cur, group, lambda resource, reserved: max(0.0, actuals.get(resource, 0.0))
        )

    def keep_worst_case(self, group: ReservationGroup) -> list:
        """Settle every reservation at its full reserved amount -- used when a call was
        sent and its outcome is unknown, so a possible charge is never under-counted."""
        with self._ledger.transaction() as cur:
            return self.keep_worst_case_in(cur, group)

    def keep_worst_case_in(self, cur, group: ReservationGroup) -> list:
        return self._resolve_in(cur, group, lambda resource, reserved: reserved)

    def release(self, group: ReservationGroup) -> list:
        """Return every reservation in full -- the call provably never went out."""
        with self._ledger.transaction() as cur:
            return self.release_in(cur, group)

    def release_in(self, cur, group: ReservationGroup) -> list:
        return self._resolve_in(cur, group, lambda resource, reserved: 0.0)

    def _resolve_in(self, cur, group: ReservationGroup, actual_fn) -> list:
        """Move each live reservation to SETTLED at its real amount.

        Returns the resources whose actual usage exceeded their reservation. The excess is
        charged in full rather than clamped: clamping to the reservation is how a call that
        cost more than its worst case got recorded as costing exactly the worst case, which
        makes the ledger agree with the budget by understating the bill.
        """
        rows = cur.execute(
            "SELECT reservation_id, resource, amount, state FROM budget_reservations"
            " WHERE attempt_id=? OR external_call_id=? OR reservation_id=?",
            (group.group_id, group.group_id, group.group_id),
        ).fetchall()
        # Fall back to deriving the per-resource ids when no attempt/call link exists.
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
        over: list = []
        for r in rows:
            if r["state"] != "RESERVED":
                continue  # already resolved; idempotent
            reserved = r["amount"]
            actual = max(0.0, actual_fn(r["resource"], reserved))
            if actual > reserved:
                over.append(BudgetUnderReserved(r["resource"], reserved, actual))
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
        return over
