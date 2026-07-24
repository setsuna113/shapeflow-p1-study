"""Every external API call as an auditable state machine.

A provider call is not a function that returns a value; it is a sequence of durable facts.
This ledger records that sequence so that after any crash we can say exactly how far a
call got and what it may have cost.

Two levels, deliberately separate:

**The logical call** is the thing we meant to buy -- one search for one query, one judgment
of one claim. It is identified by its coordinates and it is bought at most once::

    OPEN --> COMMITTED (bought; a replay is served from the frozen response)
         --> ABANDONED (given up on; a further dispatch needs a new logical call)

**A physical attempt** is one dispatch onto the wire. Every retry is a new attempt with a
new id and its own reservation, request blob, response blob and settlement::

    INTENT --> BUDGET_RESERVED --> SENT --> RESPONSE_STORED --> VALIDATED --> COMMITTED
                    |                 |            |               |
                    |                 |            |               +--> FAILED_FINAL (bad response, settle actual)
                    |                 |            +------------------> FAILED_FINAL (unparseable, settle actual)
                    |                 +-------------------------------> FAILED_UNKNOWN (sent, then silence -> keep worst-case)
                    +-------------------------------------------------> FAILED_FINAL (pre-send error -> release reservation)

Keeping them apart is what makes the cost record true. When one row served both roles, a
retry overwrote the first attempt's response ref, model and usage, so a call that was
dispatched three times looked like one call that cost whatever the last dispatch cost.

The distinction that matters most inside an attempt is between "we know it didn't go out"
and "we sent it and then lost contact". The first releases the reservation; the second
keeps the worst-case charge, because a timeout after send may still be billed and this
study refuses to under-count a possible cost.

Request and response bytes are stored **after redaction**. The request in particular
carries the credential (an ``Authorization`` header or an ``api_key`` body field), so the
raw bytes never reach the object store -- only the redacted form does. That keeps the full
audit trail without ever persisting a key, which is the whole point of storing them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

from ..experiment.budget import Budget, BudgetExceeded, ReservationGroup
from ..experiment.ledger import Ledger
from ..hashing import derive_id
from ..object_store import ObjectStore
from ..secrets import SecretRedactor

__all__ = [
    "ExternalCallLedger",
    "AttemptHandle",
    "CallAlreadyCommitted",
    "CallNotReplayable",
    "IllegalCallTransition",
]

# --- states -----------------------------------------------------------------------------

CALL_OPEN = "OPEN"
#: A logical call ends in exactly one of these. A *failed attempt* does not end the call --
#: protocol §3.5 says a retry is a new attempt, and every possible double charge is
#: recorded rather than prevented by refusing to try again. What must never repeat is a
#: successful purchase, which is why COMMITTED is the state that blocks a new dispatch.
CALL_TERMINAL = frozenset({"COMMITTED", "ABANDONED"})

ATTEMPT_SENT_STATES = frozenset({"SENT", "RESPONSE_STORED", "VALIDATED"})
ATTEMPT_TERMINAL = frozenset({"COMMITTED", "FAILED_FINAL", "FAILED_UNKNOWN"})

#: Legal forward transitions of one physical attempt. Enforced by compare-and-swap, not by
#: convention: an unconditional ``UPDATE ... WHERE attempt_id=?`` would happily move a
#: settled attempt back to SENT, and two threads racing on one row would both "succeed".
_LEGAL_ATTEMPT: dict[str, frozenset[str]] = {
    "INTENT": frozenset({"BUDGET_RESERVED", "FAILED_FINAL"}),
    "BUDGET_RESERVED": frozenset({"SENT", "FAILED_FINAL", "FAILED_UNKNOWN"}),
    "SENT": frozenset({"RESPONSE_STORED", "FAILED_FINAL", "FAILED_UNKNOWN"}),
    "RESPONSE_STORED": frozenset({"VALIDATED", "FAILED_FINAL", "FAILED_UNKNOWN"}),
    "VALIDATED": frozenset({"COMMITTED", "FAILED_FINAL", "FAILED_UNKNOWN"}),
}


class CallAlreadyCommitted(RuntimeError):
    """This logical call has already been paid for.

    Carries the frozen response so a replay can be answered from the ledger instead of
    dispatching -- and paying -- a second time.
    """

    def __init__(self, call_id: str, response_object_ref: Optional[str]) -> None:
        super().__init__(
            f"external call {call_id} is already committed; returning the frozen response "
            "rather than paying for it twice"
        )
        self.call_id = call_id
        self.response_object_ref = response_object_ref


class CallNotReplayable(RuntimeError):
    """The call was abandoned, or one of its attempts is still in flight.

    Refusing is the point. Dispatching alongside a live attempt is an unaccounted second
    charge, and re-opening an abandoned call would quietly restart a purchase somebody
    decided to stop making.
    """

    def __init__(self, call_id: str, state: str) -> None:
        super().__init__(
            f"external call {call_id} is {state}; it cannot be replayed without a new "
            "logical call, because a second dispatch is a second possible charge"
        )
        self.call_id = call_id
        self.state = state


class IllegalCallTransition(RuntimeError):
    pass


@dataclass(frozen=True)
class AttemptHandle:
    """One physical dispatch. Every state method takes this, never a bare call id."""

    call_id: str
    attempt_id: str
    attempt_ordinal: int


class ExternalCallLedger:
    def __init__(
        self,
        ledger: Ledger,
        budget: Budget,
        store: ObjectStore,
        redactor: SecretRedactor,
    ) -> None:
        self._ledger = ledger
        self._budget = budget
        self._store = store
        self._redactor = redactor

    def _conn(self):
        return self._ledger.raw_connection

    def _now(self) -> float:
        return self._ledger.now()

    # --- transitions ------------------------------------------------------------------

    def _advance_in(self, cur, attempt: AttemptHandle, state: str, **cols) -> None:
        """Compare-and-swap one attempt forward, inside the caller's transaction."""
        row = cur.execute(
            "SELECT state FROM external_call_attempts WHERE attempt_id=?",
            (attempt.attempt_id,),
        ).fetchone()
        if row is None:
            raise IllegalCallTransition(f"unknown external-call attempt {attempt.attempt_id}")
        current = row["state"]
        if state not in _LEGAL_ATTEMPT.get(current, frozenset()):
            raise IllegalCallTransition(
                f"attempt {attempt.attempt_id} cannot move {current} -> {state}"
            )
        assignments = "".join(f", {k}=?" for k in cols)
        cur.execute(
            f"UPDATE external_call_attempts SET state=?{assignments} WHERE attempt_id=?"
            " AND state=?",
            [state, *cols.values(), attempt.attempt_id, current],
        )
        if cur.rowcount != 1:
            raise IllegalCallTransition(
                f"attempt {attempt.attempt_id} was not in {current} (concurrent change?)"
            )
        cur.execute(
            "INSERT INTO external_call_transitions(call_id, attempt_id, from_state,"
            " to_state, at, reason) VALUES (?,?,?,?,?,?)",
            (attempt.call_id, attempt.attempt_id, current, state, self._now(), None),
        )
        if state == "COMMITTED":
            cur.execute(
                "UPDATE external_calls SET state='COMMITTED', updated_at=?,"
                " committed_attempt_id=? WHERE call_id=?",
                (self._now(), attempt.attempt_id, attempt.call_id),
            )
        elif state in ATTEMPT_TERMINAL:
            # The attempt is over; the call is not. It stays OPEN so a retry can be a new
            # attempt, and closing it for good is an explicit decision (close_call) made by
            # whoever owns the retry budget.
            cur.execute(
                "UPDATE external_calls SET updated_at=? WHERE call_id=?",
                (self._now(), attempt.call_id),
            )

    def _advance(self, attempt: AttemptHandle, state: str, **cols) -> None:
        with self._ledger.transaction() as cur:
            self._advance_in(cur, attempt, state, **cols)

    # --- opening --------------------------------------------------------------------

    def open_call(
        self,
        *,
        provider: str,
        op_class: str,
        call_key: str,
        work_key: Optional[str] = None,
    ) -> str:
        """Record the logical intent to buy something. Deterministic id from the
        coordinates, so the same logical purchase is recognisable across restarts."""
        call_id = derive_id(
            "external_call",
            {
                "provider": provider,
                "op_class": op_class,
                "work_key": work_key or "-",
                "call_key": call_key,
            },
        )
        with self._ledger.transaction() as cur:
            cur.execute(
                "INSERT OR IGNORE INTO external_calls(call_id, provider, op_class,"
                " call_key, work_key, state, attempt_count, created_at, updated_at)"
                " VALUES (?,?,?,?,?,?,0,?,?)",
                (
                    call_id, provider, op_class, call_key, work_key, CALL_OPEN,
                    self._now(), self._now(),
                ),
            )
        return call_id

    def begin_attempt(self, call_id: str) -> AttemptHandle:
        """Append a new physical attempt for this logical call.

        Refuses a call that already reached a terminal state. A committed call raises
        :class:`CallAlreadyCommitted` carrying its frozen response, so the caller can serve
        the replay from the ledger; any other terminal state raises
        :class:`CallNotReplayable` rather than quietly buying the same thing again.
        """
        with self._ledger.transaction() as cur:
            row = cur.execute(
                "SELECT state, attempt_count, committed_attempt_id FROM external_calls"
                " WHERE call_id=?",
                (call_id,),
            ).fetchone()
            if row is None:
                raise IllegalCallTransition(f"unknown external call {call_id}")
            if row["state"] == "COMMITTED":
                raise CallAlreadyCommitted(
                    call_id, self._response_ref_in(cur, row["committed_attempt_id"])
                )
            if row["state"] in CALL_TERMINAL:
                raise CallNotReplayable(call_id, row["state"])
            live = cur.execute(
                "SELECT attempt_id FROM external_call_attempts WHERE call_id=?"
                " AND state NOT IN ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN')",
                (call_id,),
            ).fetchone()
            if live is not None:
                raise CallNotReplayable(call_id, "attempt-in-flight")
            ordinal = int(row["attempt_count"] or 0)
            attempt_id = derive_id(
                "external_call_attempt", {"call_id": call_id, "ordinal": ordinal}
            )
            cur.execute(
                "INSERT INTO external_call_attempts(attempt_id, call_id, attempt_ordinal,"
                " state, opened_at) VALUES (?,?,?,'INTENT',?)",
                (attempt_id, call_id, ordinal, self._now()),
            )
            cur.execute(
                "INSERT INTO external_call_transitions(call_id, attempt_id, from_state,"
                " to_state, at, reason) VALUES (?,?,NULL,'INTENT',?,'attempt_opened')",
                (call_id, attempt_id, self._now()),
            )
            cur.execute(
                "UPDATE external_calls SET attempt_count=?, updated_at=? WHERE call_id=?",
                (ordinal + 1, self._now(), call_id),
            )
        return AttemptHandle(call_id=call_id, attempt_id=attempt_id, attempt_ordinal=ordinal)

    def close_call(self, call_id: str, *, reason: str) -> None:
        """Give up on a logical call for good.

        Used when retries are exhausted, the breaker is open, or a round is being frozen.
        After this, a further dispatch needs a new logical call, so the decision to stop
        buying something is recorded rather than implied by an absence of rows.
        """
        with self._ledger.transaction() as cur:
            cur.execute(
                "UPDATE external_calls SET state='ABANDONED', updated_at=? WHERE call_id=?"
                " AND state='OPEN'",
                (self._now(), call_id),
            )
            if cur.rowcount:
                cur.execute(
                    "INSERT INTO external_call_transitions(call_id, attempt_id, from_state,"
                    " to_state, at, reason) VALUES (?,NULL,'OPEN','ABANDONED',?,?)",
                    (call_id, self._now(), reason),
                )

    @staticmethod
    def _response_ref_in(cur, attempt_id: Optional[str]) -> Optional[str]:
        if not attempt_id:
            return None
        row = cur.execute(
            "SELECT response_object_ref FROM external_call_attempts WHERE attempt_id=?",
            (attempt_id,),
        ).fetchone()
        return row["response_object_ref"] if row else None

    def committed_response_ref(self, call_id: str) -> Optional[str]:
        """The frozen response of a committed call, for serving a replay without paying."""
        with self._ledger.lock:
            cur = self._conn().cursor()
            row = cur.execute(
                "SELECT committed_attempt_id FROM external_calls WHERE call_id=?"
                " AND state='COMMITTED'",
                (call_id,),
            ).fetchone()
            if row is None:
                return None
            return self._response_ref_in(cur, row["committed_attempt_id"])

    # --- the money path ---------------------------------------------------------------

    def reserve(
        self,
        attempt: AttemptHandle,
        amounts: Mapping[str, float],
        *,
        work_key: Optional[str] = None,
    ) -> ReservationGroup:
        """Reserve worst-case budget before the call goes out. On refusal, mark the attempt
        FAILED_FINAL and re-raise so the caller blocks the work item."""
        try:
            with self._ledger.transaction() as cur:
                group = self._budget.reserve_in(
                    cur, amounts, work_key=work_key,
                    external_call_id=attempt.call_id, attempt_id=attempt.attempt_id,
                )
                self._advance_in(cur, attempt, "BUDGET_RESERVED")
        except BudgetExceeded:
            self._advance(attempt, "FAILED_FINAL", error_class="budget_exceeded")
            raise
        return group

    def mark_sent(self, attempt: AttemptHandle, *, request_text: str) -> None:
        """Transition to SENT, storing the REDACTED request bytes. After this point a
        failure with no response is FAILED_UNKNOWN, never a clean release."""
        ref = self._store.put_bytes(self._redactor.redact(request_text).encode("utf-8"))
        self._ledger.register_artifact(
            ref.key, kind="external_request", raw_size=ref.raw_size,
            stored_size=ref.stored_size, work_key=attempt.call_id,
        )
        self._advance(
            attempt, "SENT", request_object_ref=ref.key, dispatched_at=self._now()
        )

    def store_response(
        self,
        attempt: AttemptHandle,
        *,
        response_text: str,
        provider_request_id: str = "",
        requested_model: str = "",
        returned_model: str = "",
        system_fingerprint: str = "",
        usage_json: str = "",
    ) -> None:
        ref = self._store.put_bytes(self._redactor.redact(response_text).encode("utf-8"))
        self._ledger.register_artifact(
            ref.key, kind="external_response", raw_size=ref.raw_size,
            stored_size=ref.stored_size, work_key=attempt.call_id,
        )
        self._advance(
            attempt,
            "RESPONSE_STORED",
            response_object_ref=ref.key,
            provider_request_id=provider_request_id,
            requested_model=requested_model,
            returned_model=returned_model,
            system_fingerprint=system_fingerprint,
            usage_json=usage_json,
        )

    def validate(self, attempt: AttemptHandle) -> None:
        self._advance(attempt, "VALIDATED")

    def commit(
        self, attempt: AttemptHandle, group: ReservationGroup, actuals: Mapping[str, float]
    ) -> None:
        """Settle budget at actual usage and mark the call done, in one transaction."""
        self._settle_and_advance(attempt, "COMMITTED", group, actuals, mode="settle")

    def fail_before_send(
        self, attempt: AttemptHandle, group: Optional[ReservationGroup], *, error_class: str
    ) -> None:
        """A pre-send failure: the call provably never went out, so release fully."""
        self._settle_and_advance(
            attempt, "FAILED_FINAL", group, {}, mode="release", error_class=error_class
        )

    def fail_unknown(
        self, attempt: AttemptHandle, group: Optional[ReservationGroup], *, error_class: str
    ) -> None:
        """Sent, then silence. Keep the worst-case charge -- it may have been billed."""
        self._settle_and_advance(
            attempt, "FAILED_UNKNOWN", group, {}, mode="worst_case", error_class=error_class
        )

    def fail_after_response(
        self,
        attempt: AttemptHandle,
        group: Optional[ReservationGroup],
        actuals: Mapping[str, float],
        *,
        error_class: str,
    ) -> None:
        """A response came back but was an error or unusable. It was likely billed, so
        settle at the actual usage the provider reported rather than releasing."""
        self._settle_and_advance(
            attempt, "FAILED_FINAL", group, actuals, mode="settle", error_class=error_class
        )

    def _settle_and_advance(
        self,
        attempt: AttemptHandle,
        state: str,
        group: Optional[ReservationGroup],
        actuals: Mapping[str, float],
        *,
        mode: str,
        error_class: Optional[str] = None,
    ) -> None:
        """Money and the state that explains it move together or not at all.

        Splitting these was how a call whose budget had already settled could be found in
        a non-terminal state after a crash and then reconciled to FAILED_UNKNOWN -- charged
        once, recorded as never having finished, and eligible to be charged again.
        """
        cols = {"settled_at": self._now()}
        if error_class is not None:
            cols["error_class"] = error_class
        over: list = []
        with self._ledger.transaction() as cur:
            if group is not None:
                if mode == "settle":
                    over = self._budget.settle_in(cur, group, actuals)
                elif mode == "worst_case":
                    over = self._budget.keep_worst_case_in(cur, group)
                else:
                    over = self._budget.release_in(cur, group)
            self._advance_in(cur, attempt, state, **cols)
        for exc in over:
            self._ledger.record_incident(
                severity="ERROR", kind="budget_under_reserved", detail=str(exc),
                work_key=attempt.call_id,
            )

    # --- reads -------------------------------------------------------------------------

    def get_state(self, call_id: str) -> Optional[str]:
        """The *logical* call's state."""
        with self._ledger.lock:
            row = self._conn().execute(
                "SELECT state FROM external_calls WHERE call_id=?", (call_id,)
            ).fetchone()
        return row["state"] if row else None

    def attempt_state(self, attempt_id: str) -> Optional[str]:
        with self._ledger.lock:
            row = self._conn().execute(
                "SELECT state FROM external_call_attempts WHERE attempt_id=?", (attempt_id,)
            ).fetchone()
        return row["state"] if row else None

    def live_attempts(self) -> list[AttemptHandle]:
        """Attempts left non-terminal by a crash, oldest first."""
        with self._ledger.lock:
            rows = self._conn().execute(
                "SELECT attempt_id, call_id, attempt_ordinal, state FROM"
                " external_call_attempts WHERE state NOT IN"
                " ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN') ORDER BY opened_at"
            ).fetchall()
        return [
            AttemptHandle(
                call_id=r["call_id"],
                attempt_id=r["attempt_id"],
                attempt_ordinal=r["attempt_ordinal"],
            )
            for r in rows
        ]

    def attempt_was_sent(self, attempt_id: str) -> bool:
        return (self.attempt_state(attempt_id) or "") in ATTEMPT_SENT_STATES
