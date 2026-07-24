"""Every external API call as an auditable state machine.

A provider call is not a function that returns a value; it is a sequence of durable facts.
This ledger records that sequence so that after any crash we can say exactly how far a
call got and what it may have cost::

    INTENT --> BUDGET_RESERVED --> SENT --> RESPONSE_STORED --> VALIDATED --> COMMITTED
                    |                 |            |               |
                    |                 |            |               +--> FAILED_FINAL (bad response, settle actual)
                    |                 |            +------------------> FAILED_FINAL (unparseable, settle actual)
                    |                 +-------------------------------> FAILED_UNKNOWN (sent, then silence -> keep worst-case)
                    +-------------------------------------------------> FAILED_FINAL (pre-send error -> release reservation)

The distinction that matters most is the one between "we know it didn't go out" and "we
sent it and then lost contact". The first releases the reservation; the second keeps the
worst-case charge, because a timeout after send may still be billed and this study refuses
to under-count a possible cost.

Request and response bytes are stored **after redaction**. The request in particular
carries the credential (an ``Authorization`` header or an ``api_key`` body field), so the
raw bytes never reach the object store -- only the redacted form does. That keeps the full
audit trail without ever persisting a key, which is the whole point of storing them.
"""

from __future__ import annotations

from typing import Mapping, Optional

from ..experiment.budget import Budget, BudgetExceeded, ReservationGroup
from ..experiment.ledger import Ledger
from ..hashing import derive_id
from ..object_store import ObjectStore
from ..secrets import SecretRedactor

__all__ = ["ExternalCallLedger"]

_SENT_STATES = frozenset({"SENT", "RESPONSE_STORED", "VALIDATED"})


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

    def _set_state(self, call_id: str, state: str, **cols) -> None:
        assignments = ", ".join(f"{k}=?" for k in cols)
        params = list(cols.values())
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                sql = "UPDATE external_calls SET state=?, updated_at=?"
                args: list = [state, self._now()]
                if assignments:
                    sql += ", " + assignments
                    args += params
                sql += " WHERE call_id=?"
                args.append(call_id)
                cur.execute(sql, args)
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise

    def open_call(
        self,
        *,
        provider: str,
        op_class: str,
        call_key: str,
        work_key: Optional[str] = None,
        attempt_ordinal: int = 0,
    ) -> str:
        """Record intent to call. Deterministic ``call_id`` from the logical coordinates
        so a retried identical call reuses the row rather than duplicating it."""
        call_id = derive_id(
            "external_call",
            {
                "provider": provider,
                "op_class": op_class,
                "work_key": work_key or "-",
                "attempt_ordinal": attempt_ordinal,
                "call_key": call_key,
            },
        )
        with self._ledger.lock:
            cur = self._conn().cursor()
            cur.execute("BEGIN IMMEDIATE;")
            try:
                cur.execute(
                    "INSERT OR IGNORE INTO external_calls(call_id, provider, op_class, state,"
                    " work_key, attempt_ordinal, created_at, updated_at)"
                    " VALUES (?,?,?,'INTENT',?,?,?,?)",
                    (call_id, provider, op_class, work_key, attempt_ordinal, self._now(), self._now()),
                )
                cur.execute("COMMIT;")
            except BaseException:
                cur.execute("ROLLBACK;")
                raise
        return call_id

    def reserve(
        self, call_id: str, amounts: Mapping[str, float], *, work_key: Optional[str] = None
    ) -> ReservationGroup:
        """Reserve worst-case budget before the call goes out. On refusal, mark the call
        FAILED_FINAL and re-raise so the caller blocks the work item."""
        try:
            group = self._budget.reserve(
                amounts, work_key=work_key, external_call_id=call_id
            )
        except BudgetExceeded:
            self._set_state(call_id, "FAILED_FINAL", error_class="budget_exceeded")
            raise
        self._set_state(call_id, "BUDGET_RESERVED")
        return group

    def mark_sent(self, call_id: str, *, request_text: str) -> None:
        """Transition to SENT, storing the REDACTED request bytes. After this point a
        failure with no response is FAILED_UNKNOWN, never a clean release."""
        ref = self._store.put_bytes(self._redactor.redact(request_text).encode("utf-8"))
        self._ledger.register_artifact(
            ref.key, kind="external_request", raw_size=ref.raw_size, stored_size=ref.stored_size
        )
        self._set_state(call_id, "SENT", request_object_ref=ref.key)

    def store_response(
        self,
        call_id: str,
        *,
        response_text: str,
        provider_request_id: str = "",
        model: str = "",
        usage_json: str = "",
    ) -> None:
        ref = self._store.put_bytes(self._redactor.redact(response_text).encode("utf-8"))
        self._ledger.register_artifact(
            ref.key, kind="external_response", raw_size=ref.raw_size, stored_size=ref.stored_size
        )
        self._set_state(
            call_id,
            "RESPONSE_STORED",
            response_object_ref=ref.key,
            provider_request_id=provider_request_id,
            model=model,
            usage_json=usage_json,
        )

    def validate(self, call_id: str) -> None:
        self._set_state(call_id, "VALIDATED")

    def commit(self, call_id: str, group: ReservationGroup, actuals: Mapping[str, float]) -> None:
        """Settle budget at actual usage and mark the call done."""
        self._budget.settle(group, actuals)
        self._set_state(call_id, "COMMITTED")

    def fail_before_send(
        self, call_id: str, group: Optional[ReservationGroup], *, error_class: str
    ) -> None:
        """A pre-send failure: the call provably never went out, so release fully."""
        if group is not None:
            self._budget.release(group)
        self._set_state(call_id, "FAILED_FINAL", error_class=error_class)

    def fail_unknown(
        self, call_id: str, group: Optional[ReservationGroup], *, error_class: str
    ) -> None:
        """Sent, then silence. Keep the worst-case charge -- it may have been billed."""
        if group is not None:
            self._budget.keep_worst_case(group)
        self._set_state(call_id, "FAILED_UNKNOWN", error_class=error_class)

    def fail_after_response(
        self,
        call_id: str,
        group: Optional[ReservationGroup],
        actuals: Mapping[str, float],
        *,
        error_class: str,
    ) -> None:
        """A response came back but was an error or unusable. It was likely billed, so
        settle at the actual usage the provider reported rather than releasing."""
        if group is not None:
            self._budget.settle(group, actuals)
        self._set_state(call_id, "FAILED_FINAL", error_class=error_class)

    def get_state(self, call_id: str) -> Optional[str]:
        with self._ledger.lock:
            row = self._conn().execute(
                "SELECT state FROM external_calls WHERE call_id=?", (call_id,)
            ).fetchone()
        return row["state"] if row else None
