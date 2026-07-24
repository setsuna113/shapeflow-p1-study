"""The coordinator loop: idempotent, crash-resumable work execution (plan §16.2).

The loop is deliberately thin and defensive. It claims a pending work item, runs the injected
executor, stores the result, then commits the terminal record **last** -- so a crash between
"stored the bytes" and "wrote COMMITTED" leaves the item non-terminal and it re-runs on resume
rather than being trusted with a commit whose bytes might be absent. Committed items are skipped
on resume (and their artifact re-verified), so re-running the whole campaign is safe and cheap.

Failures are routed by kind: a :class:`RetryableError` reopens the item (up to its retry budget),
a :class:`FatalProtocolError` fails it finally and records an incident. The executor is injected,
so the same loop drives forks, selector calls, judged reports -- and is tested here with a mock,
independently of any model or GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from ..object_store import ObjectStore
from .ledger import Ledger, WorkItem

__all__ = [
    "RetryableError",
    "FatalProtocolError",
    "Coordinator",
    "RunSummary",
]


class RetryableError(Exception):
    """A transient failure -- reopen the item while retries remain."""


class FatalProtocolError(Exception):
    """A non-recoverable failure -- fail the item finally and record an incident."""


# executor(work_item) -> raw artifact bytes; may raise RetryableError / FatalProtocolError.
Executor = Callable[[WorkItem], bytes]
# validator(work_item, raw) -> None; raise to reject a materialized result.
Validator = Callable[[WorkItem, bytes], None]


@dataclass
class RunSummary:
    committed: int = 0
    retried: int = 0
    failed_final: int = 0
    blocked: int = 0
    fatal: bool = False


class Coordinator:
    def __init__(
        self,
        ledger: Ledger,
        store: ObjectStore,
        *,
        executor: Executor,
        validator: Optional[Validator] = None,
        worker_id: str = "w0",
        lease_seconds: float = 300.0,
    ) -> None:
        self._ledger = ledger
        self._store = store
        self._executor = executor
        self._validator = validator
        self._worker_id = worker_id
        self._lease = lease_seconds

    def resume(self) -> list[str]:
        """Reclaim stale in-progress attempts before a run, reopening pure items that a crashed
        worker left mid-flight. Returns the reopened work keys."""
        return self._ledger.reclaim_stale()

    def _process(self, work_key: str, summary: RunSummary) -> None:
        item = self._ledger.get_work_item(work_key)
        if item is None or item.state != "PENDING":
            return
        attempt = self._ledger.claim(work_key, self._worker_id, lease_seconds=self._lease)
        if attempt is None:  # lost the race or no longer claimable
            return
        try:
            raw = self._executor(item)
            # Store the output BEFORE marking materialized/committed, so the terminal record is
            # always the last thing written.
            ref = self._store.put_bytes(raw)
            self._ledger.register_artifact(
                ref.key, kind="work_output", raw_size=ref.raw_size,
                stored_size=ref.stored_size, work_key=work_key,
            )
            self._ledger.advance(attempt.attempt_id, "MATERIALIZED")
            if self._validator is not None:
                self._validator(item, raw)
            self._ledger.advance(attempt.attempt_id, "VALIDATED")
            self._ledger.commit(attempt.attempt_id, result_object_ref=ref.key)
            summary.committed += 1
        except RetryableError as e:
            state = self._ledger.fail(
                attempt.attempt_id, disposition="FAILED_RETRYABLE", error_class=type(e).__name__,
                reason=str(e),
            )
            summary.retried += 1
            if state != "PENDING":
                summary.failed_final += 1
        except FatalProtocolError as e:
            self._ledger.fail(
                attempt.attempt_id, disposition="FAILED_FINAL", error_class=type(e).__name__,
                reason=str(e),
            )
            self._ledger.record_incident(
                severity="fatal", kind="protocol_error", detail=str(e), work_key=work_key
            )
            summary.failed_final += 1
            summary.fatal = True

    def run(self, *, max_items: Optional[int] = None,
            budget_remaining: Optional[Callable[[], float]] = None) -> RunSummary:
        """Drive pending work to terminal states. Stops when nothing is claimable, ``max_items``
        is reached, the budget is exhausted, or a fatal protocol error occurs."""
        summary = RunSummary()
        self.resume()
        processed = 0
        while True:
            if budget_remaining is not None and budget_remaining() <= 0:
                break
            if max_items is not None and processed >= max_items:
                break
            pending = self._ledger.pending_keys(limit=1)
            if not pending:
                break
            before = summary.committed + summary.retried + summary.failed_final
            self._process(pending[0], summary)
            after = summary.committed + summary.retried + summary.failed_final
            if after == before:
                # Nothing happened (raced away); avoid a busy spin.
                break
            processed += 1
            if summary.fatal:
                break
        return summary

    def committed_artifact_valid(self, work_key: str) -> bool:
        """Resume-time integrity check: a committed item's blob must still verify."""
        ref = self._ledger.committed_ref(work_key)
        return ref is not None and self._store.verify(ref)
