"""The acquisition driver: freeze a task's whole source pool, on budget, through the call FSM.

This ties the acquisition pieces into one auditable flow. For each frozen query in a task's sealed
acquisition spec it: reserves worst-case budget BEFORE dispatch, runs the capture client through
the external-call FSM (sent -> response stored redacted -> validated -> committed), settles at the
reported usage, and on a refusal or hard failure stops without pretending the query succeeded. Then
it builds the vendor-visible/audit source pool and freezes every page as a content-addressed
snapshot.

It is deliberately a thin orchestration over already-tested parts, and it is exercised end-to-end
here with a fake transport and the real ledger/budget -- so the "reserve, call, settle, freeze"
path is validated without a network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

from ..experiment.budget import Budget, BudgetExceeded
from ..experiment.ledger import Ledger
from ..providers.external_call_ledger import (
    CallAlreadyCommitted,
    CallNotReplayable,
    ExternalCallLedger,
)
from .snapshot_store import SnapshotStore
from .source_pool import QueryResponse, SourcePool, build_source_pool
from .exa_client import ExaCaptureClient, ExaHTTPError

__all__ = ["AcquisitionResult", "acquire_task"]


@dataclass
class AcquisitionResult:
    pool: SourcePool
    queries_ok: int = 0
    queries_failed: int = 0
    blocked_budget: bool = False


async def acquire_task(
    *,
    task_id: str,
    queries: Sequence[str],
    client: ExaCaptureClient,
    budget: Budget,
    call_ledger: ExternalCallLedger,
    snapshot_store: SnapshotStore,
    fetched_at_utc: str,
    usd_per_query: float = 0.007,
) -> AcquisitionResult:
    """Acquire and freeze one task's source pool. Stops (blocked_budget) if a reservation is
    refused, keeping already-captured queries."""
    responses: list[QueryResponse] = []
    ok = 0
    failed = 0
    blocked = False

    for query in queries:
        call_id = call_ledger.open_call(
            provider="exa", op_class="search", call_key=client.query_snapshot_id(query),
            work_key=task_id,
        )
        try:
            attempt = call_ledger.begin_attempt(call_id)
        except CallAlreadyCommitted:
            # This exact query was already frozen. Re-fetching would spend the cap twice and
            # produce a second, different world under one task id.
            ok += 1
            continue
        except CallNotReplayable:
            failed += 1
            continue
        try:
            group = call_ledger.reserve(
                attempt, {"exa_requests": 1.0, "exa_usd": usd_per_query},
                work_key=task_id,
            )
        except BudgetExceeded:
            blocked = True
            break  # admission refused -> do not dispatch, stop cleanly

        # The request body carries the key; the FSM stores only its redacted form.
        call_ledger.mark_sent(attempt, request_text=f"exa search: {query}")
        try:
            captured = await client.search(task_id, query)
        except ExaHTTPError as e:
            # A response came back as an error; settle at zero extra (request already counted).
            call_ledger.fail_after_response(
                attempt, group, {"exa_requests": 1.0}, error_class=f"http_{e.status}")
            failed += 1
            continue
        except Exception as e:  # timeout after send: unknown outcome, keep worst case
            call_ledger.fail_unknown(attempt, group, error_class=type(e).__name__)
            failed += 1
            continue

        call_ledger.store_response(
            attempt, response_text=str(captured.raw_response_sha256),
            provider_request_id=captured.request_id,
        )
        call_ledger.validate(attempt)
        call_ledger.commit(
            attempt, group, {"exa_requests": 1.0, "exa_usd": captured.cost_usd})
        responses.append(captured.response)
        ok += 1

    pool = build_source_pool(task_id, responses, snapshot_store, fetched_at_utc=fetched_at_utc)
    return AcquisitionResult(pool=pool, queries_ok=ok, queries_failed=failed, blocked_budget=blocked)
