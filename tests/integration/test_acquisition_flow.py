"""End-to-end acquisition: reserve -> call (mock) -> settle -> freeze, with the real ledger."""

from __future__ import annotations

from shapeflow_p1.acquire.acquisition import acquire_task
from shapeflow_p1.acquire.snapshot_store import SnapshotStore
from shapeflow_p1.acquire.tavily_client import TavilyCaptureClient, TavilyParams
from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.external_call_ledger import ExternalCallLedger
from shapeflow_p1.secrets import SecretRedactor

KEY = "tvly-FAKEFAKEFAKEFAKEFAKE"


def _fake_transport(pages_by_query):
    async def transport(url, body):
        q = body["query"]
        results = pages_by_query.get(q, [])
        return 200, {
            "request_id": f"rq-{q}", "response_time": 0.3, "usage": {"credits": 1},
            "failed_results": [], "results": results,
        }
    return transport


def _wire(tmp_path, transport):
    ledger = Ledger(str(tmp_path / "l.sqlite"), clock=lambda: 1.0)
    budget = Budget(ledger)
    budget.ensure_account("tavily_requests", 100)
    budget.ensure_account("tavily_credits", 100)
    store = ObjectStore(tmp_path / "obj")
    red = SecretRedactor()
    red.register(KEY, label="tavily")
    ecl = ExternalCallLedger(ledger, budget, store, red)
    client = TavilyCaptureClient(transport, TavilyParams(max_results=5), KEY)
    ss = SnapshotStore(store)
    return ledger, budget, ecl, client, ss, store


async def test_acquisition_freezes_pool_and_settles_budget(tmp_path):
    pages = {
        "q1": [{"url": "https://a", "title": "A", "content": "snip", "raw_content": "full A"}],
        "q2": [{"url": "https://b", "title": "B", "content": "snip", "raw_content": "full B"}],
    }
    ledger, budget, ecl, client, ss, store = _wire(tmp_path, _fake_transport(pages))

    result = await acquire_task(
        task_id="t1", queries=["q1", "q2"], client=client, budget=budget,
        call_ledger=ecl, snapshot_store=ss, fetched_at_utc="2026-07-24T00:00:00Z",
    )
    assert result.queries_ok == 2 and result.queries_failed == 0
    assert len(result.pool.vendor_visible) == 2
    # two requests + two credits settled; the rest released
    assert budget.available("tavily_requests") == 98
    assert budget.available("tavily_credits") == 98
    # snapshots frozen and verifiable
    for occ in result.pool.vendor_visible:
        assert occ.content_hash and store.verify(result.pool.snapshots[occ.content_hash].object_ref)


async def test_acquisition_stops_cleanly_when_budget_exhausted(tmp_path):
    pages = {"q1": [{"url": "https://a", "title": "A", "content": "s", "raw_content": "A"}]}
    ledger, budget, ecl, client, ss, store = _wire(tmp_path, _fake_transport(pages))
    # Drain the request budget so the first reservation is refused.
    budget.reserve({"tavily_requests": 100.0})

    result = await acquire_task(
        task_id="t1", queries=["q1"], client=client, budget=budget,
        call_ledger=ecl, snapshot_store=ss, fetched_at_utc="2026-07-24T00:00:00Z",
    )
    assert result.blocked_budget is True
    assert result.queries_ok == 0
    # nothing was dispatched -> empty pool, no snapshots
    assert result.pool.vendor_visible == []


async def test_no_secret_in_any_stored_call_artifact(tmp_path):
    pages = {"q1": [{"url": "https://a", "title": "A", "content": "s", "raw_content": "A"}]}
    ledger, budget, ecl, client, ss, store = _wire(tmp_path, _fake_transport(pages))
    await acquire_task(
        task_id="t1", queries=["q1"], client=client, budget=budget,
        call_ledger=ecl, snapshot_store=ss, fetched_at_utc="2026-07-24T00:00:00Z",
    )
    for blob in (tmp_path / "obj").rglob("*.zst"):
        assert KEY.encode() not in store.get_bytes(blob.stem)
