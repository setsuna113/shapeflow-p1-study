"""A complete P0 task on the real graph, with the network physically unavailable.

This is the test the whole frozen-corpus design exists for. If a treatment run can still reach
the web, then P1 -- which changes the queries a researcher issues -- also changes which pages
exist, and P0 and P1 are compared across two different worlds. So the run is executed with
outbound DNS and non-loopback connections disabled at the socket layer, and it still has to
finish: prepare, acquire, freeze, then drive vendor's own compiled ``deep_researcher``.

The provider is real (on loopback) and its upstream is a deterministic fake engine, so the run
also exercises the credential boundary, the cell tagging and the work ledger without a GPU.
"""

from __future__ import annotations

import socket
from pathlib import Path

import pytest

from shapeflow_p1.acquire.tavily_client import TavilyCaptureClient
from shapeflow_p1.campaign.acquire import (
    acquire_all,
    acquired_task_ids,
    load_frozen_pool,
    tavily_params_from,
)
from shapeflow_p1.campaign.graph_driver import CellSpec, run_cell, summarize_events
from shapeflow_p1.campaign.prepare import prepare_corpus
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.judge_client import DeepSeekJudge
from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.odr.hooks import StrategyBundle
from shapeflow_p1.runtime.provider_server import (
    ProviderConfig,
    ProviderService,
    RoleTokens,
    serve_forever,
)
from shapeflow_p1.secrets import SecretRedactor
from shapeflow_p1.strategies.p0 import VendorCloseStrategy, VendorPageStrategy

from fixtures.fake_engine import FakeEngine
from fixtures.fake_tavily import FakeTavily
from fixtures.scripted_author import ScriptedAuthor

REPO = Path(__file__).resolve().parents[2]
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"

pytestmark = pytest.mark.skipif(
    not PATCHED.exists(), reason="ODR not materialized; run scripts/materialize_vendor.sh")

TOKENS = {
    "runner": "offline-runner-token-000000000",
    "steward": "offline-steward-token-00000000",
    "evaluator": "offline-evaluator-token-000000",
}


@pytest.fixture()
def no_network(monkeypatch):
    """Refuse every connection that is not loopback, at the socket layer.

    Stronger than trusting the code not to call out: if anything reaches for the network, the
    run fails rather than quietly succeeding against a live page.
    """
    real_connect = socket.socket.connect
    real_getaddrinfo = socket.getaddrinfo

    def guarded_connect(self, address, *args, **kwargs):
        host = address[0] if isinstance(address, tuple) else str(address)
        if host not in ("127.0.0.1", "::1", "localhost"):
            raise OSError(f"network disabled in this test: refused connection to {host}")
        return real_connect(self, address, *args, **kwargs)

    def guarded_getaddrinfo(host, *args, **kwargs):
        if host not in ("127.0.0.1", "::1", "localhost", None):
            raise socket.gaierror(f"network disabled in this test: refused lookup of {host}")
        return real_getaddrinfo(host, *args, **kwargs)

    monkeypatch.setattr(socket.socket, "connect", guarded_connect)
    monkeypatch.setattr(socket, "getaddrinfo", guarded_getaddrinfo)
    yield


@pytest.fixture()
def frozen_world(tmp_path):
    """A sealed corpus with one task's world already frozen."""
    settings = Settings.load(REPO, data_root=tmp_path)
    relaxed = dict(settings.configs["task_source"])
    relaxed["audit"] = {**relaxed["audit"], "require_distinct_topic_clusters": 4}
    relaxed["splits"] = {"FORMATIVE_SCREEN": 4, "FORMATIVE_POWER_PILOT": 2, "RESERVE": 2}
    relaxed["strata_min_counts"] = {"source_conflict": 1}
    settings.configs["task_source"] = relaxed
    return settings


async def _prepare_and_acquire(settings):
    # Eight tasks over two clusters: the smallest corpus whose round-robin still reaches
    # every stratum, which build_registry requires before it will seal anything.
    author = ScriptedAuthor(clusters=4, per_cluster=2)
    judge = DeepSeekJudge(author, "deepseek-chat", "@SHAPEFLOW_PROVIDER@")
    result = await prepare_corpus(
        settings, judge=judge, authored_at_utc="2026-07-24T00:00:00Z",
        target_model="Qwen3-14B-AWQ", total=8, clusters=4,
    )
    params = tavily_params_from(settings)
    fake = FakeTavily(pages_per_query=2)
    await acquire_all(
        settings,
        client_factory=lambda task_id: TavilyCaptureClient(fake, params, "@SHAPEFLOW_PROVIDER@"),
        fetched_at_utc="2026-07-24T01:00:00Z",
    )
    return result


def _provider(settings, tmp_path, engine):
    ledger = Ledger(str(tmp_path / "provider.sqlite"))
    budget = Budget(ledger)
    for resource, cap in settings.budget_caps().items():
        budget.ensure_account(resource, cap)
    redactor = SecretRedactor()
    service = ProviderService(
        ProviderConfig(served_model="Qwen3-14B-AWQ"),
        ledger=ledger, budget=budget, store=ObjectStore(tmp_path / "provider-objects"),
        redactor=redactor, tokens=RoleTokens(TOKENS), upstream=engine,
        tavily_key=None, deepseek_key=None,
    )
    service.reconcile_on_start()
    tcp, _ = serve_forever(service, ProviderConfig(bind_port=0), redactor)
    return service, tcp, ledger


async def test_a_complete_p0_task_runs_with_no_network(frozen_world, tmp_path, no_network):
    settings = frozen_world
    result = await _prepare_and_acquire(settings)
    task_ids = acquired_task_ids(settings)
    assert task_ids, "acquisition produced no frozen pool"
    task_id = task_ids[0]
    question = next(t.question for t in result.registry.tasks if t.task_id == task_id)

    engine = FakeEngine()
    service, tcp, ledger = _provider(settings, tmp_path, engine)
    try:
        base = f"http://127.0.0.1:{tcp.server_address[1]}"
        cell = CellSpec(
            run_id="RUN-OFFLINE", task_id=task_id, arm_id="P0", page_variant="P0",
            close_variant="P0", replicate_id="0", seed=1, work_key="WK-OFFLINE",
            question=question, cell_token="cell-offline-1",
        )
        import asyncio

        from shapeflow_p1.providers.provider_client import ProviderClient

        await ProviderClient(base_url=base, token=TOKENS["runner"]).register_cell(
            cell_token=cell.cell_token, run_id=cell.run_id, task_id=cell.task_id,
            arm_id=cell.arm_id, variant_id=cell.variant_id, replicate_id=cell.replicate_id,
            work_key=cell.work_key,
        )
        pool, snapshots = load_frozen_pool(settings, task_id)
        outcome = await asyncio.wait_for(
            run_cell(
                settings, cell, pool=pool, snapshots=snapshots,
                bundle=StrategyBundle(variant_id="P0", page=VendorPageStrategy({}),
                                      close=VendorCloseStrategy()),
                provider_base_url=base, runner_token=TOKENS["runner"],
            ),
            timeout=180,
        )
    finally:
        tcp.shutdown()
        ledger.close()

    assert outcome.ok, f"the offline run failed: {outcome.error}"
    assert outcome.final_report, "the graph produced no report"
    assert engine.requests, "no model request reached the provider"

    # An explicit P0 strategy still travels the defer/checkpoint/reduce/refill path -- that is
    # what parity gate B exists to cover -- but it replays vendor's own closures, so nothing
    # falls back and every deferred batch is reduced exactly once.
    counts = summarize_events(outcome.events)
    assert counts["page_batches_deferred"] == counts["page_batches_reduced"]
    assert counts["page_fallbacks"] == 0
    assert counts["close_failed"] == 0
    assert counts["close_cancelled"] == 0

    # Every model call was attributed to this cell, and each carries an op class.
    committed = [e for e in service.events if e["kind"] == "INFERENCE_COMMITTED"]
    assert committed, "no inference was recorded"
    assert {e["task_id"] for e in committed} == {task_id}
    assert {e["arm_id"] for e in committed} == {"P0"}
    assert all(e["op_class"] for e in committed)
    # The engine saw one served model, whatever alias the graph asked for -- so P0 and P1 issue
    # byte-identical upstream requests and remain separable only in the ledger.
    assert {r["body"]["model"] for r in engine.requests} == {"Qwen3-14B-AWQ"}
    aliases = {e["requested_model"] for e in committed}
    assert {"qwen-research", "qwen-summarize", "qwen-compress", "qwen-final"} <= aliases
    ops = {e["op_class"] for e in committed}
    assert {"RESEARCHER_REACT", "PAGE_P0_SUMMARY", "COMPRESSOR_P0", "FINAL_WRITER"} <= ops


async def test_the_frozen_search_never_reaches_the_network(frozen_world, tmp_path, no_network):
    """The replacement backend has no HTTP client at all; a miss is empty, not a live query."""
    settings = frozen_world
    await _prepare_and_acquire(settings)
    task_id = acquired_task_ids(settings)[0]
    pool, snapshots = load_frozen_pool(settings, task_id)

    from shapeflow_p1.campaign.graph_driver import install_frozen_search

    with install_frozen_search(pool, snapshots, max_results=5):
        import open_deep_research.utils as vendor_utils

        payload = await vendor_utils.tavily_search_async(
            ["zzqqxx wwvvuu ttssrr"], max_results=5)
    # An empty result set is a real state of the frozen world, not a reason to look elsewhere.
    assert payload[0]["results"] == []

    with install_frozen_search(pool, snapshots, max_results=5):
        import open_deep_research.utils as vendor_utils

        hit = await vendor_utils.tavily_search_async(["measurement units observed 2025"],
                                                     max_results=5)
    assert hit[0]["results"], "the frozen corpus returned nothing for a query it contains"
    assert all(r["url"].endswith("/doc") for r in hit[0]["results"])


async def test_the_search_binding_is_restored_after_a_failure(frozen_world, tmp_path):
    """A cell that raised must not leave the next one searching the previous task's corpus."""
    settings = frozen_world
    await _prepare_and_acquire(settings)
    task_id = acquired_task_ids(settings)[0]
    pool, snapshots = load_frozen_pool(settings, task_id)

    import open_deep_research.utils as vendor_utils

    from shapeflow_p1.campaign.graph_driver import install_frozen_search

    original = vendor_utils.tavily_search_async
    with pytest.raises(RuntimeError):
        with install_frozen_search(pool, snapshots, max_results=5):
            raise RuntimeError("cell exploded")
    assert vendor_utils.tavily_search_async is original
