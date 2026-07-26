"""The P1 hooks must fire inside the FULL nested deep_researcher graph, under real awaits.

This is the regression test for the most dangerous failure the study can have: a P1 arm that
silently runs as P0 because the strategy binding did not reach the researcher nodes. Both parity
gates compare P0 to P0, so neither can see it -- only a test that binds a P1 strategy, runs the
whole graph, and asserts the hook actually fired can.

The scripted model here *awaits* (``asyncio.sleep(0)``) on every call, the way a real vLLM
request yields to the loop. That matters: the binding propagated fine when responses were
instant and broke only once a node suspended and was resumed by langgraph's background executor,
so a double that never yields would pass while the real run failed.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"

pytestmark = pytest.mark.skipif(
    not PATCHED.exists(), reason="ODR not materialized; run scripts/materialize_vendor.sh")


class AwaitingModel:
    """A scripted chat model that yields to the loop on every call, like a real engine."""

    def __init__(self, log: list) -> None:
        self._log = log
        self._structured = None
        self._config: dict = {}
        # Production refuses to claim seed pairing unless ODR's configurable-model hook has
        # the pinned runtime shape.  This double deliberately mimics that shape; omitting it
        # would test a model construction path production correctly rejects before invocation.
        self._configurable_fields = ["model", "max_tokens", "api_key"]
        # Per-conversation turn counters keyed by the first message, so the supervisor and the
        # researcher each advance their own script.
        self._turns: dict = {}

    def with_config(self, config=None, **kw):
        clone = AwaitingModel(self._log)
        clone._structured = self._structured
        clone._turns = self._turns
        clone._config = {**self._config, **(config or {}), **kw}
        return clone

    def with_structured_output(self, schema, **kw):
        clone = self.with_config()
        clone._structured = schema
        return clone

    def with_retry(self, **kw):
        return self

    def bind_tools(self, tools, **kw):
        clone = self.with_config()
        clone._config["tools"] = [getattr(t, "name", getattr(t, "__name__", "")) for t in tools]
        return clone

    async def ainvoke(self, messages, config=None, **kw):
        await asyncio.sleep(0)      # the crucial yield: suspend and be resumed by the executor
        from langchain_core.messages import AIMessage
        from open_deep_research.state import (
            ClarifyWithUser, ResearchQuestion, Summary,
        )

        if self._structured is ClarifyWithUser:
            return ClarifyWithUser(need_clarification=False, question="", verification="go")
        if self._structured is ResearchQuestion:
            return ResearchQuestion(research_brief="Research the harbour totals.")
        if self._structured is Summary:
            return Summary(summary="Harbour 4821; Union 4410.", key_excerpts="4821 / 4410")
        tools = self._config.get("tools", [])
        if "ConductResearch" in tools:
            n = self._turns.get("sup", 0)
            self._turns["sup"] = n + 1
            if n == 0:
                return AIMessage(content="", tool_calls=[{
                    "id": "s1", "name": "ConductResearch",
                    "args": {"research_topic": "harbour totals"}, "type": "tool_call"}])
            return AIMessage(content="", tool_calls=[{"id": "s2", "name": "ResearchComplete",
                                                     "args": {}, "type": "tool_call"}])
        if "tavily_search" in tools:
            n = self._turns.get("res", 0)
            self._turns["res"] = n + 1
            if n == 0:
                return AIMessage(content="", tool_calls=[{
                    "id": "t1", "name": "tavily_search",
                    "args": {"queries": ["harbour 2025 totals"]}, "type": "tool_call"}])
            return AIMessage(content="", tool_calls=[{"id": "t2", "name": "ResearchComplete",
                                                     "args": {}, "type": "tool_call"}])
        return AIMessage(content="Final report: the Harbour Authority reported 4821 [1].")


async def test_the_close_hook_fires_in_the_full_graph(monkeypatch):
    """Run the whole deep_researcher graph with a C_VISIBLE close strategy bound, and require
    that the close boundary was actually reached with the binding visible."""
    import open_deep_research.deep_researcher as dr
    import open_deep_research.utils as u

    log: list = []
    model = AwaitingModel(log)
    monkeypatch.setattr(u, "init_chat_model", lambda *a, **k: model)
    monkeypatch.setattr(dr, "configurable_model", model)
    monkeypatch.setattr(u, "get_api_key_for_model", lambda *a, **k: "x")
    monkeypatch.setattr(dr, "get_api_key_for_model", lambda *a, **k: "x")

    async def fake_search(queries, max_results=5, topic="general", include_raw_content=True,
                          config=None):
        return [{"query": q, "results": [{
            "url": "https://x.invalid/a", "title": "A", "content": "snip",
            "raw_content": "The Harbour Authority reported 4821 movements in 2025. " * 6,
        }]} for q in queries]

    monkeypatch.setattr(u, "tavily_search_async", fake_search)

    fired = {"close": 0, "page": 0}
    import shapeflow_p1.odr.vendor_hooks as vh

    real_close = vh.run_close_strategy

    async def counting_close(**kw):
        fired["close"] += 1
        return await real_close(**kw)

    monkeypatch.setattr(vh, "run_close_strategy", counting_close)

    from shapeflow_p1.campaign.graph_driver import CellSpec, run_cell
    from shapeflow_p1.campaign.settings import Settings
    from shapeflow_p1.odr.hooks import StrategyBundle
    from shapeflow_p1.strategies.close_visible import CloseSelectionStrategy, CloseStrategyConfig
    from shapeflow_p1.strategies.p0 import VendorPageStrategy
    from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer

    settings = Settings.load(REPO, data_root=Path("/tmp/hook-probe"))

    class EchoSelector:
        async def select(self, *, task_ctx, view):
            from shapeflow_p1.strategies.pipeline import WorkRecord
            ids = [c.label for c in view.candidates[:2]]
            return ({"contract": "P1_ID", "selected_ids": ids},
                    WorkRecord(selector_calls=1, completion_tokens=5))

    close = CloseSelectionStrategy(
        config=CloseStrategyConfig(variant_id="C01", node="C_VISIBLE", contract="P1_ID",
                                   aggregation="stable_union_v1", close_mode="dedicated_selector",
                                   token_budget=256),
        selector=EchoSelector(), tokenizer=WhitespaceTokenizer())
    bundle = StrategyBundle(variant_id="P0+C01", page=VendorPageStrategy({}), close=close)

    from shapeflow_p1.acquire.snapshot_store import SnapshotStore
    from shapeflow_p1.acquire.source_pool import QueryResponse, RawResult, build_source_pool
    from shapeflow_p1.object_store import ObjectStore

    store = SnapshotStore(ObjectStore(Path("/tmp/hook-probe-obj")))
    pool = build_source_pool("T", [QueryResponse("qs", "q", (
        RawResult("https://x.invalid/a", "A", 1, "s",
                  "Harbour Authority 4821 in 2025. " * 6, 0.9),))],
        store, fetched_at_utc="2026-07-24T00:00:00Z")

    cell = CellSpec(run_id="probe", task_id="T", arm_id="C_VISIBLE", page_variant="P0",
                    close_variant="C01", replicate_id="0", seed=1, work_key="WK",
                    question="Which bodies reported harbour totals?",
                    cell_token="probe-cell-1",
                    execution_binding_sha256="e" * 64,
                    protocol_document_sha256="d" * 64)
    result = await asyncio.wait_for(
        run_cell(settings, cell, pool=pool, snapshots=store, bundle=bundle,
                 provider_base_url="http://127.0.0.1:1", runner_token="x",
                 graph=dr.deep_researcher),
        timeout=60)

    # The close boundary must have been reached WITH the binding visible -- that is the whole
    # point. If the binding did not propagate, run_close_strategy returns None early and no
    # CLOSE_REDUCED event is emitted.
    assert fired["close"] > 0, "compress_research never invoked the close hook"
    kinds = [e.get("kind") for e in result.events]
    assert "CLOSE_REDUCED" in kinds, (
        f"the close strategy did not run with the binding visible; events={kinds}")
