"""The real agent, brokered: offers come from the compiled graph, and only decided forms run.

Vendor's compiled ``deep_researcher`` drives every boundary, through the real provider on
loopback (upstream: a deterministic fake engine) and a real ``BrokerService`` whose engine state
is a scripted ``/metrics``. The network is disabled at the socket layer.

What this proves is the agent-side mechanics: the graph suspends at each handoff, the offer it
submits prices exactly the requests it later sends, and the form it runs is the one decided. It
cannot prove that decisions follow a *live* engine -- the metrics here are scripted -- which is
what the live run is for.
"""

from __future__ import annotations

import asyncio
import socket
import threading
from collections import Counter
from pathlib import Path

import pytest
from fixtures.fake_engine import FakeEngine
from fixtures.fake_vllm import FakeVllm

from shapeflow.broker.client import BrokerClient
from shapeflow.broker.hosts.agent import AgentHost
from shapeflow.broker.reference import DeadlineBroker, EngineGatedBroker, GreedyJointBroker
from shapeflow.broker.service import BrokerService
from shapeflow.campaign.brokered import build_brokered_bundle
from shapeflow.campaign.graph_driver import CellSpec, run_cell
from shapeflow.campaign.pages import PageRegistry
from shapeflow.campaign.selector_client import SelectorModelCall
from shapeflow.campaign.settings import Settings
from shapeflow.evidence.model_tokenizer import WhitespaceTokenizer
from shapeflow.evidence.shared_view import budget_from_settings
from shapeflow.experiment.budget import Budget
from shapeflow.experiment.ledger import Ledger
from shapeflow.object_store import ObjectStore
from shapeflow.odr.hooks import StrategyBundle
from shapeflow.providers.provider_client import ProviderClient
from shapeflow.runtime.provider_server import (
    ProviderConfig,
    ProviderService,
    RoleTokens,
    serve_forever,
)
from shapeflow.secrets import SecretRedactor
from shapeflow.strategies.factory import load_registry
from shapeflow.strategies.offer_pricing import VendorPrompts
from shapeflow.strategies.p0 import VendorCloseStrategy, VendorPageStrategy
from shapeflow.world.search_backend import SearchRecord

REPO = Path(__file__).resolve().parents[2]
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"

pytestmark = pytest.mark.skipif(
    not PATCHED.exists(), reason="ODR not materialized; run scripts/materialize_vendor.sh")

TOKENS = {
    "runner": "brokered-runner-token-000000000",
    "steward": "brokered-steward-token-00000000",
    "evaluator": "brokered-evaluator-token-000000",
}
PAGE = "\n\n".join(f"## Section {n}\n\nMeasured value {n} was observed in 2025. " + "detail " * 30
                   for n in range(3))


class StaticBackend:
    """Every query returns the same three whole documents, the way a frozen world would."""

    def __init__(self) -> None:
        self.trace: list = []

    def search(self, query: str, *, max_results: int) -> list[SearchRecord]:
        self.trace.append(query)
        return [SearchRecord(url=f"https://doc{i}.example/doc", title=f"Doc {i}",
                             content=f"snippet {i}", raw_content=f"{PAGE}\n\npage {i}",
                             score=1.0 - i / 10, occurrence_id=f"{i:064x}")
                for i in range(3)][:max_results]


@pytest.fixture()
def no_network(monkeypatch):
    real_connect, real_getaddrinfo = socket.socket.connect, socket.getaddrinfo

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
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


def _provider(settings, tmp_path, engine):
    tmp_path.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(tmp_path / "provider.sqlite"))
    budget = Budget(ledger)
    for resource, cap in settings.budget_caps().items():
        budget.ensure_account(resource, cap)
    redactor = SecretRedactor()
    service = ProviderService(
        # Unbounded upstream, as a brokered worker requires: a queue here would gang P0.
        ProviderConfig(served_model="Qwen3-14B-AWQ", max_upstream_inflight=0),
        ledger=ledger, budget=budget, store=ObjectStore(tmp_path / "provider-objects"),
        redactor=redactor, tokens=RoleTokens(TOKENS), upstream=engine,
        tavily_key=None, deepseek_key=None)
    service.reconcile_on_start()
    tcp, _ = serve_forever(service, ProviderConfig(bind_port=0), redactor)
    return service, tcp, ledger


async def _run(settings, tmp_path, *, policy, metrics: FakeVllm, deadline_s=60.0,
               grace_s=5.0, brokered=True, name="cell-brokered-1"):
    engine = FakeEngine()
    service, tcp, ledger = _provider(settings, tmp_path / name, engine)
    broker = BrokerService(AgentHost(scrape=metrics.metrics), policy, tick_ms=20.0,
                           log_dir=tmp_path / name / "broker")
    broker_url = f"http://127.0.0.1:{broker.start()}"
    offers: list = []
    try:
        base = f"http://127.0.0.1:{tcp.server_address[1]}"
        cell = CellSpec(
            run_id="RUN-BROKERED", task_id="T-BROKERED", arm_id="BROKERED" if brokered else "P0",
            page_variant="LLM-PROMPTVIEW" if brokered else "P0",
            close_variant="C01" if brokered else "P0", replicate_id="0", seed=1,
            work_key=f"WK-{name}", question="What measured values were observed in 2025?",
            cell_token=name, execution_binding_sha256="e" * 64,
            protocol_document_sha256="d" * 64,
            form_policy="brokered" if brokered else "fixed")
        client = ProviderClient(base_url=base, token=TOKENS["runner"])
        await client.register_cell(
            cell_token=cell.cell_token, run_id=cell.run_id, task_id=cell.task_id,
            arm_id=cell.arm_id, variant_id=cell.variant_id, replicate_id=cell.replicate_id,
            work_key=cell.work_key)
        pages = PageRegistry()
        if brokered:
            model_call = SelectorModelCall(
                client, cell_token=cell.cell_token, repo=REPO, temperature=0.0, top_p=1.0,
                max_completion_tokens=768, seed=1)
            bundle = build_brokered_bundle(
                settings, load_registry(REPO / "configs"), page_variant="LLM-PROMPTVIEW",
                close_variant="C01", model_call=model_call, tokenizer=WhitespaceTokenizer(),
                pages=pages,
                client=BrokerClient(broker_url, grace_ns=int(grace_s * 1e9), poll_ms=100),
                cell_token=cell.cell_token, deadline_s=deadline_s, prompts=VendorPrompts.load(),
                log=offers.append)
        else:
            bundle = StrategyBundle(variant_id="P0", page=VendorPageStrategy({}),
                                    close=VendorCloseStrategy())
        outcome = await asyncio.wait_for(run_cell(
            settings, cell, backend=StaticBackend(), pages=pages, bundle=bundle,
            provider_base_url=base, runner_token=TOKENS["runner"],
            content_budget=budget_from_settings(settings), tokenizer=WhitespaceTokenizer()),
            timeout=240)
    finally:
        broker.stop()
        tcp.shutdown()
        ledger.close()
    committed = [e for e in service.events if e["kind"] == "INFERENCE_COMMITTED"]
    return outcome, committed, broker, offers


def _decisions(offers):
    return {row["item_id"]: row for row in offers if "offer" in row}


def _assert_only_the_chosen_form_was_sent(outcome, committed, offers):
    sent = Counter(e["prompt_sha256"] for e in committed)
    fell_back = {e["checkpoint"] for e in outcome.events
                 if e["kind"] == "PAGE_BATCH_REDUCED" and e["fell_back"]}
    for row in _decisions(offers).values():
        offer = row["offer"]
        p0 = Counter(r["prompt_sha256"] for r in offer["p0"]["requests"])
        p1 = offer["p1"]["selector"]["prompt_sha256"] if offer["p1"] else None
        if row["form"] == "P1":
            assert sent[p1] >= 1, f"{row['item_id']}: the priced P1 request was never sent"
            if row["checkpoint"] not in fell_back and offer["boundary"] == "H":
                assert not any(sent[h] for h in p0), f"{row['item_id']}: P0 ran beside P1"
        else:
            assert all(sent[h] >= n for h, n in p0.items()) or offer["boundary"] == "C", (
                f"{row['item_id']}: P0 was chosen but its priced requests were not all sent")
            assert p1 is None or sent[p1] == 0, f"{row['item_id']}: P1 ran though P0 was chosen"


def _assert_offer_follows_checkpoint(outcome):
    kinds = [(e["kind"], e.get("checkpoint")) for e in outcome.events]
    for i, (kind, checkpoint) in enumerate(kinds):
        if kind == "BROKER_OFFER":
            before = [k for k, c in kinds[:i] if c == checkpoint]
            after = [k for k, c in kinds[i + 1:] if c == checkpoint]
            assert before and before[-1] in {"H_CHECKPOINT", "C_CHECKPOINT"}
            assert after and after[0] == "BROKER_DECISION"


async def test_the_real_agent_offers_and_runs_only_the_decided_form(settings, tmp_path,
                                                                    no_network):
    outcome, committed, broker, offers = await _run(
        settings, tmp_path, policy=GreedyJointBroker(capacity_tokens=10**7), metrics=FakeVllm())
    assert outcome.ok, outcome.error
    decided = _decisions(offers)
    # P1 is the cheaper plan under the plans' own estimates, so the broker picks it at both.
    assert {(row["boundary"], row["form"]) for row in decided.values()} == {("H", "P1"),
                                                                            ("C", "P1")}
    assert {row["source"] for row in decided.values()} == {"broker"}
    _assert_offer_follows_checkpoint(outcome)
    _assert_only_the_chosen_form_was_sent(outcome, committed, offers)
    assert broker.host.committed, "the broker never committed a decision"
    ops = Counter(e["op_class"] for e in committed)
    assert ops["PAGE_P1_SELECTOR_BATCH"] == ops["COMPRESSOR_P1_SELECTOR"] == 1
    assert ops["PAGE_P0_SUMMARY"] == ops["COMPRESSOR_P0"] == 0, "a rejected plan was sent"
    assert not any(e["fell_back"] for e in outcome.events if e["kind"] == "PAGE_BATCH_REDUCED")


async def test_with_p1_never_chosen_the_agent_sends_exactly_the_priced_p0(settings, tmp_path,
                                                                         no_network):
    outcome, committed, _, offers = await _run(
        settings, tmp_path, name="cell-p0-only",
        policy=GreedyJointBroker(capacity_tokens=10**7, allow_p1=False), metrics=FakeVllm())
    assert outcome.ok, outcome.error
    decided = _decisions(offers)
    assert decided and {row["form"] for row in decided.values()} == {"P0"}
    summaries = Counter(e["prompt_sha256"] for e in committed if e["op_class"] == "PAGE_P0_SUMMARY")
    priced = Counter(r["prompt_sha256"] for row in decided.values()
                     if row["boundary"] == "H" for r in row["offer"]["p0"]["requests"])
    assert summaries == priced, "the summaries sent are not the summaries priced"
    assert not any(e["op_class"].endswith("SELECTOR") or "SELECTOR_" in e["op_class"]
                   for e in committed)

    reference, _, _, _ = await _run(settings, tmp_path, name="cell-explicit-p0",
                                    policy=GreedyJointBroker(), metrics=FakeVllm(),
                                    brokered=False)
    assert reference.ok and outcome.final_report == reference.final_report


async def test_admission_waits_for_the_engine_queue_to_drain(settings, tmp_path, no_network):
    metrics = FakeVllm(waiting=9)
    timer = threading.Timer(0.6, lambda: setattr(metrics, "waiting", 0.0))
    timer.start()
    try:
        outcome, _, broker, offers = await _run(
            settings, tmp_path, name="cell-gated",
            policy=EngineGatedBroker(GreedyJointBroker(capacity_tokens=10**7), max_waiting=4),
            metrics=metrics)
    finally:
        timer.cancel()
    assert outcome.ok, outcome.error
    reasons = [line for line in (tmp_path / "cell-gated" / "broker" / "ticks.jsonl")
               .read_text(encoding="utf-8").splitlines()]
    assert any("engine gate shut" in r for r in reasons), "the gate never shut"
    assert any("engine gate open" in r for r in reasons), "the gate never opened"
    assert {row["source"] for row in _decisions(offers).values()} == {"broker"}


async def test_an_agent_the_broker_never_answers_runs_p0_and_finishes(settings, tmp_path,
                                                                     no_network):
    outcome, committed, _, offers = await _run(
        settings, tmp_path, name="cell-timeout",
        policy=EngineGatedBroker(GreedyJointBroker(capacity_tokens=10**7), max_waiting=4),
        metrics=FakeVllm(waiting=9), deadline_s=0.2, grace_s=0.2)
    assert outcome.ok, outcome.error
    decided = _decisions(offers)
    assert decided and {(r["form"], r["source"], r["reason"]) for r in decided.values()} == {
        ("P0", "fail_closed", "AGENT_WAIT_TIMEOUT")}
    _assert_only_the_chosen_form_was_sent(outcome, committed, offers)


async def test_the_deadline_policy_answers_before_the_agent_gives_up(settings, tmp_path,
                                                                    no_network):
    outcome, _, _, offers = await _run(
        settings, tmp_path, name="cell-deadline",
        policy=DeadlineBroker(EngineGatedBroker(GreedyJointBroker(capacity_tokens=10**7),
                                                max_waiting=4)),
        metrics=FakeVllm(waiting=9), deadline_s=0.2, grace_s=3.0)
    assert outcome.ok, outcome.error
    decided = _decisions(offers)
    assert decided and {r["source"] for r in decided.values()} == {"broker"}
    assert all("deadline-forced" in (r["ticket"] or {}).get("reason", "")
               for r in decided.values())
