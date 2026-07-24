"""The campaign runner end to end: cells, blocks, resume, and freezing only what is whole.

Driven against the real provider and vendor's real compiled graph with a deterministic engine,
so what is exercised is the actual execution path -- ledger keys, artifact verification, the
whole-batch publish and the block freeze -- rather than a simulation of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.acquire.tavily_client import TavilyCaptureClient
from shapeflow_p1.campaign.acquire import acquire_all, tavily_params_from
from shapeflow_p1.campaign.prepare import prepare_corpus
from shapeflow_p1.campaign.runner import (
    CampaignRunner,
    RunnerConfig,
    available_tasks,
    questions_for,
    write_status,
)
from shapeflow_p1.campaign.schedule import ArmSpec, cell_key
from shapeflow_p1.campaign.selector_client import SelectorModelCall
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.judge_client import DeepSeekJudge
from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.provider_client import ProviderClient
from shapeflow_p1.runtime.provider_server import (
    ProviderConfig,
    ProviderService,
    RoleTokens,
    serve_forever,
)
from shapeflow_p1.secrets import SecretRedactor

from fixtures.fake_engine import FakeEngine
from fixtures.fake_tavily import FakeTavily
from fixtures.scripted_author import ScriptedAuthor

REPO = Path(__file__).resolve().parents[2]
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"

pytestmark = pytest.mark.skipif(
    not PATCHED.exists(), reason="ODR not materialized; run scripts/materialize_vendor.sh")

TOKENS = {
    "runner": "campaign-runner-token-0000000",
    "steward": "campaign-steward-token-000000",
    "evaluator": "campaign-evaluator-token-0000",
}

ARMS = [ArmSpec("P0", "P0", "P0"), ArmSpec("H_ID", "H02", "P0")]


@pytest.fixture()
def settings(tmp_path):
    s = Settings.load(REPO, data_root=tmp_path)
    relaxed = dict(s.configs["task_source"])
    relaxed["audit"] = {**relaxed["audit"], "require_distinct_topic_clusters": 4}
    relaxed["splits"] = {"FORMATIVE_SCREEN": 4, "FORMATIVE_POWER_PILOT": 2, "RESERVE": 2}
    relaxed["strata_min_counts"] = {"source_conflict": 1}
    s.configs["task_source"] = relaxed
    return s


async def _world(settings):
    author = ScriptedAuthor(clusters=4, per_cluster=2)
    judge = DeepSeekJudge(author, "deepseek-chat", "@SHAPEFLOW_PROVIDER@")
    await prepare_corpus(settings, judge=judge, authored_at_utc="2026-07-24T00:00:00Z",
                         target_model="Qwen3-14B-AWQ", total=8, clusters=4)
    params = tavily_params_from(settings)
    fake = FakeTavily(pages_per_query=2)
    await acquire_all(
        settings,
        client_factory=lambda task_id: TavilyCaptureClient(fake, params, "@SHAPEFLOW_PROVIDER@"),
        fetched_at_utc="2026-07-24T01:00:00Z")


class Harness:
    """Provider + ledger + runner, wired the way the campaign wires them."""

    def __init__(self, settings, tmp_path, engine):
        self.settings = settings
        self.engine = engine
        self.provider_ledger = Ledger(str(tmp_path / "provider.sqlite"))
        budget = Budget(self.provider_ledger)
        for resource, cap in settings.budget_caps().items():
            budget.ensure_account(resource, cap)
        redactor = SecretRedactor()
        self.service = ProviderService(
            ProviderConfig(served_model="Qwen3-14B-AWQ"),
            ledger=self.provider_ledger, budget=budget,
            store=ObjectStore(tmp_path / "provider-objects"), redactor=redactor,
            tokens=RoleTokens(TOKENS), upstream=engine, tavily_key=None, deepseek_key=None,
        )
        self.service.reconcile_on_start()
        self.tcp, _ = serve_forever(self.service, ProviderConfig(bind_port=0), redactor)
        self.base = f"http://127.0.0.1:{self.tcp.server_address[1]}"
        self.client = ProviderClient(base_url=self.base, token=TOKENS["runner"])
        self.run_ledger = Ledger(str(tmp_path / "run.sqlite"))
        self.store = ObjectStore(tmp_path / "run-objects")

    def runner(self, **kw):
        async def register(spec):
            await self.client.register_cell(
                cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
                arm_id=spec.arm_id, variant_id=spec.variant_id,
                replicate_id=spec.replicate_id, work_key=spec.work_key)

        def model_call_factory(cell_token):
            return SelectorModelCall(
                self.client, cell_token=cell_token, repo=REPO, temperature=0.0, top_p=1.0,
                max_completion_tokens=int(self.settings.get(
                    "week1", "measurement", "selector_max_completion_tokens")),
            )

        config = RunnerConfig(run_id="RUN-TEST", provider_base_url=self.base,
                              runner_token=TOKENS["runner"], **kw)
        runner = CampaignRunner(
            self.settings, ledger=self.run_ledger, store=self.store, config=config,
            model_call=model_call_factory("cell-shared-000001"), register_cell=register)
        return runner

    def close(self):
        self.tcp.shutdown()
        self.provider_ledger.close()
        self.run_ledger.close()


@pytest.fixture()
async def harness(settings, tmp_path):
    await _world(settings)
    engine = FakeEngine(selector_ids=["S1"])
    h = Harness(settings, tmp_path, engine)
    try:
        yield h
    finally:
        h.close()


async def test_a_block_of_two_arms_runs_and_freezes(harness, settings, tmp_path):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    assert tasks
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    schedule_sha = runner.freeze_schedule(manifest, tmp_path / "schedule.json")
    assert len(schedule_sha) == 64
    assert len(manifest.blocks) == 1
    assert len(manifest.cells) == 2

    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))
    states = runner.cell_states(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN")
    assert set(states.values()) == {"COMMITTED"}, states

    frozen = runner.freeze_blocks(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                  directory=tmp_path / "blocks")
    assert len(frozen) == 1
    record = frozen[0]
    assert record["complete"] is True
    assert {c["arm"]["arm_id"] for c in record["cells"]} == {"P0", "H_ID"}
    assert all(c["output_ref"] for c in record["cells"])
    assert (tmp_path / "blocks" / f"{record['block_id']}.json").exists()


async def test_an_incomplete_block_is_not_frozen(harness, settings, tmp_path):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner(max_cells=1)     # stop after the first cell of the block
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    states = runner.cell_states(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN")
    assert list(states.values()).count("COMMITTED") == 1
    frozen = runner.freeze_blocks(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                  directory=tmp_path / "blocks")
    assert frozen == [], "a half-finished block was frozen as a paired observation"


async def test_resume_fills_the_gap_without_repeating_finished_work(harness, settings, tmp_path):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    first = harness.runner(max_cells=1)
    manifest = first.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    questions = questions_for(settings, tasks)
    await first.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                          questions=questions)
    requests_after_first = len(harness.engine.requests)
    committed_first = {k: v for k, v in first.cell_states(
        manifest, phase_id="run-screen", split="FORMATIVE_SCREEN").items() if v == "COMMITTED"}
    assert len(committed_first) == 1

    second = harness.runner()               # a fresh runner, as a restart would build
    outcomes = await second.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                      questions=questions)
    assert len(outcomes) == 1, "resume re-ran a cell that was already committed"
    assert len(harness.engine.requests) > requests_after_first

    states = second.cell_states(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN")
    assert set(states.values()) == {"COMMITTED"}
    # Exactly one committed attempt per cell -- the ledger's unique index, checked from outside.
    rows = second.ledger.raw_connection.execute(
        "SELECT work_key, COUNT(*) c FROM attempts WHERE state='COMMITTED' GROUP BY work_key"
    ).fetchall()
    assert all(r["c"] == 1 for r in rows)


async def test_a_committed_cell_with_missing_bytes_is_rerun(harness, settings, tmp_path):
    """The database saying done and the bytes disagreeing is not a completion."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    cell = manifest.cells[0]
    key = runner.work_key_for(cell, phase_id="run-screen", split="FORMATIVE_SCREEN")
    ref = runner.ledger.committed_ref(key)
    blob = runner.store._path_for(ref)      # noqa: SLF001 - deliberately corrupting a blob
    blob.unlink()

    states = runner.cell_states(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN")
    assert states[cell_key(cell)] == "PENDING"
    assert runner.freeze_blocks(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                directory=tmp_path / "blocks") == []


async def test_every_cell_is_attributed_in_the_provider_ledger(harness, settings):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    committed = [e for e in harness.service.events if e["kind"] == "INFERENCE_COMMITTED"]
    assert committed
    assert {e["arm_id"] for e in committed} == {"P0", "H_ID"}
    assert all(e["work_key"] for e in committed)
    # Reservations all closed: nothing left holding budget it never settled.
    open_calls = harness.provider_ledger.raw_connection.execute(
        "SELECT COUNT(*) c FROM external_calls WHERE state NOT IN"
        " ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN')").fetchone()["c"]
    assert open_calls == 0


async def test_an_unregistered_arm_never_runs(harness, settings):
    """An arm nobody pre-registered must not execute, or the design no longer describes the run."""
    runner = harness.runner()
    assert {a.arm_id for a in runner.arms_from_config("canary")} == {
        "P0", "H_ID", "H_TYPED", "C_VISIBLE", "H_PLUS_C", "CPU_LEXICAL", "SHORT_PROSE"}

    week1 = dict(settings.configs["week1"])
    week1["canary"] = {**week1["canary"], "arms": [
        {"arm_id": "ROGUE", "page_variant": "H99-NOT-REGISTERED", "close_variant": "P0"},
    ]}
    settings.configs["week1"] = week1
    with pytest.raises(ValueError, match="unregistered variants"):
        runner.arms_from_config("canary")


async def test_status_reports_blocks_and_cells(harness, settings, tmp_path):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    body = runner.status(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN")
    assert body["cells_total"] == 2 and body["cells_committed"] == 2
    assert body["blocks_complete"] == 1
    assert body["claim_scope"] == "FORMATIVE_ONLY"

    path = tmp_path / "STATUS.json"
    write_status(body, path)
    assert json.loads(path.read_text())["run_id"] == "RUN-TEST"


async def test_a_stop_request_halts_admission_without_losing_finished_cells(
    harness, settings, tmp_path
):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:2]
    sentinel = tmp_path / "STOP_REQUESTED"
    runner = harness.runner(stop_sentinel=sentinel)
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    questions = questions_for(settings, tasks)

    sentinel.touch()
    outcomes = await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                      questions=questions)
    assert outcomes == []

    sentinel.unlink()
    outcomes = await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                      questions=questions)
    assert len(outcomes) == len(manifest.cells)
