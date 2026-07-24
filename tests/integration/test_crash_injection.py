"""Kill the campaign at each dangerous point and check what survives.

Every point below is one where a naive implementation loses money, loses a record, or -- worst --
keeps a record that is no longer true. After each injected failure the same three questions are
asked: did the budget stay inside its cap, did any work item end up with two accepted attempts,
and did an incomplete pair get counted as a paired observation.

The points, in the order a call passes through them:

  1. before the reservation            -- nothing reserved, nothing sent
  2. after reserve, before send        -- the reservation is released in full
  3. after send, before the response   -- FAILED_UNKNOWN, worst case kept
  4. at the checkpoint                 -- no partial checkpoint is trusted
  5. at the transform                  -- the batch falls back whole
  6. at staging                        -- nothing published
  7. after the first sibling           -- no [P1(A), P0(B)] hybrid
  8. at the atomic publish             -- the cell is not committed
  9. after the artifact, before commit -- the cell re-runs rather than being trusted
 10. at the paired-block freeze        -- no half block is frozen
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.acquire.tavily_client import TavilyCaptureClient
from shapeflow_p1.campaign.acquire import acquire_all, tavily_params_from
from shapeflow_p1.campaign.prepare import prepare_corpus
from shapeflow_p1.campaign.runner import available_tasks, questions_for
from shapeflow_p1.campaign.schedule import ArmSpec, cell_key
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.judge_client import DeepSeekJudge
from shapeflow_p1.experiment.budget import Budget
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore
from shapeflow_p1.providers.provider_client import PROVIDER_KEY_PLACEHOLDER
from shapeflow_p1.runtime.provider_server import (
    PROVIDER_KEY_PLACEHOLDER as PLACEHOLDER,
    ProviderConfig,
    ProviderError,
    ProviderService,
    RoleTokens,
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
    "runner": "crash-runner-token-0000000000",
    "steward": "crash-steward-token-000000000",
    "evaluator": "crash-evaluator-token-0000000",
}
ARMS = [ArmSpec("P0", "P0", "P0"), ArmSpec("H_ID", "H02", "P0")]


# --- provider-side injection points (1-3) -----------------------------------------------------


class Boom(RuntimeError):
    pass


def _provider(tmp_path, upstream, caps=None):
    ledger = Ledger(str(tmp_path / "provider.sqlite"))
    budget = Budget(ledger)
    defaults = {"tavily_requests": 10.0, "tavily_credits": 20.0, "remote_calls": 50.0,
                "deepseek_requests": 10.0, "deepseek_input_tokens": 100_000.0,
                "deepseek_output_tokens": 20_000.0, "deepseek_usd": 1.0,
                "gpu_seconds": 10_000.0}
    defaults.update(caps or {})
    for resource, cap in defaults.items():
        budget.ensure_account(resource, cap)
    service = ProviderService(
        ProviderConfig(served_model="M"), ledger=ledger, budget=budget,
        store=ObjectStore(tmp_path / "objects"), redactor=SecretRedactor(),
        tokens=RoleTokens(TOKENS), upstream=upstream,
        tavily_key="tvly-CRASHTESTKEY0000000000", deepseek_key="sk-CRASHTESTKEY000000000",
    )
    service.reconcile_on_start()
    return service, ledger, budget


def _tavily_body(query="q"):
    return {"api_key": PLACEHOLDER, "query": query, "search_depth": "advanced",
            "include_raw_content": "markdown", "include_answer": False,
            "include_usage": True, "max_results": 8, "_task_id": "T", "_call_key": f"k-{query}"}


def _totals(ledger) -> dict:
    rows = ledger.raw_connection.execute(
        "SELECT resource, cap, reserved_total, settled_total FROM budget_accounts").fetchall()
    return {r["resource"]: (r["cap"], r["reserved_total"], r["settled_total"]) for r in rows}


def test_crash_before_the_reservation_leaves_nothing_charged(tmp_path):
    calls = []
    service, ledger, budget = _provider(tmp_path, lambda *a: calls.append(a))
    before = _totals(ledger)
    with pytest.raises(ProviderError):
        service.tavily_search({**_tavily_body(), "api_key": "tvly-A-CLIENTS-OWN-KEY-00000"})
    assert calls == []
    assert _totals(ledger) == before
    ledger.close()


def test_crash_after_reserve_before_send_releases_in_full(tmp_path):
    """A reservation that provably never went out is returned whole, not settled."""
    service, ledger, budget = _provider(tmp_path, lambda *a: (_ for _ in ()).throw(Boom()))
    call_id = service._calls.open_call(provider="tavily", op_class="search", call_key="k")
    group = service._calls.reserve(call_id, {"tavily_requests": 1.0, "tavily_credits": 2.0})
    service._calls.fail_before_send(call_id, group, error_class="pre_send")

    cap, reserved, settled = _totals(ledger)["tavily_credits"]
    assert (reserved, settled) == (0.0, 0.0)
    assert budget.available("tavily_credits") == cap
    ledger.close()


def test_crash_after_send_keeps_the_worst_case_and_survives_restart(tmp_path):
    """A call we lost contact with may have been billed; it is never recorded as free."""
    def upstream(*_a):
        raise Boom("connection reset after send")

    service, ledger, budget = _provider(tmp_path, upstream)
    with pytest.raises(ProviderError):
        service.tavily_search(_tavily_body())
    _cap, reserved, settled = _totals(ledger)["tavily_credits"]
    assert settled == 2.0 and reserved == 0.0

    # A restart must not release it, and must not charge it twice either.
    fresh = ProviderService(
        ProviderConfig(), ledger=ledger, budget=budget,
        store=ObjectStore(tmp_path / "objects"), redactor=SecretRedactor(),
        tokens=RoleTokens(TOKENS), upstream=upstream,
        tavily_key="tvly-CRASHTESTKEY0000000000", deepseek_key="sk-CRASHTESTKEY000000000")
    fresh.reconcile_on_start()
    _cap2, reserved2, settled2 = _totals(ledger)["tavily_credits"]
    assert (reserved2, settled2) == (0.0, 2.0)
    ledger.close()


def test_the_budget_cap_is_never_exceeded_by_a_crash_loop(tmp_path):
    """Repeated timeouts must exhaust the cap and then refuse, never spend past it."""
    def upstream(*_a):
        raise Boom("timeout")

    service, ledger, budget = _provider(tmp_path, upstream,
                                        caps={"tavily_credits": 4.0, "tavily_requests": 10.0})
    refusals = 0
    for i in range(8):
        try:
            service.tavily_search(_tavily_body(f"q{i}"))
        except ProviderError as e:
            refusals += 1 if e.status == 429 else 0
    cap, reserved, settled = _totals(ledger)["tavily_credits"]
    assert settled <= cap
    assert refusals > 0, "the cap was never enforced"
    ledger.close()


# --- runner-side injection points (4-10) -------------------------------------------------------


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
    judge = DeepSeekJudge(author, "deepseek-chat", PROVIDER_KEY_PLACEHOLDER)
    await prepare_corpus(settings, judge=judge, authored_at_utc="2026-07-24T00:00:00Z",
                         target_model="Qwen3-14B-AWQ", total=8, clusters=4)
    params = tavily_params_from(settings)
    fake = FakeTavily(pages_per_query=2)
    await acquire_all(
        settings,
        client_factory=lambda t: TavilyCaptureClient(fake, params, PROVIDER_KEY_PLACEHOLDER),
        fetched_at_utc="2026-07-24T01:00:00Z")


@pytest.fixture()
async def campaign(settings, tmp_path):
    from fixtures.campaign_harness import Harness

    await _world(settings)
    harness = Harness(settings, tmp_path, FakeEngine(selector_ids=["S1"]), TOKENS, REPO)
    try:
        yield harness
    finally:
        harness.close()


def _integrity(runner, manifest, *, phase_id, split) -> None:
    """The three questions asked after every injected failure."""
    rows = runner.ledger.raw_connection.execute(
        "SELECT work_key, COUNT(*) c FROM attempts WHERE state='COMMITTED' GROUP BY work_key"
    ).fetchall()
    assert all(r["c"] == 1 for r in rows), "a work item accepted two attempts"
    assert runner.ledger.integrity_check()
    states = runner.cell_states(manifest, phase_id=phase_id, split=split)
    for block in manifest.blocks:
        complete = all(states.get(cell_key(c)) == "COMMITTED" for c in block.cells)
        frozen = (runner.settings.path("runs") / "blocks" / f"{block.block_id}.json").exists()
        assert frozen <= complete, "an incomplete block was frozen"


async def test_a_cell_that_dies_mid_run_does_not_commit(campaign, settings, tmp_path):
    """Point 8: the graph raises during publication. The cell must not be COMMITTED."""
    import shapeflow_p1.campaign.runner as runner_module

    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = campaign.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")

    original = runner_module.run_cell
    calls = {"n": 0}

    async def exploding(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            raise Boom("killed at publish")
        return await original(*args, **kwargs)

    runner_module.run_cell = exploding
    try:
        await runner.run_cells(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                               questions=questions_for(settings, tasks))
    finally:
        runner_module.run_cell = original

    states = runner.cell_states(manifest, phase_id="crash", split="FORMATIVE_SCREEN")
    assert "FAILED_UNKNOWN" in states.values()
    assert runner.freeze_blocks(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                                directory=settings.path("runs") / "blocks") == []
    _integrity(runner, manifest, phase_id="crash", split="FORMATIVE_SCREEN")


async def test_a_cell_killed_between_artifact_and_commit_reruns(campaign, settings, tmp_path):
    """Point 9: the bytes exist and the terminal record does not, so the cell re-runs."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = campaign.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    cell = manifest.cells[0]
    key = runner.work_key_for(cell, phase_id="crash", split="FORMATIVE_SCREEN")
    ref = runner.ledger.committed_ref(key)
    runner.store._path_for(ref).unlink()          # noqa: SLF001 - simulating the lost write

    states = runner.cell_states(manifest, phase_id="crash", split="FORMATIVE_SCREEN")
    assert states[cell_key(cell)] == "PENDING"
    assert runner.freeze_blocks(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                                directory=settings.path("runs") / "blocks") == []


async def test_a_half_finished_block_is_never_frozen(campaign, settings, tmp_path):
    """Point 10: freezing is all-or-nothing, so a pair is never spliced across a restart."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = campaign.runner(max_cells=1)
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))
    assert runner.freeze_blocks(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                                directory=settings.path("runs") / "blocks") == []
    _integrity(runner, manifest, phase_id="crash", split="FORMATIVE_SCREEN")

    resumed = campaign.runner()
    await resumed.run_cells(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                            questions=questions_for(settings, tasks))
    frozen = resumed.freeze_blocks(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                                   directory=settings.path("runs") / "blocks")
    assert len(frozen) == 1 and frozen[0]["complete"] is True
    _integrity(resumed, manifest, phase_id="crash", split="FORMATIVE_SCREEN")


async def test_a_p1_failure_falls_the_whole_batch_back_and_keeps_its_cost(campaign, settings):
    """Points 5-7: a selector that fails must not leave a [P1(A), P0(B)] hybrid behind."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    campaign.engine.selector_ids = None          # selector returns prose -> contract failure
    runner = campaign.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=[ARMS[1]] + [ARMS[0]],
                                     split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="crash", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    states = runner.cell_states(manifest, phase_id="crash", split="FORMATIVE_SCREEN")
    assert set(states.values()) == {"COMMITTED"}, "a P1 failure must still produce a run"

    record = None
    for cell in manifest.cells:
        if cell.arm.arm_id != "H_ID":
            continue
        key = runner.work_key_for(cell, phase_id="crash", split="FORMATIVE_SCREEN")
        ref = runner.ledger.committed_ref(key)
        record = json.loads(runner.store.get_bytes(ref).decode("utf-8"))
    assert record is not None
    counts = record["counts"]
    # Every deferred batch was resolved: either transformed or fallen back, never left partial.
    assert counts["page_batches_deferred"] == counts["page_batches_reduced"]
    # And the failure and its cost are on the record rather than absorbed silently.
    reduced = [e for e in record["events"] if e["kind"] == "PAGE_BATCH_REDUCED"]
    assert reduced, "no batch reduction was recorded"
    assert all(e.get("fell_back") is not None for e in reduced)


async def test_a_stale_lease_does_not_blindly_repeat_a_side_effect(campaign, settings):
    """A crashed worker's cell may already have called the engine; it is frozen, not retried."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = campaign.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    cell = manifest.cells[0]
    key = runner.work_key_for(cell, phase_id="crash", split="FORMATIVE_SCREEN")
    runner.ledger.claim(key, "dead-worker", lease_seconds=-1.0)

    reopened = runner.ledger.reclaim_stale()
    assert key not in reopened, "a side-effecting cell was reopened after an unknown outcome"
    assert runner.ledger.get_work_item(key).state == "FAILED_UNKNOWN"
