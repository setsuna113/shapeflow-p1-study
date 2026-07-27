"""The campaign runner end to end: cells, blocks, resume, and freezing only what is whole.

Driven against the real provider and vendor's real compiled graph with a deterministic engine,
so what is exercised is the actual execution path -- ledger keys, artifact verification, the
whole-batch publish and the block freeze -- rather than a simulation of it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fixtures.campaign_harness import Harness
from fixtures.fake_engine import FakeEngine
from fixtures.frozen_world import write_frozen_world

from shapeflow.campaign.runner import available_tasks, questions_for, write_status
from shapeflow.campaign.schedule import ArmSpec, cell_key
from shapeflow.campaign.settings import Settings

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
    return s


def _world(settings):
    """Two tasks with frozen worlds, written straight to the runner-readable view."""
    write_frozen_world(settings, task_ids=["T-CAMPAIGN-1", "T-CAMPAIGN-2"], pages_per_task=2)


@pytest.fixture()
async def harness(settings, tmp_path):
    _world(settings)
    engine = FakeEngine(selector_ids=["S1"])
    h = Harness(settings, tmp_path, engine, TOKENS, REPO)
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
    for cell in manifest.cells:
        work_key = runner.work_key_for(
            cell, phase_id="run-screen", split="FORMATIVE_SCREEN")
        output_ref = runner.ledger.committed_ref(work_key)
        output = json.loads(runner.store.get_bytes(output_ref).decode("utf-8"))
        assert output["e2e_latency_seconds"] >= 0.0
        assert output["work_summary"]["e2e_latency_seconds"] == (
            output["e2e_latency_seconds"])

    frozen = runner.freeze_blocks(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                  directory=tmp_path / "blocks")
    assert len(frozen) == 1
    record = frozen[0]
    assert record["complete"] is True
    assert {c["arm"]["arm_id"] for c in record["cells"]} == {"P0", "H_ID"}
    assert all(c["output_ref"] for c in record["cells"])
    assert (tmp_path / "blocks" / f"{record['block_id']}.json").exists()
    root = json.loads((tmp_path / "blocks" / "FREEZE_ROOT.json").read_text())
    assert root["run_id"] == "RUN-TEST"
    assert root["phase_id"] == "run-screen"
    assert root["execution_binding_sha256"] == runner.execution_binding_sha256
    assert root["protocol_sha256"] == runner.protocol_document_sha256
    assert root["schedule_sha256"] == manifest.digest
    assert root["terminal_frozen"] is True
    assert len(root["blocks"]) == len(manifest.blocks)
    assert {
        (cell["arm"]["arm_id"], cell["arm"]["page_variant"],
         cell["arm"]["close_variant"], cell["seed"], cell["order_index"])
        for block in root["blocks"] for cell in block["cells"]
    } == {
        (cell.arm.arm_id, cell.arm.page_variant, cell.arm.close_variant,
         cell.seed, cell.order_index)
        for cell in manifest.cells
    }
    assert all(
        cell["engine_epoch"]
        for block in root["blocks"] for cell in block["cells"])


async def test_an_anchor_cell_captures_the_continuation_a_c_fork_needs(
    harness, settings, tmp_path
):
    """The C fork's report input has to be taken mid-run, on the actual graph.

    ``final_report_generation`` returns ``{"notes": {"type": "override", "value": []}}``, so
    by the time the graph returns, the note vector every fork substitutes into is gone. The
    supervisor's ``ConductResearch`` tool-call ids -- which say *which* note each child
    produced -- are only visible at the node update that emitted them. This asserts both
    survive a real run, because the fork is unbuildable without them.
    """
    from shapeflow.odr.continuation import ContinuationStore, child_slot_key

    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")

    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))
    states = runner.cell_states(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN")
    assert set(states.values()) == {"COMMITTED"}, states

    store = ContinuationStore(settings.path("checkpoints") / "continuations")
    stored = sorted(p.stem for p in store.root.rglob("*.json"))
    assert stored, "no anchor continuation survived the run"

    envelope = store.get(stored[0])          # get() re-derives and verifies the digest
    assert envelope.research_brief
    assert envelope.root_messages
    assert envelope.anchor_engine_epoch, "the pair has no engine boot to be voided against"

    slots = envelope.notes
    assert slots, "the note vector was empty by the time it was captured"
    forkable = [slot for slot in slots if slot.tool_call_id]
    assert forkable, "no note could be attributed to a ConductResearch child"

    # Ordinals are dense and ordered, and every child's slot key is the one its researcher_id
    # carries -- that identity is what makes the substitution exact under concurrency.
    assert [slot.ordinal for slot in slots] == list(range(len(slots)))
    for slot in forkable:
        assert slot.slot_key == child_slot_key(slot.tool_call_id)
    assert len({slot.slot_key for slot in forkable}) == len(forkable), "ambiguous slot keys"

    # And a stored envelope whose notes were edited must not load at all. Tamper the file
    # backing *this* digest, then restore it, so the corruption cannot leak into another test.
    import json as _json

    path = store._path(stored[0])
    original = path.read_text(encoding="utf-8")
    try:
        body = _json.loads(original)
        body["notes"][0]["content_sha256"] = "0" * 64
        path.write_text(_json.dumps(body), encoding="utf-8")
        with pytest.raises(ValueError, match="not the one it claims"):
            store.get(stored[0])
    finally:
        path.write_text(original, encoding="utf-8")
    assert store.get(stored[0]).digest == stored[0]


async def test_new_execution_binding_cannot_reuse_old_committed_cell(harness, settings):
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    old = harness.runner(execution_binding_sha256="e" * 64)
    old_manifest = old.build_schedule(
        task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    old_cell = old_manifest.cells[0]
    old_key = old.work_key_for(
        old_cell, phase_id="run-screen", split="FORMATIVE_SCREEN")
    attempt = old.ledger.claim(old_key, "test", lease_seconds=60, run_id="RUN-TEST")
    ref = old.store.put_bytes(b"old approved execution")
    old.ledger.register_artifact(
        ref.key, kind="cell_output", raw_size=ref.raw_size,
        stored_size=ref.stored_size, work_key=old_key)
    old.ledger.advance(attempt.attempt_id, "MATERIALIZED")
    old.ledger.advance(attempt.attempt_id, "VALIDATED")
    old.ledger.commit(attempt.attempt_id, result_object_ref=ref.key)

    new = harness.runner(execution_binding_sha256="f" * 64)
    new_manifest = new.build_schedule(
        task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    new_cell = new_manifest.cells[0]
    new_key = new.work_key_for(
        new_cell, phase_id="run-screen", split="FORMATIVE_SCREEN")

    assert new_manifest.digest != old_manifest.digest
    assert new_cell.block_id != old_cell.block_id
    assert new_key != old_key
    assert new.ledger.get_work_item(new_key).state == "PENDING"


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
    assert not (tmp_path / "blocks" / "FREEZE_ROOT.json").exists()


async def test_a_terminal_failure_block_is_frozen_for_itt(harness, settings, tmp_path):
    """P1 failure stays in the offered denominator instead of deleting the whole task."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    for index, cell in enumerate(manifest.cells):
        work_key = runner.work_key_for(
            cell, phase_id="run-screen", split="FORMATIVE_SCREEN")
        attempt = runner.ledger.claim(
            work_key, "test", lease_seconds=60, run_id="RUN-TEST")
        runner.ledger.record_engine_epoch(work_key, runner.engine_epoch)
        ref = runner.store.put_bytes(
            json.dumps({"cell": cell.content(), "index": index}).encode())
        runner.ledger.register_artifact(
            ref.key, kind="cell_output", raw_size=ref.raw_size,
            stored_size=ref.stored_size, work_key=work_key)
        if index == 0:
            runner.ledger.fail(
                attempt.attempt_id, disposition="FAILED_FINAL",
                error_class="selector_error", result_object_ref=ref.key)
        else:
            runner.ledger.advance(attempt.attempt_id, "MATERIALIZED")
            runner.ledger.advance(attempt.attempt_id, "VALIDATED")
            runner.ledger.commit(attempt.attempt_id, result_object_ref=ref.key)

    frozen = runner.freeze_blocks(
        manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
        directory=tmp_path / "blocks")
    assert len(frozen) == 1
    assert frozen[0]["terminal_frozen"] is True
    assert frozen[0]["complete_success"] is False
    failed = [c for c in frozen[0]["cells"] if c["state"] == "FAILED_FINAL"]
    assert len(failed) == 1 and failed[0]["output_ref"]


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
    assert all(
        request["body"].get("seed") == 1 for request in harness.engine.requests
    ), "the replicate seed was recorded but not sent on every ODR/selector request"
    # Reservations all closed: nothing left holding budget it never settled.
    open_calls = harness.provider_ledger.raw_connection.execute(
        "SELECT COUNT(*) c FROM external_calls WHERE state NOT IN"
        " ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN')").fetchone()["c"]
    assert open_calls == 0


async def test_an_unregistered_arm_never_runs(harness, settings):
    """An arm nobody pre-registered must not execute, or the design no longer describes the run."""
    runner = harness.runner()
    configured = {
        str(arm["arm_id"])
        for arm in settings.get("week1", "screen_arms", "arms")
    }
    assert {a.arm_id for a in runner.arms_from_config("canary")} == configured

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


async def test_a_block_that_spans_two_engine_epochs_is_frozen_but_invalid(harness, settings,
                                                                          tmp_path):
    """Complete is not the same as one observation. A block half-run before an engine
    restart and half after compares two arms served by two engines, and freeze_blocks used
    to accept it because every cell said COMMITTED."""
    tasks = available_tasks(settings, "FORMATIVE_SCREEN")[:1]
    runner = harness.runner()
    manifest = runner.build_schedule(task_ids=tasks, arms=ARMS, split="FORMATIVE_SCREEN")
    await runner.run_cells(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, tasks))

    block = manifest.blocks[0]
    key = runner.work_key_for(block.cells[0], phase_id="run-screen",
                              split="FORMATIVE_SCREEN")
    with runner.ledger.transaction() as cur:
        cur.execute("UPDATE cell_epochs SET engine_epoch='a-different-engine'"
                    " WHERE work_key=?", (key,))

    frozen = runner.freeze_blocks(manifest, phase_id="run-screen", split="FORMATIVE_SCREEN",
                                  directory=tmp_path / "blocks")
    record = next(r for r in frozen if r["block_id"] == block.block_id)
    assert record["valid_for_paired_estimate"] is False
    assert record["invalid_reason"] == "SPANS_ENGINE_EPOCHS"
    incident = runner.ledger.raw_connection.execute(
        "SELECT kind FROM incidents WHERE kind='block_spans_engine_epochs'").fetchone()
    assert incident is not None


def test_logical_cell_key_does_not_change_when_vllm_restarts():
    """Boot identity is execution provenance, not a license to duplicate an assignment."""
    from shapeflow.experiment.ledger import Ledger

    coordinates = {
        "protocol_sha": "p" * 64,
        "split": "FORMATIVE_SCREEN",
        "phase_id": "e2e-screen",
        "task_id": "T1",
        "arm_id": "H",
        "variant_id": "H02+P0",
        "replicate_id": "0",
        "checkpoint_hash": "B1",
        "stage_version": "v1",
    }
    assert Ledger.work_key(**coordinates, engine_epoch="a" * 32) == Ledger.work_key(
        **coordinates, engine_epoch="b" * 32)


def test_runner_records_a_within_cell_engine_restart_as_an_invalid_epoch(
    harness, monkeypatch, tmp_path
):
    runner = harness.runner()
    runner._engine_epoch_override = None
    epoch_file = tmp_path / "engine_epoch"
    epoch_file.write_text("a" * 32 + "\n", encoding="ascii")
    monkeypatch.setenv("SHAPEFLOW_ENGINE_EPOCH_FILE", str(epoch_file))
    work_key = runner.ledger.ensure_work_item(
        protocol_sha="p" * 64, split="s", phase_id="phase", task_id="t",
        arm_id="a", variant_id="v")
    start = runner._current_engine_epoch()
    epoch_file.write_text("b" * 32 + "\n", encoding="ascii")
    recorded, end, stable = runner._execution_epoch(work_key, start)
    assert not stable
    assert end == "b" * 32
    assert recorded == f"SPANS:{'a' * 32}:{'b' * 32}"
