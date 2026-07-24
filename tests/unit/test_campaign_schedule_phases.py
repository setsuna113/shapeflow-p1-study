"""The frozen schedule and the durable phase state.

Each assertion covers a way the campaign could produce numbers that look fine: a block without
its comparator, an arm order that tracks execution time, a plan that changes after results
exist, a phase that completes twice with different evidence, a resume that repeats finished work.
"""

from __future__ import annotations

import json

import pytest

from shapeflow_p1.campaign.phases import PhaseError, PhaseStore
from shapeflow_p1.campaign.schedule import (
    ArmSpec,
    block_is_complete,
    build_blocks,
    cell_key,
    cells_needing_work,
    freeze_record,
)
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.experiment.state_machine import IllegalPhaseTransition, Phase

ARMS = [
    ArmSpec("P0", "P0", "P0"),
    ArmSpec("H_ID", "H02", "P0"),
    ArmSpec("C_VISIBLE", "P0", "C01"),
    ArmSpec("H_PLUS_C", "H02", "C01"),
]
TASKS = ["T001", "T002", "T003"]


def _manifest(**kw):
    params = dict(
        protocol_sha="a" * 64, split="FORMATIVE_SCREEN", task_ids=TASKS, arms=ARMS,
        seeds=[1, 2], layer="causal", claim_scope="FORMATIVE_ONLY",
    )
    params.update(kw)
    return build_blocks(**params)


# --- the schedule ----------------------------------------------------------------------------


def test_every_block_carries_its_comparator():
    manifest = _manifest()
    for block in manifest.blocks:
        assert any(c.arm.arm_id == "P0" for c in block.cells)


def test_a_block_without_p0_is_refused():
    """A P1 compared against an arm that ran somewhere else is not a paired observation."""
    with pytest.raises(ValueError, match="must contain P0"):
        _manifest(arms=[ArmSpec("H_ID", "H02", "P0")])


def test_the_schedule_is_deterministic_from_the_protocol_and_tasks():
    assert _manifest().digest == _manifest().digest
    # A different protocol SHA is a different experiment and must plan different work.
    assert _manifest(protocol_sha="b" * 64).digest != _manifest().digest


def test_arm_order_differs_between_blocks():
    """One fixed order would let a drift over execution land on the same arm every time."""
    orders = [
        tuple(c.arm.arm_id for c in sorted(b.cells, key=lambda c: c.order_index))
        for b in _manifest().blocks
    ]
    assert len(set(orders)) > 1


def test_a_second_replicate_is_chosen_before_anything_runs():
    with_second = _manifest(second_seed_fraction=1.0)
    assert {b.replicate_id for b in with_second.blocks} == {"0", "1"}
    # Deterministic: the same protocol and tasks select the same tasks every time.
    assert with_second.digest == _manifest(second_seed_fraction=1.0).digest
    assert {b.replicate_id for b in _manifest(second_seed_fraction=0.0).blocks} == {"0"}


def test_a_block_is_complete_only_when_every_cell_committed():
    manifest = _manifest()
    block = manifest.blocks[0]
    states = {cell_key(c): "COMMITTED" for c in block.cells}
    assert block_is_complete(block, states)

    states[cell_key(block.cells[0])] = "FAILED_FINAL"
    assert not block_is_complete(block, states)
    # An unknown external outcome is not a completion either.
    states[cell_key(block.cells[0])] = "FAILED_UNKNOWN"
    assert not block_is_complete(block, states)


def test_the_freeze_record_carries_every_cell_state():
    manifest = _manifest()
    block = manifest.blocks[0]
    states = {cell_key(c): "COMMITTED" for c in block.cells}
    outputs = {cell_key(c): f"ref-{c.arm.arm_id}" for c in block.cells}
    record = freeze_record(block, states=states, outputs=outputs)
    assert record["complete"] is True
    assert len(record["cells"]) == len(block.cells)
    assert all(c["output_ref"] for c in record["cells"])
    assert len(record["freeze_sha256"]) == 64


def test_resume_only_lists_the_cells_that_are_not_done():
    manifest = _manifest()
    states = {cell_key(c): "COMMITTED" for c in manifest.cells}
    assert cells_needing_work(manifest, states) == []
    first = manifest.cells[0]
    states[cell_key(first)] = "PENDING"
    assert [c.arm.arm_id for c in cells_needing_work(manifest, states)] == [first.arm.arm_id]


# --- phases -----------------------------------------------------------------------------------


@pytest.fixture()
def store(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    yield PhaseStore(ledger, protocol_sha="a" * 64)
    ledger.close()


def test_a_phase_survives_a_restart(tmp_path):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    first = PhaseStore(ledger, protocol_sha="a" * 64)
    first.begin(Phase.DOCTOR_PASSED)
    first.complete(Phase.DOCTOR_PASSED, {"checks": 12})
    ledger.close()

    ledger = Ledger(str(tmp_path / "l.sqlite"))
    second = PhaseStore(ledger, protocol_sha="a" * 64)
    assert second.is_complete(Phase.DOCTOR_PASSED)
    assert second.current() is Phase.DOCTOR_PASSED
    ledger.close()


def test_completing_a_phase_twice_with_different_evidence_is_fatal(store):
    store.begin(Phase.DOCTOR_PASSED)
    digest = store.complete(Phase.DOCTOR_PASSED, {"checks": 12})
    assert store.complete(Phase.DOCTOR_PASSED, {"checks": 12}) == digest
    with pytest.raises(PhaseError, match="two answers"):
        store.complete(Phase.DOCTOR_PASSED, {"checks": 13})


def test_a_gate_cannot_be_skipped_by_starting_the_phase_after_it(store):
    store.begin(Phase.DOCTOR_PASSED)
    store.complete(Phase.DOCTOR_PASSED, {})
    with pytest.raises(IllegalPhaseTransition):
        store.begin(Phase.SCREEN_RUNNING)


def test_the_legal_path_to_screening_runs_through_parity_and_smoke(store):
    for phase in (Phase.DOCTOR_PASSED, Phase.ACQUISITION_COMPLETE, Phase.SNAPSHOTS_FROZEN,
                  Phase.P0_PARITY_PASSED, Phase.GPU_SMOKE_PASSED, Phase.SCREEN_RUNNING):
        store.begin(phase)
        store.complete(phase, {"phase": phase.value})
    assert store.current() is Phase.SCREEN_RUNNING
    assert [t["to_phase"] for t in store.history()][:2] == ["DOCTOR_PASSED",
                                                           "ACQUISITION_COMPLETE"]


def test_a_failed_phase_is_recorded_not_erased(store):
    store.begin(Phase.DOCTOR_PASSED)
    store.fail(Phase.DOCTOR_PASSED, "stack mismatch")
    record = store.record(Phase.DOCTOR_PASSED)
    assert record.state == "FAILED"
    assert json.dumps(record.detail).count("stack mismatch") == 1
    assert not store.is_complete(Phase.DOCTOR_PASSED)
