"""The frozen schedule and the durable phase state.

Each assertion covers a way the campaign could produce numbers that look fine: a block without
its comparator, an arm order that tracks execution time, a plan that changes after results
exist, a phase that completes twice with different evidence, a resume that repeats finished work.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter

import pytest

from shapeflow.campaign.phases import PhaseError, PhaseStore
from shapeflow.campaign.schedule import (
    ArmSpec,
    block_is_complete,
    block_is_terminal,
    build_blocks,
    cell_key,
    cells_needing_work,
    freeze_record,
)
from shapeflow.experiment.ledger import Ledger, LedgerError
from shapeflow.experiment.state_machine import IllegalPhaseTransition, Phase

ARMS = [
    ArmSpec("P0", "P0", "P0"),
    ArmSpec("H_ID", "H02", "P0"),
    ArmSpec("C_VISIBLE", "P0", "C01"),
    ArmSpec("H_PLUS_C", "H02", "C01"),
]
TASKS = ["T001", "T002", "T003"]


def _manifest(**kw):
    params = dict(
        execution_binding_sha256="e" * 64,
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


def test_execution_binding_names_randomization_and_blocks():
    """A config/stack/approval change cannot resume the old assignment namespace."""
    original = _manifest()
    changed = _manifest(execution_binding_sha256="f" * 64)
    assert changed.digest != original.digest
    assert {block.block_id for block in changed.blocks}.isdisjoint(
        block.block_id for block in original.blocks)


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


def test_partial_replication_has_an_exact_predeclared_denominator():
    tasks = [f"T{index:03d}" for index in range(32)]
    manifest = _manifest(task_ids=tasks, second_seed_fraction=0.25)
    replicated = {
        block.task_id for block in manifest.blocks if block.replicate_id == "1"
    }
    assert len(replicated) == 8
    assert manifest.notes["second_seed_task_count"] == 8
    assert len(manifest.blocks) == 40
    # A new protocol may select different tasks, but it may not change the denominator.
    other = _manifest(
        execution_binding_sha256="f" * 64,
        task_ids=tasks,
        second_seed_fraction=0.25,
    )
    assert len({b.task_id for b in other.blocks if b.replicate_id == "1"}) == 8


def _seventeen_arm_manifest(*, execution_binding_sha256: str = "e" * 64):
    arms = [
        ArmSpec(
            "P0" if index == 0 else f"A{index:02d}",
            "P0",
            "P0",
        )
        for index in range(17)
    ]
    tasks = [f"T{index:03d}" for index in range(32)]
    return _manifest(
        execution_binding_sha256=execution_binding_sha256,
        task_ids=tasks,
        arms=arms,
        second_seed_fraction=0.25,
    )


def _orders_by_coordinate(manifest):
    return {
        (block.task_id, block.replicate_id): tuple(
            cell.arm.arm_id
            for cell in sorted(block.cells, key=lambda cell: cell.order_index)
        )
        for block in manifest.blocks
    }


def test_seventeen_arm_screen_is_globally_position_balanced():
    """The 40 frozen blocks allocate a balanced row multiset, not 40 random row draws."""
    manifest = _seventeen_arm_manifest()
    orders = list(_orders_by_coordinate(manifest).values())
    arm_ids = [arm.arm_id for arm in manifest.arms]

    assert len(orders) == 40
    assert manifest.notes["randomization_design"] == \
        "binding_ranked_balanced_williams_v2"
    assert manifest.notes["williams_full_cycles"] == 1
    assert manifest.notes["williams_remainder_rows"] == 6
    for position in range(len(arm_ids)):
        counts = Counter(order[position] for order in orders)
        assert set(counts) == set(arm_ids)
        # 40 / 17 gives exactly six arms a third appearance and eleven arms two.
        assert set(counts.values()) == {2, 3}
        assert max(counts.values()) - min(counts.values()) <= 1


def test_seventeen_arm_screen_preserves_williams_adjacency_balance():
    # Include the binding that made the old arbitrary six-row subset repeat eight directed
    # adjacencies, plus enough independent bindings to prove balance is a design invariant
    # rather than a lucky property of the default fixture.
    bindings = [
        "5feceb66ffc86f38d952786c6d696c79c2dbc239dd4e91b46729d73a27fb57e9",
        *[
            hashlib.sha256(f"binding-{index}".encode()).hexdigest()
            for index in range(64)
        ],
    ]
    for binding in bindings:
        manifest = _seventeen_arm_manifest(
            execution_binding_sha256=binding
        )
        orders = list(_orders_by_coordinate(manifest).values())
        arm_ids = [arm.arm_id for arm in manifest.arms]
        adjacency = Counter(
            (left, right)
            for order in orders
            for left, right in zip(order, order[1:], strict=False)
        )

        expected = {
            (left, right)
            for left in arm_ids
            for right in arm_ids
            if left != right
        }
        assert set(adjacency) == expected
        # One complete 34-row odd-arm design contributes every directed adjacency twice;
        # the safe six-row cyclic window contributes each affected adjacency at most once.
        assert set(adjacency.values()) == {2, 3}
        assert max(adjacency.values()) - min(adjacency.values()) <= 1
        assert manifest.notes["williams_remainder_policy"] == (
            "odd_translated_cyclic_window_carryover_balanced"
        )


def test_second_seed_uses_the_exact_full_reverse_of_the_first_row():
    manifest = _seventeen_arm_manifest()
    orders = _orders_by_coordinate(manifest)
    replicated = {
        block.task_id for block in manifest.blocks if block.replicate_id == "1"
    }

    assert len(replicated) == 8
    assert manifest.notes["second_seed_order_policy"] == "exact_full_row_reversal"
    for task_id in replicated:
        assert orders[(task_id, "1")] == tuple(reversed(orders[(task_id, "0")]))


def test_binding_change_rerandomizes_coordinate_assignment_without_changing_balance():
    first = _seventeen_arm_manifest(execution_binding_sha256="e" * 64)
    repeated = _seventeen_arm_manifest(execution_binding_sha256="e" * 64)
    changed = _seventeen_arm_manifest(execution_binding_sha256="f" * 64)

    first_orders = _orders_by_coordinate(first)
    assert first_orders == _orders_by_coordinate(repeated)
    assert first.notes["randomization_assignment_sha256"] == \
        repeated.notes["randomization_assignment_sha256"]
    assert first_orders != _orders_by_coordinate(changed)
    assert first.notes["randomization_assignment_sha256"] != \
        changed.notes["randomization_assignment_sha256"]
    # Re-randomization changes the mapping, never the pre-registered 32+8 denominator.
    assert len(changed.blocks) == 40
    assert sum(block.replicate_id == "1" for block in changed.blocks) == 8


@pytest.mark.parametrize("fraction", [-0.01, 1.01])
def test_invalid_replication_fractions_are_refused(fraction):
    with pytest.raises(ValueError, match="between 0 and 1"):
        _manifest(second_seed_fraction=fraction)


def test_replication_cannot_be_requested_with_only_one_seed():
    with pytest.raises(ValueError, match="at least two seeds"):
        _manifest(second_seed_fraction=0.25, seeds=[1])


def test_a_block_is_complete_only_when_every_cell_committed():
    manifest = _manifest()
    block = manifest.blocks[0]
    states = {cell_key(c): "COMMITTED" for c in block.cells}
    assert block_is_complete(block, states)

    states[cell_key(block.cells[0])] = "FAILED_FINAL"
    assert not block_is_complete(block, states)
    assert block_is_terminal(block, states)
    # An unknown external outcome is not a completion either.
    states[cell_key(block.cells[0])] = "FAILED_UNKNOWN"
    assert not block_is_complete(block, states)
    assert block_is_terminal(block, states)


def test_the_freeze_record_carries_every_cell_state():
    manifest = _manifest()
    block = manifest.blocks[0]
    states = {cell_key(c): "COMMITTED" for c in block.cells}
    outputs = {cell_key(c): f"ref-{c.arm.arm_id}" for c in block.cells}
    record = freeze_record(block, states=states, outputs=outputs)
    assert record["complete"] is True
    assert record["complete_success"] is True
    assert record["terminal_frozen"] is True
    assert len(record["cells"]) == len(block.cells)
    assert all(c["output_ref"] for c in record["cells"])
    assert len(record["freeze_sha256"]) == 64


def test_run_id_cannot_be_rebound_to_another_execution(tmp_path):
    ledger = Ledger(str(tmp_path / "ledger.sqlite"))
    try:
        ledger.create_run("run", "e" * 64, '{"schedule":"s1","split":"screen"}')
        # Equivalent JSON remains an idempotent resume.
        ledger.create_run("run", "e" * 64, '{"split":"screen", "schedule":"s1"}')
        with pytest.raises(LedgerError, match="different execution"):
            ledger.create_run("run", "f" * 64, '{"schedule":"s1","split":"screen"}')
        with pytest.raises(LedgerError, match="different execution"):
            ledger.create_run("run", "e" * 64, '{"schedule":"s2","split":"screen"}')
    finally:
        ledger.close()


def test_failed_assignments_are_terminal_itt_outcomes():
    block = _manifest().blocks[0]
    states = {cell_key(c): "COMMITTED" for c in block.cells}
    states[cell_key(block.cells[0])] = "FAILED_FINAL"
    outputs = {cell_key(c): f"ref-{i}" for i, c in enumerate(block.cells)}
    record = freeze_record(block, states=states, outputs=outputs)
    assert record["complete_success"] is False
    assert record["terminal_frozen"] is True
    assert all(c["output_ref"] for c in record["cells"])


def test_resume_only_lists_the_cells_that_are_not_done():
    manifest = _manifest()
    states = {cell_key(c): "COMMITTED" for c in manifest.cells}
    assert cells_needing_work(manifest, states) == []
    first = manifest.cells[0]
    states[cell_key(first)] = "PENDING"
    assert [c.arm.arm_id for c in cells_needing_work(manifest, states)] == [first.arm.arm_id]
    states[cell_key(first)] = "FAILED_FINAL"
    assert cells_needing_work(manifest, states) == []


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


def test_new_execution_binding_does_not_inherit_old_completed_gates(tmp_path):
    ledger = Ledger(str(tmp_path / "l.sqlite"))
    old = PhaseStore(ledger, protocol_sha="a" * 64)
    old.begin(Phase.DOCTOR_PASSED)
    old.complete(Phase.DOCTOR_PASSED, {"binding": "old"})

    new = PhaseStore(ledger, protocol_sha="b" * 64)
    assert new.current() is Phase.NEW
    assert not new.is_complete(Phase.DOCTOR_PASSED)
    new.begin(Phase.DOCTOR_PASSED)
    new.complete(Phase.DOCTOR_PASSED, {"binding": "new"})

    assert old.record(Phase.DOCTOR_PASSED).detail == {"binding": "old"}
    assert new.record(Phase.DOCTOR_PASSED).detail == {"binding": "new"}
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


# --- the spine has to be walkable by the commands that walk it -------------------------------


def test_the_gpu_smoke_phase_is_unreachable_without_the_gates_before_it(tmp_path):
    """The exact shape of the bug: only three phases were ever recorded, and the first legal
    edge out of NEW is DOCTOR_PASSED. A *passing* canary crashed on success -- begin() raised
    -- while a failing one exited cleanly, and run-screen could not start at all."""
    from shapeflow.campaign.phases import IllegalPhaseTransition, PhaseStore
    from shapeflow.experiment.ledger import Ledger
    from shapeflow.experiment.state_machine import Phase

    ledger = Ledger(str(tmp_path / "l.sqlite"))
    phases = PhaseStore(ledger, protocol_sha="p")
    with pytest.raises(IllegalPhaseTransition):
        phases.begin(Phase.GPU_SMOKE_PASSED)
    ledger.close()


def test_the_recorded_gates_make_the_smoke_and_screen_phases_reachable(tmp_path):
    from shapeflow.campaign.phases import PhaseStore
    from shapeflow.experiment.ledger import Ledger
    from shapeflow.experiment.state_machine import Phase

    ledger = Ledger(str(tmp_path / "l.sqlite"))
    phases = PhaseStore(ledger, protocol_sha="p")
    for phase in (Phase.DOCTOR_PASSED, Phase.ACQUISITION_COMPLETE, Phase.SNAPSHOTS_FROZEN,
                  Phase.P0_PARITY_PASSED):
        phases.begin(phase)
        phases.complete(phase, {})
    phases.begin(Phase.GPU_SMOKE_PASSED)
    phases.complete(Phase.GPU_SMOKE_PASSED, {})
    phases.begin(Phase.SCREEN_RUNNING)
    assert phases.current() is Phase.GPU_SMOKE_PASSED
    ledger.close()
