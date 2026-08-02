"""The paired-block schedule, frozen before any outcome exists.

A *block* is the recovery and inference unit: one boundary state, and every arm assigned to it,
always including P0. Two properties make it worth the machinery.

The coordinates alone do not claim checkpoint pairing. ``notes.execution_semantics`` says
whether they assign anchor trajectories for a boundary component screen or independent
task-level E2E runs. In the former case a block is *not* itself the evidence boundary; the
content-addressed fork records are.

**Complete-block freezing is an operational record, not the ITT denominator.** A block with a
missing cell is not spliced across engine epochs. It is nevertheless retained as an offered
assignment with its failure state; the analysis must not delete it merely because a P1 cell
failed.

**Order is balanced, not convenient.** Arms within a block are ordered by a Williams square, and
consecutive replicates alternate AB/BA, so a drift over the block's execution -- a warming GPU, a
slowly filling cache -- does not land preferentially on one arm. Running one arm to completion
and then the next would confound the arm with the time it ran at.

The schedule is derived from the approved execution binding, the task ids and the seeds -- never
from a clock or a counter -- so the same campaign, resumed on a different day, plans exactly the
same work.  The protocol-document SHA is retained separately for human provenance; it is not a
complete execution identity.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..canonical import canonical_json
from ..experiment.ledger import TERMINAL_STATES
from ..experiment.randomization import derive_seed, seeded_permutation, williams_square
from ..hashing import sha256_hex

__all__ = [
    "ArmSpec",
    "Block",
    "Cell",
    "FROZEN_ROOT_FILENAME",
    "ScheduleManifest",
    "block_is_complete",
    "block_is_terminal",
    "build_blocks",
    "freeze_root_record",
]

FROZEN_ROOT_FILENAME = "FREEZE_ROOT.json"


@dataclass(frozen=True)
class ArmSpec:
    arm_id: str
    page_variant: str
    close_variant: str

    def content(self) -> dict:
        return {"arm_id": self.arm_id, "page_variant": self.page_variant,
                "close_variant": self.close_variant}


@dataclass(frozen=True)
class Cell:
    block_id: str
    task_id: str
    arm: ArmSpec
    seed: int
    replicate_id: str
    order_index: int

    def content(self) -> dict:
        return {
            "block_id": self.block_id, "task_id": self.task_id, "arm": self.arm.content(),
            "seed": self.seed, "replicate_id": self.replicate_id,
            "order_index": self.order_index,
        }


@dataclass(frozen=True)
class Block:
    block_id: str
    task_id: str
    replicate_id: str
    cells: tuple[Cell, ...]

    def content(self) -> dict:
        return {
            "block_id": self.block_id, "task_id": self.task_id,
            "replicate_id": self.replicate_id,
            "cells": [c.content() for c in self.cells],
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


@dataclass
class ScheduleManifest:
    execution_binding_sha256: str
    protocol_sha: str
    split: str
    blocks: tuple[Block, ...]
    arms: tuple[ArmSpec, ...]
    seeds: tuple[int, ...]
    layer: str
    claim_scope: str
    notes: dict = field(default_factory=dict)

    def content(self) -> dict:
        return {
            "execution_binding_sha256": self.execution_binding_sha256,
            "protocol_sha": self.protocol_sha,
            "split": self.split,
            "layer": self.layer,
            "claim_scope": self.claim_scope,
            "arms": [a.content() for a in self.arms],
            "seeds": list(self.seeds),
            "blocks": [b.content() for b in self.blocks],
            "notes": dict(sorted(self.notes.items())),
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def to_json(self) -> dict:
        body = self.content()
        body["schedule_sha256"] = self.digest
        return body

    @property
    def cells(self) -> list[Cell]:
        return [cell for block in self.blocks for cell in block.cells]


def build_blocks(
    *,
    execution_binding_sha256: str,
    protocol_sha: str,
    split: str,
    task_ids: Sequence[str],
    arms: Sequence[ArmSpec],
    seeds: Sequence[int],
    layer: str,
    claim_scope: str,
    second_seed_fraction: float = 0.0,
) -> ScheduleManifest:
    """One block per (task, replicate), with a balanced arm order inside each.

    ``second_seed_fraction`` decides the *exact* number of tasks that get a second replicate.
    Tasks are ranked deterministically from the complete approved execution binding, split, and
    task id before anything runs.  An independent Bernoulli decision per task would only hit the
    requested fraction in expectation, making both power and the GPU budget unknowable until
    after the corpus was frozen.
    """
    if (
        len(execution_binding_sha256) != 64
        or any(ch not in "0123456789abcdef" for ch in execution_binding_sha256)
    ):
        raise ValueError("execution_binding_sha256 must be a lowercase SHA-256 digest")
    if len(protocol_sha) != 64 or any(ch not in "0123456789abcdef" for ch in protocol_sha):
        raise ValueError("protocol_sha must be a lowercase SHA-256 digest")
    if not arms:
        raise ValueError("a block with no arms measures nothing")
    if not 0.0 <= second_seed_fraction <= 1.0:
        raise ValueError("second_seed_fraction must be between 0 and 1 inclusive")
    if second_seed_fraction > 0.0 and len(seeds) < 2:
        raise ValueError("a positive second_seed_fraction requires at least two seeds")
    if not any(a.arm_id == "P0" or (a.page_variant == "P0" and a.close_variant == "P0")
               for a in arms):
        raise ValueError(
            "every block must contain P0: without the comparator in the same block, a P1 result "
            "is compared against an arm that ran somewhere else"
        )

    ordered_tasks = sorted(task_ids)
    if len(ordered_tasks) != len(set(ordered_tasks)):
        raise ValueError("task_ids must be unique")
    arm_ids = [a.arm_id for a in arms]
    if len(arm_ids) != len(set(arm_ids)):
        raise ValueError("arm IDs must be unique")
    by_id = {a.arm_id: a for a in arms}
    blocks: list[Block] = []
    second_seed_count = int(len(ordered_tasks) * second_seed_fraction + 0.5)
    second_seed_tasks = set(sorted(
        ordered_tasks,
        key=lambda task_id: (
            derive_seed(
                execution_binding_sha256, split, "second_seed_rank", task_id),
            task_id,
        ),
    )[:second_seed_count])
    assigned_rows, randomization_notes = _balanced_williams_assignments(
        execution_binding_sha256=execution_binding_sha256,
        split=split,
        task_ids=ordered_tasks,
        second_seed_tasks=second_seed_tasks,
        arm_count=len(arm_ids),
    )

    for task_id in ordered_tasks:
        replicates = [0]
        if task_id in second_seed_tasks:
            replicates.append(1)

        for replicate in replicates:
            seed = seeds[replicate % len(seeds)] if seeds else 0
            row = assigned_rows[(task_id, replicate)]
            order = [arm_ids[index] for index in row]
            block_id = "B" + sha256_hex(canonical_json({
                "execution_binding_sha256": execution_binding_sha256,
                "split": split, "task_id": task_id,
                "replicate": replicate,
            }))[:14]
            cells = tuple(
                Cell(block_id=block_id, task_id=task_id, arm=by_id[arm_id], seed=seed,
                     replicate_id=str(replicate), order_index=position)
                for position, arm_id in enumerate(order)
            )
            blocks.append(Block(block_id=block_id, task_id=task_id,
                                replicate_id=str(replicate), cells=cells))

    return ScheduleManifest(
        execution_binding_sha256=execution_binding_sha256,
        protocol_sha=protocol_sha, split=split, blocks=tuple(blocks), arms=tuple(arms),
        seeds=tuple(seeds), layer=layer, claim_scope=claim_scope,
        notes={"tasks": len(ordered_tasks), "blocks": len(blocks),
               "cells": sum(len(b.cells) for b in blocks),
               "second_seed_fraction": second_seed_fraction,
               "second_seed_task_count": len(second_seed_tasks),
               **randomization_notes},
    )


def _balanced_williams_assignments(
    *,
    execution_binding_sha256: str,
    split: str,
    task_ids: Sequence[str],
    second_seed_tasks: set[str],
    arm_count: int,
) -> tuple[dict[tuple[str, int], tuple[int, ...]], dict]:
    """Assign a deterministic Williams-row multiset to frozen block coordinates.

    A Williams square is balanced only when its rows are allocated as a design.  Selecting an
    independently hash-shifted row inside every task turns it back into ordinary random order:
    some arms can then occupy a warm/cold position many more often than others.  Here the
    execution binding first hash-ranks the *coordinates*, while the allocated row multiset is
    one or more complete Williams designs plus a deterministic remainder. Complete cycles,
    even-arm subsets, and the proven odd cyclic safe window are carryover-balanced; the generic
    forced-pair fallback makes no exact balance claim and records that fact in the manifest.

    A task with a second seed consumes a row and its exact full reversal.  This is the actual
    AB/BA correspondence promised by the protocol; drawing an unrelated second row and swapping
    only its first two arms is not a paired order reversal.
    """
    if arm_count == 1:
        # A single-arm campaign -- the P0 competence pilot -- has no order to counterbalance.
        # There is exactly one arm sequence, every block gets it, and carryover balance is
        # vacuously satisfied rather than unachieved. Williams squares are undefined below two
        # treatments, so this is stated here instead of arriving as "need at least 2 treatments"
        # from three frames down, which is the shape of an unimplemented case rather than a
        # designed one.
        assignments = {
            (task_id, replicate): (0,)
            for task_id in task_ids
            for replicate in ((0, 1) if task_id in second_seed_tasks else (0,))
        }
        return assignments, {
            "randomization": "single_arm_no_ordering",
            "carryover_balance": "VACUOUS_SINGLE_ARM",
            "williams_rows": 1,
        }

    rows = [tuple(row) for row in williams_square(arm_count)]
    row_index = {row: index for index, row in enumerate(rows)}
    if len(row_index) != len(rows):
        raise ValueError("Williams design contains duplicate rows")
    reversal = {
        index: row_index.get(tuple(reversed(row)), -1)
        for index, row in enumerate(rows)
    }
    if any(index < 0 for index in reversal.values()):
        raise ValueError("Williams design is not closed under full-row reversal")
    pair_types = sorted({
        (min(index, reverse_index), max(index, reverse_index))
        for index, reverse_index in reversal.items()
    })

    total_blocks = len(task_ids) + len(second_seed_tasks)
    full_cycles, remainder = divmod(total_blocks, len(rows))
    row_pool = [
        index
        for _cycle in range(full_cycles)
        for index in range(len(rows))
    ]

    # Complete cycles provide one copy of every reversal pair.  If the campaign is smaller
    # than a complete cycle (or requests more paired replicates than those cycles provide), the
    # remainder must explicitly reserve enough complete pairs before its unpaired rows are
    # chosen.  Since second_seed_tasks <= task_ids, 2 * needed_pairs always fits the remainder.
    complete_pair_capacity = full_cycles * len(pair_types)
    needed_remainder_pairs = max(
        0, len(second_seed_tasks) - complete_pair_capacity
    )
    if 2 * needed_remainder_pairs > remainder:
        raise ValueError("second-seed reversal pairs do not fit the scheduled block count")

    pair_seed = derive_seed(
        execution_binding_sha256, split, "williams_pair_type_order"
    )
    ordered_pair_types = seeded_permutation(pair_types, pair_seed)
    extra_rows: list[int] = []
    for left, right in ordered_pair_types[:needed_remainder_pairs]:
        extra_rows.extend((left, right))

    remaining_extra_count = remainder - len(extra_rows)
    # For an odd Williams design, an arbitrary subset of translated rows is position-balanced
    # but is *not* necessarily first-order carryover balanced.  A cyclic consecutive window of
    # at most floor(n/2) translated rows has both properties: each position sees distinct
    # treatments and none of its directed adjacent pairs repeat.  The binding may choose the
    # window, direction and base/reverse family, but never a statistically weaker row multiset.
    # The exact Week-1 arm count is intentionally not hard-coded into this generic constructor.
    base_seed = derive_seed(
        execution_binding_sha256, split, "williams_remainder_order"
    )
    safe_cyclic_remainder = (
        arm_count % 2 == 1
        and not extra_rows
        and remaining_extra_count <= arm_count // 2
    )
    if safe_cyclic_remainder:
        start = base_seed % arm_count
        direction = (
            1
            if derive_seed(
                execution_binding_sha256, split, "williams_remainder_direction"
            )
            % 2
            == 0
            else -1
        )
        use_reverse_family = (
            derive_seed(
                execution_binding_sha256, split, "williams_remainder_family"
            )
            % 2
            == 1
        )
        translated_window = [
            (start + direction * offset) % arm_count
            for offset in range(remaining_extra_count)
        ]
        preferred = [
            reversal[index] if use_reverse_family else index
            for index in translated_window
        ]
    else:
        # Forced second-seed pairs can make exact position/carryover balance mathematically
        # impossible for a small odd-arm remainder.  Preserve the frozen denominator and choose
        # deterministically; the manifest records that the safe-window proof did not apply.
        base_order = seeded_permutation(list(range(arm_count)), base_seed)
        preferred = [
            *base_order,
            *(reversal[index] for index in base_order),
        ]
    for index in preferred:
        if remaining_extra_count == 0:
            break
        if index in extra_rows:
            continue
        extra_rows.append(index)
        remaining_extra_count -= 1
    if remaining_extra_count:
        # This can only arise for an unusual even-arm design whose reversal rows are already in
        # the base family.  Fill deterministically without changing the frozen denominator.
        fallback = seeded_permutation(
            list(range(len(rows))),
            derive_seed(execution_binding_sha256, split, "williams_remainder_fallback"),
        )
        for index in fallback:
            if remaining_extra_count == 0:
                break
            extra_rows.append(index)
            remaining_extra_count -= 1
    row_pool.extend(extra_rows)
    if len(row_pool) != total_blocks:
        raise AssertionError("Williams row pool does not match the frozen block denominator")

    # Select actual pair copies from the frozen pool, then leave every unused row in place for
    # primary-only tasks. Removing the two rows together preserves the exact target multiset, so
    # task assignment cannot change whatever position/adjacency counts that multiset provides.
    from collections import Counter

    available = Counter(row_pool)
    pair_slots: list[tuple[int, int]] = []
    for pair in pair_types:
        pair_slots.extend([pair] * min(available[pair[0]], available[pair[1]]))
    pair_slots = seeded_permutation(
        pair_slots,
        derive_seed(execution_binding_sha256, split, "williams_pair_slot_order"),
    )
    if len(pair_slots) < len(second_seed_tasks):
        raise ValueError("balanced Williams pool lacks required second-seed reversal pairs")

    assigned: dict[tuple[str, int], tuple[int, ...]] = {}
    paired_tasks = sorted(
        second_seed_tasks,
        key=lambda task_id: (
            derive_seed(
                execution_binding_sha256, split, "williams_paired_task_rank", task_id
            ),
            task_id,
        ),
    )
    for task_id, (left, right) in zip(
        paired_tasks, pair_slots[: len(paired_tasks)], strict=True
    ):
        orientation = derive_seed(
            execution_binding_sha256, split, "williams_pair_orientation", task_id
        ) % 2
        primary_index, second_index = (
            (left, right) if orientation == 0 else (right, left)
        )
        available[primary_index] -= 1
        available[second_index] -= 1
        assigned[(task_id, 0)] = rows[primary_index]
        assigned[(task_id, 1)] = rows[second_index]
        if assigned[(task_id, 1)] != tuple(reversed(assigned[(task_id, 0)])):
            raise AssertionError("second-seed Williams row is not the full reversal")

    remaining_rows = [
        index
        for index in range(len(rows))
        for _copy in range(available[index])
    ]
    remaining_rows = seeded_permutation(
        remaining_rows,
        derive_seed(execution_binding_sha256, split, "williams_single_row_order"),
    )
    single_tasks = sorted(
        (task_id for task_id in task_ids if task_id not in second_seed_tasks),
        key=lambda task_id: (
            derive_seed(
                execution_binding_sha256, split, "williams_single_task_rank", task_id
            ),
            task_id,
        ),
    )
    if len(remaining_rows) != len(single_tasks):
        raise AssertionError("Williams single-row pool does not match primary-only tasks")
    for task_id, index in zip(single_tasks, remaining_rows, strict=True):
        assigned[(task_id, 0)] = rows[index]

    assignment_sha = sha256_hex(canonical_json({
        f"{task_id}:{replicate}": list(row)
        for (task_id, replicate), row in sorted(assigned.items())
    }))
    return assigned, {
        "randomization_design": "binding_ranked_balanced_williams_v2",
        "williams_design_rows": len(rows),
        "williams_full_cycles": full_cycles,
        "williams_remainder_rows": remainder,
        "williams_remainder_policy": (
            "odd_translated_cyclic_window_carryover_balanced"
            if safe_cyclic_remainder
            else "deterministic_general_remainder_no_exact_balance_claim"
        ),
        "second_seed_order_policy": "exact_full_row_reversal",
        "randomization_assignment_sha256": assignment_sha,
    }


def block_is_complete(block: Block, terminal_states: dict) -> bool:
    """True only when every cell in the block reached COMMITTED.

    Any other terminal state -- a final failure, an unknown external outcome, a budget block --
    leaves the block incomplete. It is kept and reported, but it is not a paired observation and
    it never enters the primary comparison.
    """
    return all(terminal_states.get(_cell_key(cell)) == "COMMITTED" for cell in block.cells)


def block_is_terminal(block: Block, states: dict) -> bool:
    """Every offered assignment has an immutable outcome, success or failure."""
    return all(states.get(_cell_key(cell)) in TERMINAL_STATES for cell in block.cells)


def _cell_key(cell: Cell) -> str:
    return f"{cell.block_id}:{cell.arm.arm_id}:{cell.replicate_id}"


def cell_key(cell: Cell) -> str:
    return _cell_key(cell)


def freeze_record(block: Block, *, states: dict, outputs: dict) -> dict:
    """The immutable record of one completed paired block.

    Carries each cell's terminal state and output reference, so "this block is frozen" can be
    re-verified later from the artifacts rather than believed from a flag.
    """
    body = {
        "block_id": block.block_id,
        "task_id": block.task_id,
        "replicate_id": block.replicate_id,
        "cells": [
            {
                **cell.content(),
                "state": states.get(_cell_key(cell), "MISSING"),
                "output_ref": outputs.get(_cell_key(cell), ""),
            }
            for cell in block.cells
        ],
        # Keep the old field for readers that mean "all successful"; it must never be used as
        # the ITT inclusion gate. terminal_frozen is the all-offered analysis unit.
        "complete": block_is_complete(block, states),
        "complete_success": block_is_complete(block, states),
        "terminal_frozen": block_is_terminal(block, states),
        "block_digest": block.digest,
    }
    body["freeze_sha256"] = sha256_hex(canonical_json(body))
    return body


def freeze_root_record(
    manifest: ScheduleManifest,
    *,
    run_id: str,
    phase_id: str,
    split: str,
    block_records: Sequence[dict],
    shard_id: int | None = None,
    owned_block_ids: Sequence[str] | None = None,
) -> dict:
    """Bind the offered schedule to every immutable terminal block.

    Individual block files are necessary but insufficient evidence for an ITT denominator:
    deleting one successful/failed block still leaves a perfectly valid directory of hashes.
    This root names *all* assignments before analysis, including arm variants, seeds and order,
    and then binds each assignment to its terminal state/output through the block freeze hash.

    ``owned_block_ids`` narrows "all" to one execution lane's frozen share when the campaign is
    task-sharded across GPUs. The completeness requirement does not weaken -- it moves: a lane
    must still freeze every block it was given, exactly, and the union of the lanes must
    reconstitute the schedule, which ``sharding.merge_shard_freeze_roots`` requires. The whole
    schedule is still carried here, so a lane's root commits to the assignments it did *not*
    execute as well, and a lane cannot quietly redefine the denominator to be its own share.
    """
    if not run_id or not phase_id:
        raise ValueError("a frozen campaign root requires non-empty run_id and phase_id")
    if split != manifest.split:
        raise ValueError(
            f"freeze split {split!r} does not match schedule split {manifest.split!r}")

    scheduled = {block.block_id for block in manifest.blocks}
    if owned_block_ids is None:
        expected_ids = [block.block_id for block in manifest.blocks]
    else:
        owned = [str(block_id) for block_id in owned_block_ids]
        unscheduled = sorted(set(owned) - scheduled)
        if unscheduled or len(set(owned)) != len(owned):
            raise ValueError(
                f"lane share is not a subset of the schedule: unscheduled={unscheduled[:5]}")
        expected_ids = [
            block.block_id for block in manifest.blocks if block.block_id in set(owned)
        ]

    by_id: dict[str, dict] = {}
    for record in block_records:
        block_id = str(record.get("block_id") or "")
        if not block_id or block_id in by_id:
            raise ValueError(f"duplicate or missing frozen block id {block_id!r}")
        by_id[block_id] = record
    if set(by_id) != set(expected_ids) or len(by_id) != len(expected_ids):
        missing = sorted(set(expected_ids) - set(by_id))
        extra = sorted(set(by_id) - set(expected_ids))
        raise ValueError(
            f"cannot freeze partial/different campaign root: missing={missing}, extra={extra}")

    frozen_blocks: list[dict] = []
    for block in manifest.blocks:
        if block.block_id not in by_id:
            continue
        record = by_id[block.block_id]
        if record.get("terminal_frozen") is not True:
            raise ValueError(f"block {block.block_id} is not terminal_frozen")
        if str(record.get("block_digest") or "") != block.digest:
            raise ValueError(f"block {block.block_id} digest differs from frozen schedule")
        freeze_sha = str(record.get("freeze_sha256") or "")
        if len(freeze_sha) != 64:
            raise ValueError(f"block {block.block_id} has no freeze_sha256")
        unsigned = {key: value for key, value in record.items() if key != "freeze_sha256"}
        if sha256_hex(canonical_json(unsigned)) != freeze_sha:
            raise ValueError(f"block {block.block_id} freeze_sha256 does not verify")
        expected_cells = {
            (
                cell.block_id, cell.task_id, cell.arm.arm_id, cell.replicate_id
            ): cell.content()
            for cell in block.cells
        }
        record_cells = list(record.get("cells") or ())
        if len(record_cells) != len(expected_cells):
            raise ValueError(f"block {block.block_id} frozen cell count differs from schedule")
        observed_epochs: set[str] = set()
        observed_keys: set[tuple[str, str, str, str]] = set()
        for raw in record_cells:
            arm = raw.get("arm") or {}
            key = (
                str(raw.get("block_id") or ""),
                str(raw.get("task_id") or ""),
                str(arm.get("arm_id") or ""),
                str(raw.get("replicate_id") or ""),
            )
            expected = expected_cells.get(key)
            scheduled_projection = {
                name: raw.get(name)
                for name in (
                    "block_id", "task_id", "arm", "seed", "replicate_id", "order_index")
            }
            if expected is None or scheduled_projection != expected or key in observed_keys:
                raise ValueError(
                    f"block {block.block_id} contains an unscheduled/duplicate cell {key}")
            observed_keys.add(key)
            epoch = str(raw.get("engine_epoch") or "")
            if not epoch:
                raise ValueError(f"block {block.block_id} cell {key} lacks engine_epoch")
            observed_epochs.add(epoch)
            if (
                str(raw.get("state") or "") not in TERMINAL_STATES
                or not str(raw.get("output_ref") or "")
            ):
                raise ValueError(f"block {block.block_id} cell {key} lacks terminal outcome")
        declared_epochs = sorted(map(str, record.get("engine_epochs") or ()))
        if declared_epochs != sorted(observed_epochs):
            raise ValueError(f"block {block.block_id} engine epoch index does not match cells")
        paired_valid = record.get("valid_for_paired_estimate")
        invalid_reason = str(record.get("invalid_reason") or "")
        if not isinstance(paired_valid, bool):
            raise ValueError(f"block {block.block_id} lacks paired-validity status")
        if paired_valid and (len(observed_epochs) != 1 or invalid_reason):
            raise ValueError(f"block {block.block_id} has inconsistent paired-validity status")
        if not paired_valid and not invalid_reason:
            raise ValueError(f"block {block.block_id} invalidity has no reason")
        frozen_blocks.append({
            "block_id": block.block_id,
            "block_digest": block.digest,
            "freeze_sha256": freeze_sha,
            "terminal_frozen": True,
            "valid_for_paired_estimate": paired_valid,
            "invalid_reason": invalid_reason,
            "engine_epochs": declared_epochs,
            # Outcomes are duplicated under the root deliberately: the root hash then commits
            # to every offered cell even if a block file is later removed or swapped.
            "cells": list(record.get("cells") or ()),
        })

    body = {
        "schema_version": "frozen_campaign_root_v1",
        "run_id": run_id,
        "phase_id": phase_id,
        "split": split,
        "execution_binding_sha256": manifest.execution_binding_sha256,
        "protocol_sha256": manifest.protocol_sha,
        "schedule_sha256": manifest.digest,
        "schedule": manifest.to_json(),
        "terminal_frozen": True,
        "blocks": frozen_blocks,
    }
    if shard_id is not None:
        body["shard_id"] = int(shard_id)
        body["owned_block_ids"] = sorted(expected_ids)
    body["freeze_root_sha256"] = sha256_hex(canonical_json(body))
    return body


def cells_needing_work(manifest: ScheduleManifest, states: dict) -> list[Cell]:
    """Non-terminal cells only; terminal failures stay in ITT and are never retried away."""
    return [
        c for c in manifest.cells
        if states.get(_cell_key(c)) not in TERMINAL_STATES
    ]
