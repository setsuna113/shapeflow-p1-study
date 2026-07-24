"""The paired-block schedule, frozen before any outcome exists.

A *block* is the recovery and inference unit: one boundary state, and every arm assigned to it,
always including P0. Two properties make it worth the machinery.

**Only complete blocks count.** A block with a missing cell is not a paired observation. Splicing
a pre-crash P0 with a post-crash P1 would compare two arms across an engine restart, and the
difference between them would include whatever the restart changed. So a block is frozen only
when every one of its cells is terminal, and a restart fills the gaps rather than re-running what
already finished.

**Order is balanced, not convenient.** Arms within a block are ordered by a Williams square, and
consecutive replicates alternate AB/BA, so a drift over the block's execution -- a warming GPU, a
slowly filling cache -- does not land preferentially on one arm. Running one arm to completion
and then the next would confound the arm with the time it ran at.

The schedule is derived from the protocol SHA, the task ids and the seeds -- never from a clock
or a counter -- so the same campaign, resumed on a different day, plans exactly the same work.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from ..canonical import canonical_json
from ..experiment.randomization import ab_ba_order, arm_order_for_block, derive_seed
from ..hashing import sha256_hex

__all__ = ["ArmSpec", "Cell", "Block", "ScheduleManifest", "build_blocks", "block_is_complete"]


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

    ``second_seed_fraction`` decides which tasks get a second replicate, deterministically from
    the protocol SHA and the task id -- never by picking after seeing which tasks looked
    interesting, which would be outcome-dependent design.
    """
    if not arms:
        raise ValueError("a block with no arms measures nothing")
    if not any(a.arm_id == "P0" or (a.page_variant == "P0" and a.close_variant == "P0")
               for a in arms):
        raise ValueError(
            "every block must contain P0: without the comparator in the same block, a P1 result "
            "is compared against an arm that ran somewhere else"
        )

    ordered_tasks = sorted(task_ids)
    arm_ids = [a.arm_id for a in arms]
    by_id = {a.arm_id: a for a in arms}
    blocks: list[Block] = []

    for index, task_id in enumerate(ordered_tasks):
        replicates = [0]
        if second_seed_fraction > 0 and seeds and len(seeds) > 1:
            # Deterministic selection: the task's own derived seed decides, so the set of tasks
            # carrying a second replicate is fixed before anything runs.
            picker = derive_seed(protocol_sha, "second_seed", task_id) % 1000
            if picker < int(second_seed_fraction * 1000):
                replicates.append(1)

        for replicate in replicates:
            seed = seeds[replicate % len(seeds)] if seeds else 0
            block_seed = derive_seed(protocol_sha, split, task_id, str(replicate))
            order = arm_order_for_block(arm_ids, block_index=index + replicate, seed=block_seed)
            # AB/BA across replicates: replicate 1 sees the reverse of replicate 0's order for
            # the leading pair, so a within-block drift does not favour one arm twice.
            if replicate % 2 == 1 and len(order) >= 2:
                first, second = ab_ba_order((order[0], order[1]), replicate=replicate,
                                            seed=block_seed)
                order = [first, second] + order[2:]
            block_id = "B" + sha256_hex(canonical_json({
                "protocol_sha": protocol_sha, "split": split, "task_id": task_id,
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
        protocol_sha=protocol_sha, split=split, blocks=tuple(blocks), arms=tuple(arms),
        seeds=tuple(seeds), layer=layer, claim_scope=claim_scope,
        notes={"tasks": len(ordered_tasks), "blocks": len(blocks),
               "cells": sum(len(b.cells) for b in blocks)},
    )


def block_is_complete(block: Block, terminal_states: dict) -> bool:
    """True only when every cell in the block reached COMMITTED.

    Any other terminal state -- a final failure, an unknown external outcome, a budget block --
    leaves the block incomplete. It is kept and reported, but it is not a paired observation and
    it never enters the primary comparison.
    """
    return all(terminal_states.get(_cell_key(cell)) == "COMMITTED" for cell in block.cells)


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
        "complete": block_is_complete(block, states),
        "block_digest": block.digest,
    }
    body["freeze_sha256"] = sha256_hex(canonical_json(body))
    return body


def cells_needing_work(manifest: ScheduleManifest, states: dict) -> list[Cell]:
    """Cells that are not yet COMMITTED. Resume fills gaps; it never repeats finished work."""
    return [c for c in manifest.cells if states.get(_cell_key(c)) != "COMMITTED"]


def optional_second_seed_tasks(manifest: ScheduleManifest) -> list[str]:
    return sorted({b.task_id for b in manifest.blocks if b.replicate_id != "0"})


def find_block(manifest: ScheduleManifest, block_id: str) -> Optional[Block]:
    for block in manifest.blocks:
        if block.block_id == block_id:
            return block
    return None
