"""Four GPUs, one schedule: task-atomic lanes frozen before treatment, merged strictly after.

Running four copies of the campaign would be four campaigns. What runs here is one
pre-registered schedule whose *tasks* are partitioned across four isolated lanes, each lane a
GPU, an engine, a provider and its own runner ledger. Cross-GPU parallelism is then independent
replication of whole tasks, which does not change the behaviour of the system under test: inside
a cell the graph's own concurrency is untouched.

One rule cannot be bent. **Every replicate and every arm of one task runs on one lane.** A block
is paired-valid only under a single engine epoch (``schedule.freeze_root_record``), and the
paired contrast is P1 against P0 on the same task -- so P0 on one card and P1 on another would
fold that card's clocks, thermals and scheduler into the treatment effect and there would be no
way afterwards to tell the two apart. The partition is therefore over tasks, never over arms.

Three properties make the assignment admissible as pre-registration:

- **It is frozen before treatment**, as a write-once content-addressed artifact, so a lane
  cannot be chosen after a result is seen.
- **It uses pre-treatment information only.** The balance proxy is the frozen corpus's own byte
  size for a task -- material that exists before any arm runs. No outcome, no trajectory, no
  observed cost.
- **It is deterministic.** Tie-breaking is seeded from the approved execution binding, so the
  same campaign resumed on a different day plans the same partition, and a lane that has to be
  re-run re-runs the same tasks.

There is no rebalancing and no mid-task migration. A lane that falls behind stays behind: moving
a half-finished task would either splice two engine epochs into one block or discard work that
is already on the ledger, and the tempting third option -- move the *remaining* arms -- is
exactly the cross-GPU pairing the rule above forbids.

**What Freeze-1 actually calls.** ``assign_tasks_to_lanes`` and ``Lane``, the latter through
``campaign.settings``. ``build_shard_manifest``, ``verify_shard_manifest`` and
``merge_shard_freeze_roots`` have no caller: they served ``freeze-shards`` and ``merge-shards``,
which are gone with the Week-1 screen, and ``campaign.bcplus`` partitions inline instead --
``sha256(binding:layer:task_id) % shards`` into ``RunnerConfig.owned_block_ids``, with
``grade-bcplus --lanes`` reading the lanes' ledgers and deduplicating rather than merging frozen
roots. They are kept, and named here as uncalled so a reader is not misled into thinking the
merge check runs, because the lane validations live inside ``build_shard_manifest``: exactly one
lane may reach a paid upstream, and no two lanes may share a GPU, a port or a runner root. Those
are properties this study still depends on, and deleting the function to tidy the tree would
delete them with it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..canonical import canonical_json
from ..experiment.randomization import derive_seed, seeded_permutation
from ..hashing import sha256_hex
from .schedule import ScheduleManifest, cell_key

__all__ = [
    "SHARD_MANIFEST_FILENAME",
    "SHARD_ASSIGNMENT_VERSION",
    "Lane",
    "ShardMergeError",
    "assign_tasks_to_lanes",
    "build_shard_manifest",
    "verify_shard_manifest",
    "merge_shard_freeze_roots",
]

SHARD_MANIFEST_FILENAME = "SHARD_MANIFEST.json"

#: Bumped when the partition would come out differently for the same inputs. It is part of the
#: manifest hash, so a changed algorithm cannot silently reuse a frozen assignment.
SHARD_ASSIGNMENT_VERSION = "task_atomic_lpt_by_frozen_corpus_bytes_v1"


class ShardMergeError(ValueError):
    """The lanes' terminal records do not reconstitute exactly the pre-registered schedule."""


@dataclass(frozen=True)
class Lane:
    """One isolated execution lane: a GPU, an engine, a provider, and its own runner tree."""

    shard_id: int
    gpu_uuid: str
    vllm_port: int
    provider_port: int
    runner_root: str
    #: Only one lane may reach a paid upstream. Four lanes each holding the full DeepSeek cap
    #: would be a 4x budget, and a cap that can be multiplied by starting another process is not
    #: admission control.
    serves_paid_upstreams: bool = False

    def content(self) -> dict:
        return {
            "shard_id": self.shard_id,
            "gpu_uuid": self.gpu_uuid,
            "vllm_port": self.vllm_port,
            "provider_port": self.provider_port,
            "runner_root": self.runner_root,
            "serves_paid_upstreams": self.serves_paid_upstreams,
        }


def assign_tasks_to_lanes(
    task_costs: Mapping[str, float],
    *,
    lane_count: int,
    execution_binding_sha256: str,
) -> dict[str, int]:
    """Partition tasks across lanes, balancing a pre-treatment cost proxy.

    Longest-processing-time-first: place the largest task on the currently lightest lane. It is
    the standard greedy for this and lands within 4/3 of optimal makespan, which matters here
    only because an idle lane is wasted GPU -- the *validity* of the partition does not depend on
    how even it is, only on its being task-atomic and pre-registered.

    Ties are broken by a permutation seeded from the approved execution binding rather than by
    task id, so the partition does not correlate with whatever ordering the corpus happens to
    have, and it is still reproducible.
    """
    if lane_count < 1:
        raise ValueError("a campaign needs at least one lane")
    task_ids = sorted(task_costs)
    if not task_ids:
        return {}
    order = seeded_permutation(
        task_ids, derive_seed(execution_binding_sha256, SHARD_ASSIGNMENT_VERSION, "tie_break")
    )
    tie_rank = {task_id: index for index, task_id in enumerate(order)}
    ordered = sorted(task_ids, key=lambda t: (-float(task_costs[t]), tie_rank[t]))

    loads = [0.0] * lane_count
    assignment: dict[str, int] = {}
    for task_id in ordered:
        lane = min(range(lane_count), key=lambda index: (loads[index], index))
        assignment[task_id] = lane
        loads[lane] += float(task_costs[task_id])
    return assignment


def build_shard_manifest(
    manifest: ScheduleManifest,
    *,
    lanes: Sequence[Lane],
    task_costs: Mapping[str, float],
    stack_manifest_sha256: str,
    protocol_sha256: str,
) -> dict:
    """The frozen partition: which lane owns which task, and every cell it therefore owns.

    Cells are enumerated, not left implied by the rule. The merge check afterwards has to be able
    to say "exactly these, exactly once" without re-deriving them from an algorithm that may by
    then have been edited.
    """
    if not lanes:
        raise ValueError("a shard manifest needs at least one lane")
    shard_ids = [lane.shard_id for lane in lanes]
    if sorted(shard_ids) != list(range(len(lanes))):
        raise ValueError("lanes must be numbered 0..n-1 exactly once")
    gpus = [lane.gpu_uuid for lane in lanes]
    if len(set(gpus)) != len(gpus) or any(not uuid.startswith("GPU-") for uuid in gpus):
        raise ValueError("each lane needs its own GPU UUID")
    ports = [lane.vllm_port for lane in lanes] + [lane.provider_port for lane in lanes]
    if len(set(ports)) != len(ports):
        raise ValueError("lanes must not share a port")
    roots = [lane.runner_root for lane in lanes]
    if len(set(roots)) != len(roots):
        raise ValueError("lanes must not share a runner root; the ledger is single-writer")
    paid = [lane for lane in lanes if lane.serves_paid_upstreams]
    if len(paid) != 1:
        raise ValueError(
            "exactly one lane may reach a paid upstream: a per-lane budget would multiply the "
            "cap by the number of processes started"
        )

    schedule_tasks = sorted({block.task_id for block in manifest.blocks})
    missing_costs = [task_id for task_id in schedule_tasks if task_id not in task_costs]
    if missing_costs:
        raise ValueError(f"no pre-treatment cost for scheduled tasks {missing_costs[:5]}")
    assignment = assign_tasks_to_lanes(
        {task_id: task_costs[task_id] for task_id in schedule_tasks},
        lane_count=len(lanes),
        execution_binding_sha256=manifest.execution_binding_sha256,
    )

    cells_by_shard: dict[int, list[str]] = {lane.shard_id: [] for lane in lanes}
    for cell in manifest.cells:
        cells_by_shard[assignment[cell.task_id]].append(cell_key(cell))
    blocks_by_shard: dict[int, list[str]] = {lane.shard_id: [] for lane in lanes}
    for block in manifest.blocks:
        blocks_by_shard[assignment[block.task_id]].append(block.block_id)

    body = {
        "schema_version": "shard_manifest_v1",
        "assignment_version": SHARD_ASSIGNMENT_VERSION,
        "execution_binding_sha256": manifest.execution_binding_sha256,
        "protocol_sha256": protocol_sha256,
        "schedule_sha256": manifest.digest,
        "stack_manifest_sha256": stack_manifest_sha256,
        "split": manifest.split,
        "layer": manifest.layer,
        "lane_count": len(lanes),
        "lanes": [lane.content() for lane in lanes],
        # The proxy is recorded, not just its result: a partition whose inputs are unknowable
        # afterwards cannot be checked for having used only pre-treatment information.
        "balance_proxy": "frozen_corpus_bytes",
        "task_costs": {task_id: float(task_costs[task_id]) for task_id in schedule_tasks},
        "task_to_shard": dict(sorted(assignment.items())),
        "cells_by_shard": {
            str(shard_id): sorted(keys) for shard_id, keys in sorted(cells_by_shard.items())
        },
        "blocks_by_shard": {
            str(shard_id): sorted(ids) for shard_id, ids in sorted(blocks_by_shard.items())
        },
        "total_cells": len(manifest.cells),
    }
    body["shard_manifest_sha256"] = sha256_hex(canonical_json(body))
    return body


def verify_shard_manifest(body: Mapping[str, object], manifest: ScheduleManifest) -> None:
    """Re-derive what the manifest claims, rather than trusting that it says so."""
    recorded = str(body.get("shard_manifest_sha256") or "")
    unsigned = {k: v for k, v in body.items() if k != "shard_manifest_sha256"}
    if not recorded or sha256_hex(canonical_json(unsigned)) != recorded:
        raise ShardMergeError("shard manifest hash does not bind its own content")
    if str(body.get("schedule_sha256") or "") != manifest.digest:
        raise ShardMergeError("shard manifest is bound to a different schedule")
    if str(body.get("execution_binding_sha256") or "") != manifest.execution_binding_sha256:
        raise ShardMergeError("shard manifest is bound to a different execution binding")

    assignment = dict(body.get("task_to_shard") or {})
    scheduled_tasks = {block.task_id for block in manifest.blocks}
    if set(assignment) != scheduled_tasks:
        raise ShardMergeError("shard manifest does not cover exactly the scheduled tasks")

    declared_cells: dict[str, int] = {}
    for shard_text, keys in dict(body.get("cells_by_shard") or {}).items():
        for key in keys:
            if key in declared_cells:
                raise ShardMergeError(f"cell {key} is assigned to more than one lane")
            declared_cells[key] = int(shard_text)
    expected = {cell_key(cell): assignment[cell.task_id] for cell in manifest.cells}
    if declared_cells != expected:
        raise ShardMergeError(
            "shard manifest cell assignment does not match the schedule's own task partition"
        )


def merge_shard_freeze_roots(
    body: Mapping[str, object],
    manifest: ScheduleManifest,
    freeze_roots: Mapping[int, Mapping[str, object]],
) -> dict:
    """Reconstitute one campaign from the lanes, or refuse.

    The merge is the only place the four lanes become one result, so it is the only place a
    partition error can still be caught. Four separately-valid lanes are not a valid campaign:
    a task silently run twice, a lane's blocks quietly missing, or one task's arms split across
    two GPUs each leave every individual lane internally consistent.
    """
    verify_shard_manifest(body, manifest)
    lane_ids = {int(lane["shard_id"]) for lane in body["lanes"]}  # type: ignore[index]
    if set(freeze_roots) != lane_ids:
        raise ShardMergeError(
            f"expected a freeze root from every lane {sorted(lane_ids)}, "
            f"got {sorted(freeze_roots)}"
        )

    assignment = {str(k): int(v) for k, v in dict(body.get("task_to_shard") or {}).items()}
    expected_blocks = {block.block_id: block for block in manifest.blocks}
    seen_blocks: dict[str, int] = {}
    seen_cells: dict[tuple[str, str, str, str], int] = {}
    epochs_by_lane: dict[int, set[str]] = {}
    merged_blocks: list[dict] = []

    for shard_id in sorted(freeze_roots):
        root = freeze_roots[shard_id]
        if str(root.get("schedule_sha256") or "") != manifest.digest:
            raise ShardMergeError(f"lane {shard_id} froze a different schedule")
        if str(root.get("split") or "") != manifest.split:
            raise ShardMergeError(f"lane {shard_id} froze a different split")
        recorded = str(root.get("freeze_root_sha256") or "")
        unsigned = {k: v for k, v in root.items() if k != "freeze_root_sha256"}
        if not recorded or sha256_hex(canonical_json(unsigned)) != recorded:
            raise ShardMergeError(f"lane {shard_id} freeze root hash does not verify")

        lane_epochs: set[str] = set()
        for record in list(root.get("blocks") or ()):
            block_id = str(record.get("block_id") or "")
            block = expected_blocks.get(block_id)
            if block is None:
                raise ShardMergeError(f"lane {shard_id} froze unscheduled block {block_id}")
            if block_id in seen_blocks:
                raise ShardMergeError(
                    f"block {block_id} was executed on lanes {seen_blocks[block_id]} and "
                    f"{shard_id}; a task must never cross a GPU"
                )
            owner = assignment.get(block.task_id)
            if owner != shard_id:
                raise ShardMergeError(
                    f"lane {shard_id} executed block {block_id}, which the frozen partition "
                    f"assigned to lane {owner}"
                )
            seen_blocks[block_id] = shard_id
            for raw in list(record.get("cells") or ()):
                arm = raw.get("arm") or {}
                key = (
                    str(raw.get("block_id") or ""),
                    str(raw.get("task_id") or ""),
                    str(arm.get("arm_id") or ""),
                    str(raw.get("replicate_id") or ""),
                )
                if key in seen_cells:
                    raise ShardMergeError(
                        f"cell {key} appears on lanes {seen_cells[key]} and {shard_id}"
                    )
                seen_cells[key] = shard_id
                lane_epochs.add(str(raw.get("engine_epoch") or ""))
            merged_blocks.append({**record, "shard_id": shard_id})
        epochs_by_lane[shard_id] = lane_epochs

    if set(seen_blocks) != set(expected_blocks):
        missing = sorted(set(expected_blocks) - set(seen_blocks))
        raise ShardMergeError(
            f"the lanes together do not cover the schedule; {len(missing)} block(s) missing, "
            f"first: {missing[:5]}"
        )
    expected_cells = {
        (cell.block_id, cell.task_id, cell.arm.arm_id, cell.replicate_id)
        for cell in manifest.cells
    }
    if set(seen_cells) != expected_cells:
        missing = sorted(expected_cells - set(seen_cells))
        extra = sorted(set(seen_cells) - expected_cells)
        raise ShardMergeError(
            f"merged cells are not the pre-registered set: missing={missing[:3]}, "
            f"extra={extra[:3]}"
        )

    # An engine epoch is one vLLM boot. Two lanes cannot share one, so a shared epoch means the
    # lanes were not the separate engines they are recorded as -- and every per-lane isolation
    # claim built on top of that would be false.
    for left in sorted(epochs_by_lane):
        for right in sorted(epochs_by_lane):
            if left >= right:
                continue
            shared = epochs_by_lane[left] & epochs_by_lane[right]
            if shared:
                raise ShardMergeError(
                    f"lanes {left} and {right} share engine epoch(s) {sorted(shared)[:3]}; "
                    "they were not separate engines"
                )

    merged = {
        "schema_version": "merged_campaign_root_v1",
        "shard_manifest_sha256": str(body.get("shard_manifest_sha256") or ""),
        "execution_binding_sha256": manifest.execution_binding_sha256,
        "protocol_sha256": manifest.protocol_sha,
        "schedule_sha256": manifest.digest,
        "split": manifest.split,
        "lane_count": len(lane_ids),
        "lane_freeze_roots": {
            str(shard_id): str(freeze_roots[shard_id].get("freeze_root_sha256") or "")
            for shard_id in sorted(freeze_roots)
        },
        "engine_epochs_by_lane": {
            str(shard_id): sorted(epochs_by_lane[shard_id])
            for shard_id in sorted(epochs_by_lane)
        },
        "blocks": sorted(merged_blocks, key=lambda item: str(item["block_id"])),
        "total_cells": len(seen_cells),
    }
    merged["merged_root_sha256"] = sha256_hex(canonical_json(merged))
    return merged
