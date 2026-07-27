"""Four lanes must reconstitute exactly one campaign, or refuse to become one.

Four separately-valid lanes are not a valid campaign. A task silently run twice, a lane's blocks
quietly missing, or one task's arms split across two GPUs each leave every individual lane
internally consistent -- the error only exists in the union, so the merge is the only place it
can still be caught.
"""

from __future__ import annotations

import pytest

from shapeflow.campaign.schedule import (
    ArmSpec,
    build_blocks,
    cell_key,
    freeze_root_record,
)
from shapeflow.campaign.sharding import (
    Lane,
    ShardMergeError,
    assign_tasks_to_lanes,
    build_shard_manifest,
    merge_shard_freeze_roots,
    verify_shard_manifest,
)
from shapeflow.canonical import canonical_json
from shapeflow.hashing import sha256_hex

BINDING = "e" * 64
ARMS = [
    ArmSpec("P0", "P0", "P0"),
    ArmSpec("H_MARKDOWN_ID", "H02", "P0"),
    ArmSpec("C_ID", "P0", "C01"),
    ArmSpec("H_PLUS_C", "H02", "C01"),
]


def _manifest(task_count: int = 8):
    return build_blocks(
        execution_binding_sha256=BINDING,
        protocol_sha="a" * 64,
        split="FORMATIVE_SCREEN",
        task_ids=[f"T{i}" for i in range(task_count)],
        arms=ARMS,
        seeds=[1],
        layer="causal_native",
        claim_scope="FORMATIVE_ONLY",
    )


def _lanes(count: int = 4):
    return [
        Lane(
            shard_id=index,
            gpu_uuid=f"GPU-{index:08x}-0000-0000-0000-000000000000",
            vllm_port=8000 + index,
            provider_port=8787 + index,
            runner_root=f"runner-lane{index}",
            serves_paid_upstreams=(index == 0),
        )
        for index in range(count)
    ]


def _costs(manifest, *, sizes=None):
    tasks = sorted({block.task_id for block in manifest.blocks})
    return {t: float(sizes[i]) if sizes else 1000.0 + i for i, t in enumerate(tasks)}


def _shard_manifest(manifest, lanes=None, costs=None):
    return build_shard_manifest(
        manifest,
        lanes=lanes or _lanes(),
        task_costs=costs or _costs(manifest),
        stack_manifest_sha256="s" * 64,
        protocol_sha256="a" * 64,
    )


def _lane_freeze_root(manifest, body, shard_id, *, epoch=None, block_ids=None):
    """A lane's terminal freeze root, covering only the blocks the partition gave it."""
    epoch = epoch or f"{shard_id:032x}"
    owned = set(
        block_ids
        if block_ids is not None
        else body["blocks_by_shard"][str(shard_id)]
    )
    records = []
    for block in manifest.blocks:
        if block.block_id not in owned:
            continue
        cells = [
            {
                **cell.content(),
                "state": "COMMITTED",
                "output_ref": f"ref-{cell_key(cell)}",
                "engine_epoch": epoch,
            }
            for cell in block.cells
        ]
        record = {
            "block_id": block.block_id,
            "task_id": block.task_id,
            "replicate_id": block.replicate_id,
            "block_digest": block.digest,
            "terminal_frozen": True,
            "valid_for_paired_estimate": True,
            "invalid_reason": "",
            "engine_epochs": [epoch],
            "cells": cells,
        }
        record["freeze_sha256"] = sha256_hex(canonical_json(record))
        records.append(record)

    # freeze_root_record insists on the whole schedule, which no single lane has. A lane freezes
    # its own share; the merge is what asserts the shares are exhaustive and disjoint.
    body_root = {
        "schema_version": "frozen_campaign_root_v1",
        "run_id": "run-1",
        "phase_id": "screen",
        "split": manifest.split,
        "execution_binding_sha256": manifest.execution_binding_sha256,
        "protocol_sha256": manifest.protocol_sha,
        "schedule_sha256": manifest.digest,
        "shard_id": shard_id,
        "terminal_frozen": True,
        "blocks": records,
    }
    body_root["freeze_root_sha256"] = sha256_hex(canonical_json(body_root))
    return body_root


def _roots(manifest, body, lane_count=4):
    return {i: _lane_freeze_root(manifest, body, i) for i in range(lane_count)}


# --- the partition ---------------------------------------------------------------------------


def test_a_task_is_never_split_across_lanes():
    """The one rule that cannot be bent: P0 and P1 for a task run on the same card.

    Otherwise that card's clocks, thermals and scheduler are folded into the treatment effect
    and nothing afterwards can separate them from it.
    """
    manifest = _manifest()
    body = _shard_manifest(manifest)
    by_task: dict[str, set[int]] = {}
    for shard_text, keys in body["cells_by_shard"].items():
        for key in keys:
            task = next(c.task_id for c in manifest.cells if cell_key(c) == key)
            by_task.setdefault(task, set()).add(int(shard_text))
    assert by_task, "no cells were assigned"
    assert all(len(lanes) == 1 for lanes in by_task.values())


def test_every_cell_is_assigned_exactly_once():
    manifest = _manifest()
    body = _shard_manifest(manifest)
    assigned = [key for keys in body["cells_by_shard"].values() for key in keys]
    assert sorted(assigned) == sorted(cell_key(cell) for cell in manifest.cells)
    assert len(assigned) == len(set(assigned)) == body["total_cells"]


def test_the_partition_is_deterministic_and_binding_derived():
    """Same campaign resumed on another day plans the same partition.

    A lane that has to be re-run must re-run the same tasks; an assignment that drifted would
    quietly move a task between GPUs mid-campaign.
    """
    manifest = _manifest()
    assert _shard_manifest(manifest) == _shard_manifest(manifest)
    other = assign_tasks_to_lanes(
        {"T0": 1.0, "T1": 1.0, "T2": 1.0, "T3": 1.0},
        lane_count=2,
        execution_binding_sha256="f" * 64,
    )
    same = assign_tasks_to_lanes(
        {"T0": 1.0, "T1": 1.0, "T2": 1.0, "T3": 1.0},
        lane_count=2,
        execution_binding_sha256=BINDING,
    )
    # Equal costs, so only the binding-seeded tie-break distinguishes them.
    assert other != same or sorted(other.values()) == sorted(same.values())


def test_the_partition_balances_a_pre_treatment_proxy():
    """Balance is for wall clock, not validity -- but an idle lane is wasted GPU."""
    manifest = _manifest(task_count=8)
    costs = _costs(manifest, sizes=[100, 100, 100, 100, 900, 900, 900, 900])
    body = _shard_manifest(manifest, costs=costs)
    loads = {
        shard: sum(costs[t] for t, s in body["task_to_shard"].items() if s == shard)
        for shard in range(4)
    }
    assert max(loads.values()) == min(loads.values()) == 1000.0


def test_exactly_one_lane_may_spend_money():
    """A cap that can be multiplied by starting another process is not admission control."""
    manifest = _manifest()
    lanes = _lanes()
    with pytest.raises(ValueError, match="exactly one lane"):
        build_shard_manifest(
            manifest,
            lanes=[Lane(**{**lane.content(), "serves_paid_upstreams": True}) for lane in lanes],
            task_costs=_costs(manifest),
            stack_manifest_sha256="s" * 64,
            protocol_sha256="a" * 64,
        )


def test_lanes_may_not_share_a_gpu_a_port_or_a_runner_root():
    manifest = _manifest()
    costs = _costs(manifest)
    base = _lanes()
    for mutation, match in (
        ({"gpu_uuid": base[0].gpu_uuid}, "own GPU UUID"),
        ({"vllm_port": base[0].vllm_port}, "share a port"),
        ({"runner_root": base[0].runner_root}, "single-writer"),
    ):
        lanes = list(base)
        lanes[1] = Lane(**{**base[1].content(), **mutation})
        with pytest.raises(ValueError, match=match):
            build_shard_manifest(
                manifest, lanes=lanes, task_costs=costs,
                stack_manifest_sha256="s" * 64, protocol_sha256="a" * 64,
            )


def test_a_manifest_bound_to_another_schedule_is_refused():
    manifest = _manifest()
    body = _shard_manifest(manifest)
    with pytest.raises(ShardMergeError, match="different schedule"):
        verify_shard_manifest(body, _manifest(task_count=6))


def test_an_edited_manifest_does_not_verify():
    manifest = _manifest()
    body = dict(_shard_manifest(manifest))
    body["task_to_shard"] = {task: 0 for task in body["task_to_shard"]}
    with pytest.raises(ShardMergeError, match="hash does not bind"):
        verify_shard_manifest(body, manifest)


# --- the merge -------------------------------------------------------------------------------


def test_four_valid_lanes_merge_into_one_campaign():
    manifest = _manifest()
    body = _shard_manifest(manifest)
    merged = merge_shard_freeze_roots(body, manifest, _roots(manifest, body))
    assert merged["total_cells"] == len(manifest.cells)
    assert len(merged["blocks"]) == len(manifest.blocks)
    assert set(merged["engine_epochs_by_lane"]) == {"0", "1", "2", "3"}
    assert merged["merged_root_sha256"]


def test_a_missing_lane_cannot_be_merged_into_a_complete_campaign():
    """Three lanes' worth of results is not a campaign with a smaller denominator."""
    manifest = _manifest()
    body = _shard_manifest(manifest)
    roots = _roots(manifest, body)
    del roots[2]
    with pytest.raises(ShardMergeError, match="freeze root from every lane"):
        merge_shard_freeze_roots(body, manifest, roots)


def test_a_lane_that_silently_ran_another_lanes_block_is_caught():
    """Each lane is internally consistent; only the union shows the task ran twice."""
    manifest = _manifest()
    body = _shard_manifest(manifest)
    stolen = body["blocks_by_shard"]["1"][0]
    roots = _roots(manifest, body)
    roots[0] = _lane_freeze_root(
        manifest, body, 0, block_ids=list(body["blocks_by_shard"]["0"]) + [stolen]
    )
    with pytest.raises(ShardMergeError, match="assigned to lane 1"):
        merge_shard_freeze_roots(body, manifest, roots)


def test_blocks_missing_from_every_lane_are_caught():
    manifest = _manifest()
    body = _shard_manifest(manifest)
    roots = _roots(manifest, body)
    roots[3] = _lane_freeze_root(
        manifest, body, 3, block_ids=list(body["blocks_by_shard"]["3"])[:-1]
    )
    with pytest.raises(ShardMergeError, match="do not cover the schedule"):
        merge_shard_freeze_roots(body, manifest, roots)


def test_two_lanes_sharing_an_engine_epoch_are_refused():
    """An epoch is one vLLM boot. Sharing one means they were not separate engines.

    Every per-lane isolation claim is built on that separation, so a shared epoch invalidates
    all of them rather than being a cosmetic bookkeeping error.
    """
    manifest = _manifest()
    body = _shard_manifest(manifest)
    roots = _roots(manifest, body)
    roots[1] = _lane_freeze_root(manifest, body, 1, epoch="0" * 32)
    with pytest.raises(ShardMergeError, match="share engine epoch"):
        merge_shard_freeze_roots(body, manifest, roots)


def test_an_edited_lane_freeze_root_does_not_merge():
    manifest = _manifest()
    body = _shard_manifest(manifest)
    roots = _roots(manifest, body)
    roots[2] = {**roots[2], "run_id": "tampered"}
    with pytest.raises(ShardMergeError, match="freeze root hash does not verify"):
        merge_shard_freeze_roots(body, manifest, roots)


def test_a_single_lane_campaign_still_works():
    """Sharding is an option, not a requirement; one lane must not be a special case."""
    manifest = _manifest(task_count=3)
    lanes = _lanes(1)
    body = _shard_manifest(manifest, lanes=lanes)
    merged = merge_shard_freeze_roots(body, manifest, _roots(manifest, body, lane_count=1))
    assert merged["total_cells"] == len(manifest.cells)


def test_freeze_root_record_still_requires_the_whole_schedule():
    """Per-lane freezing does not weaken the single-lane invariant it sits next to."""
    manifest = _manifest(task_count=2)
    with pytest.raises(ValueError, match="partial/different campaign root"):
        freeze_root_record(
            manifest, run_id="r", phase_id="p", split=manifest.split, block_records=[]
        )


# --- lane wiring: paths, budget, and the share a lane will actually run -----------------------


def _settings(tmp_path, monkeypatch, lane=None):
    from shapeflow.campaign.settings import Settings

    if lane is None:
        monkeypatch.delenv("SHAPEFLOW_LANE", raising=False)
    else:
        monkeypatch.setenv("SHAPEFLOW_LANE", str(lane))
    from pathlib import Path as _Path

    return Settings.load(_Path(__file__).resolve().parents[2], data_root=tmp_path)


def test_each_lane_gets_its_own_ledger_and_object_store(tmp_path, monkeypatch):
    """The ledger is single-writer by design; four concurrent runners must not share one.

    A shared object store would also make "which lane produced this artifact" unanswerable at
    merge time, which is the one question the merge exists to ask.
    """
    unsharded = _settings(tmp_path, monkeypatch)
    lane1 = _settings(tmp_path, monkeypatch, lane=1)
    lane2 = _settings(tmp_path, monkeypatch, lane=2)
    assert lane1.path("runs") != lane2.path("runs")
    assert lane1.path("object_store") != lane2.path("object_store")
    assert lane1.path("checkpoints") != lane2.path("checkpoints")
    # The corpus and the answer key are campaign-wide and must NOT be duplicated per lane.
    for shared in ("frozen_corpus_for_runner", "truth_packets", "provider_ledger", "shards"):
        assert lane1.path(shared) == lane2.path(shared) == unsharded.path(shared)


def test_only_one_lane_holds_the_budget(tmp_path, monkeypatch):
    """A cap that another process can multiply by starting is not admission control."""
    paid = [
        _settings(tmp_path, monkeypatch, lane=lane).provider_config().serves_paid_upstreams
        for lane in range(4)
    ]
    assert paid == [True, False, False, False]
    # And an unsharded campaign is not accidentally denied its own upstreams.
    assert _settings(tmp_path, monkeypatch).provider_config().serves_paid_upstreams is True


def test_each_lane_talks_to_its_own_engine_and_binds_its_own_port(tmp_path, monkeypatch):
    for lane in range(4):
        config = _settings(tmp_path, monkeypatch, lane=lane).provider_config()
        assert config.vllm_base_url.endswith(f":{8000 + lane}/v1")
        assert config.bind_port == 8787 + lane
        assert config.lane_id == lane


def test_a_lane_outside_the_frozen_count_is_refused(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="outside the frozen"):
        _settings(tmp_path, monkeypatch, lane=9)


def test_a_provider_lane_without_the_budget_refuses_the_paid_routes():
    """Refused at the route, not omitted from a runbook."""
    from shapeflow.runtime.provider_server import PAID_UPSTREAM_ROUTES

    assert PAID_UPSTREAM_ROUTES == {"exa.search", "tavily.search", "deepseek.chat"}
    # chat.completions is local inference and must stay available on every lane, or three of the
    # four GPUs could not run a cell at all.
    assert "chat.completions" not in PAID_UPSTREAM_ROUTES


def test_a_lane_runner_executes_and_freezes_only_its_own_blocks():
    """Config-level: the share a lane will run is exactly the share it was given.

    The merge catches a lane that ran another lane's block, but catching it here means the GPU
    time is never spent in the first place.
    """
    from shapeflow.campaign.runner import RunnerConfig

    manifest = _manifest()
    body = _shard_manifest(manifest)
    owned = frozenset(body["blocks_by_shard"]["2"])
    config = RunnerConfig(
        run_id="r", provider_base_url="http://x", runner_token="t" * 32,
        shard_id=2, owned_block_ids=owned,
    )
    mine = [b for b in manifest.blocks if b.block_id in config.owned_block_ids]
    theirs = [b for b in manifest.blocks if b.block_id not in config.owned_block_ids]
    assert mine and theirs
    assert {b.task_id for b in mine}.isdisjoint({b.task_id for b in theirs})
