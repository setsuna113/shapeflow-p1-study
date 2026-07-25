"""`run-screen`: the causal screening loop, resumable and honest about what it did not do.

One phase, driven to completion or to a clean stop. Everything that makes it survivable lives in
:mod:`shapeflow_p1.campaign.runner`; this module is the wiring -- open the ledger, build the
frozen schedule, register the run, execute the cells, freeze whole blocks, and keep
``STATUS.json`` current so a human can see what is happening without attaching to the process.

The status file is written on every pass, not only at the end. A run that has to be inspected
mid-flight is the normal case over a week, and a status that only appears on success is exactly
the one you cannot get when you need it.
"""

from __future__ import annotations

import contextlib
import json
import os
from pathlib import Path
from typing import Optional

from ..experiment.ledger import Ledger
from ..experiment.state_machine import Phase
from ..object_store import ObjectStore
from ..providers.provider_client import ProviderClient, load_role_token
from .runner import CampaignRunner, RunnerConfig, available_tasks, questions_for, write_status
from .selector_client import SelectorModelCall
from .settings import Settings


def _now_utc() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

__all__ = ["run_screening", "open_run_ledger", "provider_client_for"]


def open_run_ledger(settings: Settings) -> tuple[Ledger, ObjectStore]:
    runs = settings.path("runs")
    runs.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(runs / "ledger.sqlite"))
    store = ObjectStore(settings.path("object_store"))
    return ledger, store


def provider_client_for(settings: Settings, role: str = "runner") -> ProviderClient:
    host = settings.get("week1", "provider", "bind_host")
    port = settings.get("week1", "provider", "bind_port")
    token_dir = str(settings.get("week1", "provider", "token_dir"))
    return ProviderClient(base_url=f"http://{host}:{port}",
                          token=load_role_token(token_dir, role))


async def run_screening(
    settings: Settings,
    *,
    repo: Path,
    max_cells: Optional[int] = None,
    run_id: Optional[str] = None,
    arms_block: Optional[str] = None,
) -> dict:
    """Execute the screening split in paired blocks, resuming whatever is already committed.

    Holds the GPU lease for the run's lifetime. ops/gpu_lease.py has been correct since it
    was written -- an flock keyed by GPU UUID, released by the kernel on a crash -- and had
    no production caller at all, so the only exclusivity on a shared four-GPU host was an
    `nvidia-smi` process count in start_engine.sh, which is a race rather than a lease.
    """
    lease = _gpu_lease(settings)
    with contextlib.ExitStack() as stack:
        if lease is not None:
            stack.enter_context(lease)
        return await _run_screening_leased(
            settings, repo=repo, max_cells=max_cells, run_id=run_id, arms_block=arms_block)


def _gpu_lease(settings: Settings):
    """The lease for the device this run will use, or None when no UUID is pinned.

    The UUID comes from the environment the engine was started with (CUDA_VISIBLE_DEVICES is
    set to a UUID by start_engine.sh, never an index -- plan §5.2 is explicit that an index
    is not a device identity).
    """
    from ..ops.gpu_lease import GpuLease

    uuid = (os.environ.get("SHAPEFLOW_GPU_UUID")
            or os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not uuid.startswith("GPU-"):
        return None
    lock_file = settings.data_root / str(settings.get("week1", "runtime", "gpu_lease_file"))
    return GpuLease(uuid, lock_dir=lock_file.parent)


async def _run_screening_leased(
    settings: Settings,
    *,
    repo: Path,
    max_cells: Optional[int] = None,
    run_id: Optional[str] = None,
    arms_block: Optional[str] = None,
) -> dict:
    ledger, store = open_run_ledger(settings)
    client = provider_client_for(settings, "runner")
    token = client.token
    split = str(settings.get("week1", "screen", "split"))
    stop_sentinel = settings.data_root / str(settings.get("week1", "runtime", "stop_sentinel"))

    # The run id is derived, not generated: a restart must resume the same run rather than
    # opening a second one beside it with the same work in it.
    run_id = run_id or f"screen-{settings.shas['week1'][:12]}"

    async def register(spec):
        await client.register_cell(
            cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
            arm_id=spec.arm_id, variant_id=spec.variant_id, replicate_id=spec.replicate_id,
            work_key=spec.work_key)

    def model_call_factory(cell_token: str) -> SelectorModelCall:
        return SelectorModelCall(
            client, cell_token=cell_token, repo=repo,
            temperature=float(settings.get("stack", "sampling", "temperature")),
            top_p=float(settings.get("stack", "sampling", "top_p")),
            max_completion_tokens=int(settings.get("week1", "measurement",
                                                   "selector_max_completion_tokens")),
            guided_decoding=bool(settings.get("week1", "measurement", "guided_decoding")),
        )

    config = RunnerConfig(
        run_id=run_id,
        provider_base_url=client.base_url,
        runner_token=token,
        lease_seconds=float(settings.get("week1", "runtime", "lease_seconds")),
        max_cells=max_cells,
        stop_sentinel=stop_sentinel,
    )
    runner = CampaignRunner(settings, ledger=ledger, store=store, config=config,
                            model_call_factory=model_call_factory, register_cell=register)

    tasks = available_tasks(settings, split)
    if not tasks:
        ledger.close()
        return {"ok": False, "run_id": run_id,
                "error": f"no {split} task has a frozen world; acquisition must run first"}

    arms = runner.arms_from_config(
        arms_block or str(settings.get("week1", "screen", "arms_block")))
    manifest = runner.build_schedule(
        task_ids=tasks, arms=arms, split=split,
        second_seed_fraction=float(settings.get("week1", "screen", "second_seed_fraction")),
    )
    schedule_path = settings.path("runs") / "screen_schedule.json"
    schedule_sha = runner.freeze_schedule(manifest, schedule_path)

    runner.phases.begin(Phase.SCREEN_RUNNING, reason=f"schedule {schedule_sha[:12]}")
    ledger.create_run(run_id, settings.shas["week1"],
                      json.dumps({"split": split, "schedule_sha256": schedule_sha,
                                  "claim_scope": settings.claim_scope}, sort_keys=True))

    status_path = repo / "reports" / "STATUS.json"
    write_status({**runner.status(manifest, phase_id="run-screen", split=split),
                  "schedule_sha256": schedule_sha, "stage": "running"}, status_path)

    await runner.run_cells(manifest, phase_id="run-screen", split=split,
                           questions=questions_for(settings, tasks))

    frozen = runner.freeze_blocks(manifest, phase_id="run-screen", split=split,
                                  directory=settings.path("runs") / "blocks")
    body = runner.status(manifest, phase_id="run-screen", split=split)
    body.update({
        "schedule_sha256": schedule_sha,
        "blocks_frozen": len(frozen),
        "frozen_block_ids": sorted(b["block_id"] for b in frozen),
        "stopped_early": runner.stop_requested(),
        "stage": "complete" if body["cells_committed"] == body["cells_total"] else "partial",
        "budget": _budget_snapshot(settings),
    })
    write_status(body, status_path)

    if body["cells_committed"] == body["cells_total"]:
        runner.phases.begin(Phase.SCREEN_COMPLETE)
        runner.phases.complete(Phase.SCREEN_COMPLETE, {
            "cells": body["cells_total"], "blocks": body["blocks_total"],
            "schedule_sha256": schedule_sha,
        })
    if runner.stop_requested():
        # The counterpart to the stop sentinel. stop_safely.sh waits for this file and
        # nothing had ever written it, so every graceful stop burned its whole timeout and
        # then fell through to killing a process group.
        clean = stop_sentinel.parent / "STOPPED_CLEAN"
        clean.parent.mkdir(parents=True, exist_ok=True)
        clean.write_text(json.dumps({
            "run_id": run_id, "stopped_at_utc": _now_utc(),
            "cells_committed": body["cells_committed"],
            "cells_total": body["cells_total"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger.close()
    body["ok"] = body["blocks_complete"] > 0
    body["run_id"] = run_id
    return body


def _budget_snapshot(settings: Settings) -> dict:
    """What the provider has spent so far, read from its own ledger.

    Read-only and best effort: the runner has no business writing to the provider's accounts,
    and a status probe must never be able to disturb admission control.
    """
    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        return {}
    import sqlite3

    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT resource, cap, reserved_total, settled_total FROM budget_accounts"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return {}
    return {
        r["resource"]: {
            "cap": r["cap"], "used": r["settled_total"],
            "reserved": r["reserved_total"],
            "remaining": r["cap"] - r["settled_total"] - r["reserved_total"],
        }
        for r in rows
    }


def free_disk_bytes(path: Path) -> int:
    stat = os.statvfs(path)
    return stat.f_bavail * stat.f_frsize
