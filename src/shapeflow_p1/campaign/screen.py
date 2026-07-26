"""`run-screen`: coupled-seed end-to-end ITT, with an optional component mode.

The primary screen runs every assigned arm through the complete graph. Arms share task,
replicate and random seed; once P1 first intervenes, different searches, checkpoints and
reasoning trajectories are outcomes rather than parity failures. ``component_only=True`` asks
the separate reducer-at-an-identical-boundary question. That mode needs a validated
:class:`BoundaryForkBackend`; the pinned graph does not currently expose one, so it fails before
provider connection or ledger mutation instead of relabelling independent graph reruns as
component forks.

The status file is written on every pass, not only at the end. A run that has to be inspected
mid-flight is the normal case over a week, and a status that only appears on success is exactly
the one you cannot get when you need it.
"""

from __future__ import annotations

import contextlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from ..canonical import canonical_json
from ..experiment.ledger import Ledger
from ..experiment.state_machine import Phase
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from ..protocol import verified_execution_binding
from ..providers.provider_client import ProviderClient, load_role_token
from .fork import (
    BoundaryForkBackend,
    ForkCapabilityError,
    TrialKind,
    plan_forks,
    run_production_forks,
    write_fork_record,
)
from .runner import CampaignRunner, RunnerConfig, available_tasks, questions_for, write_status
from .selector_client import SelectorModelCall
from .settings import Settings

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
    fork_backend: Optional[BoundaryForkBackend] = None,
    component_only: bool = False,
    execution_binding_sha256: Optional[str] = None,
) -> dict:
    """Execute the primary coupled-seed end-to-end ITT screen.

    Different searches and checkpoints *after* P1's first intervention are end-to-end
    outcomes. ``component_only`` asks the separate same-checkpoint reducer question and
    therefore requires a validated boundary backend.

    Holds the GPU lease for the run's lifetime. ops/gpu_lease.py has been correct since it
    was written -- an flock keyed by GPU UUID, released by the kernel on a crash -- and had
    no production caller at all, so the only exclusivity on a shared four-GPU host was an
    `nvidia-smi` process count in start_engine.sh, which is a race rather than a lease.
    """
    if component_only and fork_backend is None:
        # This check is intentionally before provider connection, ledger mutation and GPU
        # lease.  The pinned actual graph currently has no validated intermediate-state resume
        # adapter.  Re-running seven whole graphs would be an independent E2E pilot, not the
        # forked-state component screen declared by configs/week1.yaml.
        return {
            "ok": False,
            "run_id": run_id or "screen-not-started",
            "error": (
                "NO_BOUNDARY_FORK_BACKEND: component-only screen requested but the "
                "pinned graph cannot replay H/C checkpoints. No model call was admitted."
            ),
            "required_semantics": {
                "component": "same real checkpoint; zero upstream researcher reruns",
                "h_e2e": "same first-H checkpoint; downstream trajectory may diverge",
                "c_only": "same close checkpoint; zero upstream researcher reruns",
                "hxc": "C0/C1 nested inside each natural H0/H1 close checkpoint",
            },
        }
    binding = verified_execution_binding(
        repo, expected_digest=execution_binding_sha256)
    lease = _gpu_lease(settings)
    with contextlib.ExitStack() as stack:
        stack.enter_context(lease)
        return await _run_screening_leased(
            settings, repo=repo, max_cells=max_cells, run_id=run_id, arms_block=arms_block,
            fork_backend=fork_backend, component_only=component_only,
            execution_binding_sha256=binding.digest,
            protocol_document_sha256=binding.protocol_sha)


def _gpu_lease(settings: Settings):
    """The lease for the device this run will use.

    The UUID comes from the environment the engine was started with (CUDA_VISIBLE_DEVICES is
    set to a UUID by start_engine.sh, never an index -- plan §5.2 is explicit that an index
    is not a device identity).

    Returning None when nothing was pinned -- which is what this did -- meant the lease was
    silently skipped exactly when it mattered: neither ``sfsupervise`` nor ``bootstrap``'s
    privilege-drop helper passed these variables through, so in production the mutual
    exclusion never engaged at all. On a host with several cards and more than one worker,
    that is how two runs end up on one device, and nothing downstream would show it. An
    unleasable run is refused instead.
    """
    from ..ops.gpu_lease import GpuLease

    uuid = (os.environ.get("SHAPEFLOW_GPU_UUID")
            or os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError(
            "no GPU UUID in the environment: set SHAPEFLOW_GPU_UUID (or CUDA_VISIBLE_DEVICES) "
            "to the leased device UUID. Refusing to run unleased, because the GPU lease is "
            f"what stops a second worker using the same card (saw {uuid!r})"
        )
    lock_file = settings.data_root / str(settings.get("week1", "runtime", "gpu_lease_file"))
    return GpuLease(uuid, lock_dir=lock_file.parent)


async def _run_screening_leased(
    settings: Settings,
    *,
    repo: Path,
    max_cells: Optional[int] = None,
    run_id: Optional[str] = None,
    arms_block: Optional[str] = None,
    fork_backend: Optional[BoundaryForkBackend] = None,
    component_only: bool = False,
    execution_binding_sha256: str,
    protocol_document_sha256: str,
) -> dict:
    if component_only and fork_backend is None:
        raise ForkCapabilityError(
            "component_only requires a boundary backend; use run_screening so absence is "
            "reported before any mutation"
        )
    ledger, store = open_run_ledger(settings)
    client = provider_client_for(settings, "runner")
    token = client.token
    split = str(settings.get("week1", "screen", "split"))
    stop_sentinel = settings.data_root / str(settings.get("week1", "runtime", "stop_sentinel"))

    # The run id is derived, not generated: a restart must resume the same run rather than
    # opening a second one beside it with the same work in it.
    run_id = run_id or f"screen-{execution_binding_sha256[:12]}"

    async def register(spec):
        await client.register_cell(
            cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
            arm_id=spec.arm_id, variant_id=spec.variant_id, replicate_id=spec.replicate_id,
            work_key=spec.work_key,
            layer=str(settings.get("week1", "measurement", "layer")))

    async def fetch_work_summary(work_key: str):
        return await client.work_summary(
            work_key=work_key,
            require_isolated=(
                str(settings.get("week1", "measurement", "layer")) == "causal"
            ),
        )

    def model_call_factory(cell_token: str, *, seed: int) -> SelectorModelCall:
        return SelectorModelCall(
            client, cell_token=cell_token, repo=repo,
            temperature=float(settings.get("stack", "sampling", "temperature")),
            top_p=float(settings.get("stack", "sampling", "top_p")),
            max_completion_tokens=int(settings.get("week1", "measurement",
                                                   "selector_max_completion_tokens")),
            seed=seed,
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
                            execution_binding_sha256=execution_binding_sha256,
                            protocol_document_sha256=protocol_document_sha256,
                            model_call_factory=model_call_factory, register_cell=register,
                            fetch_work_summary=fetch_work_summary)

    tasks = available_tasks(settings, split)
    if not tasks:
        ledger.close()
        return {"ok": False, "run_id": run_id,
                "execution_binding_sha256": execution_binding_sha256,
                "protocol_document_sha256": protocol_document_sha256,
                "error": f"no {split} task has a frozen world; acquisition must run first"}

    arms = runner.arms_from_config(
        arms_block or str(settings.get("week1", "screen", "arms_block")))
    manifest = runner.build_schedule(
        task_ids=tasks, arms=arms, split=split,
        second_seed_fraction=float(settings.get("week1", "screen", "second_seed_fraction")),
    )
    from ..analysis.design import load_runner_analysis_design_receipt

    analysis_receipt = load_runner_analysis_design_receipt(settings)
    manifest.notes.update({
        "execution_semantics": (
            "BOUNDARY_COMPONENT_ANCHOR_ASSIGNMENT"
            if component_only else "COUPLED_SEED_E2E_ITT"
        ),
        "full_graph_per_arm": not component_only,
        "downstream_trajectory_divergence": (
            "not_applicable_to_component_screen"
            if component_only else "measured_post_treatment_outcome"
        ),
        # Makes "features/spec were frozen before outcomes" part of the schedule and therefore
        # FREEZE_ROOT, rather than an uncheckable timestamp claim.
        "analysis_design_receipt_sha256":
            analysis_receipt["content_sha256"],
        "task_feature_registry_sha256":
            analysis_receipt["task_feature_registry_sha256"],
        "eligibility_spec_content_sha256":
            analysis_receipt["eligibility_spec_content_sha256"],
    })
    schedule_path = settings.path("runs") / "schedules" / run_id / (
        "component.json" if component_only else "e2e.json"
    )
    schedule_sha = runner.freeze_schedule(manifest, schedule_path)

    if component_only:
        # Component screening is boundary work, not one complete graph per arm.  The manifest
        # retains task/replicate assignment and arm order before outcomes exist; each block
        # supplies one P0 anchor whose H/C checkpoints are replayed.
        return await _run_component_forks(
            settings,
            repo=repo,
            runner=runner,
            ledger=ledger,
            manifest=manifest,
            questions=questions_for(settings, tasks),
            run_id=run_id,
            split=split,
            schedule_sha=schedule_sha,
            fork_backend=fork_backend,
            max_boundaries=max_cells,
        )

    return await _run_e2e_screen(
        settings,
        repo=repo,
        runner=runner,
        ledger=ledger,
        manifest=manifest,
        questions=questions_for(settings, tasks),
        run_id=run_id,
        split=split,
        schedule_sha=schedule_sha,
        stop_sentinel=stop_sentinel,
    )


async def _run_e2e_screen(
    settings: Settings,
    *,
    repo: Path,
    runner: CampaignRunner,
    ledger: Ledger,
    manifest,
    questions: dict,
    run_id: str,
    split: str,
    schedule_sha: str,
    stop_sentinel: Path,
) -> dict:
    """Run complete graphs and retain their natural post-treatment trajectories.

    Same task and replicate seed couples the arms.  Pre-treatment equality is diagnosed, not
    assumed; a mismatch stays in the ITT denominator and is surfaced for sensitivity analysis.
    After the first H/C treatment event, different queries, rounds, sources and close reasons
    are mediated outcomes and are never overwritten with P0's trajectory.
    """
    phase_id = "e2e-screen"
    execution_semantics = "COUPLED_SEED_E2E_ITT"
    runner.phases.begin(Phase.SCREEN_RUNNING, reason=f"E2E schedule {schedule_sha[:12]}")
    ledger.create_run(run_id, runner.execution_binding_sha256, json.dumps({
        "split": split,
        "schedule_sha256": schedule_sha,
        "execution_binding_sha256": runner.execution_binding_sha256,
        "protocol_document_sha256": runner.protocol_document_sha256,
        "claim_scope": settings.claim_scope,
        "execution_semantics": execution_semantics,
        "post_treatment_trajectory_divergence": "OUTCOME",
    }, sort_keys=True))

    status_path = repo / "reports" / "STATUS.json"
    write_status({
        **runner.status(manifest, phase_id=phase_id, split=split),
        "schedule_sha256": schedule_sha,
        "execution_semantics": execution_semantics,
        "stage": "running",
    }, status_path)

    await runner.run_cells(
        manifest, phase_id=phase_id, split=split, questions=questions)
    frozen = runner.freeze_blocks(
        manifest,
        phase_id=phase_id,
        split=split,
        directory=settings.path("runs") / "e2e_blocks" / run_id,
    )
    trajectory = _e2e_trajectory_diagnostics(
        runner, manifest, phase_id=phase_id, split=split)
    _write_once_json(
        settings.path("runs") / "trajectory_diagnostics" / f"{run_id}.json", trajectory)

    body = runner.status(manifest, phase_id=phase_id, split=split)
    body.update({
        "schedule_sha256": schedule_sha,
        "execution_semantics": execution_semantics,
        "post_treatment_trajectory_divergence": "OUTCOME",
        "blocks_frozen": len(frozen),
        "frozen_block_ids": sorted(b["block_id"] for b in frozen),
        "trajectory_diagnostics_sha256": trajectory["content_sha256"],
        "pre_treatment_pairs_checked": trajectory["pairs_checked"],
        "pre_treatment_pairs_matched": trajectory["pre_treatment_pairs_matched"],
        "pre_treatment_pairs_mismatched": trajectory["pre_treatment_pairs_mismatched"],
        "stopped_early": runner.stop_requested(),
        "stage": (
            "complete" if body["cells_terminal"] == body["cells_total"] else "partial"
        ),
        "budget": _budget_snapshot(settings),
    })
    write_status(body, status_path)

    if body["cells_terminal"] == body["cells_total"]:
        runner.phases.complete(Phase.SCREEN_RUNNING, {
            "cells_assigned": body["cells_total"],
            "cells_terminal": body["cells_terminal"],
            "schedule_sha256": schedule_sha,
            "execution_semantics": execution_semantics,
        })
        runner.phases.begin(Phase.SCREEN_COMPLETE)
        runner.phases.complete(Phase.SCREEN_COMPLETE, {
            "cells_assigned": body["cells_total"],
            "cells_committed": body["cells_committed"],
            "cells_terminal": body["cells_terminal"],
            "blocks_terminal": body["blocks_terminal"],
            "schedule_sha256": schedule_sha,
            "execution_semantics": execution_semantics,
        })
    if runner.stop_requested():
        clean = stop_sentinel.parent / "STOPPED_CLEAN"
        clean.parent.mkdir(parents=True, exist_ok=True)
        clean.write_text(json.dumps({
            "run_id": run_id,
            "stopped_at_utc": _now_utc(),
            "cells_terminal": body["cells_terminal"],
            "cells_total": body["cells_total"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    ledger.close()
    # A few completed blocks do not mean the coordinator finished. Unexpected partial state
    # must return non-zero so supervision resumes it; an explicit STOP_REQUESTED is the only
    # clean partial termination.
    body["ok"] = (
        body["cells_terminal"] == body["cells_total"] or runner.stop_requested()
    )
    body["run_id"] = run_id
    return body


def _e2e_trajectory_diagnostics(
    runner: CampaignRunner, manifest, *, phase_id: str, split: str
) -> dict:
    """Compare only the prefix before each P1 arm's first treatment.

    This is a diagnostic/sensitivity artifact, not an inclusion filter.  Conditioning the ITT
    result on a post-hoc "nice trajectory" subset would bias the primary estimate.  It answers a
    different question: whether common-random-number coupling actually aligned the untreated
    prefix, and where natural treatment-mediated divergence began.
    """

    def load(cell) -> dict:
        work_key = runner.work_key_for(cell, phase_id=phase_id, split=split)
        ref = runner.ledger.terminal_ref(work_key)
        if not ref or not runner.store.verify(ref):
            return {}
        try:
            return json.loads(runner.store.get_bytes(ref).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return {}

    def semantic(event: dict) -> dict:
        return {
            key: value for key, value in event.items()
            if key not in {"event_index", "event_sha256", "position"}
        }

    rows: list[dict] = []
    matched = mismatched = checked = 0
    for block in manifest.blocks:
        cells = sorted(block.cells, key=lambda cell: cell.order_index)
        baseline_cell = next(
            (cell for cell in cells if cell.arm.arm_id == "P0"), None)
        baseline = load(baseline_cell) if baseline_cell is not None else {}
        baseline_events = list(baseline.get("events") or ())
        for cell in cells:
            if cell.arm.arm_id == "P0":
                continue
            record = load(cell)
            events = list(record.get("events") or ())
            treatment_index = next(
                (i for i, event in enumerate(events)
                 if event.get("position") == "TREATMENT"),
                None,
            )
            digest = str(record.get("first_treatment_checkpoint_digest") or "")
            if treatment_index is None or not digest or not baseline_events:
                rows.append({
                    "block_id": block.block_id,
                    "task_id": block.task_id,
                    "replicate_id": block.replicate_id,
                    "arm_id": cell.arm.arm_id,
                    "status": "DIAGNOSTIC_UNAVAILABLE",
                    "first_treatment_checkpoint_digest": digest,
                })
                continue
            checked += 1
            left = [semantic(event) for event in baseline_events[:treatment_index]]
            right = [semantic(event) for event in events[:treatment_index]]
            prefix_equal = left == right
            checkpoint_seen_in_p0 = any(
                str(event.get("checkpoint") or "") == digest
                for event in baseline_events[:treatment_index + 1]
            )
            pre_match = prefix_equal and checkpoint_seen_in_p0
            matched += int(pre_match)
            mismatched += int(not pre_match)
            rows.append({
                "block_id": block.block_id,
                "task_id": block.task_id,
                "replicate_id": block.replicate_id,
                "arm_id": cell.arm.arm_id,
                "status": "MATCH" if pre_match else "PRE_TREATMENT_DIVERGENCE",
                "first_treatment_event_index": treatment_index,
                "first_treatment_checkpoint_digest": digest,
                "checkpoint_seen_in_p0_prefix": checkpoint_seen_in_p0,
                "p0_prefix_sha256": sha256_hex(canonical_json(left)),
                "p1_prefix_sha256": sha256_hex(canonical_json(right)),
                "post_treatment_is_outcome": True,
                "p0_trajectory_summary": baseline.get("trajectory_summary") or {},
                "p1_trajectory_summary": record.get("trajectory_summary") or {},
            })
    body = {
        "schema_version": "e2e_trajectory_diagnostics_v1",
        "execution_semantics": "COUPLED_SEED_E2E_ITT",
        "itt_inclusion_filter": False,
        "pairs_checked": checked,
        "pre_treatment_pairs_matched": matched,
        "pre_treatment_pairs_mismatched": mismatched,
        "rows": rows,
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _write_once_json(path: Path, body: dict) -> None:
    path = Path(path)
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing.get("content_sha256") == body.get("content_sha256"):
            return
        raise RuntimeError(f"{path} already contains different trajectory diagnostics")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _run_component_forks(
    settings: Settings,
    *,
    repo: Path,
    runner: CampaignRunner,
    ledger: Ledger,
    manifest,
    questions: dict,
    run_id: str,
    split: str,
    schedule_sha: str,
    fork_backend: BoundaryForkBackend,
    max_boundaries: Optional[int],
) -> dict:
    """Capture each anchor once, then replay reducers from its immutable H/C checkpoints."""
    page_variants = sorted({
        cell.arm.page_variant for cell in manifest.cells
        if cell.arm.page_variant != "P0"
    })
    close_variants = sorted({
        cell.arm.close_variant for cell in manifest.cells
        if cell.arm.close_variant != "P0"
    })
    if page_variants and not fork_backend.supports(TrialKind.COMPONENT, "H"):
        raise ForkCapabilityError(
            "component screen requests H variants but backend cannot replay H")
    if close_variants and not fork_backend.supports(TrialKind.COMPONENT, "C"):
        raise ForkCapabilityError(
            "component screen requests C variants but backend cannot replay C")

    runner.phases.begin(Phase.SCREEN_RUNNING, reason=f"fork schedule {schedule_sha[:12]}")
    ledger.create_run(run_id, runner.execution_binding_sha256, json.dumps({
        "split": split,
        "schedule_sha256": schedule_sha,
        "execution_binding_sha256": runner.execution_binding_sha256,
        "protocol_document_sha256": runner.protocol_document_sha256,
        "claim_scope": settings.claim_scope,
        "execution_semantics": "SHARED_BOUNDARY_COMPONENT",
    }, sort_keys=True))

    status_path = repo / "reports" / "STATUS.json"
    record_dir = settings.path("runs") / "component_forks" / run_id
    max_h = int(settings.get("week1", "screen", "max_h_states"))
    max_c = int(settings.get("week1", "screen", "max_c_states"))
    per_state = int(settings.get("week1", "screen", "variants_per_state"))
    h_seen = c_seen = offered = completed = failures = 0
    anchor_records: list[dict] = []

    write_status({
        "run_id": run_id,
        "execution_binding_sha256": runner.execution_binding_sha256,
        "protocol_document_sha256": runner.protocol_document_sha256,
        "stage": "capturing-boundaries",
        "schedule_sha256": schedule_sha,
        "execution_semantics": "SHARED_BOUNDARY_COMPONENT",
        "boundaries_offered": 0,
    }, status_path)

    for block in manifest.blocks:
        if max_boundaries is not None and offered >= max_boundaries:
            break
        seed = block.cells[0].seed
        captures = await fork_backend.capture_boundaries(
            task_id=block.task_id,
            question=questions[block.task_id],
            seed=seed,
        )
        for capture in sorted(captures, key=lambda c: (
                c.boundary_kind, c.checkpoint_digest)):
            if max_boundaries is not None and offered >= max_boundaries:
                break
            if capture.task_id != block.task_id or capture.seed != seed:
                raise ForkCapabilityError(
                    "boundary backend returned a capture for different task/seed coordinates"
                )
            if (not capture.anchor_run_ref
                    or len(capture.upstream_trace_sha256) != 64
                    or len(capture.checkpoint_digest) != 64):
                raise ForkCapabilityError(
                    "boundary capture lacks content-addressed anchor/trajectory provenance"
                )
            if not capture.seed_applied:
                raise ForkCapabilityError(
                    f"anchor {capture.anchor_run_ref} recorded seed {seed} but did not send it"
                )
            if capture.boundary_kind == "H":
                if h_seen >= max_h or not page_variants:
                    continue
                variants = page_variants[:per_state]
                h_seen += 1
            elif capture.boundary_kind == "C":
                if c_seen >= max_c or not close_variants:
                    continue
                variants = close_variants[:per_state]
                c_seen += 1
            else:
                raise ForkCapabilityError(
                    f"backend returned unknown boundary kind {capture.boundary_kind!r}"
                )

            checkpoint = runner.settings.path("checkpoints")
            from ..odr.checkpoints import CheckpointStore

            loaded = CheckpointStore(checkpoint).get(capture.checkpoint_digest)
            actual_kind = "H" if type(loaded).__name__ == "HCheckpoint" else "C"
            if actual_kind != capture.boundary_kind:
                raise ForkCapabilityError(
                    f"capture calls {capture.checkpoint_digest} {capture.boundary_kind}, "
                    f"stored object is {actual_kind}"
                )
            plan = plan_forks(
                settings,
                checkpoint_digest=capture.checkpoint_digest,
                boundary_kind=capture.boundary_kind,
                task_id=capture.task_id,
                variant_ids=variants,
                seed=seed,
                trial_kind=TrialKind.COMPONENT,
                execution_binding_sha256=runner.execution_binding_sha256,
            )
            outcomes = await run_production_forks(
                settings, plan, backend=fork_backend,
                checkpoint_store=CheckpointStore(checkpoint),
            )
            write_fork_record(record_dir, plan, outcomes)
            offered += 1
            complete_here = sum(o.state == "COMMITTED" for o in outcomes)
            completed += complete_here
            failures += len(outcomes) - complete_here
            anchor_records.append({
                "anchor_run_ref": capture.anchor_run_ref,
                "upstream_trace_sha256": capture.upstream_trace_sha256,
                "checkpoint_digest": capture.checkpoint_digest,
                "boundary_kind": capture.boundary_kind,
                "seed": seed,
                "forks": len(outcomes),
                "committed": complete_here,
            })
            write_status({
                "run_id": run_id,
                "execution_binding_sha256": runner.execution_binding_sha256,
                "protocol_document_sha256": runner.protocol_document_sha256,
                "stage": "running-component-forks",
                "schedule_sha256": schedule_sha,
                "execution_semantics": "SHARED_BOUNDARY_COMPONENT",
                "boundaries_offered": offered,
                "forks_committed": completed,
                "forks_failed": failures,
            }, status_path)

    body = {
        "ok": offered > 0,
        "run_id": run_id,
        "execution_binding_sha256": runner.execution_binding_sha256,
        "protocol_document_sha256": runner.protocol_document_sha256,
        "stage": "complete",
        "schedule_sha256": schedule_sha,
        "execution_semantics": "SHARED_BOUNDARY_COMPONENT",
        "boundaries_offered": offered,
        "h_boundaries": h_seen,
        "c_boundaries": c_seen,
        "forks_committed": completed,
        "forks_failed": failures,
        "anchors": anchor_records,
        "stopped_early": runner.stop_requested(),
        "claim_scope": settings.claim_scope,
        "budget": _budget_snapshot(settings),
    }
    write_status(body, status_path)
    if offered > 0:
        runner.phases.complete(Phase.SCREEN_RUNNING, {
            "boundaries": offered,
            "forks_committed": completed,
            "forks_failed": failures,
            "schedule_sha256": schedule_sha,
            "execution_semantics": "SHARED_BOUNDARY_COMPONENT",
        })
        runner.phases.begin(Phase.SCREEN_COMPLETE)
        runner.phases.complete(Phase.SCREEN_COMPLETE, {
            "boundaries": offered,
            "forks_committed": completed,
            "forks_failed": failures,
            "schedule_sha256": schedule_sha,
            "execution_semantics": "SHARED_BOUNDARY_COMPONENT",
        })
    ledger.close()
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
