"""The GPU canary: prove the P1 path is real, and prove nothing about whether it is good.

This is the first time the study touches a GPU, and it checks exactly one class of thing --
engineering correctness. It may never gate on a quality outcome. Deciding whether to continue
based on how good the early results look is outcome-dependent design: it turns the screen into
a selection on the very effect the screen exists to estimate, and no amount of care downstream
recovers from it.

What it does have to establish is that a green run means anything at all. The failure this is
built against is a P1 arm that *appears* to work while doing nothing: the strategy never bound,
the selector never decoded, the publication copied P0's bytes and changed a label. Each check
below closes one of those, and each is derived from an artifact the run produced rather than
from a flag the run set about itself.
"""

from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from ..canonical import canonical_json
from ..experiment.state_machine import Phase
from ..hashing import sha256_hex
from ..protocol import verified_execution_binding
from ..providers.provider_client import ProviderCallError
from ..runtime.request_tags import OpClass
from .runner import CampaignRunner, RunnerConfig, available_tasks, questions_for
from .schedule import cell_key
from .screen import open_run_ledger, provider_client_for
from .selector_client import SHORT_PROSE_OPS, STRUCTURED_SELECTOR_OPS, SelectorModelCall
from .settings import Settings

__all__ = ["run_canary", "CANARY_CHECKS"]

PASS, FAIL = "PASS", "FAIL"

GPU_SETTLEMENT_ABS_TOLERANCE_SECONDS = 1e-6
GPU_SETTLEMENT_REL_TOLERANCE = 1e-9

#: Every model-backed decode at an H or C boundary, structured *and* short-prose. Derived from
#: the selector client rather than restated, because this list had drifted: it named only the
#: structured ops, so a SHORT_PROSE control's decode was invisible to `selector_decode`, to the
#: CPU-control zero-decode check and to the decode-cap audit.
SELECTOR_OPS = STRUCTURED_SELECTOR_OPS | SHORT_PROSE_OPS

#: One bound "?" per member, so the ledger query cannot go stale when the set grows.
_PLACEHOLDERS = ",".join("?" * len(SELECTOR_OPS))

CANARY_CHECKS = (
    "patched_graph_invoked",
    "p1_strategy_invocations",
    "p1_published_output",
    "selector_decode",
    "checkpoints_present",
    "direct_denominator_provenance",
    "citation_source_lineage",
    "atomic_publication",
    "p1_bytes_differ_from_p0",
    "truth_invisible_to_treatment",
    "no_live_search_miss",
    "reservations_closed",
    "fallback_and_failures_accounted",
    "cpu_control_zero_decode",
    "short_prose_same_budget",
    "generation_cap_is_a_completion_limit",
    "provider_canary_audit",
    "projected_gpu_budget_feasible",
)


def _check(name: str, ok: bool, detail: str, data: dict | None = None) -> dict:
    """One gate's verdict, optionally carrying the per-arm numbers behind it.

    ``data`` exists so a diagnosis does not have to be reconstructed from the object store after
    the fact. When every H arm published nothing, the counters said "reduced" and the reason
    lived only in per-span records that nobody reads unless they already suspect something.
    """
    result = {"name": name, "status": PASS if ok else FAIL, "detail": detail}
    if data:
        result["data"] = data
    return result


async def run_canary(settings: Settings, *, repo: Path,
                     task_limit: int | None = None,
                     execution_binding_sha256: str | None = None) -> dict:
    """Hold the same UUID-scoped GPU lease as the campaign for the canary's lifetime."""
    from .screen import _gpu_lease

    binding = verified_execution_binding(
        repo, expected_digest=execution_binding_sha256)
    lease = _gpu_lease(settings)
    with lease:
        return await _run_canary_leased(
            settings,
            repo=repo,
            task_limit=task_limit,
            execution_binding_sha256=binding.digest,
            protocol_document_sha256=binding.protocol_sha,
        )


async def _run_canary_leased(settings: Settings, *, repo: Path,
                             task_limit: int | None = None,
                             execution_binding_sha256: str,
                             protocol_document_sha256: str) -> dict:
    """Run the canary arms and gate the full screen on engineering and cost feasibility."""
    ledger, store = open_run_ledger(settings)
    client = provider_client_for(settings, "runner")
    split = str(settings.get("week1", "screen", "split"))
    limit = task_limit or int(settings.get("week1", "canary", "tasks"))
    screen_tasks = available_tasks(settings, split)
    tasks = screen_tasks[:limit]
    if len(tasks) < limit:
        ledger.close()
        return _report([_check("frozen_world", False,
                               f"{len(tasks)} tasks have a frozen world, need {limit}")],
                       execution_binding_sha256=execution_binding_sha256,
                       protocol_document_sha256=protocol_document_sha256)

    # Under sharding each lane takes its own share of the canary's tasks, and every arm of a
    # task still runs on one lane. Four lanes each running all four canary tasks would be four
    # copies of the same smoke -- it would not exercise the partition, and it would not show
    # whether a task's arms can survive being confined to one card.
    lane = settings.lane_id
    if lane is not None:
        from .sharding import assign_tasks_to_lanes

        lane_count = int(settings.get("week1", "measurement", "shards", "lane_count"))
        assignment = assign_tasks_to_lanes(
            {task_id: 1.0 for task_id in tasks},
            lane_count=lane_count,
            execution_binding_sha256=execution_binding_sha256,
        )
        tasks = [task_id for task_id in tasks if assignment[task_id] == lane]
        if not tasks:
            ledger.close()
            return _report(
                [_check("frozen_world", False,
                        f"lane {lane} was given none of the {limit} canary tasks; the canary "
                        f"needs at least {lane_count} tasks to cover every lane")],
                execution_binding_sha256=execution_binding_sha256,
                protocol_document_sha256=protocol_document_sha256)

    layer = str(settings.get("week1", "measurement", "layer"))

    async def register(spec):
        await client.register_cell(
            cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
            arm_id=spec.arm_id, variant_id=spec.variant_id, replicate_id=spec.replicate_id,
            work_key=spec.work_key, layer=layer)

    async def fetch_work_summary(work_key: str):
        return await client.work_summary(
            work_key=work_key,
            require_isolated=settings.layer_is_serialized,
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
        run_id=f"canary-{execution_binding_sha256[:12]}",
        provider_base_url=client.base_url, runner_token=client.token,
        lease_seconds=float(settings.get("week1", "runtime", "lease_seconds")),
    )
    runner = CampaignRunner(settings, ledger=ledger, store=store, config=config,
                            execution_binding_sha256=execution_binding_sha256,
                            protocol_document_sha256=protocol_document_sha256,
                            model_call_factory=model_call_factory, register_cell=register,
                            fetch_work_summary=fetch_work_summary)
    arms = runner.arms_from_config("canary")
    manifest = runner.build_schedule(task_ids=tasks, arms=arms, split=split)

    # Freeze the exact denominator used by the cost projection before the first canary cell
    # runs.  Counting tasks from a YAML comment or assuming one seed would silently understate
    # a later design change.  Building the prospective screen schedule exercises the same
    # deterministic replicate rule as run-screen, while the task/arm hashes make the count
    # independently auditable without copying the whole screen schedule into this artifact.
    screen_arms = runner.arms_from_config(
        str(settings.get("week1", "screen", "arms_block")))
    screen_manifest = runner.build_schedule(
        task_ids=screen_tasks,
        arms=screen_arms,
        split=split,
        second_seed_fraction=float(
            settings.get("week1", "screen", "second_seed_fraction")),
    )
    canary_semantics = {
        arm.arm_id: f"{arm.page_variant}+{arm.close_variant}" for arm in arms
    }
    screen_semantics = {
        arm.arm_id: f"{arm.page_variant}+{arm.close_variant}" for arm in screen_arms
    }
    if canary_semantics != screen_semantics:
        ledger.close()
        missing = sorted(set(screen_semantics) - set(canary_semantics))
        extra = sorted(set(canary_semantics) - set(screen_semantics))
        drifted = sorted(
            arm_id for arm_id in set(canary_semantics) & set(screen_semantics)
            if canary_semantics[arm_id] != screen_semantics[arm_id]
        )
        return _report([_check(
            "screen_arm_coverage",
            False,
            "the canary must execute every screen arm before its cost can be projected: "
            f"missing={missing}, extra={extra}, semantic_drift={drifted}",
        )], execution_binding_sha256=execution_binding_sha256,
            protocol_document_sha256=protocol_document_sha256)
    canary_cells_by_arm = Counter(cell.arm.arm_id for cell in manifest.cells)
    screen_cells_by_arm = Counter(cell.arm.arm_id for cell in screen_manifest.cells)
    manifest.notes["gpu_budget_projection_plan"] = {
        "version": "canary_gpu_projection_v2",
        "projection_method": "per_arm_observed_max",
        "canary_cells": len(manifest.cells),
        "screen_cells": len(screen_manifest.cells),
        "planned_total_cells": len(manifest.cells) + len(screen_manifest.cells),
        "canary_cells_by_arm": dict(sorted(canary_cells_by_arm.items())),
        "screen_cells_by_arm": dict(sorted(screen_cells_by_arm.items())),
        "arm_variant_ids": dict(sorted(screen_semantics.items())),
        "screen_task_count": len(screen_tasks),
        "screen_arm_count": len(screen_arms),
        "screen_task_ids_sha256": sha256_hex(canonical_json(sorted(screen_tasks))),
        "screen_arm_ids_sha256": sha256_hex(canonical_json(screen_semantics)),
        "screen_assignment_sha256": screen_manifest.digest,
    }
    runner.freeze_schedule(
        manifest, settings.path("runs") / "schedules" / config.run_id / "canary.json")
    ledger.create_run(config.run_id, runner.execution_binding_sha256, json.dumps({
        "split": split,
        "schedule_sha256": manifest.digest,
        "execution_binding_sha256": runner.execution_binding_sha256,
        "protocol_document_sha256": runner.protocol_document_sha256,
        "claim_scope": settings.claim_scope,
        "execution_semantics": "GPU_ENGINEERING_CANARY",
    }, sort_keys=True))

    await runner.run_cells(manifest, phase_id="canary", split=split,
                           questions=questions_for(settings, tasks))

    states = runner.cell_states(manifest, phase_id="canary", split=split)
    outputs = {}
    audit_work_keys: list[str] = []
    for cell in manifest.cells:
        key = runner.work_key_for(cell, phase_id="canary", split=split)
        audit_work_keys.append(key)
        ref = ledger.committed_ref(key)
        if ref:
            outputs[cell_key(cell)] = json.loads(store.get_bytes(ref).decode("utf-8"))

    provider_audit = None
    provider_audit_error = ""
    if not audit_work_keys:
        # Do not send a request that violates the closed wire schema's minItems=1.
        provider_audit_error = "canary manifest produced no work keys"
    else:
        try:
            provider_audit = await client.canary_audit(work_keys=audit_work_keys)
        except ProviderCallError as exc:
            # Provider errors are already redacted at the boundary.  Keep the canary result
            # inspectable and fail its audit checks instead of falling back to opening the
            # provider-owned 0700 files from the runner process.
            provider_audit_error = str(exc)
        except Exception as exc:  # noqa: BLE001 - transport failure is a failed attestation
            # Do not copy an arbitrary HTTP exception string into the report: depending on the
            # client it can contain headers. The exception class is enough to diagnose transport.
            provider_audit_error = (
                f"{type(exc).__name__}: provider canary audit transport failed")

    checks = _verify(
        settings,
        manifest,
        states,
        outputs,
        repo=repo,
        provider_audit=provider_audit,
        provider_audit_error=provider_audit_error,
        require_provider_audit=True,
    )
    ok = all(c["status"] == PASS for c in checks)
    if ok:
        runner.phases.begin(Phase.GPU_SMOKE_PASSED)
        runner.phases.complete(Phase.GPU_SMOKE_PASSED, {
            "tasks": tasks, "arms": [a.arm_id for a in arms],
            "checks": [c["name"] for c in checks],
        })
    ledger.close()
    return _report(
        checks,
        tasks=tasks,
        arms=[a.arm_id for a in arms],
        settings=settings,
        execution_binding_sha256=execution_binding_sha256,
        protocol_document_sha256=protocol_document_sha256,
    )


def _verify(
    settings,
    manifest,
    states,
    outputs,
    *,
    repo,
    provider_audit: dict | None = None,
    provider_audit_error: str = "",
    require_provider_audit: bool = False,
) -> list[dict]:
    checks: list[dict] = []
    committed = {k: v for k, v in outputs.items()}
    # The provider ledger is intentionally append-only across campaigns.  Every canary query
    # must therefore be restricted to the exact work items whose immutable output is being
    # verified here.  Otherwise yesterday's selector decode can make today's inert arm pass,
    # and yesterday's open reservation can make a healthy canary fail.
    committed_work_keys = {
        _work_key_of(record) for record in committed.values() if _work_key_of(record)
    }

    # Every cell has to have finished; a canary that half-ran proves nothing about the half
    # that did not.
    incomplete = sorted(k for k, v in states.items() if v != "COMMITTED")
    checks.append(_check("cells_committed", not incomplete,
                         "all cells committed" if not incomplete else f"incomplete: {incomplete}"))

    # 1. The bytes that ran are the patched bytes.
    checks.append(_patched_graph_check(repo))

    p1_cells = {k: v for k, v in committed.items()
                if v["cell"]["arm"]["arm_id"] not in ("P0",)}
    p0_cells = {k: v for k, v in committed.items() if v["cell"]["arm"]["arm_id"] == "P0"}

    # 2. Every P1 arm actually ran -- not "at least one of them did".
    #    reduce_published_batch fires for any bound bundle, including arms whose page half is
    #    the vendor strategy, so a single CPU control satisfied the old total and a
    #    completely inert LLM arm passed alongside it.
    inert = []
    for arm_id, cells in _by_arm(p1_cells).items():
        reduced = sum(v["counts"].get("page_batches_reduced", 0) for v in cells)
        closed = sum(v["counts"].get("close_reduced", 0) for v in cells)
        if (reduced + closed) == 0:
            inert.append(arm_id)
    checks.append(_check("p1_strategy_invocations", bool(p1_cells) and not inert,
                         f"{len(_by_arm(p1_cells))} P1 arm(s) all reduced something"
                         if p1_cells and not inert
                         else f"arms that reduced nothing: {inert or 'no P1 cells at all'}"))

    # 2b. Something was actually PUBLISHED. Reducing a batch is not producing P1 output: a batch
    #     that fails and falls back to P0 still counts as reduced, so the check above passed on
    #     every arm of a run in which not one P1 span was ever published. That is not
    #     hypothetical -- across 19 arms and 146 cells, every H arm raised
    #     `publication handle ... exceeds frozen cap 8` on its first candidate and fell back, and
    #     the only place it was visible was the object store, after the fact.
    #
    #     The P0 fallback is a legitimate part of the ITT design, which is exactly why total
    #     inertness must be a gate: "P1 tried and fell back" and "P1 was never reachable" produce
    #     the same artifacts, and only the second is a defect.
    publication = _p1_publication_by_arm(p1_cells)
    silent = sorted(
        arm_id for arm_id, stats in publication.items() if stats["published_spans"] == 0
    )
    checks.append(_check(
        "p1_published_output",
        bool(publication) and not silent,
        f"{len(publication)} P1 arm(s) published spans; "
        + ", ".join(
            f"{arm_id}={stats['published_spans']}"
            for arm_id, stats in sorted(publication.items())
        )
        if publication and not silent
        else (
            "arms that published nothing: "
            + "; ".join(
                f"{arm_id} (selector_attempted="
                f"{publication[arm_id]['selector_attempted']}, "
                f"fallbacks={publication[arm_id]['page_fallbacks']}, "
                f"top failure={publication[arm_id]['top_failure'] or 'none recorded'})"
                for arm_id in silent
            )
            if silent else "no P1 cells at all"
        ),
        data={"by_arm": publication},
    ))

    # 3. The selector decoded. Live runs consume a provider-signed sanitized aggregate because
    # the runner cannot read the provider's 0700 ledger/object store. The direct-file branch is
    # retained only for old isolated unit fixtures; _run_canary_leased always requires the API.
    audit_view = None
    if require_provider_audit:
        audit_view = _parse_provider_audit(
            provider_audit,
            expected_work_keys=committed_work_keys,
            unavailable_reason=provider_audit_error,
        )
        inference = list(audit_view["inference"])
        ledger_error = "; ".join(audit_view["errors"])
    else:
        try:
            inference = _inference_events(settings, work_keys=committed_work_keys)
            ledger_error = ""
        except LedgerUnreadable as e:
            inference = []
            ledger_error = str(e)
    checks.append(_check(
        "provider_canary_audit",
        not ledger_error,
        (
            "runner received a closed sanitized provider attestation for the exact work set"
            if require_provider_audit and not ledger_error
            else "legacy unit harness used direct provider evidence"
            if not require_provider_audit and not ledger_error
            else ledger_error
        ),
    ))
    checks.append(_check("ledger_readable", not ledger_error,
                         ledger_error or (
                             "the provider attested its ledger through the runner-safe API"
                             if require_provider_audit else "the provider ledger was read")))
    selector_completions = sum(
        e["completion_tokens"] for e in inference if e["op_class"] in SELECTOR_OPS)
    selector_by_work_op = {
        (str(event["work_key"]), str(event["op_class"]))
        for event in inference
        if event["op_class"] in SELECTOR_OPS and int(event["completion_tokens"]) > 0
    }
    missing_selector_paths: list[str] = []
    try:
        expected_by_arm = _expected_selector_ops_by_arm(settings, manifest)
    except (KeyError, ValueError) as exc:
        expected_by_arm = {}
        missing_selector_paths.append(f"variant registry cannot be resolved: {exc}")
    for key, record in committed.items():
        arm_id = str(record["cell"]["arm"]["arm_id"])
        work_key = _work_key_of(record)
        for op_class in expected_by_arm.get(arm_id, ()):
            if (work_key, op_class) not in selector_by_work_op:
                missing_selector_paths.append(f"{key}: no {op_class} decode")
    checks.append(_check(
        "selector_decode",
        selector_completions > 0 and not missing_selector_paths,
        (
            f"{selector_completions} completion tokens; every model-backed arm exercised "
            "each required selector path"
            if selector_completions > 0 and not missing_selector_paths
            else "; ".join(missing_selector_paths[:8])
            or "no selector completion tokens"
        ),
    ))

    # 4. BOTH boundaries produced checkpoints. This was a set intersection, so H-only or
    #    C-only passed -- and a canary whose close boundary never fired is exactly the run
    #    that proves nothing about the close node.
    kinds = {c["kind"] for v in committed.values() for c in v["checkpoints"]}
    wanted = {"HCheckpoint", "CCheckpoint"}
    checks.append(_check("checkpoints_present", wanted <= kinds,
                         f"checkpoint kinds seen: {sorted(kinds)}"
                         + ("" if wanted <= kinds else f"; missing {sorted(wanted - kinds)}")))

    # A direct H score needs the raw occurrence set even when a chunker emitted no candidates.
    # Checking only span IDs lets a broken chunker delete hard facts from its own denominator.
    provenance_errors = []
    for key, record in p1_cells.items():
        page_variant = str(
            ((record.get("cell") or {}).get("arm") or {}).get("page_variant") or "")
        if page_variant == "P0":
            continue
        h_records = [
            item for item in (record.get("direct_node_records") or ())
            if str(item.get("node") or "").upper().startswith("H")
        ]
        if not h_records:
            provenance_errors.append(f"{key}: no H direct record")
            continue
        for item in h_records:
            if "offered_source_occurrence_ids" not in item:
                provenance_errors.append(
                    f"{key}:{item.get('stage', 'single')}: missing source occurrences")
            elif item.get("offered_span_ids") and not item.get(
                "offered_source_occurrence_ids"):
                provenance_errors.append(
                    f"{key}:{item.get('stage', 'single')}: spans have no source occurrence")
    checks.append(_check(
        "direct_denominator_provenance", not provenance_errors,
        "every H stage carries raw source occurrence identity"
        if not provenance_errors else "; ".join(provenance_errors[:8]),
    ))

    citation_lineage_errors = [
        f"{key}: SEARCH_QUERY event {index} lacks source occurrence ids"
        for key, record in committed.items()
        for index, event in enumerate(record.get("events") or ())
        if isinstance(event, dict)
        and event.get("kind") == "SEARCH_QUERY"
        and "source_occurrence_ids" not in event
    ]
    checks.append(_check(
        "citation_source_lineage",
        not citation_lineage_errors,
        "every retrieved result is bound to an arm-local source occurrence"
        if not citation_lineage_errors else "; ".join(citation_lineage_errors[:8]),
    ))

    # 5. Publication was whole-batch.  Deferred calls and reduced turns are different units:
    # a turn with two sibling searches legitimately has deferred=2 and reduced=1.  The only
    # useful proof is a publication event emitted at the single-Command boundary, naming the
    # whole sibling set.  Absence or malformed metadata fails closed.
    atomic, atomic_detail = _atomic_publications_are_whole(committed)
    checks.append(_check("atomic_publication", atomic, atomic_detail))

    # 6. EVERY P1 arm is not P0 with a different label. One differing cell out of six used
    #    to be enough, so five silently-inert arms passed behind the one that worked.
    same_as_p0 = _arms_identical_to_p0(p0_cells, p1_cells)
    differing, compared = _p1_differs_from_p0(p0_cells, p1_cells)
    checks.append(_check("p1_bytes_differ_from_p0",
                         compared > 0 and not same_as_p0,
                         f"{differing}/{compared} P1 cells differ from P0 on the same task"
                         + ("" if not same_as_p0
                            else f"; arms identical to P0: {same_as_p0}")))

    # 7. Neither the answer key nor the steward's audit graph is reachable from here.
    from ..ops.acceptance import tree_isolation

    isolated, detail = tree_isolation(repo, settings, ("evaluator_root", "steward_root"))
    checks.append(_check("truth_invisible_to_treatment", isolated, detail))

    # 8. Replay never missed into a live search.
    misses = [k for k, v in committed.items()
              if "ReplayMiss" in json.dumps(v.get("events", []))]
    checks.append(_check("no_live_search_miss", not misses,
                         "no replay miss" if not misses else f"replay miss in {misses}"))

    # 9. Every reservation is closed or explicitly unknown.
    if require_provider_audit:
        open_calls = int(audit_view["open_attempts"]) if audit_view is not None else 0
        checks.append(_check(
            "reservations_closed",
            not ledger_error and open_calls == 0,
            ledger_error or f"{open_calls} exact-work external attempt(s) still open",
        ))
    else:
        try:
            open_calls = _open_external_calls(settings, work_keys=committed_work_keys)
            checks.append(_check("reservations_closed", open_calls == 0,
                                 f"{open_calls} external call(s) still open"))
        except LedgerUnreadable as e:
            checks.append(_check("reservations_closed", False, str(e)))

    # 10. A strategy failure is normally a *committed* provider call followed by invalid JSON,
    #     contract rejection or preflight failure.  It is not a provider FAILED attempt.  What
    #     must reconcile is the strategy incident's checkpoint/reason/direct trace and the
    #     selector work already spent by that cell.
    fallbacks = sum(v["counts"].get("page_fallbacks", 0) for v in committed.values())
    close_failures = sum(v["counts"].get("close_failed", 0) for v in committed.values())
    accounted, accounting_detail = _fallbacks_are_on_the_ledger(
        committed, inference=inference, expected=fallbacks + close_failures)
    checks.append(_check("fallback_and_failures_accounted", accounted,
                         f"{fallbacks} page fallback(s), {close_failures} close failure(s); "
                         + accounting_detail))

    # 11. The CPU control decoded nothing. If it did, it is not a CPU control.
    cpu_arm_ids = {
        "CPU_LEXICAL", "H_CPU_CONTROL", "C_CPU_CONTROL",
    }
    cpu_cells = [
        value for value in committed.values()
        if value["cell"]["arm"]["arm_id"] in cpu_arm_ids
    ]
    cpu_work_keys = {v["cell"] and _work_key_of(v) for v in cpu_cells}
    cpu_completions = sum(e["completion_tokens"] for e in inference
                          if e["op_class"] in SELECTOR_OPS and e["work_key"] in cpu_work_keys)
    checks.append(_check("cpu_control_zero_decode", cpu_cells and cpu_completions == 0,
                         f"{cpu_completions} selector completion tokens in the CPU control"))

    # 12. SHORT_PROSE was held to the same rendered-token budget as the P1 arms.
    budget = int(settings.get("week1", "measurement", "selected_token_budget"))
    prose_arm_ids = {
        "SHORT_PROSE", "H_PROSE_CONTROL", "C_PROSE_CONTROL",
    }
    prose_cells = [
        value for value in committed.values()
        if value["cell"]["arm"]["arm_id"] in prose_arm_ids
    ]
    prose_keys = {_work_key_of(v) for v in prose_cells}
    prose_calls = [e for e in inference if e["work_key"] in prose_keys]
    prose_max = max(
        (e.get("max_completion_tokens_observed", e["completion_tokens"])
         for e in prose_calls),
        default=0,
    )
    prose_attempts = sum(int(e.get("committed_attempt_count", 1)) for e in prose_calls)
    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    # An arm that issued no request at all had max(..., default=0) <= cap and passed. And
    # the comparison was against the selector cap while the detail line quoted the rendered
    # budget, which is the number the check is named for.
    rendered_counts = [
        int(v["counts"].get("max_rendered_tokens", 0) or 0) for v in prose_cells
    ]
    output_counts = [
        int(v["counts"].get("rendered_output_count", 0) or 0) for v in prose_cells
    ]
    rendered = max(rendered_counts, default=0)
    prose_ok = (
        bool(prose_cells)
        and prose_attempts > 0
        and all(count > 0 for count in output_counts)
        and prose_max <= cap
        and rendered <= budget
    )
    checks.append(_check("short_prose_same_budget", prose_ok,
                         f"{prose_attempts} request(s), max {prose_max} completion tokens "
                         f"against a {cap} cap, {rendered} rendered tokens against a "
                         f"{budget} budget"))

    # 13. The cap that bounded generation is a completion limit, not a schema bound. This
    #     read two config values and multiplied them out to a constant True; it never looked
    #     at a request. It looks at the requests now.
    if require_provider_audit:
        capped, capping_detail = _audit_requests_carry_a_completion_cap(
            audit_view, cap, ledger_error=ledger_error)
    else:
        capped, capping_detail = _requests_carry_a_completion_cap(
            settings, cap, work_keys=committed_work_keys)
    checks.append(_check("generation_cap_is_a_completion_limit", capped, capping_detail))

    # This gate is deliberately blind to answer quality.  It asks only whether the observed
    # isolated treatment work makes the predeclared full design affordable.  It is a planning
    # forecast, not an alternative budget authority: every inference request remains subject
    # to the provider ledger's hard admission cap.
    checks.append(_gpu_budget_projection_check(
        settings,
        manifest=manifest,
        states=states,
        committed=committed,
        live_gpu_budget=(audit_view or {}).get("gpu_budget"),
        live_settled_gpu_seconds_by_work=(
            (audit_view or {}).get("settled_gpu_seconds_by_work")
        ),
        provider_audit_errors=tuple((audit_view or {}).get("errors") or ()),
        require_live_budget=require_provider_audit,
    ))
    return checks


def _parse_provider_audit(
    provider_audit: dict | None,
    *,
    expected_work_keys: set[str],
    unavailable_reason: str = "",
) -> dict:
    """Validate and normalize the runner-safe provider attestation.

    The provider validates its response before sending it; the runner validates it again before
    using it. This makes the response schema a two-sided contract and prevents a future provider
    field from becoming observable treatment metadata merely because one producer added it.
    """

    view = {
        "inference": [],
        "open_attempts": 0,
        "selector_request_max_tokens": [],
        "settled_gpu_seconds_by_work": {},
        "gpu_budget": None,
        "errors": [],
    }
    errors: list[str] = view["errors"]
    if not expected_work_keys:
        errors.append("provider canary audit expected work set is empty")
        return view
    if unavailable_reason:
        errors.append(f"provider canary audit unavailable: {unavailable_reason}")
        return view
    if not isinstance(provider_audit, dict):
        errors.append("provider canary audit is missing")
        return view

    from ..runtime.provider_server import ProviderError, validate_request

    try:
        validate_request("canary.audit.response", provider_audit)
    except ProviderError as exc:
        errors.append(f"provider canary audit response violates its closed schema: {exc}")
        return view

    attested_keys = list(map(str, provider_audit["work_keys"]))
    expected = sorted(expected_work_keys)
    if attested_keys != expected:
        errors.append(
            "provider canary audit work set differs from committed canary work: "
            f"missing={sorted(set(expected) - set(attested_keys))}, "
            f"extra={sorted(set(attested_keys) - set(expected))}"
        )

    rows = provider_audit["work"]
    by_key: dict[str, dict] = {}
    for row in rows:
        work_key = str(row["work_key"])
        if work_key in by_key:
            errors.append(f"provider canary audit repeats work_key {work_key!r}")
        by_key[work_key] = row
    if set(by_key) != set(attested_keys):
        errors.append("provider canary audit work rows do not match its work_keys")

    open_total = 0
    selector_caps: list[int | None] = []
    settled_by_work: dict[str, float] = {}
    inference: list[dict] = []
    for work_key in expected:
        row = by_key.get(work_key)
        if row is None:
            continue
        open_total += int(row["open_attempts"])
        seen_ops: set[str] = set()
        op_settled_total = 0.0
        for op in row["ops"]:
            op_class = str(op["op_class"])
            if op_class in seen_ops:
                errors.append(
                    f"provider canary audit repeats {work_key!r}/{op_class!r}")
            seen_ops.add(op_class)
            attempts = int(op["attempt_count"])
            committed_attempts = int(op["committed_attempt_count"])
            if committed_attempts > attempts:
                errors.append(
                    f"provider canary audit has committed_attempt_count > attempt_count "
                    f"for {work_key!r}/{op_class!r}")
            op_settled = float(op["settled_gpu_seconds"])
            if not math.isfinite(op_settled) or op_settled < 0:
                errors.append(
                    f"provider canary audit has invalid settled_gpu_seconds "
                    f"for {work_key!r}/{op_class!r}")
            else:
                op_settled_total += op_settled
            inference.append({
                "work_key": work_key,
                "op_class": op_class,
                "prompt_tokens": int(op["prompt_tokens"]),
                "completion_tokens": int(op["completion_tokens"]),
                "max_completion_tokens_observed":
                    int(op["max_completion_tokens_observed"]),
                "committed_attempt_count": committed_attempts,
                "settled_gpu_seconds": op_settled,
            })
            if op_class in SELECTOR_OPS:
                selector_caps.extend(op["selector_request_max_tokens"])
        work_settled = float(row["settled_gpu_seconds"])
        if not math.isfinite(work_settled) or work_settled < 0:
            errors.append(
                f"provider canary audit has invalid work settlement for {work_key!r}")
        elif not math.isclose(
            work_settled,
            op_settled_total,
            rel_tol=GPU_SETTLEMENT_REL_TOLERANCE,
            abs_tol=GPU_SETTLEMENT_ABS_TOLERANCE_SECONDS,
        ):
            errors.append(
                f"provider canary audit work/op settlement mismatch for {work_key!r}")
        settled_by_work[work_key] = work_settled

    attested_open = int(provider_audit["open_attempts"])
    if open_total != attested_open:
        errors.append(
            f"provider canary audit open-attempt total {attested_open} "
            f"does not reconcile with per-work total {open_total}")
    view.update({
        "inference": inference,
        "open_attempts": attested_open,
        "selector_request_max_tokens": selector_caps,
        "settled_gpu_seconds_by_work": settled_by_work,
        "gpu_budget": dict(provider_audit["gpu_budget"]),
    })
    return view


def _audit_requests_carry_a_completion_cap(
    audit_view: dict | None,
    cap: int,
    *,
    ledger_error: str = "",
) -> tuple[bool, str]:
    if ledger_error or audit_view is None:
        return False, ledger_error or "provider canary audit is unavailable"
    limits = list(audit_view["selector_request_max_tokens"])
    if not limits:
        return False, "provider attested no selector request completion limits"
    invalid = [
        value for value in limits
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value <= 0
            or value > cap
        )
    ]
    if invalid:
        return (
            False,
            f"{len(invalid)}/{len(limits)} selector request(s) carried no usable max_tokens",
        )
    return True, f"{len(limits)} selector request(s) all carried max_tokens <= {cap}"


def _gpu_budget_projection_check(
    settings: Settings,
    *,
    manifest,
    states: dict,
    committed: dict,
    live_gpu_budget: dict | None = None,
    live_settled_gpu_seconds_by_work: dict[str, float] | None = None,
    provider_audit_errors: tuple[str, ...] = (),
    require_live_budget: bool = False,
) -> dict:
    """Project full-design GPU work from every committed canary cell, fail closed.

    ``service_seconds`` is the sum of isolated treatment intervals for the cell.  In the causal
    layer those intervals do not overlap, and their sum is comparable to the provider ledger's
    settled ``gpu_seconds`` resource.  The projection is intentionally a simple pre-treatment
    engineering forecast: it reads no report, score, truth packet, or quality outcome.
    """

    plan = (getattr(manifest, "notes", {}) or {}).get("gpu_budget_projection_plan")
    expected_keys = {cell_key(cell) for cell in manifest.cells}
    errors: list[str] = list(provider_audit_errors)
    values: list[float] = []

    # Causal means no cross-arm prefix-cache carry-over. It does not mean serialized: the
    # projection needs comparable per-cell work, which the interval union provides in either
    # regime.
    if not settings.layer_is_causal:
        errors.append(
            f"measurement layer {settings.measurement_layer!r} has prefix caching on; "
            "one arm's prefill could subsidise another's")

    if not isinstance(plan, dict):
        errors.append("frozen manifest has no gpu_budget_projection_plan")
        plan = {}

    def positive_int(name: str, *, allow_zero: bool = False) -> int | None:
        value = plan.get(name)
        minimum = 0 if allow_zero else 1
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            errors.append(f"projection plan has invalid {name}={value!r}")
            return None
        return value

    canary_cells = positive_int("canary_cells")
    screen_cells = positive_int("screen_cells")
    planned_total_cells = positive_int("planned_total_cells")
    screen_task_count = positive_int("screen_task_count")
    screen_arm_count = positive_int("screen_arm_count")
    if plan.get("version") != "canary_gpu_projection_v2":
        errors.append(f"unknown projection plan version {plan.get('version')!r}")
    if plan.get("projection_method") != "per_arm_observed_max":
        errors.append(
            f"unknown projection method {plan.get('projection_method')!r}")
    for name in (
        "screen_task_ids_sha256",
        "screen_arm_ids_sha256",
        "screen_assignment_sha256",
    ):
        value = plan.get(name)
        if not isinstance(value, str) or len(value) != 64:
            errors.append(f"projection plan has invalid {name}")
    if canary_cells is not None and canary_cells != len(expected_keys):
        errors.append(
            f"projection plan names {canary_cells} canary cells, manifest has "
            f"{len(expected_keys)}")
    if (
        canary_cells is not None
        and screen_cells is not None
        and planned_total_cells is not None
        and planned_total_cells != canary_cells + screen_cells
    ):
        errors.append(
            "planned_total_cells does not equal frozen canary_cells + screen_cells")

    def count_map(name: str) -> dict[str, int]:
        raw = plan.get(name)
        if not isinstance(raw, dict) or not raw:
            errors.append(f"projection plan has invalid {name}")
            return {}
        result: dict[str, int] = {}
        for arm_id, count in raw.items():
            if (
                not str(arm_id)
                or not isinstance(count, int)
                or isinstance(count, bool)
                or count <= 0
            ):
                errors.append(f"projection plan has invalid {name}[{arm_id!r}]")
                continue
            result[str(arm_id)] = count
        return result

    canary_counts_by_arm = count_map("canary_cells_by_arm")
    screen_counts_by_arm = count_map("screen_cells_by_arm")
    arm_variant_ids = plan.get("arm_variant_ids")
    if (
        not isinstance(arm_variant_ids, dict)
        or set(map(str, arm_variant_ids)) != set(screen_counts_by_arm)
        or any(not str(value) for value in (arm_variant_ids or {}).values())
    ):
        errors.append("projection plan has invalid arm_variant_ids")
    if set(canary_counts_by_arm) != set(screen_counts_by_arm):
        errors.append(
            "canary does not cover the exact screen arm set: "
            f"missing={sorted(set(screen_counts_by_arm) - set(canary_counts_by_arm))}, "
            f"extra={sorted(set(canary_counts_by_arm) - set(screen_counts_by_arm))}"
        )
    if canary_cells is not None and sum(canary_counts_by_arm.values()) != canary_cells:
        errors.append("canary_cells_by_arm does not sum to canary_cells")
    if screen_cells is not None and sum(screen_counts_by_arm.values()) != screen_cells:
        errors.append("screen_cells_by_arm does not sum to screen_cells")
    if screen_task_count is not None and screen_arm_count is not None and screen_cells is not None:
        # With second replicates the screen may have *more* than tasks x arms, never fewer.
        minimum_screen_cells = screen_task_count * screen_arm_count
        if screen_cells < minimum_screen_cells:
            errors.append(
                f"screen_cells={screen_cells} is below task x arm minimum "
                f"{minimum_screen_cells}")

    unexpected = sorted(set(committed) - expected_keys)
    if unexpected:
        errors.append(f"outputs contain non-manifest cells: {unexpected[:4]}")
    values_by_arm: dict[str, list[float]] = {}
    observed_service_by_work: dict[str, float] = {}
    observed_counts_by_arm: Counter[str] = Counter()
    for key in sorted(expected_keys):
        if states.get(key) != "COMMITTED":
            errors.append(f"{key}: state is {states.get(key, 'MISSING')!r}")
            continue
        record = committed.get(key)
        if not isinstance(record, dict):
            errors.append(f"{key}: committed output is missing")
            continue
        summary = record.get("work_summary")
        if not isinstance(summary, dict):
            errors.append(f"{key}: work_summary is missing")
            continue
        if summary.get("telemetry_complete") is not True:
            errors.append(f"{key}: telemetry_complete is not true")
        if summary.get("overlap_valid") is not True:
            errors.append(f"{key}: overlap_valid is not true")
        raw = summary.get("service_seconds")
        if isinstance(raw, bool):
            errors.append(f"{key}: service_seconds is not numeric")
            continue
        try:
            value = float(raw)
        except (TypeError, ValueError):
            errors.append(f"{key}: service_seconds is not numeric")
            continue
        if not math.isfinite(value) or value <= 0:
            errors.append(f"{key}: service_seconds must be finite and positive")
            continue
        values.append(value)
        work_key = _work_key_of(record)
        if not work_key:
            errors.append(f"{key}: frozen output has no work_key")
        elif work_key in observed_service_by_work:
            errors.append(f"{key}: duplicate work_key {work_key!r}")
        else:
            observed_service_by_work[work_key] = value
        arm_id = str(((record.get("cell") or {}).get("arm") or {}).get("arm_id") or "")
        if not arm_id:
            errors.append(f"{key}: frozen output has no arm_id")
            continue
        observed_counts_by_arm[arm_id] += 1
        values_by_arm.setdefault(arm_id, []).append(value)
    if dict(sorted(observed_counts_by_arm.items())) != canary_counts_by_arm:
        errors.append(
            "observed canary cells do not reconcile with canary_cells_by_arm")

    if require_live_budget:
        if not isinstance(live_settled_gpu_seconds_by_work, dict):
            errors.append("provider canary audit has no exact-work gpu settlement totals")
        else:
            attested_keys = set(live_settled_gpu_seconds_by_work)
            observed_keys = set(observed_service_by_work)
            if attested_keys != observed_keys:
                errors.append(
                    "provider gpu settlements do not match output work keys: "
                    f"missing={sorted(observed_keys - attested_keys)}, "
                    f"extra={sorted(attested_keys - observed_keys)}"
                )
            for work_key in sorted(observed_keys & attested_keys):
                try:
                    settled_for_work = float(
                        live_settled_gpu_seconds_by_work[work_key])
                except (TypeError, ValueError):
                    errors.append(
                        f"{work_key}: provider settled_gpu_seconds is not numeric")
                    continue
                observed_for_work = observed_service_by_work[work_key]
                if (
                    not math.isfinite(settled_for_work)
                    or settled_for_work < 0
                    or not math.isclose(
                        settled_for_work,
                        observed_for_work,
                        rel_tol=GPU_SETTLEMENT_REL_TOLERANCE,
                        abs_tol=GPU_SETTLEMENT_ABS_TOLERANCE_SECONDS,
                    )
                ):
                    errors.append(
                        f"{work_key}: provider settled_gpu_seconds "
                        f"{settled_for_work!r} does not match output service_seconds "
                        f"{observed_for_work!r} within abs_tol="
                        f"{GPU_SETTLEMENT_ABS_TOLERANCE_SECONDS}s, rel_tol="
                        f"{GPU_SETTLEMENT_REL_TOLERANCE}"
                    )

    configured_hard_cap = float(settings.budget_caps()["gpu_seconds"])
    hard_cap = configured_hard_cap
    ledger_reserved = 0.0
    ledger_settled = 0.0
    ledger_remaining = configured_hard_cap
    if require_live_budget:
        if not isinstance(live_gpu_budget, dict):
            errors.append("provider canary audit has no gpu_seconds budget account")
        else:
            try:
                resource = str(live_gpu_budget["resource"])
                hard_cap = float(live_gpu_budget["cap"])
                ledger_reserved = float(live_gpu_budget["reserved"])
                ledger_settled = float(live_gpu_budget["settled"])
                ledger_remaining = float(live_gpu_budget["remaining"])
            except (KeyError, TypeError, ValueError):
                errors.append("provider canary audit has an invalid gpu_seconds budget account")
            else:
                budget_values = (
                    hard_cap, ledger_reserved, ledger_settled, ledger_remaining)
                if resource != "gpu_seconds":
                    errors.append(
                        f"provider attested budget resource {resource!r}, expected gpu_seconds")
                if (
                    any(not math.isfinite(value) for value in budget_values)
                    or hard_cap < 0
                    or ledger_reserved < 0
                    or ledger_settled < 0
                    or ledger_remaining < 0
                ):
                    errors.append("provider gpu_seconds budget contains invalid totals")
                if not math.isclose(
                    hard_cap, configured_hard_cap, rel_tol=0.0, abs_tol=1e-6
                ):
                    errors.append(
                        f"provider gpu_seconds cap drift: live={hard_cap}, "
                        f"approved={configured_hard_cap}")
                expected_remaining = hard_cap - ledger_reserved - ledger_settled
                if not math.isclose(
                    ledger_remaining, expected_remaining, rel_tol=0.0, abs_tol=1e-6
                ):
                    errors.append(
                        "provider gpu_seconds remaining does not equal cap-reserved-settled")
                exact_work_settled = (
                    sum(live_settled_gpu_seconds_by_work.values())
                    if isinstance(live_settled_gpu_seconds_by_work, dict)
                    and live_settled_gpu_seconds_by_work else None
                )
                if exact_work_settled is not None and (
                    not math.isfinite(exact_work_settled)
                    or ledger_settled + GPU_SETTLEMENT_ABS_TOLERANCE_SECONDS
                    < exact_work_settled
                ):
                    errors.append(
                        f"provider global settled gpu_seconds {ledger_settled} is below "
                        f"the exact-work settled total {exact_work_settled}")

    actual = (
        sum(values)
        if not errors and len(values) == len(expected_keys) and expected_keys else None
    )
    mean = actual / len(expected_keys) if actual is not None else None
    max_by_arm = (
        {
            arm_id: max(values_by_arm[arm_id])
            for arm_id in sorted(screen_counts_by_arm)
        }
        if (
            actual is not None
            and set(values_by_arm) == set(screen_counts_by_arm)
        )
        else None
    )
    projected_screen = (
        sum(
            max_by_arm[arm_id] * screen_counts_by_arm[arm_id]
            for arm_id in screen_counts_by_arm
        )
        if max_by_arm is not None else None
    )
    if require_live_budget:
        # Canary and any historical work are already present in settled/reserved. Only the
        # prospective screen forecast consumes the current live headroom.
        projected = (
            ledger_settled + ledger_reserved + projected_screen
            if actual is not None and projected_screen is not None else None
        )
        margin = (
            ledger_remaining - projected_screen
            if projected_screen is not None else None
        )
        feasible = (
            not errors
            and projected is not None
            and projected_screen is not None
            and projected_screen <= ledger_remaining
        )
    else:
        # Compatibility for isolated unit fixtures without a provider process.
        projected = (
            actual + projected_screen
            if actual is not None and projected_screen is not None else None
        )
        margin = hard_cap - projected if projected is not None else None
        feasible = not errors and projected is not None and projected <= hard_cap

    if errors:
        detail = (
            "; ".join(errors[:8])
            + ". Projection unavailable; the provider budget ledger remains the hard authority."
        )
    else:
        detail = (
            f"canary actual={actual:.3f}s across {len(expected_keys)} cells; "
            f"mean={mean:.3f}s/cell; planned={planned_total_cells} cells; "
            f"screen forecast={projected_screen:.3f}s using each arm's observed maximum; "
            f"ledger settled={ledger_settled:.3f}s, reserved={ledger_reserved:.3f}s, "
            f"remaining={ledger_remaining:.3f}s; projected={projected:.3f}s; "
            f"hard cap={hard_cap:.3f}s; "
            f"margin={margin:.3f}s. Cost-only forecast; provider ledger remains authoritative."
        )

    check = _check("projected_gpu_budget_feasible", feasible, detail)
    check.update({
        "canary_actual_gpu_seconds": actual,
        "canary_mean_gpu_seconds_per_cell": mean,
        "canary_max_gpu_seconds_by_arm": max_by_arm,
        "planned_total_cells": planned_total_cells,
        "projected_screen_gpu_seconds": projected_screen,
        "projected_total_gpu_seconds": projected,
        "hard_cap_gpu_seconds": hard_cap,
        "configured_hard_cap_gpu_seconds": configured_hard_cap,
        "ledger_reserved_gpu_seconds": ledger_reserved,
        "ledger_settled_gpu_seconds": ledger_settled,
        "ledger_remaining_gpu_seconds": ledger_remaining,
        "gpu_settlement_abs_tolerance_seconds":
            GPU_SETTLEMENT_ABS_TOLERANCE_SECONDS,
        "gpu_settlement_rel_tolerance": GPU_SETTLEMENT_REL_TOLERANCE,
        "projected_margin_gpu_seconds": margin,
        "measurement": "summed_isolated_treatment_service_intervals",
        "projection_method": "per_arm_observed_max",
        "gate_scope": "cost_only_no_quality",
        "hard_budget_authority": "provider_budget_ledger",
    })
    return check


def _by_arm(cells: dict) -> dict:
    grouped: dict = {}
    for record in cells.values():
        grouped.setdefault(record["cell"]["arm"]["arm_id"], []).append(record)
    return grouped


def _expected_selector_ops_by_arm(settings: Settings, manifest) -> dict[str, tuple[str, ...]]:
    """Resolve the selector paths each frozen arm must exercise from variant semantics."""
    from ..strategies.factory import load_registry

    registry = load_registry(settings.repo / "configs")
    expected: dict[str, tuple[str, ...]] = {}
    for arm in manifest.arms:
        ops: list[str] = []
        page = registry.get(str(arm.page_variant))
        close = registry.get(str(arm.close_variant))
        if page is None or close is None:
            raise ValueError(
                f"{arm.arm_id} references unknown variants "
                f"{arm.page_variant!r}+{arm.close_variant!r}")
        # A SHORT_PROSE control is LLM-backed but does not run the structured selector, and it
        # carries its own op class. Keying only on selector_backend would demand the selector op
        # from an arm that never emits one -- and used to pass only because the prose aliases
        # were mis-mapped onto the selector classes.
        if page.node == "WEBPAGE_P1" and page.selector_backend == "LLM":
            if page.contract == "SHORT_PROSE":
                ops.append(OpClass.PAGE_P1_SHORT_PROSE.value)
            else:
                ops.append(OpClass.PAGE_P1_SELECTOR_LOCAL.value)
                if page.scope == "hierarchical":
                    ops.append(OpClass.PAGE_P1_SELECTOR_GLOBAL.value)
        if close.node in {"C_VISIBLE", "C_REGISTRY", "C_FUSED_EXT"} \
                and close.selector_backend == "LLM":
            ops.append(
                OpClass.COMPRESSOR_SHORT_PROSE.value if close.contract == "SHORT_PROSE"
                else OpClass.COMPRESSOR_P1_SELECTOR.value
            )
        expected[str(arm.arm_id)] = tuple(ops)
    return expected


def _p1_publication_by_arm(p1_cells: dict) -> dict[str, dict]:
    """Per-arm evidence that P1 output actually reached the graph, plus why it did not.

    Read from the direct-node records rather than the cell counters, because the counters
    describe attempts: ``page_batches_reduced`` is incremented for a batch that failed and fell
    back, so an arm can look invoked, look reduced, and have published nothing at all. The
    diagnostics travel with the verdict so a failure is legible from the canary summary instead
    of requiring the object store to be read afterwards.
    """
    summary: dict[str, dict] = {}
    for arm_id, records in _by_arm(p1_cells).items():
        stats = summary.setdefault(arm_id, {
            "cells": 0,
            "published_spans": 0,
            "selector_attempted": 0,
            "direct_node_records": 0,
            "page_fallbacks": 0,
            "failures": {},
            "top_failure": "",
        })
        for record in records:
            stats["cells"] += 1
            stats["page_fallbacks"] += int(record.get("counts", {}).get("page_fallbacks", 0))
            for node_record in record.get("direct_node_records") or ():
                stats["direct_node_records"] += 1
                stats["published_spans"] += len(node_record.get("published_span_ids") or ())
                if node_record.get("selector_attempted"):
                    stats["selector_attempted"] += 1
                failure = node_record.get("failure")
                if failure:
                    # Group by reason and keep one example: 2,067 occurrences of the same
                    # publication-handle overflow is one defect, not 2,067 of them.
                    if isinstance(failure, dict):
                        reason = str(failure.get("reason") or "")
                        detail = str(failure.get("detail") or "")
                    else:
                        reason, _, detail = str(failure).partition(":")
                    key = f"{reason.strip()}: {detail.strip()[:120]}"
                    stats["failures"][key] = stats["failures"].get(key, 0) + 1
        if stats["failures"]:
            reason, count = max(stats["failures"].items(), key=lambda item: item[1])
            stats["top_failure"] = f"{reason} (x{count})"
    return summary


def _arms_identical_to_p0(p0_cells: dict, p1_cells: dict) -> list:
    """P1 arms whose every compared cell is byte-identical to P0 on the same task."""
    p0_by_task = {v["cell"]["task_id"]: v.get("final_report", "")
                  for v in p0_cells.values()}
    identical = []
    for arm_id, records in sorted(_by_arm(p1_cells).items()):
        compared = 0
        same = 0
        for record in records:
            baseline = p0_by_task.get(record["cell"]["task_id"])
            if baseline is None:
                continue
            compared += 1
            if record.get("final_report", "") == baseline:
                same += 1
        if compared and same == compared:
            identical.append(arm_id)
    return identical


def _atomic_publications_are_whole(committed: dict) -> tuple[bool, str]:
    """Reconcile reduced H turns with the one event emitted at atomic publication.

    ``PAGE_BATCH_DEFERRED`` counts tool calls while ``PAGE_BATCH_REDUCED`` counts assistant
    turns; comparing those counts rejects the normal multi-sibling case.  A
    ``TOOL_BATCH_PUBLISHED`` record instead attests to the single graph update and names every
    sibling in it.  This consumer deliberately fails closed until the producer supplies that
    event -- a count heuristic is not an acceptable substitute.
    """

    problems: list[str] = []
    published_batches = 0
    for cell_key_, record in committed.items():
        events = list(record.get("events") or ())
        reductions = [
            (index, event) for index, event in enumerate(events)
            if event.get("kind") == "PAGE_BATCH_REDUCED"
        ]
        publications = [
            (index, event) for index, event in enumerate(events)
            if event.get("kind") == "TOOL_BATCH_PUBLISHED"
        ]
        published_batches += len(publications)

        reduced_by_checkpoint: dict[str, list[tuple[int, dict]]] = {}
        for index, event in reductions:
            checkpoint = str(event.get("checkpoint") or "")
            if not checkpoint:
                problems.append(f"{cell_key_}: reduction missing checkpoint")
                continue
            reduced_by_checkpoint.setdefault(checkpoint, []).append((index, event))

        published_by_checkpoint: dict[str, list[tuple[int, dict]]] = {}
        for index, event in publications:
            checkpoint = str(event.get("checkpoint") or "")
            if not checkpoint:
                problems.append(f"{cell_key_}: publication missing checkpoint")
                continue
            published_by_checkpoint.setdefault(checkpoint, []).append((index, event))
            sibling_count = event.get("sibling_count")
            tool_ids = event.get("tool_call_ids")
            if event.get("atomic_publish") is not True:
                problems.append(f"{cell_key_}:{checkpoint}: atomic_publish is not true")
            if (
                not isinstance(sibling_count, int)
                or isinstance(sibling_count, bool)
                or sibling_count <= 0
            ):
                problems.append(f"{cell_key_}:{checkpoint}: invalid sibling_count")
            if (
                not isinstance(tool_ids, list)
                or not tool_ids
                or any(not isinstance(value, str) or not value for value in tool_ids)
                or len(set(tool_ids)) != len(tool_ids)
            ):
                problems.append(f"{cell_key_}:{checkpoint}: invalid tool_call_ids")
            elif isinstance(sibling_count, int) and len(tool_ids) != sibling_count:
                problems.append(
                    f"{cell_key_}:{checkpoint}: {len(tool_ids)} ids != "
                    f"sibling_count {sibling_count}"
                )

        checkpoints = set(reduced_by_checkpoint) | set(published_by_checkpoint)
        for checkpoint in checkpoints:
            reduced = reduced_by_checkpoint.get(checkpoint, [])
            published = published_by_checkpoint.get(checkpoint, [])
            if len(reduced) != 1 or len(published) != 1:
                problems.append(
                    f"{cell_key_}:{checkpoint}: reductions={len(reduced)}, "
                    f"publications={len(published)}"
                )
                continue
            reduced_index, reduced_event = reduced[0]
            published_index, published_event = published[0]
            if published_index <= reduced_index:
                problems.append(f"{cell_key_}:{checkpoint}: publication precedes reduction")
            reduced_siblings = reduced_event.get("siblings")
            if (
                not isinstance(reduced_siblings, int)
                or isinstance(reduced_siblings, bool)
                or reduced_siblings <= 0
                or reduced_siblings != published_event.get("sibling_count")
            ):
                problems.append(
                    f"{cell_key_}:{checkpoint}: reduced siblings do not match publication"
                )

        reported = int((record.get("counts") or {}).get("page_batches_reduced", 0) or 0)
        if reported != len(reductions):
            problems.append(
                f"{cell_key_}: counts report {reported} reductions, events contain "
                f"{len(reductions)}"
            )
    if problems:
        return False, "; ".join(problems[:8])
    return True, f"{published_batches} reduced batch(es) each had one atomic publication"


def _fallbacks_are_on_the_ledger(
    committed: dict,
    *,
    inference: list[dict],
    expected: int,
) -> tuple[bool, str]:
    """Reconcile strategy incidents with direct provenance and already-spent selector work.

    Invalid JSON, contract rejection and preflight rejection happen *after* a successful model
    response, so their provider attempt should normally be COMMITTED.  Requiring a provider
    FAILED row reverses the state machine and loses the selector tokens.  We instead require
    the strategy boundary record, its direct-node failure trace, and both durable work-summary
    and request-ledger evidence for the selector call (except the CPU-only control).
    """

    problems: list[str] = []
    observed = 0
    selector_events = {
        (str(event.get("work_key") or ""), str(event.get("op_class") or ""))
        for event in inference
        if event.get("op_class") in SELECTOR_OPS
    }
    for cell_key_, record in committed.items():
        events = list(record.get("events") or ())
        direct = list(record.get("direct_node_records") or ())
        counts = record.get("counts") or {}
        arm_id = str((record.get("cell") or {}).get("arm", {}).get("arm_id") or "")
        work_key = _work_key_of(record)
        page_incidents = [
            ("H", event) for event in events
            if event.get("kind") == "PAGE_BATCH_REDUCED"
            and (event.get("fell_back") or event.get("failure"))
        ]
        close_incidents = [
            ("C", event) for event in events if event.get("kind") == "CLOSE_FAILED"
        ]
        incidents = page_incidents + close_incidents
        observed += len(incidents)
        if int(counts.get("page_fallbacks", 0) or 0) != len(
            [event for _, event in page_incidents if event.get("fell_back")]
        ):
            problems.append(f"{cell_key_}: page fallback count/event mismatch")
        if int(counts.get("close_failed", 0) or 0) != len(close_incidents):
            problems.append(f"{cell_key_}: close failure count/event mismatch")
        if not incidents:
            continue

        work_summary = record.get("work_summary")
        if not isinstance(work_summary, dict) or work_summary.get("telemetry_complete") is not True:
            problems.append(f"{cell_key_}: incident cell has incomplete work telemetry")
            work_summary = {}
        by_op = work_summary.get("by_op") if isinstance(work_summary, dict) else {}
        if not isinstance(by_op, dict):
            by_op = {}

        for node, event in incidents:
            checkpoint = str(event.get("checkpoint") or "")
            reason = str(event.get("failure") or event.get("reason") or "")
            if not checkpoint:
                problems.append(f"{cell_key_}:{node}: incident missing checkpoint")
            if not reason:
                problems.append(f"{cell_key_}:{node}:{checkpoint}: incident missing reason")

            matching = [
                item for item in direct
                if str(item.get("node") or "") == node
                and str(item.get("checkpoint_hash") or item.get("checkpoint") or "")
                == checkpoint
                and bool(item.get("failure"))
            ]
            if not matching:
                problems.append(
                    f"{cell_key_}:{node}:{checkpoint}: no direct-node failure record"
                )

            # CPU_LEXICAL is intentionally selector-free.  Every model-backed incident must
            # still show the selector in both the sanitized work summary and immutable request
            # ledger for this exact work key.
            if arm_id in {"CPU_LEXICAL", "H_CPU_CONTROL", "C_CPU_CONTROL"}:
                continue
            # SHORT_PROSE controls are model-backed too; they just emit their own op class.
            # Listing only the structured ops would fail an arm that did exactly what it should.
            allowed = (
                {OpClass.PAGE_P1_SELECTOR_LOCAL.value,
                 OpClass.PAGE_P1_SELECTOR_GLOBAL.value,
                 OpClass.PAGE_P1_SHORT_PROSE.value}
                if node == "H"
                else {OpClass.COMPRESSOR_P1_SELECTOR.value,
                      OpClass.COMPRESSOR_SHORT_PROSE.value}
            )
            if not work_key:
                problems.append(f"{cell_key_}:{node}:{checkpoint}: missing work_key")
                continue
            if not any(
                isinstance(by_op.get(op), dict)
                and int(by_op[op].get("count", 0) or 0) > 0
                for op in allowed
            ):
                problems.append(
                    f"{cell_key_}:{node}:{checkpoint}: selector absent from work_summary"
                )
            if not any((work_key, op) in selector_events for op in allowed):
                problems.append(
                    f"{cell_key_}:{node}:{checkpoint}: selector absent from provider ledger"
                )

    if observed != expected:
        problems.append(f"counts report {expected} incident(s), events contain {observed}")
    if problems:
        return False, "; ".join(problems[:8])
    return True, f"{observed} strategy incident(s) reconciled to direct trace and selector work"


def _requests_carry_a_completion_cap(
    settings: Settings,
    cap: int,
    *,
    work_keys: set[str] | None = None,
) -> tuple:
    """Read the requests that were actually sent, not the config that describes them."""
    import sqlite3

    keys = sorted(str(key) for key in (work_keys or ()) if key)
    if not keys:
        return False, "no committed canary work keys were available to inspect"
    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        return False, f"no provider ledger at {path}; no request could be inspected"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        # Placeholders are generated from the set size, not written out: a literal (?,?,?)
        # meant that adding the SHORT_PROSE op classes became a runtime binding error. Only
        # "?" characters are interpolated -- every value is still bound -- so S608's injection
        # concern does not apply here.
        query = (
            "SELECT c.work_key, a.request_object_ref FROM external_call_attempts a"  # noqa: S608
            " JOIN external_calls c ON c.call_id = a.call_id"
            " WHERE c.provider='vllm' AND a.request_object_ref IS NOT NULL"
            f" AND c.op_class IN ({_PLACEHOLDERS})"
        )
        rows = conn.execute(query, sorted(SELECTOR_OPS)).fetchall()
        conn.close()
        rows = [row for row in rows if str(row["work_key"] or "") in set(keys)]
    except sqlite3.Error as e:
        return False, f"the ledger could not be read: {e}"
    if not rows:
        return False, "no selector request was recorded for the committed canary cells"

    from ..object_store import ObjectStore

    store = ObjectStore(settings.data_root
                        / str(settings.get("week1", "paths", "provider_root")) / "objects")
    uncapped = 0
    unreadable = 0
    seen = 0
    for row in rows:
        try:
            body = json.loads(store.get_bytes(row["request_object_ref"]).decode("utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable request is not an observed cap
            unreadable += 1
            continue
        seen += 1
        limit = body.get("max_tokens")
        if (
            not isinstance(limit, int)
            or isinstance(limit, bool)
            or limit <= 0
            or limit > cap
        ):
            uncapped += 1
    if unreadable:
        return False, f"{unreadable}/{len(rows)} selector request body/bodies could not be read"
    if not seen:
        return False, "no inference request body could be read back"
    if uncapped:
        return False, f"{uncapped}/{seen} selector request(s) carried no usable max_tokens"
    return True, f"{seen} selector request(s) all carried max_tokens <= {cap}"


class LedgerUnreadable(RuntimeError):
    """The canary could not read the ledger it verifies against.

    Its own absence used to be reported as zero open reservations and zero selector
    decode -- the two ledger-derived checks then passed on a host with no ledger at all.
    """


def _work_key_of(record: dict) -> str:
    return record.get("work_key", "") or record.get("cell", {}).get("work_key", "")


def _patched_graph_check(repo: Path) -> dict:
    """The installed ODR must hash to the patched tree. A path check cannot answer this: ODR is
    a namespace package and its ``__file__`` is None."""
    from ..treehash import tree_sha256

    try:
        import open_deep_research

        installed = Path(open_deep_research.__path__[0])
        patched = repo / ".build" / "open_deep_research-patched" / "src" / "open_deep_research"
        live = tree_sha256(installed)
        want = tree_sha256(patched)
        return _check("patched_graph_invoked", live == want,
                      f"installed tree {live[:12]} vs patched {want[:12]}")
    except Exception as e:  # noqa: BLE001
        return _check("patched_graph_invoked", False, f"{type(e).__name__}: {e}")


def _p1_differs_from_p0(p0_cells: dict, p1_cells: dict) -> tuple[int, int]:
    """Compare each P1 cell's report bytes against P0's on the same task."""
    p0_by_task = {v["cell"]["task_id"]: sha256_hex(canonical_json(v["final_report"]))
                  for v in p0_cells.values()}
    differing = compared = 0
    for record in p1_cells.values():
        task = record["cell"]["task_id"]
        if task not in p0_by_task:
            continue
        compared += 1
        if sha256_hex(canonical_json(record["final_report"])) != p0_by_task[task]:
            differing += 1
    return differing, compared


def _inference_events(
    settings: Settings,
    *,
    work_keys: set[str] | None = None,
) -> list[dict]:
    """Per-request work, read from the provider's own ledger, read-only.

    The runner's ledger records cells; the provider's records calls. Reading the runner's would
    return nothing and every token-based check would pass vacuously -- which is exactly the
    shape of failure the canary exists to catch.
    """
    import sqlite3

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        # Not "no calls". A canary that cannot read the ledger has not verified anything
        # about what was dispatched, and returning [] made every ledger-derived check read
        # the absence of evidence as evidence of absence.
        raise LedgerUnreadable(f"no provider ledger at {path}")
    keys = sorted(str(key) for key in (work_keys or ()) if key)
    if work_keys is not None and not keys:
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        sql = (
            "SELECT c.op_class, c.work_key, a.usage_json FROM external_calls c"
            " JOIN external_call_attempts a ON a.call_id = c.call_id"
            " WHERE c.provider='vllm' AND a.state='COMMITTED'"
        )
        params: list[str] = []
        if work_keys is not None:
            sql += " AND c.work_key IN (" + ",".join("?" for _ in keys) + ")"
            params.extend(keys)
        rows = conn.execute(sql, params).fetchall()
        conn.close()
    except sqlite3.Error as e:
        raise LedgerUnreadable(f"{path} is unreadable: {e}") from e
    events = []
    for row in rows:
        try:
            usage = json.loads(row["usage_json"] or "{}")
        except (json.JSONDecodeError, TypeError) as e:
            raise LedgerUnreadable(
                f"{path} has invalid usage_json for work_key {row['work_key']!r}: {e}"
            ) from e
        events.append({
            "op_class": row["op_class"], "work_key": row["work_key"] or "",
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        })
    return events


def _open_external_calls(
    settings: Settings,
    *,
    work_keys: set[str] | None = None,
) -> int:
    import sqlite3

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        raise LedgerUnreadable(f"no provider ledger at {path}")
    keys = sorted(str(key) for key in (work_keys or ()) if key)
    if work_keys is not None and not keys:
        return 0
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        sql = (
            "SELECT COUNT(*) FROM external_call_attempts a"
            " JOIN external_calls c ON c.call_id = a.call_id"
            " WHERE a.state NOT IN ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN')"
        )
        params: list[str] = []
        if work_keys is not None:
            sql += " AND c.work_key IN (" + ",".join("?" for _ in keys) + ")"
            params.extend(keys)
        count = conn.execute(sql, params).fetchone()[0]
        conn.close()
        return int(count)
    except sqlite3.Error as e:
        raise LedgerUnreadable(f"{path} is unreadable: {e}") from e


def _listable(path: Path) -> bool:
    import os

    try:
        os.listdir(path)
        return True
    except OSError:
        return False


def _report(
    checks: list[dict],
    *,
    tasks=(),
    arms=(),
    settings=None,
    execution_binding_sha256: str = "",
    protocol_document_sha256: str = "",
) -> dict:
    ok = all(c["status"] == PASS for c in checks)
    body = {
        "ok": ok,
        "checks": checks,
        "tasks": list(tasks),
        "arms": list(arms),
        "execution_binding_sha256": execution_binding_sha256,
        "protocol_document_sha256": protocol_document_sha256,
        "claim_scope": settings.claim_scope if settings else "FORMATIVE_ONLY",
    }
    lines = [
        "# GPU smoke (canary)",
        "",
        f"Result: **{'PASS' if ok else 'FAIL'}**",
        "",
        "This canary checks engineering correctness and P1 non-inertness only. It does not,",
        "and may not, gate on a quality outcome: continuing based on how good early results",
        "look would select on the very effect the screen exists to estimate.",
        "",
        f"Claim scope: **{body['claim_scope']}**",
        "",
        f"Tasks: {', '.join(body['tasks']) or '(none)'}",
        f"Arms: {', '.join(body['arms']) or '(none)'}",
        "",
        "| Check | Status | Detail |",
        "|---|---|---|",
    ]
    lines += [f"| {c['name']} | {c['status']} | {c['detail']} |" for c in checks]

    # The per-arm publication table is printed whether or not its gate passed. A summary that
    # only shows numbers when something is already known to be wrong is a summary nobody reads
    # in time: these counts were all zero for 19 arms and 146 cells, and the run looked healthy.
    by_arm = next(
        (c.get("data", {}).get("by_arm") for c in checks
         if c["name"] == "p1_published_output"),
        None,
    )
    if by_arm:
        lines += [
            "",
            "## P1 output per arm",
            "",
            "| Arm | Cells | Published spans | Selector attempts | Page fallbacks | Top failure |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {arm_id} | {s['cells']} | {s['published_spans']} | "
            f"{s['selector_attempted']} | {s['page_fallbacks']} | "
            f"{s['top_failure'] or '-'} |"
            for arm_id, s in sorted(by_arm.items())
        ]
    return {"json": body, "markdown": "\n".join(lines) + "\n"}
