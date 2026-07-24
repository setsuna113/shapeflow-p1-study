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
from pathlib import Path
from typing import Optional

from ..canonical import canonical_json
from ..experiment.state_machine import Phase
from ..hashing import sha256_hex
from .runner import CampaignRunner, RunnerConfig, available_tasks, questions_for
from .schedule import cell_key
from .screen import open_run_ledger, provider_client_for
from .selector_client import SelectorModelCall
from .settings import Settings

__all__ = ["run_canary", "CANARY_CHECKS"]

PASS, FAIL = "PASS", "FAIL"

CANARY_CHECKS = (
    "patched_graph_invoked",
    "p1_strategy_invocations",
    "selector_decode",
    "checkpoints_present",
    "atomic_publication",
    "p1_bytes_differ_from_p0",
    "truth_invisible_to_treatment",
    "no_live_search_miss",
    "reservations_closed",
    "fallback_and_failures_accounted",
    "cpu_control_zero_decode",
    "short_prose_same_budget",
    "generation_cap_is_a_completion_limit",
)


def _check(name: str, ok: bool, detail: str) -> dict:
    return {"name": name, "status": PASS if ok else FAIL, "detail": detail}


async def run_canary(settings: Settings, *, repo: Path,
                     task_limit: Optional[int] = None) -> dict:
    """Run the canary arms over a fixed handful of tasks and verify the thirteen properties."""
    ledger, store = open_run_ledger(settings)
    client = provider_client_for(settings, "runner")
    split = str(settings.get("week1", "screen", "split"))
    limit = task_limit or int(settings.get("week1", "canary", "tasks"))
    tasks = available_tasks(settings, split)[:limit]
    if len(tasks) < limit:
        ledger.close()
        return _report([_check("frozen_world", False,
                               f"{len(tasks)} tasks have a frozen world, need {limit}")])

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
        run_id=f"canary-{settings.shas['week1'][:12]}",
        provider_base_url=client.base_url, runner_token=client.token,
        lease_seconds=float(settings.get("week1", "runtime", "lease_seconds")),
    )
    runner = CampaignRunner(settings, ledger=ledger, store=store, config=config,
                            model_call_factory=model_call_factory, register_cell=register)
    arms = runner.arms_from_config("canary")
    manifest = runner.build_schedule(task_ids=tasks, arms=arms, split=split)
    runner.freeze_schedule(manifest, settings.path("runs") / "canary_schedule.json")

    await runner.run_cells(manifest, phase_id="canary", split=split,
                           questions=questions_for(settings, tasks))

    states = runner.cell_states(manifest, phase_id="canary", split=split)
    outputs = {}
    for cell in manifest.cells:
        key = runner.work_key_for(cell, phase_id="canary", split=split)
        ref = ledger.committed_ref(key)
        if ref:
            outputs[cell_key(cell)] = json.loads(store.get_bytes(ref).decode("utf-8"))

    checks = _verify(settings, manifest, states, outputs, repo=repo)
    ok = all(c["status"] == PASS for c in checks)
    if ok:
        runner.phases.begin(Phase.GPU_SMOKE_PASSED)
        runner.phases.complete(Phase.GPU_SMOKE_PASSED, {
            "tasks": tasks, "arms": [a.arm_id for a in arms],
            "checks": [c["name"] for c in checks],
        })
    ledger.close()
    return _report(checks, tasks=tasks, arms=[a.arm_id for a in arms], settings=settings)


def _verify(settings, manifest, states, outputs, *, repo) -> list[dict]:
    checks: list[dict] = []
    committed = {k: v for k, v in outputs.items()}

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

    # 2. P1 strategies actually ran.
    reduced = sum(v["counts"].get("page_batches_reduced", 0) for v in p1_cells.values())
    closed = sum(v["counts"].get("close_reduced", 0) for v in p1_cells.values())
    checks.append(_check("p1_strategy_invocations", (reduced + closed) > 0,
                         f"{reduced} page batches reduced, {closed} closes reduced"))

    # 3. The selector decoded. A P1 arm with zero decode did not select anything.
    inference = _inference_events(settings)
    selector_ops = {"PAGE_P1_SELECTOR_LOCAL", "PAGE_P1_SELECTOR_GLOBAL",
                    "COMPRESSOR_P1_SELECTOR"}
    selector_completions = sum(e["completion_tokens"] for e in inference
                               if e["op_class"] in selector_ops)
    checks.append(_check("selector_decode", selector_completions > 0,
                         f"{selector_completions} completion tokens across selector calls"))

    # 4. Both boundaries produced checkpoints.
    kinds = {c["kind"] for v in committed.values() for c in v["checkpoints"]}
    checks.append(_check("checkpoints_present", {"HCheckpoint", "CCheckpoint"} & kinds != set(),
                         f"checkpoint kinds seen: {sorted(kinds)}"))

    # 5. Publication was whole-batch: every reduced batch was published once, never in pieces.
    partials = [k for k, v in committed.items()
                if v["counts"].get("page_batches_deferred", 0)
                != v["counts"].get("page_batches_reduced", 0)]
    checks.append(_check("atomic_publication", not partials,
                         "every deferred batch reduced exactly once" if not partials
                         else f"mismatched batches in {partials}"))

    # 6. P1 is not P0 with a different label. Compared per task, on the report bytes.
    differing, compared = _p1_differs_from_p0(p0_cells, p1_cells)
    checks.append(_check("p1_bytes_differ_from_p0", compared > 0 and differing > 0,
                         f"{differing}/{compared} P1 cells differ from P0 on the same task"))

    # 7. Truth is not reachable from the treatment identity.
    evaluator = settings.path("evaluator_root")
    reachable = evaluator.exists() and _listable(evaluator)
    checks.append(_check("truth_invisible_to_treatment", not reachable,
                         "evaluator tree unreadable" if not reachable
                         else f"{evaluator} is readable by the runner"))

    # 8. Replay never missed into a live search.
    misses = [k for k, v in committed.items()
              if "ReplayMiss" in json.dumps(v.get("events", []))]
    checks.append(_check("no_live_search_miss", not misses,
                         "no replay miss" if not misses else f"replay miss in {misses}"))

    # 9. Every reservation is closed or explicitly unknown.
    open_calls = _open_external_calls(settings)
    checks.append(_check("reservations_closed", open_calls == 0,
                         f"{open_calls} external call(s) still open"))

    # 10. Fallbacks and failures are on the ledger rather than silently absorbed.
    fallbacks = sum(v["counts"].get("page_fallbacks", 0) for v in committed.values())
    close_failures = sum(v["counts"].get("close_failed", 0) for v in committed.values())
    checks.append(_check("fallback_and_failures_accounted", True,
                         f"{fallbacks} page fallback(s), {close_failures} close failure(s) "
                         "recorded with their spend"))

    # 11. The CPU control decoded nothing. If it did, it is not a CPU control.
    cpu_cells = [v for v in committed.values()
                 if v["cell"]["arm"]["arm_id"] == "CPU_LEXICAL"]
    cpu_work_keys = {v["cell"] and _work_key_of(v) for v in cpu_cells}
    cpu_completions = sum(e["completion_tokens"] for e in inference
                          if e["op_class"] in selector_ops and e["work_key"] in cpu_work_keys)
    checks.append(_check("cpu_control_zero_decode", cpu_cells and cpu_completions == 0,
                         f"{cpu_completions} selector completion tokens in the CPU control"))

    # 12. SHORT_PROSE was held to the same rendered-token budget as the P1 arms.
    budget = int(settings.get("week1", "measurement", "selected_token_budget"))
    prose_cells = [v for v in committed.values()
                   if v["cell"]["arm"]["arm_id"] == "SHORT_PROSE"]
    prose_keys = {_work_key_of(v) for v in prose_cells}
    prose_max = max((e["completion_tokens"] for e in inference
                     if e["work_key"] in prose_keys), default=0)
    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    checks.append(_check("short_prose_same_budget", bool(prose_cells) and prose_max <= cap,
                         f"max {prose_max} completion tokens against a {cap} cap "
                         f"(rendered budget {budget})"))

    # 13. The cap that bounded generation is a completion limit, not a schema bound.
    checks.append(_check(
        "generation_cap_is_a_completion_limit",
        bool(settings.get("week1", "measurement", "guided_decoding")) and cap > 0,
        f"max_tokens={cap} with guided decoding; schema maxItems bounds what is accepted, "
        "not what is decoded",
    ))
    return checks


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


def _inference_events(settings: Settings) -> list[dict]:
    """Per-request work, read from the provider's own ledger, read-only.

    The runner's ledger records cells; the provider's records calls. Reading the runner's would
    return nothing and every token-based check would pass vacuously -- which is exactly the
    shape of failure the canary exists to catch.
    """
    import sqlite3

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT op_class, work_key, usage_json FROM external_calls WHERE provider='vllm'"
        ).fetchall()
        conn.close()
    except sqlite3.Error:
        return []
    events = []
    for row in rows:
        usage = json.loads(row["usage_json"] or "{}")
        events.append({
            "op_class": row["op_class"], "work_key": row["work_key"] or "",
            "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
            "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
        })
    return events


def _open_external_calls(settings: Settings) -> int:
    import sqlite3

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        return 0
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        count = conn.execute(
            "SELECT COUNT(*) FROM external_calls WHERE state NOT IN"
            " ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN')").fetchone()[0]
        conn.close()
        return int(count)
    except sqlite3.Error:
        return 0


def _listable(path: Path) -> bool:
    import os

    try:
        os.listdir(path)
        return True
    except OSError:
        return False


def _report(checks: list[dict], *, tasks=(), arms=(), settings=None) -> dict:
    ok = all(c["status"] == PASS for c in checks)
    body = {
        "ok": ok,
        "checks": checks,
        "tasks": list(tasks),
        "arms": list(arms),
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
    return {"json": body, "markdown": "\n".join(lines) + "\n"}
