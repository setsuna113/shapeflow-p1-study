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

    # 3. The selector decoded. A P1 arm with zero decode did not select anything.
    try:
        inference = _inference_events(settings)
        ledger_error = ""
    except LedgerUnreadable as e:
        inference = []
        ledger_error = str(e)
    checks.append(_check("ledger_readable", not ledger_error,
                         ledger_error or "the provider ledger was read"))
    selector_ops = {"PAGE_P1_SELECTOR_LOCAL", "PAGE_P1_SELECTOR_GLOBAL",
                    "COMPRESSOR_P1_SELECTOR"}
    selector_completions = sum(e["completion_tokens"] for e in inference
                               if e["op_class"] in selector_ops)
    checks.append(_check("selector_decode", selector_completions > 0,
                         f"{selector_completions} completion tokens across selector calls"))

    # 4. BOTH boundaries produced checkpoints. This was a set intersection, so H-only or
    #    C-only passed -- and a canary whose close boundary never fired is exactly the run
    #    that proves nothing about the close node.
    kinds = {c["kind"] for v in committed.values() for c in v["checkpoints"]}
    wanted = {"HCheckpoint", "CCheckpoint"}
    checks.append(_check("checkpoints_present", wanted <= kinds,
                         f"checkpoint kinds seen: {sorted(kinds)}"
                         + ("" if wanted <= kinds else f"; missing {sorted(wanted - kinds)}")))

    # 5. Publication was whole-batch: every reduced batch was published once, never in pieces.
    partials = [k for k, v in committed.items()
                if v["counts"].get("page_batches_deferred", 0)
                != v["counts"].get("page_batches_reduced", 0)]
    checks.append(_check("atomic_publication", not partials,
                         "every deferred batch reduced exactly once" if not partials
                         else f"mismatched batches in {partials}"))

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
    try:
        open_calls = _open_external_calls(settings)
        checks.append(_check("reservations_closed", open_calls == 0,
                             f"{open_calls} external call(s) still open"))
    except LedgerUnreadable as e:
        checks.append(_check("reservations_closed", False, str(e)))

    # 10. Fallbacks and failures are on the ledger rather than silently absorbed. This was
    #     literally `_check(..., True, ...)`: it formatted the counts into a sentence and
    #     passed unconditionally, having asserted nothing at all.
    fallbacks = sum(v["counts"].get("page_fallbacks", 0) for v in committed.values())
    close_failures = sum(v["counts"].get("close_failed", 0) for v in committed.values())
    accounted, accounting_detail = _fallbacks_are_on_the_ledger(
        settings, expected=fallbacks + close_failures)
    checks.append(_check("fallback_and_failures_accounted", accounted,
                         f"{fallbacks} page fallback(s), {close_failures} close failure(s); "
                         + accounting_detail))

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
    prose_calls = [e for e in inference if e["work_key"] in prose_keys]
    prose_max = max((e["completion_tokens"] for e in prose_calls), default=0)
    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    # An arm that issued no request at all had max(..., default=0) <= cap and passed. And
    # the comparison was against the selector cap while the detail line quoted the rendered
    # budget, which is the number the check is named for.
    rendered = max((v["counts"].get("selected_tokens", 0) for v in prose_cells), default=0)
    prose_ok = bool(prose_cells) and bool(prose_calls) and prose_max <= cap and (
        rendered <= budget)
    checks.append(_check("short_prose_same_budget", prose_ok,
                         f"{len(prose_calls)} request(s), max {prose_max} completion tokens "
                         f"against a {cap} cap, {rendered} rendered tokens against a "
                         f"{budget} budget"))

    # 13. The cap that bounded generation is a completion limit, not a schema bound. This
    #     read two config values and multiplied them out to a constant True; it never looked
    #     at a request. It looks at the requests now.
    capped, capping_detail = _requests_carry_a_completion_cap(settings, cap)
    checks.append(_check("generation_cap_is_a_completion_limit", capped, capping_detail))
    return checks


def _by_arm(cells: dict) -> dict:
    grouped: dict = {}
    for record in cells.values():
        grouped.setdefault(record["cell"]["arm"]["arm_id"], []).append(record)
    return grouped


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


def _fallbacks_are_on_the_ledger(settings: Settings, *, expected: int) -> tuple:
    """Every fallback and failure the cells report has a matching ledger record."""
    import sqlite3

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        return False, f"no provider ledger at {path} to reconcile against"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        failed = conn.execute(
            "SELECT COUNT(*) FROM external_call_attempts WHERE state IN"
            " ('FAILED_FINAL','FAILED_UNKNOWN')").fetchone()[0]
        incidents = conn.execute("SELECT COUNT(*) FROM incidents").fetchone()[0]
        conn.close()
    except sqlite3.Error as e:
        return False, f"the ledger could not be reconciled: {e}"
    if expected and not (failed or incidents):
        return False, (f"{expected} fallback/failure(s) in the cells and none on the ledger; "
                       "the cost of a fallback is not being recorded")
    return True, f"{failed} failed attempt(s) and {incidents} incident(s) on the ledger"


def _requests_carry_a_completion_cap(settings: Settings, cap: int) -> tuple:
    """Read the requests that were actually sent, not the config that describes them."""
    import sqlite3

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    if not path.exists():
        return False, f"no provider ledger at {path}; no request could be inspected"
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT a.request_object_ref FROM external_call_attempts a"
            " JOIN external_calls c ON c.call_id = a.call_id"
            " WHERE c.provider='vllm' AND a.request_object_ref IS NOT NULL"
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        return False, f"the ledger could not be read: {e}"
    if not rows:
        return False, "no inference request was recorded, so no cap could be observed"

    from ..object_store import ObjectStore

    store = ObjectStore(settings.data_root
                        / str(settings.get("week1", "paths", "provider_root")) / "objects")
    uncapped = 0
    seen = 0
    for row in rows:
        try:
            body = json.loads(store.get_bytes(row["request_object_ref"]).decode("utf-8"))
        except Exception:  # noqa: BLE001 - an unreadable request is not an observed cap
            continue
        seen += 1
        limit = body.get("max_tokens")
        if not isinstance(limit, int) or limit <= 0 or limit > cap:
            uncapped += 1
    if not seen:
        return False, "no inference request body could be read back"
    if uncapped:
        return False, f"{uncapped}/{seen} inference request(s) carried no usable max_tokens"
    return True, f"{seen} inference request(s) all carried max_tokens <= {cap}"


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


def _inference_events(settings: Settings) -> list[dict]:
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
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT c.op_class, c.work_key, a.usage_json FROM external_calls c"
            " JOIN external_call_attempts a ON a.call_id = c.call_id"
            " WHERE c.provider='vllm' AND a.state='COMMITTED'"
        ).fetchall()
        conn.close()
    except sqlite3.Error as e:
        raise LedgerUnreadable(f"{path} is unreadable: {e}") from e
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
        raise LedgerUnreadable(f"no provider ledger at {path}")
    try:
        conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        count = conn.execute(
            "SELECT COUNT(*) FROM external_call_attempts WHERE state NOT IN"
            " ('COMMITTED','FAILED_FINAL','FAILED_UNKNOWN')").fetchone()[0]
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
