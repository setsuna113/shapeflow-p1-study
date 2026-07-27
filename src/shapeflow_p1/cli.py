"""The `shapeflow-p1` command surface (plan §19).

The commands map one-to-one to campaign phases, and each asserts the identity it must run as:
the steward acquires and freezes, the runner executes treatments, the evaluator reads truth. A
command that checked nothing but a docstring would leave the whole UID separation resting on
whoever typed it.

Once ``reports/LAUNCH_GATE_PASSED.json`` exists, mutation commands accept only
``--resume --protocol-sha <exact>``: the historically named flag carries the complete approved
execution-binding digest. A diagnostic override after launch would silently alter an approved
run, and any real change to budget, sample or phase order mints a new execution binding and needs
a new approval instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import typer

app = typer.Typer(add_completion=False, help="ShapeFlow P1 Week-1 study CLI")

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = {
    "budget": _REPO / "configs" / "budget_v1.yaml",
}
_SCHEMAS = _REPO / "schemas"
_CFG = typer.Option(None, "--config", help="Campaign config (configs/week1.yaml)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


def _require_role(role: str) -> None:
    """Assert the effective identity. The separation is the boundary; this enforces it."""
    from .doctor import check_identity

    result = check_identity(role)
    if result.status == "FAIL":
        _fail(f"  FAIL  {result.name}: {result.detail}")


def _settings():
    from .campaign.settings import Settings

    return Settings.load(_REPO)


def _launched() -> bool:
    return (_REPO / "reports" / "LAUNCH_GATE_PASSED.json").exists()


def _assert_post_launch_flags(protocol_sha: Optional[str], resume: bool) -> None:
    """After launch, resume only under the exact complete approved execution binding."""
    if not _launched():
        return
    from .protocol import ApprovalError, verified_execution_binding

    if not resume:
        _fail("this run has already launched; mutation requires --resume", code=2)
    if not protocol_sha:
        _fail(
            "this run has already launched; --protocol-sha must carry the exact approved "
            "execution-binding digest",
            code=2,
        )
    try:
        verified_execution_binding(_REPO, expected_digest=protocol_sha)
    except ApprovalError as exc:
        _fail(
            "--protocol-sha is the approved execution-binding digest after launch; "
            f"resume identity did not verify: {exc}",
            code=2,
        )


def _content_sha(body: dict) -> str:
    """The digest of everything except the digest field itself."""
    from .canonical import canonical_json
    from .hashing import sha256_hex

    return sha256_hex(canonical_json(
        {key: value for key, value in body.items() if key != "content_sha256"}))


def _write_once_json(path: Path, body: dict, *, digest_field: str, label: str) -> None:
    """Create, or verify an identical existing artifact. Never replace.

    Exclusive creation rather than exists()-then-write: two resumptions reaching a freeze at once
    is a truncation race, and a pre-registration artifact that can be silently rewritten is not
    pre-registration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return
    except FileExistsError:
        pass
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get(digest_field) != body.get(digest_field):
        _fail(
            f"{path} already contains a different {label}; it is write-once. "
            f"Existing {existing.get(digest_field, '')[:12]}, "
            f"computed {str(body.get(digest_field, ''))[:12]}."
        )


def _require_approval():
    """Every command that can spend verifies the approval itself.

    Not in the launch script. A shell wrapper checks the approval once, at the top, for a
    sequence of commands each of which can be run on its own -- and each of which spends
    money or GPU hours when it is. `_assert_post_launch_flags` is not this check either: it
    returns immediately until reports/LAUNCH_GATE_PASSED.json exists, and reports/ is
    gitignored, so before the first launch it is a pass-through.
    """
    from .protocol import ApprovalError, verified_execution_binding

    try:
        return verified_execution_binding(_REPO)
    except ApprovalError as e:
        _fail(f"refusing to run: the approval does not bind this configuration.\n{e}")


def _record_phase(phase_name: str, detail: dict) -> None:
    """Record a completed campaign phase.

    Only three phases were ever written -- GPU_SMOKE_PASSED, SCREEN_RUNNING and
    SCREEN_COMPLETE -- and the state machine's first legal edge from NEW is DOCTOR_PASSED.
    So with the earlier gates unrecorded, `begin(GPU_SMOKE_PASSED)` raised
    IllegalPhaseTransition: a *passing* canary crashed on success while a failing one exited
    cleanly, and `run-screen` could not start at all.
    """
    from .campaign.phases import PhaseStore
    from .experiment.ledger import Ledger
    from .experiment.state_machine import Phase
    from .protocol import ApprovalError, verified_execution_binding

    settings = _settings()
    try:
        binding = verified_execution_binding(_REPO)
    except ApprovalError as exc:
        _fail(f"cannot record phase without a verified execution binding: {exc}")
    # Campaign phase ownership is the runner's ledger, the same ledger CampaignRunner reads.
    # Writing gate phases into the provider ledger made the state machine split-brain: the
    # canary saw NEW even though doctor/acquisition/parity had "completed" elsewhere. It also
    # asked non-provider UIDs to open a 0700 provider directory.
    path = settings.path("runs") / "ledger.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(path))
    try:
        phases = PhaseStore(ledger, protocol_sha=binding.digest)
        phase = Phase(phase_name)
        phases.begin(phase)
        phases.complete(phase, detail)
    finally:
        ledger.close()


def _provider_client(settings, role: str = "runner"):
    from .providers.provider_client import ProviderClient, load_role_token

    token_dir = str(settings.get("week1", "provider", "token_dir"))
    host = settings.get("week1", "provider", "bind_host")
    port = settings.get("week1", "provider", "bind_port")
    return ProviderClient(base_url=f"http://{host}:{port}",
                          token=load_role_token(token_dir, role))


# --- read-only ---------------------------------------------------------------------------


@app.command()
def doctor(
    config: Path = _CFG,
    role: str = typer.Option(None, "--role", help="Assert the effective identity for this role"),
) -> None:
    """Verify the environment and stack, read-only. Fails closed on any non-PASS check."""
    from .doctor import check_git_clean, check_stack_manifest, run_pure_checks

    report = run_pure_checks(repo=_REPO, configs=_CONFIGS, schema_dir=_SCHEMAS, role=role)
    report.add(check_git_clean(_REPO))
    report.add(check_stack_manifest(
        _REPO, _REPO / "configs" / "stack.yaml",
        measurement_layer=_settings().measurement_layer,
    ))
    if config is not None and not config.exists():
        _fail(f"  FAIL  config: {config} not found")
    for check in report.checks:
        typer.echo(f"  {check.status:4}  {check.name}: {check.detail}")
    if not report.ok:
        skipped = report.skipped
        if skipped:
            _fail(f"doctor: FAILED -- {len(skipped)} check(s) could not run. A check that did "
                  "not run has not been satisfied; it is not a pass.")
        _fail("doctor: FAILED")
    _record_phase("DOCTOR_PASSED", {"role": role or "unspecified",
                                    "checks": len(report.checks)})
    typer.echo("doctor: ok")


@app.command("verify-approval")
def verify_approval(
    approval: Path = typer.Option(
        ..., "--approval", help="External append-only approval store's current pointer"),
    config: Path = _CFG,
) -> None:
    """Assert the approval binds the whole live configuration, not just three of its hashes.

    The protocol SHA is read from the tracked document, never from the environment: a value the
    caller supplies and then compares against itself is a gate that cannot fail. A caller who
    *claims* a different SHA is a disagreement about which experiment is running, and fatal.
    """
    from .protocol import ApprovalError, verified_execution_binding

    try:
        binding = verified_execution_binding(_REPO, approval_path=approval)
    except ApprovalError as e:
        _fail(f"approval mismatch: {e}")

    claimed = os.environ.get("SHAPEFLOW_PROTOCOL_SHA")
    if claimed and claimed != binding.protocol_sha:
        _fail(f"SHAPEFLOW_PROTOCOL_SHA={claimed[:12]} does not match the protocol document "
              f"({binding.protocol_sha[:12]}); the caller and the repository disagree about "
              "which protocol is being run")
    typer.echo(f"approval ok (protocol={binding.protocol_sha[:12]} binding={binding.digest[:12]})")


@app.command()
def status(status_json: Path = typer.Option(None, help="Path to runs/STATUS.json")) -> None:
    """Print the latest STATUS.json if present."""
    path = status_json or (_REPO / "reports" / "STATUS.json")
    if not path.exists():
        typer.echo("no STATUS.json yet")
        raise typer.Exit(code=0)
    typer.echo(path.read_text(encoding="utf-8"))


@app.command("verify-artifacts")
def verify_artifacts(ledger_path: Path, object_store: Path) -> None:
    """Confirm every committed work item's artifact still verifies in the object store."""
    from .experiment.ledger import Ledger
    from .object_store import ObjectStore

    ledger = Ledger(str(ledger_path))
    store = ObjectStore(object_store)
    counts = ledger.state_counts()
    committed = counts.get("COMMITTED", 0)
    rows = ledger.raw_connection.execute(
        "SELECT result_object_ref FROM attempts WHERE state='COMMITTED'").fetchall()
    bad = sum(1 for r in rows if not (r["result_object_ref"]
                                      and store.verify(r["result_object_ref"])))
    typer.echo(f"committed={committed} verified={committed - bad} corrupt={bad}")
    ledger.close()
    if bad:
        raise typer.Exit(code=1)


@app.command("authorize-budget-raise")
def authorize_budget_raise(
    config: Path = _CFG,
    reason: str = typer.Option(..., "--reason", help="why the original ceiling was wrong"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Provider-only: apply raised caps from the frozen budget config, on the record.

    ``Budget.ensure_account`` refuses to widen a ceiling, and that refusal is correct -- a cap
    that drifted upward as a side effect of loading a config would let a round spend past what
    its approval was granted against. So a raise cannot happen implicitly, and editing
    ``budget_v1.yaml`` alone changes nothing about what this ledger will allow.

    This is the explicit act that applies one. It requires a verified approval, so the wider
    ceiling is bound to a protocol version someone froze deliberately; it records an incident
    per resource carrying the old value, the new value and how much was already spent; and it
    refuses to lower anything, because a call that moved caps in both directions would just be
    an ordinary write with a longer name.
    """
    from .experiment.budget import Budget
    from .experiment.ledger import Ledger

    _require_role("provider")
    binding = _require_approval()
    settings = _settings()
    ledger_path = settings.path("provider_ledger")
    if not ledger_path.exists():
        _fail(f"no provider ledger at {ledger_path}")

    caps = settings.budget_caps()
    ledger = Ledger(str(ledger_path))
    try:
        budget = Budget(ledger)
        current = {
            r["resource"]: (r["cap"], r["settled_total"])
            for r in ledger.raw_connection.execute(
                "SELECT resource, cap, settled_total FROM budget_accounts")
        }
        changes = []
        for resource, cap in sorted(caps.items()):
            have = current.get(resource)
            if have is None or cap <= have[0]:
                continue
            changes.append((resource, have[0], cap, have[1]))
        if not changes:
            typer.echo("no cap in the frozen config is above the ledger; nothing to raise")
            return
        for resource, was, now, spent in changes:
            typer.echo(f"  {resource}: {was} -> {now}   (already spent {spent})")
        if dry_run:
            typer.echo("dry run; nothing was changed")
            return
        for resource, _was, now, _spent in changes:
            budget.authorize_cap_raise(
                resource, now,
                authorization=f"approval binding {binding.digest[:12]}",
                reason=reason,
            )
    finally:
        ledger.close()
    typer.echo(f"raised {len(changes)} cap(s) under binding {binding.digest[:12]}")


#: Why the first round's treatment artifacts cannot be analysed. Each is independently
#: sufficient; together they mean no cell in that round observed the treatment it is labelled
#: with. They are enumerated rather than summarised because a later reader deciding whether some
#: subset is salvageable needs to see all four.
TREATMENT_INVALIDATION_REASONS = {
    "P1_PUBLISHED_OUTPUT_COUNT_ZERO": (
        "Across 19 arms and 146 cells, not one P1 span was published. Every H batch raised "
        "`publication handle 'H3_1_0_1' costs 9 exact model tokens, exceeding the frozen cap 8` "
        "on its first candidate and fell back to P0 as a whole, so every H cell is a P0 cell "
        "wearing an H label."
    ),
    "H_VIEW_CONSTRUCTION_FALLBACK_ALL": (
        "The failure was total rather than partial: 2,067 recorded view-construction failures "
        "across 87 distinct handles, with no arm publishing anything. A fallback rate is an "
        "outcome; a 100% fallback rate before the first selector call is an apparatus that "
        "never ran the treatment."
    ),
    "P0_NON_VENDOR_OUTPUT_CAP_1024": (
        "summarization_model_max_tokens was 1024 against the pinned vendor default of 8192. "
        "Vendor's summariser falls back to the raw page when truncated, so 263 of 1839 P0 "
        "summaries (14.3%) published whole pages as compressed notes -- worst on the largest "
        "pages, and P1 never reaches that code path. The baseline was handicapped exactly where "
        "P1 was meant to win."
    ),
    "SERIAL_REGIME_NOT_TARGET_ODR": (
        "The engine admitted one upstream request at a time, which was a precondition of the "
        "summed-service metric rather than of causal isolation. The pinned graph summarises a "
        "result set with asyncio.gather, so the serialization queued eight concurrent summaries "
        "behind each other and produced 212 vendor timeouts that exist in no native run. The "
        "measured system is not the system the study is about."
    ),
}


@app.command("invalidate-treatment")
def invalidate_treatment(
    config: Path = _CFG,
    attempt_id: str = typer.Option(..., "--attempt-id", help="e.g. serial-engineering-smoke"),
    state: str = typer.Option("INVALIDATED_SERIAL_ENGINEERING_SMOKE", "--state"),
    note: str = typer.Option("", "--note"),
) -> None:
    """Seal a round of treatment artifacts as unanalysable, without deleting any of them.

    Nothing is removed and nothing is refunded. The point is that a later reader can tell the
    difference between "this round produced no P1 effect" and "this round never executed P1" --
    which the artifacts alone cannot say, because a P0 fallback is a legitimate part of the ITT
    design and a run that fell back 146 times out of 146 looks exactly like a run.

    The new campaign writes to per-lane trees, so the sealed round is not overwritten either.
    """
    _require_role("runner")
    settings = _settings()
    root = settings.data_root / str(settings.get("week1", "paths", "runner_root"))
    marker = root / f"{state}.json"

    inventory: dict[str, int] = {}
    for name in ("runs", "object_store", "checkpoints"):
        directory = settings.data_root / str(settings.get("week1", "paths", name))
        inventory[name] = (
            sum(1 for path in directory.rglob("*") if path.is_file())
            if directory.exists() else 0
        )

    body = {
        "schema_version": "invalidated_treatment_attempt_v1",
        "attempt_id": attempt_id,
        "state": state,
        "reasons": TREATMENT_INVALIDATION_REASONS,
        "note": note,
        "sealed_at_utc": _now(),
        "sealed_tree": str(root),
        "artifact_counts": inventory,
        "protocol_sha_at_sealing": __import__(
            "shapeflow_p1.protocol", fromlist=["protocol_sha"]).protocol_sha(_REPO),
        "analysis_permitted": False,
    }
    body["content_sha256"] = _content_sha(body)
    _write_once_json(marker, body, digest_field="content_sha256",
                     label="treatment invalidation record")
    typer.echo(json.dumps({
        "state": state, "sealed_tree": str(root),
        "artifact_counts": inventory,
        "reasons": sorted(TREATMENT_INVALIDATION_REASONS),
        "content_sha256": body["content_sha256"],
    }, indent=2, sort_keys=True))


# --- steward -----------------------------------------------------------------------------


@app.command("freeze-approval")
def freeze_approval(
    approved_commit: str = typer.Option("", "--approved-commit"),
    approval: Path = typer.Option(None, "--approval"),
) -> None:
    """Steward-only: record approval outside the clean Git execution tree."""
    from .protocol import ApprovalError, write_approval_file

    _require_role("steward")
    try:
        binding = write_approval_file(_REPO, approved_at_utc=_now(),
                                      approved_commit=approved_commit, approval_path=approval)
    except ApprovalError as e:
        _fail(f"cannot write approval: {e}")
    typer.echo(f"approval recorded (binding={binding.digest[:12]})")


@app.command("freeze-stack")
def freeze_stack(
    config: Path = _CFG,
    engine_pid: int = typer.Option(None, "--engine-pid", help="Running vLLM pid, for its flags"),
    engine_log: Path = typer.Option(None, "--engine-log",
                                    help="The engine's startup log, for its attention backend"),
) -> None:
    """Steward-only: resolve every @STEWARD_FREEZES@ field into protocol/stack_manifest.json."""
    from .ops.live_stack import (
        StackError,
        attention_backend_from_log,
        freeze_stack as do_freeze,
        observe,
    )

    _require_role("steward")
    settings = _settings()
    stack = settings.configs["stack"]
    observation = observe(
        gpu_uuid=str(stack["host"]["gpu_uuid"]),
        model_dir=Path(str(stack["model"]["path"])),
        vllm_python=Path(str(stack["engine"]["vllm_venv"])) / "bin" / "python",
        engine_pid=engine_pid,
    )
    # The attention backend is only knowable from a served engine: vLLM picks it at startup.
    # Recorded from the engine's own log when it is running, and left unresolved otherwise so
    # the freeze refuses rather than writing down a plausible guess.
    backend = attention_backend_from_log(engine_log) if engine_log else ""
    if backend:
        observation.values["attention_backend"] = backend
    try:
        body = do_freeze(_REPO, stack, observation, frozen_at_utc=_now())
    except StackError as e:
        _fail(f"freeze-stack failed: {e}")
    typer.echo(f"stack frozen (manifest={body['manifest_sha256'][:12]}, "
               f"unavailable={len(body['unavailable'])})")


# --- gates --------------------------------------------------------------------------------


@app.command("test-p0-parity")
def test_p0_parity(config: Path = _CFG) -> None:
    """Assert the patched hooks-off graph reproduces vendor, and that explicit P0 does too."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/integration/test_p0_parity.py"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=3600,
    )
    typer.echo(result.stdout[-4000:])
    if result.returncode != 0:
        typer.echo(result.stderr[-4000:], err=True)
        _fail("P0 parity failed; all GPU screening is barred")
    from .canonical import canonical_json
    from .hashing import sha256_hex
    from .protocol import protocol_sha

    probe_paths = (
        _REPO / "tests" / "integration" / "test_p0_parity.py",
        _REPO / "tests" / "integration" / "parity_probe.py",
        _REPO / "patches" / "odr_p1_hooks.patch",
    )
    report = {
        "status": "PASS",
        "protocol_sha": protocol_sha(_REPO),
        "probe": "tests/integration/test_p0_parity.py",
        "probe_inputs_sha256": sha256_hex(canonical_json({
            str(path.relative_to(_REPO)): sha256_hex(path.read_bytes())
            for path in probe_paths
        })),
    }
    report["content_sha256"] = sha256_hex(canonical_json(report))
    path = _REPO / "reports" / "P0_PARITY_PASSED.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo("p0 parity: ok")


@app.command()
def accept(config: Path = _CFG) -> None:
    """Run the acceptance matrix and write reports/ACCEPTANCE.json."""
    from .ops.acceptance import run_acceptance

    settings = _settings()
    body = run_acceptance(settings, repo=_REPO)
    path = _REPO / "reports" / "ACCEPTANCE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for gate in body["gates"]:
        typer.echo(f"  {gate['status']:4}  {gate['name']}: {gate['detail']}")
    if not body["ok"]:
        _fail("acceptance: FAILED")
    typer.echo("acceptance: ok")


@app.command()
def preflight(
    config: Path = _CFG,
    approved_protocol_sha: str = typer.Option(None, "--approved-protocol-sha"),
) -> None:
    """Campaign preflight: the approval, the sealed corpus, the frozen world, the schedule."""
    from .ops.acceptance import run_preflight

    settings = _settings()
    body = run_preflight(settings, repo=_REPO, approved_protocol_sha=approved_protocol_sha)
    for check in body["checks"]:
        typer.echo(f"  {check['status']:4}  {check['name']}: {check['detail']}")
    if not body["ok"]:
        _fail("preflight: FAILED")
    # Preflight is run by the runner before treatment. It has just re-derived every frozen
    # world and the P0 parity receipt, so it is the one identity that can safely advance the
    # runner-owned phase spine without giving the truth-holding steward write access to
    # treatment state.
    parity = json.loads(
        (_REPO / "reports" / "P0_PARITY_PASSED.json").read_text(encoding="utf-8"))
    _record_phase("ACQUISITION_COMPLETE", {
        "preflight_protocol_sha": body["protocol_sha"],
    })
    _record_phase("SNAPSHOTS_FROZEN", {
        "frozen_world": next(
            c["detail"] for c in body["checks"] if c["name"] == "frozen_world"),
    })
    _record_phase("P0_PARITY_PASSED", {
        "receipt_sha256": parity["content_sha256"],
        "probe_inputs_sha256": parity["probe_inputs_sha256"],
    })
    typer.echo("preflight: ok")


@app.command()
def smoke(config: Path = _CFG,
          tasks: int = typer.Option(None, "--tasks"),
          protocol_sha: str = typer.Option(None, "--protocol-sha"),
          resume: bool = typer.Option(False, "--resume")) -> None:
    """Runner-only: the real GPU canary. Engineering correctness only, never a quality gate."""
    from .campaign.canary import run_canary

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    binding = _require_approval()
    settings = _settings()
    body = asyncio.run(run_canary(
        settings,
        repo=_REPO,
        task_limit=tasks,
        execution_binding_sha256=binding.digest,
    ))
    path = _REPO / "reports" / "GPU_SMOKE_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body["markdown"], encoding="utf-8")
    (_REPO / "reports" / "GPU_SMOKE.json").write_text(
        json.dumps(body["json"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for check in body["json"]["checks"]:
        typer.echo(f"  {check['status']:4}  {check['name']}: {check['detail']}")
    if not body["json"]["ok"]:
        _fail("gpu smoke: FAILED")
    typer.echo("gpu smoke: ok")


# --- the campaign ---------------------------------------------------------------------------


@app.command("freeze-shards")
def freeze_shards(config: Path = _CFG) -> None:
    """Steward-only: freeze which GPU lane runs which task, before any of them runs.

    Task-atomic. Every replicate and every arm of one task stays on one lane, because a block is
    paired-valid only under a single engine epoch and the paired contrast is P1 against P0 on
    the same task -- P0 on one card and P1 on another would fold that card's clocks and thermals
    into the treatment effect with nothing afterwards able to separate them.

    Balanced on the frozen corpus's own byte size, which exists before any arm runs. Written
    write-once: a lane chosen after a result is seen is not a partition, it is a selection.
    """
    from .campaign.acquire import acquired_task_ids
    from .campaign.schedule import build_blocks
    from .campaign.screen import available_tasks
    from .campaign.sharding import SHARD_MANIFEST_FILENAME, Lane, build_shard_manifest

    _require_role("steward")
    settings = _settings()
    binding = _require_approval()
    split = str(settings.get("week1", "screen", "split"))
    tasks = available_tasks(settings, split)
    if not tasks:
        _fail(f"freeze-shards: no {split} task has a frozen world")

    from .campaign.runner import CampaignRunner  # arm resolution lives with the runner

    arms = CampaignRunner.arms_from_config_static(
        settings, str(settings.get("week1", "screen", "arms_block")))
    manifest = build_blocks(
        execution_binding_sha256=binding.digest,
        protocol_sha=binding.protocol_sha,
        split=split,
        task_ids=tasks,
        arms=arms,
        seeds=[int(s) for s in settings.get("week1", "screen", "seeds")],
        layer=settings.measurement_layer,
        claim_scope=settings.claim_scope,
        second_seed_fraction=float(settings.get("week1", "screen", "second_seed_fraction")),
    )

    shards = settings.get("week1", "measurement", "shards")
    lane_count = int(shards["lane_count"])
    pool = list(settings.get("stack", "host", "gpu_uuid_pool"))
    if len(pool) < lane_count:
        _fail(f"freeze-shards: {lane_count} lanes but only {len(pool)} GPUs in the frozen pool")
    lanes = [
        Lane(
            shard_id=index,
            gpu_uuid=str(pool[index]),
            vllm_port=int(shards["vllm_base_port"]) + index,
            provider_port=int(shards["provider_base_port"]) + index,
            runner_root=str(shards["runner_root_template"]).format(lane=index),
            serves_paid_upstreams=(index == int(shards["paid_upstream_lane"])),
        )
        for index in range(lane_count)
    ]

    # Pre-treatment only: the frozen corpus's bytes for a task exist before any arm runs.
    costs: dict[str, float] = {}
    del acquired_task_ids
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    for task_id in tasks:
        try:
            costs[task_id] = float((pool_dir / f"{task_id}.json").stat().st_size)
        except OSError as exc:
            _fail(f"freeze-shards: no frozen pool for {task_id}: {exc}")

    stack_manifest = _REPO / "protocol" / "stack_manifest.json"
    try:
        stack_sha = str(json.loads(
            stack_manifest.read_text(encoding="utf-8")).get("manifest_sha256") or "")
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"freeze-shards: stack manifest unreadable ({exc}); run freeze-stack first")
    body = build_shard_manifest(
        manifest, lanes=lanes, task_costs=costs,
        stack_manifest_sha256=stack_sha, protocol_sha256=binding.protocol_sha,
    )
    directory = settings.path("shards")
    directory.mkdir(parents=True, exist_ok=True)
    _write_once_json(directory / SHARD_MANIFEST_FILENAME, body,
                     digest_field="shard_manifest_sha256", label="shard manifest")
    typer.echo(json.dumps({
        "shard_manifest_sha256": body["shard_manifest_sha256"],
        "lane_count": lane_count,
        "tasks": len(tasks),
        "cells_by_shard": {k: len(v) for k, v in body["cells_by_shard"].items()},
    }, indent=2, sort_keys=True))


@app.command("merge-shards")
def merge_shards(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Steward-only: reconstitute one campaign from the lanes, or refuse.

    Four separately-valid lanes are not a valid campaign. A task silently run twice, a lane's
    blocks quietly missing, or one task's arms split across two GPUs each leave every individual
    lane internally consistent -- the error exists only in the union, so this is the last place
    it can be caught. Nothing downstream reads a lane's root directly.
    """
    from .campaign.schedule import FROZEN_ROOT_FILENAME, build_blocks
    from .campaign.screen import available_tasks
    from .campaign.sharding import (
        SHARD_MANIFEST_FILENAME,
        ShardMergeError,
        merge_shard_freeze_roots,
    )

    _require_role("steward")
    settings = _settings()
    binding = _require_approval()
    split = str(settings.get("week1", "screen", "split"))

    from .campaign.runner import CampaignRunner

    manifest = build_blocks(
        execution_binding_sha256=binding.digest,
        protocol_sha=binding.protocol_sha,
        split=split,
        task_ids=available_tasks(settings, split),
        arms=CampaignRunner.arms_from_config_static(
            settings, str(settings.get("week1", "screen", "arms_block"))),
        seeds=[int(s) for s in settings.get("week1", "screen", "seeds")],
        layer=settings.measurement_layer,
        claim_scope=settings.claim_scope,
        second_seed_fraction=float(settings.get("week1", "screen", "second_seed_fraction")),
    )
    shard_dir = settings.path("shards")
    body = json.loads((shard_dir / SHARD_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    template = str(settings.get("week1", "measurement", "shards", "runner_root_template"))
    suffix = str(settings.get("week1", "paths", "runs"))[
        len(str(settings.get("week1", "paths", "runner_root"))):
    ].lstrip("/")
    roots: dict[int, dict] = {}
    for lane in body["lanes"]:
        shard_id = int(lane["shard_id"])
        path = (
            settings.data_root / template.format(lane=shard_id) / suffix
            / "e2e_blocks" / run_id / FROZEN_ROOT_FILENAME
        )
        try:
            roots[shard_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"merge-shards: lane {shard_id} freeze root unreadable at {path}: {exc}")
    try:
        merged = merge_shard_freeze_roots(body, manifest, roots)
    except ShardMergeError as exc:
        _fail(f"merge-shards: BLOCKED: {exc}")
    _write_once_json(shard_dir / f"MERGED_ROOT_{run_id}.json", merged,
                     digest_field="merged_root_sha256", label="merged campaign root")
    typer.echo(json.dumps({
        "merged_root_sha256": merged["merged_root_sha256"],
        "lane_count": merged["lane_count"],
        "total_cells": merged["total_cells"],
        "blocks": len(merged["blocks"]),
    }, indent=2, sort_keys=True))


@app.command("run-week1")
def run_week1(config: Path = _CFG,
              resume: bool = typer.Option(False, "--resume"),
              protocol_sha: str = typer.Option(None, "--protocol-sha")) -> None:
    """Runner-only: the Week-1 campaign. Idempotent; a restart fills gaps."""
    from .campaign.screen import run_screening

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    binding = _require_approval()
    body = asyncio.run(run_screening(
        _settings(),
        repo=_REPO,
        max_cells=None,
        execution_binding_sha256=binding.digest,
    ))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    if not body.get("ok", False):
        raise typer.Exit(code=1)


@app.command("stop-safely")
def stop_safely(config: Path = _CFG) -> None:
    """Stop admission without modifying the protocol. Finished work is kept."""
    settings = _settings()
    sentinel = settings.data_root / str(settings.get("week1", "runtime", "stop_sentinel"))
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(_now() + "\n", encoding="utf-8")
    typer.echo(f"stop requested at {sentinel}")


@app.command("release-holdout")
def release_holdout(config: Path = _CFG) -> None:
    """Holdout gate: materialize the holdout corpus for the runner, once.

    Fail-closed, and this round it must fail: a formative machine-authored corpus has no
    confirmatory holdout, and the campaign never reaches POLICY_FROZEN. This is a gate that
    refuses, not a stub -- it checks the real preconditions and reports which one is unmet.
    """
    from .campaign.phases import PhaseStore
    from .experiment.ledger import Ledger
    from .experiment.state_machine import Phase

    settings = _settings()
    if not bool(settings.get("week1", "campaign", "open_holdout")):
        _fail("configs/week1.yaml sets open_holdout: false -- this corpus is FORMATIVE_ONLY and "
              "has no confirmatory holdout to release")
    binding = _require_approval()
    ledger = Ledger(str(settings.data_root / str(settings.get("week1", "paths", "runs"))
                        / "ledger.sqlite"))
    phases = PhaseStore(ledger, protocol_sha=binding.digest)
    if not phases.is_complete(Phase.POLICY_FROZEN):
        ledger.close()
        _fail("holdout release requires POLICY_FROZEN; releasing earlier would let the holdout "
              "be seen before the policy that it validates was fixed")
    ledger.close()
    _fail("holdout release is not authorized under this approval")


@app.command("serve-provider")
def serve_provider(config: Path = _CFG) -> None:  # pragma: no cover - process entry point
    """Run the provider. The only process that reads a credential, and never as root."""
    from .runtime.provider_main import run

    _require_role("provider")
    if config is not None and not config.exists():
        _fail(f"  FAIL  config: {config} not found")
    run(_settings())


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
