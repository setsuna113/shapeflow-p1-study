"""The `shapeflow` command surface (plan §19).

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

app = typer.Typer(add_completion=False, help="ShapeFlow experiment CLI")

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
            "shapeflow.protocol", fromlist=["protocol_sha"]).protocol_sha(_REPO),
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


# --- BrowseComp-Plus (Freeze-1) ---------------------------------------------------------------


@app.command("run-bcplus")
def run_bcplus_cmd(
    config: Path = _CFG,
    layer: str = typer.Option(..., "--layer", help="Design layer, e.g. b1_select"),
    arms_block: str = typer.Option(..., "--arms", help="Arm block in configs/week1.yaml"),
    retrieval_url: str = typer.Option("http://127.0.0.1:8710", "--retrieval-url"),
    tasks: Optional[int] = typer.Option(None, "--tasks", help="Prefix of the layer to run"),
    max_cells: Optional[int] = typer.Option(None, "--max-cells"),
    phase_id: str = typer.Option("bcplus", "--phase-id"),
    run_id: Optional[str] = typer.Option(None, "--run-id"),
    shard: int = typer.Option(0, "--shard", help="This lane's index in a task-atomic partition"),
    shards: int = typer.Option(1, "--shards", help="How many lanes share this layer"),
    allow_ineffective_freeze: bool = typer.Option(
        False, "--allow-ineffective-freeze",
        help="Run against a retrieval freeze the competence pilot has not yet validated. Only "
             "the pilot itself and the pre-pilot liveness smoke may: every other run refuses, "
             "which is what stops an unvalidated retriever from quietly serving a campaign."),
    protocol_sha: str = typer.Option(None, "--protocol-sha"),
    resume: bool = typer.Option(False, "--resume"),
) -> None:
    """Runner-only: one design layer of the BrowseComp-Plus campaign. Idempotent under --resume."""
    from .campaign.bcplus import run_bcplus

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    binding = _require_approval()
    settings = _settings()
    stop = settings.data_root / str(settings.get("week1", "runtime", "stop_sentinel"))
    body = asyncio.run(run_bcplus(
        settings,
        repo=_REPO,
        layer=layer,
        arms_block=arms_block,
        retrieval_base_url=retrieval_url,
        task_limit=tasks,
        max_cells=max_cells,
        execution_binding_sha256=binding.digest,
        phase_id=phase_id,
        # The competence pilot is what *makes* the freeze effective, so it and the smoke that
        # proves the apparatus works are the only runs that legitimately precede it.
        require_effective_freeze=not allow_ineffective_freeze,
        stop_sentinel=stop,
        run_id=run_id,
        shard=shard,
        shards=shards,
    ))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    if not body.get("ok", False):
        raise typer.Exit(code=1)


@app.command("grade-bcplus")
def grade_bcplus(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    layer: str = typer.Option(..., "--layer"),
    phase_id: str = typer.Option("bcplus", "--phase-id"),
    baseline_arm: str = typer.Option("P0", "--baseline-arm"),
    out: Optional[Path] = typer.Option(None, "--out"),
    lanes: Optional[str] = typer.Option(
        None, "--lanes",
        help="Comma-separated lane ids whose ledgers hold this run, e.g. '0,1'. Omit for one "
             "unsharded lane."),
    skip_grading: bool = typer.Option(
        False, "--skip-grading", help="Recall and work only; no judge calls."),
    ran_under_binding: Optional[str] = typer.Option(
        None, "--ran-under-binding",
        help="Execution binding the run committed under, when the tree has since moved. "
             "Grading is read-only; refusing to name it would mean a grading bug could only "
             "ever be fixed by re-running the campaign."),
) -> None:
    """Evaluator-only: grade a finished BC+ run and emit the paired analysis.

    Reads the answer key. Runs as the evaluator identity, whose import of the qrels module is
    exactly what the treatment identities are refused.
    """
    from .bench.bcplus.analysis import attach_grades, attach_recall, build_report, load_cells
    from .bench.bcplus.qrels import load_bcplus_evaluator_queries
    from .bench.bcplus.render import render_markdown
    from .campaign.bcplus import benchdata_root
    from .campaign.runner import STAGE_VERSION
    from .campaign.session import open_run_ledger

    _require_role("evaluator")
    binding = _require_approval()
    # Both reach the filesystem: run_id names a schedule directory and a report file, layer names
    # the schedule inside it. They come off the command line, so they get the same one-component
    # grammar the runner applies before it writes anything.
    from .scoped_paths import safe_scope_component

    run_id = safe_scope_component(run_id, name="run_id")
    layer = safe_scope_component(layer, name="layer")
    # A work key is namespaced by the execution binding, and the binding contains
    # ``approved_commit`` -- so every commit renames every work key ever written. For the runner
    # that is the point: a cell committed under different bytes is a different cell. For the
    # evaluator it is a trap. Grading reads a finished run and writes no cell, yet with the
    # namespace pinned to HEAD the only way to fix a grading bug was to re-run the campaign that
    # exposed it. Twice already a judge-parsing fix would have cost a pilot.
    #
    # So the run's own binding may be named. It is checked against the approval chain, not taken
    # on trust: an unrecorded digest is refused, which keeps this from becoming a way to point
    # the analysis at an arbitrary namespace.
    namespace = binding.digest
    if ran_under_binding and ran_under_binding != binding.digest:
        from .protocol import approval_chain_digests

        known = approval_chain_digests(_REPO)
        if ran_under_binding not in known:
            _fail(
                f"--ran-under-binding {ran_under_binding[:12]} is not in this repository's "
                f"approval chain ({len(known)} recorded binding(s)). Grading a namespace that "
                "was never approved would be grading a run this protocol never authorised."
            )
        typer.echo(f"grading cells committed under {ran_under_binding[:12]} "
                   f"(the live approval is {binding.digest[:12]})")
        namespace = ran_under_binding

    def work_key_for(ledger, cell: dict) -> str:
        arm = cell["arm"]
        return ledger.work_key(
            protocol_sha=namespace, split=layer, phase_id=phase_id,
            task_id=str(cell["task_id"]), arm_id=str(arm["arm_id"]),
            variant_id=f"{arm['page_variant']}+{arm['close_variant']}",
            replicate_id=str(cell["replicate_id"]), checkpoint_hash=str(cell["block_id"]),
            stage_version=STAGE_VERSION)

    # Sharded lanes each keep their own ledger and object store, and the analysis needs their
    # union: a lane holds one half of the task partition, so grading one of them would report
    # half a campaign as a whole one. Read per lane, concatenate, and record which lanes
    # contributed.
    lane_ids = [lane.strip() for lane in (lanes or "").split(",") if lane.strip()] or [None]
    records, schedules, seen_keys = [], [], set()
    for lane in lane_ids:
        if lane is None:
            os.environ.pop("SHAPEFLOW_LANE", None)
        else:
            os.environ["SHAPEFLOW_LANE"] = lane
        settings = _settings()
        ledger, store = open_run_ledger(settings)
        schedule_path = settings.path("runs") / "schedules" / run_id / f"{layer}.json"
        if not schedule_path.exists():
            ledger.close()
            _fail(f"no frozen schedule at {schedule_path}")
        schedules.append(str(schedule_path))
        for record in load_cells(schedule_path=schedule_path, ledger=ledger, store=store,
                                 work_key_for=lambda cell, _l=ledger: work_key_for(_l, cell)):
            # Every lane's schedule names every block; only its own share ran. Keep the first
            # terminal record for a cell and drop the other lane's empty view of it, or a cell
            # another lane committed would be counted here as a missing one.
            key = (record.task_id, record.arm_id, record.replicate_id)
            if key in seen_keys and not record.output_ref:
                continue
            if key in seen_keys:
                records = [r for r in records
                           if (r.task_id, r.arm_id, r.replicate_id) != key]
            seen_keys.add(key)
            records.append(record)
        ledger.close()

    queries = load_bcplus_evaluator_queries(
        benchdata_root() / "browsecomp-plus" / "data")
    attach_recall(records, queries)

    grading = {"graded": 0, "errors": [], "ungraded": 0, "skipped": True}
    if not skip_grading:
        # The judge must be reached through the *paid* lane's provider. Only one lane may reach
        # an upstream that costs money -- four providers each admitting against the full DeepSeek
        # cap would be a four-fold budget -- and the others refuse `deepseek.chat` outright. The
        # lane loop above left SHAPEFLOW_LANE on whichever lane it read last, so the settings the
        # grader is built from are resolved here rather than inherited from it.
        os.environ.pop("SHAPEFLOW_LANE", None)
        paid_lane = _settings().get("week1", "measurement", "shards", "paid_upstream_lane")
        os.environ["SHAPEFLOW_LANE"] = str(paid_lane)
        grader, judge_meta = _bcplus_grader(_settings())
        grading = attach_grades(records, queries, grader,
                                on_progress=lambda i, n: typer.echo(f"  graded {i}/{n}", err=True))
        grading["judge"] = judge_meta
        grading["skipped"] = False

    body = build_report(records, baseline_arm=baseline_arm, context={
        "run_id": run_id, "layer": layer, "phase_id": phase_id,
        "lanes": lane_ids,
        "schedules": schedules,
        "execution_binding_sha256": binding.digest,
        # The namespace the cells were actually read from, which is the live binding unless
        # --ran-under-binding moved it. Recorded because it is the one field a reader needs to
        # re-run this analysis and get this table: without it the artifact names the binding of
        # the tree that graded the run, under which not one of its cells was produced.
        "cells_committed_under_sha256": namespace,
        "protocol_document_sha256": binding.protocol_sha,
        "evaluator_source_sha256": queries.source_sha256,
        "grading": grading,
    })
    destination = out or (_REPO / "reports" / f"BCPLUS_{run_id}.json")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    markdown = destination.with_suffix(".md")
    markdown.write_text(render_markdown(body), encoding="utf-8")
    typer.echo(f"wrote {destination} and {markdown}")

    for arm, contrast in sorted(body["contrasts"].items()):
        pairing = contrast.get("pairing") or {}
        if not pairing.get("reportable", True):
            typer.echo(
                f"  WARN  {arm}: only {pairing['n_pairs']} of "
                f"{pairing['tasks_either_arm_committed']} tasks paired "
                f"({pairing['survival']:.0%}); this contrast is survivorship-biased",
                err=True)
    # An unavailable grade is not a zero, and a report whose accuracy is mostly missing must not
    # read as a report whose accuracy is low. Anything past a twentieth and this exits non-zero
    # rather than being quietly published.
    graded, ungraded = int(grading.get("graded", 0)), int(grading.get("ungraded", 0))
    if graded and ungraded > 0.05 * graded:
        for problem in (grading.get("errors") or [])[:3]:
            typer.echo(f"  {problem}", err=True)
        _fail(f"grading: {ungraded}/{graded} tasks have no judgment. Accuracy computed over the "
              "subset a judge happened to answer is not the accuracy that was pre-registered.")


def _bcplus_grader(settings):
    """The official BC+ grader, judged by DeepSeek through the provider.

    Through the provider and not directly: the key is readable only by the provider UID, the
    budget cap is enforced there, and every judge call has to land in the same ledger as
    everything else or the cost of scoring is off the books.

    Model, decoding envelope and retry budget all come from ``configs/judge.yaml`` through
    Settings, never from literals here. ``judge_sha`` is inside the execution binding precisely
    so the scoring policy is pinned; a second copy of it in this function is a policy that can
    drift without changing a single recorded digest. It nearly did: this used to stringify the
    whole ``judge.model`` mapping into the ``model`` field -- which DeepSeek answers with a 400,
    which is fail-fast, which makes every task ``UNAVAILABLE``, which the report shows as
    ``accuracy: null``. That reads as "grading has not run yet", not as "grading ran and every
    call was rejected".
    """
    from .bench.bcplus.grader import Grader, validate_verdict
    from .bench.grading.judge_client import DeepSeekJudge
    from .campaign.session import provider_client_for
    from .runtime.provider_server import PROVIDER_KEY_PLACEHOLDER

    client = provider_client_for(settings, "evaluator")
    model = settings.judge_model()
    sampling = settings.judge_sampling()
    retries = settings.judge_max_retries()
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_REPORT"),
        model, PROVIDER_KEY_PLACEHOLDER, max_retries=retries, sampling=sampling)

    def judge_fn(system: str, user: str):
        response = asyncio.run(judge.judge(system, user, validate=validate_verdict))
        return response.data

    return Grader(judge_fn), {"model": model, "op_class": "JUDGE_REPORT",
                              "max_retries": retries, "sampling": sampling.content()}


@app.command("bcplus-competence")
def bcplus_competence(
    config: Path = _CFG,
    report: Path = typer.Option(..., "--report", help="reports/BCPLUS_<pilot run>.json"),
    arm: str = typer.Option("P0", "--arm"),
) -> None:
    """Steward-only: decide protocol §5.3's competence gate and, on PASS, make the freeze effective.

    The gate is read out of the frozen prereg, never restated here: a threshold written in two
    places is a threshold that will eventually differ in two places, and the whole point of the
    hash lock is that the number the verdict used is the number that was registered.

    On FAIL this stops. It does not climb encoder sizes, widen top-k, or lower the floor -- each
    of those is a change to the frozen object made after seeing agent data, which is a declared
    deviation and a human's decision. The report says so and exits non-zero.
    """
    from .config import load_config
    from .retrieval.freeze import load_freeze, write_freeze

    _require_role("steward")
    body = json.loads(Path(report).read_text(encoding="utf-8"))
    arms = body.get("arms") or {}
    if arm not in arms:
        _fail(f"{report} has no {arm!r} arm; the competence gate is a P0 measurement")
    measured = arms[arm]
    accuracy = measured.get("accuracy")
    recall = measured.get("evidence_recall_mean")
    # Read straight out of prereg.yaml, the way protocol.py hashes it, rather than through
    # Settings: adding prereg to Settings' config set would change the shas the binding is
    # computed from and invalidate the approval for a reason that has nothing to do with the
    # experiment.
    prereg, prereg_sha = load_config(_REPO / "configs" / "prereg.yaml")
    competence = prereg["pilots"]["competence"]
    criteria = [
        {"name": "p0_accuracy", "measured": accuracy,
         "floor": float(competence["accuracy_floor"]), "n": measured.get("accuracy_n")},
        # Against ``evidence_docids``, not ``gold_docids``. The prereg says "agent-level gold
        # evidence recall", which reads both ways; evidence is the larger set and therefore the
        # stricter floor (the freeze measured 0.4831 evidence vs 0.5654 gold at Recall@100), so
        # this is the conservative reading. Recorded in the gate file so the choice is on the
        # record instead of inside a code path.
        {"name": "agent_evidence_recall", "measured": recall,
         "floor": float(competence["evidence_recall_floor"]), "n": measured.get("recall_n"),
         "floor_applies_to": "evidence_docids",
         "also_measured_gold_recall": measured.get("gold_recall_mean")},
    ]
    for criterion in criteria:
        value = criterion["measured"]
        criterion["status"] = (
            "INPUTS_UNAVAILABLE" if value is None
            else "PASS" if float(value) >= criterion["floor"] else "FAIL")
        typer.echo(f"  {criterion['status']:18}  {criterion['name']}: "
                   f"{value} vs floor {criterion['floor']} (n={criterion['n']})")

    verdict = ("FAIL" if any(c["status"] == "FAIL" for c in criteria)
               else "INPUTS_UNAVAILABLE" if any(c["status"] != "PASS" for c in criteria)
               else "PASS")
    gate = {
        "gate": "RETRIEVAL_COMPETENCE",
        "verdict": verdict,
        "criteria": criteria,
        "arm": arm,
        "source_report": str(report),
        "source_sha256": _content_sha(body),
        "prereg_sha256": prereg_sha,
        "decided_at_utc": _now(),
    }
    out = _REPO / "reports" / "gates" / "RETRIEVAL_COMPETENCE.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    if verdict != "PASS":
        _fail(f"competence gate: {verdict} -- see {out}. Stopping. Changing the encoder, the "
              "top-k or the floor now would be a change to the frozen retriever made after "
              "seeing agent data; that is a declared deviation and needs a human.", code=1)

    freeze_path = _REPO / "protocol" / "retrieval_freeze.json"
    current = load_freeze(freeze_path)
    if current.effective:
        typer.echo(f"freeze already effective after {current.effective_after[:12]}")
        return
    from dataclasses import replace as _replace

    effective = _replace(current, effective_after=gate["source_sha256"])
    # The freeze is write-once and effective_after is inside its digest, so the validated
    # retriever is a *different object* from the unvalidated one rather than an annotation on it.
    # The old file is superseded by rename, which is how every other write-once artifact here
    # records that it was replaced instead of edited.
    superseded = freeze_path.parent / f"superseded_retrieval_freeze_{current.digest[:12]}.json"
    freeze_path.rename(superseded)
    write_freeze(effective, freeze_path)
    typer.echo(f"competence gate: PASS -- retrieval freeze is now effective after "
               f"{gate['source_sha256'][:12]} (was {current.digest[:12]}, kept at {superseded})")
    typer.echo("the execution binding has changed: re-run freeze-approval before the campaign.")


def _has(settings, config: str, *keys: str) -> bool:
    try:
        settings.get(config, *keys)
        return True
    except Exception:  # noqa: BLE001 - a missing key is the answer, not a failure
        return False


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
