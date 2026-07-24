"""The `shapeflow-p1` command surface (plan §19).

The commands map one-to-one to campaign phases, and each asserts the identity it must run as:
the steward acquires and freezes, the runner executes treatments, the evaluator reads truth. A
command that checked nothing but a docstring would leave the whole UID separation resting on
whoever typed it.

Once ``reports/LAUNCH_GATE_PASSED.json`` exists, mutation commands accept only
``--resume --protocol-sha <exact>``: a diagnostic override after launch would silently alter an
approved run, and any real change to budget, sample or phase order mints a new protocol SHA and
needs a new approval instead.
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
    "decision": _REPO / "configs" / "decision.yaml",
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
    """After the gate has passed, only `--resume --protocol-sha <exact>` may mutate a run."""
    if not _launched():
        return
    from .protocol import protocol_sha as document_sha

    if not resume:
        _fail("this run has already launched; mutation requires --resume", code=2)
    expected = document_sha(_REPO)
    if protocol_sha != expected:
        _fail(
            f"--protocol-sha must be the exact approved SHA {expected[:12]}...; a budget, "
            "sample or phase change mints a new protocol SHA and needs a new approval",
            code=2,
        )


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
    report.add(check_stack_manifest(_REPO, _REPO / "configs" / "stack.yaml"))
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
    typer.echo("doctor: ok")


@app.command("verify-approval")
def verify_approval(
    approval: Path = typer.Option(..., "--approval", help="protocol/launch_approval.json"),
    config: Path = _CFG,
) -> None:
    """Assert the approval binds the whole live configuration, not just three of its hashes.

    The protocol SHA is read from the tracked document, never from the environment: a value the
    caller supplies and then compares against itself is a gate that cannot fail. A caller who
    *claims* a different SHA is a disagreement about which experiment is running, and fatal.
    """
    from .protocol import ApprovalError, verify_approval_file

    try:
        binding = verify_approval_file(_REPO, approval)
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


@app.command()
def report(
    decision_json: Path = typer.Argument(..., help="Path to a WEEK1_P1_DECISION.json"),
) -> None:
    """Render a decision JSON to Markdown on stdout (both come from one object upstream)."""
    from .analysis.decision import Verdict
    from .analysis.report import DecisionObject, EffectWithCI, NodeDecision, render_markdown

    data = json.loads(decision_json.read_text(encoding="utf-8"))

    def node(key: str) -> NodeDecision:
        n = data[key]
        ws = n.get("work_saving")
        return NodeDecision(
            node=n["node"], verdict=Verdict(n["verdict"]),
            champion_variant=n.get("champion_variant"),
            work_saving=EffectWithCI(**ws) if ws else None,
        )

    decision = DecisionObject(
        webpage_p1=node("WEBPAGE_P1"), c_visible=node("C_VISIBLE"),
        c_registry=node("C_REGISTRY"), h_plus_c_visible=node("H_PLUS_C_VISIBLE"),
        verdict_status=data["verdict_status"],
        confirmatory_power_shortfall=data.get("confirmatory_power_shortfall", False),
        human_audit_status=data.get("human_audit_status", ""),
        protocol_sha=data.get("protocol_sha", ""), freeze_sha=data.get("freeze_sha", ""),
        generated_at_utc=data.get("generated_at_utc", ""),
    )
    typer.echo(render_markdown(decision))


@app.command("freeze-corpus-attempt")
def freeze_corpus_attempt(
    config: Path = _CFG,
    attempt_id: str = typer.Option(..., "--attempt-id", help="e.g. attempt-1"),
    corpus_version: str = typer.Option(..., "--corpus-version"),
    state: str = typer.Option("FAILED_AUTHORING_ATTEMPT", "--state"),
    reason: str = typer.Option("", "--reason"),
    note: str = typer.Option("", "--note"),
) -> None:
    """Close out one round of corpus work without deleting or refunding anything.

    Everything this round spent stays spent and stays on the books. The point is not to
    tidy up -- it is to make the failure a recorded fact, so a later round is visibly a new
    attempt rather than a silent continuation of a corpus nobody can reconstruct. The
    ledger is copied through the backup API first, so the frozen state of the round
    survives whatever happens to the live database next.
    """
    from .experiment.ledger import Ledger
    from .protocol import protocol_sha

    _require_role("provider")
    settings = _settings()
    ledger_path = settings.path("provider_ledger")
    if not ledger_path.exists():
        _fail(f"no provider ledger at {ledger_path}")

    ledger = Ledger(str(ledger_path))
    try:
        snapshot_dir = ledger_path.parent / "ledger_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{attempt_id}-{_now().replace(':', '')}.sqlite"
        ledger.backup_to(str(snapshot))
        os.chmod(snapshot, 0o400)

        ledger.record_corpus_attempt(
            attempt_id=attempt_id, corpus_version=corpus_version, state=state,
            reason=reason, protocol_sha=protocol_sha(_REPO), note=note,
        )
        spend = {
            r["resource"]: {"cap": r["cap"], "reserved": r["reserved_total"],
                            "settled": r["settled_total"]}
            for r in ledger.raw_connection.execute(
                "SELECT resource, cap, reserved_total, settled_total FROM budget_accounts")
        }
        calls = [
            dict(r) for r in ledger.raw_connection.execute(
                "SELECT provider, op_class, state, COUNT(*) n FROM external_call_attempts"
                " a JOIN external_calls c USING(call_id) GROUP BY 1,2,3 ORDER BY 1,2,3")
        ]
    finally:
        ledger.close()

    body = {
        "attempt_id": attempt_id, "corpus_version": corpus_version, "state": state,
        "reason": reason, "note": note, "frozen_at_utc": _now(),
        "ledger_snapshot": str(snapshot), "spend": spend, "attempts_by_outcome": calls,
    }
    out = _REPO / "reports" / f"{state}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"{state} recorded as {attempt_id}; ledger snapshot at {snapshot}")
    for resource, totals in sorted(spend.items()):
        if totals["settled"]:
            typer.echo(f"  kept on the books: {resource} = {totals['settled']}")


# --- steward -----------------------------------------------------------------------------


@app.command("freeze-approval")
def freeze_approval(
    approved_commit: str = typer.Option("", "--approved-commit"),
    approval: Path = typer.Option(None, "--approval"),
) -> None:
    """Steward-only: record the approval for the configuration that is live right now."""
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
    from .campaign.truth import truth_prompt_sha256
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
    observation.values["truth_prompt_sha256"] = truth_prompt_sha256()
    observation.values["atomize_prompt_sha256"] = truth_prompt_sha256()
    observation.values["report_prompt_sha256"] = truth_prompt_sha256()
    try:
        body = do_freeze(_REPO, stack, observation, frozen_at_utc=_now())
    except StackError as e:
        _fail(f"freeze-stack failed: {e}")
    typer.echo(f"stack frozen (manifest={body['manifest_sha256'][:12]}, "
               f"unavailable={len(body['unavailable'])})")


@app.command()
def prepare(config: Path = _CFG,
            protocol_sha: str = typer.Option(None, "--protocol-sha"),
            resume: bool = typer.Option(False, "--resume")) -> None:
    """Steward-only: author, audit, split and seal the task registry."""
    from .campaign.prepare import prepare_corpus
    from .evaluation.judge_client import DeepSeekJudge
    from .providers.provider_client import PROVIDER_KEY_PLACEHOLDER

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("steward")
    settings = _settings()
    client = _provider_client(settings, "steward")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="TASK_AUTHOR", work_key="prepare"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
        sampling=settings.authoring_sampling(),
    )
    result = asyncio.run(prepare_corpus(
        settings, judge=judge, authored_at_utc=_now(),
        target_model=str(settings.get("stack", "model", "repo")),
    ))
    typer.echo(f"sealed {len(result.registry.tasks)} tasks "
               f"(registry={result.registry_sha256[:12]}, "
               f"steward={result.steward_task_count}, runner={result.runner_task_count})")


@app.command()
def acquire(config: Path = _CFG,
            protocol_sha: str = typer.Option(None, "--protocol-sha"),
            resume: bool = typer.Option(False, "--resume")) -> None:
    """Steward-only: call Tavily once per task and freeze the world."""
    from .acquire.tavily_client import TavilyCaptureClient
    from .campaign.acquire import acquire_all, tavily_params_from
    from .providers.provider_client import PROVIDER_KEY_PLACEHOLDER

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("steward")
    settings = _settings()
    client = _provider_client(settings, "steward")
    params = tavily_params_from(settings)

    def factory(task_id: str) -> TavilyCaptureClient:
        return TavilyCaptureClient(
            client.tavily_transport(task_id=task_id), params, PROVIDER_KEY_PLACEHOLDER)

    outcome = asyncio.run(acquire_all(settings, client_factory=factory, fetched_at_utc=_now()))
    typer.echo(f"acquired={outcome.tasks_acquired} skipped={outcome.tasks_skipped} "
               f"queries ok/empty/failed={outcome.queries_ok}/{outcome.queries_empty}/"
               f"{outcome.queries_failed} root={outcome.campaign_sha256[:12]}")
    if outcome.blocked_budget:
        _fail("acquisition stopped on a refused budget reservation; already-frozen worlds kept")


@app.command("build-truth")
def build_truth(config: Path = _CFG) -> None:
    """Evaluator-only: build a TruthPacket per acquired task from the frozen sources."""
    from .campaign.acquire import acquired_task_ids
    from .campaign.truth import build_truth_for_task
    from .evaluation.judge_client import DeepSeekJudge
    from .providers.provider_client import PROVIDER_KEY_PLACEHOLDER

    _require_role("evaluator")
    settings = _settings()
    client = _provider_client(settings, "evaluator")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_TRUTH"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
    )
    steward_tasks = settings.path("tasks")
    built = 0
    for task_id in acquired_task_ids(settings):
        record = json.loads((steward_tasks / f"{task_id}.json").read_text(encoding="utf-8"))
        asyncio.run(build_truth_for_task(
            settings, judge=judge, task_id=task_id,
            question=record["treatment_visible"]["original_question"],
            required_facets=record["acquisition_spec"]["authored_facets"],
        ))
        built += 1
    typer.echo(f"truth packets built: {built}")


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
    settings = _settings()
    body = asyncio.run(run_canary(settings, repo=_REPO, task_limit=tasks))
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


@app.command("run-screen")
def run_screen(config: Path = _CFG,
               resume: bool = typer.Option(False, "--resume"),
               protocol_sha: str = typer.Option(None, "--protocol-sha"),
               max_cells: int = typer.Option(None, "--max-cells")) -> None:
    """Runner-only: causal screening over the FORMATIVE_SCREEN split, in paired blocks."""
    from .campaign.screen import run_screening

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    body = asyncio.run(run_screening(_settings(), repo=_REPO, max_cells=max_cells))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    if not body.get("ok", False):
        raise typer.Exit(code=1)


@app.command("run-week1")
def run_week1(config: Path = _CFG,
              resume: bool = typer.Option(False, "--resume"),
              protocol_sha: str = typer.Option(None, "--protocol-sha")) -> None:
    """Runner-only: the Week-1 campaign. Idempotent; a restart fills gaps."""
    from .campaign.screen import run_screening

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    body = asyncio.run(run_screening(_settings(), repo=_REPO, max_cells=None))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    if not body.get("ok", False):
        raise typer.Exit(code=1)


@app.command()
def evaluate(config: Path = _CFG) -> None:
    """Evaluator-only: score frozen outputs against the truth packets."""
    from .campaign.evaluate import evaluate_frozen

    _require_role("evaluator")
    body = asyncio.run(evaluate_frozen(_settings(), repo=_REPO))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))


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
    ledger = Ledger(str(settings.data_root / str(settings.get("week1", "paths", "runs"))
                        / "ledger.sqlite"))
    phases = PhaseStore(ledger, protocol_sha=settings.shas["week1"])
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
