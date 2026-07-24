"""The `shapeflow-p1` command surface (plan §19).

The commands map one-to-one to campaign phases. Those that run anywhere (doctor,
verify-approval, report, verify-artifacts, status) are wired here. The phase commands that are
not built yet are declared with the exact names and options the launch gate passes, and each
exits 3 with "not implemented" -- so a missing option can never be mistaken for a missing
capability, and an unbuilt phase can never be mistaken for a passing one.

Nothing here approximates a phase it cannot perform. Once `LAUNCH_GATE_PASSED.json` exists,
mutation commands accept only `--resume --protocol-sha <exact>`, so a diagnostic override can
never silently alter an approved run.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

app = typer.Typer(add_completion=False, help="ShapeFlow P1 Week-1 study CLI")

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = {
    "decision": _REPO / "configs" / "decision.yaml",
    "budget": _REPO / "configs" / "budget_v1.yaml",
}
_SCHEMAS = _REPO / "schemas"


_NOT_IMPLEMENTED = {
    "prepare": "Block 4 (frozen corpus + task registry)",
    "smoke": "Block 10 (real GPU canary)",
    "run-screen": "Block 7 (campaign runner)",
    "run-holdout": "Block 7 (campaign runner)",
    "run-week1": "Block 7 (campaign runner)",
    "release-holdout": "Block 8 (holdout gate)",
    "acquire": "Block 4 (Tavily acquisition)",
    "test-p0-parity": "Block 2 (ODR adapter + parity harness)",
    "accept": "Block 10 (acceptance matrix)",
    "freeze-stack": "Block 9 (stewarded stack manifest)",
    "preflight": "Block 7 (campaign preflight)",
}


def _not_implemented(name: str) -> None:
    """Refuse loudly. Never approximate a phase that has not been built.

    Exit 3 is distinct from a check *failing* (exit 1): the launch gate must be able to tell
    "this stack is wrong" from "this command does not exist yet", and neither may ever be
    mistaken for success.
    """
    typer.echo(
        f"[{name}] is not implemented yet -- {_NOT_IMPLEMENTED[name]}. It refuses to run rather "
        "than approximate a phase, so the launch gate cannot pass an incomplete stack.",
        err=True,
    )
    raise typer.Exit(code=3)


@app.command()
def doctor(
    config: Path = typer.Option(None, "--config", help="Campaign config (configs/week1.yaml)"),
    role: str = typer.Option(None, "--role", help="Assert the effective identity for this role"),
) -> None:
    """Verify the environment and stack, read-only. Fails closed on any non-PASS check."""
    from .doctor import check_git_clean, check_stack_manifest, run_pure_checks

    report = run_pure_checks(repo=_REPO, configs=_CONFIGS, schema_dir=_SCHEMAS, role=role)
    report.add(check_git_clean(_REPO))
    report.add(check_stack_manifest(_REPO, _REPO / "configs" / "stack.yaml"))
    if config is not None and not config.exists():
        typer.echo(f"  FAIL  config: {config} not found", err=True)
        raise typer.Exit(code=1)
    for c in report.checks:
        typer.echo(f"  {c.status:4}  {c.name}: {c.detail}")
    if not report.ok:
        skipped = report.skipped
        if skipped:
            typer.echo(
                f"doctor: FAILED -- {len(skipped)} check(s) could not run. A check that did not "
                "run has not been satisfied; it is not a pass.",
                err=True,
            )
        else:
            typer.echo("doctor: FAILED", err=True)
        raise typer.Exit(code=1)
    typer.echo("doctor: ok")


@app.command("verify-approval")
def verify_approval(
    approval: Path = typer.Option(..., "--approval", help="protocol/launch_approval.json"),
    config: Path = typer.Option(None, "--config", help="Campaign config (unused for hashing)"),
) -> None:
    """Assert the approval pins the hashes of the live budget and decision configs.

    The launch gate runs this BEFORE any paid step. Checking only that the file exists, and
    verifying its hashes afterwards, means an edited threshold or budget can already have spent
    Tavily credits and GPU hours by the time the mismatch surfaces.
    """
    import json
    import os

    from .config import load_config
    from .experiment.freeze import ApprovalMismatch, verify_launch_approval

    # protocol_sha is deliberately NOT read from the approval file: comparing a value to itself
    # is self-pinning -- it always passes and proves nothing. It is the SHA of the protocol
    # document, supplied by the caller, and the campaign runner re-checks it.
    protocol_sha = os.environ.get("SHAPEFLOW_PROTOCOL_SHA")
    if not protocol_sha:
        typer.echo(
            "SHAPEFLOW_PROTOCOL_SHA is not set. The approval's protocol_sha must be checked "
            "against the protocol document, not against itself.",
            err=True,
        )
        raise typer.Exit(code=1)
    if not approval.exists():
        typer.echo(f"approval file {approval} not found", err=True)
        raise typer.Exit(code=1)
    try:
        data = json.loads(approval.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as e:
        typer.echo(f"approval file {approval} is not readable JSON: {e}", err=True)
        raise typer.Exit(code=1)
    # config_sha takes the config's parsed contents, not a Path -- passing the Path raised
    # CanonicalizationError, so this "gate" could never have run at all.
    _, live_budget = load_config(_CONFIGS["budget"])
    _, live_thresholds = load_config(_CONFIGS["decision"])
    try:
        verify_launch_approval(
            data,
            protocol_sha=protocol_sha,
            budget_sha=live_budget,
            decision_thresholds_sha=live_thresholds,
        )
    except ApprovalMismatch as e:
        typer.echo(f"approval mismatch: {e}", err=True)
        raise typer.Exit(code=1)
    typer.echo(f"approval ok (mode={data['approval_mode']}, protocol={protocol_sha[:12]})")


@app.command()
def report(decision_json: Path = typer.Argument(..., help="Path to a WEEK1_P1_DECISION.json")) -> None:
    """Render a decision JSON to Markdown on stdout (both come from one object upstream)."""
    import json

    from .analysis.report import render_markdown
    from .analysis.decision import Verdict
    from .analysis.report import DecisionObject, EffectWithCI, NodeDecision

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


@app.command("verify-artifacts")
def verify_artifacts(ledger_path: Path, object_store: Path) -> None:
    """Confirm every committed work item's artifact still verifies in the object store."""
    from .experiment.ledger import Ledger
    from .object_store import ObjectStore

    lg = Ledger(str(ledger_path))
    store = ObjectStore(object_store)
    bad = 0
    counts = lg.state_counts()
    committed = counts.get("COMMITTED", 0)
    rows = lg.raw_connection.execute(
        "SELECT result_object_ref FROM attempts WHERE state='COMMITTED'"
    ).fetchall()
    for r in rows:
        ref = r["result_object_ref"]
        if not (ref and store.verify(ref)):
            bad += 1
    typer.echo(f"committed={committed} verified={committed - bad} corrupt={bad}")
    lg.close()
    if bad:
        raise typer.Exit(code=1)


@app.command()
def status(status_json: Path = typer.Option(None, help="Path to runs/STATUS.json")) -> None:
    """Print the latest STATUS.json if present."""
    path = status_json or (_REPO / "runs" / "STATUS.json")
    if not path.exists():
        typer.echo("no STATUS.json yet")
        raise typer.Exit(code=0)
    typer.echo(path.read_text(encoding="utf-8"))


# --- phases not yet built: declared so the contract is stable, refuse to fake work ------
#
# Each accepts the options the launch gate passes, so a missing flag can never be mistaken for
# a missing capability, and each exits 3. The gate then blocks with an accurate slug instead of
# dying on "No such option: --config".

_CFG = typer.Option(None, "--config", help="Campaign config (configs/week1.yaml)")


@app.command()
def acquire(config: Path = _CFG) -> None:  # noqa: D401
    """Acquire and freeze the Tavily source pool."""
    _not_implemented("acquire")


@app.command("test-p0-parity")
def test_p0_parity(config: Path = _CFG) -> None:
    """Assert the patched hooks-off graph matches vendor byte-for-byte."""
    _not_implemented("test-p0-parity")


@app.command()
def prepare(config: Path = _CFG) -> None:
    """Build the frozen task registry and corpus."""
    _not_implemented("prepare")


@app.command()
def smoke(config: Path = _CFG) -> None:
    """Real GPU smoke over a tiny task set."""
    _not_implemented("smoke")


@app.command()
def preflight(
    config: Path = _CFG,
    approved_protocol_sha: str = typer.Option(None, "--approved-protocol-sha"),
) -> None:
    """Campaign-level preflight against the approved protocol SHA."""
    _not_implemented("preflight")


@app.command()
def accept(config: Path = _CFG) -> None:
    """Run the acceptance matrix and write reports/ACCEPTANCE.json."""
    _not_implemented("accept")


@app.command("freeze-stack")
def freeze_stack(config: Path = _CFG) -> None:
    """Steward-only: resolve every @STEWARD_FREEZES@ field into protocol/stack_manifest.json."""
    _not_implemented("freeze-stack")


@app.command("run-screen")
def run_screen(config: Path = _CFG) -> None:
    """Run component screening."""
    _not_implemented("run-screen")


@app.command("run-week1")
def run_week1(
    config: Path = _CFG,
    resume: bool = typer.Option(False, "--resume"),
    protocol_sha: str = typer.Option(None, "--protocol-sha"),
) -> None:
    """Run the Week-1 campaign."""
    _not_implemented("run-week1")


@app.command("run-holdout")
def run_holdout(config: Path = _CFG) -> None:
    """Run the honest holdout."""
    _not_implemented("run-holdout")


@app.command("release-holdout")
def release_holdout(config: Path = _CFG) -> None:
    """Holdout gate: materialize the holdout corpus for the runner, once."""
    _not_implemented("release-holdout")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
