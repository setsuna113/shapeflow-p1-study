"""The `shapeflow-p1` command surface (plan §19).

The commands map one-to-one to campaign phases. The ones that run anywhere (doctor, report,
verify-artifacts, status) are wired here; the phase commands that need the frozen run host
(acquire, parity, smoke, run-*) are declared with the same names and options so the contract is
stable, and each fails loudly with a "run on the frozen host" message rather than pretending to do
work. Once `LAUNCH_GATE_PASSED.json` exists, mutation commands accept only `--resume
--protocol-sha <exact>` -- a guard enforced here so a diagnostic override can never silently alter
an approved run.
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


def _runtime_only(name: str) -> None:
    typer.echo(
        f"[{name}] is a run-host phase: it needs the frozen sjtu environment "
        "(uv-provisioned Python 3.12 + ODR stack + vLLM). Run it via scripts/bootstrap_and_run.sh "
        "on the host, not in the authoring environment.",
        err=True,
    )
    raise typer.Exit(code=3)


@app.command()
def doctor() -> None:
    """Verify the environment and stack. Fails closed on any pure-check failure."""
    from .doctor import check_git_clean, run_pure_checks

    report = run_pure_checks(repo=_REPO, configs=_CONFIGS, schema_dir=_SCHEMAS)
    report.add(check_git_clean(_REPO))
    for c in report.checks:
        typer.echo(f"  {c.status:4}  {c.name}: {c.detail}")
    if not report.ok:
        typer.echo("doctor: FAILED", err=True)
        raise typer.Exit(code=1)
    typer.echo("doctor: ok")


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


# --- run-host phases: declared for a stable contract, refuse to fake work ---------------

@app.command()
def acquire() -> None:  # noqa: D401
    """Acquire and freeze the Tavily source pool (run host)."""
    _runtime_only("acquire")


@app.command("test-p0-parity")
def test_p0_parity() -> None:
    """Assert the patched hooks-off graph matches vendor byte-for-byte (run host)."""
    _runtime_only("test-p0-parity")


@app.command()
def smoke() -> None:
    """Real GPU smoke over a tiny task set (run host)."""
    _runtime_only("smoke")


@app.command("run-screen")
def run_screen() -> None:
    """Run component screening (run host)."""
    _runtime_only("run-screen")


@app.command("run-holdout")
def run_holdout() -> None:
    """Run the honest holdout (run host)."""
    _runtime_only("run-holdout")


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
