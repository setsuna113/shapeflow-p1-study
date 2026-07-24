"""Doctor pure checks and CLI wiring."""

from __future__ import annotations

from pathlib import Path

from shapeflow_p1.doctor import (
    FAIL,
    PASS,
    check_configs,
    check_schemas_closed,
    check_secret_present,
    run_pure_checks,
)

REPO = Path(__file__).resolve().parents[2]


def test_real_schemas_are_closed():
    res = check_schemas_closed(REPO / "schemas")
    assert res.status == PASS, res.detail


def test_real_configs_load_and_hash():
    res = check_configs({
        "decision": REPO / "configs" / "decision.yaml",
        "budget": REPO / "configs" / "budget_v1.yaml",
    })
    assert res.status == PASS, res.detail


def test_secret_present_from_file_without_leaking(tmp_path, monkeypatch):
    key_file = tmp_path / "k.key"
    key_file.write_text("tvly-FAKEFAKEFAKEFAKEFAKE")
    monkeypatch.setenv("TAVILY_API_KEY_FILE", str(key_file))
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    res = check_secret_present("tavily", env_var="TAVILY_API_KEY", file_env_var="TAVILY_API_KEY_FILE")
    assert res.status == PASS
    # the detail reports presence + a fingerprint, never the value
    assert "tvly-FAKEFAKEFAKEFAKEFAKE" not in res.detail
    assert "fp" in res.detail


def test_secret_absent_fails(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY_FILE", raising=False)
    res = check_secret_present("tavily", env_var="TAVILY_API_KEY", file_env_var="TAVILY_API_KEY_FILE")
    assert res.status == FAIL


def test_run_pure_checks_report_structure(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    monkeypatch.delenv("TAVILY_API_KEY_FILE", raising=False)
    report = run_pure_checks(
        repo=REPO,
        configs={"decision": REPO / "configs" / "decision.yaml"},
        schema_dir=REPO / "schemas",
    )
    names = {c.name for c in report.checks}
    assert "configs" in names and "schemas" in names and "identity" in names
    # with no secrets set, the report is not ok (fail-closed)
    assert not report.ok


def test_cli_app_constructs_and_has_commands():
    from shapeflow_p1.cli import app

    # Typer app exposes the declared commands.
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    assert "doctor" in " ".join(str(n) for n in names) or any(
        "doctor" == (c.name or getattr(c.callback, "__name__", "")) for c in app.registered_commands
    )
