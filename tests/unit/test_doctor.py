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


def test_a_skip_is_not_a_pass():
    """`doctor: ok` was obtainable on a machine with no GPU, no vLLM and no model.

    Every runtime check skipped, `ok` accepted anything that was not FAIL, and the absence of
    evidence was read as evidence of correctness. A check that could not run has not been
    satisfied.
    """
    from shapeflow_p1.doctor import SKIP, CheckResult, DoctorReport

    report = DoctorReport()
    report.add(CheckResult("a", PASS, ""))
    report.add(CheckResult("b", SKIP, "no GPU here"))
    assert not report.ok
    assert [c.name for c in report.skipped] == ["b"]


def test_an_empty_report_is_not_green():
    from shapeflow_p1.doctor import DoctorReport

    assert not DoctorReport().ok


def test_identity_asserts_the_role_instead_of_always_passing():
    """check_identity returned PASS unconditionally -- it recorded a uid and asserted nothing.

    The provider/runner/evaluator separation (who may read credentials, who may read the answer
    key) was documented but unenforced.
    """
    from shapeflow_p1.doctor import SKIP, check_identity

    # No role asserted -> SKIP, which no longer counts as green.
    assert check_identity().status == SKIP
    # A role this process is definitely not running as must FAIL.
    assert check_identity("runner").status == FAIL
    assert check_identity("no-such-role").status == FAIL


def test_stack_manifest_is_required_and_never_self_frozen():
    """Doctor compares against a stewarded manifest; it must not create one.

    A doctor that froze what it observed at launch would launder an already-drifted engine,
    driver or model into a legitimate baseline.
    """
    from shapeflow_p1.doctor import check_stack_manifest

    res = check_stack_manifest(REPO, REPO / "configs" / "stack.yaml")
    assert res.status == FAIL
    assert "freeze-stack" in res.detail
    assert not (REPO / "protocol" / "stack_manifest.json").exists(), (
        "checking the manifest must not have created it"
    )


def test_a_present_manifest_is_not_evidence_until_the_verifier_exists(tmp_path, monkeypatch):
    """A hand-written manifest with arbitrary values satisfied the check.

    Presence of a JSON file with the expected keys is not verification. Until the digest,
    schema and every live value are compared against the running host, the honest status is
    "not built" -- a gate that cannot fail is not a gate.
    """
    import json
    import shutil

    from shapeflow_p1.doctor import check_stack_manifest

    fake_repo = tmp_path / "repo"
    (fake_repo / "protocol").mkdir(parents=True)
    (fake_repo / "configs").mkdir()
    shutil.copy(REPO / "configs" / "stack.yaml", fake_repo / "configs" / "stack.yaml")
    (fake_repo / "protocol" / "stack_manifest.json").write_text(json.dumps({
        "resolved": {
            "model.artifact_merkle_root": "lol",
            "engine.vllm_package_tree_sha256": "nope",
            "engine.attention_backend": "whatever",
        }
    }))
    res = check_stack_manifest(fake_repo, fake_repo / "configs" / "stack.yaml")
    assert res.status == FAIL
    assert "NOT_IMPLEMENTED" in res.detail


def test_verify_approval_requires_an_external_protocol_sha():
    """Comparing the approval's protocol_sha to itself always passes and proves nothing."""
    from typer.testing import CliRunner

    from shapeflow_p1.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app, ["verify-approval", "--approval", "protocol/launch_approval.json"],
        env={"SHAPEFLOW_PROTOCOL_SHA": ""},
    )
    assert result.exit_code == 1
    assert "SHAPEFLOW_PROTOCOL_SHA" in result.output


def test_verify_approval_hashes_config_contents_not_the_path(monkeypatch):
    """`config_sha(Path(...))` raised CanonicalizationError, so the gate could never run."""
    from shapeflow_p1.config import config_sha, load_config

    data, sha = load_config(REPO / "configs" / "decision.yaml")
    assert config_sha(data) == sha


def test_launch_gate_phase_commands_exit_3_and_accept_the_gates_options():
    """A missing option must never be mistaken for a missing capability.

    bootstrap_and_run.sh calls these with --config. Before they accepted it, the gate died on
    Typer's "No such option: --config" -- an argument-parsing error dressed as a stack failure.
    """
    from typer.testing import CliRunner

    from shapeflow_p1.cli import app

    runner = CliRunner()
    for command in ("prepare", "smoke", "accept", "freeze-stack", "test-p0-parity",
                    "run-screen", "run-holdout", "release-holdout", "acquire"):
        result = runner.invoke(app, [command, "--config", "configs/week1.yaml"])
        assert result.exit_code == 3, f"{command}: expected exit 3, got {result.exit_code}"
        assert "not implemented" in result.output.lower() or "not implemented" in str(result.stderr)
    # run-week1 additionally takes the flags the gate execs it with.
    result = runner.invoke(app, ["run-week1", "--config", "configs/week1.yaml",
                                 "--resume", "--protocol-sha", "deadbeef"])
    assert result.exit_code == 3


def test_cli_app_constructs_and_has_commands():
    from shapeflow_p1.cli import app

    # Typer app exposes the declared commands.
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    assert "doctor" in " ".join(str(n) for n in names) or any(
        "doctor" == (c.name or getattr(c.callback, "__name__", "")) for c in app.registered_commands
    )
