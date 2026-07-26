"""Doctor pure checks and CLI wiring."""

from __future__ import annotations

from pathlib import Path

from shapeflow_p1.doctor import (
    FAIL,
    PASS,
    check_configs,
    check_schemas_closed,
    check_credential_isolation,
    check_provider_ready,
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


def _proof(tmp_path, **overrides):
    import json

    body = {
        "denied": {"sfrunner": True, "sfinfer": True, "sfsteward": True,
                   "sfevaluator": True},
        "provider_can_read": True,
    }
    body.update(overrides)
    (tmp_path / "reports").mkdir(exist_ok=True)
    (tmp_path / "reports" / "CREDENTIAL_ISOLATION.json").write_text(
        json.dumps(body), encoding="utf-8")
    return tmp_path


def test_isolation_is_proven_by_failed_reads_not_by_a_successful_one(tmp_path):
    """Doctor no longer opens a key file. The old check could only pass for the identity
    that holds the credential, so `doctor --role runner` -- the one the gate runs -- failed
    by construction, and it proved nobody else could read the secret by reading it."""
    res = check_credential_isolation(_proof(tmp_path))
    assert res.status == PASS


def test_an_identity_that_could_read_a_credential_fails_the_check(tmp_path):
    res = check_credential_isolation(
        _proof(tmp_path, denied={"sfrunner": False, "sfinfer": True,
                                 "sfsteward": True, "sfevaluator": True}))
    assert res.status == FAIL
    assert "sfrunner" in res.detail


def test_an_uncovered_identity_is_not_a_pass(tmp_path):
    res = check_credential_isolation(
        _proof(tmp_path, denied={"sfrunner": True, "sfinfer": True}))
    assert res.status == FAIL
    assert "sfsteward" in res.detail


def test_a_missing_proof_fails_rather_than_skipping(tmp_path):
    assert check_credential_isolation(tmp_path).status == FAIL


def test_a_rejected_upstream_credential_is_a_doctor_failure():
    """Present-and-wrong is what sent 262 requests and received 262 rejections."""
    res = check_provider_ready(lambda: (401, "unauthorized"))
    assert res.status == FAIL
    assert "401" in res.detail


def test_a_ready_provider_passes():
    assert check_provider_ready(lambda: (200, "exa ok, deepseek ok")).status == PASS


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


def test_stack_manifest_is_required_and_never_self_frozen(tmp_path):
    """Doctor compares against a stewarded manifest; it must not create one.

    A doctor that froze what it observed at launch would launder an already-drifted engine,
    driver or model into a legitimate baseline. Run against a fake repo with NO manifest: the
    real repo now carries a committed host freeze, so the missing-manifest path is only reachable
    on a clean tree.
    """
    import shutil

    from shapeflow_p1.doctor import check_stack_manifest

    fake_repo = tmp_path / "repo"
    (fake_repo / "protocol").mkdir(parents=True)
    (fake_repo / "configs").mkdir()
    shutil.copy(REPO / "configs" / "stack.yaml", fake_repo / "configs" / "stack.yaml")

    res = check_stack_manifest(fake_repo, fake_repo / "configs" / "stack.yaml")
    assert res.status == FAIL
    assert "freeze-stack" in res.detail
    assert not (fake_repo / "protocol" / "stack_manifest.json").exists(), (
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
    # The verifier now names what disagreed rather than reporting that it does not exist.
    assert "mismatch" in res.detail
    assert "artifact_merkle_root" in res.detail or "stack.yaml changed" in res.detail


def test_engine_log_defaults_under_the_repo_and_is_overridable(tmp_path, monkeypatch):
    """A log path inside our own repo is safe to default; a missing one is not observable."""
    from shapeflow_p1.doctor import _engine_log

    monkeypatch.delenv("SHAPEFLOW_ENGINE_LOG", raising=False)
    assert _engine_log(tmp_path) is None                       # default absent -> None
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "vllm-causal.log").write_text("x")
    assert _engine_log(tmp_path) == tmp_path / "logs" / "vllm-causal.log"

    other = tmp_path / "elsewhere.log"
    other.write_text("y")
    monkeypatch.setenv("SHAPEFLOW_ENGINE_LOG", str(other))
    assert _engine_log(tmp_path) == other                      # override wins
    monkeypatch.setenv("SHAPEFLOW_ENGINE_LOG", str(tmp_path / "nope.log"))
    assert _engine_log(tmp_path) is None                       # named but missing -> None


def test_stack_check_reads_the_backend_from_the_engine_log(tmp_path, monkeypatch):
    """observe() cannot see the attention backend; the check reads it from the engine's own log,

    the same source the freeze used. So an engine restarted onto a different backend is caught,
    and the gate is not vacuous.
    """
    import shutil

    from shapeflow_p1 import doctor as doc
    from shapeflow_p1.config import load_config
    from shapeflow_p1.ops import live_stack as ls
    from shapeflow_p1.ops.live_stack import _OBSERVED_KEY, StackObservation, freeze_stack

    fake_repo = tmp_path / "repo"
    (fake_repo / "protocol").mkdir(parents=True)
    (fake_repo / "configs").mkdir()
    shutil.copy(REPO / "configs" / "stack.yaml", fake_repo / "configs" / "stack.yaml")
    stack_yaml = fake_repo / "configs" / "stack.yaml"
    declared, _ = load_config(stack_yaml)

    # A full observation that resolves every steward-frozen field, the backend included.
    values: dict[str, str] = {}
    for section, block in declared.items():
        if not isinstance(block, dict):
            continue
        for key, value in block.items():
            if isinstance(value, str) and value.startswith("@"):
                obskey = _OBSERVED_KEY.get(key, key)
                values[obskey] = "FLASH_ATTN" if obskey == "attention_backend" else f"{len(values):064x}"
    from shapeflow_p1.campaign.evaluate import relation_prompt_sha256
    from shapeflow_p1.campaign.truth import truth_prompt_sha256
    from shapeflow_p1.evaluation.atomizer import atomize_protocol_sha256

    values.update({
        "atomize_prompt_sha256": atomize_protocol_sha256(),
        "truth_prompt_sha256": truth_prompt_sha256(),
        "report_prompt_sha256": relation_prompt_sha256(),
    })
    freeze_stack(fake_repo, declared, StackObservation(values=dict(values)),
                 frozen_at_utc="2026-07-24T00:00:00Z")

    # observe() (patched at its source, since check_stack_manifest imports it fresh) returns
    # everything EXCEPT the backend, which the real observe genuinely cannot see.
    seen = {k: v for k, v in values.items() if k != "attention_backend"}
    monkeypatch.setattr(ls, "observe", lambda **kw: StackObservation(values=dict(seen)))
    monkeypatch.delenv("SHAPEFLOW_ENGINE_PID", raising=False)

    # No readable log -> the backend is unobservable and the check fails closed on that field.
    monkeypatch.setenv("SHAPEFLOW_ENGINE_LOG", str(fake_repo / "absent.log"))
    miss = doc.check_stack_manifest(fake_repo, stack_yaml)
    assert miss.status == FAIL and "attention_backend" in miss.detail

    # A log naming the frozen backend -> the check passes.
    log = fake_repo / "engine.log"
    log.write_text("(EngineCore pid=7) INFO Using FLASH_ATTN attention backend out of [...]\n")
    monkeypatch.setenv("SHAPEFLOW_ENGINE_LOG", str(log))
    assert doc.check_stack_manifest(fake_repo, stack_yaml).status == PASS

    # A log showing a different backend -> caught as drift, not laundered into a pass.
    log.write_text("(EngineCore pid=8) INFO Using FLASHINFER attention backend out of [...]\n")
    drift = doc.check_stack_manifest(fake_repo, stack_yaml)
    assert drift.status == FAIL and "attention_backend" in drift.detail


def test_verify_approval_reads_the_protocol_sha_from_the_tracked_document(
    tmp_path, monkeypatch
):
    """Comparing the approval's protocol_sha to itself always passes and proves nothing.

    The SHA is now a fact about a file in the repository, so a caller-supplied value cannot make
    the gate pass. An absent approval is still fatal, and a caller who *claims* a protocol SHA
    that disagrees with the document is fatal too -- that is a disagreement about which
    experiment is being run, not a formatting difference.
    """
    from typer.testing import CliRunner

    from shapeflow_p1.cli import app
    import shapeflow_p1.protocol as protocol_module

    monkeypatch.setattr(protocol_module, "_require_clean_execution_tree", lambda _repo: None)
    runner = CliRunner()
    result = runner.invoke(
        app, ["verify-approval", "--approval", str(tmp_path / "does_not_exist.json")],
        env={"SHAPEFLOW_PROTOCOL_SHA": ""},
    )
    assert result.exit_code == 1
    assert "nothing authorises" in result.output


def test_verify_approval_rejects_a_caller_who_names_a_different_protocol(
    tmp_path, monkeypatch
):
    import json

    from typer.testing import CliRunner

    from shapeflow_p1.cli import app
    import shapeflow_p1.protocol as protocol_module
    from shapeflow_p1.protocol import compute_binding, read_head_commit

    monkeypatch.setattr(protocol_module, "_require_clean_execution_tree", lambda _repo: None)
    # Bound to the live HEAD, so the only thing left to disagree about is the SHA the
    # caller names.
    binding = compute_binding(REPO, approved_commit=read_head_commit(REPO))
    approval = tmp_path / "launch_approval.json"
    approval.write_text(json.dumps({
        "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
        "approved_commit": binding.approved_commit,
        "binding": binding.content(),
        "binding_sha256": binding.digest,
    }))
    result = CliRunner().invoke(
        app, ["verify-approval", "--approval", str(approval)],
        env={"SHAPEFLOW_PROTOCOL_SHA": "0" * 64},
    )
    assert result.exit_code == 1
    assert "disagree about which" in result.output


def test_verify_approval_hashes_config_contents_not_the_path(monkeypatch):
    """`config_sha(Path(...))` raised CanonicalizationError, so the gate could never run."""
    from shapeflow_p1.config import config_sha, load_config

    data, sha = load_config(REPO / "configs" / "decision.yaml")
    assert config_sha(data) == sha


def test_the_launch_gate_commands_accept_the_options_the_gate_passes():
    """A missing option must never be mistaken for a missing capability.

    bootstrap_and_run.sh calls these with --config. Before they accepted it, the gate died on
    Typer's "No such option: --config" -- an argument-parsing error dressed as a stack failure.
    Now they are implemented, so the option must parse and the command must fail for a *real*
    reason (wrong identity, missing corpus), never on argument parsing.
    """
    from typer.testing import CliRunner

    from shapeflow_p1.cli import app

    runner = CliRunner()
    for command in ("prepare", "smoke", "freeze-stack", "run-screen", "release-holdout",
                    "acquire", "build-truth", "evaluate"):
        result = runner.invoke(app, [command, "--config", "configs/week1.yaml"])
        assert result.exit_code != 0, f"{command} succeeded without its preconditions"
        assert "No such option" not in result.output, f"{command} rejected --config"
    result = runner.invoke(app, ["run-week1", "--config", "configs/week1.yaml",
                                 "--resume", "--protocol-sha", "deadbeef"])
    assert "No such option" not in result.output


def test_a_phase_command_refuses_the_wrong_identity():
    """The UID separation is the boundary; a command that only documented it enforced nothing."""
    from typer.testing import CliRunner

    from shapeflow_p1.cli import app

    result = CliRunner().invoke(app, ["prepare", "--config", "configs/week1.yaml"])
    assert result.exit_code == 1
    assert "must run as sfsteward" in result.output


def test_release_holdout_is_a_gate_that_refuses_not_a_stub():
    """This round has no confirmatory holdout, and the command must say which precondition failed."""
    from typer.testing import CliRunner

    from shapeflow_p1.cli import app

    result = CliRunner().invoke(app, ["release-holdout", "--config", "configs/week1.yaml"])
    assert result.exit_code == 1
    assert "FORMATIVE_ONLY" in result.output and "open_holdout" in result.output


def test_cli_app_constructs_and_has_commands():
    from shapeflow_p1.cli import app

    # Typer app exposes the declared commands.
    names = {c.name or c.callback.__name__ for c in app.registered_commands}
    assert "doctor" in " ".join(str(n) for n in names) or any(
        "doctor" == (c.name or getattr(c.callback, "__name__", "")) for c in app.registered_commands
    )
