"""The unit templates, checked as configuration rather than as prose.

Two failure modes here are silent by construction, which is why they get tests:

- ``StartLimitIntervalSec``/``StartLimitBurst`` under ``[Service]`` are *ignored* by systemd.
  The unit still parses, still starts, and simply has no crash ceiling -- so a service that
  crash-loops forever looks exactly like one governed by the plan's "3 restarts in 30 min".
- The two vLLM layers differ only in engine flags. Nothing about a running engine announces
  which layer it is, so if both units can be active at once, or if a layer's flags drift from
  ``configs/stack.yaml``, the measurement layers mix and nothing downstream notices.
"""

from __future__ import annotations

import configparser
import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

SYSTEMD = Path(__file__).resolve().parents[2] / "systemd"
STACK = Path(__file__).resolve().parents[2] / "configs" / "stack.yaml"
INSTALL_HOST = Path(__file__).resolve().parents[2] / "scripts" / "install_host.sh"
BOOTSTRAP = Path(__file__).resolve().parents[2] / "scripts" / "bootstrap_and_run.sh"
SUPERVISOR = Path(__file__).resolve().parents[2] / "scripts" / "sfsupervise.sh"
UNITS = sorted(SYSTEMD.glob("*.service.template"))


def _parse(path: Path) -> configparser.ConfigParser:
    # systemd allows repeated keys (e.g. Environment=); collapse them for these assertions.
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read_string(path.read_text(encoding="utf-8"))
    return cp


def _environment(unit: Path) -> str:
    """Every ``Environment=`` value in one string.

    ``_parse`` keeps only the *last* value of a repeated key, so an assertion written against
    ``cp.get("Service", "Environment")`` silently stops checking what it was written to check
    the moment another Environment line is added below it -- it does not fail, it just tests a
    different line. Reading them all is what makes these assertions stable.
    """
    return "\n".join(
        line.split("=", 1)[1].strip()
        for line in unit.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("Environment=")
    )


def test_there_is_at_least_one_unit():
    assert UNITS


def _provider_launchers() -> dict[str, str]:
    """Every place the provider is actually started, by name.

    The unit is not enough on this host: systemd is not PID 1 there, so install_host.sh renders
    the units "for the record, not started" and the real launch goes through sfsupervise --
    which passes no environment of its own. Both paths must inject the same credentials or the
    provider comes up unable to acquire.
    """
    unit = SYSTEMD / "shapeflow-api-provider.service.template"
    start_provider = Path(__file__).resolve().parents[2] / "scripts" / "start_provider.sh"
    return {
        "systemd unit": unit.read_text(encoding="utf-8"),
        "scripts/start_provider.sh": start_provider.read_text(encoding="utf-8"),
    }


@pytest.mark.parametrize("name", sorted(_provider_launchers()))
def test_every_provider_launcher_injects_the_mandatory_exa_credential(name: str):
    """provider_main.build_service calls load_exa_key unconditionally.

    Acquisition moved from Tavily to Exa, but neither the unit nor any start script was
    updated, so a deployed provider raised "no Exa credential" on startup and nothing could be
    acquired. Tavily is deliberately not asserted here: it is optional now, and requiring it
    would re-encode the assumption that broke this.
    """
    body = _provider_launchers()[name]
    assert "EXA_API_KEY_FILE" in body, (
        f"{name} does not inject EXA_API_KEY_FILE; the provider will refuse to start"
    )
    assert "DEEPSEEK_API_KEY_FILE" in body


@pytest.mark.parametrize("name", sorted(_provider_launchers()))
def test_no_provider_launcher_embeds_a_credential_value(name: str):
    """Paths only. A key in a unit file or a start script would land in the repository."""
    body = _provider_launchers()[name]
    for line in body.splitlines():
        if "API_KEY" not in line or line.lstrip().startswith("#"):
            continue
        # Accept `NAME_FILE=<path>` and bare `NAME=` forms; reject an assigned literal value.
        assert "_FILE" in line or line.rstrip().endswith("="), (
            f"{name} appears to assign a credential value directly: {line.strip()!r}"
        )


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_start_limit_directives_live_in_the_unit_section(unit: Path):
    cp = _parse(unit)
    for key in ("StartLimitIntervalSec", "StartLimitBurst"):
        assert not cp.has_option("Service", key), (
            f"{unit.name}: {key} under [Service] is silently ignored by systemd, which removes "
            "the crash ceiling entirely"
        )
    if cp.has_option("Service", "Restart"):
        assert cp.has_option("Unit", "StartLimitBurst"), (
            f"{unit.name}: Restart= without a [Unit] StartLimit* is an unbounded restart loop"
        )
        assert cp.get("Unit", "StartLimitBurst") == "3"
        assert cp.get("Unit", "StartLimitIntervalSec") == "1800"


@pytest.mark.parametrize("unit", UNITS, ids=lambda p: p.name)
def test_units_are_hardened_and_bind_locally(unit: Path):
    cp = _parse(unit)
    assert cp.get("Service", "NoNewPrivileges", fallback="") == "true"
    assert cp.get("Service", "UMask", fallback="") == "0077"


def test_install_host_grants_evaluator_write_only_below_judgments():
    """The evaluator must emit scores without gaining write access to frozen truth."""
    text = INSTALL_HOST.read_text(encoding="utf-8")
    truth_owner = 'chown -R sfsteward:sfevaluator "$DATA_ROOT/evaluator"'
    judgment_owner = (
        'chown -R sfevaluator:sfevaluator "$DATA_ROOT/evaluator/judgments"')
    assert truth_owner in text
    assert judgment_owner in text
    assert text.index(truth_owner) < text.index(judgment_owner)
    assert (
        'find "$DATA_ROOT/evaluator/judgments" -type d -exec chmod 0700 {} +'
        in text
    )
    assert (
        'find "$DATA_ROOT/evaluator/judgments" -type f -exec chmod 0400 {} +'
        in text
    )
    assert 'chmod 0770 "$DATA_ROOT/evaluator"' not in text
    assert "command -v \"$command\"" in text
    assert 'readonly_acl "$DATA_ROOT/runner/$published" sfevaluator' in text
    assert (
        'readonly_acl "$DATA_ROOT/runner/frozen_corpus" sfrunner sfevaluator'
        in text
    )
    assert 'readonly_acl "$DATA_ROOT/steward/acquisition" sfevaluator' in text
    assert 'default_args+=("d:u:$reader:r-x")' in text


def test_the_two_vllm_layers_are_mutually_exclusive():
    """Both engines serve the same port and GPU with different semantics.

    Without Conflicts=, starting one while the other runs leaves the coordinator talking to
    whichever bound first -- so a causal-layer measurement could silently be served by an
    engine with prefix caching on.
    """
    causal = _parse(SYSTEMD / "shapeflow-vllm-causal.service.template")
    operational = _parse(SYSTEMD / "shapeflow-vllm-operational.service.template")
    assert causal.get("Unit", "Conflicts") == "shapeflow-vllm-operational.service"
    assert operational.get("Unit", "Conflicts") == "shapeflow-vllm-causal.service"


def test_causal_unit_matches_the_frozen_causal_isolation_config():
    """The causal engine's flags are the experiment's validity conditions, not tuning."""
    causal = _parse(SYSTEMD / "shapeflow-vllm-causal.service.template")
    exec_start = causal.get("Service", "ExecStart")
    iso = yaml.safe_load(STACK.read_text(encoding="utf-8"))["isolation"]["causal"]
    assert iso["max_num_seqs"] == 1
    assert f"--max-num-seqs {iso['max_num_seqs']}" in exec_start
    assert iso["enable_prefix_caching"] is False
    assert "--no-enable-prefix-caching" in exec_start
    assert "--enable-prefix-caching" not in exec_start.replace("--no-enable-prefix-caching", "")
    assert iso["enable_chunked_prefill"] is False
    assert "--no-enable-chunked-prefill" in exec_start


def test_operational_unit_matches_the_frozen_operational_isolation_config():
    operational = _parse(SYSTEMD / "shapeflow-vllm-operational.service.template")
    exec_start = operational.get("Service", "ExecStart")
    iso = yaml.safe_load(STACK.read_text(encoding="utf-8"))["isolation"]["operational"]
    assert f"--max-num-seqs {iso['max_num_seqs']}" in exec_start
    assert iso["enable_prefix_caching"] is True
    assert "--enable-prefix-caching" in exec_start
    assert "--no-enable-prefix-caching" not in exec_start


@pytest.mark.parametrize(
    "unit",
    [SYSTEMD / "shapeflow-vllm-causal.service.template",
     SYSTEMD / "shapeflow-vllm-operational.service.template"],
    ids=lambda p: p.name,
)
def test_gpu_is_selected_by_uuid_not_index(unit: Path):
    """Indices renumber across driver reloads; a renumbered index moves the campaign silently."""
    text = unit.read_text(encoding="utf-8")
    assert "CUDA_VISIBLE_DEVICES=@GPU_UUID@" in text
    assert "@GPU_INDEX@" not in text


def test_no_unit_still_references_the_removed_single_vllm_service():
    """The old single unit hard-coded APC on for every layer; nothing may depend on it again."""
    assert not (SYSTEMD / "shapeflow-vllm.service.template").exists()
    for unit in UNITS:
        text = unit.read_text(encoding="utf-8")
        assert "shapeflow-vllm.service" not in text, f"{unit.name} references the removed unit"


def test_causal_engine_mints_and_runner_reads_a_per_boot_epoch():
    causal = _parse(SYSTEMD / "shapeflow-vllm-causal.service.template")
    start_pre = causal.get("Service", "ExecStartPre")
    assert "write_engine_epoch.py" in start_pre
    assert "/run/shapeflow-vllm-causal/engine_epoch" in start_pre
    assert "SHAPEFLOW_ENGINE_EPOCH_FILE=/run/shapeflow-vllm-causal/engine_epoch" in (
        _environment(SYSTEMD / "shapeflow-p1-week1.service.template"))


def test_the_coordinator_receives_the_leased_device():
    """The coordinator takes an flock on this UUID; without it the lease never engages.

    Nothing passed SHAPEFLOW_GPU_UUID through -- not this unit, not sfsupervise, not the
    bootstrap privilege-drop helper -- so `gpu_lease` returned None in production and the
    mutual exclusion that stops two workers sharing a card was silently absent.
    """
    assert "SHAPEFLOW_GPU_UUID=" in _environment(
        SYSTEMD / "shapeflow-p1-week1.service.template")
    for launcher in ("sfsupervise.sh", "bootstrap_and_run.sh"):
        body = (SUPERVISOR.parent / launcher).read_text(encoding="utf-8")
        assert "SHAPEFLOW_GPU_UUID" in body, f"{launcher} does not pass the leased device through"


def test_every_screening_lane_receives_the_approved_execution_binding():
    """The binding has to reach whatever actually starts the screen.

    It used to reach a single systemd coordinator, which the bootstrap patched in place. The
    campaign now starts four lane runners from run_lanes.sh, so the assertion follows the launch
    rather than the mechanism it used to go through -- a check pinned to the old path would have
    stayed green while the binding reached nothing.
    """
    install = INSTALL_HOST.read_text(encoding="utf-8")
    bootstrap = BOOTSTRAP.read_text(encoding="utf-8")
    supervisor = SUPERVISOR.read_text(encoding="utf-8")
    lanes = (Path(__file__).resolve().parents[2] / "scripts" / "run_lanes.sh").read_text(
        encoding="utf-8")
    runner_text = (
        SYSTEMD / "shapeflow-p1-week1.service.template").read_text(encoding="utf-8")
    assert "SHAPEFLOW_EXECUTION_BINDING_SHA" in install
    # The bootstrap derives it and hands over; run_lanes.sh derives it again for the launch, so
    # a lane cannot start under a binding nobody verified.
    assert "verified_execution_binding(Path('.')).digest" in bootstrap
    assert 'exec "$REPO/scripts/run_lanes.sh"' in bootstrap
    assert "verified_execution_binding(Path('$REPO')).digest" in lanes
    assert '--protocol-sha "$BINDING"' in lanes
    # Every lane, not just the first: a lane started without it would run unverified.
    assert 'for lane in $(seq 0 $((LANE_COUNT - 1))); do' in lanes
    assert 'SHAPEFLOW_LANE="$lane" setsid /usr/local/bin/sfsupervise' in lanes
    assert "SHAPEFLOW_APPROVAL_FILE=@DATA_ROOT@/approvals/launch_approval.json" in runner_text
    assert "SHAPEFLOW_APPROVAL_FILE=" in supervisor
    assert 'readonly_acl "$DATA_ROOT/approvals" sfrunner sfevaluator' in install


def test_engine_epoch_writer_uses_systemd_invocation_id_and_replaces_on_restart(tmp_path):
    script = Path(__file__).resolve().parents[2] / "scripts" / "write_engine_epoch.py"
    target = tmp_path / "run" / "engine_epoch"
    first = "a" * 32
    second = "b" * 32
    for expected in (first, second):
        subprocess.run(
            [sys.executable, str(script), str(target)],
            check=True,
            env={**os.environ, "INVOCATION_ID": expected},
        )
        assert target.read_text(encoding="ascii").strip() == expected
