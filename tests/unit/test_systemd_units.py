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
from pathlib import Path

import pytest
import yaml

SYSTEMD = Path(__file__).resolve().parents[2] / "systemd"
STACK = Path(__file__).resolve().parents[2] / "configs" / "stack.yaml"
UNITS = sorted(SYSTEMD.glob("*.service.template"))


def _parse(path: Path) -> configparser.ConfigParser:
    # systemd allows repeated keys (e.g. Environment=); collapse them for these assertions.
    cp = configparser.ConfigParser(strict=False, interpolation=None)
    cp.optionxform = str
    cp.read_string(path.read_text(encoding="utf-8"))
    return cp


def test_there_is_at_least_one_unit():
    assert UNITS


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
