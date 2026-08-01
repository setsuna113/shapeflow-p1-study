"""The actual materialized supervisor node refines every ConductResearch child."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "tests" / "integration" / "child_identity_probe.py"
PRISTINE = REPO / ".build" / "odr-pristine" / "src"
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"

pytestmark = pytest.mark.skipif(
    not (PRISTINE.exists() and PATCHED.exists()),
    reason="ODR trees not materialized",
)


def _probe(root: Path, mode: str, scenario: str) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{root}:{REPO / 'src'}"
    env["PYTHONHASHSEED"] = "0"
    env["TZ"] = "UTC"
    completed = subprocess.run(
        [
            sys.executable,
            str(PROBE),
            "--mode",
            mode,
            "--scenario",
            scenario,
        ],
        env=env,
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert "<<<TRACE>>>" in completed.stdout, completed.stderr[-4000:]
    return json.loads(completed.stdout.split("<<<TRACE>>>", 1)[1])


def test_sequential_children_across_supervisor_iterations_do_not_alias():
    vendor = _probe(PRISTINE, "hooks-off", "sequential")
    hooks_off = _probe(PATCHED, "hooks-off", "sequential")
    bound = _probe(PATCHED, "bound", "sequential")
    assert vendor["commands"] == hooks_off["commands"] == bound["commands"]

    children = bound["children"]
    assert {tuple(child["coordinate"]) for child in children} == {
        (1, 0, "conduct-a"),
        (2, 0, "conduct-b"),
    }
    assert len({child["researcher_id"] for child in children}) == 2
    assert len({child["handle"] for child in children}) == 2
    assert all(child["markers"] == [f"marker:{child['topic']}"] for child in children)
    assert bound["parent_sidecar"] == {}


def test_two_concurrent_children_have_isolated_context_and_handles():
    hooks_off = _probe(PATCHED, "hooks-off", "concurrent")
    bound = _probe(PATCHED, "bound", "concurrent")
    assert hooks_off["commands"] == bound["commands"]

    children = bound["children"]
    assert {tuple(child["coordinate"]) for child in children} == {
        (7, 0, "conduct-a"),
        (7, 1, "conduct-b"),
    }
    assert len({child["researcher_id"] for child in children}) == 2
    assert len({child["handle"] for child in children}) == 2
    assert all(child["markers"] == [f"marker:{child['topic']}"] for child in children)
    assert bound["parent_sidecar"] == {}
