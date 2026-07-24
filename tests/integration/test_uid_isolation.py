"""The separation as a filesystem fact, checked by actually being a different user.

Every other isolation test in this suite runs as one identity and asks what that identity
can reach. That question cannot answer this one. A same-process ``os.listdir`` on the
steward tree returns everything when the steward runs it, nothing when the directory is
absent, and says nothing at all about the runner -- which is the only identity the
invariant is about.

So these tests drop privileges for real. They need root (to become another user) and the
service accounts, so they skip everywhere else; the skip is loud rather than silent,
because a green suite that never ran them is exactly how "separation by uid, not by
convention" turned out to be by convention.
"""

from __future__ import annotations

import json
import os
import pwd
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
ROLES = ("sfrunner", "sfsteward", "sfevaluator")


def _have_roles() -> bool:
    try:
        for role in ROLES:
            pwd.getpwnam(role)
    except KeyError:
        return False
    return True


pytestmark = pytest.mark.skipif(
    os.geteuid() != 0 or not _have_roles(),
    reason="needs root and the sf* service accounts; run on the host after install_host.sh",
)


def _data_root() -> Path:
    return Path(os.environ.get("SHAPEFLOW_DATA_ROOT", "/storage/nvme/shapeflow-data"))


def _as(role: str, *argv: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["runuser", "-u", role, "--", *argv],
        capture_output=True, text=True, timeout=60,
    )


def _require(path: Path) -> Path:
    if not path.exists():
        pytest.skip(f"{path} has not been created yet; run install_host.sh")
    return path


@pytest.mark.parametrize("tree", ["steward", "evaluator"])
def test_the_runner_cannot_read_the_trees_that_hold_the_answer(tree):
    """The steward tree matters as much as the evaluator one.

    Its acquisition manifests carry the audit occurrence graph -- ranks, scores, duplicate
    links -- which AGENTS.md §2 makes evaluator-only, and its permissions were
    ``sfsteward:sfrunner 0750``: r-x for the identity under measurement.
    """
    path = _require(_data_root() / tree)
    listed = _as("sfrunner", "ls", "-A", str(path))
    assert listed.returncode != 0, (
        f"sfrunner listed {path}:\n{listed.stdout[:400]}"
    )


def test_the_runner_cannot_read_a_task_record_or_an_acquisition_manifest():
    """Not just the directory: the files inside it, by absolute path."""
    root = _require(_data_root() / "steward")
    for subdir in ("tasks", "acquisition"):
        directory = root / subdir
        if not directory.exists():
            continue
        for path in sorted(directory.glob("*.json"))[:5]:
            read = _as("sfrunner", "cat", str(path))
            assert read.returncode != 0, f"sfrunner read {path}"


def test_the_evaluator_can_reach_what_it_needs_to_build_truth():
    """Isolation that also blocks the evaluator would just move the problem."""
    tasks = _require(_data_root() / "evaluator" / "tasks")
    listed = _as("sfevaluator", "ls", "-A", str(tasks))
    assert listed.returncode == 0, f"sfevaluator cannot list its own view: {listed.stderr}"

    files = sorted(tasks.glob("*.json"))
    if not files:
        pytest.skip("no evaluator task views yet; run prepare")
    read = _as("sfevaluator", "cat", str(files[0]))
    assert read.returncode == 0, f"sfevaluator cannot read its own view: {read.stderr}"
    body = json.loads(read.stdout)
    assert body["authored_facets"], "the evaluator view carries no facets to score against"
    assert body["original_question"]


def test_the_runner_can_read_the_corpus_it_is_supposed_to_run_on():
    """The runner view is published *into* the runner's tree, and must stay reachable."""
    tasks = _require(_data_root() / "runner" / "frozen_corpus" / "tasks")
    listed = _as("sfrunner", "ls", "-A", str(tasks))
    assert listed.returncode == 0, f"sfrunner cannot list its own corpus: {listed.stderr}"


def test_no_role_but_the_provider_can_read_a_credential():
    cred_dir = Path("/etc/shapeflow")
    keys = sorted(cred_dir.glob("*.key")) if cred_dir.exists() else []
    if not keys:
        pytest.skip("no credentials installed on this host")
    for role in ROLES + ("sfinfer",):
        for key in keys:
            read = _as(role, "cat", str(key))
            assert read.returncode != 0, f"{role} read {key}"
    provider = _as("sfprovider", "cat", str(keys[0]))
    assert provider.returncode == 0, "the provider cannot read its own credential"
