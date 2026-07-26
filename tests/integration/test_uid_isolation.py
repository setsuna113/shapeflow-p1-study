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
import sys
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


def _create_with_service_umask(role: str, path: Path) -> subprocess.CompletedProcess:
    return _as(role, "sh", "-c", 'umask 077; : > "$1"', "sh", str(path))


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


def test_the_evaluator_can_read_what_it_needs_to_score_truth():
    """The steward writes truth; isolation must still leave the evaluator able to score it."""
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


def test_the_evaluator_can_write_only_its_judgment_subtree():
    """Scoring needs a write capability, but it must not extend to the answer key."""
    evaluator_root = _require(_data_root() / "evaluator")
    judgments = _require(evaluator_root / "judgments")
    marker = judgments / f".uid-isolation-{os.getpid()}"
    answer_key_marker = evaluator_root / "tasks" / f".uid-isolation-{os.getpid()}"
    try:
        created = _as("sfevaluator", "touch", str(marker))
        assert created.returncode == 0, (
            f"sfevaluator cannot write judgments: {created.stderr}")

        runner_read = _as("sfrunner", "cat", str(marker))
        assert runner_read.returncode != 0, "sfrunner read an evaluator judgment"

        mutated_truth = _as("sfevaluator", "touch", str(answer_key_marker))
        assert mutated_truth.returncode != 0, (
            "sfevaluator can mutate the steward-owned answer key")
    finally:
        _as("sfevaluator", "rm", "-f", str(marker))
        if answer_key_marker.exists():
            answer_key_marker.unlink()


def test_new_0077_runtime_artifacts_inherit_the_evaluator_read_only_acl():
    """The real service umask must not erase the post-run evaluator's read capability."""
    root = _data_root()
    created: list[Path] = []
    try:
        for relative in ("runner/runs", "runner/object_store", "runner/checkpoints"):
            directory = _require(root / relative)
            marker = directory / f".uid-isolation-{os.getpid()}"
            created.append(marker)
            made = _create_with_service_umask("sfrunner", marker)
            assert made.returncode == 0, f"sfrunner cannot create {marker}: {made.stderr}"
            assert _as("sfevaluator", "cat", str(marker)).returncode == 0, (
                f"sfevaluator cannot read new 0600-style runner artifact {marker}")
            denied = marker.with_name(marker.name + "-evaluator-write")
            created.append(denied)
            assert _as("sfevaluator", "touch", str(denied)).returncode != 0, (
                f"sfevaluator can mutate runner directory {directory}")

        frozen = _require(root / "runner" / "frozen_corpus")
        publication = frozen / f".uid-isolation-{os.getpid()}"
        created.append(publication)
        assert _create_with_service_umask("sfsteward", publication).returncode == 0
        assert _as("sfrunner", "cat", str(publication)).returncode == 0
        assert _as("sfevaluator", "cat", str(publication)).returncode == 0
        runner_write = publication.with_name(publication.name + "-runner-write")
        created.append(runner_write)
        assert _as("sfrunner", "touch", str(runner_write)).returncode != 0

        acquisition = _require(root / "steward" / "acquisition")
        provenance = acquisition / f".uid-isolation-{os.getpid()}"
        created.append(provenance)
        assert _create_with_service_umask("sfsteward", provenance).returncode == 0
        assert _as("sfevaluator", "cat", str(provenance)).returncode == 0
        assert _as("sfrunner", "cat", str(provenance)).returncode != 0
    finally:
        for path in created:
            if path.exists():
                path.unlink()


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


def test_runner_uses_the_sanitized_canary_api_without_reading_provider_files():
    """The live UID boundary: deny the file, allow only the closed aggregate route."""

    ledger = _require(_data_root() / "provider" / "provider_ledger.sqlite")
    direct = _as("sfrunner", "cat", str(ledger))
    assert direct.returncode != 0, "sfrunner read the provider-owned ledger directly"

    token = _require(Path("/etc/shapeflow-tokens/runner.token"))
    url = os.environ.get(
        "SHAPEFLOW_PROVIDER_URL", "http://127.0.0.1:8787/v1/canary/audit")
    program = (
        "import json,urllib.request;"
        f"token=open({str(token)!r},encoding='utf-8').read().strip();"
        "body=json.dumps({'work_keys':['uid-boundary-probe']}).encode();"
        f"req=urllib.request.Request({url!r},data=body,"
        "headers={'Authorization':'Bearer '+token,'Content-Type':'application/json'});"
        "print(urllib.request.urlopen(req,timeout=10).read().decode())"
    )
    result = _as("sfrunner", sys.executable, "-c", program)
    if result.returncode != 0 and "Connection refused" in result.stderr:
        pytest.skip("live provider is not running")
    assert result.returncode == 0, result.stderr
    attestation = json.loads(result.stdout)
    assert attestation["work_keys"] == ["uid-boundary-probe"]
    assert attestation["work"][0] == {
        "work_key": "uid-boundary-probe",
        "open_attempts": 0,
        "settled_gpu_seconds": 0.0,
        "ops": [],
    }
    dumped = json.dumps(attestation, sort_keys=True)
    for forbidden in (
        "request_object_ref", "response_object_ref", "prompt_sha256",
        "messages", "headers", "credential",
    ):
        assert forbidden not in dumped
