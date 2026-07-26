"""Cross-uid artifacts must not zero the inherited POSIX ACL mask.

The multi-uid layout shares data with default ACLs rather than a shared group (see
``scripts/install_host.sh``). A writer that asks the kernel for an explicitly restrictive
mode folds that mode into the inherited ACL -- ``mask &= (mode >> 3)`` -- so a file created
at 0600 lands with mask ``---`` and every ``u:<role>:r-x`` entry becomes ``#effective:---``.
``getfacl`` still lists the grant, so nothing looks wrong until the other uid gets EACCES.

That is not hypothetical: ``tempfile.mkstemp`` requests 0600 by design, and four writers on
the cross-uid path used it -- the object store (steward-written page bytes the runner reads,
runner-written outputs the evaluator scores), the checkpoint store (the C fork's input), the
acquisition manifests, and the approval file that every ``sfrunner`` command verifies. The
acquisition and approval failures would both have landed *after* money was spent.

These tests assert the mechanism directly rather than dropping privileges, so they run
without root; ``tests/integration/test_uid_isolation.py`` covers the real uid drop. Each one
is paired with a control proving the assertion can fail, because a test for a mask that is
already ``rwx`` by accident would pass against the bug it is meant to catch.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

import pytest

from shapeflow_p1.fsmode import SHARED_READ_MODE, chmod_shared
from shapeflow_p1.object_store import ObjectStore

#: Any name that resolves; the entry only has to exist for the kernel to compute a mask.
ACL_PRINCIPAL = "nobody"


def _acl_supported(directory: Path) -> bool:
    if not (shutil.which("setfacl") and shutil.which("getfacl")):
        return False
    probe = directory / "acl-probe"
    probe.mkdir()
    result = subprocess.run(
        ["setfacl", "-d", "-m", f"u:{ACL_PRINCIPAL}:r-x", str(probe)],
        capture_output=True,
    )
    return result.returncode == 0


@pytest.fixture()
def shared_dir(tmp_path):
    """A directory carrying the same default ACL install_host.sh grants."""
    if not _acl_supported(tmp_path):
        pytest.skip("setfacl/getfacl unavailable or filesystem has no ACL support")
    target = tmp_path / "shared"
    target.mkdir()
    subprocess.run(
        ["setfacl", "-d", "-m", f"u:{ACL_PRINCIPAL}:r-x", str(target)],
        check=True,
        capture_output=True,
    )
    return target


def _mask_of(path: Path) -> str:
    """The file's effective ACL mask, e.g. ``r--`` or ``---``."""
    out = subprocess.run(
        ["getfacl", "-pE", str(path)], check=True, capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if line.startswith("mask::"):
            return line.split("::", 1)[1].strip()
    raise AssertionError(f"no mask entry in the ACL of {path}:\n{out}")


def _named_entry_is_effective(path: Path) -> bool:
    """False when getfacl annotates the named-user grant as ``#effective:---``."""
    out = subprocess.run(
        ["getfacl", "-p", str(path)], check=True, capture_output=True, text=True
    ).stdout
    for line in out.splitlines():
        if line.startswith(f"user:{ACL_PRINCIPAL}:"):
            return "#effective:---" not in line.replace(" ", "")
    raise AssertionError(f"no u:{ACL_PRINCIPAL} entry in the ACL of {path}:\n{out}")


def _assert_readable_by_the_named_role(path: Path) -> None:
    assert _mask_of(path) != "---", (
        f"{path} published with ACL mask '---'; every named-user grant on it is masked out "
        "and the other uid will get EACCES"
    )
    assert "r" in _mask_of(path)
    assert _named_entry_is_effective(path)


# --------------------------------------------------------------------------------------
# The control: prove the assertion above can actually fail.
# --------------------------------------------------------------------------------------


def test_a_mkstemp_write_would_have_zeroed_the_mask(shared_dir):
    """Reproduces the defect, so the assertions below are not vacuous.

    This is exactly what every ``mkstemp`` + ``os.replace`` writer did before the fix.
    """
    fd, tmp = tempfile.mkstemp(dir=str(shared_dir), suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(b"payload")
    published = shared_dir / "written-at-0600"
    os.replace(tmp, published)

    assert _mask_of(published) == "---"
    assert not _named_entry_is_effective(published)
    with pytest.raises(AssertionError, match="masked out"):
        _assert_readable_by_the_named_role(published)


def test_chmod_shared_repairs_the_mask(shared_dir):
    fd, tmp = tempfile.mkstemp(dir=str(shared_dir), suffix=".tmp")
    with os.fdopen(fd, "wb") as handle:
        handle.write(b"payload")
    chmod_shared(tmp)
    published = shared_dir / "written-through-chmod-shared"
    os.replace(tmp, published)

    _assert_readable_by_the_named_role(published)
    assert published.stat().st_mode & 0o777 == SHARED_READ_MODE
    # World access is still denied -- widening the mask is not opening the file up.
    assert not published.stat().st_mode & 0o007


# --------------------------------------------------------------------------------------
# The four real writers.
# --------------------------------------------------------------------------------------


def test_object_store_blobs_stay_readable_across_the_uid_boundary(shared_dir):
    store = ObjectStore(shared_dir)
    ref = store.put_bytes(b"a frozen page the runner has to read")

    blobs = list(shared_dir.rglob("*.zst"))
    assert len(blobs) == 1
    _assert_readable_by_the_named_role(blobs[0])
    # The store's own contract is unaffected by the mode change.
    assert store.get_bytes(ref.key) == b"a frozen page the runner has to read"


def test_checkpoint_store_documents_stay_readable_across_the_uid_boundary(shared_dir):
    from shapeflow_p1.odr.checkpoints import (
        CCheckpoint,
        CheckpointStore,
        EvidenceManifest,
        FrozenMessage,
        SamplingEnvelope,
    )

    checkpoint = CCheckpoint(
        task_id="task-acl",
        researcher_id="researcher-0",
        researcher_messages=(FrozenMessage(role="human", content="find the answer"),),
        evidence_manifest=EvidenceManifest(span_ids=()),
        query_attempt_ids=(),
        close_reason="RESEARCH_COMPLETE",
        sampling=SamplingEnvelope(
            model="Qwen3-14B-AWQ", temperature=0.3, top_p=1.0, max_tokens=4096, seed=1
        ),
    )
    digest = CheckpointStore(shared_dir).put(checkpoint)

    stored = next(shared_dir.rglob(f"{digest}.json"))
    _assert_readable_by_the_named_role(stored)


def test_acquisition_manifests_stay_readable_across_the_uid_boundary(shared_dir):
    from shapeflow_p1.campaign.acquire import _write_json_atomic

    path = shared_dir / "pools" / "task-acl.json"
    _write_json_atomic(path, {"task_id": "task-acl", "sources": []})

    _assert_readable_by_the_named_role(path)
    assert json.loads(path.read_text(encoding="utf-8"))["task_id"] == "task-acl"


def test_approval_artifacts_stay_readable_across_the_uid_boundary(shared_dir):
    from shapeflow_p1.protocol import _write_atomic, _write_new

    pointer = shared_dir / "launch_approval.json"
    _write_atomic(pointer, {"approval_mode": "USER_EXPLICIT_AUTO_LAUNCH"})
    _assert_readable_by_the_named_role(pointer)

    link = shared_dir / "history" / "0001-abcdef123456.json"
    _write_new(link, {"approval_mode": "USER_EXPLICIT_AUTO_LAUNCH"})
    _assert_readable_by_the_named_role(link)
