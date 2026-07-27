"""The approval must bind everything a result depends on, and the protocol SHA must be a fact."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.protocol import (
    PROTOCOL_DOCUMENT,
    ApprovalError,
    compute_binding,
    protocol_sha,
    verify_approval_file,
    verified_execution_binding,
)

REPO = Path(__file__).resolve().parents[2]


def test_the_protocol_document_is_tracked():
    """A protocol that lives outside the repository cannot be hashed.

    The SHA used to come from SHAPEFLOW_PROTOCOL_SHA, so the launch gate compared a number the
    caller supplied against a number the caller supplied. Thresholds guarded by a check that
    cannot fail are not pre-registered.
    """
    assert (REPO / PROTOCOL_DOCUMENT).exists()
    assert len(protocol_sha(REPO)) == 64


def test_protocol_sha_follows_the_document():
    original = (REPO / PROTOCOL_DOCUMENT).read_bytes()
    before = protocol_sha(REPO)
    try:
        (REPO / PROTOCOL_DOCUMENT).write_bytes(original + b"\n<!-- edit -->\n")
        assert protocol_sha(REPO) != before
    finally:
        (REPO / PROTOCOL_DOCUMENT).write_bytes(original)
    assert protocol_sha(REPO) == before


def test_the_binding_covers_every_input_that_changes_the_result():
    """Pinning protocol/budget/thresholds alone left the rest free to move.

    The variant registry could gain an arm, the stack manifest could be replaced, the patch
    could change what the graph does -- all under an approval that still verified.
    """
    binding = compute_binding(REPO)
    covered = set(binding.content())
    assert {"protocol_sha", "budget_sha", "variants_sha", "retrieval_sha", "judge_sha",
            "week1_sha", "stack_sha", "stack_manifest_sha", "vendor_commit",
            "patched_tree_sha", "approved_commit"} <= covered
    assert binding.vendor_commit == "408da442a661ea5e40a6163329f82e3f22628949"
    assert binding.patched_tree_sha


def test_a_missing_approval_is_an_error_not_a_default(tmp_path):
    with pytest.raises(ApprovalError, match="nothing authorises"):
        verify_approval_file(REPO, tmp_path / "does_not_exist.json")


def test_an_approval_that_pins_a_stale_variant_registry_is_rejected(tmp_path):
    binding = compute_binding(REPO)
    pinned = binding.content()
    pinned["variants_sha"] = "0" * 64
    approval = tmp_path / "launch_approval.json"
    approval.write_text(json.dumps({
        "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
        "binding": pinned,
        "binding_sha256": binding.digest,
    }))
    with pytest.raises(ApprovalError, match="variants_sha"):
        verify_approval_file(REPO, approval)


# --- the approval must be able to fail -------------------------------------------------------


def test_the_vendor_pin_comes_from_the_gitlink_not_the_submodule_head():
    """The gitlink is what this repository pins; the submodule's HEAD is wherever someone
    left it. Reading the latter makes drift undetectable by construction."""
    from shapeflow_p1.protocol import read_vendor_pin

    pin = read_vendor_pin(REPO)
    assert len(pin) == 40 and all(c in "0123456789abcdef" for c in pin)


def test_an_unobservable_vendor_pin_is_fatal_rather_than_empty(tmp_path):
    """An empty string used to be written into the approval and compared against itself."""
    import pytest

    from shapeflow_p1.protocol import VendorCommitUnobservable, read_vendor_pin

    with pytest.raises(VendorCommitUnobservable):
        read_vendor_pin(tmp_path)


def test_an_approval_that_names_a_different_commit_is_refused(tmp_path):
    import json

    import pytest

    from shapeflow_p1.protocol import (
        ApprovalError,
        compute_binding,
        read_head_commit,
        verify_approval_file,
    )

    head = read_head_commit(REPO)
    binding = compute_binding(REPO, approved_commit="0" * 40)
    approval = tmp_path / "launch_approval.json"
    approval.write_text(json.dumps({
        "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
        "approved_commit": "0" * 40,
        "binding": binding.content(),
        "binding_sha256": binding.digest,
    }), encoding="utf-8")

    with pytest.raises(ApprovalError) as excinfo:
        verify_approval_file(REPO, approval)
    assert "approved_commit" in str(excinfo.value)
    assert head[:12] in str(excinfo.value)


def test_an_approval_bound_to_the_live_head_verifies(tmp_path):
    import json

    from shapeflow_p1.protocol import compute_binding, read_head_commit, verify_approval_file

    binding = compute_binding(REPO, approved_commit=read_head_commit(REPO))
    approval = tmp_path / "launch_approval.json"
    approval.write_text(json.dumps({
        "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
        "approved_commit": binding.approved_commit,
        "binding": binding.content(),
        "binding_sha256": binding.digest,
    }), encoding="utf-8")
    assert verify_approval_file(REPO, approval).digest == binding.digest


def test_launcher_claim_must_equal_the_independently_verified_binding(tmp_path, monkeypatch):
    import shapeflow_p1.protocol as protocol_module

    from shapeflow_p1.protocol import read_head_commit

    monkeypatch.setattr(protocol_module, "_require_clean_execution_tree", lambda _repo: None)
    binding = compute_binding(REPO, approved_commit=read_head_commit(REPO))
    approval = tmp_path / "launch_approval.json"
    approval.write_text(json.dumps({
        "approval_mode": "USER_EXPLICIT_AUTO_LAUNCH",
        "approved_commit": binding.approved_commit,
        "binding": binding.content(),
        "binding_sha256": binding.digest,
    }), encoding="utf-8")
    assert verified_execution_binding(
        REPO, expected_digest=binding.digest, approval_path=approval
    ).digest == binding.digest
    with pytest.raises(ApprovalError, match="execution binding mismatch"):
        verified_execution_binding(
            REPO, expected_digest="0" * 64, approval_path=approval)


def test_execution_binding_refuses_dirty_source_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import shapeflow_p1.protocol as protocol_module

    monkeypatch.setattr(
        protocol_module,
        "_git",
        lambda *_args: SimpleNamespace(
            returncode=0, stdout=" M src/shapeflow_p1/campaign/runner.py\n", stderr=""),
    )
    with pytest.raises(ApprovalError, match="execution tree has tracked or untracked changes"):
        protocol_module._require_clean_execution_tree(tmp_path)


def test_recording_an_approval_appends_rather_than_overwrites(tmp_path, monkeypatch):
    """One mutable path can only show the current answer. What a past run was checked
    against then has no artifact behind it."""
    import shutil

    from shapeflow_p1 import protocol
    from shapeflow_p1.protocol import approval_chain, write_approval_file

    repo = tmp_path / "repo"
    (repo / "protocol").mkdir(parents=True)
    shutil.copytree(REPO / "configs", repo / "configs")
    shutil.copy(REPO / PROTOCOL_DOCUMENT, repo / PROTOCOL_DOCUMENT)
    # git state is exercised by its own tests above; here the subject is the chain.
    monkeypatch.setattr(protocol, "read_vendor_pin", lambda _repo: "a" * 40)
    monkeypatch.setattr(protocol, "read_head_commit", lambda _repo: "b" * 40)
    monkeypatch.setattr(protocol, "_require_clean_execution_tree", lambda _repo: None)
    approval = tmp_path / "approval-store" / "launch_approval.json"
    monkeypatch.setenv("SHAPEFLOW_APPROVAL_FILE", str(approval))

    first = write_approval_file(repo, approved_at_utc="2026-07-24T00:00:00Z")
    week1 = repo / "configs" / "week1.yaml"
    week1.write_text(
        week1.read_text(encoding="utf-8").replace(
            "id: week1_formative_v1", "id: week1_formative_v2"),
        encoding="utf-8")
    second = write_approval_file(repo, approved_at_utc="2026-07-25T00:00:00Z")

    assert first.digest != second.digest
    chain = approval_chain(repo)
    assert [c["binding_sha256"] for c in chain] == [first.digest, second.digest]
    history = approval.parent / "history"
    assert all((history / c["file"]).exists() for c in chain)
    # The first approval is still readable after the second is recorded.
    assert json.loads(
        (history / chain[0]["file"]).read_text(encoding="utf-8")
    )["binding_sha256"] == first.digest


def test_external_approval_has_a_non_self_referential_live_path(tmp_path, monkeypatch):
    """Approve clean code HEAD, write elsewhere, then verify that exact clean tree."""
    import shutil
    import subprocess

    from shapeflow_p1 import protocol
    from shapeflow_p1.protocol import verified_execution_binding, write_approval_file

    repo = tmp_path / "repo"
    (repo / "protocol").mkdir(parents=True)
    (repo / "patches").mkdir()
    shutil.copytree(REPO / "configs", repo / "configs")
    shutil.copy(REPO / PROTOCOL_DOCUMENT, repo / PROTOCOL_DOCUMENT)
    shutil.copy(REPO / "protocol" / "stack_manifest.json",
                repo / "protocol" / "stack_manifest.json")
    shutil.copy(REPO / "patches" / "patched_tree.sha256",
                repo / "patches" / "patched_tree.sha256")
    (repo / "code.py").write_text("VALUE = 1\n", encoding="utf-8")
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "test@example.invalid"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "ShapeFlow Test"],
                   check=True)
    subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "approved code"], check=True)

    approval = tmp_path / "external-approvals" / "launch_approval.json"
    monkeypatch.setenv("SHAPEFLOW_APPROVAL_FILE", str(approval))
    monkeypatch.setattr(protocol, "read_vendor_pin", lambda _repo: "a" * 40)
    binding = write_approval_file(repo, approved_at_utc="2026-07-25T00:00:00Z")

    assert approval.is_file()
    assert subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain"],
        check=True, capture_output=True, text=True,
    ).stdout == ""
    assert verified_execution_binding(repo).digest == binding.digest

    # Code bytes differing from approved HEAD fail even if no bound config changed.
    (repo / "code.py").write_text("VALUE = 2\n", encoding="utf-8")
    with pytest.raises(ApprovalError, match="execution tree has tracked or untracked changes"):
        verified_execution_binding(repo)
    (repo / "code.py").write_text("VALUE = 1\n", encoding="utf-8")

    # A clean, newly committed config is still a different execution and cannot use the old
    # external approval.
    week1 = repo / "configs" / "week1.yaml"
    week1.write_text(
        week1.read_text(encoding="utf-8").replace(
            "id: week1_formative_v1", "id: week1_formative_v2"),
        encoding="utf-8",
    )
    subprocess.run(["git", "-C", str(repo), "add", "configs/week1.yaml"], check=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-qm", "change config"], check=True)
    with pytest.raises(ApprovalError, match="approval does not bind"):
        verified_execution_binding(repo)


def test_approval_artifact_inside_repo_is_refused(tmp_path):
    from shapeflow_p1.protocol import verify_approval_file

    with pytest.raises(ApprovalError, match="self-referential approval"):
        verify_approval_file(REPO, REPO / "protocol" / "launch_approval.json")
