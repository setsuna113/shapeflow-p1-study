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
    assert {"protocol_sha", "decision_thresholds_sha", "budget_sha", "variants_sha",
            "stack_sha", "stack_manifest_sha", "vendor_commit", "patched_tree_sha",
            "approved_commit"} <= covered
    assert binding.vendor_commit == "408da442a661ea5e40a6163329f82e3f22628949"
    assert binding.patched_tree_sha


def test_a_missing_approval_is_an_error_not_a_default():
    with pytest.raises(ApprovalError, match="nothing authorises"):
        verify_approval_file(REPO, REPO / "protocol" / "does_not_exist.json")


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
