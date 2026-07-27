"""Config hashing and the freeze / launch-approval logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from shapeflow_p1.config import ConfigBundle, ConfigError, config_sha, load_config, require
from shapeflow_p1.experiment.freeze import (
    APPROVAL_MODE_AUTO,
    ApprovalMismatch,
    FreezeRecord,
    build_launch_approval,
    verify_launch_approval,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_real_budget_config_loads_and_hashes():
    data, sha = load_config(CONFIGS / "budget_v1.yaml")
    assert isinstance(data, dict) and len(sha) == 64


def test_budget_caps_are_the_v01_hard_bounds():
    data, _ = load_config(CONFIGS / "budget_v1.yaml")
    assert require(data, "api_budget", "tavily_max_requests") == 500
    # Re-derived 2026-07-26 at verified prices; see configs/budget_v1.yaml. The earlier set was
    # denominated in an invented price snapshot that over-stated spend ~19x. All four move
    # together and are sized to bind at roughly the same point, because leaving any one behind
    # stops the run just as dead having spent the others.
    assert require(data, "api_budget", "deepseek_max_usd") == 400.00
    assert require(data, "api_budget", "deepseek_max_requests") == 150000
    assert require(data, "api_budget", "deepseek_max_output_tokens") == 140000000
    assert require(data, "campaign_budget", "max_gpu_hours") == 150


def test_config_sha_is_canonical_and_stable():
    assert config_sha({"a": 1, "b": 2}) == config_sha({"b": 2, "a": 1})


def test_require_rejects_missing_key_without_defaulting():
    with pytest.raises(ConfigError, match="missing required config key"):
        require({"a": {}}, "a", "b")


def test_bundle_protocol_sha_changes_when_any_config_changes(tmp_path):
    (tmp_path / "x.yaml").write_text("k: 1\n")
    (tmp_path / "y.yaml").write_text("k: 2\n")
    b1 = ConfigBundle.load({"x": tmp_path / "x.yaml", "y": tmp_path / "y.yaml"})
    (tmp_path / "y.yaml").write_text("k: 3\n")
    b2 = ConfigBundle.load({"x": tmp_path / "x.yaml", "y": tmp_path / "y.yaml"})
    assert b1.protocol_sha != b2.protocol_sha


def test_freeze_record_digest_is_deterministic_and_content_addressed():
    rec = FreezeRecord(
        kind="p1_holdout_v1", protocol_sha="p", champion_variant={"variant_id": "H03"},
        thresholds_sha="t", budget_sha="b", holdout_task_ids=("t1", "t2"),
        randomization_schedule_sha="r", software_manifest_sha="s",
        sample_size_plan={"n_plan": 48},
    )
    assert rec.digest == rec.digest
    # A different champion => a different freeze.
    rec2 = FreezeRecord(
        kind="p1_holdout_v1", protocol_sha="p", champion_variant={"variant_id": "H02"},
        thresholds_sha="t", budget_sha="b", holdout_task_ids=("t1", "t2"),
        randomization_schedule_sha="r", software_manifest_sha="s",
        sample_size_plan={"n_plan": 48},
    )
    assert rec.digest != rec2.digest


def test_launch_approval_is_auto_launch_per_protocol_v01():
    approval = build_launch_approval(
        protocol_sha="p", budget_sha="b", decision_thresholds_sha="d",
        approved_at_utc="2026-07-24T18:00:00Z",
    )
    # Protocol v0.1 sections 0 and 4.1: the user's 2026-07-24 "build it and start running"
    # instruction IS the launch authorization. Gates still fail closed.
    assert approval["approval_mode"] == APPROVAL_MODE_AUTO
    # No "requires a human to launch" flag may exist -- inventing one would encode a decision
    # the user never made.
    assert "requires_human_launch" not in approval


def test_launch_approval_refuses_an_unauthorized_mode():
    """A mode the protocol does not authorize is a hard error, not a new policy.

    This is the regression guard for the fabricated USER_EXPLICIT_GATE_GREEN_THEN_PAUSE mode:
    a coding agent may materialize and hash the protocol's values, never choose different ones.
    """
    with pytest.raises(ApprovalMismatch, match="not authorized"):
        build_launch_approval(
            protocol_sha="p", budget_sha="b", decision_thresholds_sha="d",
            approved_at_utc="2026-07-24T18:00:00Z",
            mode="USER_EXPLICIT_GATE_GREEN_THEN_PAUSE",
        )


def test_verify_launch_approval_rejects_a_forged_mode_on_disk():
    """An approval file edited to a different mode must not verify."""
    approval = build_launch_approval(
        protocol_sha="p", budget_sha="b", decision_thresholds_sha="d",
        approved_at_utc="2026-07-24T18:00:00Z",
    )
    approval["approval_mode"] = "USER_EXPLICIT_GATE_GREEN_THEN_PAUSE"
    with pytest.raises(ApprovalMismatch, match="not authorized"):
        verify_launch_approval(approval, protocol_sha="p", budget_sha="b",
                               decision_thresholds_sha="d")


def test_verify_launch_approval_detects_drift():
    approval = build_launch_approval(
        protocol_sha="p", budget_sha="b", decision_thresholds_sha="d",
        approved_at_utc="2026-07-24T18:00:00Z",
    )
    verify_launch_approval(approval, protocol_sha="p", budget_sha="b", decision_thresholds_sha="d")
    # A changed threshold config invalidates the approval.
    with pytest.raises(ApprovalMismatch, match="decision_thresholds_sha"):
        verify_launch_approval(approval, protocol_sha="p", budget_sha="b",
                               decision_thresholds_sha="CHANGED")
