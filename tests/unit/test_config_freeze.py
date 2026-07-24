"""Config hashing and the freeze / launch-approval logic."""

from __future__ import annotations

from pathlib import Path

import pytest

from shapeflow_p1.config import ConfigBundle, ConfigError, config_sha, load_config, require
from shapeflow_p1.experiment.freeze import (
    APPROVAL_MODE_AUTO,
    APPROVAL_MODE_PAUSE,
    ApprovalMismatch,
    FreezeRecord,
    build_launch_approval,
    verify_launch_approval,
)

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_real_decision_and_budget_configs_load_and_hash():
    for name in ("decision.yaml", "budget_v1.yaml"):
        data, sha = load_config(CONFIGS / name)
        assert isinstance(data, dict) and len(sha) == 64


def test_decision_thresholds_are_the_v01_values():
    data, _ = load_config(CONFIGS / "decision.yaml")
    assert require(data, "utility", "minimum_meaningful_work_reduction") == 0.10
    assert require(data, "coverage", "conditional_task_exposure_coverage_lcb_min") == 0.30
    assert require(data, "quality_ni_margin", "critical_harm_risk_pp_max") == 3


def test_budget_caps_are_the_v01_hard_bounds():
    data, _ = load_config(CONFIGS / "budget_v1.yaml")
    assert require(data, "api_budget", "tavily_max_requests") == 500
    assert require(data, "api_budget", "deepseek_max_usd") == 10.00
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


def test_launch_approval_defaults_to_pause_and_requires_human_launch():
    approval = build_launch_approval(
        protocol_sha="p", budget_sha="b", decision_thresholds_sha="d",
        approved_at_utc="2026-07-24T18:00:00Z",
    )
    # The user's override: gate-green-then-pause, not auto-launch.
    assert approval["approval_mode"] == APPROVAL_MODE_PAUSE
    assert approval["approval_mode"] != APPROVAL_MODE_AUTO
    assert approval["requires_human_launch"] is True


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
