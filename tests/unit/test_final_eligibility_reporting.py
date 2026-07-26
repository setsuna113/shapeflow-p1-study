"""Exploratory eligibility is fully reported but isolated from primary decisions."""

from __future__ import annotations

import copy
import json
import runpy
from pathlib import Path

import pytest

from shapeflow_p1.analysis.finalize import _exploratory_eligibility_snapshot
from shapeflow_p1.analysis.report import render_json, render_markdown

_TEST_DIR = Path(__file__).parent
_FINAL_HELPERS = runpy.run_path(str(_TEST_DIR / "test_final_decision.py"))
_REPORT_HELPERS = runpy.run_path(str(_TEST_DIR / "test_report.py"))
_build = _FINAL_HELPERS["_build"]
_inputs = _FINAL_HELPERS["_inputs"]
_reseal = _FINAL_HELPERS["_reseal"]
_decision = _REPORT_HELPERS["_decision"]


_SCHEMA = "e2e_eligibility_result_v3"
_SCOPE = "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY"


def _configured_target_ids(decision: dict) -> list[str]:
    configured = (decision.get("eligibility_learner") or {}).get("targets")
    if isinstance(configured, dict):
        return sorted(map(str, configured))
    if isinstance(configured, list):
        return [str(target["target_id"]) for target in configured]
    return ["C_STANDALONE", "H_STANDALONE", "HC_JOINT_CHOICE"]


def _eligibility(target_ids: list[str], *, status: str) -> dict:
    return {
        "schema_version": _SCHEMA,
        "analysis_scope": _SCOPE,
        "status": "OK",
        "target_results": [
            {
                "target_id": target_id,
                "status": status,
                "rule": None if status != "STABLE_FORMATIVE_RULE" else "source_count >= 3",
                "oof_task_coverage": 0.0 if status != "STABLE_FORMATIVE_RULE" else 0.25,
            }
            for target_id in reversed(target_ids)
        ],
    }


def _with_eligibility(inputs: tuple, eligibility: dict) -> tuple:
    values = list(inputs)
    e2e = copy.deepcopy(values[1])
    effects = copy.deepcopy(e2e["e2e_effects"])
    effects["eligibility"] = eligibility
    e2e["e2e_effects"] = _reseal(effects)
    values[1] = _reseal(e2e)
    return tuple(values)


def test_v3_family_is_canonicalized_and_keeps_every_frozen_target() -> None:
    decision = {
        "eligibility_learner": {
            "targets": [
                {"target_id": "H_STANDALONE"},
                {"target_id": "C_STANDALONE"},
            ]
        }
    }
    effects = {
        "eligibility": {
            "schema_version": _SCHEMA,
            "analysis_scope": _SCOPE,
            "target_results": {
                "H_STANDALONE": {"status": "NULL_NO_ELIGIBLE_TASKS"},
                "C_STANDALONE": {
                    "target_id": "C_STANDALONE",
                    "status": "UNSTABLE_RULE",
                },
            },
        }
    }

    status, snapshot, limitations = _exploratory_eligibility_snapshot(
        effects, decision_config=decision
    )

    assert status == _SCOPE
    assert [row["target_id"] for row in snapshot["target_results"]] == [
        "C_STANDALONE",
        "H_STANDALONE",
    ]
    assert [row["status"] for row in snapshot["target_results"]] == [
        "UNSTABLE_RULE",
        "NULL_NO_ELIGIBLE_TASKS",
    ]
    assert "NO_PRIMARY_VERDICT" in snapshot["decision_use"]
    assert any("DOES_NOT_CHANGE_PRIMARY_VERDICTS" in item for item in limitations)


def test_v3_family_rejects_missing_frozen_target_and_duplicate_id() -> None:
    decision = {
        "eligibility_learner": {
            "targets": {
                "H_STANDALONE": {"treatment_semantic": "h"},
                "C_STANDALONE": {"treatment_semantic": "c"},
            }
        }
    }
    missing = {
        "eligibility": {
            "schema_version": _SCHEMA,
            "analysis_scope": _SCOPE,
            "target_results": [
                {"target_id": "H_STANDALONE", "status": "NOT_ESTIMABLE"}
            ],
        }
    }
    with pytest.raises(ValueError, match="every frozen target"):
        _exploratory_eligibility_snapshot(missing, decision_config=decision)

    duplicate = copy.deepcopy(missing)
    duplicate["eligibility"]["target_results"] *= 2
    with pytest.raises(ValueError, match="uniquely named"):
        _exploratory_eligibility_snapshot(duplicate, decision_config={})


def test_opposite_exploratory_findings_do_not_change_any_primary_arm_decision() -> None:
    base = _inputs()
    target_ids = _configured_target_ids(base[2])
    stable = _build(
        _with_eligibility(
            base,
            _eligibility(target_ids, status="STABLE_FORMATIVE_RULE"),
        )
    )
    null = _build(
        _with_eligibility(
            base,
            _eligibility(target_ids, status="NULL_NO_ELIGIBLE_TASKS"),
        )
    )

    assert stable.eligibility_status == _SCOPE
    assert null.eligibility_status == _SCOPE
    for stable_node, null_node in zip(stable.nodes, null.nodes, strict=True):
        assert stable_node.verdict is null_node.verdict
        assert stable_node.champion_variant == null_node.champion_variant
        assert stable_node.arm_results == null_node.arm_results
        assert stable_node.coverage is None
        assert stable_node.where_it_works == ()


def test_json_and_markdown_report_null_unstable_and_not_estimable_targets() -> None:
    rows = [
        {"target_id": "A", "status": "NULL_NO_ELIGIBLE_TASKS"},
        {"target_id": "B", "status": "UNSTABLE_RULE"},
        {"target_id": "C", "status": "NOT_ESTIMABLE"},
    ]
    decision = _decision()
    decision = decision.__class__(
        **{
            **decision.__dict__,
            "eligibility_status": _SCOPE,
            "exploratory_task_level_eligibility": {
                "schema_version": _SCHEMA,
                "analysis_scope": _SCOPE,
                "target_results": rows,
            },
        }
    )

    parsed = json.loads(render_json(decision))
    markdown = render_markdown(decision)

    assert parsed["exploratory_task_level_eligibility"]["target_results"] == rows
    for row in rows:
        assert row["target_id"] in markdown
        assert row["status"] in markdown
    assert "never change a primary verdict" in markdown
    assert "not invocation coverage" in markdown
