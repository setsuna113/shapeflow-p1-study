"""Matched contrasts preserve pairing, semantics, missingness, and task-level weights."""

from __future__ import annotations

from copy import deepcopy

import pytest

from shapeflow_p1.analysis.e2e_effects import (
    QUALITY_GUARD_DIRECTIONS,
    TRAJECTORY_ENDPOINT_DIRECTIONS,
)
from shapeflow_p1.analysis.matched import build_matched_contrasts
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.hashing import sha256_hex

LEFT = "H_ID"
RIGHT = "H_TYPED"


def _semantics() -> dict[str, dict]:
    common = {
        "node": "WEBPAGE_P1",
        "chunker": "markdown_structure_v1",
        "scope": "per_page",
        "aggregation": "stable_union_v1",
        "close_mode": "separate",
        "selector_backend": "LLM",
        "publication_path": "STRUCTURED_SELECTION",
    }
    return {
        LEFT: {
            **common,
            "variant_id": "H02",
            "contract": "P1_ID",
            "output_representation": "EVIDENCE_IDS",
        },
        RIGHT: {
            **common,
            "variant_id": "H_TYPED_STABLE",
            "contract": "P1_TYPED",
            "output_representation": "TYPED_EVIDENCE",
        },
    }


def _design() -> dict:
    return {
        "version": "matched_contrasts_v2",
        "executable_variant_fields": [
            "node",
            "chunker",
            "scope",
            "contract",
            "aggregation",
            "close_mode",
            "selector_backend",
            "publication_path",
            "output_representation",
            "bridge_token_cap_each",
            "bridge_token_cap_total",
        ],
        "pairs": [
            {
                "contrast_id": "H_ID_VS_TYPED",
                "kind": "ID_vs_TYPED",
                "left_arm_id": LEFT,
                "right_arm_id": RIGHT,
                "target_factor": "contract",
                "left_level": "P1_ID",
                "right_level": "P1_TYPED",
                "factor_fields": ["contract", "output_representation"],
            }
        ],
    }


def _scope(*, valid: bool = True) -> dict:
    return {
        "freeze_root_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "block_freeze_sha256": "c" * 64,
        "block_digest": "d" * 64,
        "valid_for_paired_estimate": valid,
        "invalid_reason": "" if valid else "ENGINE_EPOCH_CHANGED",
        "engine_epochs": ["epoch-1"] if valid else ["epoch-1", "epoch-2"],
        "engine_epoch_by_arm": {"left": "epoch-1", "right": "epoch-1"},
    }


def _arm(
    arm_id: str,
    *,
    replicate: str,
    quality: float,
    work: float,
    latency: float = 10.0,
    valid: bool = True,
) -> dict:
    metrics = {
        metric: (1.0 - quality if metric == "critical_harm" else quality)
        for metric in QUALITY_GUARD_DIRECTIONS
    }
    trajectory = {
        endpoint: 0.0 for endpoint in TRAJECTORY_ENDPOINT_DIRECTIONS
    }
    trajectory["query_count"] = 8.0 if arm_id == LEFT else 10.0
    return {
        "arm_id": arm_id,
        "replicate_id": replicate,
        "variant_id": _semantics()[arm_id]["variant_id"],
        "quality_views": {
            view: dict(metrics)
            for view in ("strict", "fallback_assisted", "worst_case", "best_case")
        },
        "work_summary": {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": work,
            "tokens": {
                "prompt_tokens": 100.0,
                "completion_tokens": 20.0,
                "cached_prompt_tokens": 5.0,
            },
        },
        "e2e_latency_seconds": latency,
        "trajectory_metrics": {"status": "OK", **trajectory},
        "frozen_scope": _scope(valid=valid),
    }


def _block(
    task: str,
    replicate: str,
    *,
    left_quality: float = 0.9,
    right_quality: float = 0.8,
    left_work: float = 8.0,
    right_work: float = 10.0,
    valid: bool = True,
) -> dict:
    return {
        "block_id": f"block-{task}-{replicate}",
        "task_id": task,
        "replicate_id": replicate,
        "per_arm": {
            f"{LEFT}:{replicate}": _arm(
                LEFT,
                replicate=replicate,
                quality=left_quality,
                work=left_work,
                valid=valid,
            ),
            f"{RIGHT}:{replicate}": _arm(
                RIGHT,
                replicate=replicate,
                quality=right_quality,
                work=right_work,
                valid=valid,
            ),
        },
    }


def _study() -> tuple[list[dict], dict[str, str]]:
    records = [
        _block("task-a", "0"),
        _block("task-a", "1"),
        _block("task-b", "0"),
    ]
    return records, {"task-a": "cluster-a", "task-b": "cluster-b"}


def _bind_scope(records: list[dict]) -> dict:
    score_index = []
    for record in records:
        record["run_id"] = "run-matched"
        record["phase_id"] = "screen"
        record["execution_binding_sha256"] = "e" * 64
        record["protocol_document_sha256"] = "f" * 64
        record["content_sha256"] = sha256_hex(
            canonical_json(
                {key: value for key, value in record.items() if key != "content_sha256"}
            )
        )
        first = next(iter(record["per_arm"].values()))["frozen_scope"]
        epochs = list(first["engine_epochs"])
        arm_keys = list(record["per_arm"])
        by_arm = {
            arm_key: epochs[min(index, len(epochs) - 1)]
            for index, arm_key in enumerate(arm_keys)
        }
        score_index.append(
            {
                "block_id": record["block_id"],
                "task_id": record["task_id"],
                "replicate_id": record["replicate_id"],
                "score_content_sha256": record["content_sha256"],
                "execution_binding_sha256": "e" * 64,
                "protocol_document_sha256": "f" * 64,
                "block_freeze_sha256": first["block_freeze_sha256"],
                "block_digest": first["block_digest"],
                "valid_for_paired_estimate": first["valid_for_paired_estimate"],
                "invalid_reason": first["invalid_reason"],
                "engine_epochs": epochs,
                "engine_epoch_by_arm": by_arm,
            }
        )
    receipt = {
        "schema_version": "evaluated_itt_scope_v1",
        "run_id": "run-matched",
        "phase_id": "screen",
        "execution_binding_sha256": "e" * 64,
        "protocol_document_sha256": "f" * 64,
        "schedule_sha256": "b" * 64,
        "freeze_root_sha256": "a" * 64,
        "analysis_design_receipt_sha256": "1" * 64,
        "task_feature_registry_sha256": "2" * 64,
        "eligibility_spec_content_sha256": "3" * 64,
        "all_offered_blocks": len(score_index),
        "scores": score_index,
    }
    receipt["evaluation_scope_sha256"] = sha256_hex(canonical_json(receipt))
    return receipt


def _run(records: list[dict], clusters: dict[str, str]) -> dict:
    return build_matched_contrasts(
        records,
        scope_receipt=_bind_scope(records),
        cluster_by_task=clusters,
        matched_contrasts=_design(),
        arm_variants=_semantics(),
        n_boot=100,
        seed=17,
    )


def test_reports_all_quality_views_metrics_and_operational_endpoints():
    records, clusters = _study()
    result = _run(records, clusters)
    contrast = result["contrasts"][0]

    assert result["content_sha256"] == sha256_hex(
        canonical_json(
            {key: value for key, value in result.items() if key != "content_sha256"}
        )
    )
    assert contrast["pairing_status"] == "OK"
    assert set(contrast["quality"]) == {
        "strict",
        "fallback_assisted",
        "worst_case",
        "best_case",
    }
    assert all(
        set(by_metric) == set(QUALITY_GUARD_DIRECTIONS)
        for by_metric in contrast["quality"].values()
    )
    assert set(contrast["operational"]) == {
        "service_work_seconds",
        "e2e_latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
    }
    quality = contrast["quality"]["strict"]["weighted_required_atom_recall"]
    assert quality["status"] == "OK"
    assert quality["point"] == pytest.approx(0.1)
    assert quality["replicates_by_task"] == {"task-a": 2, "task-b": 1}
    work = contrast["operational"]["service_work_seconds"]
    assert work["status"] == "OK"
    assert work["point"] == pytest.approx(0.2)
    assert work["one_sided_95_lcb"] == pytest.approx(0.2)
    assert set(contrast["trajectory"]) == set(TRAJECTORY_ENDPOINT_DIRECTIONS)
    assert contrast["trajectory"]["query_count"]["status"] == "OK"
    assert contrast["trajectory"]["query_count"]["point"] == -2.0
    assert result["contrast_convention"]["trajectory_checkpoint_divergence"].startswith(
        "mediated_end_to_end"
    )


def test_accepts_the_hash_locked_quality_metric_name_list():
    records, clusters = _study()
    result = build_matched_contrasts(
        records,
        scope_receipt=_bind_scope(records),
        cluster_by_task=clusters,
        matched_contrasts=_design(),
        arm_variants=_semantics(),
        quality_metrics=list(QUALITY_GUARD_DIRECTIONS),
        n_boot=10,
    )

    assert result["contrasts"][0]["quality"]["strict"][
        "weighted_required_atom_recall"
    ]["status"] == "OK"


def test_unequal_replicates_are_task_averaged_before_cluster_bootstrap():
    records = [
        _block("task-a", "0", left_quality=0.0, right_quality=0.0),
        _block("task-a", "1", left_quality=1.0, right_quality=0.0),
        _block("task-b", "0", left_quality=1.0, right_quality=0.0),
    ]
    result = _run(records, {"task-a": "cluster-a", "task-b": "cluster-b"})
    endpoint = result["contrasts"][0]["quality"]["strict"][
        "weighted_required_atom_recall"
    ]

    # task-a contributes mean(.0, 1.0)=.5 once and task-b contributes 1.0 once.
    assert endpoint["point"] == pytest.approx(0.75)
    assert endpoint["replicates_by_task"] == {"task-a": 2, "task-b": 1}


def test_missing_arm_is_structural_and_never_complete_case_filtered():
    records, clusters = _study()
    del records[0]["per_arm"][f"{RIGHT}:0"]

    result = _run(records, clusters)
    contrast = result["contrasts"][0]

    assert contrast["pairing_status"] == "NOT_ESTIMABLE"
    assert contrast["blocks_offered"] == 3
    assert contrast["blocks_paired"] == 2
    assert contrast["structural_issues"][0]["code"] == "MISSING_ARM"
    assert (
        contrast["quality"]["strict"]["weighted_required_atom_recall"]["status"]
        == "NOT_ESTIMABLE"
    )
    assert (
        contrast["operational"]["service_work_seconds"]["reason"]
        == "ALL_OFFERED_CONTAINS_STRUCTURALLY_INVALID_PAIR"
    )


def test_endpoint_missingness_is_independent_and_fail_closed():
    records, clusters = _study()
    records[0]["per_arm"][f"{LEFT}:0"]["work_summary"]["service_seconds"] = None

    result = _run(records, clusters)
    contrast = result["contrasts"][0]

    work = contrast["operational"]["service_work_seconds"]
    assert contrast["pairing_status"] == "OK"
    assert work["status"] == "NOT_ESTIMABLE"
    assert work["missing_counts"]["by_side"] == {"left": 1, "right": 0}
    assert contrast["operational"]["e2e_latency_seconds"]["status"] == "OK"
    assert (
        contrast["quality"]["strict"]["weighted_required_atom_recall"]["status"]
        == "OK"
    )


def test_paired_invalid_block_is_reported_and_invalidates_whole_contrast():
    records, clusters = _study()
    records[0] = _block("task-a", "0", valid=False)

    result = _run(records, clusters)
    contrast = result["contrasts"][0]

    assert contrast["pairing_status"] == "NOT_ESTIMABLE"
    assert contrast["structural_issues"][0] == {
        "code": "PAIRED_INVALID_BLOCK",
        "block_id": "block-task-a-0",
        "reason": "ENGINE_EPOCH_CHANGED",
        "engine_epochs": ["epoch-1", "epoch-2"],
    }
    assert contrast["operational"]["prompt_tokens"]["status"] == "NOT_ESTIMABLE"


def test_observed_variant_mismatch_is_structural():
    records, clusters = _study()
    records[0]["per_arm"][f"{LEFT}:0"]["variant_id"] = "H03"

    result = _run(records, clusters)
    contrast = result["contrasts"][0]

    assert contrast["pairing_status"] == "NOT_ESTIMABLE"
    assert contrast["structural_issues"][0]["code"] == "ARM_ROW_SEMANTIC_MISMATCH"
    assert contrast["structural_issues"][0]["observed_variant_ids"][LEFT] == "H03"


def test_declared_variants_may_not_differ_outside_registered_factor():
    records, clusters = _study()
    semantics = deepcopy(_semantics())
    semantics[RIGHT]["scope"] = "hierarchical"

    with pytest.raises(ValueError, match="differ outside factor_fields"):
        build_matched_contrasts(
            records,
            scope_receipt=_bind_scope(records),
            cluster_by_task=clusters,
            matched_contrasts=_design(),
            arm_variants=semantics,
            n_boot=10,
        )
