"""Fail-closed tests for the all-offered E2E factorial/eligibility bridge."""

from __future__ import annotations

import math
from copy import deepcopy
from pathlib import Path

import pytest

from shapeflow_p1.analysis.e2e_effects import (
    ELIGIBILITY_TARGET_FAMILY,
    QUALITY_GUARD_DIRECTIONS,
    TRAJECTORY_ENDPOINT_DIRECTIONS,
    build_e2e_effects,
    build_factorial_effects,
    task_feature_registry_sha256,
)
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.config import load_config
from shapeflow_p1.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[2]


def _seal(body: dict, field: str = "content_sha256") -> dict:
    body[field] = sha256_hex(
        canonical_json({key: value for key, value in body.items() if key != field})
    )
    return body


def _arm(
    *,
    task_id: str,
    replicate_id: str,
    arm_id: str,
    quality: float,
    work: float,
    latency: float,
    prompt: int,
    completion: int,
    cached: int,
    frozen_scope: dict,
) -> dict:
    metrics = {
        metric: (
            1.0
            if metric == "qualified_report"
            else 1.0 - quality
            if metric == "critical_harm"
            else quality
        )
        for metric in QUALITY_GUARD_DIRECTIONS
    }
    trajectory = {endpoint: 0.0 for endpoint in TRAJECTORY_ENDPOINT_DIRECTIONS}
    trajectory["query_count"] = {
        "P0": 10.0,
        "H_MARKDOWN_ID": 8.0,
        "C_ID": 9.0,
        "H_PLUS_C": 7.0,
    }[arm_id]
    trajectory["unique_query_count"] = trajectory["query_count"]
    return {
        "task_id": task_id,
        "replicate_id": replicate_id,
        "arm_id": arm_id,
        "variant_id": {
            "P0": "P0+P0",
            "H_MARKDOWN_ID": "H02+P0",
            "C_ID": "P0+C01",
            "H_PLUS_C": "H02+C01",
        }[arm_id],
        "quality_views": {
            view: dict(metrics)
            for view in ("strict", "fallback_assisted", "worst_case", "best_case")
        },
        "work_summary": {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": work,
            # The primary work endpoint under the native-concurrent layer. Kept distinct from
            # `service_seconds` in the fixture so a test cannot pass by reading the one that
            # only exists when the engine was serialized.
            "interval_union_seconds": work * 0.8,
            "max_concurrent_treatment_requests": 3,
            "energy_joules": work * 250.0,
            "tokens": {
                "prompt_tokens": prompt,
                "completion_tokens": completion,
                "cached_prompt_tokens": cached,
            },
        },
        "e2e_latency_seconds": latency,
        "trajectory_metrics": {"status": "OK", **trajectory},
        "frozen_scope": frozen_scope,
    }


def _make_study(*, replicates: int = 1) -> tuple[list[dict], dict, dict, dict]:
    task_features: dict[str, dict] = {}
    records: list[dict] = []
    task_rows: list[tuple[str, str, str, bool]] = []
    for cluster_index in range(4):
        for within_cluster in range(8):
            task_rows.append(
                (
                    f"train-{cluster_index}-{within_cluster}",
                    "FORMATIVE_SCREEN",
                    f"train-cluster-{cluster_index}",
                    within_cluster < 4,
                )
            )
    for heldout_index in range(4):
        task_rows.append(
            (
                f"heldout-{heldout_index}",
                "FORMATIVE_POWER_PILOT",
                f"heldout-cluster-{heldout_index // 2}",
                heldout_index % 2 == 0,
            )
        )

    for task_id, split, cluster_id, eligible in task_rows:
        task_features[task_id] = _seal(
            {
                "schema_version": "frozen_task_features_v1",
                "task_id": task_id,
                "split": split,
                "cluster_id": cluster_id,
                "features": {
                    "source_count": 1.0 if eligible else 10.0,
                    "redundancy": 0.1 if eligible else 0.9,
                },
            }
        )
        for replicate_index in range(replicates):
            replicate_id = str(replicate_index)
            block_id = f"block-{task_id}-{replicate_id}"
            frozen_scope = {
                "freeze_root_sha256": "a" * 64,
                "schedule_sha256": "b" * 64,
                "block_freeze_sha256": sha256_hex(f"freeze:{block_id}".encode()),
                "block_digest": sha256_hex(f"block:{block_id}".encode()),
                "valid_for_paired_estimate": True,
                "invalid_reason": "",
                "engine_epochs": ["epoch-1"],
                "engine_epoch_by_arm": {
                    f"{arm}:{replicate_id}": "epoch-1"
                    for arm in ("P0", "H_MARKDOWN_ID", "C_ID", "H_PLUS_C")
                },
            }
            values = {
                "P0": (0.80, 10.0, 20.0, 100, 20, 0),
                "H_MARKDOWN_ID": (
                    0.80,
                    8.0 if eligible else 9.5,
                    18.0,
                    80,
                    10,
                    5,
                ),
                "C_ID": (0.79, 9.0, 19.0, 90, 15, 3),
                "H_PLUS_C": (0.82, 7.0, 16.0, 70, 8, 8),
            }
            per_arm = {
                f"{arm_id}:{replicate_id}": _arm(
                    task_id=task_id,
                    replicate_id=replicate_id,
                    arm_id=arm_id,
                    quality=quality,
                    work=work,
                    latency=latency,
                    prompt=prompt,
                    completion=completion,
                    cached=cached,
                    frozen_scope=frozen_scope,
                )
                for arm_id, (
                    quality,
                    work,
                    latency,
                    prompt,
                    completion,
                    cached,
                ) in values.items()
            }
            records.append(
                _seal(
                    {
                        "schema_version": "evaluated_block_v1",
                        "run_id": "run-1",
                        "phase_id": "phase-1",
                        "execution_binding_sha256": "c" * 64,
                        "protocol_document_sha256": "d" * 64,
                        "block_id": block_id,
                        "task_id": task_id,
                        "replicate_id": replicate_id,
                        "per_arm": per_arm,
                    }
                )
            )

    receipt = {
        "schema_version": "evaluated_itt_scope_v1",
        "run_id": "run-1",
        "phase_id": "phase-1",
        "execution_binding_sha256": "c" * 64,
        "protocol_document_sha256": "d" * 64,
        "schedule_sha256": "b" * 64,
        "freeze_root_sha256": "a" * 64,
        "all_offered_blocks": len(records),
        "scores": [
            {
                "block_id": record["block_id"],
                "task_id": record["task_id"],
                "replicate_id": record["replicate_id"],
                "execution_binding_sha256": "c" * 64,
                "protocol_document_sha256": "d" * 64,
                "score_content_sha256": record["content_sha256"],
                "block_freeze_sha256": next(iter(record["per_arm"].values()))["frozen_scope"][
                    "block_freeze_sha256"
                ],
                "block_digest": next(iter(record["per_arm"].values()))["frozen_scope"][
                    "block_digest"
                ],
                "valid_for_paired_estimate": True,
                "invalid_reason": "",
                "engine_epochs": ["epoch-1"],
                "engine_epoch_by_arm": next(iter(record["per_arm"].values()))["frozen_scope"][
                    "engine_epoch_by_arm"
                ],
            }
            for record in records
        ],
    }
    _seal(receipt, "evaluation_scope_sha256")
    arm_map = {
        "p0": "P0",
        "h": "H_MARKDOWN_ID",
        "c": "C_ID",
        "hc": "H_PLUS_C",
    }
    spec = {
        "schema_version": "e2e_eligibility_spec_v2",
        "analysis_scope": "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY",
        "targets": [
            {
                "target_id": target_id,
                "treatment_semantic": treatment,
                "treatment_arm_id": arm_map[treatment],
                "comparator_semantics": list(comparators),
                "comparator_arm_ids": [arm_map[item] for item in comparators],
                "comparator_logic": "ALL",
            }
            for target_id, (treatment, comparators) in ELIGIBILITY_TARGET_FAMILY.items()
        ],
        "quality_view": "strict",
        "quality_guards": {
            "higher_is_better_min_effect": {
                "weighted_required_atom_recall": -0.05,
                "grounded_claim_precision": -0.03,
                "citation_correctness": -0.03,
                "citation_association": -0.03,
                "required_facet_coverage": -0.05,
                "qualified_report": -0.05,
            },
            "lower_is_better_max_effect": {"critical_harm": 0.03},
            "inapplicability_policy":
                "BOTH_ARMS_INAPPLICABLE_PASS_OTHERWISE_FAIL_CLOSED",
        },
        "absolute_treatment_requirements": {
            "qualified_report_min": 1.0,
            "replicate_policy": "ALL_REPLICATES_MUST_QUALIFY",
        },
        "minimum_work_saving": 0.10,
        "feature_names": ["source_count", "redundancy"],
        "task_feature_registry_sha256": task_feature_registry_sha256(task_features),
        "training_splits": ["FORMATIVE_SCREEN"],
        "heldout_splits": ["FORMATIVE_POWER_PILOT"],
        "excluded_splits": [],
        "max_depth": 2,
        "min_tasks_per_leaf": 8,
        "eligible_threshold": 0.5,
        "cv_folds": 2,
        "stability_bootstraps": 20,
        "minimum_root_stability": 0.70,
    }
    _seal(spec)
    return records, receipt, task_features, spec


def _build(study):
    records, receipt, features, spec = study
    decision, _ = load_config(REPO / "configs" / "decision.yaml")
    return build_e2e_effects(
        records,
        task_features=features,
        eligibility_spec=spec,
        scope_receipt=receipt,
        expected_variant_ids={
            "p0": "P0+P0",
            "h": "H02+P0",
            "c": "P0+C01",
            "hc": "H02+C01",
        },
        task_level_joint_outcomes_policy=decision["task_level_joint_outcomes"],
        n_boot=100,
    )


def _reseal_scope(records: list[dict], receipt: dict) -> None:
    for record in records:
        _seal(record)
    receipt["scores"] = [
        {
            "block_id": record["block_id"],
            "task_id": record["task_id"],
            "replicate_id": record["replicate_id"],
            "execution_binding_sha256": receipt["execution_binding_sha256"],
            "protocol_document_sha256": receipt["protocol_document_sha256"],
            "score_content_sha256": record["content_sha256"],
            "block_freeze_sha256": next(iter(record["per_arm"].values()))["frozen_scope"][
                "block_freeze_sha256"
            ],
            "block_digest": next(iter(record["per_arm"].values()))["frozen_scope"]["block_digest"],
            "valid_for_paired_estimate": next(iter(record["per_arm"].values()))["frozen_scope"][
                "valid_for_paired_estimate"
            ],
            "invalid_reason": next(iter(record["per_arm"].values()))["frozen_scope"][
                "invalid_reason"
            ],
            "engine_epochs": next(iter(record["per_arm"].values()))["frozen_scope"][
                "engine_epochs"
            ],
            "engine_epoch_by_arm": next(iter(record["per_arm"].values()))["frozen_scope"][
                "engine_epoch_by_arm"
            ],
        }
        for record in records
    ]
    receipt["all_offered_blocks"] = len(records)
    _seal(receipt, "evaluation_scope_sha256")


def _rebind_features(features: dict, spec: dict) -> None:
    spec["task_feature_registry_sha256"] = task_feature_registry_sha256(features)
    _seal(spec)


def test_builds_separate_factorial_endpoints_and_interaction():
    result = _build(_make_study())
    endpoints = result["factorial_endpoints"]
    assert set(endpoints) == {
        "quality",
        "interval_union_seconds",
        "service_work_seconds",
        "energy_joules",
        "e2e_latency_seconds",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
    }
    assert endpoints["quality"]["direction"] == "higher_is_better"
    assert endpoints["service_work_seconds"]["direction"] == "higher_saving_is_better"
    assert endpoints["cached_prompt_tokens"]["direction"] == "descriptive_only"
    assert endpoints["service_work_seconds"]["contrast_scale"] == ("paired_log_ratio_saving")
    assert endpoints["quality"]["contrast_scale"] == "treatment_minus_baseline"
    assert endpoints["quality"]["h_simple"]["point"] == pytest.approx(0.0)
    assert endpoints["quality"]["c_simple"]["point"] == pytest.approx(-0.01)
    assert endpoints["quality"]["joint"]["point"] == pytest.approx(0.02)
    assert endpoints["quality"]["interaction"]["point"] == pytest.approx(0.03)
    cluster_equal_h_saving = 1.0 - math.exp((math.log(0.8) + math.log(0.95)) / 2)
    assert endpoints["service_work_seconds"]["h_simple"]["point"] == pytest.approx(
        cluster_equal_h_saving
    )
    assert endpoints["service_work_seconds"]["h_simple"]["one_sided_95_lcb"] == pytest.approx(
        cluster_equal_h_saving
    )
    assert endpoints["prompt_tokens"]["h_simple"]["point"] == pytest.approx(-20)
    assert endpoints["completion_tokens"]["interaction"]["point"] == pytest.approx(3)
    work_tail = endpoints["service_work_seconds"]["task_effect_distribution"]["h_simple"]
    assert work_tail["scale"] == "per_task_fraction_saved"
    assert work_tail["n_tasks"] == 36
    assert work_tail["median"] == pytest.approx(0.125)
    assert work_tail["p05"] <= work_tail["median"] <= work_tail["p95"]
    latency_tail = endpoints["e2e_latency_seconds"]["task_effect_distribution"]["joint"]
    assert latency_tail["scale"] == "per_task_treatment_minus_baseline"
    assert latency_tail["median"] == pytest.approx(-4.0)
    joint = result["task_level_joint_outcomes"]
    assert joint["status"] == "OK"
    assert joint["arms"]["h_simple"]["saving_threshold_proportions"]["saving_ge_10pct"][
        "point"
    ] == pytest.approx(0.5)
    assert joint["arms"]["joint"]["saving_threshold_proportions"]["saving_ge_25pct"][
        "point"
    ] == pytest.approx(1.0)
    assert joint["arms"]["joint"]["quality_qualified_pareto_win_proportion"][
        "point"
    ] == pytest.approx(1.0)
    assert joint["arms"]["c_simple"]["slower_and_quality_harmed_proportion"][
        "point"
    ] == pytest.approx(0.0)
    assert result["quality_guards"]["strict"]["critical_harm"]["direction"] == "lower_is_better"
    assert set(result["trajectory_endpoints"]) == set(TRAJECTORY_ENDPOINT_DIRECTIONS)
    assert result["trajectory_endpoints"]["query_count"]["status"] == "OK"
    assert result["trajectory_endpoints"]["query_count"]["h_simple"]["point"] == -2.0
    assert result["trajectory_endpoints"]["query_count"]["interaction"]["point"] == 0.0
    assert result["variant_validation"] == "VERIFIED"
    assert result["content_sha256"] == sha256_hex(
        canonical_json({key: value for key, value in result.items() if key != "content_sha256"})
    )


def test_eligibility_is_training_only_content_addressed_and_task_counted():
    result = _build(_make_study(replicates=2))
    eligibility = result["eligibility"]
    assert eligibility["schema_version"] == "e2e_eligibility_result_v3"
    assert eligibility["analysis_scope"] == (
        "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY"
    )
    assert set(eligibility["target_results"]) == set(ELIGIBILITY_TARGET_FAMILY)
    h_result = eligibility["target_results"]["H_STANDALONE"]
    dataset = h_result["training_dataset"]
    assert h_result["fit_scope"] == "TRAINING_ONLY"
    assert h_result["training_rows"] == 32
    assert h_result["training_tasks"] == 32
    assert h_result["composite_label"]["successes"] == 16
    assert {row["split"] for row in dataset["rows"]} == {"FORMATIVE_SCREEN"}
    assert sum(row["composite_success"] for row in dataset["rows"]) == 16
    assert dataset["content_sha256"] == sha256_hex(
        canonical_json({key: value for key, value in dataset.items() if key != "content_sha256"})
    )
    assert h_result["rule"]["root_split_feature"] == "source_count"
    assert h_result["rule"]["min_tasks_per_leaf"] == 8
    assert h_result["heldout_evaluation"]["fit_used_heldout"] is False
    assert h_result["heldout_evaluation"]["confirmatory_claim_allowed"] is False
    assert h_result["heldout_evaluation"]["rows"] == 4
    assert "cluster_bootstrap_root_stability" in h_result
    assert "cluster_equal_task_coverage" in h_result["coverage_on_training_registry"]
    assert "invocation_coverage" not in str(eligibility)
    assert all(
        target["target_id"] == target_id
        for target_id, target in eligibility["target_results"].items()
    )


def test_eligibility_target_family_cannot_be_reduced_or_relabelled_after_outcomes():
    records, receipt, features, spec = _make_study()
    reduced = deepcopy(spec)
    reduced["targets"].pop()
    _seal(reduced)
    with pytest.raises(ValueError, match="target family"):
        _build((records, receipt, features, reduced))

    relabelled = deepcopy(spec)
    joint = next(
        item for item in relabelled["targets"]
        if item["target_id"] == "HC_JOINT_CHOICE"
    )
    joint["comparator_semantics"] = ["p0"]
    joint["comparator_arm_ids"] = ["P0"]
    _seal(relabelled)
    with pytest.raises(ValueError, match="predeclared family"):
        _build((records, receipt, features, relabelled))


def test_hc_joint_choice_does_not_relabel_an_h_only_benefit_as_joint():
    records, receipt, features, spec = _make_study()
    for record in records:
        summary = record["per_arm"][f"H_MARKDOWN_ID:{record['replicate_id']}"][
            "work_summary"
        ]
        # Both, kept consistent: eligibility reads the primary work endpoint, which is the
        # interval union, and moving only the serialized sum would leave the test measuring an
        # endpoint the decision path no longer uses.
        summary["service_seconds"] = 6.0
        summary["interval_union_seconds"] = 6.0 * 0.8
    _reseal_scope(records, receipt)

    result = _build((records, receipt, features, spec))["eligibility"]["target_results"]

    assert result["H_STANDALONE"]["composite_label"]["successes"] == 32
    assert result["HC_JOINT_CHOICE"]["composite_label"]["successes"] == 0
    assert all(
        not row["composite_success"]
        for row in result["HC_JOINT_CHOICE"]["training_dataset"]["rows"]
    )
    assert result["C_INCREMENT_GIVEN_H"]["composite_label"]["successes"] == 0


def test_equal_failures_cannot_be_labelled_useful_merely_because_p1_fails_faster():
    records, receipt, features, spec = _make_study()
    for record in records:
        replicate = record["replicate_id"]
        for arm_id in ("P0", "H_MARKDOWN_ID"):
            record["per_arm"][f"{arm_id}:{replicate}"]["quality_views"]["strict"][
                "qualified_report"
            ] = 0.0
    _reseal_scope(records, receipt)

    h_result = _build((records, receipt, features, spec))["eligibility"][
        "target_results"
    ]["H_STANDALONE"]

    assert h_result["composite_label"]["successes"] == 0
    assert all(
        not row["comparator_results"][0]["absolute_treatment_requirement"]["pass"]
        for row in h_result["training_dataset"]["rows"]
    )


def test_eligibility_rejects_trajectory_mediators_even_when_rehashed():
    records, receipt, features, spec = _make_study()
    for forbidden in (
        "visible_message_tokens",
        "tool_call_count",
        "query_attempt_count",
        "close_reason_research_complete",
    ):
        tampered = deepcopy(spec)
        tampered["feature_names"] = [forbidden]
        _seal(tampered)
        with pytest.raises(ValueError, match="non-pre-treatment"):
            _build((records, receipt, features, tampered))


def test_one_arm_quality_applicability_failure_is_target_scoped_and_reported():
    records, receipt, features, spec = _make_study()
    first = records[0]
    first["per_arm"]["H_MARKDOWN_ID:0"]["quality_views"]["strict"][
        "citation_association"
    ] = None
    _reseal_scope(records, receipt)

    targets = _build((records, receipt, features, spec))["eligibility"]["target_results"]

    assert targets["H_STANDALONE"]["status"] == "NOT_ESTIMABLE"
    assert "arm-asymmetric applicability" in targets["H_STANDALONE"]["reason"]
    assert targets["C_STANDALONE"]["status"] != "NOT_ESTIMABLE"
    assert set(targets) == set(ELIGIBILITY_TARGET_FAMILY)


def test_rejects_missing_factorial_arm_even_when_scope_is_resealed():
    study = list(_make_study())
    records, receipt = study[:2]
    records[0]["per_arm"].pop("H_PLUS_C:0")
    _reseal_scope(records, receipt)
    with pytest.raises(ValueError, match="exactly one H_PLUS_C"):
        _build(tuple(study))


def test_rejects_scope_coordinate_swap():
    records, receipt, features, spec = _make_study()
    receipt["scores"][0]["task_id"] = "another-task"
    _seal(receipt, "evaluation_scope_sha256")
    with pytest.raises(ValueError, match="coordinates differ"):
        _build((records, receipt, features, spec))


def test_rejects_unverified_or_overlapping_service_telemetry():
    study = list(_make_study())
    records, receipt = study[:2]
    records[0]["per_arm"]["H_MARKDOWN_ID:0"]["work_summary"]["overlap_valid"] = False
    _reseal_scope(records, receipt)
    result = _build(tuple(study))
    endpoint = result["factorial_endpoints"]["service_work_seconds"]
    assert endpoint["status"] == "NOT_ESTIMABLE"
    assert endpoint["reason"] == "PARTIAL_ARM_MEASUREMENT_MISSING_FAIL_CLOSED"
    assert endpoint["missing_counts"]["by_arm"]["h"] == 1
    # A missing work endpoint cannot erase independently observed quality.
    assert result["factorial_endpoints"]["quality"]["status"] == "OK"


def test_rejects_post_treatment_feature_and_feature_registry_swap():
    records, receipt, features, spec = _make_study()
    bad_spec = deepcopy(spec)
    bad_spec["feature_names"] = ["fallback_count"]
    _seal(bad_spec)
    with pytest.raises(ValueError, match="non-pre-treatment"):
        _build((records, receipt, features, bad_spec))

    swapped_features = deepcopy(features)
    swapped_features["train-0-0"]["features"]["source_count"] = 999
    _seal(swapped_features["train-0-0"])
    with pytest.raises(ValueError, match="does not bind the exact"):
        _build((records, receipt, swapped_features, spec))


def test_rejects_train_holdout_cluster_leakage_and_unknown_splits():
    records, receipt, features, spec = _make_study()
    leaky = deepcopy(features)
    leaky["heldout-0"]["cluster_id"] = "train-cluster-0"
    _seal(leaky["heldout-0"])
    leaky_spec = deepcopy(spec)
    _rebind_features(leaky, leaky_spec)
    result = _build((records, receipt, leaky, leaky_spec))
    assert result["factorial_endpoints"]["quality"]["status"] == "OK"
    assert result["eligibility"]["status"] == "NOT_ESTIMABLE"
    assert "both training and heldout" in result["eligibility"]["reason"]

    unknown = deepcopy(features)
    unknown["heldout-0"]["split"] = "UNDECLARED"
    _seal(unknown["heldout-0"])
    unknown_spec = deepcopy(spec)
    _rebind_features(unknown, unknown_spec)
    result = _build((records, receipt, unknown, unknown_spec))
    assert result["eligibility"]["status"] == "NOT_ESTIMABLE"
    assert "does not classify observed splits" in result["eligibility"]["reason"]


def test_expected_variant_ids_detect_arm_label_variant_drift():
    records, receipt, features, spec = _make_study()
    records[0]["per_arm"]["H_MARKDOWN_ID:0"]["variant_id"] = "H03+P0"
    _reseal_scope(records, receipt)
    with pytest.raises(ValueError, match="expected variant"):
        _build((records, receipt, features, spec))


def test_all_quality_views_and_guards_are_reported_with_fail_closed_missingness():
    records, receipt, features, spec = _make_study()
    result = _build((records, receipt, features, spec))
    assert set(result["quality_guards"]) == {
        "strict",
        "fallback_assisted",
        "worst_case",
        "best_case",
    }
    assert all(
        set(metrics) == set(QUALITY_GUARD_DIRECTIONS)
        for metrics in result["quality_guards"].values()
    )

    # No truth contradiction in one task is an applicability condition, not an arm failure.
    first = records[0]
    for arm in first["per_arm"].values():
        for view in arm["quality_views"].values():
            view["contradiction_handling"] = None
    _reseal_scope(records, receipt)
    result = _build((records, receipt, features, spec))
    contradiction = result["quality_guards"]["strict"]["contradiction_handling"]
    assert contradiction["status"] == "OK"
    assert contradiction["not_applicable_blocks"] == [first["block_id"]]

    # A missing value in only one randomized corner is measurement failure and cannot be
    # silently complete-case filtered.
    second = records[1]
    second["per_arm"]["H_MARKDOWN_ID:0"]["quality_views"]["strict"]["citation_completeness"] = None
    _reseal_scope(records, receipt)
    result = _build((records, receipt, features, spec))
    citation = result["quality_guards"]["strict"]["citation_completeness"]
    assert citation["status"] == "NOT_ESTIMABLE"
    assert citation["reason"] == "PARTIAL_ARM_MEASUREMENT_MISSING_FAIL_CLOSED"
    assert citation["missing_counts"]["by_arm"]["h"] == 1

    for record in records:
        for arm in record["per_arm"].values():
            for view in arm["quality_views"].values():
                view["grounded_negative_recall"] = None
    _reseal_scope(records, receipt)
    result = _build((records, receipt, features, spec))
    negative = result["quality_guards"]["best_case"]["grounded_negative_recall"]
    assert negative["status"] == "NOT_APPLICABLE"

    # A missing judge view is not evidence that the truth guard was inapplicable.
    for record in records:
        for arm in record["per_arm"].values():
            arm["quality_views"]["strict"] = None
    _reseal_scope(records, receipt)
    result = _build((records, receipt, features, spec))
    strict = result["quality_guards"]["strict"]["weighted_required_atom_recall"]
    assert strict["status"] == "NOT_ESTIMABLE"
    assert strict["reason"] == "PARTIAL_ARM_MEASUREMENT_MISSING_FAIL_CLOSED"


def test_endpoint_missingness_is_isolated_and_reports_counts():
    records, receipt, features, spec = _make_study()
    records[0]["per_arm"]["C_ID:0"]["e2e_latency_seconds"] = None
    records[0]["per_arm"]["C_ID:0"]["work_summary"].pop("e2e_latency_seconds", None)
    _reseal_scope(records, receipt)
    result = _build((records, receipt, features, spec))
    latency = result["factorial_endpoints"]["e2e_latency_seconds"]
    assert latency["status"] == "NOT_ESTIMABLE"
    assert latency["missing_counts"]["by_arm"]["c"] == 1
    assert result["factorial_endpoints"]["prompt_tokens"]["status"] == "OK"
    assert result["factorial_endpoints"]["service_work_seconds"]["status"] == "OK"


def test_unequal_replicates_are_task_averaged_before_cluster_bootstrap():
    records, receipt, features, spec = _make_study()
    target = records[0]
    for view in target["per_arm"]["H_MARKDOWN_ID:0"]["quality_views"].values():
        view["weighted_required_atom_recall"] = 0.10
    _reseal_scope(records, receipt)
    original = _build((records, receipt, features, spec))

    duplicate = deepcopy(target)
    duplicate["block_id"] = f"{target['block_id']}-duplicate"
    duplicate["replicate_id"] = "extra"
    duplicate_arms: dict[str, dict] = {}
    for key, row in duplicate["per_arm"].items():
        arm_id = key.rsplit(":", 1)[0]
        row["replicate_id"] = "extra"
        scope = deepcopy(row["frozen_scope"])
        scope["block_freeze_sha256"] = sha256_hex(f"freeze:{duplicate['block_id']}".encode())
        scope["block_digest"] = sha256_hex(f"block:{duplicate['block_id']}".encode())
        scope["engine_epoch_by_arm"] = {
            f"{name.rsplit(':', 1)[0]}:extra": "epoch-1" for name in duplicate["per_arm"]
        }
        row["frozen_scope"] = scope
        duplicate_arms[f"{arm_id}:extra"] = row
    duplicate["per_arm"] = duplicate_arms
    records.append(duplicate)
    _reseal_scope(records, receipt)
    duplicated = _build((records, receipt, features, spec))
    assert duplicated["factorial_endpoints"]["quality"]["h_simple"]["point"] == (
        pytest.approx(original["factorial_endpoints"]["quality"]["h_simple"]["point"])
    )
    assert (
        duplicated["factorial_endpoints"]["quality"]["replicate_reduction"]
        == "task_mean_before_cluster_bootstrap"
    )


def test_paired_invalid_is_retained_but_factorial_is_not_estimable():
    records, receipt, features, spec = _make_study()
    target = records[0]
    for arm in target["per_arm"].values():
        arm["frozen_scope"]["valid_for_paired_estimate"] = False
        arm["frozen_scope"]["invalid_reason"] = "CELL_SPANS_ENGINE_EPOCHS"
        arm["frozen_scope"]["engine_epochs"] = ["epoch-1", "epoch-2"]
        arm["frozen_scope"]["engine_epoch_by_arm"]["H_PLUS_C:0"] = "epoch-2"
    _reseal_scope(records, receipt)
    result = _build((records, receipt, features, spec))
    assert result["structural_accounting"]["all_offered_retained"] is True
    assert (
        result["structural_accounting"]["paired_invalid_blocks"][0]["block_id"]
        == target["block_id"]
    )
    assert all(
        endpoint["status"] == "NOT_ESTIMABLE" for endpoint in result["factorial_endpoints"].values()
    )
    assert result["eligibility"]["status"] == "NOT_ESTIMABLE"


def test_feature_registry_may_be_a_frozen_superset_of_analysis_tasks():
    records, receipt, features, spec = _make_study()
    features["future-confirmatory-task"] = _seal(
        {
            "schema_version": "frozen_task_features_v1",
            "task_id": "future-confirmatory-task",
            "split": "FORMATIVE_POWER_PILOT",
            "cluster_id": "future-cluster",
            "features": {"source_count": 3.0, "redundancy": 0.2},
        }
    )
    _rebind_features(features, spec)
    result = _build((records, receipt, features, spec))
    assert result["tasks"] == len({record["task_id"] for record in records})
    assert result["task_feature_registry_sha256"] == task_feature_registry_sha256(features)


def test_factorial_can_be_built_without_any_eligibility_spec():
    records, receipt, features, _ = _make_study()
    result = build_factorial_effects(
        records,
        task_features=features,
        scope_receipt=receipt,
        expected_variant_ids={
            "p0": "P0+P0",
            "h": "H02+P0",
            "c": "P0+C01",
            "hc": "H02+C01",
        },
        n_boot=50,
    )
    assert result["factorial_endpoints"]["quality"]["status"] == "OK"
    assert "eligibility" not in result


def test_scope_block_provenance_swap_is_rejected_even_when_receipt_is_resealed():
    records, receipt, features, spec = _make_study()
    receipt["scores"][0]["block_digest"] = "f" * 64
    _seal(receipt, "evaluation_scope_sha256")
    with pytest.raises(ValueError, match="exact frozen scope"):
        _build((records, receipt, features, spec))


def test_optional_schedule_and_freeze_root_objects_are_rehashed_end_to_end():
    records, receipt, features, _ = _make_study()
    schedule = _seal(
        {
            "protocol_sha": "9" * 64,
            "split": "FORMATIVE_SCREEN",
            "layer": "complete_e2e",
            "claim_scope": "FORMATIVE",
            "arms": [],
            "seeds": [0],
            "blocks": [],
            "notes": {},
        },
        "schedule_sha256",
    )
    receipt["schedule_sha256"] = schedule["schedule_sha256"]
    for record in records:
        for row in record["per_arm"].values():
            row["frozen_scope"]["schedule_sha256"] = schedule["schedule_sha256"]
    _reseal_scope(records, receipt)
    root = {
        "schema_version": "frozen_campaign_root_v1",
        "run_id": "run-1",
        "phase_id": "phase-1",
        "split": "FORMATIVE_SCREEN",
        "protocol_sha256": "9" * 64,
        "schedule_sha256": schedule["schedule_sha256"],
        "schedule": schedule,
        "terminal_frozen": True,
        "blocks": [
            {
                "block_id": row["block_id"],
                "block_digest": row["block_digest"],
                "freeze_sha256": row["block_freeze_sha256"],
                "terminal_frozen": True,
                "valid_for_paired_estimate": row["valid_for_paired_estimate"],
                "invalid_reason": row["invalid_reason"],
                "engine_epochs": row["engine_epochs"],
                "cells": [],
            }
            for row in receipt["scores"]
        ],
    }
    _seal(root, "freeze_root_sha256")
    receipt["freeze_root_sha256"] = root["freeze_root_sha256"]
    for record in records:
        for row in record["per_arm"].values():
            row["frozen_scope"]["freeze_root_sha256"] = root["freeze_root_sha256"]
    _reseal_scope(records, receipt)
    result = build_factorial_effects(
        records,
        task_features=features,
        scope_receipt=receipt,
        freeze_root=root,
        schedule_manifest=schedule,
        n_boot=20,
    )
    assert result["provenance"]["schedule_object_verified"] is True
    assert result["provenance"]["freeze_root_object_verified"] is True
    tampered = deepcopy(root)
    tampered["split"] = "CONFIRMATORY"
    with pytest.raises(ValueError, match="does not verify"):
        build_factorial_effects(
            records,
            task_features=features,
            scope_receipt=receipt,
            freeze_root=tampered,
            schedule_manifest=schedule,
            n_boot=20,
        )
