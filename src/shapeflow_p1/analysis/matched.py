"""Fail-closed matched-ablation analysis over frozen all-offered score records.

The end-to-end factorial asks whether enabling H and/or C helps the whole workflow.  This
module answers the narrower mechanism questions in ``matched_contrasts``: for example, ID
selection versus typed selection while every other executable variant field is held fixed.

The caller supplies already-verified frozen per-block score records, the source/topic cluster
for every task, the frozen matched-contrast declaration, and the semantic variant registry
indexed by arm ID.  We still verify the pairing and semantic facts that are specific to this
analysis:

* every task/replicate block has exactly one left and one right arm;
* both rows refer to the same frozen, pair-valid block;
* the executed ``variant_id`` matches the declared arm semantics; and
* the two declared variants differ only in the contrast's allowed ``factor_fields``.

Replicates are averaged within task before source/topic cluster bootstrap.  Missingness is
endpoint-specific: a missing work measurement cannot erase an observed quality comparison.
In contrast, a missing arm, variant mismatch, duplicate task/replicate coordinate, or any
pair-invalid block makes the entire contrast structurally ``NOT_ESTIMABLE``.  Such blocks are
reported, never complete-case filtered.

Trajectory or checkpoint divergence is deliberately absent from the acceptance rules.  In an
end-to-end treatment it is a mediated outcome, not a pairing violation.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np

from ..canonical import canonical_json
from ..hashing import sha256_hex
from .bootstrap import cluster_bootstrap_ci, paired_log_ratio_saving, seed_from
from .e2e_effects import (
    PRIMARY_WORK_ENDPOINT,
    QUALITY_GUARD_DIRECTIONS,
    SAVING_SCALE_ENDPOINTS,
    TRAJECTORY_ENDPOINT_DIRECTIONS,
    _verify_score_scope,
)

__all__ = ["build_matched_contrasts", "resolve_arm_semantics"]


_QUALITY_VIEWS = ("strict", "fallback_assisted", "worst_case", "best_case")
_OPERATIONAL_ENDPOINTS = {
    # interval_union_seconds, not service_work_seconds, is the primary work endpoint: a sum of
    # per-request intervals is only a work figure when nothing overlaps, and requiring that made
    # the engine serialize a graph that is natively concurrent. See e2e_effects.PRIMARY_WORK_ENDPOINT.
    "interval_union_seconds": "higher_saving_is_better",
    "service_work_seconds": "higher_saving_is_better",
    "energy_joules": "lower_is_better",
    "e2e_latency_seconds": "lower_is_better",
    "prompt_tokens": "lower_is_better",
    "completion_tokens": "lower_is_better",
    "cached_prompt_tokens": "descriptive_only",
}
_REQUIRED_VARIANT_FIELDS = {
    "node",
    "chunker",
    "scope",
    "contract",
    "aggregation",
    "close_mode",
    "selector_backend",
    "publication_path",
    "output_representation",
}


def resolve_arm_semantics(
    arm_id: str,
    arm: Mapping[str, Any],
    variants: Mapping[str, Mapping[str, Any]],
    executable_fields: Sequence[str],
) -> dict[str, Any]:
    """Resolve one E2E arm without collapsing a joint H+C configuration.

    A one-node matched arm retains the historical scalar field representation.  A joint arm
    stores each executable field as ``{"page": ..., "close": ...}``.  Explicit node keys avoid
    turning list position into an undocumented semantic contract.  This makes H+C-vs-joint-
    control comparisons check both interventions while preserving the existing one-node
    contracts and their frozen level declarations.
    """
    page = str(arm.get("page_variant") or "")
    close = str(arm.get("close_variant") or "")
    if not page or not close:
        raise ValueError(f"matched arm {arm_id!r} lacks page/close variant identity")

    active: list[tuple[str, Mapping[str, Any]]] = []
    for position, variant_id, expected_node in (
        ("page", page, "WEBPAGE_P1"),
        ("close", close, "C_VISIBLE"),
    ):
        if variant_id == "P0":
            continue
        source = variants.get(variant_id)
        if not isinstance(source, Mapping):
            raise ValueError(
                f"matched arm {arm_id!r} names unknown {position} variant {variant_id!r}"
            )
        if str(source.get("node") or "") != expected_node:
            raise ValueError(
                f"matched arm {arm_id!r} uses {variant_id!r} at the wrong {position} node"
            )
        active.append((variant_id, source))
    if not active:
        raise ValueError(f"matched arm {arm_id!r} has no active P1/control variant")

    if len(active) == 1:
        semantics = {
            str(field): active[0][1].get(str(field))
            for field in executable_fields
        }
    else:
        by_position = {
            "page": active[0][1],
            "close": active[1][1],
        }
        semantics = {
            str(field): {
                position: source.get(str(field))
                for position, source in by_position.items()
            }
            for field in executable_fields
        }
    semantics["variant_id"] = f"{page}+{close}"
    return semantics


def _finite(
    value: Any,
    *,
    nonnegative: bool = False,
    strictly_positive: bool = False,
) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if (
        not math.isfinite(number)
        or (nonnegative and number < 0)
        or (strictly_positive and number <= 0)
    ):
        return None
    return number


def _seal(body: dict[str, Any]) -> dict[str, Any]:
    body["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in body.items() if key != "content_sha256"})
    )
    return body


def _input_sha(value: Any) -> str:
    return sha256_hex(canonical_json(value))


def _validate_semantic_registry(
    matched_contrasts: Mapping[str, Any],
    arm_variants: Mapping[str, Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], tuple[str, ...]]:
    if str(matched_contrasts.get("version") or "") != "matched_contrasts_v2":
        raise ValueError("unsupported matched_contrasts version")
    executable_fields = tuple(
        map(str, matched_contrasts.get("executable_variant_fields") or ())
    )
    if (
        not executable_fields
        or len(executable_fields) != len(set(executable_fields))
        or not _REQUIRED_VARIANT_FIELDS <= set(executable_fields)
    ):
        raise ValueError("executable_variant_fields is incomplete or duplicated")
    raw_pairs = matched_contrasts.get("pairs")
    if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, str | bytes) or not raw_pairs:
        raise ValueError("matched_contrasts.pairs must be a non-empty sequence")

    pairs: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for raw in raw_pairs:
        if not isinstance(raw, Mapping):
            raise ValueError("each matched contrast must be a mapping")
        pair = dict(raw)
        contrast_id = str(pair.get("contrast_id") or "")
        left_arm = str(pair.get("left_arm_id") or "")
        right_arm = str(pair.get("right_arm_id") or "")
        target = str(pair.get("target_factor") or "")
        factor_fields = tuple(map(str, pair.get("factor_fields") or ()))
        if (
            not contrast_id
            or contrast_id in seen_ids
            or not left_arm
            or not right_arm
            or left_arm == right_arm
        ):
            raise ValueError("contrast IDs and distinct left/right arm IDs are required")
        seen_ids.add(contrast_id)
        if (
            not factor_fields
            or len(factor_fields) != len(set(factor_fields))
            or target not in factor_fields
            or not set(factor_fields) <= set(executable_fields)
        ):
            raise ValueError(f"{contrast_id}: factor_fields do not define the target factor")
        if left_arm not in arm_variants or right_arm not in arm_variants:
            raise ValueError(f"{contrast_id}: arm semantics are absent from the variant mapping")
        left = dict(arm_variants[left_arm])
        right = dict(arm_variants[right_arm])
        for arm_id, variant in ((left_arm, left), (right_arm, right)):
            if not str(variant.get("variant_id") or ""):
                raise ValueError(f"{contrast_id}: {arm_id} has no variant_id")
            missing = sorted(_REQUIRED_VARIANT_FIELDS - set(variant))
            if missing:
                raise ValueError(
                    f"{contrast_id}: {arm_id} lacks executable semantic fields {missing}"
                )
        left_level = pair.get("left_level")
        right_level = pair.get("right_level")
        if left.get(target) != left_level or right.get(target) != right_level:
            raise ValueError(
                f"{contrast_id}: target-factor levels do not match the arm variant semantics"
            )
        joint_levels = isinstance(left.get(target), Mapping) or isinstance(
            right.get(target), Mapping
        )
        if joint_levels and (
            not isinstance(left.get(target), Mapping)
            or not isinstance(right.get(target), Mapping)
            or pair.get("factor_scope") != "JOINT_E2E_POLICY"
            or list(map(str, pair.get("affected_nodes") or ()))
            != ["WEBPAGE_P1", "C_VISIBLE"]
            or pair.get("first_treatment_boundary") != "WEBPAGE_P1"
        ):
            raise ValueError(
                f"{contrast_id}: joint contrast lacks its ordered E2E intervention scope"
            )
        unexpected_differences = sorted(
            field
            for field in executable_fields
            if field not in factor_fields and left.get(field) != right.get(field)
        )
        if unexpected_differences:
            raise ValueError(
                f"{contrast_id}: variants differ outside factor_fields: "
                f"{unexpected_differences}"
            )
        pair["_factor_fields"] = factor_fields
        pairs.append(pair)
    return pairs, executable_fields


def _scope_issue(
    block_id: str,
    left_scope: Any,
    right_scope: Any,
) -> dict[str, Any] | None:
    if not isinstance(left_scope, Mapping) or not isinstance(right_scope, Mapping):
        return {
            "code": "MISSING_FROZEN_PAIR_SCOPE",
            "block_id": block_id,
        }
    coordinates = (
        "freeze_root_sha256",
        "schedule_sha256",
        "block_freeze_sha256",
        "block_digest",
        "valid_for_paired_estimate",
        "invalid_reason",
        "engine_epochs",
        "engine_epoch_by_arm",
    )
    different = [field for field in coordinates if left_scope.get(field) != right_scope.get(field)]
    if different:
        return {
            "code": "ARM_FROZEN_SCOPE_MISMATCH",
            "block_id": block_id,
            "fields": different,
        }
    validity = left_scope.get("valid_for_paired_estimate")
    if not isinstance(validity, bool):
        return {
            "code": "MISSING_PAIR_VALIDITY",
            "block_id": block_id,
        }
    if not validity:
        return {
            "code": "PAIRED_INVALID_BLOCK",
            "block_id": block_id,
            "reason": str(left_scope.get("invalid_reason") or "UNSPECIFIED"),
            "engine_epochs": list(map(str, left_scope.get("engine_epochs") or ())),
        }
    if str(left_scope.get("invalid_reason") or ""):
        return {
            "code": "INCONSISTENT_PAIR_VALIDITY",
            "block_id": block_id,
            "reason": str(left_scope["invalid_reason"]),
        }
    return None


def _pair_blocks(
    records: Sequence[Mapping[str, Any]],
    *,
    left_arm: str,
    right_arm: str,
    expected_left_variant: str,
    expected_right_variant: str,
    cluster_by_task: Mapping[str, str],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    paired: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    seen_coordinates: set[tuple[str, str]] = set()
    for ordinal, record in enumerate(records):
        block_id = str(record.get("block_id") or f"<record:{ordinal}>")
        task_id = str(record.get("task_id") or "")
        replicate_id = str(record.get("replicate_id") or "")
        coordinate = (task_id, replicate_id)
        if not task_id or not replicate_id:
            issues.append(
                {
                    "code": "MISSING_TASK_REPLICATE_COORDINATE",
                    "block_id": block_id,
                }
            )
            continue
        if coordinate in seen_coordinates:
            issues.append(
                {
                    "code": "DUPLICATE_TASK_REPLICATE_COORDINATE",
                    "block_id": block_id,
                    "task_id": task_id,
                    "replicate_id": replicate_id,
                }
            )
            continue
        seen_coordinates.add(coordinate)
        cluster_id = str(cluster_by_task.get(task_id) or "")
        if not cluster_id:
            issues.append(
                {
                    "code": "MISSING_TASK_CLUSTER",
                    "block_id": block_id,
                    "task_id": task_id,
                }
            )
            continue
        per_arm = record.get("per_arm")
        if not isinstance(per_arm, Mapping):
            issues.append({"code": "MISSING_PER_ARM_SCORES", "block_id": block_id})
            continue
        left_key = f"{left_arm}:{replicate_id}"
        right_key = f"{right_arm}:{replicate_id}"
        left = per_arm.get(left_key)
        right = per_arm.get(right_key)
        missing = [
            arm
            for arm, row in ((left_arm, left), (right_arm, right))
            if not isinstance(row, Mapping)
        ]
        if missing:
            issues.append(
                {
                    "code": "MISSING_ARM",
                    "block_id": block_id,
                    "task_id": task_id,
                    "replicate_id": replicate_id,
                    "arm_ids": missing,
                }
            )
            continue
        left = dict(left)
        right = dict(right)
        row_mismatches = []
        for arm_id, expected_variant, row in (
            (left_arm, expected_left_variant, left),
            (right_arm, expected_right_variant, right),
        ):
            if row.get("arm_id") not in (None, arm_id):
                row_mismatches.append(f"{arm_id}:arm_id")
            if row.get("task_id") not in (None, task_id):
                row_mismatches.append(f"{arm_id}:task_id")
            if row.get("replicate_id") not in (None, replicate_id):
                row_mismatches.append(f"{arm_id}:replicate_id")
            if str(row.get("variant_id") or "") != expected_variant:
                row_mismatches.append(f"{arm_id}:variant_id")
        if row_mismatches:
            issues.append(
                {
                    "code": "ARM_ROW_SEMANTIC_MISMATCH",
                    "block_id": block_id,
                    "fields": row_mismatches,
                    "expected_variant_ids": {
                        left_arm: expected_left_variant,
                        right_arm: expected_right_variant,
                    },
                    "observed_variant_ids": {
                        left_arm: str(left.get("variant_id") or ""),
                        right_arm: str(right.get("variant_id") or ""),
                    },
                }
            )
            continue
        scope_problem = _scope_issue(
            block_id,
            left.get("frozen_scope"),
            right.get("frozen_scope"),
        )
        if scope_problem is not None:
            issues.append(scope_problem)
            continue
        paired.append(
            {
                "block_id": block_id,
                "task_id": task_id,
                "replicate_id": replicate_id,
                "cluster_id": cluster_id,
                "left": left,
                "right": right,
            }
        )
    return paired, issues


def _quality_measurement(
    row: Mapping[str, Any],
    *,
    view: str,
    metric: str,
) -> tuple[float | None, str]:
    views = row.get("quality_views")
    selected = views.get(view) if isinstance(views, Mapping) else None
    if not isinstance(selected, Mapping) or metric not in selected:
        return None, "MISSING"
    if selected[metric] is None:
        return None, "INAPPLICABLE"
    value = _finite(selected[metric])
    return (value, "VALUE") if value is not None else (None, "MISSING")


def _operational_measurement(
    row: Mapping[str, Any],
    *,
    endpoint: str,
) -> tuple[float | None, str]:
    summary = row.get("work_summary")
    if not isinstance(summary, Mapping):
        return None, "MISSING"
    # Telemetry completeness and interval non-overlap are separate questions. Completeness asks
    # whether every request was observed; overlap asks only whether they happened to be
    # serialized, which nothing but the summed-service endpoint depends on. Conflating them
    # dropped token counts -- which have no scheduling dependence at all -- for every cell in
    # any concurrent regime.
    if summary.get("telemetry_complete") is not True:
        return None, "MISSING"
    if endpoint == "service_work_seconds":
        if summary.get("overlap_valid") is not True:
            return None, "MISSING"
        value = _finite(summary.get("service_seconds"), strictly_positive=True)
    elif endpoint == "interval_union_seconds":
        value = _finite(summary.get("interval_union_seconds"), strictly_positive=True)
    elif endpoint == "energy_joules":
        value = _finite(summary.get("energy_joules"), strictly_positive=True)
    elif endpoint == "e2e_latency_seconds":
        raw = row.get("e2e_latency_seconds")
        if raw is None:
            raw = summary.get("e2e_latency_seconds")
        value = _finite(raw, nonnegative=True)
    else:
        tokens = summary.get("tokens")
        value = (
            _finite(tokens.get(endpoint), nonnegative=True)
            if isinstance(tokens, Mapping)
            else None
        )
    return (value, "VALUE") if value is not None else (None, "MISSING")


def _trajectory_measurement(
    row: Mapping[str, Any],
    *,
    endpoint: str,
) -> tuple[float | None, str]:
    trajectory = row.get("trajectory_metrics")
    if not isinstance(trajectory, Mapping) or trajectory.get("status") != "OK":
        return None, "MISSING"
    value = _finite(trajectory.get(endpoint), nonnegative=True)
    if (
        endpoint in {"query_redundancy", "retrieval_empty_rate"}
        and value is not None
        and value > 1.0
    ):
        value = None
    return (value, "VALUE") if value is not None else (None, "MISSING")


def _not_estimable_endpoint(
    *,
    direction: str,
    scale: str,
    reason: str,
    structural_issues: Sequence[Mapping[str, Any]] = (),
    missing: Sequence[Mapping[str, Any]] = (),
    not_applicable_blocks: Sequence[str] = (),
) -> dict[str, Any]:
    counts = Counter(
        str(item["side"])
        for item in missing
        if item.get("side") in {"left", "right"}
    )
    return {
        "status": "NOT_ESTIMABLE",
        "direction": direction,
        "contrast_scale": scale,
        "reason": reason,
        "missing_counts": {
            "total": len(missing),
            "by_side": {"left": counts["left"], "right": counts["right"]},
            "details": [dict(item) for item in missing],
        },
        "not_applicable_blocks": sorted(map(str, not_applicable_blocks)),
        "structural_issues": [dict(item) for item in structural_issues],
    }


def _task_means(
    measured: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in measured:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    result: list[dict[str, Any]] = []
    for task_id, rows in sorted(by_task.items()):
        clusters = {str(row["cluster_id"]) for row in rows}
        if len(clusters) != 1:
            raise ValueError(f"task {task_id} belongs to multiple source/topic clusters")
        result.append(
            {
                "task_id": task_id,
                "cluster_id": next(iter(clusters)),
                "replicates": len(rows),
                "left": sum(float(row["left_value"]) for row in rows) / len(rows),
                "right": sum(float(row["right_value"]) for row in rows) / len(rows),
            }
        )
    return result


def _verified_first_boundary_receipt(raw: Any) -> dict[str, Any] | None:
    if not isinstance(raw, Mapping):
        return None
    body = dict(raw)
    if body.get("schema_version") != "first_boundary_input_receipt_v1":
        return None
    recorded = str(body.pop("content_sha256", "") or "")
    if not recorded or recorded != sha256_hex(canonical_json(body)):
        return None
    body["content_sha256"] = recorded
    return body


def _first_boundary_input_comparability(
    paired: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Prove equality only at the first treatment boundary.

    Later candidate views, checkpoints, queries and reasoning are post-treatment mediators in
    the E2E estimand.  Requiring them to remain equal would condition on an effect of P1.
    """
    details: list[dict[str, Any]] = []
    compared_fields = (
        "node",
        "candidate_input_count",
        "candidate_inputs",
    )
    for block in paired:
        left = _verified_first_boundary_receipt(
            block["left"].get("first_boundary_input")
        )
        right = _verified_first_boundary_receipt(
            block["right"].get("first_boundary_input")
        )
        issue = ""
        if left is None or right is None:
            issue = "MISSING_OR_NONVERIFYING_RECEIPT"
        elif left.get("status") != "OK" or right.get("status") != "OK":
            issue = "FIRST_BOUNDARY_RECEIPT_NOT_OK"
        else:
            unequal = [
                field for field in compared_fields if left.get(field) != right.get(field)
            ]
            if unequal:
                issue = "REALIZED_FIRST_BOUNDARY_INPUT_MISMATCH"
        if issue:
            details.append(
                {
                    "block_id": str(block["block_id"]),
                    "task_id": str(block["task_id"]),
                    "replicate_id": str(block["replicate_id"]),
                    "issue": issue,
                    "left_receipt_sha256": str((left or {}).get("content_sha256") or ""),
                    "right_receipt_sha256": str((right or {}).get("content_sha256") or ""),
                    "unequal_fields": (
                        [
                            field
                            for field in compared_fields
                            if left is not None
                            and right is not None
                            and left.get(field) != right.get(field)
                        ]
                        if issue == "REALIZED_FIRST_BOUNDARY_INPUT_MISMATCH"
                        else []
                    ),
                }
            )
    return {
        "schema_version": "first_boundary_input_comparability_v1",
        "status": "OK" if paired and not details else "NOT_ESTABLISHED",
        "scope": "COMPLETE_ORDERED_FIRST_ATOMIC_H_OR_C_BOUNDARY_INPUT_SET",
        "compared_fields": list(compared_fields),
        "blocks_offered": len(paired),
        "blocks_equal": len(paired) - len(details),
        "issues": details,
        "later_boundary_policy": "MEDIATED_E2E_OUTCOME_NOT_PAIRING_FILTER",
    }


def _paired_task_values(
    paired: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    quality_metric: str | None = None,
) -> tuple[list[float], list[float], list[str], dict[str, Any]]:
    """Return task-level left/right means for one complete paired estimand."""
    measured: list[dict[str, Any]] = []
    inapplicable: list[str] = []
    missing: list[dict[str, str]] = []
    for block in paired:
        if quality_metric is not None:
            left, left_state = _quality_measurement(
                block["left"], view="strict", metric=quality_metric
            )
            right, right_state = _quality_measurement(
                block["right"], view="strict", metric=quality_metric
            )
            if left_state == right_state == "INAPPLICABLE":
                inapplicable.append(str(block["block_id"]))
                continue
        else:
            left, left_state = _operational_measurement(
                block["left"], endpoint=endpoint
            )
            right, right_state = _operational_measurement(
                block["right"], endpoint=endpoint
            )
        for side, value, state in (
            ("left", left, left_state),
            ("right", right, right_state),
        ):
            if value is None:
                missing.append(
                    {
                        "block_id": str(block["block_id"]),
                        "side": side,
                        "state": state,
                    }
                )
        if left is not None and right is not None:
            measured.append(
                {
                    **block,
                    "left_value": left,
                    "right_value": right,
                }
            )
    if missing:
        return [], [], [], {
            "status": "NOT_ESTIMABLE",
            "reason": "PARTIAL_PAIR_MEASUREMENT_MISSING_FAIL_CLOSED",
            "missing": missing,
        }
    if not measured:
        return [], [], [], {
            "status": "NOT_APPLICABLE" if inapplicable else "NOT_ESTIMABLE",
            "reason": (
                "TRUTH_GUARD_INAPPLICABLE_FOR_BOTH_ARMS"
                if inapplicable
                else "ENDPOINT_MISSING_IN_ALL_BLOCKS"
            ),
            "not_applicable_blocks": sorted(inapplicable),
        }
    task_rows = _task_means(measured)
    clusters = [str(row["cluster_id"]) for row in task_rows]
    if len(set(clusters)) < 2:
        return [], [], [], {
            "status": "NOT_ESTIMABLE",
            "reason": "FEWER_THAN_TWO_ESTIMABLE_SOURCE_TOPIC_CLUSTERS",
        }
    return (
        [float(row["left"]) for row in task_rows],
        [float(row["right"]) for row in task_rows],
        clusters,
        {
            "status": "OK",
            "tasks_estimable": len(task_rows),
            "clusters_estimable": len(set(clusters)),
            "not_applicable_blocks": sorted(inapplicable),
        },
    )


def _cluster_bootstrap_samples(
    values: Sequence[float],
    cluster_ids: Sequence[str],
    statistic: Callable[[np.ndarray], float],
    *,
    n_boot: int,
    seed: int,
) -> tuple[float, np.ndarray]:
    """Mirror ``cluster_bootstrap_ci`` while retaining the resamples for Holm inversion."""
    by_cluster: dict[str, list[float]] = {}
    for value, cluster_id in zip(values, cluster_ids, strict=True):
        by_cluster.setdefault(str(cluster_id), []).append(float(value))
    clusters = sorted(by_cluster)
    if len(clusters) < 2:
        raise ValueError("need at least two clusters for structured-increment bootstrap")
    cluster_values = np.asarray(
        [float(np.mean(by_cluster[cluster])) for cluster in clusters],
        dtype=float,
    )
    point = float(statistic(cluster_values))
    rng = np.random.default_rng(seed)
    samples = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        draw = rng.integers(0, len(clusters), size=len(clusters))
        samples[index] = statistic(cluster_values[draw])
    return point, samples


def _effect_bootstrap(
    paired: Sequence[Mapping[str, Any]],
    *,
    effect: str,
    n_boot: int,
    seed: int,
    quality_metric: str | None = None,
) -> dict[str, Any]:
    left, right, clusters, status = _paired_task_values(
        paired,
        endpoint=PRIMARY_WORK_ENDPOINT if effect == "work_saving" else "quality",
        quality_metric=quality_metric,
    )
    if status["status"] != "OK":
        return status
    if effect == "work_saving":
        if any(value <= 0 for value in (*left, *right)):
            return {
                "status": "NOT_ESTIMABLE",
                "reason": "NONPOSITIVE_COMPLETE_SERVICE_WORK",
            }
        values = np.log(np.asarray(left, dtype=float) / np.asarray(right, dtype=float))

        def statistic(rows: np.ndarray) -> float:
            return 1.0 - float(np.exp(np.mean(rows)))

        unit = "fraction_complete_service_work_saved_left_relative_to_right"
    elif effect == "quality_difference":
        values = np.asarray(left, dtype=float) - np.asarray(right, dtype=float)

        def statistic(rows: np.ndarray) -> float:
            return float(np.mean(rows))

        unit = "fraction_left_minus_right"
    else:
        raise ValueError(f"unknown structured-increment effect {effect!r}")
    point, samples = _cluster_bootstrap_samples(
        values,
        clusters,
        statistic,
        n_boot=n_boot,
        seed=seed,
    )
    return {
        **status,
        "point": point,
        "samples": samples,
        "unit": unit,
        "bootstrap_unit": "source_topic_cluster",
        "replicate_reduction": "task_mean_before_cluster_bootstrap",
        "n_boot": n_boot,
    }


def _structured_raw_contract_adverse(row: Mapping[str, Any]) -> tuple[bool, list[str]]:
    """Fail closed over every applicable structured selector node."""

    reasons: list[str] = []
    direct = row.get("direct_node_metrics")
    by_node = direct.get("by_node") if isinstance(direct, Mapping) else None
    nodes = [
        (str(node), value)
        for node, value in (by_node.items() if isinstance(by_node, Mapping) else ())
        if isinstance(value, Mapping) and value.get("applicable") is True
    ]
    if not nodes:
        reasons.append("NO_APPLICABLE_STRUCTURED_SELECTOR_NODE")
    for node, metrics in nodes:
        summary = metrics.get("selector_normalization")
        if (
            not isinstance(summary, Mapping)
            or summary.get("status") != "OK"
            or type(summary.get("no_repair_adverse")) is not bool
        ):
            reasons.append(f"{node}:SELECTOR_NORMALIZATION_TRACE_NOT_ESTABLISHED")
            continue
        if summary["no_repair_adverse"]:
            reasons.append(f"{node}:SELECTOR_RAW_CONTRACT_NONADHERENT")
    if row.get("assignment_state") not in {None, "COMMITTED"} or row.get("fell_back"):
        reasons.append("STRUCTURED_ASSIGNMENT_FAILED_OR_FELL_BACK")
    return bool(reasons), reasons


def _prose_control_state(row: Mapping[str, Any]) -> dict[str, Any]:
    summary = row.get("prose_control_normalization")
    reasons: list[str] = []
    if not isinstance(summary, Mapping):
        return {
            "integrity_ok": False,
            "raw_contract_adverse": True,
            "reasons": ["PROSE_CONTROL_NORMALIZATION_SUMMARY_MISSING"],
            "summary": {},
        }
    if summary.get("schema_version") != "prose_control_normalization_summary_v1":
        reasons.append("PROSE_CONTROL_NORMALIZATION_SCHEMA_MISMATCH")
    if summary.get("status") != "OK" or summary.get("trace_complete") is not True:
        reasons.append("PROSE_CONTROL_NORMALIZATION_TRACE_INCOMPLETE")
    attempts = summary.get("attempt_count")
    if isinstance(attempts, bool) or not isinstance(attempts, int) or attempts <= 0:
        reasons.append("PROSE_CONTROL_HAS_NO_OBSERVED_ATTEMPT")
    semantic_repairs = summary.get("semantic_repair_count")
    if (
        isinstance(semantic_repairs, bool)
        or not isinstance(semantic_repairs, int)
        or semantic_repairs != 0
    ):
        reasons.append("PROSE_CONTROL_SEMANTIC_REPAIR_PRESENT_OR_UNVERIFIABLE")
    raw_adverse = summary.get("raw_contract_adverse")
    if type(raw_adverse) is not bool:
        reasons.append("PROSE_CONTROL_RAW_CONTRACT_STATE_MISSING")
        raw_adverse = True
    if row.get("assignment_state") not in {None, "COMMITTED"} or row.get("fell_back"):
        raw_adverse = True
    return {
        "integrity_ok": not reasons,
        "raw_contract_adverse": bool(raw_adverse),
        "reasons": reasons,
        "summary": dict(summary),
    }


def _prose_integrity_gate(
    paired: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    details: list[dict[str, Any]] = []
    raw_nonadherent_blocks = 0
    structured_nonadherent_blocks = 0
    for block in paired:
        prose = _prose_control_state(block["right"])
        structured_adverse, structured_reasons = _structured_raw_contract_adverse(
            block["left"]
        )
        raw_nonadherent_blocks += int(prose["raw_contract_adverse"])
        structured_nonadherent_blocks += int(structured_adverse)
        if not prose["integrity_ok"] or prose["raw_contract_adverse"] or structured_adverse:
            details.append(
                {
                    "block_id": str(block["block_id"]),
                    "task_id": str(block["task_id"]),
                    "replicate_id": str(block["replicate_id"]),
                    "prose_integrity_reasons": list(prose["reasons"]),
                    "prose_raw_contract_adverse": bool(
                        prose["raw_contract_adverse"]
                    ),
                    "structured_raw_contract_adverse": structured_adverse,
                    "structured_raw_contract_reasons": structured_reasons,
                }
            )
    integrity_ok = bool(paired) and all(
        _prose_control_state(block["right"])["integrity_ok"]
        for block in paired
    )
    return {
        "schema_version": "prose_control_integrity_gate_v1",
        "status": "PASS" if integrity_ok else "NOT_ESTABLISHED",
        "trace_complete_all_offered": integrity_ok,
        "semantic_repair_count": sum(
            value
            for block in paired
            for value in [
                (_prose_control_state(block["right"])["summary"]).get(
                    "semantic_repair_count", 0
                )
            ]
            if isinstance(value, int) and not isinstance(value, bool) and value >= 0
        ),
        "blocks_offered": len(paired),
        "prose_raw_contract_nonadherent_blocks": raw_nonadherent_blocks,
        "structured_raw_contract_nonadherent_blocks": structured_nonadherent_blocks,
        "raw_contract_adherence_all_offered": bool(paired)
        and raw_nonadherent_blocks == 0
        and structured_nonadherent_blocks == 0,
        "details": details,
    }


def _raw_contract_quality_bootstrap(
    paired: Sequence[Mapping[str, Any]],
    *,
    metric: str,
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Adverse all-offered bound: dirty ID is worst, dirty prose is best."""

    measured: list[dict[str, Any]] = []
    inapplicable: list[str] = []
    for block in paired:
        left, left_state = _quality_measurement(
            block["left"], view="strict", metric=metric
        )
        right, right_state = _quality_measurement(
            block["right"], view="strict", metric=metric
        )
        if left_state == right_state == "INAPPLICABLE":
            inapplicable.append(str(block["block_id"]))
            continue
        if left_state == "INAPPLICABLE" or right_state == "INAPPLICABLE":
            return {
                "status": "NOT_ESTIMABLE",
                "reason": "ASYMMETRIC_QUALITY_APPLICABILITY",
            }
        structured_adverse, _ = _structured_raw_contract_adverse(block["left"])
        prose = _prose_control_state(block["right"])
        left_dirty = structured_adverse or left is None
        right_dirty = prose["raw_contract_adverse"] or right is None
        if metric == "critical_harm":
            bounded_left = 1.0 if left_dirty else float(left)
            bounded_right = 0.0 if right_dirty else float(right)
        else:
            bounded_left = 0.0 if left_dirty else float(left)
            bounded_right = 1.0 if right_dirty else float(right)
        measured.append(
            {
                **block,
                "left_value": bounded_left,
                "right_value": bounded_right,
            }
        )
    if not measured:
        return {
            "status": "NOT_APPLICABLE" if inapplicable else "NOT_ESTIMABLE",
            "reason": (
                "TRUTH_GUARD_INAPPLICABLE_FOR_BOTH_ARMS"
                if inapplicable
                else "ENDPOINT_MISSING_IN_ALL_BLOCKS"
            ),
        }
    task_rows = _task_means(measured)
    clusters = [str(row["cluster_id"]) for row in task_rows]
    if len(set(clusters)) < 2:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "FEWER_THAN_TWO_ESTIMABLE_SOURCE_TOPIC_CLUSTERS",
        }
    values = np.asarray(
        [float(row["left"]) - float(row["right"]) for row in task_rows],
        dtype=float,
    )
    point, samples = _cluster_bootstrap_samples(
        values,
        clusters,
        lambda rows: float(np.mean(rows)),
        n_boot=n_boot,
        seed=seed,
    )
    return {
        "status": "OK",
        "point": point,
        "samples": samples,
        "unit": "fraction_left_minus_right_adverse_raw_contract_bound",
        "tasks_estimable": len(task_rows),
        "clusters_estimable": len(set(clusters)),
        "not_applicable_blocks": sorted(inapplicable),
    }


def _estimate_endpoint(
    paired: Sequence[Mapping[str, Any]],
    *,
    endpoint: str,
    direction: str,
    n_boot: int,
    seed: int,
    quality_view: str | None = None,
    quality_metric: str | None = None,
) -> dict[str, Any]:
    measured: list[dict[str, Any]] = []
    missing: list[dict[str, Any]] = []
    not_applicable: list[str] = []
    for block in paired:
        if quality_view is not None and quality_metric is not None:
            left_value, left_state = _quality_measurement(
                block["left"], view=quality_view, metric=quality_metric
            )
            right_value, right_state = _quality_measurement(
                block["right"], view=quality_view, metric=quality_metric
            )
            if left_state == right_state == "INAPPLICABLE":
                not_applicable.append(str(block["block_id"]))
                continue
        else:
            measurement = (
                _trajectory_measurement
                if endpoint in TRAJECTORY_ENDPOINT_DIRECTIONS
                else _operational_measurement
            )
            left_value, left_state = measurement(block["left"], endpoint=endpoint)
            right_value, right_state = measurement(block["right"], endpoint=endpoint)
        for side, value, state in (
            ("left", left_value, left_state),
            ("right", right_value, right_state),
        ):
            if value is None:
                missing.append(
                    {
                        "block_id": str(block["block_id"]),
                        "task_id": str(block["task_id"]),
                        "replicate_id": str(block["replicate_id"]),
                        "side": side,
                        "state": state,
                    }
                )
        if left_value is not None and right_value is not None:
            measured.append(
                {
                    **block,
                    "left_value": left_value,
                    "right_value": right_value,
                }
            )
    scale = (
        "paired_log_ratio_saving"
        if endpoint in SAVING_SCALE_ENDPOINTS
        else "left_minus_right"
    )
    if missing:
        return _not_estimable_endpoint(
            direction=direction,
            scale=scale,
            reason="PARTIAL_PAIR_MEASUREMENT_MISSING_FAIL_CLOSED",
            missing=missing,
            not_applicable_blocks=not_applicable,
        )
    if not measured:
        if quality_view is not None and not_applicable:
            return {
                "status": "NOT_APPLICABLE",
                "direction": direction,
                "contrast_scale": scale,
                "reason": "TRUTH_GUARD_INAPPLICABLE_FOR_BOTH_ARMS",
                "not_applicable_blocks": sorted(not_applicable),
                "missing_counts": {
                    "total": 0,
                    "by_side": {"left": 0, "right": 0},
                    "details": [],
                },
            }
        return _not_estimable_endpoint(
            direction=direction,
            scale=scale,
            reason="ENDPOINT_MISSING_IN_ALL_BLOCKS",
        )
    task_rows = _task_means(measured)
    clusters = [str(row["cluster_id"]) for row in task_rows]
    if len(set(clusters)) < 2:
        return _not_estimable_endpoint(
            direction=direction,
            scale=scale,
            reason="FEWER_THAN_TWO_ESTIMABLE_SOURCE_TOPIC_CLUSTERS",
            not_applicable_blocks=not_applicable,
        )
    if endpoint in SAVING_SCALE_ENDPOINTS:
        ci = paired_log_ratio_saving(
            [float(row["left"]) for row in task_rows],
            [float(row["right"]) for row in task_rows],
            clusters,
            n_boot=n_boot,
            seed=seed,
        )
        interval = {
            "point": float(ci.point),
            "lower": float(ci.lower),
            "upper": None,
            "one_sided_95_lcb": float(ci.lower),
        }
    else:
        differences = [
            float(row["left"]) - float(row["right"])
            for row in task_rows
        ]
        ci = cluster_bootstrap_ci(
            differences,
            clusters,
            np.mean,
            n_boot=n_boot,
            seed=seed,
            side="two",
        )
        interval = {
            "point": float(ci.point),
            "lower": float(ci.lower),
            "upper": float(ci.upper),
        }
    return {
        "status": "OK",
        "direction": direction,
        "contrast_scale": scale,
        "contrast_convention": (
            "positive_means_left_saves_work_relative_to_right"
            if endpoint in SAVING_SCALE_ENDPOINTS
            else "left_arm_minus_right_arm"
        ),
        "replicate_reduction": "task_mean_before_cluster_bootstrap",
        "tasks_estimable": len(task_rows),
        "clusters_estimable": len(set(clusters)),
        "replicates_by_task": {
            str(row["task_id"]): int(row["replicates"])
            for row in task_rows
        },
        "not_applicable_blocks": sorted(not_applicable),
        "missing_counts": {
            "total": 0,
            "by_side": {"left": 0, "right": 0},
            "details": [],
        },
        "n_boot": int(ci.n_boot),
        "n_clusters": int(ci.n_clusters),
        **interval,
    }


def _structurally_not_estimable(
    *,
    direction: str,
    endpoint: str,
    issues: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    return _not_estimable_endpoint(
        direction=direction,
        scale=(
            "paired_log_ratio_saving"
            if endpoint in SAVING_SCALE_ENDPOINTS
            else "left_minus_right"
        ),
        reason="ALL_OFFERED_CONTAINS_STRUCTURALLY_INVALID_PAIR",
        structural_issues=issues,
    )


def _percentile_bootstrap_one_sided_p(
    samples: np.ndarray,
    *,
    threshold: float,
) -> float:
    """Invert the percentile-bootstrap lower bound for ``H1: theta > threshold``.

    The structured-increment gate reports a lower percentile bound from these exact
    bootstrap draws.  Its raw p-value must therefore use the matching lower tail:
    the fraction of bootstrap estimates at or below the frozen null threshold.
    A centered-null upper-tail calculation would instead invert a basic-bootstrap
    interval and can disagree materially for the skewed work-saving estimand.
    """

    finite = np.asarray(samples, dtype=float)
    if finite.ndim != 1 or len(finite) == 0 or not np.all(np.isfinite(finite)):
        raise ValueError("one-sided bootstrap samples must be a finite non-empty vector")
    return float(
        (1 + np.count_nonzero(finite <= float(threshold))) / (len(finite) + 1)
    )


def _build_structured_increment_gates(
    *,
    paired_by_contrast: Mapping[str, Sequence[Mapping[str, Any]]],
    contrast_results: Sequence[Mapping[str, Any]],
    matched_contrasts: Mapping[str, Any],
    policy: Mapping[str, Any],
    n_boot: int,
    seed: int,
) -> dict[str, Any]:
    """Build the executable six-member Holm family required by the final decision.

    The six primary tests are jointly frozen before outcomes: standalone H, standalone C, and
    joint H+C each receive an LLM-vs-CPU strict-recall test and an ID-vs-prose complete-work
    saving test.  Each raw tail probability and Holm-rank LCB is derived from the same paired-
    task/source-cluster bootstrap; an ordinary descriptive CI is never reused as a Holm result.
    """
    schema = str(policy.get("schema_version") or "")
    family = policy.get("multiplicity_family")
    estimands = policy.get("control_estimands")
    ni_policy = policy.get("strict_quality_noninferiority")
    by_node_policy = policy.get("by_node")
    if (
        schema != "structured_increment_gate_v3"
        or not isinstance(family, Mapping)
        or family.get("method") != "holm_one_sided_across_six_primary_tests"
        or not isinstance(estimands, Mapping)
        or set(estimands) != {"LLM_vs_CPU", "structured_selection_vs_prose"}
        or not isinstance(ni_policy, Mapping)
        or not isinstance(by_node_policy, Mapping)
    ):
        raise ValueError("structured-increment producer policy is incomplete")
    family_id = str(family.get("family_id") or "")
    alpha = _finite(family.get("familywise_alpha"))
    member_ids = tuple(map(str, family.get("member_contrast_ids") or ()))
    if (
        not family_id
        or alpha is None
        or not 0.0 < alpha < 1.0
        or len(member_ids) != 6
        or len(set(member_ids)) != 6
    ):
        raise ValueError("structured-increment Holm family must have six unique members")
    higher_margins = ni_policy.get("higher_is_better_min_effect")
    lower_margins = ni_policy.get("lower_is_better_max_effect")
    if (
        not isinstance(higher_margins, Mapping)
        or not higher_margins
        or not isinstance(lower_margins, Mapping)
        or not lower_margins
        or set(higher_margins) & set(lower_margins)
        or not (
            (set(higher_margins) | set(lower_margins))
            <= set(QUALITY_GUARD_DIRECTIONS)
        )
    ):
        raise ValueError("structured-increment strict NI metric family is incomplete")

    design_by_id = {
        str(item.get("contrast_id") or ""): item
        for item in (matched_contrasts.get("pairs") or ())
        if isinstance(item, Mapping)
    }
    result_by_id = {
        str(item.get("contrast_id") or ""): item
        for item in contrast_results
        if isinstance(item, Mapping)
    }
    if set(member_ids) - set(design_by_id):
        raise ValueError("structured-increment Holm member is absent from matched design")
    nodes = ("WEBPAGE_P1", "C_VISIBLE", "H_PLUS_C_VISIBLE")
    declared_members = {
        str(contrast_id)
        for node in nodes
        for contrast_id in (
            (by_node_policy.get(node) or {}).get("required_contrast_ids") or ()
        )
        if isinstance(by_node_policy.get(node), Mapping)
    }
    if declared_members != set(member_ids):
        raise ValueError(
            "structured-increment Holm family must cover both controls for H, C, and H+C"
        )

    internal: dict[str, dict[str, Any]] = {}
    for contrast_id in member_ids:
        design = design_by_id[contrast_id]
        kind = str(design.get("kind") or "")
        if kind not in {"LLM_vs_CPU", "structured_selection_vs_prose"}:
            raise ValueError(
                f"structured-increment member {contrast_id} has unsupported kind {kind!r}"
            )
        result = result_by_id.get(contrast_id)
        paired = paired_by_contrast.get(contrast_id) or ()
        comparability = (
            result.get("first_boundary_input_comparability")
            if isinstance(result, Mapping)
            else None
        )
        if (
            not isinstance(result, Mapping)
            or result.get("pairing_status") != "OK"
            or not isinstance(comparability, Mapping)
            or comparability.get("status") != "OK"
        ):
            internal[contrast_id] = {
                "status": "NOT_ESTIMABLE",
                "reason": (
                    "PAIRING_OR_REALIZED_FIRST_BOUNDARY_INPUT_COMPARABILITY_NOT_ESTABLISHED"
                ),
                "kind": kind,
                "comparability": dict(comparability or {}),
            }
            continue
        contract = estimands[kind]
        if not isinstance(contract, Mapping):
            raise ValueError(f"structured-increment estimand contract for {kind} is absent")
        if kind == "LLM_vs_CPU":
            primary = _effect_bootstrap(
                paired,
                effect="quality_difference",
                quality_metric="weighted_required_atom_recall",
                n_boot=n_boot,
                seed=seed_from(str(seed), family_id, contrast_id, "primary"),
            )
            threshold = (
                float(contract.get("adjusted_lcb_min_pp")) / 100.0
                if _finite(contract.get("adjusted_lcb_min_pp")) is not None
                else None
            )
            primary_unit = "fraction_left_llm_minus_right_cpu"
        else:
            primary = _effect_bootstrap(
                paired,
                effect="work_saving",
                n_boot=n_boot,
                seed=seed_from(str(seed), family_id, contrast_id, "primary"),
            )
            threshold = _finite(contract.get("adjusted_lcb_min"))
            primary_unit = "fraction_complete_service_work_saved"
        if threshold is None or primary.get("status") != "OK":
            internal[contrast_id] = {
                "status": "NOT_ESTIMABLE",
                "reason": "PRIMARY_HOLM_ESTIMAND_NOT_ESTIMABLE",
                "kind": kind,
                "primary": {
                    key: value
                    for key, value in primary.items()
                    if key != "samples"
                },
            }
            continue
        point = float(primary["point"])
        samples = np.asarray(primary["samples"], dtype=float)
        # This is the one-sided p-value obtained by inverting the same percentile-bootstrap
        # lower bound reported below.  Keeping the p-value and confidence bound in one
        # bootstrap family is essential for skewed complete-work-saving effects.
        raw_p = _percentile_bootstrap_one_sided_p(
            samples,
            threshold=threshold,
        )
        internal[contrast_id] = {
            "status": "ESTIMABLE",
            "kind": kind,
            "threshold": threshold,
            "point": point,
            "samples": samples,
            "raw_one_sided_p": raw_p,
            "primary_unit": primary_unit,
            "tasks_estimable": primary["tasks_estimable"],
            "clusters_estimable": primary["clusters_estimable"],
            "comparability": dict(comparability),
        }

    all_primary_estimable = all(
        internal.get(contrast_id, {}).get("status") == "ESTIMABLE"
        for contrast_id in member_ids
    )
    ordered: list[str] = []
    if all_primary_estimable:
        ordered = sorted(
            member_ids,
            key=lambda contrast_id: (
                float(internal[contrast_id]["raw_one_sided_p"]),
                contrast_id,
            ),
        )
        continue_rejecting = True
        for rank, contrast_id in enumerate(ordered, start=1):
            member = internal[contrast_id]
            critical_alpha = float(alpha) / (len(member_ids) - rank + 1)
            raw_p = float(member["raw_one_sided_p"])
            rejected = continue_rejecting and raw_p <= critical_alpha
            if not rejected:
                continue_rejecting = False
            adjusted_lcb = float(
                np.quantile(
                    member["samples"],
                    critical_alpha,
                    method="lower",
                )
            )
            member.update(
                {
                    "holm_rank": rank,
                    "holm_critical_alpha": critical_alpha,
                    "holm_rejected": rejected,
                    "holm_rank_adjusted_lcb": adjusted_lcb,
                    "adjusted_primary_gate_pass": (
                        rejected and adjusted_lcb > float(member["threshold"])
                    ),
                }
            )
    else:
        for contrast_id in member_ids:
            internal[contrast_id].update(
                {
                    "holm_rank": None,
                    "holm_critical_alpha": None,
                    "holm_rejected": False,
                    "holm_rank_adjusted_lcb": None,
                    "adjusted_primary_gate_pass": False,
                    "family_failure_reason": "ONE_OR_MORE_FAMILY_MEMBERS_NOT_ESTIMABLE",
                }
            )

    component_by_id: dict[str, dict[str, Any]] = {}
    for contrast_id in member_ids:
        design = design_by_id[contrast_id]
        kind = str(design["kind"])
        contract = estimands[kind]
        paired = paired_by_contrast.get(contrast_id) or ()
        primary = internal[contrast_id]
        ni_results: dict[str, dict[str, Any]] = {}
        ni_pass = True
        for metric, raw_margin in sorted(higher_margins.items()):
            margin = _finite(raw_margin)
            if margin is None:
                raise ValueError(f"strict NI margin for {metric} is invalid")
            estimate = _effect_bootstrap(
                paired,
                effect="quality_difference",
                quality_metric=str(metric),
                n_boot=n_boot,
                seed=seed_from(str(seed), contrast_id, "strict-ni", str(metric)),
            )
            if estimate.get("status") == "NOT_APPLICABLE":
                ni_results[str(metric)] = {
                    "status": "NOT_APPLICABLE_PASS",
                    "reason": estimate.get("reason"),
                    "margin": margin,
                }
                continue
            if estimate.get("status") != "OK":
                ni_pass = False
                ni_results[str(metric)] = {
                    "status": "NOT_ESTABLISHED",
                    "reason": estimate.get("reason"),
                    "margin": margin,
                }
                continue
            lower = float(np.quantile(estimate["samples"], 0.05, method="lower"))
            passed = lower >= margin
            ni_pass = ni_pass and passed
            ni_results[str(metric)] = {
                "status": "PASS" if passed else "FAIL",
                "point": float(estimate["point"]),
                "one_sided_95_lcb": lower,
                "margin": margin,
                "unit": "fraction_left_minus_right",
            }
        for metric, raw_margin in sorted(lower_margins.items()):
            margin = _finite(raw_margin)
            if margin is None:
                raise ValueError(f"strict NI margin for {metric} is invalid")
            estimate = _effect_bootstrap(
                paired,
                effect="quality_difference",
                quality_metric=str(metric),
                n_boot=n_boot,
                seed=seed_from(str(seed), contrast_id, "strict-ni", str(metric)),
            )
            if estimate.get("status") == "NOT_APPLICABLE":
                ni_results[str(metric)] = {
                    "status": "NOT_APPLICABLE_PASS",
                    "reason": estimate.get("reason"),
                    "margin": margin,
                }
                continue
            if estimate.get("status") != "OK":
                ni_pass = False
                ni_results[str(metric)] = {
                    "status": "NOT_ESTABLISHED",
                    "reason": estimate.get("reason"),
                    "margin": margin,
                }
                continue
            upper = float(np.quantile(estimate["samples"], 0.95, method="higher"))
            passed = upper <= margin
            ni_pass = ni_pass and passed
            ni_results[str(metric)] = {
                "status": "PASS" if passed else "FAIL",
                "point": float(estimate["point"]),
                "one_sided_95_ucb": upper,
                "margin": margin,
                "unit": "fraction_left_minus_right",
            }

        prose_integrity: dict[str, Any] | None = None
        raw_contract_ni_results: dict[str, dict[str, Any]] = {}
        raw_contract_ni_pass: bool | None = None
        raw_contract_work: dict[str, Any] | None = None
        pointer_only_attribution_status: str | None = None
        if kind == "structured_selection_vs_prose":
            if (
                contract.get("prose_control_integrity_required") is not True
                or contract.get(
                    "raw_contract_adherence_sensitivity_required_for_pointer_only_attribution"
                )
                is not True
            ):
                raise ValueError(
                    "structured-selection/prose policy lacks integrity and raw-contract guards"
                )
            prose_integrity = _prose_integrity_gate(paired)
            raw_contract_ni_pass = True
            for metric, raw_margin in sorted(higher_margins.items()):
                margin = _finite(raw_margin)
                estimate = _raw_contract_quality_bootstrap(
                    paired,
                    metric=str(metric),
                    n_boot=n_boot,
                    seed=seed_from(
                        str(seed), contrast_id, "raw-contract-ni", str(metric)
                    ),
                )
                if estimate.get("status") == "NOT_APPLICABLE":
                    raw_contract_ni_results[str(metric)] = {
                        "status": "NOT_APPLICABLE_PASS",
                        "reason": estimate.get("reason"),
                        "margin": margin,
                    }
                    continue
                if estimate.get("status") != "OK" or margin is None:
                    raw_contract_ni_pass = False
                    raw_contract_ni_results[str(metric)] = {
                        "status": "NOT_ESTABLISHED",
                        "reason": estimate.get("reason"),
                        "margin": margin,
                    }
                    continue
                lower = float(
                    np.quantile(estimate["samples"], 0.05, method="lower")
                )
                passed = lower >= margin
                raw_contract_ni_pass = raw_contract_ni_pass and passed
                raw_contract_ni_results[str(metric)] = {
                    "status": "PASS" if passed else "FAIL",
                    "point": float(estimate["point"]),
                    "one_sided_95_lcb": lower,
                    "margin": margin,
                    "unit": estimate["unit"],
                }
            for metric, raw_margin in sorted(lower_margins.items()):
                margin = _finite(raw_margin)
                estimate = _raw_contract_quality_bootstrap(
                    paired,
                    metric=str(metric),
                    n_boot=n_boot,
                    seed=seed_from(
                        str(seed), contrast_id, "raw-contract-ni", str(metric)
                    ),
                )
                if estimate.get("status") == "NOT_APPLICABLE":
                    raw_contract_ni_results[str(metric)] = {
                        "status": "NOT_APPLICABLE_PASS",
                        "reason": estimate.get("reason"),
                        "margin": margin,
                    }
                    continue
                if estimate.get("status") != "OK" or margin is None:
                    raw_contract_ni_pass = False
                    raw_contract_ni_results[str(metric)] = {
                        "status": "NOT_ESTABLISHED",
                        "reason": estimate.get("reason"),
                        "margin": margin,
                    }
                    continue
                upper = float(
                    np.quantile(estimate["samples"], 0.95, method="higher")
                )
                passed = upper <= margin
                raw_contract_ni_pass = raw_contract_ni_pass and passed
                raw_contract_ni_results[str(metric)] = {
                    "status": "PASS" if passed else "FAIL",
                    "point": float(estimate["point"]),
                    "one_sided_95_ucb": upper,
                    "margin": margin,
                    "unit": estimate["unit"],
                }
            if prose_integrity["raw_contract_adherence_all_offered"]:
                estimate = _effect_bootstrap(
                    paired,
                    effect="work_saving",
                    n_boot=n_boot,
                    seed=seed_from(
                        str(seed), contrast_id, "raw-contract-work-sensitivity"
                    ),
                )
                raw_contract_work = {
                    key: value
                    for key, value in estimate.items()
                    if key != "samples"
                }
            else:
                raw_contract_work = {
                    "status": "NOT_ESTIMABLE",
                    "reason": (
                        "DIRTY_RAW_CONTRACT_HAS_NO_IDENTIFIED_COUNTERFACTUAL_SERVICE_WORK"
                    ),
                    "prose_raw_contract_nonadherent_blocks": prose_integrity[
                        "prose_raw_contract_nonadherent_blocks"
                    ],
                    "structured_raw_contract_nonadherent_blocks": prose_integrity[
                        "structured_raw_contract_nonadherent_blocks"
                    ],
                }

        cost_guard_pass: bool | None = None
        cost_guard: dict[str, Any] | None = None
        if kind == "LLM_vs_CPU":
            work = _effect_bootstrap(
                paired,
                effect="work_saving",
                n_boot=n_boot,
                seed=seed_from(str(seed), contrast_id, "service-work-cost-guard"),
            )
            maximum = _finite(contract.get("service_work_increase_ucb_max"))
            if work.get("status") == "OK" and maximum is not None:
                increases = -np.asarray(work["samples"], dtype=float)
                upper = float(np.quantile(increases, 0.95, method="higher"))
                cost_guard_pass = upper <= maximum
                cost_guard = {
                    "status": "PASS" if cost_guard_pass else "FAIL",
                    "point": -float(work["point"]),
                    "one_sided_95_ucb": upper,
                    "maximum": maximum,
                    "unit": "fraction_geometric_service_work_increase_left_over_right",
                }
            else:
                cost_guard_pass = False
                cost_guard = {
                    "status": "NOT_ESTABLISHED",
                    "reason": work.get("reason"),
                    "maximum": maximum,
                }
        comparability_pass = (
            isinstance(primary.get("comparability"), Mapping)
            and primary["comparability"].get("status") == "OK"
        )
        component_established = (
            primary.get("adjusted_primary_gate_pass") is True
            and ni_pass
            and comparability_pass
            and (cost_guard_pass is True if kind == "LLM_vs_CPU" else True)
            and (
                prose_integrity is not None
                and prose_integrity.get("status") == "PASS"
                if kind == "structured_selection_vs_prose"
                else True
            )
        )
        if kind == "structured_selection_vs_prose":
            pointer_only_attribution_status = (
                "ESTABLISHED"
                if (
                    component_established
                    and raw_contract_ni_pass is True
                    and prose_integrity is not None
                    and prose_integrity.get(
                        "raw_contract_adherence_all_offered"
                    )
                    is True
                    and isinstance(raw_contract_work, Mapping)
                    and raw_contract_work.get("status") == "OK"
                )
                else "NOT_ESTABLISHED"
            )
        component_by_id[contrast_id] = {
            "status": "ESTABLISHED" if component_established else "NOT_ESTABLISHED",
            "contrast_id": contrast_id,
            "holm_family_id": family_id,
            "estimand_contract_sha256": sha256_hex(canonical_json(dict(contract))),
            "adjusted_primary_gate_pass": (
                primary.get("adjusted_primary_gate_pass") is True
            ),
            "primary_estimand": {
                key: value
                for key, value in primary.items()
                if key not in {"samples", "comparability"}
            },
            "all_strict_quality_ni_guards_pass": ni_pass,
            "strict_quality_ni_guards": ni_results,
            "first_boundary_input_comparability_pass": comparability_pass,
            "first_boundary_input_comparability": dict(
                primary.get("comparability") or {}
            ),
            **(
                {
                    "service_work_cost_guard_pass": cost_guard_pass,
                    "service_work_cost_guard": cost_guard,
                }
                if kind == "LLM_vs_CPU"
                else {
                    "prose_control_integrity_gate_pass": (
                        prose_integrity is not None
                        and prose_integrity.get("status") == "PASS"
                    ),
                    "prose_control_integrity_gate": prose_integrity,
                    "all_raw_contract_quality_ni_guards_pass":
                        raw_contract_ni_pass,
                    "raw_contract_quality_ni_guards": raw_contract_ni_results,
                    "raw_contract_work_sensitivity": raw_contract_work,
                    "pointer_only_attribution_status":
                        pointer_only_attribution_status,
                    "main_estimand_scope":
                        "STRUCTURED_ID_POLICY_VS_BOUNDED_SHORT_PROSE_POLICY",
                }
            ),
        }

    by_node: dict[str, dict[str, Any]] = {}
    for node in nodes:
        node_policy = by_node_policy.get(node)
        if not isinstance(node_policy, Mapping):
            raise ValueError(f"structured-increment policy for {node} is absent")
        primary_arm = str(node_policy.get("primary_arm_id") or "")
        required = tuple(map(str, node_policy.get("required_contrast_ids") or ()))
        if (
            not primary_arm
            or len(required) != 2
            or not set(required) <= set(member_ids)
            or {
                str(design_by_id[contrast_id].get("kind") or "")
                for contrast_id in required
            }
            != {"LLM_vs_CPU", "structured_selection_vs_prose"}
        ):
            raise ValueError(f"structured-increment policy for {node} needs both control types")
        if any(
            str(design_by_id[contrast_id].get("left_arm_id") or "") != primary_arm
            for contrast_id in required
        ):
            raise ValueError(
                f"structured-increment controls for {node} do not target its primary arm"
            )
        components = {
            str(design_by_id[contrast_id]["kind"]): component_by_id[contrast_id]
            for contrast_id in required
        }
        established = all(
            component.get("status") == "ESTABLISHED"
            for component in components.values()
        )
        pointer_only = (
            components["structured_selection_vs_prose"].get(
                "pointer_only_attribution_status"
            )
            == "ESTABLISHED"
        )
        by_node[node] = {
            "status": "ESTABLISHED" if established else "NOT_ESTABLISHED",
            "primary_arm_id": primary_arm,
            "required_contrast_ids": list(required),
            "component_gates": components,
            "compound_status": "ESTABLISHED" if established else "NOT_ESTABLISHED",
            "compound_claim": str(
                (policy.get("compound_claim") or {}).get("label") or ""
            ),
            "pointer_only_attribution_status": (
                "ESTABLISHED" if established and pointer_only
                else "NOT_ESTABLISHED"
            ),
            "attribution_scope": (
                "LLM_PLUS_STRUCTURED_POINTER_REPRESENTATION"
                if established and pointer_only
                else "LLM_PLUS_STRUCTURED_BOUNDED_POLICY_ONLY"
            ),
        }

    family_results = {
        contrast_id: {
            key: value
            for key, value in internal[contrast_id].items()
            if key not in {"samples", "comparability"}
        }
        for contrast_id in member_ids
    }
    return {
        "schema_version": schema,
        "policy_sha256": sha256_hex(canonical_json(dict(policy))),
        "multiplicity_family": dict(family),
        "family_status": (
            "ESTIMABLE"
            if all_primary_estimable
            else "NOT_ESTIMABLE_ONE_OR_MORE_MEMBERS"
        ),
        "holm_order": ordered,
        "family_results": family_results,
        "by_node": by_node,
        "bootstrap_contract": {
            "unit": "source_topic_cluster",
            "replicate_reduction": "task_mean_before_cluster_bootstrap",
            "raw_p_method": "percentile_paired_cluster_bootstrap_lcb_inversion_one_sided",
            "adjusted_bound_method": "holm_rank_alpha_percentile_lcb",
            "n_boot": n_boot,
            "seed": seed,
        },
    }


def build_matched_contrasts(
    records: Sequence[Mapping[str, Any]],
    *,
    scope_receipt: Mapping[str, Any],
    cluster_by_task: Mapping[str, str],
    matched_contrasts: Mapping[str, Any],
    arm_variants: Mapping[str, Mapping[str, Any]],
    quality_views: Sequence[str] = _QUALITY_VIEWS,
    quality_metrics: Mapping[str, str] | Sequence[str] = QUALITY_GUARD_DIRECTIONS,
    structured_increment_policy: Mapping[str, Any] | None = None,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, Any]:
    """Build every preregistered matched contrast and content-address the report.

    ``arm_variants`` is indexed by *arm ID*, not variant ID.  Each value must include a
    ``variant_id`` plus the executable semantic fields named by ``matched_contrasts``.  This
    makes it possible to verify both the declared one-factor comparison and what each score
    row actually executed.
    """
    if not records:
        raise ValueError("matched analysis needs frozen score records")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    (
        scope_run_id,
        scope_phase_id,
        evaluation_scope_sha256,
        _expected_scope_rows,
        verified_scope,
    ) = _verify_score_scope(records, scope_receipt)
    used_tasks = sorted({str(record.get("task_id") or "") for record in records})
    if "" in used_tasks or any(
        not str(cluster_by_task.get(task_id) or "") for task_id in used_tasks
    ):
        raise ValueError("cluster_by_task does not cover every frozen score task")
    frozen_cluster_by_task = {
        task_id: str(cluster_by_task[task_id]) for task_id in used_tasks
    }
    views = tuple(map(str, quality_views))
    if len(views) != len(set(views)) or set(views) != set(_QUALITY_VIEWS):
        raise ValueError("quality_views must contain every preregistered view exactly once")
    if isinstance(quality_metrics, Mapping):
        metrics = {str(key): str(value) for key, value in quality_metrics.items()}
    else:
        names = tuple(map(str, quality_metrics))
        if len(names) != len(set(names)):
            raise ValueError("quality_metrics contains duplicates")
        metrics = {
            name: QUALITY_GUARD_DIRECTIONS.get(name, "")
            for name in names
        }
    if set(metrics) != set(QUALITY_GUARD_DIRECTIONS) or any(
        direction not in {"higher_is_better", "lower_is_better", "descriptive_only"}
        for direction in metrics.values()
    ):
        raise ValueError("quality_metrics must contain every preregistered metric and direction")
    pairs, executable_fields = _validate_semantic_registry(
        matched_contrasts,
        arm_variants,
    )

    results: list[dict[str, Any]] = []
    paired_by_contrast: dict[str, list[dict[str, Any]]] = {}
    for pair in pairs:
        contrast_id = str(pair["contrast_id"])
        left_arm = str(pair["left_arm_id"])
        right_arm = str(pair["right_arm_id"])
        left_variant = dict(arm_variants[left_arm])
        right_variant = dict(arm_variants[right_arm])
        paired, issues = _pair_blocks(
            records,
            left_arm=left_arm,
            right_arm=right_arm,
            expected_left_variant=str(left_variant["variant_id"]),
            expected_right_variant=str(right_variant["variant_id"]),
            cluster_by_task=cluster_by_task,
        )
        paired_by_contrast[contrast_id] = paired
        first_boundary_comparability = _first_boundary_input_comparability(paired)
        quality_results: dict[str, dict[str, Any]] = {}
        operational_results: dict[str, dict[str, Any]] = {}
        trajectory_results: dict[str, dict[str, Any]] = {}
        if issues:
            for view in views:
                quality_results[view] = {
                    metric: _structurally_not_estimable(
                        direction=direction,
                        endpoint=f"quality.{view}.{metric}",
                        issues=issues,
                    )
                    for metric, direction in metrics.items()
                }
            operational_results = {
                endpoint: _structurally_not_estimable(
                    direction=direction,
                    endpoint=endpoint,
                    issues=issues,
                )
                for endpoint, direction in _OPERATIONAL_ENDPOINTS.items()
            }
            trajectory_results = {
                endpoint: _structurally_not_estimable(
                    direction=direction,
                    endpoint=endpoint,
                    issues=issues,
                )
                for endpoint, direction in TRAJECTORY_ENDPOINT_DIRECTIONS.items()
            }
        else:
            for view in views:
                quality_results[view] = {
                    metric: _estimate_endpoint(
                        paired,
                        endpoint=f"quality.{view}.{metric}",
                        direction=direction,
                        quality_view=view,
                        quality_metric=metric,
                        n_boot=n_boot,
                        seed=seed_from(str(seed), contrast_id, view, metric),
                    )
                    for metric, direction in metrics.items()
                }
            operational_results = {
                endpoint: _estimate_endpoint(
                    paired,
                    endpoint=endpoint,
                    direction=direction,
                    n_boot=n_boot,
                    seed=seed_from(str(seed), contrast_id, endpoint),
                )
                for endpoint, direction in _OPERATIONAL_ENDPOINTS.items()
            }
            trajectory_results = {
                endpoint: _estimate_endpoint(
                    paired,
                    endpoint=endpoint,
                    direction=direction,
                    n_boot=n_boot,
                    seed=seed_from(str(seed), contrast_id, "trajectory", endpoint),
                )
                for endpoint, direction in TRAJECTORY_ENDPOINT_DIRECTIONS.items()
            }
        statuses = Counter(
            result["status"]
            for by_metric in quality_results.values()
            for result in by_metric.values()
        )
        statuses.update(result["status"] for result in operational_results.values())
        statuses.update(result["status"] for result in trajectory_results.values())
        results.append(
            {
                "contrast_id": contrast_id,
                "kind": str(pair.get("kind") or ""),
                "left_arm_id": left_arm,
                "right_arm_id": right_arm,
                "target_factor": str(pair["target_factor"]),
                "factor_fields": list(pair["_factor_fields"]),
                "left_level": pair.get("left_level"),
                "right_level": pair.get("right_level"),
                **(
                    {
                        "factor_scope": pair.get("factor_scope"),
                        "affected_nodes": list(map(str, pair.get("affected_nodes") or ())),
                        "first_treatment_boundary": pair.get("first_treatment_boundary"),
                    }
                    if pair.get("factor_scope") is not None
                    else {}
                ),
                "semantic_validation": {
                    "status": "OK",
                    "executable_variant_fields": list(executable_fields),
                    "left_variant": {
                        field: left_variant.get(field)
                        for field in ("variant_id", *executable_fields)
                    },
                    "right_variant": {
                        field: right_variant.get(field)
                        for field in ("variant_id", *executable_fields)
                    },
                },
                "pairing_status": "NOT_ESTIMABLE" if issues else "OK",
                "first_boundary_input_comparability": first_boundary_comparability,
                "blocks_offered": len(records),
                "blocks_paired": len(paired),
                "structural_issues": issues,
                "endpoint_status_counts": dict(sorted(statuses.items())),
                "quality": quality_results,
                "operational": operational_results,
                "trajectory": trajectory_results,
            }
        )

    structured_increment_gates = (
        _build_structured_increment_gates(
            paired_by_contrast=paired_by_contrast,
            contrast_results=results,
            matched_contrasts=matched_contrasts,
            policy=structured_increment_policy,
            n_boot=n_boot,
            seed=seed,
        )
        if structured_increment_policy is not None
        else None
    )
    body: dict[str, Any] = {
        "schema_version": "matched_contrast_analysis_v2",
        "run_id": scope_run_id,
        "phase_id": scope_phase_id,
        "evaluation_scope_sha256": evaluation_scope_sha256,
        "execution_binding_sha256": str(
            verified_scope["execution_binding_sha256"]
        ),
        "protocol_document_sha256": str(
            verified_scope["protocol_document_sha256"]
        ),
        "contrast_convention": {
            "ordinary_endpoints": "left_arm_minus_right_arm",
            "service_work": (
                "one_minus_geometric_mean_of_left_over_right; "
                "positive_means_left_saves_work"
            ),
            "trajectory_checkpoint_divergence": "mediated_end_to_end_outcome_not_pairing_error",
        },
        "bootstrap": {
            "unit": "source_topic_cluster",
            "replicate_reduction": "task_mean_before_cluster_bootstrap",
            "n_boot": n_boot,
            "seed": seed,
        },
        "input_provenance": {
            "score_records_sha256": _input_sha(list(records)),
            "source_cluster_map_sha256": _input_sha(frozen_cluster_by_task),
            "matched_contrasts_sha256": _input_sha(dict(matched_contrasts)),
            "arm_variants_sha256": _input_sha(
                {
                    key: dict(value)
                    for key, value in sorted(arm_variants.items())
                }
            ),
            "records": len(records),
            "tasks": len({str(record.get("task_id") or "") for record in records}),
        },
        "contrasts": results,
    }
    if structured_increment_gates is not None:
        body["structured_increment_gates"] = structured_increment_gates
    return _seal(body)
