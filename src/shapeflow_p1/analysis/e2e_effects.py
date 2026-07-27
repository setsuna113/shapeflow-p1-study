"""End-to-end H x C effects and a training-only eligibility bridge.

This module consumes the evaluator's *verified all-offered* score records.  It does not scan a
directory or choose whichever cells happened to succeed.  Every block must carry exactly one
P0, H, C and H+C assignment for the same task/replicate.  Unequal replicates are first averaged
within task; source/topic clusters are then the bootstrap unit.

Quality, latency, and ordinary token endpoints retain treatment-minus-P0 signs.  Service work is
the preregistered paired log-ratio saving (positive means P1 saves work) with a one-sided lower
confidence bound.  Cached prompt tokens are descriptive because a larger cache count can mean
either more reusable work or more input work; it is never labeled "lower is better".

Each endpoint has an independent OK/NOT_APPLICABLE/NOT_ESTIMABLE status.  One missing endpoint
therefore cannot erase other measured outcomes, while a one-arm missing value is never silently
complete-case filtered.  All four strict/assisted/worst/best quality views are crossed with every
preregistered safety, citation, coverage, contradiction, negative, and gap guard.

The eligibility family is a formative task-level heterogeneity analysis fitted only on
pre-declared training splits and genuinely pre-treatment numeric features.  Every predeclared
H/C/H+C target requires all strict quality guards and complete service-work saving against all of
its named comparators.  Cluster-held-out CV prevents correlated tasks in one source/topic cluster
from leaking across train and validation.  Heldout rows may be evaluated by the frozen rule but
never enter fitting, split selection, thresholds, or stability estimation; no result is an online
invocation policy or a confirmatory/deployment envelope.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from ..canonical import canonical_json
from ..hashing import sha256_hex
from .bootstrap import cluster_bootstrap_ci, paired_log_ratio_saving, seed_from
from .factorial import ArmOutcomes, estimate_contrasts
from .heterogeneity import (
    Sample,
    fit_eligibility_tree,
    rule_stability,
)

__all__ = [
    "ALLOWED_PRETREATMENT_FEATURES",
    "CORE_FACTORIAL_ARMS",
    "ELIGIBILITY_TARGET_FAMILY",
    "QUALITY_GUARD_DIRECTIONS",
    "build_factorial_effects",
    "build_eligibility_effects",
    "build_e2e_effects",
    "task_feature_registry_sha256",
    "validate_factorial_variants",
]


ALLOWED_PRETREATMENT_FEATURES = frozenset(
    {
        # Frozen acquired source pool.  These exist before the first randomized treatment cell.
        "candidate_evidence_tokens",
        "source_count",
        "span_count",
        "span_length_q25",
        "span_length_q50",
        "span_length_q75",
        "span_length_q90",
        "table_list_fraction",
        "redundancy",
        "raw_content_available_fraction",
        # Frozen evaluator task view.  These may be used for formative task-level
        # heterogeneity, but are not claimed to be an online invocation policy.
        "question_token_count",
        "authored_facet_count",
        "fixed_query_count",
        "declared_stratum_evidence_volume_low",
        "declared_stratum_evidence_volume_medium",
        "declared_stratum_evidence_volume_high",
        "declared_stratum_facets_1_2",
        "declared_stratum_facets_3_4",
        "declared_stratum_facets_5_plus",
        "declared_stratum_single_source_fact",
        "declared_stratum_multi_source_synthesis",
        "declared_stratum_source_conflict",
        "declared_stratum_negative_evidence",
        "declared_stratum_citation_dense",
        "declared_stratum_table_list_heavy",
        "declared_stratum_high_redundancy",
        "declared_stratum_raw_content_missing",
    }
)

_FORBIDDEN_FEATURE_FRAGMENTS = (
    "fallback",
    "retry",
    "reuse",
    "p0_output",
    "treatment_output",
    "selected_",
    "published_",
    "visible_message",
    "tool_output",
    "tool_call",
    "query_attempt",
    "close_reason",
    "checkpoint",
    "trajectory",
)

CORE_FACTORIAL_ARMS = {
    "p0": "P0",
    # The core 2x2 in configs/week1.yaml.  H02/C01 are the representative ID-selection
    # mechanisms; the other ten arms remain design variants and controls, not substitutes for
    # either factorial corner.
    "h": "H_MARKDOWN_ID",
    "c": "C_ID",
    "hc": "H_PLUS_C",
}

# This family is deliberately exact and code-checked.  A hash-valid spec cannot silently add a
# post-hoc subgroup target or relabel the benefit of H as an H+C benefit.  The final target is an
# intersection-union choice: H+C must clear the same strict guards and work threshold against
# every simpler alternative.
ELIGIBILITY_TARGET_FAMILY = {
    "H_STANDALONE": ("h", ("p0",)),
    "C_STANDALONE": ("c", ("p0",)),
    "C_INCREMENT_GIVEN_H": ("hc", ("h",)),
    "H_INCREMENT_GIVEN_C": ("hc", ("c",)),
    "HC_JOINT_CHOICE": ("hc", ("p0", "h", "c")),
}

_ELIGIBILITY_SCOPE = "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY"

_QUALITY_VIEWS = ("strict", "fallback_assisted", "worst_case", "best_case")

# These are the report-quality estimands that can change the scientific answer.  A metric that
# is inapplicable to a truth packet is represented as None in all four arms; a one-arm None is
# measurement failure, never an invitation to analyze the convenient subset.
QUALITY_GUARD_DIRECTIONS = {
    "weighted_required_atom_recall": "higher_is_better",
    "critical_atom_safety": "higher_is_better",
    "grounded_claim_precision": "higher_is_better",
    "citation_correctness": "higher_is_better",
    "citation_association": "higher_is_better",
    "citation_completeness": "higher_is_better",
    "required_facet_coverage": "higher_is_better",
    "contradiction_handling": "higher_is_better",
    "critical_harm": "lower_is_better",
    "qualified_report": "higher_is_better",
    "grounded_negative_recall": "higher_is_better",
    "unresolved_gap_reporting_recall": "higher_is_better",
}

#: The work endpoint the product claim rests on under the native-concurrent primary layer.
#:
#: It used to be ``service_work_seconds`` -- the sum of per-request service intervals -- and that
#: choice reached back and changed the system under test: a sum of intervals is only a work
#: figure when the intervals do not overlap, so the engine was configured to admit one upstream
#: request at a time. The pinned graph summarises a result set with ``asyncio.gather``, so the
#: serialization queued concurrent summaries behind each other and produced 212 timeouts that
#: exist in no native run, on exactly the largest pages. The metric was distorting the
#: measurement it was supposed to take.
PRIMARY_WORK_ENDPOINT = "interval_union_seconds"

#: Endpoints expressed as a paired log-ratio saving rather than a difference. All are strictly
#: positive work quantities where "30% less" is the meaningful statement and an absolute
#: difference is not comparable across tasks of different size.
SAVING_SCALE_ENDPOINTS = frozenset({
    "interval_union_seconds",
    "service_work_seconds",
    "energy_joules",
})

# End-to-end trajectory changes are outcomes, not pairing violations.  These are deliberately
# bounded, interpretable counters/rates recomputed by the evaluator from each frozen event
# stream.  No endpoint requires P1 to reproduce P0's queries, checkpoint IDs, or close path.
TRAJECTORY_ENDPOINT_DIRECTIONS = {
    "event_count": "descriptive_only",
    "query_count": "descriptive_only",
    "unique_query_count": "descriptive_only",
    "query_redundancy": "lower_is_better",
    "retrieved_result_count": "descriptive_only",
    "unique_source_occurrence_count": "descriptive_only",
    "retrieval_empty_rate": "lower_is_better",
    "research_rounds": "descriptive_only",
    "tool_calls_observed": "descriptive_only",
    "model_tool_decision_count": "descriptive_only",
    "conduct_research_calls": "descriptive_only",
    "think_calls": "descriptive_only",
    "research_complete_calls": "descriptive_only",
    "h_checkpoint_count": "descriptive_only",
    "c_checkpoint_count": "descriptive_only",
    "fallback_count": "lower_is_better",
    "failure_count": "lower_is_better",
}


def _unsigned_sha(body: Mapping[str, Any], hash_field: str = "content_sha256") -> str:
    return sha256_hex(
        canonical_json({key: value for key, value in body.items() if key != hash_field})
    )


def task_feature_registry_sha256(
    task_features: Mapping[str, Mapping[str, Any]],
) -> str:
    """Address the exact task-feature registry committed to by an eligibility spec.

    Individual records remain independently content addressed.  The registry digest binds
    their task coordinates and hashes, preventing an analyst from swapping in a different
    pre-treatment feature table after outcomes are visible.
    """
    records = [
        {
            "task_id": str(task_id),
            "feature_content_sha256": str(record.get("content_sha256") or ""),
        }
        for task_id, record in sorted(task_features.items())
    ]
    return sha256_hex(
        canonical_json(
            {
                "schema_version": "task_feature_registry_v1",
                "records": records,
            }
        )
    )


def _verify_score_scope(
    records: Sequence[Mapping[str, Any]],
    scope_receipt: Mapping[str, Any] | None,
) -> tuple[str, str, str, dict[str, dict], dict]:
    if not records:
        raise ValueError("no all-offered score records")
    receipt = scope_receipt or getattr(records, "scope_receipt", None)
    if not isinstance(receipt, Mapping):
        raise ValueError(
            "verified all-offered scope receipt is required; an arbitrary score list is not ITT"
        )
    if receipt.get("schema_version") != "evaluated_itt_scope_v1":
        raise ValueError("unsupported evaluation scope receipt schema")
    recorded_scope_sha = str(receipt.get("evaluation_scope_sha256") or "")
    if recorded_scope_sha != _unsigned_sha(receipt, "evaluation_scope_sha256"):
        raise ValueError("evaluation scope receipt content hash does not verify")
    expected_rows = receipt.get("scores")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError("evaluation scope receipt has no offered score index")
    expected: dict[str, dict] = {}
    execution_binding_sha = str(receipt.get("execution_binding_sha256") or "")
    protocol_document_sha = str(receipt.get("protocol_document_sha256") or "")
    if len(execution_binding_sha) != 64 or len(protocol_document_sha) != 64:
        raise ValueError("evaluation scope receipt lacks execution/protocol identity")
    for row in expected_rows:
        block_id = str((row or {}).get("block_id") or "")
        score_sha = str((row or {}).get("score_content_sha256") or "")
        if not block_id or block_id in expected or len(score_sha) != 64:
            raise ValueError("evaluation scope has duplicate or incomplete score coordinates")
        expected_row = dict(row)
        if (
            str(expected_row.get("execution_binding_sha256") or "")
            != execution_binding_sha
            or str(expected_row.get("protocol_document_sha256") or "")
            != protocol_document_sha
        ):
            raise ValueError(
                f"evaluation scope score {block_id} has another execution/protocol identity"
            )
        epochs = list(map(str, expected_row.get("engine_epochs") or ()))
        by_arm = expected_row.get("engine_epoch_by_arm")
        paired_valid = expected_row.get("valid_for_paired_estimate")
        if (
            len(str(expected_row.get("block_freeze_sha256") or "")) != 64
            or len(str(expected_row.get("block_digest") or "")) != 64
            or not isinstance(paired_valid, bool)
            or not epochs
            or not isinstance(by_arm, Mapping)
            or not by_arm
            or set(map(str, by_arm.values())) != set(epochs)
            or (paired_valid and str(expected_row.get("invalid_reason") or ""))
            or (not paired_valid and not str(expected_row.get("invalid_reason") or ""))
        ):
            raise ValueError(f"evaluation scope has incomplete block provenance for {block_id}")
        expected_row["engine_epochs"] = epochs
        expected_row["engine_epoch_by_arm"] = dict(
            sorted((str(key), str(value)) for key, value in by_arm.items())
        )
        expected[block_id] = expected_row
    if int(receipt.get("all_offered_blocks") or -1) != len(expected):
        raise ValueError("evaluation scope all_offered_blocks does not match its score index")

    run_ids: set[str] = set()
    phase_ids: set[str] = set()
    observed: set[str] = set()
    for record in records:
        block_id = str(record.get("block_id") or "")
        if not block_id or block_id in observed:
            raise ValueError(f"duplicate or unnamed score block {block_id!r}")
        observed.add(block_id)
        score_sha = str(record.get("content_sha256") or "")
        expected_row = expected.get(block_id) or {}
        if score_sha != _unsigned_sha(record) or score_sha != str(
            expected_row.get("score_content_sha256") or ""
        ):
            raise ValueError(f"score block {block_id} does not match the verified scope")
        if str(record.get("task_id") or "") != str(expected_row.get("task_id") or "") or str(
            record.get("replicate_id") or ""
        ) != str(expected_row.get("replicate_id") or ""):
            raise ValueError(f"score block {block_id} coordinates differ from verified scope")
        if (
            str(record.get("execution_binding_sha256") or "")
            != execution_binding_sha
            or str(record.get("protocol_document_sha256") or "")
            != protocol_document_sha
        ):
            raise ValueError(
                f"score block {block_id} has another execution/protocol identity"
            )
        run_ids.add(str(record.get("run_id") or ""))
        phase_ids.add(str(record.get("phase_id") or ""))
    if observed != set(expected):
        raise ValueError(
            "score records are not the complete all-offered set: "
            f"missing={sorted(set(expected) - observed)}, "
            f"extra={sorted(observed - set(expected))}"
        )
    if len(run_ids) != 1 or "" in run_ids or len(phase_ids) != 1 or "" in phase_ids:
        raise ValueError("score records must belong to one exact non-empty run and phase")
    run_id = next(iter(run_ids))
    phase_id = next(iter(phase_ids))
    if str(receipt.get("run_id") or "") != run_id or str(receipt.get("phase_id") or "") != phase_id:
        raise ValueError("evaluation scope receipt belongs to another run or phase")
    if (
        len(str(receipt.get("schedule_sha256") or "")) != 64
        or len(str(receipt.get("freeze_root_sha256") or "")) != 64
    ):
        raise ValueError("evaluation scope receipt lacks schedule/freeze-root provenance")
    return run_id, phase_id, recorded_scope_sha, expected, dict(receipt)


def _verify_root_provenance(
    receipt: Mapping[str, Any],
    expected_rows: Mapping[str, Mapping[str, Any]],
    *,
    freeze_root: Mapping[str, Any] | None,
    schedule_manifest: Mapping[str, Any] | None,
) -> dict:
    """Verify the full schedule/root chain when supplied, while always verifying its digests.

    ``load_scoped_scores`` already validates these files before attaching the receipt.  Direct
    programmatic users may additionally pass the frozen objects here, allowing this builder to
    recompute every link rather than merely trust the content-addressed receipt.
    """
    schedule_sha = str(receipt["schedule_sha256"])
    root_sha = str(receipt["freeze_root_sha256"])
    status = {
        "execution_binding_sha256": str(receipt["execution_binding_sha256"]),
        "protocol_document_sha256": str(receipt["protocol_document_sha256"]),
        "schedule_sha256": schedule_sha,
        "freeze_root_sha256": root_sha,
        "schedule_object_verified": schedule_manifest is not None,
        "freeze_root_object_verified": freeze_root is not None,
    }
    if schedule_manifest is not None:
        schedule = dict(schedule_manifest)
        recorded = str(schedule.get("schedule_sha256") or "")
        if recorded != schedule_sha or recorded != _unsigned_sha(schedule, "schedule_sha256"):
            raise ValueError("schedule manifest does not verify against evaluation scope")
    if freeze_root is not None:
        root = dict(freeze_root)
        if root.get("schema_version") != "frozen_campaign_root_v1":
            raise ValueError("unsupported frozen campaign root schema")
        recorded = str(root.get("freeze_root_sha256") or "")
        if recorded != root_sha or recorded != _unsigned_sha(root, "freeze_root_sha256"):
            raise ValueError("frozen campaign root does not verify against evaluation scope")
        if (
            str(root.get("run_id") or "") != str(receipt["run_id"])
            or str(root.get("phase_id") or "") != str(receipt["phase_id"])
            or str(root.get("schedule_sha256") or "") != schedule_sha
            or root.get("terminal_frozen") is not True
        ):
            raise ValueError("frozen campaign root coordinates differ from evaluation scope")
        root_blocks = {str(item.get("block_id") or ""): item for item in root.get("blocks") or ()}
        if set(root_blocks) != set(expected_rows):
            raise ValueError("frozen campaign root block set differs from evaluation scope")
        for block_id, expected in expected_rows.items():
            rooted = root_blocks[block_id]
            if (
                rooted.get("freeze_sha256") != expected["block_freeze_sha256"]
                or rooted.get("block_digest") != expected["block_digest"]
                or rooted.get("valid_for_paired_estimate") != expected["valid_for_paired_estimate"]
                or str(rooted.get("invalid_reason") or "")
                != str(expected.get("invalid_reason") or "")
                or list(map(str, rooted.get("engine_epochs") or ()))
                != list(expected["engine_epochs"])
            ):
                raise ValueError(f"frozen root provenance differs for block {block_id}")
        embedded = root.get("schedule")
        if schedule_manifest is not None and embedded != dict(schedule_manifest):
            raise ValueError("frozen root embeds a different schedule manifest")
        if isinstance(embedded, Mapping):
            recorded_embedded = str(embedded.get("schedule_sha256") or "")
            if recorded_embedded != schedule_sha or recorded_embedded != _unsigned_sha(
                embedded, "schedule_sha256"
            ):
                raise ValueError("frozen root embeds a non-verifying schedule")
            status["embedded_schedule_verified"] = True
    return status


def _verify_task_features(
    task_features: Mapping[str, Mapping[str, Any]],
    task_ids: set[str],
) -> dict[str, dict]:
    if not task_ids <= set(task_features):
        raise ValueError(
            "frozen task feature registry does not cover analyzed tasks: "
            f"missing={sorted(task_ids - set(task_features))}"
        )
    verified: dict[str, dict] = {}
    for task_id, raw in task_features.items():
        body = dict(raw)
        if str(body.get("task_id") or "") != task_id:
            raise ValueError(f"task feature record {task_id} names another task")
        if str(body.get("content_sha256") or "") != _unsigned_sha(body):
            raise ValueError(f"task feature record {task_id} content hash does not verify")
        if not body.get("cluster_id") or not body.get("split"):
            raise ValueError(f"task feature record {task_id} lacks cluster_id/split")
        if not isinstance(body.get("features"), Mapping):
            raise ValueError(f"task feature record {task_id} lacks numeric features")
        verified[task_id] = body
    return verified


def validate_factorial_variants(
    blocks: Sequence[Mapping[str, Any]],
    expected_variant_ids: Mapping[str, str],
) -> dict[str, str]:
    """Prove that each semantic 2x2 corner executed its preregistered variant.

    Arm labels alone are insufficient: an ``H_MARKDOWN_ID`` row whose ledger actually ran
    ``H03+P0`` would otherwise contaminate the core factorial without changing the score key.
    """
    expected = {str(key): str(value) for key, value in expected_variant_ids.items()}
    if (
        set(expected) != {"p0", "h", "c", "hc"}
        or any(not value for value in expected.values())
        or len(set(expected.values())) != 4
    ):
        raise ValueError(
            "expected_variant_ids must name four distinct non-empty p0/h/c/hc variants"
        )
    for block in blocks:
        for semantic, variant_id in expected.items():
            observed = str(block["arms"][semantic].get("variant_id") or "")
            if observed != variant_id:
                raise ValueError(
                    f"block {block['block_id']} {semantic} expected variant "
                    f"{variant_id!r}, observed {observed!r}"
                )
    return expected


def _arm_id(key: str) -> str:
    return key.rsplit(":", 1)[0]


def _one_arm(per_arm: Mapping[str, Any], arm_id: str, replicate_id: str, block_id: str) -> dict:
    rows = [
        row
        for key, row in per_arm.items()
        if _arm_id(str(key)) == arm_id and str(key).rsplit(":", 1)[-1] == replicate_id
    ]
    if len(rows) != 1 or not isinstance(rows[0], Mapping):
        raise ValueError(
            f"block {block_id} needs exactly one {arm_id}:{replicate_id}, found {len(rows)}"
        )
    row = dict(rows[0])
    if row.get("arm_id") not in (None, arm_id):
        raise ValueError(f"block {block_id} {arm_id} row names another arm")
    if row.get("replicate_id") not in (None, replicate_id):
        raise ValueError(f"block {block_id} {arm_id} row names another replicate")
    return row


def _finite(value: Any, *, label: str, nonnegative: bool = False) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is unavailable or non-numeric") from exc
    if not math.isfinite(number) or (nonnegative and number < 0):
        raise ValueError(f"{label} is not a finite{' non-negative' if nonnegative else ''} value")
    return number


def _maybe_finite(
    value: Any,
    *,
    nonnegative: bool = False,
    strictly_positive: bool = False,
) -> float | None:
    if value is None:
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


def _quality_optional(
    row: Mapping[str, Any],
    *,
    view: str,
    metric: str,
) -> float | None:
    value, _ = _quality_measurement(row, view=view, metric=metric)
    return value


def _quality_measurement(
    row: Mapping[str, Any],
    *,
    view: str,
    metric: str,
) -> tuple[float | None, str]:
    """Return value plus VALUE/INAPPLICABLE/MISSING measurement state."""
    views = row.get("quality_views")
    selected = views.get(view) if isinstance(views, Mapping) else None
    if not isinstance(selected, Mapping):
        return None, "MISSING"
    if metric not in selected:
        return None, "MISSING"
    if selected[metric] is None:
        return None, "INAPPLICABLE"
    value = _maybe_finite(selected[metric])
    return (value, "VALUE") if value is not None else (None, "MISSING")


def _quality(row: Mapping[str, Any], *, view: str, metric: str) -> float:
    value = _quality_optional(row, view=view, metric=metric)
    if value is None:
        raise ValueError(f"quality.{view}.{metric} is unavailable or non-numeric")
    return value


def _work_summary_optional(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """A work summary whose *telemetry* is complete.

    Telemetry completeness and interval non-overlap were one condition, and they are not the
    same question. Completeness asks whether every request was observed; non-overlap asks
    whether the requests happened to be serialized. Merging them meant that under any regime
    where the agent issues concurrent requests -- which is what the pinned graph does, via
    ``asyncio.gather`` over a result set -- *every* endpoint went missing, including token
    counts, which do not depend on scheduling at all.

    Overlap is now checked only by the endpoints it actually invalidates; see
    :func:`_serialized_work_summary_optional`.
    """
    value = row.get("work_summary")
    if not isinstance(value, Mapping):
        return None
    if value.get("telemetry_complete") is not True:
        return None
    return value


def _serialized_work_summary_optional(row: Mapping[str, Any]) -> Mapping[str, Any] | None:
    """A work summary whose summed service time is also meaningful.

    ``service_seconds`` adds up per-request intervals, so it counts one wall-clock second once
    per concurrent request. That is a number only in a regime that admits one request at a time,
    and it is the reason the engine was previously made to admit one request at a time. The
    metric now carries its own precondition instead of imposing it on the system under test.
    """
    value = _work_summary_optional(row)
    if value is None or value.get("overlap_valid") is not True:
        return None
    return value


def _endpoint_optional(
    row: Mapping[str, Any],
    endpoint: str,
    *,
    quality_view: str,
    quality_metric: str,
) -> float | None:
    if endpoint == "quality":
        return _quality_optional(row, view=quality_view, metric=quality_metric)
    if endpoint in TRAJECTORY_ENDPOINT_DIRECTIONS:
        trajectory = row.get("trajectory_metrics")
        if not isinstance(trajectory, Mapping) or trajectory.get("status") != "OK":
            return None
        value = _maybe_finite(trajectory.get(endpoint), nonnegative=True)
        if (
            endpoint in {"query_redundancy", "retrieval_empty_rate"}
            and value is not None
            and value > 1.0
        ):
            return None
        return value
    summary = _work_summary_optional(row)
    if summary is None:
        return None
    if endpoint == "service_work_seconds":
        # Summed intervals mean nothing when they overlap, so this endpoint -- and only this
        # endpoint -- requires the serialized regime. A log-ratio has no defined value at zero.
        serialized = _serialized_work_summary_optional(row)
        if serialized is None:
            return None
        return _maybe_finite(serialized.get("service_seconds"), strictly_positive=True)
    if endpoint == "interval_union_seconds":
        return _maybe_finite(summary.get("interval_union_seconds"), strictly_positive=True)
    if endpoint == "energy_joules":
        return _maybe_finite(summary.get("energy_joules"), strictly_positive=True)
    if endpoint == "e2e_latency_seconds":
        value = row.get("e2e_latency_seconds")
        if value is None:
            value = summary.get("e2e_latency_seconds")
        return _maybe_finite(value, nonnegative=True)
    token_name = endpoint.removesuffix("_tokens")
    tokens = summary.get("tokens")
    if not isinstance(tokens, Mapping):
        return None
    return _maybe_finite(tokens.get(f"{token_name}_tokens"), nonnegative=True)


def _endpoint(
    row: Mapping[str, Any], endpoint: str, *, quality_view: str, quality_metric: str
) -> float:
    value = _endpoint_optional(
        row, endpoint, quality_view=quality_view, quality_metric=quality_metric
    )
    if value is None:
        raise ValueError(f"{endpoint} is unavailable or invalid")
    return value


def _ci(ci) -> dict:
    return {
        "point": float(ci.point),
        "lower": float(ci.lower),
        "upper": float(ci.upper),
        "n_clusters": int(ci.n_clusters),
        "n_boot": int(ci.n_boot),
    }


def _saving_ci(ci) -> dict:
    return {
        "point": float(ci.point),
        "lower": float(ci.lower),
        "one_sided_95_lcb": float(ci.lower),
        "upper": None,
        "n_clusters": int(ci.n_clusters),
        "n_boot": int(ci.n_boot),
    }


def _not_estimable(
    *,
    direction: str,
    scale: str,
    reason: str,
    missing_by_arm: Mapping[str, int] | None = None,
    partial_missing_blocks: Sequence[str] = (),
    paired_invalid_blocks: Sequence[Mapping[str, Any]] = (),
) -> dict:
    return {
        "status": "NOT_ESTIMABLE",
        "direction": direction,
        "contrast_scale": scale,
        "reason": reason,
        "missing_counts": {
            "total": sum((missing_by_arm or {}).values()),
            "by_arm": dict(missing_by_arm or {}),
            "partial_missing_blocks": list(partial_missing_blocks),
        },
        "paired_invalid_blocks": [dict(item) for item in paired_invalid_blocks],
    }


def _task_mean_outcomes(
    complete: Sequence[Mapping[str, Any]],
) -> list[ArmOutcomes]:
    by_task: dict[str, list[Mapping[str, Any]]] = {}
    for row in complete:
        by_task.setdefault(str(row["task_id"]), []).append(row)
    outcomes: list[ArmOutcomes] = []
    for task_id, rows in sorted(by_task.items()):
        clusters = {str(row["cluster_id"]) for row in rows}
        if len(clusters) != 1:
            raise ValueError(f"task {task_id} appears in multiple source/topic clusters")
        outcomes.append(
            ArmOutcomes(
                task_id=task_id,
                cluster_id=next(iter(clusters)),
                p0=sum(float(row["values"]["p0"]) for row in rows) / len(rows),
                h=sum(float(row["values"]["h"]) for row in rows) / len(rows),
                c=sum(float(row["values"]["c"]) for row in rows) / len(rows),
                hc=sum(float(row["values"]["hc"]) for row in rows) / len(rows),
            )
        )
    return outcomes


def _descriptive_distribution(values: Sequence[float]) -> dict[str, float | int]:
    """Freeze the observed task-level effect distribution without implying iid precision.

    The inferential unit remains the source/topic cluster in the bootstrap above.  These
    quantiles answer the separate operational question "was the average hiding a bad tail?"
    and are explicitly descriptive: correlated tasks do not become independent merely because
    we report their median or p05.
    """
    array = np.asarray(values, dtype=float)
    if not len(array) or not np.all(np.isfinite(array)):
        raise ValueError("task-level effect distribution requires finite observations")
    return {
        "n_tasks": int(len(array)),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "p05": float(np.quantile(array, 0.05)),
        "p10": float(np.quantile(array, 0.10)),
        "p90": float(np.quantile(array, 0.90)),
        "p95": float(np.quantile(array, 0.95)),
        "minimum": float(np.min(array)),
        "maximum": float(np.max(array)),
    }


def _task_effect_distributions(
    outcomes: Sequence[ArmOutcomes], *, service_work: bool
) -> dict[str, dict[str, float | int]]:
    if service_work:
        values = {
            "h_simple": [1.0 - item.h / item.p0 for item in outcomes],
            "c_simple": [1.0 - item.c / item.p0 for item in outcomes],
            "joint": [1.0 - item.hc / item.p0 for item in outcomes],
            # Positive means joint work is lower than the multiplicative combination of the
            # two simple work ratios, matching the inferential interaction orientation.
            "interaction": [1.0 - (item.hc * item.p0) / (item.h * item.c) for item in outcomes],
        }
        scale = "per_task_fraction_saved"
    else:
        values = {
            "h_simple": [item.h - item.p0 for item in outcomes],
            "c_simple": [item.c - item.p0 for item in outcomes],
            "joint": [item.hc - item.p0 for item in outcomes],
            "interaction": [item.hc - item.h - item.c + item.p0 for item in outcomes],
        }
        scale = "per_task_treatment_minus_baseline"
    return {
        name: {
            **_descriptive_distribution(observed),
            "scale": scale,
            "inference": "DESCRIPTIVE_ONLY_CLUSTER_BOOTSTRAP_REMAINS_PRIMARY",
        }
        for name, observed in values.items()
    }


def _proportion_with_cluster_ci(
    values: Sequence[bool],
    clusters: Sequence[str],
    *,
    seed: int,
    n_boot: int,
) -> dict[str, Any]:
    numeric = [float(value) for value in values]
    if len(set(clusters)) < 2:
        return {
            "status": "NOT_ESTIMABLE_FEWER_THAN_TWO_SOURCE_TOPIC_CLUSTERS",
            "point": float(np.mean(numeric)) if numeric else None,
            "n_tasks": len(numeric),
        }
    ci = cluster_bootstrap_ci(
        numeric,
        clusters,
        np.mean,
        n_boot=n_boot,
        seed=seed,
        side="two",
    )
    return {
        "status": "ESTIMABLE",
        "point": float(ci.point),
        "lower": float(ci.lower),
        "upper": float(ci.upper),
        "n_tasks": len(numeric),
        "n_clusters": int(ci.n_clusters),
        "n_boot": int(ci.n_boot),
        "interval_type": "two_sided_95_percent_source_topic_cluster_bootstrap",
    }


def _task_level_joint_outcomes(
    blocks: Sequence[Mapping[str, Any]],
    *,
    policy: Mapping[str, Any] | None,
    n_boot: int,
    seed_namespace: str,
) -> dict[str, Any]:
    """Joint complete-work/quality outcomes for the exact primary H/C/H+C arms."""
    if not isinstance(policy, Mapping):
        return {
            "status": "NOT_ESTABLISHED",
            "reason": "FROZEN_TASK_LEVEL_JOINT_OUTCOME_POLICY_MISSING",
        }
    higher = policy.get("higher_is_better_ni_margins")
    lower = policy.get("lower_is_better_increase_max")
    thresholds_raw = policy.get("service_work_saving_thresholds")
    if (
        policy.get("schema_version") != "task_level_joint_outcomes_policy_v1"
        or policy.get("quality_view") != "strict"
        or not isinstance(higher, Mapping)
        or not isinstance(lower, Mapping)
        or set(lower) != {"critical_harm"}
        or not isinstance(thresholds_raw, Sequence)
        or isinstance(thresholds_raw, str | bytes)
    ):
        raise ValueError("task-level joint-outcome policy is incomplete")
    expected_higher = {
        "weighted_required_atom_recall",
        "grounded_claim_precision",
        "citation_correctness",
        "citation_association",
        "required_facet_coverage",
        "qualified_report",
    }
    thresholds = tuple(float(value) for value in thresholds_raw)
    if (
        set(higher) != expected_higher
        or len(thresholds) != 3
        or tuple(sorted(thresholds)) != thresholds
        or len(set(thresholds)) != 3
        or any(not 0.0 < value < 1.0 for value in thresholds)
        or any(not math.isfinite(float(value)) for value in higher.values())
        or any(not math.isfinite(float(value)) for value in lower.values())
    ):
        raise ValueError("task-level joint-outcome margins or thresholds are invalid")

    invalid = [block for block in blocks if not bool(block["valid_for_paired_estimate"])]
    if invalid:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "ALL_OFFERED_CONTAINS_PAIRED_INVALID_BLOCK",
            "paired_invalid_blocks": [str(block["block_id"]) for block in invalid],
        }

    by_task: dict[str, dict[str, Any]] = {}
    missing: list[dict[str, str]] = []
    metrics = (*sorted(higher), *sorted(lower))
    for block in blocks:
        task_id = str(block["task_id"])
        cluster_id = str(block["cluster_id"])
        task = by_task.setdefault(
            task_id,
            {
                "cluster_id": cluster_id,
                "work": {semantic: [] for semantic in ("p0", "h", "c", "hc")},
                "quality": {
                    semantic: {metric: [] for metric in metrics}
                    for semantic in ("p0", "h", "c", "hc")
                },
            },
        )
        if task["cluster_id"] != cluster_id:
            raise ValueError(f"task {task_id!r} appears in multiple source/topic clusters")
        for semantic in ("p0", "h", "c", "hc"):
            row = block["arms"][semantic]
            work = _endpoint_optional(
                row,
                PRIMARY_WORK_ENDPOINT,
                quality_view="strict",
                quality_metric="weighted_required_atom_recall",
            )
            if work is None or work <= 0:
                missing.append(
                    {
                        "block_id": str(block["block_id"]),
                        "semantic": semantic,
                        "endpoint": PRIMARY_WORK_ENDPOINT,
                    }
                )
            else:
                task["work"][semantic].append(float(work))
            for metric in metrics:
                value, state = _quality_measurement(row, view="strict", metric=metric)
                if value is None:
                    missing.append(
                        {
                            "block_id": str(block["block_id"]),
                            "semantic": semantic,
                            "endpoint": f"strict.{metric}",
                            "state": state,
                        }
                    )
                else:
                    task["quality"][semantic][metric].append(float(value))
    if missing:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "PARTIAL_JOINT_WORK_OR_QUALITY_MEASUREMENT_MISSING_FAIL_CLOSED",
            "missing": missing,
        }

    task_rows: list[dict[str, Any]] = []
    for task_id, task in sorted(by_task.items()):
        if any(not values for values in task["work"].values()) or any(
            not values for per_metric in task["quality"].values() for values in per_metric.values()
        ):
            raise ValueError(f"task {task_id!r} has an empty joint-outcome replicate family")
        task_rows.append(
            {
                "task_id": task_id,
                "cluster_id": task["cluster_id"],
                "work": {
                    semantic: float(np.mean(values)) for semantic, values in task["work"].items()
                },
                "quality": {
                    semantic: {
                        metric: float(np.mean(values)) for metric, values in per_metric.items()
                    }
                    for semantic, per_metric in task["quality"].items()
                },
            }
        )
    clusters = [str(row["cluster_id"]) for row in task_rows]
    if len(set(clusters)) < 2:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "FEWER_THAN_TWO_SOURCE_TOPIC_CLUSTERS",
            "tasks": len(task_rows),
        }

    output: dict[str, Any] = {}
    for label, semantic in (("h_simple", "h"), ("c_simple", "c"), ("joint", "hc")):
        savings: list[float] = []
        quality_passes: list[bool] = []
        for row in task_rows:
            saving = 1.0 - float(row["work"][semantic]) / float(row["work"]["p0"])
            quality_pass = all(
                row["quality"][semantic][metric] - row["quality"]["p0"][metric] >= float(margin)
                for metric, margin in higher.items()
            ) and all(
                row["quality"][semantic][metric] - row["quality"]["p0"][metric] <= float(maximum)
                for metric, maximum in lower.items()
            )
            savings.append(saving)
            quality_passes.append(quality_pass)
        threshold_results = {
            f"saving_ge_{round(threshold * 100):02d}pct": _proportion_with_cluster_ci(
                [saving >= threshold for saving in savings],
                clusters,
                seed=seed_from(seed_namespace, "task_joint", label, str(threshold)),
                n_boot=n_boot,
            )
            for threshold in thresholds
        }
        output[label] = {
            "saving_threshold_proportions": threshold_results,
            "quality_guard_pass_proportion": _proportion_with_cluster_ci(
                quality_passes,
                clusters,
                seed=seed_from(seed_namespace, "task_joint", label, "quality_pass"),
                n_boot=n_boot,
            ),
            "quality_qualified_pareto_win_proportion": _proportion_with_cluster_ci(
                [
                    saving > 0.0 and quality_pass
                    for saving, quality_pass in zip(savings, quality_passes, strict=True)
                ],
                clusters,
                seed=seed_from(seed_namespace, "task_joint", label, "pareto"),
                n_boot=n_boot,
            ),
            "slower_and_quality_harmed_proportion": _proportion_with_cluster_ci(
                [
                    saving < 0.0 and not quality_pass
                    for saving, quality_pass in zip(savings, quality_passes, strict=True)
                ],
                clusters,
                seed=seed_from(seed_namespace, "task_joint", label, "joint_adverse"),
                n_boot=n_boot,
            ),
        }
    return {
        "status": "OK",
        "scope": "TASK_LEVEL_PRIMARY_H_C_HC_JOINT_OUTCOMES",
        "policy": json.loads(canonical_json(dict(policy))),
        "policy_sha256": sha256_hex(canonical_json(dict(policy))),
        "tasks": len(task_rows),
        "clusters": len(set(clusters)),
        "replicate_reduction": "task_mean_before_source_topic_cluster_bootstrap",
        "arms": output,
    }


def _factorial_for(
    blocks: Sequence[dict],
    *,
    endpoint: str,
    quality_view: str,
    quality_metric: str,
    direction: str,
    n_boot: int,
    seed: int,
) -> dict:
    scale = (
        "paired_log_ratio_saving"
        if endpoint in SAVING_SCALE_ENDPOINTS
        else "treatment_minus_baseline"
    )
    invalid = [
        {
            "block_id": block["block_id"],
            "reason": str(block["invalid_reason"]),
            "engine_epochs": list(block["engine_epochs"]),
        }
        for block in blocks
        if not block["valid_for_paired_estimate"]
    ]
    if invalid:
        return _not_estimable(
            direction=direction,
            scale=scale,
            reason="ALL_OFFERED_CONTAINS_PAIRED_INVALID_BLOCK",
            paired_invalid_blocks=invalid,
        )

    missing = {semantic: 0 for semantic in ("p0", "h", "c", "hc")}
    partial: list[str] = []
    not_applicable: list[str] = []
    complete: list[dict] = []
    for block in blocks:
        quality_states: dict[str, str] = {}
        if endpoint == "quality":
            measured = {
                semantic: _quality_measurement(row, view=quality_view, metric=quality_metric)
                for semantic, row in block["arms"].items()
            }
            values = {semantic: item[0] for semantic, item in measured.items()}
            quality_states = {semantic: item[1] for semantic, item in measured.items()}
        else:
            values = {
                semantic: _endpoint_optional(
                    row,
                    endpoint,
                    quality_view=quality_view,
                    quality_metric=quality_metric,
                )
                for semantic, row in block["arms"].items()
            }
        absent = [semantic for semantic, value in values.items() if value is None]
        for semantic in absent:
            missing[semantic] += 1
        if endpoint == "quality" and set(quality_states.values()) == {"INAPPLICABLE"}:
            not_applicable.append(block["block_id"])
            continue
        if absent:
            partial.append(block["block_id"])
            continue
        complete.append({**block, "values": values})

    if partial:
        return _not_estimable(
            direction=direction,
            scale=scale,
            reason="PARTIAL_ARM_MEASUREMENT_MISSING_FAIL_CLOSED",
            missing_by_arm=missing,
            partial_missing_blocks=partial,
        )
    if endpoint == "quality" and not complete:
        return {
            "status": "NOT_APPLICABLE",
            "direction": direction,
            "contrast_scale": scale,
            "reason": "TRUTH_GUARD_INAPPLICABLE_IN_ALL_FOUR_CORNERS",
            "not_applicable_blocks": sorted(not_applicable),
            "missing_counts": {
                "total": sum(missing.values()),
                "by_arm": missing,
                "partial_missing_blocks": [],
            },
        }
    if not complete:
        return _not_estimable(
            direction=direction,
            scale=scale,
            reason="ENDPOINT_MISSING_IN_ALL_BLOCKS",
            missing_by_arm=missing,
        )
    outcomes = _task_mean_outcomes(complete)
    if len({outcome.cluster_id for outcome in outcomes}) < 2:
        return _not_estimable(
            direction=direction,
            scale=scale,
            reason="FEWER_THAN_TWO_ESTIMABLE_SOURCE_TOPIC_CLUSTERS",
            missing_by_arm=missing,
        )
    common = {
        "status": "OK",
        "direction": direction,
        "replicate_reduction": "task_mean_before_cluster_bootstrap",
        "tasks_estimable": len(outcomes),
        "clusters_estimable": len({outcome.cluster_id for outcome in outcomes}),
        "not_applicable_blocks": sorted(not_applicable),
        "missing_counts": {
            "total": sum(missing.values()),
            "by_arm": missing,
            "partial_missing_blocks": [],
        },
    }
    if endpoint in SAVING_SCALE_ENDPOINTS:
        p0 = [item.p0 for item in outcomes]
        clusters = [item.cluster_id for item in outcomes]
        comparisons = {
            "h_simple": [item.h for item in outcomes],
            "c_simple": [item.c for item in outcomes],
            "joint": [item.hc for item in outcomes],
        }
        result = {
            name: _saving_ci(
                paired_log_ratio_saving(
                    values,
                    p0,
                    clusters,
                    n_boot=n_boot,
                    seed=seed_from(str(seed), name),
                )
            )
            for name, values in comparisons.items()
        }
        # Positive interaction means H+C saves more work than the product of the two simple
        # work ratios would predict.
        result["interaction"] = _saving_ci(
            paired_log_ratio_saving(
                [item.hc * item.p0 for item in outcomes],
                [item.h * item.c for item in outcomes],
                clusters,
                n_boot=n_boot,
                seed=seed_from(str(seed), "interaction"),
            )
        )
        return {
            **common,
            "contrast_scale": scale,
            "positive_means": "P1_SAVES_SERVICE_WORK",
            "interval": "one_sided_95_percent_lower_confidence_bound",
            "task_effect_distribution": _task_effect_distributions(outcomes, service_work=True),
            **result,
        }
    contrasts = estimate_contrasts(outcomes, n_boot=n_boot, seed=seed)
    return {
        **common,
        "contrast_scale": scale,
        "task_effect_distribution": _task_effect_distributions(outcomes, service_work=False),
        "h_simple": _ci(contrasts.h_simple),
        "c_simple": _ci(contrasts.c_simple),
        "joint": _ci(contrasts.joint),
        "interaction": _ci(contrasts.interaction),
    }


def _verify_eligibility_spec(spec: Mapping[str, Any]) -> dict:
    body = dict(spec)
    if str(body.get("content_sha256") or "") != _unsigned_sha(body):
        raise ValueError("eligibility spec is not hash-frozen")
    if body.get("schema_version") != "e2e_eligibility_spec_v2":
        raise ValueError("unsupported eligibility spec schema")
    if body.get("analysis_scope") != _ELIGIBILITY_SCOPE:
        raise ValueError("eligibility is restricted to formative task-level pre-treatment analysis")
    if body.get("quality_view") != "strict":
        raise ValueError("eligibility labels must use the strict quality view")
    if len(str(body.get("task_feature_registry_sha256") or "")) != 64:
        raise ValueError("eligibility spec does not bind a task feature registry")
    raw_targets = body.get("targets")
    if not isinstance(raw_targets, list):
        raise ValueError("eligibility spec lacks its exact target family")
    targets: list[dict] = []
    target_ids: list[str] = []
    for raw in raw_targets:
        if not isinstance(raw, Mapping):
            raise ValueError("eligibility target is malformed")
        target = dict(raw)
        expected_fields = {
            "target_id",
            "treatment_semantic",
            "treatment_arm_id",
            "comparator_semantics",
            "comparator_arm_ids",
            "comparator_logic",
        }
        if set(target) != expected_fields:
            raise ValueError("eligibility target fields differ from the frozen schema")
        target_id = str(target["target_id"])
        target_ids.append(target_id)
        expected = ELIGIBILITY_TARGET_FAMILY.get(target_id)
        treatment = str(target["treatment_semantic"])
        comparators = tuple(map(str, target["comparator_semantics"]))
        comparator_arm_ids = tuple(map(str, target["comparator_arm_ids"]))
        if (
            expected is None
            or (treatment, comparators) != expected
            or str(target["comparator_logic"]) != "ALL"
            or not str(target["treatment_arm_id"])
            or len(comparator_arm_ids) != len(comparators)
            or any(not item for item in comparator_arm_ids)
        ):
            raise ValueError(
                f"eligibility target {target_id!r} differs from the predeclared family"
            )
        targets.append(target)
    if target_ids != list(ELIGIBILITY_TARGET_FAMILY):
        raise ValueError("eligibility target family is missing, reordered, duplicated, or extended")
    features = tuple(map(str, body.get("feature_names") or ()))
    if not features or len(features) != len(set(features)):
        raise ValueError("eligibility feature_names must be a non-empty unique list")
    forbidden = sorted(
        feature
        for feature in features
        if feature not in ALLOWED_PRETREATMENT_FEATURES
        or any(fragment in feature.lower() for fragment in _FORBIDDEN_FEATURE_FRAGMENTS)
    )
    if forbidden:
        raise ValueError(f"eligibility spec uses non-pre-treatment features: {forbidden}")
    raw_guards = body.get("quality_guards")
    if not isinstance(raw_guards, Mapping):
        raise ValueError("eligibility spec lacks strict quality guards")
    higher_raw = raw_guards.get("higher_is_better_min_effect")
    lower_raw = raw_guards.get("lower_is_better_max_effect")
    if not isinstance(higher_raw, Mapping) or not isinstance(lower_raw, Mapping):
        raise ValueError("eligibility strict quality guards are malformed")
    higher = {
        str(metric): _finite(value, label=f"higher guard {metric}")
        for metric, value in higher_raw.items()
    }
    lower = {
        str(metric): _finite(value, label=f"lower guard {metric}", nonnegative=True)
        for metric, value in lower_raw.items()
    }
    expected_higher = {
        "weighted_required_atom_recall",
        "grounded_claim_precision",
        "citation_correctness",
        "citation_association",
        "required_facet_coverage",
        "qualified_report",
    }
    if (
        set(higher) != expected_higher
        or set(lower) != {"critical_harm"}
        or any(value > 0 for value in higher.values())
        or raw_guards.get("inapplicability_policy")
        != "BOTH_ARMS_INAPPLICABLE_PASS_OTHERWISE_FAIL_CLOSED"
    ):
        raise ValueError("eligibility spec does not carry the complete frozen strict guard set")
    absolute = body.get("absolute_treatment_requirements")
    if (
        not isinstance(absolute, Mapping)
        or set(absolute) != {"qualified_report_min", "replicate_policy"}
        or _finite(
            absolute.get("qualified_report_min"),
            label="absolute qualified_report_min",
            nonnegative=True,
        )
        != 1.0
        or absolute.get("replicate_policy") != "ALL_REPLICATES_MUST_QUALIFY"
    ):
        raise ValueError(
            "eligibility must require a qualified strict treatment report on every replicate"
        )
    if int(body.get("max_depth", -1)) != 2:
        raise ValueError("eligibility max_depth is frozen at 2")
    if int(body.get("min_tasks_per_leaf", 0)) < 8:
        raise ValueError("eligibility needs at least 8 independent tasks per leaf")
    training = tuple(map(str, body.get("training_splits") or ()))
    heldout = tuple(map(str, body.get("heldout_splits") or ()))
    excluded = tuple(map(str, body.get("excluded_splits") or ()))
    split_groups = (set(training), set(heldout), set(excluded))
    if not training or any(
        left & right
        for index, left in enumerate(split_groups)
        for right in split_groups[index + 1 :]
    ):
        raise ValueError("training, heldout, and excluded eligibility splits must be disjoint")
    saving = _finite(body.get("minimum_work_saving"), label="minimum_work_saving", nonnegative=True)
    if saving >= 1:
        raise ValueError("minimum_work_saving must be below 1")
    threshold = _finite(body.get("eligible_threshold"), label="eligible_threshold")
    if not 0 <= threshold <= 1:
        raise ValueError("eligible_threshold must be in [0,1]")
    if int(body.get("cv_folds", 0)) < 2:
        raise ValueError("eligibility cv_folds must be at least 2")
    if int(body.get("stability_bootstraps", 0)) <= 0:
        raise ValueError("eligibility stability_bootstraps must be positive")
    minimum_stability = _finite(body.get("minimum_root_stability"), label="minimum_root_stability")
    if minimum_stability != 0.70:
        raise ValueError("eligibility minimum_root_stability is frozen at 0.70")
    body["_features"] = features
    body["_targets"] = targets
    body["_higher_guards"] = higher
    body["_lower_guards"] = lower
    body["_qualified_report_min"] = 1.0
    body["_training_splits"] = training
    body["_heldout_splits"] = heldout
    body["_excluded_splits"] = excluded
    body["_saving"] = saving
    body["_threshold"] = threshold
    body["_minimum_stability"] = minimum_stability
    return body


def _numeric_features(task: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    source = task["features"]
    result: dict[str, float] = {}
    for name in names:
        value = _finite(source.get(name), label=f"{task['task_id']}.{name}")
        result[name] = value
    return result


def _quality_guard_for_comparator(
    task_blocks: Sequence[Mapping[str, Any]],
    *,
    treatment_semantic: str,
    comparator_semantic: str,
    quality_view: str,
    higher_guards: Mapping[str, float],
    lower_guards: Mapping[str, float],
) -> tuple[bool, dict[str, dict]]:
    results: dict[str, dict] = {}
    for metric, direction, margin in (
        *(
            (metric, "higher_is_better", margin)
            for metric, margin in higher_guards.items()
        ),
        *(
            (metric, "lower_is_better", margin)
            for metric, margin in lower_guards.items()
        ),
    ):
        paired_values: list[tuple[float, float]] = []
        inapplicable = 0
        for replicate in task_blocks:
            treatment_value, treatment_state = _quality_measurement(
                replicate["arms"][treatment_semantic],
                view=quality_view,
                metric=metric,
            )
            comparator_value, comparator_state = _quality_measurement(
                replicate["arms"][comparator_semantic],
                view=quality_view,
                metric=metric,
            )
            if treatment_state == comparator_state == "INAPPLICABLE":
                inapplicable += 1
                continue
            if treatment_state != "VALUE" or comparator_state != "VALUE":
                raise ValueError(
                    f"eligibility {metric} is missing or has arm-asymmetric applicability "
                    f"for {treatment_semantic} vs {comparator_semantic}"
                )
            assert treatment_value is not None and comparator_value is not None
            paired_values.append((treatment_value, comparator_value))
        if inapplicable and paired_values:
            raise ValueError(
                f"eligibility {metric} applicability changes across replicates for one task"
            )
        if not paired_values:
            results[metric] = {
                "direction": direction,
                "margin": margin,
                "status": "NOT_APPLICABLE_PASS",
                "treatment_mean": None,
                "comparator_mean": None,
                "effect_treatment_minus_comparator": None,
                "pass": True,
            }
            continue
        treatment_mean = sum(pair[0] for pair in paired_values) / len(paired_values)
        comparator_mean = sum(pair[1] for pair in paired_values) / len(paired_values)
        effect = treatment_mean - comparator_mean
        passed = (
            effect + 1e-12 >= margin
            if direction == "higher_is_better"
            else effect <= margin + 1e-12
        )
        results[metric] = {
            "direction": direction,
            "margin": margin,
            "status": "VALUE",
            "treatment_mean": treatment_mean,
            "comparator_mean": comparator_mean,
            "effect_treatment_minus_comparator": effect,
            "pass": passed,
        }
    return all(item["pass"] for item in results.values()), results


def _target_observation(
    task_blocks: Sequence[Mapping[str, Any]],
    spec: Mapping[str, Any],
    target: Mapping[str, Any],
) -> tuple[Sample, dict]:
    if not task_blocks:
        raise ValueError("eligibility task has no offered blocks")
    block = task_blocks[0]
    target_id = str(target["target_id"])
    treatment = str(target["treatment_semantic"])
    comparators = tuple(map(str, target["comparator_semantics"]))
    comparator_results: list[dict] = []
    for comparator in comparators:
        quality_pass, quality_results = _quality_guard_for_comparator(
            task_blocks,
            treatment_semantic=treatment,
            comparator_semantic=comparator,
            quality_view=str(spec["quality_view"]),
            higher_guards=spec["_higher_guards"],
            lower_guards=spec["_lower_guards"],
        )
        qualified_values: list[float] = []
        for replicate in task_blocks:
            qualified, qualified_state = _quality_measurement(
                replicate["arms"][treatment],
                view=str(spec["quality_view"]),
                metric="qualified_report",
            )
            if qualified_state != "VALUE" or qualified is None:
                raise ValueError(
                    f"eligibility treatment {treatment} has no strict qualified_report value"
                )
            qualified_values.append(qualified)
        absolute_treatment_pass = all(
            value + 1e-12 >= float(spec["_qualified_report_min"])
            for value in qualified_values
        )
        treatment_work_values = [
            _endpoint(
                replicate["arms"][treatment],
                PRIMARY_WORK_ENDPOINT,
                quality_view=str(spec["quality_view"]),
                quality_metric="weighted_required_atom_recall",
            )
            for replicate in task_blocks
        ]
        comparator_work_values = [
            _endpoint(
                replicate["arms"][comparator],
                PRIMARY_WORK_ENDPOINT,
                quality_view=str(spec["quality_view"]),
                quality_metric="weighted_required_atom_recall",
            )
            for replicate in task_blocks
        ]
        treatment_work = sum(treatment_work_values) / len(treatment_work_values)
        comparator_work = sum(comparator_work_values) / len(comparator_work_values)
        work_saving = 1.0 - treatment_work / comparator_work
        work_pass = work_saving + 1e-12 >= float(spec["_saving"])
        comparator_results.append({
            "comparator_semantic": comparator,
            "comparator_arm_id": spec["_arm_map"][comparator],
            "strict_quality_guards_pass": quality_pass,
            "strict_quality_guards": quality_results,
            "absolute_treatment_requirement": {
                "metric": "qualified_report",
                "minimum": spec["_qualified_report_min"],
                "replicate_policy": "ALL_REPLICATES_MUST_QUALIFY",
                "replicate_values": qualified_values,
                "pass": absolute_treatment_pass,
            },
            "treatment_service_work_mean": treatment_work,
            "comparator_service_work_mean": comparator_work,
            "service_work_saving_fraction": work_saving,
            "minimum_service_work_saving": spec["_saving"],
            "service_work_saving_pass": work_pass,
            "composite_pass": (
                absolute_treatment_pass and quality_pass and work_pass
            ),
        })
    success = all(item["composite_pass"] for item in comparator_results)
    numeric_features = _numeric_features(block["task_features"], spec["_features"])
    sample = Sample(
        # Replicates from one task are correlated.  Keeping the real task ID here makes the
        # existing CART's distinct-task leaf guard and task bootstrap count them once.
        task_id=str(block["task_id"]),
        features=numeric_features,
        training_success=success,
        cluster_id=str(block["cluster_id"]),
    )
    label_record = {
        "task_id": str(block["task_id"]),
        "replicate_ids": sorted(str(item["replicate_id"]) for item in task_blocks),
        "replicate_reduction": "task_mean",
        "cluster_id": str(block["cluster_id"]),
        "split": str(block["task_features"]["split"]),
        "task_feature_content_sha256": str(block["task_features"]["content_sha256"]),
        "features": numeric_features,
        "target_id": target_id,
        "treatment_semantic": treatment,
        "treatment_arm_id": spec["_arm_map"][treatment],
        "comparator_logic": "ALL",
        "comparator_results": comparator_results,
        "composite_success": success,
    }
    return sample, label_record


def _cluster_cv(
    samples: Sequence[Sample],
    cluster_by_sample: Mapping[str, str],
    *,
    feature_names: Sequence[str],
    min_tasks_per_leaf: int,
    eligible_threshold: float,
    folds: int,
) -> dict:
    clusters = sorted(set(cluster_by_sample.values()))
    if len(clusters) < 2:
        raise ValueError("eligibility cluster CV needs at least two source/topic clusters")
    n_folds = min(max(2, folds), len(clusters))
    cluster_folds = [clusters[index::n_folds] for index in range(n_folds)]
    predictions: list[tuple[Sample, bool]] = []
    root_features: list[str | None] = []
    fold_records: list[dict] = []
    for fold_index, validation_clusters in enumerate(cluster_folds):
        validation_set = set(validation_clusters)
        train = [
            sample for sample in samples if cluster_by_sample[sample.task_id] not in validation_set
        ]
        validation = [
            sample for sample in samples if cluster_by_sample[sample.task_id] in validation_set
        ]
        if not train or not validation:
            raise ValueError("cluster CV produced an empty train or validation fold")
        if len({sample.task_id for sample in train}) < 2 * min_tasks_per_leaf:
            raise ValueError("cluster CV training fold cannot support two minimum-size leaves")
        rule = fit_eligibility_tree(
            train,
            feature_names,
            max_depth=2,
            min_tasks_per_leaf=min_tasks_per_leaf,
            eligible_threshold=eligible_threshold,
        )
        root_features.append(rule.root_split_feature)
        fold_predictions = [(sample, rule.predict(sample.features)) for sample in validation]
        predictions.extend(fold_predictions)
        fold_records.append(
            {
                "fold": fold_index,
                "validation_clusters": validation_clusters,
                "training_tasks": len({sample.task_id for sample in train}),
                "validation_tasks": len({sample.task_id for sample in validation}),
                "root_split_feature": rule.root_split_feature,
                "rule_hash": rule.rule_hash,
            }
        )
    eligible = [(sample, pred) for sample, pred in predictions if pred]
    accuracy = sum(int(pred == sample.training_success) for sample, pred in predictions) / len(
        predictions
    )
    precision = (
        sum(int(sample.training_success) for sample, _ in eligible) / len(eligible)
        if eligible
        else 0.0
    )
    return {
        "folds": fold_records,
        "predictions": len(predictions),
        "oof_task_accuracy": accuracy,
        "eligible_task_precision": precision,
        "eligible_task_count": len(eligible),
        "oof_task_coverage": len(eligible) / len(predictions),
        "_root_features": root_features,
    }


def _cluster_equal_coverage(
    rule,
    samples: Sequence[Sample],
    cluster_by_sample: Mapping[str, str],
) -> dict[str, float]:
    by_cluster_tasks: dict[str, dict[str, bool]] = {}
    for sample in samples:
        cluster = cluster_by_sample[sample.task_id]
        eligible = rule.predict(sample.features)
        tasks = by_cluster_tasks.setdefault(cluster, {})
        tasks[sample.task_id] = tasks.get(sample.task_id, False) or eligible
    return {
        "cluster_equal_task_coverage": sum(
            sum(tasks.values()) / len(tasks) for tasks in by_cluster_tasks.values()
        )
        / len(by_cluster_tasks),
    }


def _eligibility(
    blocks: Sequence[dict],
    spec_raw: Mapping[str, Any],
    *,
    arm_map: Mapping[str, str],
    seed_namespace: str,
) -> dict:
    spec = _verify_eligibility_spec(spec_raw)
    spec["_arm_map"] = dict(arm_map)
    for target in spec["_targets"]:
        treatment = str(target["treatment_semantic"])
        comparators = tuple(map(str, target["comparator_semantics"]))
        if (
            str(target["treatment_arm_id"]) != str(arm_map.get(treatment) or "")
            or tuple(map(str, target["comparator_arm_ids"]))
            != tuple(str(arm_map.get(item) or "") for item in comparators)
        ):
            raise ValueError(
                f"eligibility target {target['target_id']} does not bind the executed arm map"
            )
    observed_splits = {str(block["task_features"]["split"]) for block in blocks}
    declared_splits = (
        set(spec["_training_splits"]) | set(spec["_heldout_splits"]) | set(spec["_excluded_splits"])
    )
    if observed_splits - declared_splits:
        raise ValueError(
            "eligibility spec does not classify observed splits: "
            f"{sorted(observed_splits - declared_splits)}"
        )
    training_blocks = [
        block for block in blocks if block["task_features"]["split"] in spec["_training_splits"]
    ]
    heldout_blocks = [
        block for block in blocks if block["task_features"]["split"] in spec["_heldout_splits"]
    ]
    if not training_blocks:
        raise ValueError("eligibility spec selected no training blocks")
    train_clusters = {block["cluster_id"] for block in training_blocks}
    heldout_clusters = {block["cluster_id"] for block in heldout_blocks}
    if train_clusters & heldout_clusters:
        raise ValueError(
            "a source/topic cluster appears in both training and heldout eligibility splits"
        )
    cluster_by_sample = {str(block["task_id"]): str(block["cluster_id"]) for block in blocks}
    min_leaf = int(spec["min_tasks_per_leaf"])
    threshold = float(spec["_threshold"])

    def observations_by_task(
        selected: Sequence[dict],
        target: Mapping[str, Any],
    ) -> list[tuple[Sample, dict]]:
        grouped: dict[str, list[dict]] = {}
        for item in selected:
            grouped.setdefault(str(item["task_id"]), []).append(item)
        return [
            _target_observation(
                sorted(rows, key=lambda row: str(row["replicate_id"])),
                spec,
                target,
            )
            for _, rows in sorted(grouped.items())
        ]
    target_results: dict[str, dict] = {}
    for target in spec["_targets"]:
        target_id = str(target["target_id"])
        try:
            training_observations = observations_by_task(training_blocks, target)
            training_samples = [sample for sample, _ in training_observations]
            training_label_rows = [row for _, row in training_observations]
            if len({sample.task_id for sample in training_samples}) < 2 * min_leaf:
                raise ValueError(
                    "eligibility training registry cannot support two minimum-size leaves"
                )
            rule = fit_eligibility_tree(
                training_samples,
                spec["_features"],
                max_depth=2,
                min_tasks_per_leaf=min_leaf,
                eligible_threshold=threshold,
            )
            cv = _cluster_cv(
                training_samples,
                cluster_by_sample,
                feature_names=spec["_features"],
                min_tasks_per_leaf=min_leaf,
                eligible_threshold=threshold,
                folds=int(spec.get("cv_folds", 5)),
            )
            full_root = rule.root_split_feature
            roots = cv.pop("_root_features")
            cv_stability = (
                sum(root == full_root for root in roots) / len(roots)
                if full_root is not None
                else 0.0
            )
            bootstrap_stability = rule_stability(
                training_samples,
                spec["_features"],
                n_boot=int(spec.get("stability_bootstraps", 200)),
                seed=seed_from(
                    seed_namespace,
                    target_id,
                    "eligibility-stability",
                ),
                min_tasks_per_leaf=min_leaf,
                eligible_threshold=threshold,
            )
            stability_pass = (
                full_root is not None
                and bootstrap_stability >= float(spec["_minimum_stability"])
            )
            if full_root is None:
                target_status = "NO_SUBGROUP_SPLIT"
            elif stability_pass:
                target_status = "STABLE_EXPLORATORY_RULE"
            else:
                target_status = "UNSTABLE_EXPLORATORY_RULE"
            training_dataset = {
                "schema_version": "eligibility_task_label_dataset_v2",
                "analysis_scope": _ELIGIBILITY_SCOPE,
                "eligibility_spec_content_sha256": spec["content_sha256"],
                "target_id": target_id,
                "rows": sorted(training_label_rows, key=lambda row: row["task_id"]),
            }
            training_dataset["content_sha256"] = sha256_hex(
                canonical_json(training_dataset)
            )
            training_predictions = [
                rule.predict(sample.features) for sample in training_samples
            ]
            target_result = {
                "target_id": target_id,
                "status": target_status,
                "analysis_scope": _ELIGIBILITY_SCOPE,
                "fit_scope": "TRAINING_ONLY",
                "treatment_semantic": target["treatment_semantic"],
                "treatment_arm_id": target["treatment_arm_id"],
                "comparator_semantics": list(target["comparator_semantics"]),
                "comparator_arm_ids": list(target["comparator_arm_ids"]),
                "comparator_logic": "ALL",
                "training_splits": list(spec["_training_splits"]),
                "heldout_splits": list(spec["_heldout_splits"]),
                "excluded_splits": list(spec["_excluded_splits"]),
                "training_rows": len(training_samples),
                "training_tasks": len({sample.task_id for sample in training_samples}),
                "training_clusters": len(train_clusters),
                "training_dataset": training_dataset,
                "composite_label": {
                    "quality_view": spec["quality_view"],
                    "quality_guards": {
                        "higher_is_better_min_effect": spec["_higher_guards"],
                        "lower_is_better_max_effect": spec["_lower_guards"],
                    },
                    "absolute_treatment_requirements": {
                        "qualified_report_min": spec["_qualified_report_min"],
                        "replicate_policy": "ALL_REPLICATES_MUST_QUALIFY",
                    },
                    "minimum_service_work_saving": spec["_saving"],
                    "comparator_logic": "ALL",
                    "successes": sum(
                        sample.training_success for sample in training_samples
                    ),
                    "tasks": len(training_samples),
                },
                "rule": {
                    "readable": rule.to_readable(),
                    "rule_hash": rule.rule_hash,
                    "root_split_feature": rule.root_split_feature,
                    "root_leaf_eligible_when_unsplit": (
                        bool(rule.root.eligible) if rule.root_split_feature is None else None
                    ),
                    "feature_names": list(rule.features),
                    "min_tasks_per_leaf": min_leaf,
                    "eligible_threshold": threshold,
                },
                "cluster_cv": {
                    **cv,
                    "root_split_stability": cv_stability,
                },
                "cluster_bootstrap_root_stability": bootstrap_stability,
                "minimum_root_stability": spec["_minimum_stability"],
                "exploratory_rule_stability_pass": stability_pass,
                "coverage_on_training_registry": {
                    "task_coverage": (
                        sum(training_predictions) / len(training_predictions)
                    ),
                    **_cluster_equal_coverage(
                        rule,
                        training_samples,
                        cluster_by_sample,
                    ),
                },
            }
            try:
                heldout_observations = observations_by_task(heldout_blocks, target)
                heldout_samples = [sample for sample, _ in heldout_observations]
                heldout_label_rows = [row for _, row in heldout_observations]
                if heldout_samples:
                    heldout_predictions = [
                        rule.predict(sample.features) for sample in heldout_samples
                    ]
                    eligible_count = sum(heldout_predictions)
                    heldout_dataset = {
                        "schema_version": "eligibility_task_label_dataset_v2",
                        "analysis_scope": _ELIGIBILITY_SCOPE,
                        "eligibility_spec_content_sha256": spec["content_sha256"],
                        "target_id": target_id,
                        "rows": sorted(
                            heldout_label_rows,
                            key=lambda row: row["task_id"],
                        ),
                    }
                    heldout_dataset["content_sha256"] = sha256_hex(
                        canonical_json(heldout_dataset)
                    )
                    target_result["heldout_evaluation"] = {
                        "status": (
                            "FORMATIVE_PREDICTION_ONLY"
                            if eligible_count
                            else "NOT_ESTIMABLE_ZERO_ELIGIBLE_TASKS"
                        ),
                        "fit_used_heldout": False,
                        "confirmatory_claim_allowed": False,
                        "rows": len(heldout_samples),
                        "clusters": len(heldout_clusters),
                        "eligible_tasks": eligible_count,
                        "task_coverage": eligible_count / len(heldout_samples),
                        **_cluster_equal_coverage(
                            rule,
                            heldout_samples,
                            cluster_by_sample,
                        ),
                        "eligible_success_rate": (
                            sum(
                                sample.training_success
                                for sample, prediction in zip(
                                    heldout_samples,
                                    heldout_predictions,
                                    strict=True,
                                )
                                if prediction
                            )
                            / eligible_count
                            if eligible_count
                            else None
                        ),
                        "label_dataset": heldout_dataset,
                    }
                else:
                    target_result["heldout_evaluation"] = {
                        "status": "NOT_PRESENT_IN_THIS_RUN_PHASE",
                        "fit_used_heldout": False,
                        "confirmatory_claim_allowed": False,
                        "rows": 0,
                    }
            except ValueError as exc:
                target_result["heldout_evaluation"] = {
                    "status": "NOT_ESTIMABLE",
                    "fit_used_heldout": False,
                    "confirmatory_claim_allowed": False,
                    "reason": str(exc),
                }
            target_results[target_id] = target_result
        except ValueError as exc:
            target_results[target_id] = {
                "target_id": target_id,
                "status": "NOT_ESTIMABLE",
                "analysis_scope": _ELIGIBILITY_SCOPE,
                "fit_scope": "TRAINING_ONLY",
                "treatment_semantic": target["treatment_semantic"],
                "treatment_arm_id": target["treatment_arm_id"],
                "comparator_semantics": list(target["comparator_semantics"]),
                "comparator_arm_ids": list(target["comparator_arm_ids"]),
                "comparator_logic": "ALL",
                "reason": str(exc),
            }
    return {
        "schema_version": "e2e_eligibility_result_v3",
        "analysis_scope": _ELIGIBILITY_SCOPE,
        "spec_content_sha256": spec["content_sha256"],
        "fit_scope": "TRAINING_ONLY",
        "claim_policy": {
            "primary_verdict_effect": "NONE",
            "deployment_or_invocation_envelope_allowed": False,
            "confirmatory_claim_allowed": False,
            "all_predeclared_targets_reported": True,
        },
        "target_results": target_results,
    }


def _prepare_blocks(
    records: Sequence[Mapping[str, Any]],
    *,
    task_features: Mapping[str, Mapping[str, Any]],
    scope_receipt: Mapping[str, Any] | None = None,
    arm_map: Mapping[str, str] | None = None,
    expected_variant_ids: Mapping[str, str] | None = None,
    freeze_root: Mapping[str, Any] | None = None,
    schedule_manifest: Mapping[str, Any] | None = None,
) -> dict:
    arms = dict(CORE_FACTORIAL_ARMS if arm_map is None else arm_map)
    if set(arms) != {"p0", "h", "c", "hc"} or len(set(arms.values())) != 4:
        raise ValueError("arm_map must name four distinct p0/h/c/hc arms")
    (
        run_id,
        phase_id,
        scope_sha,
        expected_rows,
        receipt,
    ) = _verify_score_scope(records, scope_receipt)
    provenance = _verify_root_provenance(
        receipt,
        expected_rows,
        freeze_root=freeze_root,
        schedule_manifest=schedule_manifest,
    )
    tasks = {str(record.get("task_id") or "") for record in records}
    if "" in tasks:
        raise ValueError("score record lacks task_id")
    features = _verify_task_features(task_features, tasks)
    feature_registry_sha = task_feature_registry_sha256(features)

    blocks: list[dict] = []
    seen_coordinates: set[tuple[str, str]] = set()
    for record in records:
        block_id = str(record["block_id"])
        task_id = str(record["task_id"])
        replicate_id = str(record.get("replicate_id") or "")
        if not replicate_id:
            raise ValueError(f"block {block_id} lacks replicate_id")
        coordinate = (task_id, replicate_id)
        if coordinate in seen_coordinates:
            raise ValueError(
                f"duplicate factorial task/replicate coordinate {task_id}:{replicate_id}"
            )
        seen_coordinates.add(coordinate)
        per_arm = record.get("per_arm")
        if not isinstance(per_arm, Mapping):
            raise ValueError(f"block {block_id} lacks per_arm records")
        selected = {
            semantic: _one_arm(per_arm, arm_id, replicate_id, block_id)
            for semantic, arm_id in arms.items()
        }
        for semantic, row in selected.items():
            if row.get("task_id") not in (None, task_id):
                raise ValueError(f"block {block_id} {semantic} row names another task")
        expected = expected_rows[block_id]
        expected_scope = {
            "freeze_root_sha256": str(receipt["freeze_root_sha256"]),
            "schedule_sha256": str(receipt["schedule_sha256"]),
            "block_freeze_sha256": str(expected["block_freeze_sha256"]),
            "block_digest": str(expected["block_digest"]),
            "valid_for_paired_estimate": expected["valid_for_paired_estimate"],
            "invalid_reason": str(expected.get("invalid_reason") or ""),
            "engine_epochs": list(expected["engine_epochs"]),
            "engine_epoch_by_arm": dict(expected["engine_epoch_by_arm"]),
        }
        if set(expected_scope["engine_epoch_by_arm"]) != {str(key) for key in per_arm}:
            raise ValueError(f"block {block_id} engine-epoch index differs from offered arms")
        for arm_key, row in per_arm.items():
            if not isinstance(row, Mapping) or row.get("frozen_scope") != expected_scope:
                raise ValueError(
                    f"block {block_id} arm {arm_key!r} is not bound to exact frozen scope"
                )
        blocks.append(
            {
                "block_id": block_id,
                "task_id": task_id,
                "replicate_id": replicate_id,
                "cluster_id": str(features[task_id]["cluster_id"]),
                "task_features": features[task_id],
                "arms": selected,
                "valid_for_paired_estimate": bool(expected["valid_for_paired_estimate"]),
                "invalid_reason": str(expected.get("invalid_reason") or ""),
                "engine_epochs": list(expected["engine_epochs"]),
                "block_freeze_sha256": str(expected["block_freeze_sha256"]),
                "block_digest": str(expected["block_digest"]),
            }
        )
    verified_variants = (
        validate_factorial_variants(blocks, expected_variant_ids)
        if expected_variant_ids is not None
        else None
    )
    return {
        "run_id": run_id,
        "phase_id": phase_id,
        "scope_sha": scope_sha,
        "receipt": receipt,
        "provenance": provenance,
        "feature_registry_sha": feature_registry_sha,
        "features": features,
        "arm_map": arms,
        "expected_variant_ids": verified_variants,
        "blocks": blocks,
    }


def build_factorial_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    task_features: Mapping[str, Mapping[str, Any]],
    scope_receipt: Mapping[str, Any] | None = None,
    arm_map: Mapping[str, str] | None = None,
    expected_variant_ids: Mapping[str, str] | None = None,
    freeze_root: Mapping[str, Any] | None = None,
    schedule_manifest: Mapping[str, Any] | None = None,
    quality_view: str = "fallback_assisted",
    quality_metric: str = "weighted_required_atom_recall",
    task_level_joint_outcomes_policy: Mapping[str, Any] | None = None,
    n_boot: int = 2000,
    seed_namespace: str = "e2e-effects-v2",
) -> dict:
    """Build the core all-offered factorial without depending on eligibility fitting."""
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    prepared = _prepare_blocks(
        records,
        task_features=task_features,
        scope_receipt=scope_receipt,
        arm_map=arm_map,
        expected_variant_ids=expected_variant_ids,
        freeze_root=freeze_root,
        schedule_manifest=schedule_manifest,
    )
    blocks = prepared["blocks"]
    run_id = prepared["run_id"]
    phase_id = prepared["phase_id"]
    if quality_metric not in QUALITY_GUARD_DIRECTIONS:
        raise ValueError(f"unregistered quality guard metric {quality_metric!r}")

    # `service_work_seconds` is kept but is the *serialized* layer's metric: it is a sum of
    # per-request intervals and is only defined when nothing overlaps, so under the
    # native-concurrent primary layer it is simply absent. `interval_union_seconds` answers the
    # same question -- how long was the engine busy on this cell -- without requiring the system
    # under test to be serialized to make the arithmetic work.
    #
    # Token counts stay split. Prompt, completion and cached tokens are priced differently, come
    # from different phases of inference, and move in opposite directions between P0 and P1
    # (P1 trades a longer selector prefill for a much shorter decode). Summing them into one
    # "tokens" number and thresholding it would hide exactly the trade the study is about.
    operational_endpoints = (
        "interval_union_seconds",
        "service_work_seconds",
        "e2e_latency_seconds",
        "energy_joules",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
    )
    directions = {
        "interval_union_seconds": "higher_saving_is_better",
        "service_work_seconds": "higher_saving_is_better",
        "e2e_latency_seconds": "lower_is_better",
        "energy_joules": "lower_is_better",
        "prompt_tokens": "lower_is_better",
        "completion_tokens": "lower_is_better",
        "cached_prompt_tokens": "descriptive_only",
    }
    endpoint_results: dict[str, dict] = {
        "quality": _factorial_for(
            blocks,
            endpoint="quality",
            quality_view=quality_view,
            quality_metric=quality_metric,
            direction=QUALITY_GUARD_DIRECTIONS[quality_metric],
            n_boot=n_boot,
            seed=seed_from(
                seed_namespace, run_id, phase_id, "quality", quality_view, quality_metric
            ),
        ),
    }
    endpoint_results.update(
        {
            endpoint: _factorial_for(
                blocks,
                endpoint=endpoint,
                quality_view=quality_view,
                quality_metric=quality_metric,
                direction=directions[endpoint],
                n_boot=n_boot,
                seed=seed_from(seed_namespace, run_id, phase_id, endpoint),
            )
            for endpoint in operational_endpoints
        }
    )
    quality_guards = {
        view: {
            metric: _factorial_for(
                blocks,
                endpoint="quality",
                quality_view=view,
                quality_metric=metric,
                direction=direction,
                n_boot=n_boot,
                seed=seed_from(seed_namespace, run_id, phase_id, "quality", view, metric),
            )
            for metric, direction in QUALITY_GUARD_DIRECTIONS.items()
        }
        for view in _QUALITY_VIEWS
    }
    trajectory_results = {
        endpoint: _factorial_for(
            blocks,
            endpoint=endpoint,
            quality_view=quality_view,
            quality_metric=quality_metric,
            direction=direction,
            n_boot=n_boot,
            seed=seed_from(seed_namespace, run_id, phase_id, "trajectory", endpoint),
        )
        for endpoint, direction in TRAJECTORY_ENDPOINT_DIRECTIONS.items()
    }
    task_joint_outcomes = _task_level_joint_outcomes(
        blocks,
        policy=task_level_joint_outcomes_policy,
        n_boot=n_boot,
        seed_namespace=f"{seed_namespace}:{run_id}:{phase_id}",
    )
    paired_invalid = [
        {
            "block_id": block["block_id"],
            "task_id": block["task_id"],
            "replicate_id": block["replicate_id"],
            "reason": block["invalid_reason"],
            "engine_epochs": block["engine_epochs"],
        }
        for block in blocks
        if not block["valid_for_paired_estimate"]
    ]
    body = {
        "schema_version": "e2e_factorial_effects_v2",
        "run_id": run_id,
        "phase_id": phase_id,
        "evaluation_scope_sha256": prepared["scope_sha"],
        "provenance": prepared["provenance"],
        "all_offered_blocks": len(blocks),
        "tasks": len({block["task_id"] for block in blocks}),
        "clusters": len({block["cluster_id"] for block in blocks}),
        "arm_map": prepared["arm_map"],
        "expected_variant_ids": prepared["expected_variant_ids"],
        "variant_validation": (
            "VERIFIED" if prepared["expected_variant_ids"] is not None else "NOT_REQUESTED"
        ),
        "quality_view": quality_view,
        "quality_metric": quality_metric,
        "task_feature_registry_sha256": prepared["feature_registry_sha"],
        "structural_accounting": {
            "paired_valid_blocks": len(blocks) - len(paired_invalid),
            "paired_invalid_blocks": paired_invalid,
            "all_offered_retained": True,
        },
        "factorial_endpoints": endpoint_results,
        "trajectory_endpoints": trajectory_results,
        "quality_guards": quality_guards,
        "task_level_joint_outcomes": task_joint_outcomes,
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def build_eligibility_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    task_features: Mapping[str, Mapping[str, Any]],
    eligibility_spec: Mapping[str, Any],
    scope_receipt: Mapping[str, Any] | None = None,
    arm_map: Mapping[str, str] | None = None,
    expected_variant_ids: Mapping[str, str] | None = None,
    freeze_root: Mapping[str, Any] | None = None,
    schedule_manifest: Mapping[str, Any] | None = None,
    seed_namespace: str = "e2e-effects-v2",
) -> dict:
    """Build only the training/heldout eligibility bridge.

    Measurement absence and paired-invalid blocks are returned as an auditable status.  Hash or
    scope tampering still raises because it is not a statistical non-estimability condition.
    """
    verified_spec = _verify_eligibility_spec(eligibility_spec)
    prepared = _prepare_blocks(
        records,
        task_features=task_features,
        scope_receipt=scope_receipt,
        arm_map=arm_map,
        expected_variant_ids=expected_variant_ids,
        freeze_root=freeze_root,
        schedule_manifest=schedule_manifest,
    )
    spec_registry = str(eligibility_spec.get("task_feature_registry_sha256") or "")
    if spec_registry != prepared["feature_registry_sha"]:
        raise ValueError("eligibility spec does not bind the exact frozen task feature registry")

    def family_not_estimable(reason: str, **details: Any) -> dict[str, Any]:
        return {
            "status": "NOT_ESTIMABLE",
            "schema_version": "e2e_eligibility_result_v3",
            "analysis_scope": _ELIGIBILITY_SCOPE,
            "spec_content_sha256": verified_spec["content_sha256"],
            "fit_scope": "TRAINING_ONLY",
            "reason": reason,
            "claim_policy": {
                "primary_verdict_effect": "NONE",
                "deployment_or_invocation_envelope_allowed": False,
                "confirmatory_claim_allowed": False,
                "all_predeclared_targets_reported": True,
            },
            "target_results": {
                str(target["target_id"]): {
                    "target_id": str(target["target_id"]),
                    "status": "NOT_ESTIMABLE",
                    "analysis_scope": _ELIGIBILITY_SCOPE,
                    "treatment_semantic": str(target["treatment_semantic"]),
                    "treatment_arm_id": str(target["treatment_arm_id"]),
                    "comparator_semantics": list(
                        map(str, target["comparator_semantics"])
                    ),
                    "comparator_arm_ids": list(
                        map(str, target["comparator_arm_ids"])
                    ),
                    "comparator_logic": "ALL",
                    "reason": reason,
                }
                for target in verified_spec["_targets"]
            },
            **details,
        }

    invalid = [block for block in prepared["blocks"] if not block["valid_for_paired_estimate"]]
    if invalid:
        eligibility = family_not_estimable(
            "ALL_OFFERED_CONTAINS_PAIRED_INVALID_BLOCK",
            paired_invalid_blocks=[
                {
                    "block_id": block["block_id"],
                    "reason": block["invalid_reason"],
                }
                for block in invalid
            ],
        )
    else:
        try:
            eligibility = {
                "status": "OK",
                **_eligibility(
                    prepared["blocks"],
                    eligibility_spec,
                    arm_map=prepared["arm_map"],
                    seed_namespace=seed_namespace,
                ),
            }
        except ValueError as exc:
            eligibility = family_not_estimable(str(exc))
    body = {
        "schema_version": "e2e_eligibility_effects_v3",
        "run_id": prepared["run_id"],
        "phase_id": prepared["phase_id"],
        "evaluation_scope_sha256": prepared["scope_sha"],
        "task_feature_registry_sha256": prepared["feature_registry_sha"],
        "arm_map": prepared["arm_map"],
        "expected_variant_ids": prepared["expected_variant_ids"],
        "eligibility": eligibility,
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def build_e2e_effects(
    records: Sequence[Mapping[str, Any]],
    *,
    task_features: Mapping[str, Mapping[str, Any]],
    eligibility_spec: Mapping[str, Any] | None = None,
    scope_receipt: Mapping[str, Any] | None = None,
    arm_map: Mapping[str, str] | None = None,
    expected_variant_ids: Mapping[str, str] | None = None,
    freeze_root: Mapping[str, Any] | None = None,
    schedule_manifest: Mapping[str, Any] | None = None,
    quality_view: str = "fallback_assisted",
    quality_metric: str = "weighted_required_atom_recall",
    task_level_joint_outcomes_policy: Mapping[str, Any] | None = None,
    n_boot: int = 2000,
    seed_namespace: str = "e2e-effects-v2",
) -> dict:
    """Build independently valid factorial and optional eligibility artifacts."""
    factorial = build_factorial_effects(
        records,
        task_features=task_features,
        scope_receipt=scope_receipt,
        arm_map=arm_map,
        expected_variant_ids=expected_variant_ids,
        freeze_root=freeze_root,
        schedule_manifest=schedule_manifest,
        quality_view=quality_view,
        quality_metric=quality_metric,
        task_level_joint_outcomes_policy=task_level_joint_outcomes_policy,
        n_boot=n_boot,
        seed_namespace=seed_namespace,
    )
    if eligibility_spec is None:
        eligibility = {
            "status": "NOT_REQUESTED",
            "reason": "NO_FROZEN_ELIGIBILITY_SPEC",
        }
    else:
        built = build_eligibility_effects(
            records,
            task_features=task_features,
            eligibility_spec=eligibility_spec,
            scope_receipt=scope_receipt,
            arm_map=arm_map,
            expected_variant_ids=expected_variant_ids,
            freeze_root=freeze_root,
            schedule_manifest=schedule_manifest,
            seed_namespace=seed_namespace,
        )
        eligibility = built["eligibility"]
    body = {
        key: value
        for key, value in factorial.items()
        if key not in {"schema_version", "content_sha256"}
    }
    body["schema_version"] = "e2e_effects_v2"
    body["eligibility"] = eligibility
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body
