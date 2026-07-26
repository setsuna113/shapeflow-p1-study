"""Production ITT estimands from frozen evaluation artifacts.

This is the missing bridge between the runner/evaluator ledgers and a verdict.  It refuses a
directory of hand-written summaries: inputs must be the block-scoped score records emitted by
``campaign.evaluate`` for one exact run and phase.  Every assigned arm stays in the accounting
table.  Judge-missing outcomes are shown as observed-missing plus explicit worst/best bounds;
terminal failures and strict-P1 fallbacks receive their pre-registered adverse outcome.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..scoped_paths import resolve_scoped_path
from .bootstrap import cluster_bootstrap_ci, paired_log_ratio_saving, seed_from

__all__ = [
    "load_scoped_scores",
    "build_verdict_inputs",
    "write_verdict_inputs",
]

_EVALUATION_SCOPE_FILENAME = "EVALUATION_SCOPE.json"


class _ScopedScoreRecords(list):
    """A list carrying the verified all-offered receipt that made it complete."""

    def __init__(self, records: Sequence[dict], scope_receipt: dict) -> None:
        super().__init__(records)
        self.scope_receipt = scope_receipt


_CORE_QUALITY_METRICS = (
    "weighted_required_atom_recall",
    "grounded_claim_precision",
    "citation_correctness",
    "citation_association",
    "required_facet_coverage",
    "qualified_report",
    "critical_harm",
)
_CONDITIONAL_QUALITY_METRICS = (
    "grounded_negative_recall",
    "unresolved_gap_reporting_recall",
)
_QUALITY_METRICS = _CORE_QUALITY_METRICS + _CONDITIONAL_QUALITY_METRICS

_NODE_DIRECT_FIELDS = (
    "candidate_coverage",
    "selector_conditional_recall",
    "weighted_evidence_recall",
    "total_published_prechunk_recall",
    "critical_truth_recall",
    "contradiction_pair_recall",
    "grounded_negative_atom_recall",
    "negative_query_trace_recall",
    "unresolved_gap_recall",
    "negative_gap_recall",
    "typed_relation_coverage",
    "contradiction_role_pair_recall",
)
_DIRECT_METRICS = {
    f"{node.lower()}_{field}": (node, field) for node in ("H", "C") for field in _NODE_DIRECT_FIELDS
}
_SELECTOR_EFFICIENCY_FIELDS = (
    "selected_token_precision",
    "weighted_truth_per_100_rendered_tokens",
    "materialization_ratio",
)


def load_scoped_scores(judgments_root: Path, *, run_id: str, phase_id: str) -> list[dict]:
    """Load exactly the score set sealed by the evaluator's all-offered receipt."""
    directory = resolve_scoped_path(judgments_root, run_id=run_id, phase_id=phase_id)
    if not directory.is_dir():
        raise ValueError(f"scoped judgment directory does not exist: {directory}")
    receipt_path = resolve_scoped_path(
        judgments_root,
        run_id=run_id,
        phase_id=phase_id,
        tail=(_EVALUATION_SCOPE_FILENAME,),
    )
    if not receipt_path.is_file():
        raise ValueError(f"all-offered evaluation receipt is missing: {receipt_path}")
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid evaluation scope receipt {receipt_path}: {exc}") from exc
    if receipt.get("schema_version") != "evaluated_itt_scope_v1":
        raise ValueError(f"{receipt_path} has unsupported scope schema")
    receipt_sha = str(receipt.get("evaluation_scope_sha256") or "")
    actual_receipt_sha = sha256_hex(
        canonical_json(
            {key: value for key, value in receipt.items() if key != "evaluation_scope_sha256"}
        )
    )
    if receipt_sha != actual_receipt_sha:
        raise ValueError(
            f"{receipt_path} was edited: records {receipt_sha!r}, "
            f"hashes to {actual_receipt_sha}"
        )
    if receipt.get("run_id") != run_id or receipt.get("phase_id") != phase_id:
        raise ValueError(f"{receipt_path} belongs to another run/phase")
    if (
        len(str(receipt.get("execution_binding_sha256") or "")) != 64
        or len(str(receipt.get("protocol_document_sha256") or "")) != 64
        or len(str(receipt.get("schedule_sha256") or "")) != 64
        or len(str(receipt.get("freeze_root_sha256") or "")) != 64
        or len(str(receipt.get("analysis_design_receipt_sha256") or "")) != 64
        or len(str(receipt.get("task_feature_registry_sha256") or "")) != 64
        or len(str(receipt.get("eligibility_spec_content_sha256") or "")) != 64
    ):
        raise ValueError(
            f"{receipt_path} lacks execution/protocol/schedule/frozen-root/design provenance"
        )
    expected_rows = receipt.get("scores")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise ValueError(f"{receipt_path} has no offered score records")
    expected_by_id: dict[str, dict] = {}
    for expected in expected_rows:
        block_id = str((expected or {}).get("block_id") or "")
        if not block_id or block_id in expected_by_id:
            raise ValueError(f"{receipt_path} has duplicate/unnamed block {block_id!r}")
        expected_by_id[block_id] = expected
    if int(receipt.get("all_offered_blocks") or -1) != len(expected_by_id):
        raise ValueError(f"{receipt_path} all_offered_blocks does not match its score index")

    primary_files = {
        path.name
        for path in directory.glob("*.json")
        if path.name != _EVALUATION_SCOPE_FILENAME and not path.name.endswith(".judge.json")
    }
    expected_files = {f"{block_id}.json" for block_id in expected_by_id}
    if primary_files != expected_files:
        raise ValueError(
            "scoped judgment directory differs from all-offered receipt: "
            f"missing={sorted(expected_files - primary_files)}, "
            f"extra={sorted(primary_files - expected_files)}"
        )
    records: list[dict] = []
    seen: set[str] = set()
    for block_id in sorted(expected_by_id):
        path = resolve_scoped_path(
            judgments_root,
            run_id=run_id,
            phase_id=phase_id,
            tail=(f"{block_id}.json",),
        )
        body = json.loads(path.read_text(encoding="utf-8"))
        if body.get("run_id") != run_id or body.get("phase_id") != phase_id:
            raise ValueError(f"{path} belongs to another run/phase")
        observed_block_id = str(body.get("block_id") or "")
        if observed_block_id != block_id or block_id in seen:
            raise ValueError(f"duplicate or missing block_id in {path}")
        recorded = str(body.get("content_sha256") or "")
        unsigned = {k: v for k, v in body.items() if k != "content_sha256"}
        actual = sha256_hex(canonical_json(unsigned))
        if recorded != actual:
            raise ValueError(f"{path} was edited: records {recorded}, hashes to {actual}")
        expected = expected_by_id[block_id]
        if (
            str(expected.get("execution_binding_sha256") or "")
            != str(receipt["execution_binding_sha256"])
            or str(expected.get("protocol_document_sha256") or "")
            != str(receipt["protocol_document_sha256"])
            or str(body.get("execution_binding_sha256") or "")
            != str(receipt["execution_binding_sha256"])
            or str(body.get("protocol_document_sha256") or "")
            != str(receipt["protocol_document_sha256"])
        ):
            raise ValueError(f"{path} execution/protocol identity differs from evaluation scope")
        if recorded != str(expected.get("score_content_sha256") or ""):
            raise ValueError(f"{path} content hash differs from all-offered receipt")
        if str(body.get("task_id") or "") != str(expected.get("task_id") or "") or str(
            body.get("replicate_id") or ""
        ) != str(expected.get("replicate_id") or ""):
            raise ValueError(f"{path} coordinates differ from all-offered receipt")
        per_arm = body.get("per_arm")
        if not isinstance(per_arm, dict) or not per_arm:
            raise ValueError(f"{path} has no offered arms")
        expected_scope = {
            "freeze_root_sha256": str(receipt["freeze_root_sha256"]),
            "schedule_sha256": str(receipt["schedule_sha256"]),
            "block_freeze_sha256": str(expected.get("block_freeze_sha256") or ""),
            "block_digest": str(expected.get("block_digest") or ""),
            "valid_for_paired_estimate": expected.get("valid_for_paired_estimate"),
            "invalid_reason": str(expected.get("invalid_reason") or ""),
            "engine_epochs": list(map(str, expected.get("engine_epochs") or ())),
            "engine_epoch_by_arm": dict(
                sorted(
                    (str(key), str(value))
                    for key, value in (expected.get("engine_epoch_by_arm") or {}).items()
                )
            ),
        }
        if (
            len(expected_scope["block_freeze_sha256"]) != 64
            or len(expected_scope["block_digest"]) != 64
            or not isinstance(expected_scope["valid_for_paired_estimate"], bool)
            or not expected_scope["engine_epochs"]
            or len(expected_scope["engine_epoch_by_arm"]) != len(per_arm)
        ):
            raise ValueError(f"{receipt_path} has incomplete block provenance for {block_id}")
        for arm_key, row in per_arm.items():
            if not isinstance(row, dict) or row.get("frozen_scope") != expected_scope:
                raise ValueError(
                    f"{path} arm {arm_key!r} is not bound to the all-offered frozen scope"
                )
        seen.add(block_id)
        records.append(body)
    return _ScopedScoreRecords(records, receipt)


def _arm_id(key: str) -> str:
    return key.rsplit(":", 1)[0]


def _view(row: dict, mode: str) -> dict | None:
    views = row.get("quality_views") or {}
    value = views.get(mode)
    return value if isinstance(value, dict) else None


def _work(row: dict) -> float | None:
    summary = row.get("work_summary")
    if not isinstance(summary, Mapping):
        return None
    if summary.get("telemetry_complete") is not True or summary.get("overlap_valid") is not True:
        return None
    for key in ("service_seconds", "total_service_seconds", "complete_service_seconds"):
        value = summary.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                return None
            return value if np.isfinite(value) and value > 0 else None
    return None


def _direct_value(row: dict, metric: str) -> float | None:
    direct = row.get("direct_node_metrics") or {}
    if metric not in _DIRECT_METRICS:
        raise ValueError(f"unknown node-scoped direct metric {metric!r}")
    node, field = _DIRECT_METRICS[metric]
    node_metrics = (direct.get("by_node") or {}).get(node)
    if not isinstance(node_metrics, dict):
        return None
    value = node_metrics.get(field)
    return float(value) if value is not None else None


def _applicable_direct_nodes(row: dict) -> list[dict]:
    """Return only node-scoped measurements that the assigned arm actually exercised.

    The flat direct status predates joint H+C arms.  Reading it for a joint arm can let a healthy
    H trace hide a missing C trace (or vice versa), so all structural accounting below uses the
    independent node records whenever they exist.
    """
    direct = row.get("direct_node_metrics") or {}
    by_node = direct.get("by_node") or {}
    return [
        value
        for value in by_node.values()
        if isinstance(value, dict) and bool(value.get("applicable"))
    ]


def _ci_obj(ci) -> dict:
    def finite(value):
        return float(value) if np.isfinite(value) else None

    return {
        "point": finite(ci.point),
        "lower": finite(ci.lower),
        "upper": finite(ci.upper),
        "n_clusters": ci.n_clusters,
        "n_boot": ci.n_boot,
    }


def _difference_ci(
    values: Sequence[float], clusters: Sequence[str], *, seed: int, n_boot: int
) -> dict | None:
    if len(set(clusters)) < 2:
        return None
    ci = cluster_bootstrap_ci(values, clusters, np.mean, seed=seed, n_boot=n_boot, side="lower")
    return {
        **_ci_obj(ci),
        "interval_type": "one_sided_95_percent_lower_confidence_bound",
        "bound_direction": "lower",
        "effect_orientation": "positive_favors_treatment",
        "decision_direction": "lower_bound_must_exceed_preregistered_noninferiority_margin",
    }


def _task_means(
    values: Sequence[float],
    task_ids: Sequence[str],
    cluster_ids: Sequence[str],
) -> tuple[list[float], list[str]]:
    """Give each task one vote before resampling its source/topic cluster.

    Replicates are repeated measurements of a task, not extra independent tasks.  Reducing them
    here prevents an accidentally over-replicated task from receiving extra weight in either the
    point estimate or the cluster bootstrap.
    """
    if not (len(values) == len(task_ids) == len(cluster_ids)):
        raise ValueError("values, task ids and cluster ids must align")
    grouped: dict[str, list[float]] = {}
    task_cluster: dict[str, str] = {}
    for value, task_id, cluster_id in zip(values, task_ids, cluster_ids, strict=True):
        previous = task_cluster.setdefault(str(task_id), str(cluster_id))
        if previous != str(cluster_id):
            raise ValueError(f"task {task_id!r} appears in multiple source clusters")
        grouped.setdefault(str(task_id), []).append(float(value))
    ordered = sorted(grouped)
    return (
        [float(np.mean(grouped[task_id])) for task_id in ordered],
        [task_cluster[task_id] for task_id in ordered],
    )


def _selector_efficiency_summary(
    observations: Sequence[Mapping],
    *,
    node: str,
    seed_namespace: str,
    arm: str,
    n_boot: int,
) -> dict:
    """Aggregate exact token-materialization traces without making them primary utility.

    These are absolute selector-efficiency diagnostics, not P1-vs-P0 causal effects.  Every
    applicable offered row must have a complete trace; otherwise the selector-efficiency
    sub-conclusion is explicitly not established while the row remains in the primary ITT.
    """
    applicable: list[tuple[Mapping, Mapping]] = []
    for observation in observations:
        metrics = ((observation["p1"].get("direct_node_metrics") or {}).get("by_node") or {}).get(
            node
        ) or {}
        if bool(metrics.get("applicable")):
            applicable.append((observation, metrics))
    if not applicable:
        return {
            "status": "NOT_APPLICABLE",
            "node": node,
            "all_offered_applicable_rows": 0,
            "token_trace_complete_rows": 0,
            "decision_use": "SECONDARY_DESCRIPTIVE_NOT_PRIMARY_CAUSAL_UTILITY",
        }

    complete_rows = sum(metrics.get("token_trace_complete") is True for _, metrics in applicable)
    malformed_rows = 0
    for _, metrics in applicable:
        for field in (
            *_SELECTOR_EFFICIENCY_FIELDS,
            "published_rendered_tokens",
            "offered_evidence_tokens",
        ):
            value = metrics.get(field)
            if (
                isinstance(value, bool)
                or not isinstance(value, int | float)
                or not np.isfinite(float(value))
                or float(value) < 0
            ):
                malformed_rows += 1
                break
    common = {
        "node": node,
        "all_offered_applicable_rows": len(applicable),
        "token_trace_complete_rows": complete_rows,
        "token_trace_incomplete_rows": len(applicable) - complete_rows,
        "malformed_metric_rows": malformed_rows,
        "decision_use": "SECONDARY_DESCRIPTIVE_NOT_PRIMARY_CAUSAL_UTILITY",
    }
    if complete_rows != len(applicable) or malformed_rows:
        return {
            **common,
            "status": "NOT_ESTABLISHED_INCOMPLETE_TOKEN_TRACE",
            "metrics": {
                field: {"status": "NOT_ESTABLISHED_INCOMPLETE_TOKEN_TRACE"}
                for field in _SELECTOR_EFFICIENCY_FIELDS
            },
            "published_rendered_tokens": None,
            "offered_evidence_tokens": None,
        }

    metric_results: dict[str, dict] = {}
    for field in _SELECTOR_EFFICIENCY_FIELDS:
        values = [float(metrics[field]) for _, metrics in applicable]
        tasks = [str(observation["task_id"]) for observation, _ in applicable]
        clusters = [str(observation["cluster_id"]) for observation, _ in applicable]
        task_values, task_clusters = _task_means(values, tasks, clusters)
        if len(set(task_clusters)) < 2:
            metric_results[field] = {
                "status": "NOT_ESTABLISHED_FEWER_THAN_TWO_SOURCE_CLUSTERS",
                "tasks": len(task_values),
            }
            continue
        ci = cluster_bootstrap_ci(
            task_values,
            task_clusters,
            np.mean,
            seed=seed_from(seed_namespace, arm, node, field),
            n_boot=n_boot,
            side="two",
        )
        metric_results[field] = {
            "status": "ESTIMABLE",
            "ci": {
                **_ci_obj(ci),
                "interval_type": "two_sided_95_percent_cluster_bootstrap",
                "replicate_reduction": "task_mean_before_source_topic_cluster_bootstrap",
            },
            "tasks": len(task_values),
        }
    status = (
        "ESTIMABLE"
        if all(value["status"] == "ESTIMABLE" for value in metric_results.values())
        else "NOT_ESTABLISHED"
    )
    return {
        **common,
        "status": status,
        "metrics": metric_results,
        "published_rendered_tokens": {
            "status": "OBSERVED_ROW_TOTAL",
            "total": int(
                sum(float(metrics["published_rendered_tokens"]) for _, metrics in applicable)
            ),
        },
        "offered_evidence_tokens": {
            "status": "OBSERVED_ROW_TOTAL",
            "total": int(
                sum(float(metrics["offered_evidence_tokens"]) for _, metrics in applicable)
            ),
        },
    }


def _absolute_critical_harm_summary(
    observations: Sequence[Mapping],
    *,
    seed_namespace: str,
    arm: str,
    n_boot: int,
) -> dict:
    """Absolute P1 harm rates, kept separate from paired treatment effects."""

    def summarize(
        values: Sequence[float],
        tasks: Sequence[str],
        clusters: Sequence[str],
        *,
        label: str,
        missing_pairs: int,
    ) -> dict:
        task_values, task_clusters = _task_means(values, tasks, clusters)
        common = {
            "n_pairs": len(values),
            "n_tasks": len(task_values),
            "missing_pairs": missing_pairs,
            "denominator_definition": "all_offered_assigned_treatment_pairs",
        }
        if missing_pairs:
            return {
                **common,
                "status": "NOT_ESTIMABLE_MISSING_STRICT_HARM_MEASUREMENT",
                "ci": None,
            }
        if len(set(task_clusters)) < 2:
            return {
                **common,
                "status": "NOT_ESTIMABLE_FEWER_THAN_TWO_SOURCE_CLUSTERS",
                "ci": None,
            }
        ci = cluster_bootstrap_ci(
            task_values,
            task_clusters,
            np.mean,
            seed=seed_from(seed_namespace, arm, "absolute_critical_harm", label),
            n_boot=n_boot,
            side="two",
        )
        return {
            **common,
            "status": "ESTIMABLE",
            "ci": {
                **_ci_obj(ci),
                "interval_type": "two_sided_95_percent_source_topic_cluster_bootstrap",
                "effect_orientation": "absolute_treatment_critical_harm_rate",
                "replicate_reduction": "task_mean_before_source_topic_cluster_bootstrap",
            },
        }

    strict_values: list[float] = []
    strict_tasks: list[str] = []
    strict_clusters: list[str] = []
    strict_missing = 0
    adverse_values: list[float] = []
    adverse_tasks: list[str] = []
    adverse_clusters: list[str] = []
    for observation in observations:
        strict = _view(dict(observation["p1"]), "strict")
        value = strict.get("critical_harm") if isinstance(strict, Mapping) else None
        if value is None:
            strict_missing += 1
        else:
            strict_values.append(float(value))
            strict_tasks.append(str(observation["task_id"]))
            strict_clusters.append(str(observation["cluster_id"]))
        worst = _view(dict(observation["p1"]), "worst_case")
        adverse = worst.get("critical_harm") if isinstance(worst, Mapping) else None
        # Keep every assignment. If even the evaluator's worst-case view is unavailable, the
        # adverse bound is one rather than deleting the pair.
        adverse_values.append(float(adverse) if adverse is not None else 1.0)
        adverse_tasks.append(str(observation["task_id"]))
        adverse_clusters.append(str(observation["cluster_id"]))

    return {
        "strict_observed": summarize(
            strict_values,
            strict_tasks,
            strict_clusters,
            label="strict_observed",
            missing_pairs=strict_missing,
        ),
        "all_offered_adverse": {
            **summarize(
                adverse_values,
                adverse_tasks,
                adverse_clusters,
                label="all_offered_adverse",
                missing_pairs=0,
            ),
            "missing_strict_pairs_assigned_adverse": strict_missing,
            "definition": ("P1 worst_case critical_harm; unavailable worst_case is assigned 1.0"),
        },
    }


def _task_mean_pairs(
    left: Sequence[float],
    right: Sequence[float],
    task_ids: Sequence[str],
    cluster_ids: Sequence[str],
) -> tuple[list[float], list[float], list[str]]:
    if not (len(left) == len(right) == len(task_ids) == len(cluster_ids)):
        raise ValueError("paired values, task ids and cluster ids must align")
    left_by_task: dict[str, list[float]] = {}
    right_by_task: dict[str, list[float]] = {}
    task_cluster: dict[str, str] = {}
    for a, b, task_id, cluster_id in zip(left, right, task_ids, cluster_ids, strict=True):
        task_id = str(task_id)
        cluster_id = str(cluster_id)
        previous = task_cluster.setdefault(task_id, cluster_id)
        if previous != cluster_id:
            raise ValueError(f"task {task_id!r} appears in multiple source clusters")
        left_by_task.setdefault(task_id, []).append(float(a))
        right_by_task.setdefault(task_id, []).append(float(b))
    ordered = sorted(left_by_task)
    return (
        [float(np.mean(left_by_task[task_id])) for task_id in ordered],
        [float(np.mean(right_by_task[task_id])) for task_id in ordered],
        [task_cluster[task_id] for task_id in ordered],
    )


def _binary_indicator(row: Mapping, *, field: str) -> float | None:
    if field == "terminal_failure":
        state = row.get("assignment_state")
        if state == "COMMITTED":
            return 0.0
        if state in {"FAILED_FINAL", "FAILED_UNKNOWN", "BLOCKED_BUDGET"}:
            return 1.0
        return None
    if field == "fallback":
        value = row.get("fell_back")
        return float(value) if isinstance(value, bool) else None
    raise ValueError(f"unknown binary endpoint {field!r}")


def _paired_risk_difference(
    observations: Sequence[Mapping],
    *,
    field: str,
    seed: int,
    n_boot: int,
) -> dict:
    """Treatment-minus-P0 risk in percentage points, with all missingness exposed."""
    diffs: list[float] = []
    tasks: list[str] = []
    clusters: list[str] = []
    missing_blocks: list[str] = []
    for obs in observations:
        p0 = _binary_indicator(obs["p0"], field=field)
        p1 = _binary_indicator(obs["p1"], field=field)
        if p0 is None or p1 is None:
            missing_blocks.append(str(obs["block_id"]))
            continue
        diffs.append(p1 - p0)
        tasks.append(str(obs["task_id"]))
        clusters.append(str(obs["cluster_id"]))
    if missing_blocks:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "ASSIGNED_BINARY_OUTCOME_MISSING",
            "orientation": "treatment_minus_p0_percentage_points",
            "all_offered_pairs": len(observations),
            "observed_pairs": len(diffs),
            "missing_blocks": sorted(missing_blocks),
            "ci": None,
        }
    task_diffs, task_clusters = _task_means(diffs, tasks, clusters)
    if len(set(task_clusters)) < 2:
        return {
            "status": "NOT_ESTIMABLE",
            "reason": "FEWER_THAN_TWO_SOURCE_CLUSTERS",
            "orientation": "treatment_minus_p0_percentage_points",
            "all_offered_pairs": len(observations),
            "observed_pairs": len(diffs),
            "tasks": len(task_diffs),
            "ci": None,
        }
    ci = cluster_bootstrap_ci(
        task_diffs,
        task_clusters,
        np.mean,
        n_boot=n_boot,
        seed=seed,
        side="upper",
    )
    return {
        "status": "OK",
        "orientation": "treatment_minus_p0_percentage_points",
        "all_offered_pairs": len(observations),
        "observed_pairs": len(diffs),
        "tasks": len(task_diffs),
        "ci": {
            **_ci_obj(ci),
            "point": float(ci.point) * 100.0,
            "lower": None,
            "upper": float(ci.upper) * 100.0,
            "unit": "percentage_points",
            "interval_type": "one_sided_95_percent_upper_confidence_bound",
            "bound_direction": "upper",
            "effect_orientation": "positive_is_increased_adverse_risk",
            "decision_direction": "upper_bound_must_not_exceed_preregistered_risk_margin",
        },
    }


def _micro_counts(row: Mapping, metric: str) -> tuple[float, float] | None:
    node = ((row.get("direct_node_metrics") or {}).get("by_node") or {}).get("C") or {}
    raw = (node.get("micro_counts") or {}).get(metric)
    if not isinstance(raw, Mapping):
        return None
    try:
        numerator = float(raw["numerator"])
        denominator = float(raw["denominator"])
    except (KeyError, TypeError, ValueError):
        return None
    if (
        not np.isfinite(numerator)
        or not np.isfinite(denominator)
        or numerator < 0
        or denominator <= 0
        or numerator > denominator
    ):
        return None
    return numerator, denominator


def _cluster_micro_ratio_ci(
    numerators: Sequence[float],
    denominators: Sequence[float],
    task_ids: Sequence[str],
    cluster_ids: Sequence[str],
    *,
    seed: int,
    n_boot: int,
) -> dict | None:
    """Pooled opportunity ratio with task-mean replicate reduction and cluster resampling."""
    if not (len(numerators) == len(denominators) == len(task_ids) == len(cluster_ids)):
        raise ValueError("micro counts and coordinates must align")
    by_task_num: dict[str, list[float]] = {}
    by_task_den: dict[str, list[float]] = {}
    task_cluster: dict[str, str] = {}
    for numerator, denominator, task_id, cluster_id in zip(
        numerators, denominators, task_ids, cluster_ids, strict=True
    ):
        task_id = str(task_id)
        cluster_id = str(cluster_id)
        previous = task_cluster.setdefault(task_id, cluster_id)
        if previous != cluster_id:
            raise ValueError(f"task {task_id!r} appears in multiple source clusters")
        by_task_num.setdefault(task_id, []).append(float(numerator))
        by_task_den.setdefault(task_id, []).append(float(denominator))
    cluster_totals: dict[str, list[float]] = {}
    for task_id in sorted(by_task_num):
        total = cluster_totals.setdefault(task_cluster[task_id], [0.0, 0.0])
        total[0] += float(np.mean(by_task_num[task_id]))
        total[1] += float(np.mean(by_task_den[task_id]))
    clusters = sorted(cluster_totals)
    if len(clusters) < 2:
        return None
    point_num = sum(cluster_totals[c][0] for c in clusters)
    point_den = sum(cluster_totals[c][1] for c in clusters)
    if point_den <= 0:
        return None
    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for index in range(n_boot):
        sampled = rng.integers(0, len(clusters), size=len(clusters))
        numerator = sum(cluster_totals[clusters[j]][0] for j in sampled)
        denominator = sum(cluster_totals[clusters[j]][1] for j in sampled)
        draws[index] = numerator / denominator
    return {
        "point": point_num / point_den,
        "lower": float(np.quantile(draws, 0.05)),
        "upper": None,
        "n_clusters": len(clusters),
        "n_boot": n_boot,
        "interval": "one_sided_95_percent_lower_confidence_bound",
        "replicate_reduction": "task_mean_counts_before_cluster_bootstrap",
        "weighting": "pooled_visible_truth_atom_opportunities",
    }


_SELECTOR_COUNT_FIELDS = (
    "selector_attempt_count",
    "strict_valid_count",
    "repaired_attempt_count",
    "rejected_attempt_count",
    "invalid_id_count",
    "schema_rejection_count",
    "semantic_conflict_attempt_count",
    "semantic_conflict_event_count",
    "failure_before_parse_count",
    "normalization_missing_count",
    "normalization_invalid_count",
    "normalization_trace_error_count",
    "component_failure_count",
)


def _selector_validity_counts(row: Mapping) -> dict:
    """Sum independent H/C selector summaries without treating an absent trace as clean."""
    nodes = _applicable_direct_nodes(dict(row))
    direct = row.get("direct_node_metrics") or {}
    totals = {field: 0 for field in _SELECTOR_COUNT_FIELDS}
    missing_summaries = 0
    adverse = bool(row.get("assignment_state") != "COMMITTED" or row.get("fell_back"))
    for node_metrics in nodes:
        summary = node_metrics.get("selector_normalization")
        if not isinstance(summary, Mapping):
            missing_summaries += 1
            adverse = True
            continue
        malformed = False
        for field in _SELECTOR_COUNT_FIELDS:
            value = summary.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                malformed = True
                continue
            totals[field] += value
        adverse = adverse or bool(summary.get("no_repair_adverse"))
        if malformed:
            missing_summaries += 1
            adverse = True
    if not nodes and str(direct.get("status") or "") not in {
        "NOT_APPLICABLE",
        "NOT_APPLICABLE_NO_SELECTOR",
    }:
        # A treatment row without any node-scoped selector audit cannot establish that the
        # model output was strict-valid. The other ITT layers may still explain why the node
        # is absent, but this sensitivity analysis must assign the adverse endpoint.
        missing_summaries += 1
        adverse = True
    totals["normalization_trace_error_count"] += missing_summaries
    return {
        **totals,
        "applicable_node_count": len(nodes),
        "missing_node_summary_count": missing_summaries,
        "no_repair_adverse": adverse,
    }


def _selector_rate_estimand(
    observations: Sequence[Mapping],
    *,
    numerator_field: str,
    side: str,
    seed: int,
    n_boot: int,
) -> dict:
    """Task-equal selector rate after pooling attempts within each task."""
    by_task_num: dict[str, float] = {}
    by_task_den: dict[str, float] = {}
    task_cluster: dict[str, str] = {}
    incomplete_rows = 0
    no_attempt_rows = 0
    for obs in observations:
        counts = _selector_validity_counts(obs["p1"])
        if counts["normalization_trace_error_count"] > 0:
            incomplete_rows += 1
        attempts = int(counts["selector_attempt_count"])
        if attempts == 0:
            no_attempt_rows += 1
            continue
        task_id = str(obs["task_id"])
        cluster_id = str(obs["cluster_id"])
        previous = task_cluster.setdefault(task_id, cluster_id)
        if previous != cluster_id:
            raise ValueError(f"task {task_id!r} appears in multiple source clusters")
        by_task_num[task_id] = by_task_num.get(task_id, 0.0) + float(counts[numerator_field])
        by_task_den[task_id] = by_task_den.get(task_id, 0.0) + float(attempts)
    task_ids = sorted(by_task_den)
    values = [by_task_num[task] / by_task_den[task] for task in task_ids]
    clusters = [task_cluster[task] for task in task_ids]
    if incomplete_rows:
        return {
            "status": "NOT_ESTIMABLE_INCOMPLETE_NORMALIZATION_TRACE",
            "ci": None,
            "tasks": len(task_ids),
            "attempts": int(sum(by_task_den.values())),
            "incomplete_rows": incomplete_rows,
            "no_attempt_rows": no_attempt_rows,
        }
    if len(set(clusters)) < 2:
        return {
            "status": "NOT_ESTIMABLE_FEWER_THAN_TWO_SOURCE_CLUSTERS",
            "ci": None,
            "tasks": len(task_ids),
            "attempts": int(sum(by_task_den.values())),
            "incomplete_rows": 0,
            "no_attempt_rows": no_attempt_rows,
        }
    ci = cluster_bootstrap_ci(
        values,
        clusters,
        np.mean,
        n_boot=n_boot,
        seed=seed,
        side=side,
    )
    direction = "lower" if side == "lower" else "upper"
    return {
        "status": "ESTIMABLE",
        "ci": {
            **_ci_obj(ci),
            "interval_type": f"one_sided_95_percent_{direction}_confidence_bound",
            "bound_direction": direction,
            "replicate_reduction": (
                "pool_attempt_counts_within_task_then_cluster_bootstrap_task_rates"
            ),
        },
        "tasks": len(task_ids),
        "attempts": int(sum(by_task_den.values())),
        "incomplete_rows": 0,
        "no_attempt_rows": no_attempt_rows,
    }


def _quality_applicable(row: Mapping, metric: str) -> bool:
    if metric in _CORE_QUALITY_METRICS:
        return True
    return any(
        isinstance(view, Mapping) and view.get(metric) is not None
        for view in (row.get("quality_views") or {}).values()
    )


def _no_repair_quality_value(
    row: Mapping, metric: str, *, treatment: bool, force_adverse: bool
) -> float | None:
    """Keep the pair, assigning the decision-adverse endpoint when it is unknowable/dirty."""
    if not _quality_applicable(row, metric):
        return None
    strict = _view(dict(row), "strict")
    if not treatment and isinstance(strict, Mapping) and strict.get(metric) is not None:
        return float(strict[metric])
    if (
        treatment
        and not force_adverse
        and isinstance(strict, Mapping)
        and strict.get(metric) is not None
    ):
        return float(strict[metric])
    # P0 judge-missing is favourable to the comparator; P1 judge-missing or any P1
    # repair/reject/failure is adverse to treatment. This bounds the paired effect rather than
    # deleting the offered row.
    if metric == "critical_harm":
        return 1.0 if treatment else 0.0
    return 0.0 if treatment else 1.0


def build_verdict_inputs(
    records: Sequence[dict],
    *,
    cluster_by_task: Mapping[str, str],
    comparator_arm: str = "P0",
    n_boot: int = 2000,
    seed_namespace: str = "verdict-input-v1",
    scope_receipt: Mapping | None = None,
    human_audit_receipt: Mapping | None = None,
) -> dict:
    """Build all-offered paired effects and sensitivity bounds, one treatment arm at a time.

    This function does not invent a final KEEP/KILL when a required layer is absent.  It emits
    the sufficient, content-addressed evidence object consumed by the decision stage: exact
    denominators, failures/fallbacks, direct-trace integrity, work-saving CIs and quality-effect
    CIs under strict, fallback-assisted, worst-case and best-case policies.
    """
    if not records:
        raise ValueError("no frozen evaluation records")
    verified_scope = scope_receipt or getattr(records, "scope_receipt", None)
    if not isinstance(verified_scope, Mapping):
        raise ValueError(
            "all-offered evaluation scope receipt is required; a list of whichever score "
            "files exist cannot establish ITT completeness"
        )
    receipt_unsigned = {
        key: value for key, value in verified_scope.items() if key != "evaluation_scope_sha256"
    }
    recorded_scope_sha = str(verified_scope.get("evaluation_scope_sha256") or "")
    if recorded_scope_sha != sha256_hex(canonical_json(receipt_unsigned)):
        raise ValueError("evaluation scope receipt content hash does not verify")
    expected_score_rows = verified_scope.get("scores")
    if not isinstance(expected_score_rows, list) or not expected_score_rows:
        raise ValueError("evaluation scope receipt contains no offered scores")
    expected_scores: dict[str, dict] = {}
    for expected in expected_score_rows:
        block_id = str((expected or {}).get("block_id") or "")
        if not block_id or block_id in expected_scores:
            raise ValueError(f"evaluation scope has duplicate/unnamed block {block_id!r}")
        expected_scores[block_id] = expected
    if int(verified_scope.get("all_offered_blocks") or -1) != len(expected_scores):
        raise ValueError("evaluation scope all_offered_blocks is inconsistent")
    observed_ids = [str(record.get("block_id") or "") for record in records]
    if (
        len(observed_ids) != len(set(observed_ids))
        or set(observed_ids) != set(expected_scores)
        or len(records) != len(expected_scores)
    ):
        raise ValueError(
            "analysis records are not the complete all-offered score set: "
            f"missing={sorted(set(expected_scores) - set(observed_ids))}, "
            f"extra={sorted(set(observed_ids) - set(expected_scores))}"
        )
    for record in records:
        block_id = str(record["block_id"])
        recorded_score_sha = str(record.get("content_sha256") or "")
        actual_score_sha = sha256_hex(
            canonical_json({key: value for key, value in record.items() if key != "content_sha256"})
        )
        if recorded_score_sha != actual_score_sha or recorded_score_sha != str(
            expected_scores[block_id].get("score_content_sha256") or ""
        ):
            raise ValueError(f"block {block_id} score does not match evaluation scope")
    run_ids = {str(r.get("run_id") or "") for r in records}
    phase_ids = {str(r.get("phase_id") or "") for r in records}
    if len(run_ids) != 1 or "" in run_ids or len(phase_ids) != 1 or "" in phase_ids:
        raise ValueError("records must belong to one exact non-empty run and phase")
    if str(verified_scope.get("run_id") or "") != next(iter(run_ids)) or str(
        verified_scope.get("phase_id") or ""
    ) != next(iter(phase_ids)):
        raise ValueError("evaluation scope receipt belongs to another run/phase")
    execution_binding_sha = str(verified_scope.get("execution_binding_sha256") or "")
    protocol_document_sha = str(verified_scope.get("protocol_document_sha256") or "")
    schedule_sha = str(verified_scope.get("schedule_sha256") or "")
    freeze_root_sha = str(verified_scope.get("freeze_root_sha256") or "")
    if (
        len(execution_binding_sha) != 64
        or len(protocol_document_sha) != 64
        or len(schedule_sha) != 64
        or len(freeze_root_sha) != 64
        or len(str(verified_scope.get("analysis_design_receipt_sha256") or "")) != 64
        or len(str(verified_scope.get("task_feature_registry_sha256") or "")) != 64
        or len(str(verified_scope.get("eligibility_spec_content_sha256") or "")) != 64
    ):
        raise ValueError(
            "evaluation scope lacks execution/protocol/schedule/frozen-root/design provenance"
        )
    for record in records:
        block_id = str(record["block_id"])
        expected = expected_scores[block_id]
        if (
            str(expected.get("execution_binding_sha256") or "") != execution_binding_sha
            or str(expected.get("protocol_document_sha256") or "") != protocol_document_sha
            or str(record.get("execution_binding_sha256") or "") != execution_binding_sha
            or str(record.get("protocol_document_sha256") or "") != protocol_document_sha
        ):
            raise ValueError(
                f"block {block_id} execution/protocol identity differs from evaluation scope"
            )
    score_sha_by_block = {
        str(record["block_id"]): str(record["content_sha256"]) for record in records
    }
    truth_sha_by_task: dict[str, str] = {}
    for record in records:
        task_id = str(record.get("task_id") or "")
        truth_sha = str(record.get("truth_packet_sha256") or "")
        if truth_sha:
            previous = truth_sha_by_task.setdefault(task_id, truth_sha)
            if previous != truth_sha:
                raise ValueError(f"task {task_id!r} was scored against multiple truth packets")

    truth_audited = False
    human_audit_receipt_sha: str | None = None
    if human_audit_receipt is not None:
        from ..evaluation.human_audit_workflow import (
            validate_human_audit_receipt,
        )

        if len(truth_sha_by_task) != len(
            {str(record.get("task_id") or "") for record in records}
        ) or any(len(value) != 64 for value in truth_sha_by_task.values()):
            raise ValueError("analysis rows do not bind one truth packet per task")
        verified_audit = validate_human_audit_receipt(
            human_audit_receipt,
            run_id=next(iter(run_ids)),
            phase_id=next(iter(phase_ids)),
            evaluation_scope_sha256=recorded_scope_sha,
            execution_binding_sha256=execution_binding_sha,
            protocol_document_sha256=protocol_document_sha,
            truth_packet_sha256_by_task=truth_sha_by_task,
            score_content_sha256_by_block=score_sha_by_block,
        )
        human_audit_receipt_sha = str(verified_audit["content_sha256"])
        truth_audited = True
    judge_policies = {str(r.get("judge_policy_sha256") or "") for r in records}
    if len(judge_policies) != 1:
        raise ValueError("score records do not share one uniform judge_policy_sha256")
    judge_policy_sha = next(iter(judge_policies))
    try:
        valid_judge_policy_sha = (
            len(judge_policy_sha) == 64
            and judge_policy_sha == judge_policy_sha.lower()
            and int(judge_policy_sha, 16) >= 0
        )
    except ValueError:
        valid_judge_policy_sha = False
    if not valid_judge_policy_sha:
        raise ValueError(
            "score records require one non-empty lowercase SHA-256 judge_policy_sha256"
        )
    claim_scopes = {str(r.get("claim_scope") or "") for r in records}
    truth_verifier_statuses = {str(r.get("truth_verifier_status") or "") for r in records}
    judge_signatures = {
        canonical_json(
            {
                "requested_model": provenance.get("requested_model"),
                "returned_models": provenance.get("returned_models") or [],
                "system_fingerprints": provenance.get("system_fingerprints") or [],
            }
        ).decode("utf-8")
        for r in records
        for provenance in [r.get("judge_provenance") or {}]
        if int(provenance.get("judgments") or 0) > 0
    }
    judge_drift = len(judge_signatures) > 1

    # arm -> list of paired block observations
    pairs: dict[str, list[dict]] = {}
    blocks_seen: set[str] = set()
    for record in records:
        block_id = str(record["block_id"])
        if block_id in blocks_seen:
            raise ValueError(f"duplicate block {block_id}")
        blocks_seen.add(block_id)
        task_id = str(record.get("task_id") or "")
        if task_id not in cluster_by_task:
            raise ValueError(f"task {task_id!r} has no frozen source cluster")
        per_arm = record.get("per_arm") or {}
        comparators = [(k, v) for k, v in per_arm.items() if _arm_id(k) == comparator_arm]
        if len(comparators) != 1:
            raise ValueError(
                f"block {block_id} needs exactly one {comparator_arm} assignment, "
                f"found {len(comparators)}"
            )
        _, p0 = comparators[0]
        for key, treatment in per_arm.items():
            arm = _arm_id(key)
            if arm == comparator_arm:
                continue
            pairs.setdefault(arm, []).append(
                {
                    "block_id": block_id,
                    "task_id": task_id,
                    "cluster_id": str(cluster_by_task[task_id]),
                    "p0": p0,
                    "p1": treatment,
                    "valid_for_paired_estimate": bool(
                        expected_scores[block_id].get("valid_for_paired_estimate")
                    ),
                    "paired_invalid_reason": str(
                        expected_scores[block_id].get("invalid_reason") or ""
                    ),
                }
            )

    arms: dict[str, dict] = {}
    for arm, observations in sorted(pairs.items()):
        clusters_all = [o["cluster_id"] for o in observations]
        state_counts: dict[str, int] = {}
        fallbacks = 0
        judge_unavailable = 0
        trace_unavailable = 0
        trace_invalid = 0
        selector_guard_unavailable = 0
        paired_invalid = 0
        paired_invalid_reasons: dict[str, int] = {}
        for obs in observations:
            row = obs["p1"]
            state = str(row.get("assignment_state") or "UNKNOWN")
            state_counts[state] = state_counts.get(state, 0) + 1
            fallbacks += int(bool(row.get("fell_back")))
            judge_unavailable += int(row.get("evaluation_status") == "JUDGE_UNAVAILABLE")
            direct = row.get("direct_node_metrics") or {}
            node_metrics = _applicable_direct_nodes(row)
            if node_metrics:
                node_statuses = {str(node.get("status") or "") for node in node_metrics}
                trace_invalid += int("INVALID_DIRECT_TRACE" in node_statuses)
                trace_unavailable += int(
                    any(status not in {"OK", "INVALID_DIRECT_TRACE"} for status in node_statuses)
                )
                selector_guard_unavailable += int(
                    any(not bool(node.get("selector_guards_complete")) for node in node_metrics)
                )
            else:
                # Legacy/no-trace rows have no node record to inspect.  Keep the flat field only
                # as a fail-closed absence signal; it is never used as an H or C measurement.
                trace_status = str(direct.get("status") or "")
                trace_unavailable += int(trace_status != "INVALID_DIRECT_TRACE")
                trace_invalid += int(trace_status == "INVALID_DIRECT_TRACE")
                selector_guard_unavailable += 1
            if not obs["valid_for_paired_estimate"]:
                paired_invalid += 1
                reason = obs["paired_invalid_reason"] or "UNSPECIFIED_PAIRED_INVALIDITY"
                paired_invalid_reasons[reason] = paired_invalid_reasons.get(reason, 0) + 1

        quality: dict[str, dict] = {}
        for metric in _QUALITY_METRICS:
            mode_effects: dict[str, dict | None] = {}
            for mode in ("strict", "fallback_assisted", "worst_case", "best_case"):
                # These are bounds on the *paired treatment effect*, not marginal
                # imputations applied symmetrically to both arms.  If both outcomes are
                # unavailable, setting both to zero would spuriously collapse uncertainty
                # to an effect of zero.  The lower bound makes treatment bad and comparator
                # good; the upper bound does the converse.
                p0_mode, p1_mode = {
                    "strict": ("strict", "strict"),
                    "fallback_assisted": ("fallback_assisted", "fallback_assisted"),
                    "worst_case": ("best_case", "worst_case"),
                    "best_case": ("worst_case", "best_case"),
                }[mode]
                diffs: list[float] = []
                tasks: list[str] = []
                clusters: list[str] = []
                missing_pairs = 0
                for obs in observations:
                    p0_view = _view(obs["p0"], p0_mode)
                    p1_view = _view(obs["p1"], p1_mode)
                    if p0_view is None or p1_view is None:
                        missing_pairs += 1
                        continue
                    a, b = p0_view.get(metric), p1_view.get(metric)
                    if a is None or b is None:
                        # Conditional truth guards may be jointly inapplicable.  They remain
                        # outside this metric's denominator, but the exact count is exposed.
                        if not (a is None and b is None):
                            missing_pairs += 1
                        continue
                    # Positive always means favourable to P1, including harm (where lower wins).
                    diff = float(a) - float(b) if metric == "critical_harm" else float(b) - float(a)
                    diffs.append(diff)
                    tasks.append(obs["task_id"])
                    clusters.append(obs["cluster_id"])
                task_diffs, task_clusters = _task_means(diffs, tasks, clusters)
                mode_effects[mode] = _difference_ci(
                    task_diffs,
                    task_clusters,
                    seed=seed_from(seed_namespace, arm, metric, mode),
                    n_boot=n_boot,
                )
                mode_effects[f"{mode}_n_pairs"] = len(diffs)
                mode_effects[f"{mode}_n_tasks"] = len(task_diffs)
                mode_effects[f"{mode}_missing_pairs"] = missing_pairs
            no_repair_diffs: list[float] = []
            no_repair_tasks: list[str] = []
            no_repair_clusters: list[str] = []
            no_repair_adverse_pairs = 0
            no_repair_judge_missing_pairs = 0
            no_repair_not_applicable_pairs = 0
            for obs in observations:
                p0_applicable = _quality_applicable(obs["p0"], metric)
                p1_applicable = _quality_applicable(obs["p1"], metric)
                if not p0_applicable and not p1_applicable:
                    no_repair_not_applicable_pairs += 1
                    continue
                validity = _selector_validity_counts(obs["p1"])
                adverse = bool(validity["no_repair_adverse"])
                no_repair_adverse_pairs += int(adverse)
                p0_strict = _view(obs["p0"], "strict")
                p1_strict = _view(obs["p1"], "strict")
                no_repair_judge_missing_pairs += int(
                    not isinstance(p0_strict, Mapping)
                    or p0_strict.get(metric) is None
                    or not isinstance(p1_strict, Mapping)
                    or p1_strict.get(metric) is None
                )
                a = _no_repair_quality_value(
                    obs["p0"], metric, treatment=False, force_adverse=False
                )
                b = _no_repair_quality_value(
                    obs["p1"], metric, treatment=True, force_adverse=adverse
                )
                if a is None:
                    a = 0.0 if metric == "critical_harm" else 1.0
                if b is None:
                    b = 1.0 if metric == "critical_harm" else 0.0
                diff = a - b if metric == "critical_harm" else b - a
                no_repair_diffs.append(float(diff))
                no_repair_tasks.append(obs["task_id"])
                no_repair_clusters.append(obs["cluster_id"])
            no_repair_task_diffs, no_repair_task_clusters = _task_means(
                no_repair_diffs, no_repair_tasks, no_repair_clusters
            )
            mode_effects["no_repair_adverse"] = _difference_ci(
                no_repair_task_diffs,
                no_repair_task_clusters,
                seed=seed_from(seed_namespace, arm, metric, "no_repair_adverse"),
                n_boot=n_boot,
            )
            mode_effects["no_repair_adverse_n_pairs"] = len(no_repair_diffs)
            mode_effects["no_repair_adverse_n_tasks"] = len(no_repair_task_diffs)
            mode_effects["no_repair_adverse_dirty_pairs"] = no_repair_adverse_pairs
            mode_effects["no_repair_adverse_judge_missing_pairs"] = no_repair_judge_missing_pairs
            mode_effects["no_repair_adverse_not_applicable_pairs"] = no_repair_not_applicable_pairs
            quality[metric] = mode_effects

        p1_work: list[float] = []
        p0_work: list[float] = []
        work_tasks: list[str] = []
        work_clusters: list[str] = []
        for obs in observations:
            p0_value, p1_value = _work(obs["p0"]), _work(obs["p1"])
            if p0_value is None or p1_value is None:
                continue
            p0_work.append(p0_value)
            p1_work.append(p1_value)
            work_tasks.append(obs["task_id"])
            work_clusters.append(obs["cluster_id"])
        p1_work_tasks, p0_work_tasks, work_task_clusters = _task_mean_pairs(
            p1_work, p0_work, work_tasks, work_clusters
        )
        work_ci = None
        if len(set(work_task_clusters)) >= 2:
            work_seed = seed_from(seed_namespace, arm, "work")
            work_lcb = paired_log_ratio_saving(
                p1_work_tasks,
                p0_work_tasks,
                work_task_clusters,
                n_boot=n_boot,
                seed=work_seed,
                side="lower",
            )
            work_ucb = paired_log_ratio_saving(
                p1_work_tasks,
                p0_work_tasks,
                work_task_clusters,
                n_boot=n_boot,
                seed=work_seed,
                side="upper",
            )
            work_ci = {
                "point": float(work_lcb.point),
                "lower": float(work_lcb.lower),
                "upper": float(work_ucb.upper),
                "n_clusters": work_lcb.n_clusters,
                "n_boot": work_lcb.n_boot,
                "interval_type": "paired_one_sided_95_percent_lower_and_upper_confidence_bounds",
                "effect_orientation": "positive_is_work_saving",
                "lower_bound": {
                    "value": float(work_lcb.lower),
                    "interval_type": "one_sided_95_percent_lower_confidence_bound",
                    "decision_direction": (
                        "lower_bound_establishes_minimum_meaningful_work_reduction"
                    ),
                },
                "upper_bound": {
                    "value": float(work_ucb.upper),
                    "interval_type": "one_sided_95_percent_upper_confidence_bound",
                    "decision_direction": "upper_bound_tests_remaining_work_reduction_headroom",
                },
            }

        selector_estimands: dict[str, dict] = {}
        for metric in _DIRECT_METRICS:
            node, field = _DIRECT_METRICS[metric]
            observed_values: list[float] = []
            observed_tasks: list[str] = []
            observed_clusters: list[str] = []
            worst_values: list[float] = []
            worst_tasks: list[str] = []
            worst_clusters: list[str] = []
            denominator_missing = 0
            eligible_denominator_rows = 0
            eligible_denominator_opportunities = 0.0
            not_applicable_rows = 0
            adverse_missing_value_rows = 0
            micro_observed_num: list[float] = []
            micro_observed_den: list[float] = []
            micro_observed_tasks: list[str] = []
            micro_observed_clusters: list[str] = []
            micro_worst_num: list[float] = []
            micro_worst_den: list[float] = []
            micro_worst_tasks: list[str] = []
            micro_worst_clusters: list[str] = []
            micro_denominator_missing = 0
            for obs in observations:
                row = obs["p1"]
                node_metrics = ((row.get("direct_node_metrics") or {}).get("by_node") or {}).get(
                    node
                ) or {}
                if not bool(node_metrics.get("applicable")):
                    continue
                value = _direct_value(row, metric)
                raw_denominator = (node_metrics.get("eligible_denominators") or {}).get(field)
                denominator_known = (
                    not isinstance(raw_denominator, bool)
                    and isinstance(raw_denominator, int | float)
                    and np.isfinite(float(raw_denominator))
                    and float(raw_denominator) >= 0
                )
                if denominator_known and float(raw_denominator) == 0:
                    if value is None:
                        not_applicable_rows += 1
                        eligible = False
                    else:
                        # A reported ratio with zero opportunities is internally inconsistent.
                        denominator_missing += 1
                        eligible = False
                else:
                    # Legacy records without explicit counts are readable only when they carry
                    # an actual value. A missing value with an unknown denominator is not NA.
                    eligible = bool(
                        (denominator_known and float(raw_denominator) > 0)
                        or (not denominator_known and value is not None)
                    )
                    if eligible:
                        eligible_denominator_rows += 1
                        eligible_denominator_opportunities += (
                            float(raw_denominator) if denominator_known else 1.0
                        )
                    else:
                        denominator_missing += 1
                strict_valid = (
                    row.get("assignment_state") == "COMMITTED"
                    and not row.get("fell_back")
                    and node_metrics.get("status") == "OK"
                    and bool(node_metrics.get("selector_guards_complete"))
                )
                if eligible and value is not None and strict_valid:
                    observed_values.append(value)
                    observed_tasks.append(obs["task_id"])
                    observed_clusters.append(obs["cluster_id"])
                # Only eligible denominators produce a direct value. A failed/fallback reducer
                # with a relevant direct denominator is adverse; a genuinely NA checkpoint is
                # not turned into a fabricated zero.
                if eligible:
                    # The denominator is the offered opportunity set. Once it is known, a
                    # failed/missing selector value is an adverse zero in all-offered ITT,
                    # never a reason to delete this row from the gate.
                    observed_value = value is not None and np.isfinite(float(value))
                    adverse_missing_value_rows += int(not observed_value)
                    worst_values.append(float(value) if strict_valid and observed_value else 0.0)
                    worst_tasks.append(obs["task_id"])
                    worst_clusters.append(obs["cluster_id"])

                if node == "C" and field in {
                    "candidate_coverage",
                    "total_published_prechunk_recall",
                }:
                    counts = _micro_counts(row, field)
                    if counts is None:
                        micro_denominator_missing += 1
                    else:
                        numerator, denominator = counts
                        micro_value_observed = value is not None and np.isfinite(float(value))
                        if strict_valid and micro_value_observed:
                            micro_observed_num.append(numerator)
                            micro_observed_den.append(denominator)
                            micro_observed_tasks.append(obs["task_id"])
                            micro_observed_clusters.append(obs["cluster_id"])
                        micro_worst_num.append(
                            numerator if strict_valid and micro_value_observed else 0.0
                        )
                        micro_worst_den.append(denominator)
                        micro_worst_tasks.append(obs["task_id"])
                        micro_worst_clusters.append(obs["cluster_id"])
            observed_task_values, observed_task_clusters = _task_means(
                observed_values, observed_tasks, observed_clusters
            )
            observed_ci = None
            if len(set(observed_task_clusters)) >= 2:
                observed_ci = _ci_obj(
                    cluster_bootstrap_ci(
                        observed_task_values,
                        observed_task_clusters,
                        np.mean,
                        n_boot=n_boot,
                        seed=seed_from(seed_namespace, arm, metric, "direct_observed"),
                        side="lower",
                    )
                )
            worst_task_values, worst_task_clusters = _task_means(
                worst_values, worst_tasks, worst_clusters
            )
            worst_ci = None
            if len(set(worst_task_clusters)) >= 2 and denominator_missing == 0:
                worst_ci = _ci_obj(
                    cluster_bootstrap_ci(
                        worst_task_values,
                        worst_task_clusters,
                        np.mean,
                        n_boot=n_boot,
                        seed=seed_from(seed_namespace, arm, metric, "direct_worst"),
                        side="lower",
                    )
                )
            if denominator_missing:
                gate_status = "INCOMPLETE_ELIGIBLE_DENOMINATOR"
            elif len(worst_values) != eligible_denominator_rows:
                gate_status = "INCOMPLETE_ALL_OFFERED_NUMERATOR"
            elif eligible_denominator_rows == 0:
                gate_status = "NOT_APPLICABLE_NO_ELIGIBLE_DENOMINATOR"
            elif worst_ci is None:
                gate_status = "NOT_ESTIMABLE_FEWER_THAN_TWO_SOURCE_CLUSTERS"
            else:
                gate_status = "ESTIMABLE"
            micro_observed = None
            micro_worst = None
            if node == "C" and field in {
                "candidate_coverage",
                "total_published_prechunk_recall",
            }:
                micro_observed = _cluster_micro_ratio_ci(
                    micro_observed_num,
                    micro_observed_den,
                    micro_observed_tasks,
                    micro_observed_clusters,
                    n_boot=n_boot,
                    seed=seed_from(seed_namespace, arm, metric, "direct_micro_observed"),
                )
                if micro_denominator_missing == 0:
                    micro_worst = _cluster_micro_ratio_ci(
                        micro_worst_num,
                        micro_worst_den,
                        micro_worst_tasks,
                        micro_worst_clusters,
                        n_boot=n_boot,
                        seed=seed_from(seed_namespace, arm, metric, "direct_micro_worst"),
                    )
            selector_estimands[metric] = {
                "strict_observed": observed_ci,
                "strict_observed_n": len(observed_values),
                "strict_observed_tasks": len(observed_task_values),
                "all_offered_worst_case": worst_ci,
                "all_offered_lcb": (worst_ci["lower"] if worst_ci is not None else None),
                "all_offered_worst_case_n": len(worst_values),
                "all_offered_worst_case_tasks": len(worst_task_values),
                "missing_direct_denominator_count": denominator_missing,
                "eligible_denominator_rows": eligible_denominator_rows,
                "eligible_denominator_opportunities": eligible_denominator_opportunities,
                "not_applicable_rows": not_applicable_rows,
                "adverse_missing_value_rows": adverse_missing_value_rows,
                "all_offered_eligible_rows_covered": (
                    len(worst_values) == eligible_denominator_rows
                ),
                "all_offered_gate_status": gate_status,
                "decision_direction": "lower_bound_must_meet_preregistered_absolute_gate",
                "macro_definition": (
                    "task_mean_of_within_run_pooled_frozen_weight_ratios"
                    if field == "weighted_evidence_recall"
                    else "task_mean_of_within_run_pooled_negative_gap_opportunities"
                    if field == "negative_gap_recall"
                    else "task_mean_of_checkpoint_macro_means"
                ),
                "strict_observed_micro": micro_observed,
                "all_offered_worst_case_micro": micro_worst,
                "all_offered_worst_case_micro_n": len(micro_worst_den),
                "missing_micro_denominator_count": (
                    micro_denominator_missing
                    if node == "C"
                    and field
                    in {
                        "candidate_coverage",
                        "total_published_prechunk_recall",
                    }
                    else None
                ),
            }

        has_h = any(
            bool(
                (
                    ((obs["p1"].get("direct_node_metrics") or {}).get("by_node") or {}).get("H")
                    or {}
                ).get("applicable")
            )
            for obs in observations
        )
        has_c = any(
            bool(
                (
                    ((obs["p1"].get("direct_node_metrics") or {}).get("by_node") or {}).get("C")
                    or {}
                ).get("applicable")
            )
            for obs in observations
        )
        required_direct_estimands = (
            [
                "h_candidate_coverage",
                "h_total_published_prechunk_recall",
            ]
            if has_h
            else []
        ) + (
            [
                "c_candidate_coverage",
                "c_total_published_prechunk_recall",
            ]
            if has_c
            else []
        )
        required_c_micro_estimands = (
            [
                "c_candidate_coverage",
                "c_total_published_prechunk_recall",
            ]
            if has_c
            else []
        )
        required_selector_gate_estimands = (
            [
                "h_weighted_evidence_recall",
                "h_critical_truth_recall",
                "h_contradiction_pair_recall",
                "h_negative_gap_recall",
            ]
            if has_h
            else []
        ) + (
            [
                "c_weighted_evidence_recall",
                "c_critical_truth_recall",
                "c_contradiction_pair_recall",
                "c_negative_gap_recall",
            ]
            if has_c
            else []
        )
        selector_validity_rows = [_selector_validity_counts(obs["p1"]) for obs in observations]
        selector_validity_totals = {
            field: sum(int(row[field]) for row in selector_validity_rows)
            for field in _SELECTOR_COUNT_FIELDS
        }
        selector_validity_totals["missing_node_summary_count"] = sum(
            int(row["missing_node_summary_count"]) for row in selector_validity_rows
        )
        selector_validity_totals["no_repair_adverse_row_count"] = sum(
            int(bool(row["no_repair_adverse"])) for row in selector_validity_rows
        )
        strict_valid_rate = _selector_rate_estimand(
            observations,
            numerator_field="strict_valid_count",
            side="lower",
            seed=seed_from(seed_namespace, arm, "selector_strict_valid_rate"),
            n_boot=n_boot,
        )
        repair_rate = _selector_rate_estimand(
            observations,
            numerator_field="repaired_attempt_count",
            side="upper",
            seed=seed_from(seed_namespace, arm, "selector_repair_rate"),
            n_boot=n_boot,
        )
        invalid_id_count = selector_validity_totals["invalid_id_count"]
        normalization_trace_errors = selector_validity_totals["normalization_trace_error_count"]
        selector_output_validity = {
            **selector_validity_totals,
            "all_offered_rows": len(observations),
            "strict_valid_rate_lcb": strict_valid_rate,
            "repair_rate_ucb": repair_rate,
            "invalid_id_hard_gate": {
                "status": (
                    "NOT_ESTABLISHED_INCOMPLETE_NORMALIZATION_TRACE"
                    if normalization_trace_errors
                    else "PASS"
                    if invalid_id_count == 0
                    else "FAIL"
                ),
                "invalid_id_count": invalid_id_count,
                "required_maximum": 0,
                "definition": "out_of_set_label rejections across all offered selector attempts",
                "duplicates_are_invalid_ids": False,
            },
        }
        selector_efficiency = {
            node: _selector_efficiency_summary(
                observations,
                node=node,
                seed_namespace=seed_namespace,
                arm=arm,
                n_boot=n_boot,
            )
            for node in ("H", "C")
        }
        critical_harm_absolute = _absolute_critical_harm_summary(
            observations,
            seed_namespace=seed_namespace,
            arm=arm,
            n_boot=n_boot,
        )
        terminal_failure_effect = _paired_risk_difference(
            observations,
            field="terminal_failure",
            seed=seed_from(seed_namespace, arm, "terminal_failure_risk"),
            n_boot=n_boot,
        )
        fallback_effect = _paired_risk_difference(
            observations,
            field="fallback",
            seed=seed_from(seed_namespace, arm, "fallback_risk"),
            n_boot=n_boot,
        )

        machine_estimands_ready = (
            len(set(clusters_all)) >= 2
            and paired_invalid == 0
            and trace_unavailable == 0
            and trace_invalid == 0
            and selector_guard_unavailable == 0
            and len(p1_work) == len(observations)
            and work_ci is not None
            and bool(required_direct_estimands)
            and all(
                selector_estimands[metric]["all_offered_worst_case"] is not None
                and selector_estimands[metric]["missing_direct_denominator_count"] == 0
                and selector_estimands[metric]["all_offered_worst_case_n"]
                == selector_estimands[metric]["eligible_denominator_rows"]
                for metric in required_direct_estimands
            )
            and all(
                selector_estimands[metric]["all_offered_worst_case_micro"] is not None
                and selector_estimands[metric]["missing_micro_denominator_count"] == 0
                and selector_estimands[metric]["all_offered_worst_case_micro_n"]
                == selector_estimands[metric]["eligible_denominator_rows"]
                for metric in required_c_micro_estimands
            )
            and all(
                selector_estimands[metric]["all_offered_gate_status"]
                in {
                    "ESTIMABLE",
                    "NOT_APPLICABLE_NO_ELIGIBLE_DENOMINATOR",
                }
                for metric in required_selector_gate_estimands
            )
            and selector_output_validity["invalid_id_hard_gate"]["status"] == "PASS"
            and strict_valid_rate["status"] == "ESTIMABLE"
            and repair_rate["status"] == "ESTIMABLE"
            and all(
                quality[m]["worst_case"] is not None
                and quality[m]["worst_case_n_pairs"] == len(observations)
                and quality[m]["no_repair_adverse"] is not None
                and quality[m]["no_repair_adverse_n_pairs"] == len(observations)
                for m in _CORE_QUALITY_METRICS
            )
            and all(
                quality[m]["worst_case_n_pairs"] == 0 or quality[m]["worst_case"] is not None
                for m in _CONDITIONAL_QUALITY_METRICS
            )
            and all(
                quality[m]["no_repair_adverse_n_pairs"] == 0
                or quality[m]["no_repair_adverse"] is not None
                for m in _CONDITIONAL_QUALITY_METRICS
            )
            and not judge_drift
            and terminal_failure_effect["status"] == "OK"
            and fallback_effect["status"] == "OK"
            and len(claim_scopes) == 1
            and "" not in claim_scopes
        )
        arms[arm] = {
            "all_offered_pairs": len(observations),
            "independent_clusters": len(set(clusters_all)),
            "assignment_states": dict(sorted(state_counts.items())),
            "fallback_count": fallbacks,
            "judge_unavailable_count": judge_unavailable,
            "direct_trace_unavailable_count": trace_unavailable,
            "direct_trace_invalid_count": trace_invalid,
            "selector_guard_unavailable_count": selector_guard_unavailable,
            "paired_invalid_block_count": paired_invalid,
            "paired_invalid_reasons": dict(sorted(paired_invalid_reasons.items())),
            "structural_pass": (
                paired_invalid == 0
                and trace_unavailable == 0
                and trace_invalid == 0
                and selector_guard_unavailable == 0
            ),
            "work_pairs_observed": len(p1_work),
            "work_pairs_missing": len(observations) - len(p1_work),
            "work_tasks_observed": len(p1_work_tasks),
            "work_saving": work_ci,
            "terminal_failure_risk_difference": terminal_failure_effect,
            "fallback_risk_difference": fallback_effect,
            "selector_estimands": selector_estimands,
            "required_direct_estimands": required_direct_estimands,
            "required_c_micro_estimands": required_c_micro_estimands,
            "required_selector_gate_estimands": required_selector_gate_estimands,
            "selector_output_validity": selector_output_validity,
            "selector_efficiency": selector_efficiency,
            "critical_harm_absolute": critical_harm_absolute,
            "quality_effects": quality,
            "machine_estimands_ready": machine_estimands_ready,
            "verdict_ready": machine_estimands_ready and truth_audited,
            "verdict_blockers": (
                []
                if machine_estimands_ready and truth_audited
                else (["TRUTH_NOT_HUMAN_AUDITED"] if not truth_audited else [])
                + (["PAIRED_BLOCK_INVALID"] if paired_invalid else [])
                + (["MACHINE_ESTIMANDS_INCOMPLETE"] if not machine_estimands_ready else [])
            ),
        }

    used_tasks = sorted({str(r.get("task_id") or "") for r in records})
    frozen_clusters = {task: str(cluster_by_task[task]) for task in used_tasks}
    body = {
        "schema_version": "itt_verdict_input_v1",
        "run_id": next(iter(run_ids)),
        "phase_id": next(iter(phase_ids)),
        "comparator_arm": comparator_arm,
        "blocks_offered": len(records),
        "execution_binding_sha256": execution_binding_sha,
        "protocol_document_sha256": protocol_document_sha,
        "schedule_sha256": schedule_sha,
        "freeze_root_sha256": freeze_root_sha,
        "analysis_design_receipt_sha256": str(verified_scope["analysis_design_receipt_sha256"]),
        "task_feature_registry_sha256": str(verified_scope["task_feature_registry_sha256"]),
        "eligibility_spec_content_sha256": str(verified_scope["eligibility_spec_content_sha256"]),
        "evaluation_scope_sha256": recorded_scope_sha,
        "judge_policy_sha256": judge_policy_sha,
        "judge_policy_sha256s": sorted(judge_policies),
        "judge_runtime_drift": judge_drift,
        "claim_scope": next(iter(claim_scopes)) if len(claim_scopes) == 1 else None,
        "score_declared_truth_verifier_statuses": sorted(truth_verifier_statuses),
        "score_truth_verifier_status_is_decision_input": False,
        "human_audit_receipt_sha256": human_audit_receipt_sha,
        "truth_audited": truth_audited,
        "source_cluster_map_sha256": sha256_hex(canonical_json(frozen_clusters)),
        "effect_orientation": (
            "positive favors treatment; critical_harm is comparator minus treatment"
        ),
        "arms": arms,
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def write_verdict_inputs(body: dict, path: Path) -> str:
    """Write-once verdict input; a changed analysis is a new artifact, not an overwrite."""
    path = Path(path)
    digest = str(body.get("content_sha256") or "")
    actual = sha256_hex(canonical_json({k: v for k, v in body.items() if k != "content_sha256"}))
    if digest != actual:
        raise ValueError("verdict-input content_sha256 does not match its content")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o440)
    except FileExistsError:
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != body:
            raise ValueError(f"{path} already contains a different verdict input") from None
    return digest
