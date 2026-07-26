"""Content-addressed human-audit workflow for upgrading provisional measurements.

Machine-authored truth and machine relation judgments may support a formative result, but they
cannot mark themselves ``AUDITED``.  This module creates the deterministic review queue, accepts
an explicit reviewer attestation over every queued task, applies the frozen gate, and emits the
only receipt analysis is allowed to treat as human-audited.

Truth packets and score files remain immutable.  The receipt binds their hashes rather than
rewriting ``verifier_status`` in place, which preserves both the pre-audit artifact and the
reviewer's decision.  Consequently, finding *any* error makes the original truth/score pair
ineligible for an audited receipt.  A corrected truth packet would require a content-addressed
all-arm re-score; that workflow is deliberately not emulated here.  The currently supported
upgrade is therefore correction-free only.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from pathlib import Path

from ..analysis.design import REGISTRY_FILENAME, task_feature_registry_sha256
from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..scoped_paths import resolve_scoped_path
from .human_audit import audit_gate_ready, select_audit_queue

__all__ = [
    "AUDIT_FINALIZATION_FILENAME",
    "AUDITED_RECEIPT_FILENAME",
    "finalize_human_audit",
    "load_human_audit_receipt",
    "prepare_human_audit",
    "validate_human_audit_receipt",
]

AUDIT_QUEUE_FILENAME = "AUDIT_QUEUE.json"
AUDIT_FINALIZATION_FILENAME = "AUDIT_FINALIZATION.json"
AUDITED_RECEIPT_FILENAME = "AUDITED_RECEIPT.json"

_REVIEW_CHECKS = (
    "critical_truth_atoms_correct",
    "critical_source_edges_correct",
    "noncritical_truth_atoms_correct",
    "noncritical_source_edges_correct",
    "relation_judgments_correct",
    "critical_harm_complete",
)
_CRITICAL_ERROR_CHECKS = frozenset(
    {
        "critical_truth_atoms_correct",
        "critical_source_edges_correct",
    }
)
_NONCRITICAL_ERROR_CHECKS = frozenset(
    {
        "noncritical_truth_atoms_correct",
        "noncritical_source_edges_correct",
        "relation_judgments_correct",
    }
)


def _unsigned_sha(body: Mapping) -> str:
    return sha256_hex(
        canonical_json({key: value for key, value in body.items() if key != "content_sha256"})
    )


def _verified(path: Path, *, hash_field: str = "content_sha256") -> dict:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid audit artifact {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"audit artifact {path} is not an object")
    recorded = str(body.get(hash_field) or "")
    actual = sha256_hex(
        canonical_json({key: value for key, value in body.items() if key != hash_field})
    )
    if len(recorded) != 64 or recorded != actual:
        raise ValueError(f"audit artifact {path} does not verify")
    return body


def _scope_path(
    settings,
    run_id: str,
    phase_id: str,
    *tail: str,
) -> Path:
    return resolve_scoped_path(
        settings.path("judgments"),
        run_id=run_id,
        phase_id=phase_id,
        tail=tuple(tail),
    )


def _write_once(path: Path, body: Mapping) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(dict(body), indent=2, sort_keys=True) + "\n"
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(raw)
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o440)
    except FileExistsError:
        if _verified(path) != dict(body):
            raise ValueError(f"{path} already contains a different audit artifact") from None


def _frozen_registry(settings, scope_receipt: Mapping) -> dict:
    path = settings.path("evaluator_root") / "analysis_design" / REGISTRY_FILENAME
    registry = _verified(path)
    records = registry.get("records")
    if not isinstance(records, dict) or not records:
        raise ValueError("frozen task feature registry has no records")
    for task_id, record in records.items():
        if (
            not isinstance(record, dict)
            or str(record.get("task_id") or "") != str(task_id)
            or str(record.get("content_sha256") or "") != _unsigned_sha(record)
        ):
            raise ValueError(f"frozen feature record {task_id!r} does not verify")
    calculated_registry_sha = task_feature_registry_sha256(records)
    if (
        str(registry.get("task_feature_registry_sha256") or "") != calculated_registry_sha
        or str(scope_receipt.get("task_feature_registry_sha256") or "") != calculated_registry_sha
    ):
        raise ValueError("audit queue feature registry is not the evaluation-scoped registry")
    return registry


def _strict(row: Mapping, metric: str) -> float | None:
    views = row.get("quality_views")
    strict = views.get("strict") if isinstance(views, Mapping) else None
    value = strict.get(metric) if isinstance(strict, Mapping) else None
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def prepare_human_audit(settings, *, run_id: str, phase_id: str) -> dict:
    """Freeze the review queue for one exact all-offered evaluation scope."""
    from ..analysis.estimands import load_scoped_scores

    scores = load_scoped_scores(settings.path("judgments"), run_id=run_id, phase_id=phase_id)
    scope = getattr(scores, "scope_receipt", None)
    if not isinstance(scope, Mapping):
        raise ValueError("verified evaluation scope is unavailable")
    registry = _frozen_registry(settings, scope)
    features = registry["records"]
    tasks = sorted({str(score.get("task_id") or "") for score in scores})
    if not tasks or any(task not in features for task in tasks):
        raise ValueError("audit score tasks are not a subset of the frozen registry")

    tasks_by_cluster: dict[str, list[str]] = {}
    for task_id in tasks:
        cluster_id = str(features[task_id].get("cluster_id") or "")
        if not cluster_id:
            raise ValueError(f"frozen task {task_id} has no cluster")
        tasks_by_cluster.setdefault(cluster_id, []).append(task_id)

    critical_misses: set[str] = set()
    disagreements: set[str] = set()
    near_margin: set[str] = set()
    epsilon = float(settings.get("judge", "human_audit", "near_margin_epsilon"))
    ni_margin = (
        abs(
            float(settings.get("decision", "quality_ni_margin", "weighted_required_atom_recall_pp"))
        )
        / 100.0
    )
    truth_by_task: dict[str, str] = {}
    score_hashes: dict[str, str] = {}
    blocks_by_task: dict[str, list[str]] = {}

    for score in scores:
        task_id = str(score["task_id"])
        block_id = str(score["block_id"])
        truth_sha = str(score.get("truth_packet_sha256") or "")
        score_sha = str(score.get("content_sha256") or "")
        if len(truth_sha) != 64 or len(score_sha) != 64:
            raise ValueError(f"score {block_id} lacks truth/score content identity")
        previous_truth = truth_by_task.setdefault(task_id, truth_sha)
        if previous_truth != truth_sha:
            raise ValueError(f"task {task_id} was scored against multiple truth packets")
        score_hashes[block_id] = score_sha
        blocks_by_task.setdefault(task_id, []).append(block_id)
        for item in score.get("human_queue") or ():
            if isinstance(item, Mapping):
                disagreements.add(task_id)

        per_arm = score.get("per_arm")
        if not isinstance(per_arm, Mapping):
            raise ValueError(f"score {block_id} has no per_arm map")
        p0_rows = [row for key, row in per_arm.items() if str(key).split(":", 1)[0] == "P0"]
        if len(p0_rows) != 1:
            raise ValueError(f"score {block_id} needs exactly one P0 row")
        p0 = p0_rows[0]
        p0_recall = _strict(p0, "weighted_required_atom_recall")
        for key, row in per_arm.items():
            if str(key).split(":", 1)[0] == "P0" or not isinstance(row, Mapping):
                continue
            harm = _strict(row, "critical_harm")
            safety = _strict(row, "critical_atom_safety")
            if (harm is not None and harm > 0.0) or (safety is not None and safety < 1.0):
                critical_misses.add(task_id)
            recall = _strict(row, "weighted_required_atom_recall")
            if (
                recall is not None
                and p0_recall is not None
                and abs((recall - p0_recall) + ni_margin) <= epsilon
            ):
                near_margin.add(task_id)

    policy = settings.get("judge", "human_audit")
    selected = select_audit_queue(
        tasks_by_stratum=tasks_by_cluster,
        sample_fraction=float(policy["random_sample_fraction"]),
        critical_misses=sorted(critical_misses),
        disagreements=sorted(disagreements),
        near_margin=sorted(near_margin),
        seed=int(settings.get("judge", "model", "seed")),
    )
    items = []
    for item in selected:
        payload = {
            "task_id": item.task_id,
            "kind": item.kind,
            "detail": item.detail,
            "critical": item.kind == "CRITICAL_MISS",
            "truth_packet_sha256": truth_by_task[item.task_id],
            "score_block_ids": sorted(blocks_by_task[item.task_id]),
        }
        payload["item_id"] = sha256_hex(canonical_json(payload))
        items.append(payload)
    body = {
        "schema_version": "human_audit_queue_v2",
        "run_id": run_id,
        "phase_id": phase_id,
        "execution_binding_sha256": str(scope["execution_binding_sha256"]),
        "protocol_document_sha256": str(scope["protocol_document_sha256"]),
        "evaluation_scope_sha256": str(scope["evaluation_scope_sha256"]),
        "task_feature_registry_sha256": str(scope["task_feature_registry_sha256"]),
        "policy": dict(policy),
        "policy_sha256": sha256_hex(canonical_json(policy)),
        "tasks_in_scope": len(tasks),
        "task_ids_sha256": sha256_hex(canonical_json(tasks)),
        "truth_packet_sha256_by_task": dict(sorted(truth_by_task.items())),
        "score_content_sha256_by_block": dict(sorted(score_hashes.items())),
        "review_contract": {
            "version": "correction_free_all_dimensions_v1",
            "required_boolean_checks": list(_REVIEW_CHECKS),
            "correction_summary_required_on_error": True,
            "any_error_requires_new_truth_and_all_arm_rescore": True,
            "corrected_truth_rescore_acceptance_supported": False,
        },
        "items": sorted(items, key=lambda value: value["item_id"]),
    }
    body["content_sha256"] = _unsigned_sha(body)
    _write_once(_scope_path(settings, run_id, phase_id, "human_audit", AUDIT_QUEUE_FILENAME), body)
    return body


def finalize_human_audit(
    settings,
    *,
    run_id: str,
    phase_id: str,
    results: Mapping,
) -> dict:
    """Finalize one correction-free audit attempt for an immutable score scope.

    A false review check is evidence that at least one frozen truth/score input is wrong.
    Statistical audit tolerances do not repair those bytes, so such an attempt is recorded as
    ``CORRECTIONS_REQUIRED`` and can never create ``AUDITED_RECEIPT.json``.  The finalization
    pointer is write-once: a later clean-looking submission cannot overwrite an earlier
    discovered correction for the same queue.
    """
    queue = _verified(_scope_path(settings, run_id, phase_id, "human_audit", AUDIT_QUEUE_FILENAME))
    expected_contract = {
        "version": "correction_free_all_dimensions_v1",
        "required_boolean_checks": list(_REVIEW_CHECKS),
        "correction_summary_required_on_error": True,
        "any_error_requires_new_truth_and_all_arm_rescore": True,
        "corrected_truth_rescore_acceptance_supported": False,
    }
    if (
        queue.get("schema_version") != "human_audit_queue_v2"
        or queue.get("run_id") != run_id
        or queue.get("phase_id") != phase_id
        or queue.get("review_contract") != expected_contract
    ):
        raise ValueError("audit queue does not carry the frozen correction-free v2 contract")
    expected_result_keys = {
        "schema_version",
        "queue_content_sha256",
        "reviewer_id",
        "reviewed_at_utc",
        "reviewer_attestation",
        "items",
        "content_sha256",
    }
    if set(results) != expected_result_keys:
        raise ValueError(
            "human-audit results schema is not closed; missing="
            f"{sorted(expected_result_keys - set(results))}, "
            f"extra={sorted(set(results) - expected_result_keys)}"
        )
    if results.get("schema_version") != "human_audit_results_v2":
        raise ValueError("unsupported human-audit result schema")
    recorded = str(results.get("content_sha256") or "")
    if recorded != _unsigned_sha(results):
        raise ValueError("human-audit results do not verify")
    if str(results.get("queue_content_sha256") or "") != str(queue["content_sha256"]):
        raise ValueError("human-audit results belong to another queue")
    reviewer = str(results.get("reviewer_id") or "").strip()
    reviewed_at = str(results.get("reviewed_at_utc") or "").strip()
    attestation = str(results.get("reviewer_attestation") or "").strip()
    if not reviewer or not reviewed_at or len(attestation) < 20:
        raise ValueError("reviewer identity, time, and explicit attestation are required")

    expected = {str(item["item_id"]): item for item in queue["items"]}
    rows = results.get("items")
    if not isinstance(rows, list):
        raise ValueError("human-audit results.items must be a list")
    observed: dict[str, dict] = {}
    expected_row_keys = {"item_id", *_REVIEW_CHECKS, "correction_summary"}
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("each human-audit result must be an object")
        item_id = str(row.get("item_id") or "")
        if item_id not in expected or item_id in observed:
            raise ValueError(f"unknown or duplicate audit item {item_id!r}")
        if set(row) != expected_row_keys:
            raise ValueError(
                f"audit item {item_id} schema is not closed; missing="
                f"{sorted(expected_row_keys - set(row))}, "
                f"extra={sorted(set(row) - expected_row_keys)}"
            )
        for name in _REVIEW_CHECKS:
            if not isinstance(row.get(name), bool):
                raise ValueError(f"audit item {item_id} needs boolean {name}")
        summary = row.get("correction_summary")
        if not isinstance(summary, str):
            raise ValueError(f"audit item {item_id} needs string correction_summary")
        has_error = any(not bool(row[name]) for name in _REVIEW_CHECKS)
        if has_error and not summary.strip():
            raise ValueError(f"audit item {item_id} found an error but has no correction summary")
        if not has_error and summary.strip():
            raise ValueError(
                f"audit item {item_id} claims no error but supplies a correction summary"
            )
        observed[item_id] = row
    if set(observed) != set(expected):
        raise ValueError(
            "review must cover the exact queue; missing="
            f"{sorted(set(expected) - set(observed))}, "
            f"extra={sorted(set(observed) - set(expected))}"
        )

    random_tasks = {
        item["task_id"] for item in expected.values() if item["kind"] == "RANDOM_SAMPLE"
    }
    sampled_fraction = len(random_tasks) / int(queue["tasks_in_scope"])
    error_counts_by_check = {
        name: sum(not bool(row[name]) for row in observed.values()) for name in _REVIEW_CHECKS
    }
    correction_count = sum(error_counts_by_check.values())
    items_requiring_correction = sum(
        any(not bool(row[name]) for name in _REVIEW_CHECKS) for row in observed.values()
    )
    critical_errors = sum(error_counts_by_check[name] for name in _CRITICAL_ERROR_CHECKS)
    critical_harm_misses = error_counts_by_check["critical_harm_complete"]
    noncritical_accuracy = (
        sum(all(bool(row[name]) for name in _NONCRITICAL_ERROR_CHECKS) for row in observed.values())
        / len(observed)
        if observed
        else 1.0
    )
    policy = queue["policy"]
    gate = audit_gate_ready(
        sampled_fraction=sampled_fraction,
        required_fraction=float(policy["random_sample_fraction"]),
        critical_errors=critical_errors,
        noncritical_accuracy=noncritical_accuracy,
        noncritical_accuracy_min=float(policy["non_critical_accuracy_gate"]),
        critical_harm_misses=critical_harm_misses,
    )
    if correction_count:
        status = "CORRECTIONS_REQUIRED"
        gate_reasons = [
            *gate.reasons,
            (
                f"{correction_count} audited correction(s) across "
                f"{items_requiring_correction} item(s); immutable original scores "
                "require corrected truth plus all-arm re-score"
            ),
        ]
        correction_resolution = "REQUIRES_NEW_TRUTH_AND_ALL_ARM_RESCORE"
    elif gate.ready:
        status = "AUDITED"
        gate_reasons = []
        correction_resolution = "NOT_REQUIRED"
    else:
        status = "FAILED_AUDIT_GATE"
        gate_reasons = list(gate.reasons)
        correction_resolution = "NOT_APPLICABLE_GATE_FAILED"

    results_path = _scope_path(
        settings, run_id, phase_id, "human_audit", "results", f"{recorded}.json"
    )
    _write_once(results_path, results)
    receipt = {
        "schema_version": "human_audit_receipt_v2",
        "status": status,
        "run_id": run_id,
        "phase_id": phase_id,
        "execution_binding_sha256": queue["execution_binding_sha256"],
        "protocol_document_sha256": queue["protocol_document_sha256"],
        "evaluation_scope_sha256": queue["evaluation_scope_sha256"],
        "queue_content_sha256": queue["content_sha256"],
        "results_content_sha256": recorded,
        "reviewer_id": reviewer,
        "reviewed_at_utc": reviewed_at,
        "reviewer_attestation_sha256": sha256_hex(attestation.encode("utf-8")),
        "truth_packet_sha256_by_task": queue["truth_packet_sha256_by_task"],
        "score_content_sha256_by_block": queue["score_content_sha256_by_block"],
        "sampled_fraction": sampled_fraction,
        "required_fraction": float(policy["random_sample_fraction"]),
        "audit_basis": "ORIGINAL_ARTIFACTS_CORRECTION_FREE",
        "correction_count": correction_count,
        "items_requiring_correction": items_requiring_correction,
        "error_counts_by_check": error_counts_by_check,
        "correction_resolution": correction_resolution,
        "corrected_truth_packet_sha256_by_task": None,
        "all_arm_rescore_manifest_sha256": None,
        "critical_errors": critical_errors,
        "critical_harm_misses": critical_harm_misses,
        "noncritical_accuracy": noncritical_accuracy,
        "gate_reasons": gate_reasons,
    }
    receipt["content_sha256"] = _unsigned_sha(receipt)
    _write_once(
        _scope_path(
            settings,
            run_id,
            phase_id,
            "human_audit",
            AUDIT_FINALIZATION_FILENAME,
        ),
        receipt,
    )
    history_path = _scope_path(
        settings,
        run_id,
        phase_id,
        "human_audit",
        "receipts",
        f"{receipt['content_sha256']}.json",
    )
    _write_once(history_path, receipt)
    if status == "AUDITED":
        _write_once(
            _scope_path(settings, run_id, phase_id, "human_audit", AUDITED_RECEIPT_FILENAME),
            receipt,
        )
    return receipt


def load_human_audit_receipt(
    settings,
    *,
    run_id: str,
    phase_id: str,
    evaluation_scope_sha256: str,
    execution_binding_sha256: str,
    protocol_document_sha256: str,
    truth_packet_sha256_by_task: Mapping[str, str],
    score_content_sha256_by_block: Mapping[str, str],
) -> dict | None:
    """Return a fully bound ready receipt; absence means the result remains provisional."""
    finalization_path = _scope_path(
        settings, run_id, phase_id, "human_audit", AUDIT_FINALIZATION_FILENAME
    )
    if not finalization_path.exists():
        return None
    finalization = _verified(finalization_path)
    if (
        finalization.get("schema_version") != "human_audit_receipt_v2"
        or finalization.get("run_id") != run_id
        or finalization.get("phase_id") != phase_id
    ):
        raise ValueError("human-audit finalization identity/schema is invalid")
    if finalization.get("status") != "AUDITED":
        return None
    path = _scope_path(settings, run_id, phase_id, "human_audit", AUDITED_RECEIPT_FILENAME)
    if not path.exists():
        raise ValueError("audited finalization has no audited receipt pointer")
    receipt = _verified(path)
    if receipt != finalization:
        raise ValueError("human-audit receipt differs from the immutable finalization")
    return validate_human_audit_receipt(
        receipt,
        run_id=run_id,
        phase_id=phase_id,
        evaluation_scope_sha256=evaluation_scope_sha256,
        execution_binding_sha256=execution_binding_sha256,
        protocol_document_sha256=protocol_document_sha256,
        truth_packet_sha256_by_task=truth_packet_sha256_by_task,
        score_content_sha256_by_block=score_content_sha256_by_block,
    )


def validate_human_audit_receipt(
    receipt: Mapping,
    *,
    run_id: str,
    phase_id: str,
    evaluation_scope_sha256: str,
    execution_binding_sha256: str,
    protocol_document_sha256: str,
    truth_packet_sha256_by_task: Mapping[str, str] | None = None,
    score_content_sha256_by_block: Mapping[str, str] | None = None,
) -> dict:
    """Validate the only audit upgrade currently supported: correction-free v2.

    An ``AUDITED`` label alone is not enough.  Any reviewer correction means the immutable
    truth/score bytes being analyzed are known to be wrong and require a new truth packet plus
    all-arm re-score before a later audit can bind them.
    """
    if not isinstance(receipt, Mapping):
        raise ValueError("human-audit receipt must be an object")
    body = dict(receipt)
    if str(body.get("content_sha256") or "") != _unsigned_sha(body):
        raise ValueError("human-audit receipt content hash does not verify")
    counts = body.get("error_counts_by_check")
    integer_zero_fields = (
        "correction_count",
        "items_requiring_correction",
        "critical_errors",
        "critical_harm_misses",
    )
    zero_counts_valid = all(
        not isinstance(body.get(field), bool)
        and isinstance(body.get(field), int)
        and body.get(field) == 0
        for field in integer_zero_fields
    )
    per_check_valid = (
        isinstance(counts, Mapping)
        and set(counts) == set(_REVIEW_CHECKS)
        and all(
            not isinstance(value, bool) and isinstance(value, int) and value == 0
            for value in counts.values()
        )
    )
    sampled = body.get("sampled_fraction")
    required = body.get("required_fraction")
    coverage_valid = (
        not isinstance(sampled, bool)
        and isinstance(sampled, int | float)
        and not isinstance(required, bool)
        and isinstance(required, int | float)
        and float(sampled) >= float(required)
    )
    identity_valid = (
        body.get("schema_version") == "human_audit_receipt_v2"
        and body.get("status") == "AUDITED"
        and body.get("run_id") == run_id
        and body.get("phase_id") == phase_id
        and body.get("evaluation_scope_sha256") == evaluation_scope_sha256
        and body.get("execution_binding_sha256") == execution_binding_sha256
        and body.get("protocol_document_sha256") == protocol_document_sha256
    )
    correction_free = (
        body.get("audit_basis") == "ORIGINAL_ARTIFACTS_CORRECTION_FREE"
        and zero_counts_valid
        and per_check_valid
        and body.get("correction_resolution") == "NOT_REQUIRED"
        and body.get("corrected_truth_packet_sha256_by_task") is None
        and body.get("all_arm_rescore_manifest_sha256") is None
        and body.get("gate_reasons") == []
        and coverage_valid
    )
    if not identity_valid or not correction_free:
        raise ValueError(
            "human-audit receipt is not a correction-free v2 receipt bound to this scope"
        )
    if truth_packet_sha256_by_task is not None and body.get("truth_packet_sha256_by_task") != dict(
        sorted(truth_packet_sha256_by_task.items())
    ):
        raise ValueError("human-audit receipt binds different truth packets")
    if score_content_sha256_by_block is not None and body.get(
        "score_content_sha256_by_block"
    ) != dict(sorted(score_content_sha256_by_block.items())):
        raise ValueError("human-audit receipt binds different score bytes")
    return body
