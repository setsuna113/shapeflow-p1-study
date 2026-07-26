"""A machine result can become AUDITED only through a bound reviewer receipt."""

from __future__ import annotations

from copy import deepcopy

import pytest

from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.evaluation.human_audit_workflow import (
    finalize_human_audit,
    load_human_audit_receipt,
)
from shapeflow_p1.hashing import sha256_hex


def _seal(body):
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _queue(tmp_path):
    directory = tmp_path / "judgments" / "RUN1" / "PHASE1" / "human_audit"
    directory.mkdir(parents=True)
    items = [
        {
            "item_id": "i1",
            "task_id": "T1",
            "kind": "RANDOM_SAMPLE",
            "detail": "cluster-a",
            "critical": False,
            "truth_packet_sha256": "a" * 64,
            "score_block_ids": ["B1"],
        },
        {
            "item_id": "i2",
            "task_id": "T2",
            "kind": "CRITICAL_MISS",
            "detail": "",
            "critical": True,
            "truth_packet_sha256": "b" * 64,
            "score_block_ids": ["B2"],
        },
    ]
    queue = _seal(
        {
            "schema_version": "human_audit_queue_v2",
            "run_id": "RUN1",
            "phase_id": "PHASE1",
            "execution_binding_sha256": "3" * 64,
            "protocol_document_sha256": "4" * 64,
            "evaluation_scope_sha256": "c" * 64,
            "task_feature_registry_sha256": "d" * 64,
            "policy": {
                "random_sample_fraction": 0.5,
                "non_critical_accuracy_gate": 0.95,
            },
            "policy_sha256": "e" * 64,
            "tasks_in_scope": 2,
            "task_ids_sha256": "f" * 64,
            "truth_packet_sha256_by_task": {"T1": "a" * 64, "T2": "b" * 64},
            "score_content_sha256_by_block": {"B1": "1" * 64, "B2": "2" * 64},
            "review_contract": {
                "version": "correction_free_all_dimensions_v1",
                "required_boolean_checks": [
                    "critical_truth_atoms_correct",
                    "critical_source_edges_correct",
                    "noncritical_truth_atoms_correct",
                    "noncritical_source_edges_correct",
                    "relation_judgments_correct",
                    "critical_harm_complete",
                ],
                "correction_summary_required_on_error": True,
                "any_error_requires_new_truth_and_all_arm_rescore": True,
                "corrected_truth_rescore_acceptance_supported": False,
            },
            "items": items,
        }
    )
    (directory / "AUDIT_QUEUE.json").write_text(__import__("json").dumps(queue), encoding="utf-8")
    return queue


class _Settings:
    def __init__(self, root):
        self.root = root

    def path(self, name):
        assert name == "judgments"
        return self.root / "judgments"


_CHECKS = (
    "critical_truth_atoms_correct",
    "critical_source_edges_correct",
    "noncritical_truth_atoms_correct",
    "noncritical_source_edges_correct",
    "relation_judgments_correct",
    "critical_harm_complete",
)


def _results(queue, *, false_check=None):
    def row(item_id):
        checks = {name: name != false_check for name in _CHECKS}
        return {
            "item_id": item_id,
            **checks,
            "correction_summary": (
                f"Reviewer found an error in {false_check}." if false_check else ""
            ),
        }

    return _seal(
        {
            "schema_version": "human_audit_results_v2",
            "queue_content_sha256": queue["content_sha256"],
            "reviewer_id": "reviewer-1",
            "reviewed_at_utc": "2026-07-25T18:00:00Z",
            "reviewer_attestation": "I reviewed every referenced packet and score.",
            "items": [row("i1"), row("i2")],
        }
    )


def test_ready_receipt_binds_queue_results_truth_and_scores(tmp_path):
    queue = _queue(tmp_path)
    receipt = finalize_human_audit(
        _Settings(tmp_path),
        run_id="RUN1",
        phase_id="PHASE1",
        results=_results(queue),
    )
    assert receipt["status"] == "AUDITED"
    assert receipt["evaluation_scope_sha256"] == "c" * 64
    assert receipt["truth_packet_sha256_by_task"] == {"T1": "a" * 64, "T2": "b" * 64}
    assert (
        tmp_path / "judgments" / "RUN1" / "PHASE1" / "human_audit" / "AUDITED_RECEIPT.json"
    ).exists()
    loaded = load_human_audit_receipt(
        _Settings(tmp_path),
        run_id="RUN1",
        phase_id="PHASE1",
        evaluation_scope_sha256="c" * 64,
        execution_binding_sha256="3" * 64,
        protocol_document_sha256="4" * 64,
        truth_packet_sha256_by_task={"T1": "a" * 64, "T2": "b" * 64},
        score_content_sha256_by_block={"B1": "1" * 64, "B2": "2" * 64},
    )
    assert loaded == receipt
    assert loaded["schema_version"] == "human_audit_receipt_v2"
    assert loaded["correction_count"] == 0
    assert loaded["audit_basis"] == "ORIGINAL_ARTIFACTS_CORRECTION_FREE"


@pytest.mark.parametrize("false_check", _CHECKS)
def test_any_reviewer_error_requires_correction_and_never_audits(tmp_path, false_check):
    """A statistical tolerance does not repair an incorrect truth edge or score."""
    queue = _queue(tmp_path)
    receipt = finalize_human_audit(
        _Settings(tmp_path),
        run_id="RUN1",
        phase_id="PHASE1",
        results=_results(queue, false_check=false_check),
    )
    assert receipt["status"] == "CORRECTIONS_REQUIRED"
    assert receipt["correction_count"] == 2
    assert receipt["items_requiring_correction"] == 2
    assert receipt["correction_resolution"] == "REQUIRES_NEW_TRUTH_AND_ALL_ARM_RESCORE"
    assert receipt["corrected_truth_packet_sha256_by_task"] is None
    assert receipt["all_arm_rescore_manifest_sha256"] is None
    assert not (
        tmp_path / "judgments" / "RUN1" / "PHASE1" / "human_audit" / "AUDITED_RECEIPT.json"
    ).exists()


def test_partial_or_retargeted_results_are_rejected(tmp_path):
    queue = _queue(tmp_path)
    partial = _results(queue)
    partial["items"].pop()
    _seal({key: value for key, value in partial.items() if key != "content_sha256"})
    # Re-seal the actual object after mutation.
    partial["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in partial.items() if key != "content_sha256"})
    )
    with pytest.raises(ValueError, match="exact queue"):
        finalize_human_audit(_Settings(tmp_path), run_id="RUN1", phase_id="PHASE1", results=partial)

    retargeted = deepcopy(_results(queue))
    retargeted["queue_content_sha256"] = "0" * 64
    retargeted["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in retargeted.items() if key != "content_sha256"})
    )
    with pytest.raises(ValueError, match="another queue"):
        finalize_human_audit(
            _Settings(tmp_path), run_id="RUN1", phase_id="PHASE1", results=retargeted
        )


def test_error_needs_an_explicit_correction_summary(tmp_path):
    queue = _queue(tmp_path)
    results = _results(queue, false_check="noncritical_source_edges_correct")
    results["items"][0]["correction_summary"] = ""
    results["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in results.items() if key != "content_sha256"})
    )
    with pytest.raises(ValueError, match="no correction summary"):
        finalize_human_audit(_Settings(tmp_path), run_id="RUN1", phase_id="PHASE1", results=results)


def test_discovered_correction_cannot_be_overwritten_by_a_clean_resubmission(tmp_path):
    queue = _queue(tmp_path)
    settings = _Settings(tmp_path)
    first = finalize_human_audit(
        settings,
        run_id="RUN1",
        phase_id="PHASE1",
        results=_results(queue, false_check="relation_judgments_correct"),
    )
    assert first["status"] == "CORRECTIONS_REQUIRED"

    with pytest.raises(ValueError, match="different audit artifact"):
        finalize_human_audit(
            settings,
            run_id="RUN1",
            phase_id="PHASE1",
            results=_results(queue),
        )
