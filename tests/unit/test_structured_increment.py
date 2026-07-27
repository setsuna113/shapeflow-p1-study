"""The structured P1 attribution gate is executable, multiplicity controlled, and fail closed."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from shapeflow_p1.analysis.e2e_effects import (
    QUALITY_GUARD_DIRECTIONS,
    TRAJECTORY_ENDPOINT_DIRECTIONS,
)
from shapeflow_p1.analysis.matched import (
    _percentile_bootstrap_one_sided_p,
    build_matched_contrasts,
    resolve_arm_semantics,
)
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.config import load_config
from shapeflow_p1.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[2]
CONTROL_ARMS = (
    "H_MARKDOWN_ID",
    "H_CPU_CONTROL",
    "H_PROSE_CONTROL",
    "C_ID",
    "C_CPU_CONTROL",
    "C_PROSE_CONTROL",
    "H_PLUS_C",
    "H_PLUS_C_CPU_CONTROL",
    "H_PLUS_C_PROSE_CONTROL",
)


def _configs() -> tuple[dict, dict, dict]:
    decision, _ = load_config(REPO / "configs" / "decision.yaml")
    week1, _ = load_config(REPO / "configs" / "week1.yaml")
    variants, _ = load_config(REPO / "configs" / "variants.yaml")
    return decision, week1, variants


def _arm_variants(week1: dict, variants: dict) -> dict[str, dict]:
    variant_by_id = {item["variant_id"]: item for item in variants["variants"]}
    arm_by_id = {item["arm_id"]: item for item in week1["screen_arms"]["arms"]}
    fields = week1["matched_contrasts"]["executable_variant_fields"]
    requested = {
        str(pair[side])
        for pair in week1["matched_contrasts"]["pairs"]
        for side in ("left_arm_id", "right_arm_id")
    }
    result = {}
    for arm_id in sorted(requested):
        arm = arm_by_id[arm_id]
        result[arm_id] = resolve_arm_semantics(
            arm_id,
            arm,
            variant_by_id,
            fields,
        )
    return result


def _receipt(node: str, digest: str) -> dict:
    inputs = [
        {
            "candidate_view_sha256": digest,
            "offered_span_ids": ["s1", "s2"],
            "offered_source_occurrence_ids": ["occ-1"],
        },
        {
            "candidate_view_sha256": sha256_hex(f"{node}:second".encode()),
            "offered_span_ids": ["s3"],
            "offered_source_occurrence_ids": ["occ-2"],
        },
    ]
    body = {
        "schema_version": "first_boundary_input_receipt_v1",
        "status": "OK",
        "node": node,
        "candidate_input_count": len(inputs),
        "candidate_inputs": inputs,
        "boundary_checkpoint_hash": sha256_hex(f"{node}:checkpoint".encode()),
        "first_event_index": 4,
        "event_kind": "NODE_SELECTION",
        "comparison_scope": "COMPLETE_ORDERED_FIRST_ATOMIC_H_OR_C_BOUNDARY_INPUT_SET",
        "later_boundary_policy": "MEDIATED_E2E_OUTCOME_NOT_PAIRING_FILTER",
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _scope() -> dict:
    return {
        "freeze_root_sha256": "a" * 64,
        "schedule_sha256": "b" * 64,
        "block_freeze_sha256": "c" * 64,
        "block_digest": "d" * 64,
        "valid_for_paired_estimate": True,
        "invalid_reason": "",
        "engine_epochs": ["epoch-1"],
        "engine_epoch_by_arm": {arm_id: "epoch-1" for arm_id in CONTROL_ARMS},
    }


def _bind_scope(records: list[dict]) -> dict:
    scores = []
    for record in records:
        record.update(
            {
                "run_id": "run-structured",
                "phase_id": "screen",
                "execution_binding_sha256": "e" * 64,
                "protocol_document_sha256": "f" * 64,
            }
        )
        record["content_sha256"] = sha256_hex(
            canonical_json(
                {key: value for key, value in record.items() if key != "content_sha256"}
            )
        )
        first = next(iter(record["per_arm"].values()))["frozen_scope"]
        scores.append(
            {
                "block_id": record["block_id"],
                "task_id": record["task_id"],
                "replicate_id": record["replicate_id"],
                "score_content_sha256": record["content_sha256"],
                "execution_binding_sha256": "e" * 64,
                "protocol_document_sha256": "f" * 64,
                "block_freeze_sha256": first["block_freeze_sha256"],
                "block_digest": first["block_digest"],
                "valid_for_paired_estimate": True,
                "invalid_reason": "",
                "engine_epochs": ["epoch-1"],
                "engine_epoch_by_arm": {
                    arm_key: "epoch-1" for arm_key in record["per_arm"]
                },
            }
        )
    receipt = {
        "schema_version": "evaluated_itt_scope_v1",
        "run_id": "run-structured",
        "phase_id": "screen",
        "execution_binding_sha256": "e" * 64,
        "protocol_document_sha256": "f" * 64,
        "schedule_sha256": "b" * 64,
        "freeze_root_sha256": "a" * 64,
        "analysis_design_receipt_sha256": "1" * 64,
        "task_feature_registry_sha256": "2" * 64,
        "eligibility_spec_content_sha256": "3" * 64,
        "all_offered_blocks": len(scores),
        "scores": scores,
    }
    receipt["evaluation_scope_sha256"] = sha256_hex(canonical_json(receipt))
    return receipt


def _arm_row(
    arm_id: str,
    *,
    variant_id: str,
    replicate_id: str,
    quality: float,
    work: float,
) -> dict:
    node = "H" if arm_id.startswith("H_") else "C"
    metrics = {
        metric: (0.01 if metric == "critical_harm" else quality)
        for metric in QUALITY_GUARD_DIRECTIONS
    }
    structured_nodes = (
        ("H", "C") if arm_id == "H_PLUS_C"
        else ("H",) if arm_id == "H_MARKDOWN_ID"
        else ("C",) if arm_id == "C_ID"
        else ()
    )
    prose_nodes = (
        ("H", "C") if arm_id == "H_PLUS_C_PROSE_CONTROL"
        else ("H",) if arm_id == "H_PROSE_CONTROL"
        else ("C",) if arm_id == "C_PROSE_CONTROL"
        else ()
    )
    row = {
        "arm_id": arm_id,
        "replicate_id": replicate_id,
        "variant_id": variant_id,
        "quality_views": {
            view: dict(metrics)
            for view in ("strict", "fallback_assisted", "worst_case", "best_case")
        },
        "work_summary": {
            "telemetry_complete": True,
            "overlap_valid": True,
            "service_seconds": work,
            # The primary work endpoint. Distinct from `service_seconds` in the fixture so a
            # test cannot pass by reading the one that only exists when the engine was
            # serialized.
            "interval_union_seconds": work * 0.8,
            "max_concurrent_treatment_requests": 3,
            "energy_joules": work * 250.0,
            "tokens": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "cached_prompt_tokens": 5,
            },
        },
        "e2e_latency_seconds": work,
        "trajectory_metrics": {
            "status": "OK",
            **{endpoint: 0.0 for endpoint in TRAJECTORY_ENDPOINT_DIRECTIONS},
        },
        "first_boundary_input": _receipt(
            node,
            "1" * 64 if node == "H" else "2" * 64,
        ),
        "frozen_scope": _scope(),
    }
    if structured_nodes:
        row["direct_node_metrics"] = {
            "status": "OK",
            "by_node": {
                direct_node: {
                    "applicable": True,
                    "selector_normalization": {
                        "status": "OK",
                        "no_repair_adverse": False,
                    },
                }
                for direct_node in structured_nodes
            },
        }
    if prose_nodes:
        row["prose_control_normalization"] = {
            "schema_version": "prose_control_normalization_summary_v1",
            "status": "OK",
            "expected_nodes": list(prose_nodes),
            "observed_nodes": list(prose_nodes),
            "attempt_count": len(prose_nodes),
            "published_count": len(prose_nodes),
            "rejected_count": 0,
            "fallback_discarded_count": 0,
            "call_failed_count": 0,
            "cancelled_count": 0,
            "raw_contract_adherent_count": len(prose_nodes),
            "truncated_attempt_count": 0,
            "semantic_repair_count": 0,
            "raw_contract_nonadherent_count": 0,
            "normalization_trace_error_count": 0,
            "trace_complete": True,
            "published_policy_adverse": False,
            "raw_contract_adverse": False,
            "unexpected_nodes": [],
            "by_node": {},
        }
    return row


def _study() -> tuple[list[dict], dict[str, str], dict, dict, dict]:
    decision, week1, variants = _configs()
    semantics = _arm_variants(week1, variants)
    arm_values = {
        "H_MARKDOWN_ID": (0.95, 8.0),
        "H_CPU_CONTROL": (0.70, 7.5),
        "H_PROSE_CONTROL": (0.94, 12.0),
        "C_ID": (0.95, 8.0),
        "C_CPU_CONTROL": (0.70, 7.5),
        "C_PROSE_CONTROL": (0.94, 12.0),
        "H_PLUS_C": (0.95, 8.0),
        "H_PLUS_C_CPU_CONTROL": (0.70, 7.5),
        "H_PLUS_C_PROSE_CONTROL": (0.94, 12.0),
    }
    records = []
    clusters = {}
    for index in range(4):
        task_id = f"task-{index}"
        replicate_id = "0"
        clusters[task_id] = f"cluster-{index}"
        records.append(
            {
                "block_id": f"block-{index}",
                "task_id": task_id,
                "replicate_id": replicate_id,
                "per_arm": {
                    f"{arm_id}:{replicate_id}": _arm_row(
                        arm_id,
                        variant_id=semantics[arm_id]["variant_id"],
                        replicate_id=replicate_id,
                        quality=arm_values[arm_id][0],
                        work=arm_values[arm_id][1],
                    )
                    for arm_id in CONTROL_ARMS
                },
            }
        )
    return records, clusters, decision, week1, semantics


def _run(
    records: list[dict],
    clusters: dict[str, str],
    decision: dict,
    week1: dict,
    semantics: dict,
):
    return build_matched_contrasts(
        records,
        scope_receipt=_bind_scope(records),
        cluster_by_task=clusters,
        matched_contrasts=week1["matched_contrasts"],
        arm_variants=semantics,
        structured_increment_policy=decision["structured_increment"],
        n_boot=399,
        seed=71,
    )


def test_six_member_holm_producer_can_establish_all_three_node_composites() -> None:
    records, clusters, decision, week1, semantics = _study()

    result = _run(records, clusters, decision, week1, semantics)
    gates = result["structured_increment_gates"]

    assert gates["policy_sha256"] == sha256_hex(
        canonical_json(decision["structured_increment"])
    )
    assert gates["family_status"] == "ESTIMABLE"
    assert set(gates["holm_order"]) == set(
        decision["structured_increment"]["multiplicity_family"]["member_contrast_ids"]
    )
    assert all(
        row["holm_rejected"] and row["adjusted_primary_gate_pass"]
        for row in gates["family_results"].values()
    )
    assert gates["by_node"]["WEBPAGE_P1"]["status"] == "ESTABLISHED"
    assert gates["by_node"]["C_VISIBLE"]["status"] == "ESTABLISHED"
    assert gates["by_node"]["H_PLUS_C_VISIBLE"]["status"] == "ESTABLISHED"
    assert all(
        component["first_boundary_input_comparability_pass"]
        for node in gates["by_node"].values()
        for component in node["component_gates"].values()
    )


def test_dirty_prose_raw_contract_keeps_policy_itt_but_blocks_pointer_only_claim() -> None:
    records, clusters, decision, week1, semantics = _study()
    dirty = records[0]["per_arm"]["H_PROSE_CONTROL:0"][
        "prose_control_normalization"
    ]
    dirty["raw_contract_adherent_count"] = 0
    dirty["raw_contract_nonadherent_count"] = 1
    dirty["raw_contract_adverse"] = True

    gates = _run(records, clusters, decision, week1, semantics)[
        "structured_increment_gates"
    ]
    component = gates["by_node"]["WEBPAGE_P1"]["component_gates"][
        "structured_selection_vs_prose"
    ]

    assert component["status"] == "ESTABLISHED"
    assert component["prose_control_integrity_gate_pass"] is True
    assert component["pointer_only_attribution_status"] == "NOT_ESTABLISHED"
    assert (
        component["raw_contract_work_sensitivity"]["status"]
        == "NOT_ESTIMABLE"
    )
    assert (
        component["raw_contract_quality_ni_guards"][
            "grounded_claim_precision"
        ]["status"]
        == "FAIL"
    )
    assert gates["by_node"]["WEBPAGE_P1"]["attribution_scope"] == (
        "LLM_PLUS_STRUCTURED_BOUNDED_POLICY_ONLY"
    )


def test_missing_joint_c_prose_trace_cannot_borrow_standalone_control_gates() -> None:
    records, clusters, decision, week1, semantics = _study()
    for record in records:
        summary = record["per_arm"]["H_PLUS_C_PROSE_CONTROL:0"][
            "prose_control_normalization"
        ]
        summary["status"] = "INVALID_NORMALIZATION_TRACE"
        summary["trace_complete"] = False
        summary["normalization_trace_error_count"] = 1
        summary["missing_expected_nodes"] = ["C"]

    gates = _run(records, clusters, decision, week1, semantics)[
        "structured_increment_gates"
    ]

    assert gates["by_node"]["WEBPAGE_P1"]["status"] == "ESTABLISHED"
    assert gates["by_node"]["C_VISIBLE"]["status"] == "ESTABLISHED"
    assert gates["by_node"]["H_PLUS_C_VISIBLE"]["status"] == "NOT_ESTABLISHED"
    joint = gates["by_node"]["H_PLUS_C_VISIBLE"]["component_gates"][
        "structured_selection_vs_prose"
    ]
    assert joint["prose_control_integrity_gate_pass"] is False


def test_one_sided_raw_p_inverts_the_reported_lcb_for_skewed_draws() -> None:
    # Work-saving draws can be strongly left-skewed after the log-ratio transform.  A
    # centered-null upper-tail test would return 0.9775 here even though the matching
    # percentile lower bound is positive.  The frozen gate must use one coherent family.
    samples = np.asarray([-10.0] * 9 + [0.4] * 390)
    alpha = 0.05
    threshold = 0.0

    raw_p = _percentile_bootstrap_one_sided_p(samples, threshold=threshold)
    lower = float(np.quantile(samples, alpha, method="lower"))

    assert raw_p == 0.025
    assert (raw_p <= alpha) is (lower > threshold)


def test_one_failed_holm_member_blocks_only_the_affected_compound() -> None:
    records, clusters, decision, week1, semantics = _study()
    for record in records:
        record["per_arm"]["C_CPU_CONTROL:0"]["quality_views"]["strict"][
            "weighted_required_atom_recall"
        ] = 0.96

    gates = _run(records, clusters, decision, week1, semantics)[
        "structured_increment_gates"
    ]

    assert gates["family_results"]["C_LLM_VS_CPU"]["holm_rejected"] is False
    assert gates["by_node"]["C_VISIBLE"]["status"] == "NOT_ESTABLISHED"
    assert gates["by_node"]["WEBPAGE_P1"]["status"] == "ESTABLISHED"


def test_missing_family_measurement_fails_closed_without_reusing_plain_ci() -> None:
    records, clusters, decision, week1, semantics = _study()
    del records[0]["per_arm"]["C_CPU_CONTROL:0"]["quality_views"]["strict"][
        "weighted_required_atom_recall"
    ]

    gates = _run(records, clusters, decision, week1, semantics)[
        "structured_increment_gates"
    ]

    assert gates["family_status"] == "NOT_ESTIMABLE_ONE_OR_MORE_MEMBERS"
    assert all(
        row["adjusted_primary_gate_pass"] is False
        for row in gates["family_results"].values()
    )
    assert gates["by_node"]["WEBPAGE_P1"]["status"] == "NOT_ESTABLISHED"
    assert gates["by_node"]["C_VISIBLE"]["status"] == "NOT_ESTABLISHED"


def test_second_page_mismatch_blocks_attribution_but_later_trajectory_is_irrelevant() -> None:
    records, clusters, decision, week1, semantics = _study()
    bad = records[0]["per_arm"]["H_CPU_CONTROL:0"]["first_boundary_input"]
    bad["candidate_inputs"][1]["candidate_view_sha256"] = "9" * 64
    bad["content_sha256"] = sha256_hex(
        canonical_json({key: value for key, value in bad.items() if key != "content_sha256"})
    )
    # Post-treatment trajectories are deliberately different and never enter comparability.
    records[1]["per_arm"]["H_MARKDOWN_ID:0"]["trajectory_metrics"]["query_count"] = 99

    result = _run(records, clusters, decision, week1, semantics)
    contrast = next(
        row for row in result["contrasts"] if row["contrast_id"] == "H_LLM_VS_CPU"
    )

    assert contrast["first_boundary_input_comparability"]["status"] == "NOT_ESTABLISHED"
    assert result["structured_increment_gates"]["by_node"]["WEBPAGE_P1"][
        "status"
    ] == "NOT_ESTABLISHED"
