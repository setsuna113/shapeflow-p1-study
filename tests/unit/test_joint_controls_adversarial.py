"""Adversarial checks for the preregistered joint controls and 19-arm schedule.

These tests deliberately exercise the seams most likely to yield an attractive but invalid
H+C claim: collapsing the two nodes into one scalar variant, silently accepting a partial or
node-swapped joint arm, borrowing standalone gates for the joint mechanism, accepting stale
semantic provenance, or relying on one lucky randomization binding.
"""

from __future__ import annotations

import copy
import hashlib
from collections import Counter
from pathlib import Path

import pytest

from shapeflow_p1.analysis.finalize import _structured_increment_gates
from shapeflow_p1.analysis.matched import resolve_arm_semantics
from shapeflow_p1.campaign.schedule import ArmSpec, build_blocks
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.config import load_config
from shapeflow_p1.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[2]
JOINT_CONTROL_IDS = {
    "HC_LLM_VS_CPU",
    "HC_ID_VS_PROSE",
}
NODES = ("WEBPAGE_P1", "C_VISIBLE", "H_PLUS_C_VISIBLE")


def _configs() -> tuple[dict, dict, dict]:
    decision, _ = load_config(REPO / "configs" / "decision.yaml")
    week1, _ = load_config(REPO / "configs" / "week1.yaml")
    variants, _ = load_config(REPO / "configs" / "variants.yaml")
    return decision, week1, variants


def _registries(
    week1: dict,
    variants_config: dict,
) -> tuple[dict[str, dict], dict[str, dict], tuple[str, ...]]:
    arms = {
        str(row["arm_id"]): row
        for row in week1["screen_arms"]["arms"]
    }
    variants = {
        str(row["variant_id"]): row
        for row in variants_config["variants"]
    }
    fields = tuple(map(str, week1["matched_contrasts"]["executable_variant_fields"]))
    return arms, variants, fields


def _resolved(
    arm_id: str,
    *,
    week1: dict,
    variants_config: dict,
) -> dict:
    arms, variants, fields = _registries(week1, variants_config)
    return resolve_arm_semantics(arm_id, arms[arm_id], variants, fields)


def _field_differences(
    left: dict,
    right: dict,
    executable_fields: tuple[str, ...],
) -> set[str]:
    return {
        field
        for field in executable_fields
        if left.get(field) != right.get(field)
    }


def test_joint_cpu_control_changes_only_both_selector_backends() -> None:
    _, week1, variants_config = _configs()
    _, _, fields = _registries(week1, variants_config)
    treatment = _resolved("H_PLUS_C", week1=week1, variants_config=variants_config)
    cpu = _resolved(
        "H_PLUS_C_CPU_CONTROL",
        week1=week1,
        variants_config=variants_config,
    )

    assert _field_differences(treatment, cpu, fields) == {"selector_backend"}
    assert treatment["selector_backend"] == {"page": "LLM", "close": "LLM"}
    assert cpu["selector_backend"] == {
        "page": "CPU_LEXICAL",
        "close": "CPU_LEXICAL",
    }
    assert all(
        isinstance(treatment[field], dict)
        and set(treatment[field]) == {"page", "close"}
        for field in fields
    )


def test_joint_prose_control_changes_only_declared_structured_output_fields() -> None:
    _, week1, variants_config = _configs()
    _, _, fields = _registries(week1, variants_config)
    treatment = _resolved("H_PLUS_C", week1=week1, variants_config=variants_config)
    prose = _resolved(
        "H_PLUS_C_PROSE_CONTROL",
        week1=week1,
        variants_config=variants_config,
    )

    expected = {"contract", "publication_path", "output_representation"}
    assert _field_differences(treatment, prose, fields) == expected
    assert treatment["selector_backend"] == prose["selector_backend"] == {
        "page": "LLM",
        "close": "LLM",
    }
    pairs = {
        str(row["contrast_id"]): row
        for row in week1["matched_contrasts"]["pairs"]
    }
    assert set(pairs["HC_ID_VS_PROSE"]["factor_fields"]) == expected


def test_joint_resolver_rejects_node_swaps_and_incomplete_identity() -> None:
    _, week1, variants_config = _configs()
    _, variants, fields = _registries(week1, variants_config)

    with pytest.raises(ValueError, match="wrong page node"):
        resolve_arm_semantics(
            "FAKE_JOINT",
            {"page_variant": "C01", "close_variant": "H02"},
            variants,
            fields,
        )
    with pytest.raises(ValueError, match="lacks page/close variant identity"):
        resolve_arm_semantics(
            "FAKE_JOINT",
            {"page_variant": "H02"},
            variants,
            fields,
        )
    with pytest.raises(ValueError, match="unknown close variant"):
        resolve_arm_semantics(
            "FAKE_JOINT",
            {"page_variant": "H02", "close_variant": "DOES_NOT_EXIST"},
            variants,
            fields,
        )


def _valid_finalizer_inputs() -> tuple[dict, dict, dict, dict, dict]:
    decision, week1, variants_config = _configs()
    arms, variants, fields = _registries(week1, variants_config)
    design = week1["matched_contrasts"]
    requested_arms = {
        str(row[side])
        for row in design["pairs"]
        for side in ("left_arm_id", "right_arm_id")
    }
    resolved = {
        arm_id: resolve_arm_semantics(
            arm_id,
            arms[arm_id],
            variants,
            fields,
        )
        for arm_id in sorted(requested_arms)
    }
    policy = decision["structured_increment"]
    estimands = policy["control_estimands"]

    def component(control_type: str, contrast_id: str) -> dict:
        return {
            "status": "ESTABLISHED",
            "contrast_id": contrast_id,
            "holm_family_id": policy["multiplicity_family"]["family_id"],
            "estimand_contract_sha256": sha256_hex(
                canonical_json(estimands[control_type])
            ),
            "adjusted_primary_gate_pass": True,
            "all_strict_quality_ni_guards_pass": True,
            "first_boundary_input_comparability_pass": True,
            **(
                {"service_work_cost_guard_pass": True}
                if control_type == "LLM_vs_CPU"
                else {
                    "prose_control_integrity_gate_pass": True,
                    "main_estimand_scope":
                        "STRUCTURED_ID_POLICY_VS_BOUNDED_SHORT_PROSE_POLICY",
                    "pointer_only_attribution_status": "ESTABLISHED",
                    "raw_contract_quality_ni_guards": {},
                    "raw_contract_work_sensitivity": {"status": "OK"},
                }
            ),
        }

    contrasts_by_id = {
        str(row["contrast_id"]): {
            **copy.deepcopy(row),
            "pairing_status": "OK",
            "semantic_validation": {"status": "OK"},
        }
        for row in design["pairs"]
    }
    by_node: dict[str, dict] = {}
    for node, node_policy in policy["by_node"].items():
        required = list(map(str, node_policy["required_contrast_ids"]))
        by_kind = {
            str(contrasts_by_id[contrast_id]["kind"]): contrast_id
            for contrast_id in required
        }
        by_node[node] = {
            "status": "ESTABLISHED",
            "primary_arm_id": str(node_policy["primary_arm_id"]),
            "required_contrast_ids": required,
            "compound_status": "ESTABLISHED",
            "pointer_only_attribution_status": "ESTABLISHED",
            "attribution_scope": "LLM_PLUS_STRUCTURED_POINTER_REPRESENTATION",
            "component_gates": {
                control_type: component(control_type, contrast_id)
                for control_type, contrast_id in by_kind.items()
            },
        }

    family_ids = set(map(str, policy["multiplicity_family"]["member_contrast_ids"]))
    matched = {
        "input_provenance": {
            "matched_contrasts_sha256": sha256_hex(canonical_json(design)),
            "arm_variants_sha256": sha256_hex(canonical_json(resolved)),
        },
        "contrasts": [
            contrasts_by_id[contrast_id]
            for contrast_id in sorted(family_ids)
        ],
        "structured_increment_gates": {
            "schema_version": policy["schema_version"],
            "policy_sha256": sha256_hex(canonical_json(policy)),
            "multiplicity_family": copy.deepcopy(policy["multiplicity_family"]),
            "by_node": by_node,
        },
    }
    return matched, decision, design, variants, arms


def _consume_gates(
    matched: dict,
    decision: dict,
    design: dict,
    variants: dict,
    arms: dict,
) -> dict[str, dict]:
    return _structured_increment_gates(
        matched,
        decision_config=decision,
        primary_arm_by_node={
            str(node): str(policy["primary_arm_id"])
            for node, policy in decision["structured_increment"]["by_node"].items()
        },
        matched_design=design,
        variants=variants,
        arms=arms,
    )


def test_standalone_gates_cannot_substitute_for_a_missing_joint_gate() -> None:
    matched, decision, design, variants, arms = _valid_finalizer_inputs()
    del matched["structured_increment_gates"]["by_node"]["H_PLUS_C_VISIBLE"]

    gates = _consume_gates(matched, decision, design, variants, arms)

    assert gates["WEBPAGE_P1"]["status"] == "ESTABLISHED"
    assert gates["C_VISIBLE"]["status"] == "ESTABLISHED"
    assert gates["H_PLUS_C_VISIBLE"]["status"] == "NOT_ESTABLISHED"
    assert gates["H_PLUS_C_VISIBLE"]["reason"] == "STRUCTURED_INCREMENT_NODE_GATE_MISSING"


def test_tampered_joint_contrast_semantics_fail_only_joint_attribution() -> None:
    matched, decision, design, variants, arms = _valid_finalizer_inputs()
    joint = next(
        row
        for row in matched["contrasts"]
        if row["contrast_id"] == "HC_LLM_VS_CPU"
    )
    joint["affected_nodes"] = ["C_VISIBLE", "WEBPAGE_P1"]

    gates = _consume_gates(matched, decision, design, variants, arms)

    assert gates["WEBPAGE_P1"]["status"] == "ESTABLISHED"
    assert gates["C_VISIBLE"]["status"] == "ESTABLISHED"
    assert gates["H_PLUS_C_VISIBLE"]["status"] == "NOT_ESTABLISHED"


def test_tampered_joint_semantic_registry_hash_fails_closed_for_all_nodes() -> None:
    matched, decision, design, variants, arms = _valid_finalizer_inputs()
    matched["input_provenance"]["arm_variants_sha256"] = "0" * 64

    gates = _consume_gates(matched, decision, design, variants, arms)

    assert set(gates) == set(NODES)
    assert all(row["status"] == "NOT_ESTABLISHED" for row in gates.values())
    assert all(
        row["reason"]
        == "MULTIPLICITY_CONTROLLED_STRUCTURED_INCREMENT_COMPOSITE_NOT_EMITTED"
        for row in gates.values()
    )


def _real_arms(week1: dict) -> list[ArmSpec]:
    return [
        ArmSpec(
            str(row["arm_id"]),
            str(row["page_variant"]),
            str(row["close_variant"]),
        )
        for row in week1["screen_arms"]["arms"]
    ]


def _manifest(
    *,
    binding: str,
    split: str,
    task_count: int,
    arms: list[ArmSpec],
    second_seed_fraction: float,
):
    return build_blocks(
        execution_binding_sha256=binding,
        protocol_sha="a" * 64,
        split=split,
        task_ids=[f"{split}-T{index:03d}" for index in range(task_count)],
        arms=arms,
        seeds=[1, 2],
        layer="causal",
        claim_scope="FORMATIVE_ONLY",
        second_seed_fraction=second_seed_fraction,
    )


def test_real_19_arm_schedule_is_balanced_and_unique_across_many_bindings() -> None:
    _, week1, _ = _configs()
    arms = _real_arms(week1)
    arm_ids = {arm.arm_id for arm in arms}
    bindings = [
        "e" * 64,
        *[
            hashlib.sha256(f"joint-schedule-binding-{index}".encode()).hexdigest()
            for index in range(64)
        ],
    ]
    assignment_hashes: set[str] = set()

    assert len(arms) == 19
    for binding in bindings:
        screen = _manifest(
            binding=binding,
            split="FORMATIVE_SCREEN",
            task_count=32,
            arms=arms,
            second_seed_fraction=0.25,
        )
        canary = _manifest(
            binding=binding,
            split="CANARY",
            task_count=4,
            arms=arms,
            second_seed_fraction=0.0,
        )
        assignment_hashes.add(str(screen.notes["randomization_assignment_sha256"]))

        assert len(screen.blocks) == 40
        assert len(screen.cells) == 760
        assert len(canary.blocks) == 4
        assert len(canary.cells) == 76
        assert len(screen.cells) + len(canary.cells) == 836
        assert screen.notes["williams_design_rows"] == 38
        assert screen.notes["williams_full_cycles"] == 1
        assert screen.notes["williams_remainder_rows"] == 2
        assert (
            screen.notes["williams_remainder_policy"]
            == "odd_translated_cyclic_window_carryover_balanced"
        )

        coordinates = [
            (manifest.split, cell.task_id, cell.replicate_id, cell.arm.arm_id)
            for manifest in (screen, canary)
            for cell in manifest.cells
        ]
        assert len(coordinates) == len(set(coordinates)) == 836
        for manifest in (screen, canary):
            for block in manifest.blocks:
                ordered = sorted(block.cells, key=lambda cell: cell.order_index)
                assert len(ordered) == 19
                assert {cell.arm.arm_id for cell in ordered} == arm_ids
                assert [cell.order_index for cell in ordered] == list(range(19))

        orders = [
            tuple(
                cell.arm.arm_id
                for cell in sorted(block.cells, key=lambda cell: cell.order_index)
            )
            for block in screen.blocks
        ]
        for position in range(19):
            counts = Counter(order[position] for order in orders)
            assert set(counts) == arm_ids
            assert set(counts.values()) == {2, 3}
            assert max(counts.values()) - min(counts.values()) == 1

        adjacency = Counter(
            (left, right)
            for order in orders
            for left, right in zip(order, order[1:], strict=False)
        )
        assert set(adjacency) == {
            (left, right)
            for left in arm_ids
            for right in arm_ids
            if left != right
        }
        assert set(adjacency.values()) == {2, 3}
        assert max(adjacency.values()) - min(adjacency.values()) == 1

    assert len(assignment_hashes) > 1
