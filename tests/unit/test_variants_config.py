"""The variant registry and stack config load and are structurally sound."""

from __future__ import annotations

from pathlib import Path

from shapeflow_p1.config import load_config

CONFIGS = Path(__file__).resolve().parents[2] / "configs"


def test_variants_load_and_have_required_fields():
    data, sha = load_config(CONFIGS / "variants.yaml")
    variants = data["variants"]
    assert len(sha) == 64
    required = {"variant_id", "node", "chunker", "scope", "contract", "aggregation", "close_mode"}
    for v in variants:
        assert required <= set(v), f"{v.get('variant_id')} missing fields"


def test_registry_has_the_expected_nodes_and_controls():
    data, _ = load_config(CONFIGS / "variants.yaml")
    variants = {v["variant_id"]: v for v in data["variants"]}
    # P0 and the CPU/PROSE variants are controls.
    assert variants["P0"]["is_control"] is True
    assert variants["H00-CPU"]["is_control"] is True
    assert variants["C00-PROSE"]["is_control"] is True
    # The real candidates are not controls.
    assert variants["H03"].get("is_control", False) is False
    assert variants["H03"]["contract"] == "P1_TYPED"
    # The fused and registry extensions carry their own distinct nodes.
    assert variants["C05-FUSED-EXT"]["node"] == "C_FUSED_EXT"
    assert variants["C06-REG"]["node"] == "C_REGISTRY"
    # C_VISIBLE (compressor-only) is distinct from those extensions.
    assert variants["C02"]["node"] == "C_VISIBLE"


def test_stack_config_freezes_isolation_modes():
    data, _ = load_config(CONFIGS / "stack.yaml")
    iso = data["isolation"]
    # Primary causal mode is single-in-flight, APC off.
    assert iso["causal_max_num_seqs"] == 1
    assert iso["causal_gateway_max_upstream_inflight"] == 1
    assert iso["causal_apc"] is False
    # Operational mode turns APC on.
    assert iso["operational_apc"] is True
    assert data["determinism"]["pythonhashseed"] == 0
