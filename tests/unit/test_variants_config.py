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
    # Primary causal mode is single-in-flight, APC off, chunked prefill off.
    causal = iso["causal"]
    assert causal["max_num_seqs"] == 1
    assert causal["gateway_max_upstream_inflight"] == 1
    assert causal["enable_prefix_caching"] is False
    assert causal["enable_chunked_prefill"] is False
    # Operational mode turns native APC on and uses real batching.
    operational = iso["operational"]
    assert operational["enable_prefix_caching"] is True
    assert operational["enable_chunked_prefill"] is True
    assert operational["max_num_seqs"] > 1
    assert data["determinism"]["pythonhashseed"] == 0


def test_the_two_layers_have_separate_units_and_never_share_apc_state():
    """The layers must not collapse into one engine config.

    A single shared `engine.enable_prefix_caching` is what let the causal arm run with APC on
    while `configs/stack.yaml` claimed it was off. Each layer therefore names its own systemd
    unit and carries its own flags, and the top-level `engine` block must not carry either flag.
    """
    data, _ = load_config(CONFIGS / "stack.yaml")
    iso = data["isolation"]
    assert iso["causal"]["unit"] != iso["operational"]["unit"]
    assert iso["causal"]["enable_prefix_caching"] != iso["operational"]["enable_prefix_caching"]
    for leaked in ("enable_prefix_caching", "enable_chunked_prefill", "max_num_seqs"):
        assert leaked not in data["engine"], (
            f"engine.{leaked} is layer-specific and must live under isolation.<layer>"
        )
    # Cross-arm cache carry-over must be proved absent, not assumed (plan 5.5).
    assert iso["operational"]["cache_salt_probe_required"] is True


def test_model_identity_pins_both_the_revision_and_the_artifact():
    """The model dir is not a git checkout, but a revision IS recoverable -- pin both.

    Verified read-only on sjtu 2026-07-24: /storage/nvme/reme/models/Qwen3-14B-AWQ has no .git,
    yet all 11 .cache/huggingface/download/*.metadata files record commit 31c69efc. Dropping
    the revision because there is no git ref would discard a real, checkable identity; pinning
    only the revision would miss on-disk corruption. Doctor requires both.
    """
    data, _ = load_config(CONFIGS / "stack.yaml")
    model = data["model"]
    assert model["revision"] == "31c69efc29464b6bb0aee1398b5a7b50a99340c3"
    assert model["revision_source"] == "hf_download_metadata"
    assert "artifact_merkle_root" in model


def test_both_vllm_version_strings_are_pinned():
    """`vllm.__version__` and the distribution version disagree, so both are recorded.

    Observed on sjtu: distribution "0.24.0+cu129", vllm.__version__ "0.24.0". Checking against
    one alone silently accepts a stack whose other half changed.
    """
    data, _ = load_config(CONFIGS / "stack.yaml")
    engine = data["engine"]
    assert engine["vllm_distribution_version"] == "0.24.0+cu129"
    assert engine["vllm_dunder_version"] == "0.24.0"
    assert "vllm_package_tree_sha256" in engine


def test_driver_cuda_capability_is_not_conflated_with_the_torch_runtime():
    """nvidia-smi reports 12.4 (driver capability); torch is built against 12.9. Both, apart."""
    data, _ = load_config(CONFIGS / "stack.yaml")
    assert data["host"]["driver_cuda_capability"] == "12.4"
    assert data["engine"]["torch_compiled_cuda"] == "12.9"


def test_frozen_stack_values_are_stewarded_not_self_frozen_at_launch():
    """Placeholders say STEWARD, not DOCTOR, and that distinction is the point.

    A campaign doctor that froze whatever it observed would launder an already-drifted
    environment into a legitimate baseline. The manifest is produced by an explicit
    `freeze-stack` run before approval; the campaign's doctor only ever compares.
    """
    text = (CONFIGS / "stack.yaml").read_text(encoding="utf-8")
    assert "@DOCTOR_FREEZES@" not in text
    assert "@STEWARD_FREEZES@" in text
