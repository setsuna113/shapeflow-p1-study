from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from shapeflow_p1 import cli
from shapeflow_p1.analysis.design import build_eligibility_spec
from shapeflow_p1.analysis.e2e_effects import task_feature_registry_sha256
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.hashing import sha256_hex

REPO = Path(__file__).resolve().parents[2]


def _seal(body: dict) -> dict:
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _write_design(settings: Settings) -> tuple[dict, dict, dict]:
    record = _seal({
        "schema_version": "frozen_task_features_v1",
        "task_id": "task-1",
        "split": "FORMATIVE_SCREEN",
        "cluster_id": "cluster-1",
        "features": {"candidate_evidence_tokens": 42.0},
    })
    records = {"task-1": record}
    registry = _seal({
        "schema_version": "frozen_task_feature_registry_v1",
        "records": records,
        "task_feature_registry_sha256": task_feature_registry_sha256(records),
    })
    spec = build_eligibility_spec(settings, registry)
    directory = settings.path("evaluator_root") / "analysis_design"
    directory.mkdir(parents=True)
    (directory / "TASK_FEATURE_REGISTRY.json").write_text(
        json.dumps(registry), encoding="utf-8")
    (directory / "ELIGIBILITY_SPEC.json").write_text(
        json.dumps(spec), encoding="utf-8")
    receipt = _seal({
        "schema_version": "pre_treatment_analysis_design_receipt_v1",
        "task_feature_registry_sha256":
            registry["task_feature_registry_sha256"],
        "feature_registry_content_sha256": registry["content_sha256"],
        "eligibility_spec_content_sha256": spec["content_sha256"],
        "decision_config_sha256": settings.shas["decision"],
        "variants_config_sha256": settings.shas["variants"],
        "week1_config_sha256": settings.shas["week1"],
    })
    (directory / "ANALYSIS_DESIGN_RECEIPT.json").write_text(
        json.dumps(receipt), encoding="utf-8")
    scope = {
        "schema_version": "evaluated_itt_scope_v1",
        "run_id": "run-1",
        "phase_id": "phase-1",
        "execution_binding_sha256": "1" * 64,
        "protocol_document_sha256": "2" * 64,
        "evaluation_scope_sha256": "e" * 64,
        "analysis_design_receipt_sha256": receipt["content_sha256"],
        "task_feature_registry_sha256":
            registry["task_feature_registry_sha256"],
        "eligibility_spec_content_sha256": spec["content_sha256"],
    }
    return registry, spec, scope


def test_analysis_approval_is_a_second_binding_not_the_treatment_binding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import shapeflow_p1.protocol as protocol_module

    calls: list[tuple[Path, dict]] = []
    live = SimpleNamespace(
        digest="8" * 64,
        protocol_sha="2" * 64,
        approved_commit="1" * 40,
    )

    def fake_verified(repo: Path, **kwargs):
        calls.append((repo, kwargs))
        return live

    monkeypatch.setattr(protocol_module, "verified_execution_binding", fake_verified)
    observed = cli._verified_analysis_binding(
        execution_binding_sha256="1" * 64,
        protocol_document_sha256="2" * 64,
    )

    assert observed is live
    assert calls == [(REPO, {})]
    assert observed.digest != "1" * 64


def test_semantic_mappings_are_derived_from_frozen_configs(tmp_path: Path) -> None:
    settings = Settings.load(REPO, data_root=tmp_path)

    arm_map, expected, matched = cli._e2e_semantic_mappings(settings)

    assert arm_map == {
        "p0": "P0",
        "h": "H_MARKDOWN_ID",
        "c": "C_ID",
        "hc": "H_PLUS_C",
    }
    assert expected == {
        "p0": "P0+P0",
        "h": "H02+P0",
        "c": "P0+C01",
        "hc": "H02+C01",
    }
    assert matched["H_MARKDOWN_ID"] == {
        "variant_id": "H02+P0",
        "node": "WEBPAGE_P1",
        "chunker": "markdown_structure_v1",
        "scope": "per_page",
        "contract": "P1_ID",
            "aggregation": "stable_union_v1",
            "close_mode": "separate",
            "selector_backend": "LLM",
            "publication_path": "STRUCTURED_SELECTION",
            "output_representation": "EVIDENCE_IDS",
            "bridge_token_cap_each": None,
        "bridge_token_cap_total": None,
    }
    assert matched["C_ID"]["variant_id"] == "P0+C01"
    assert matched["C_ID"]["node"] == "C_VISIBLE"


def test_analysis_design_loader_rejects_scope_bound_spec_swap(
    tmp_path: Path,
) -> None:
    settings = Settings.load(REPO, data_root=tmp_path)
    _, spec, scope = _write_design(settings)
    records, loaded_spec, bindings = cli._load_e2e_analysis_design(
        settings, scope)
    assert set(records) == {"task-1"}
    assert loaded_spec == spec
    assert bindings["analysis_design_receipt_sha256"] == (
        scope["analysis_design_receipt_sha256"])

    swapped = _seal({
        key: value for key, value in spec.items() if key != "content_sha256"
    } | {"quality_view": "fallback_assisted"})
    path = (
        settings.path("evaluator_root") / "analysis_design"
        / "ELIGIBILITY_SPEC.json"
    )
    path.write_text(json.dumps(swapped), encoding="utf-8")
    with pytest.raises(
        ValueError, match="differs from the hash-locked decision design"
    ):
        cli._load_e2e_analysis_design(settings, scope)


def test_analysis_design_loader_rejects_record_and_receipt_digest_tampering(
    tmp_path: Path,
) -> None:
    settings = Settings.load(REPO, data_root=tmp_path)
    registry, _, scope = _write_design(settings)
    registry["records"]["task-1"]["features"]["candidate_evidence_tokens"] = 99
    registry["content_sha256"] = cli._unsigned_content_sha256(registry)
    path = (
        settings.path("evaluator_root") / "analysis_design"
        / "TASK_FEATURE_REGISTRY.json"
    )
    path.write_text(json.dumps(registry), encoding="utf-8")
    with pytest.raises(ValueError, match="record 'task-1' does not verify"):
        cli._load_e2e_analysis_design(settings, scope)

    _write_clean_root = tmp_path / "second"
    second = Settings.load(REPO, data_root=_write_clean_root)
    _, _, second_scope = _write_design(second)
    second_scope["analysis_design_receipt_sha256"] = "G" * 64
    with pytest.raises(ValueError, match="different analysis-design receipt"):
        cli._load_e2e_analysis_design(second, second_scope)


def test_e2e_analysis_output_is_write_once_and_idempotent(
    tmp_path: Path,
) -> None:
    path = tmp_path / "analysis" / "E2E_ANALYSIS.json"
    body = _seal({"schema_version": "e2e_analysis_bundle_v1", "value": 1})
    assert cli._write_e2e_analysis_once(path, body) == "CREATED"
    assert cli._write_e2e_analysis_once(path, body) == "EXISTING_IDENTICAL"

    changed = _seal({"schema_version": "e2e_analysis_bundle_v1", "value": 2})
    with pytest.raises(ValueError, match="different E2E analysis"):
        cli._write_e2e_analysis_once(path, changed)


def test_analyze_e2e_wires_verified_inputs_and_controlled_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shapeflow_p1.analysis import e2e_effects, estimands, matched

    settings = Settings.load(REPO, data_root=tmp_path)
    _, _, scope = _write_design(settings)

    class Records(list):
        scope_receipt = scope

    records = Records([{
        "task_id": "task-1",
        "block_id": "block-1",
        "run_id": "run-1",
        "phase_id": "phase-1",
        "truth_packet_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }])
    calls: dict[str, object] = {}

    def fake_load(_root, *, run_id, phase_id):
        assert (run_id, phase_id) == ("run-1", "phase-1")
        return records

    def fake_e2e(passed_records, **kwargs):
        assert passed_records is records
        assert kwargs["arm_map"]["h"] == "H_MARKDOWN_ID"
        assert kwargs["expected_variant_ids"]["hc"] == "H02+C01"
        assert kwargs["scope_receipt"] is scope
        assert kwargs["task_features"]["task-1"]["cluster_id"] == "cluster-1"
        calls["e2e"] = kwargs
        return _seal({
            "schema_version": "e2e_effects_v2",
            "all_offered_blocks": 1,
            "tasks": 1,
            "clusters": 1,
            "eligibility": {"status": "OK"},
        })

    def fake_matched(passed_records, **kwargs):
        assert passed_records is records
        assert kwargs["cluster_by_task"] == {"task-1": "cluster-1"}
        assert kwargs["arm_variants"]["H_MARKDOWN_ID"]["variant_id"] == "H02+P0"
        assert kwargs["arm_variants"]["C_ID"]["variant_id"] == "P0+C01"
        assert kwargs["structured_increment_policy"] == settings.get(
            "decision", "structured_increment"
        )
        calls["matched"] = kwargs
        return _seal({
            "schema_version": "matched_contrast_analysis_v2",
            "contrasts": [{"contrast_id": "H_ID_VS_TYPED"}],
        })

    monkeypatch.setattr(cli, "_require_role", lambda role: None)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "_verified_analysis_binding",
        lambda **_kwargs: SimpleNamespace(
            digest="8" * 64,
            protocol_sha=scope["protocol_document_sha256"],
            approved_commit="1" * 40,
        ),
    )
    monkeypatch.setattr(estimands, "load_scoped_scores", fake_load)
    monkeypatch.setattr(e2e_effects, "build_e2e_effects", fake_e2e)
    monkeypatch.setattr(matched, "build_matched_contrasts", fake_matched)

    result = CliRunner().invoke(
        cli.app,
        ["analyze-e2e", "--run-id", "run-1", "--phase-id", "phase-1"],
    )
    assert result.exit_code == 0, result.output
    assert set(calls) == {"e2e", "matched"}
    output = (
        settings.path("judgments") / "run-1" / "phase-1" / "analysis"
        / "E2E_ANALYSIS.json"
    )
    body = cli._load_content_addressed_json(output)
    assert body["analysis_policy"]["trajectory_checkpoint_divergence"] == (
        "mediated_end_to_end_outcome_not_pairing_error"
    )
    assert body["execution_binding_sha256"] == scope["execution_binding_sha256"]
    assert body["protocol_document_sha256"] == scope["protocol_document_sha256"]
    assert (
        body["analysis_policy"]["analysis_execution_binding_sha256"]
        == "8" * 64
    )
    assert body["semantic_mapping"]["core_expected_variant_ids"]["h"] == "H02+P0"
    assert body["design_bindings"]["eligibility_spec_content_sha256"] == (
        scope["eligibility_spec_content_sha256"]
    )

    repeated = CliRunner().invoke(
        cli.app,
        ["analyze-e2e", "--run-id", "run-1", "--phase-id", "phase-1"],
    )
    assert repeated.exit_code == 0, repeated.output
    assert '"write_status": "EXISTING_IDENTICAL"' in repeated.output


def test_analyze_itt_uses_scope_bound_feature_registry_for_clusters(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shapeflow_p1.analysis import estimands

    settings = Settings.load(REPO, data_root=tmp_path)
    _, _, scope = _write_design(settings)

    # A mutable task view carries a conflicting value.  The command must never consult it.
    task_dir = settings.path("evaluator_root") / "tasks"
    task_dir.mkdir(parents=True)
    (task_dir / "task-1.json").write_text(
        json.dumps({"task_id": "task-1", "cluster_id": "post-outcome-poison"}),
        encoding="utf-8",
    )

    class Records(list):
        scope_receipt = scope

    records = Records([{
        "task_id": "task-1",
        "block_id": "block-1",
        "run_id": "run-1",
        "phase_id": "phase-1",
        "truth_packet_sha256": "a" * 64,
        "content_sha256": "b" * 64,
    }])
    captured: dict[str, object] = {}

    def fake_load(_root, *, run_id, phase_id):
        assert (run_id, phase_id) == ("run-1", "phase-1")
        return records

    def fake_build(passed_records, **kwargs):
        assert passed_records is records
        captured["cluster_by_task"] = kwargs["cluster_by_task"]
        assert kwargs["human_audit_receipt"] is None
        return {
            "blocks_offered": 1,
            "arms": {"H": {"verdict_ready": False}},
            "execution_binding_sha256": scope["execution_binding_sha256"],
            "protocol_document_sha256": scope["protocol_document_sha256"],
            "content_sha256": "f" * 64,
        }

    def fake_write(body, path):
        captured["body"] = body
        captured["path"] = path
        return body["content_sha256"]

    monkeypatch.setattr(cli, "_require_role", lambda role: None)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(
        cli,
        "_verified_analysis_binding",
        lambda **_kwargs: SimpleNamespace(
            digest="8" * 64,
            protocol_sha=scope["protocol_document_sha256"],
            approved_commit="1" * 40,
        ),
    )
    monkeypatch.setattr(estimands, "load_scoped_scores", fake_load)
    monkeypatch.setattr(estimands, "build_verdict_inputs", fake_build)
    monkeypatch.setattr(estimands, "write_verdict_inputs", fake_write)

    result = CliRunner().invoke(
        cli.app,
        ["analyze-itt", "--run-id", "run-1", "--phase-id", "phase-1"],
    )
    assert result.exit_code == 0, result.output
    assert captured["cluster_by_task"] == {"task-1": "cluster-1"}
    assert captured["body"]["execution_binding_sha256"] == (
        scope["execution_binding_sha256"]
    )
    assert captured["body"]["analysis_execution_binding_sha256"] == "8" * 64
    assert captured["body"]["analysis_approved_commit"] == "1" * 40
    assert captured["path"] == (
        settings.path("judgments")
        / "run-1"
        / "phase-1"
        / "analysis"
        / "ITT_VERDICT_INPUTS.json"
    ).resolve()


def test_human_audit_cli_is_evaluator_only_and_passes_exact_results_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from shapeflow_p1.evaluation import human_audit_workflow

    settings = Settings.load(REPO, data_root=tmp_path)
    roles: list[str] = []
    calls: dict[str, object] = {}
    results = {
        "schema_version": "human_audit_results_v1",
        "content_sha256": "a" * 64,
        "items": [],
    }
    results_path = tmp_path / "review-results.json"
    results_path.write_text(json.dumps(results), encoding="utf-8")

    def fake_prepare(passed_settings, *, run_id, phase_id):
        assert passed_settings is settings
        calls["prepare"] = (run_id, phase_id)
        return {"schema_version": "human_audit_queue_v1", "items": []}

    def fake_finalize(passed_settings, *, run_id, phase_id, results):
        assert passed_settings is settings
        calls["finalize"] = (run_id, phase_id, results)
        return {"schema_version": "human_audit_receipt_v2", "status": "AUDITED"}

    monkeypatch.setattr(cli, "_require_role", roles.append)
    monkeypatch.setattr(cli, "_settings", lambda: settings)
    monkeypatch.setattr(human_audit_workflow, "prepare_human_audit", fake_prepare)
    monkeypatch.setattr(human_audit_workflow, "finalize_human_audit", fake_finalize)

    prepared = CliRunner().invoke(
        cli.app,
        ["prepare-human-audit", "--run-id", "run-1", "--phase-id", "phase-1"],
    )
    assert prepared.exit_code == 0, prepared.output
    finalized = CliRunner().invoke(
        cli.app,
        [
            "finalize-human-audit",
            "--run-id", "run-1",
            "--phase-id", "phase-1",
            "--results-json", str(results_path),
        ],
    )
    assert finalized.exit_code == 0, finalized.output
    assert roles == ["evaluator", "evaluator"]
    assert calls["prepare"] == ("run-1", "phase-1")
    assert calls["finalize"] == ("run-1", "phase-1", results)
