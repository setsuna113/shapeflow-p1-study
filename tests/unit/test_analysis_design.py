"""Pre-treatment analysis inputs are derived, sealed and runner-safe."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.acquire.manifest import campaign_manifest
from shapeflow_p1.analysis.design import (
    build_task_feature_registry,
    freeze_analysis_design,
    load_runner_analysis_design_receipt,
)
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.hashing import sha256_hex
from shapeflow_p1.object_store import ObjectStore

REPO = Path(__file__).resolve().parents[2]


def _world(tmp_path: Path) -> Settings:
    settings = Settings.load(REPO, data_root=tmp_path)
    task_id = "T-design"
    task_dir = settings.path("evaluator_root") / "tasks"
    task_dir.mkdir(parents=True)
    (task_dir / f"{task_id}.json").write_text(json.dumps({
        "task_id": task_id,
        "split": "FORMATIVE_SCREEN",
        "cluster_id": "cluster-a",
        "original_question": "Which stated values answer this frozen research question?",
        "authored_facets": ["secret-facet-must-not-enter-runner-receipt"],
        "fixed_queries": ["frozen whole-question query", "frozen facet query"],
        "strata": [
            "evidence_volume_low",
            "facets_1_2",
            "table_list_heavy",
        ],
    }), encoding="utf-8")

    objects = ObjectStore(
        settings.path("frozen_corpus_for_runner") / "objects")
    text = "# Results\n\n- Alpha was 41%.\n- Beta was 39%."
    ref = objects.put_bytes(text.encode("utf-8"))
    pool = {
        "task_id": task_id,
        "occurrences": [{
            "occurrence_id": "o1",
            "url": "https://example.test/a",
            "title": "A",
            "snippet_content": "Alpha 41%",
            "content_hash": "h1",
            "vendor_visible_order": 0,
        }],
        "snapshots": {
            "h1": {
                "object_ref": ref.key,
                "byte_len": len(text.encode()),
                "raw_content_format": "markdown",
                "normalization_version": "v1",
                "fetched_at_utc": "2026-07-25T00:00:00Z",
            },
        },
    }
    pool["pool_sha256"] = sha256_hex(canonical_json(pool))
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    pool_dir.mkdir(parents=True)
    (pool_dir / f"{task_id}.json").write_text(
        json.dumps(pool), encoding="utf-8")

    acquisition = campaign_manifest(
        {task_id: "a" * 64},
        registry_sha256="b" * 64,
        claim_scope=settings.claim_scope,
        corpus_tier=settings.corpus_tier,
    )
    acquisition_dir = settings.path("acquisition")
    acquisition_dir.mkdir(parents=True)
    (acquisition_dir / "campaign_acquisition.json").write_text(
        json.dumps(acquisition), encoding="utf-8")
    return settings


def test_freezes_static_features_and_publishes_only_hashes_to_runner(tmp_path):
    settings = _world(tmp_path)
    result = freeze_analysis_design(settings)

    record = result["registry"]["records"]["T-design"]
    assert record["features"]["source_count"] == 1.0
    assert record["features"]["candidate_evidence_tokens"] > 0
    assert record["features"]["table_list_fraction"] > 0
    assert record["features"]["question_token_count"] > 0
    assert record["features"]["authored_facet_count"] == 1.0
    assert record["features"]["fixed_query_count"] == 2.0
    assert record["features"]["declared_stratum_table_list_heavy"] == 1.0
    assert record["features"]["declared_stratum_source_conflict"] == 0.0
    assert len(record["tokenizer_sha256"]) == 64
    assert result["registry"]["tokenizer_sha256"] == record["tokenizer_sha256"]
    assert result["eligibility_spec"]["task_feature_registry_sha256"] == (
        result["registry"]["task_feature_registry_sha256"]
    )

    receipt = load_runner_analysis_design_receipt(settings)
    evaluator_receipt = json.loads(
        (
            settings.path("evaluator_root")
            / "analysis_design"
            / "ANALYSIS_DESIGN_RECEIPT.json"
        ).read_text(encoding="utf-8")
    )
    assert evaluator_receipt == receipt
    assert result["evaluator_receipt_path"].endswith(
        "evaluator/analysis_design/ANALYSIS_DESIGN_RECEIPT.json")
    encoded = json.dumps(receipt)
    assert "secret-facet" not in encoded
    assert "candidate_evidence_tokens" not in encoded
    assert receipt["task_pools"] == [{
        "task_id": "T-design",
        "split": "FORMATIVE_SCREEN",
        "source_pool_sha256": record["source_pool_sha256"],
    }]


def test_design_is_idempotent_but_cannot_first_be_authored_after_runner_state(tmp_path):
    settings = _world(tmp_path)
    first = freeze_analysis_design(settings)
    settings.path("runs").mkdir(parents=True)
    (settings.path("runs") / "ledger.sqlite").write_bytes(b"treatment may exist")
    second = freeze_analysis_design(settings)
    assert first["receipt"]["content_sha256"] == second["receipt"]["content_sha256"]

    other = _world(tmp_path / "late")
    other.path("runs").mkdir(parents=True)
    (other.path("runs") / "ledger.sqlite").write_bytes(b"treatment may exist")
    with pytest.raises(ValueError, match="cannot be authored after treatment"):
        freeze_analysis_design(other)


def test_feature_build_fails_on_corrupt_source_bytes(tmp_path):
    settings = _world(tmp_path)
    pool = json.loads(
        (settings.path("frozen_corpus_for_runner") / "pools" / "T-design.json")
        .read_text(encoding="utf-8")
    )
    ref = pool["snapshots"]["h1"]["object_ref"]
    ObjectStore(
        settings.path("frozen_corpus_for_runner") / "objects"
    )._path_for(ref).write_bytes(b"not zstd")
    with pytest.raises(ValueError, match="unavailable or corrupt"):
        build_task_feature_registry(settings)


def test_existing_write_once_design_cannot_mask_an_obsolete_spec(tmp_path):
    settings = _world(tmp_path)
    result = freeze_analysis_design(settings)
    path = Path(result["eligibility_spec_path"])
    obsolete = dict(result["eligibility_spec"])
    obsolete["schema_version"] = "e2e_eligibility_spec_v1"
    obsolete["content_sha256"] = sha256_hex(canonical_json({
        key: value for key, value in obsolete.items() if key != "content_sha256"
    }))
    path.chmod(0o640)
    path.write_text(json.dumps(obsolete), encoding="utf-8")

    with pytest.raises(ValueError, match="current hash-locked design"):
        freeze_analysis_design(settings)
