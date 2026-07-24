"""`prepare` end to end against a scripted author: seal once, and split by who may read it.

The assertion that matters most is the last one: the runner's copy of a task must contain the
question and nothing else. If the authored facets or the fixed queries reach the treatment
identity, P1 receives a decomposition of the question that P0 never got, and any advantage it
then shows is an artifact of the harness.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.campaign.prepare import load_sealed_registry, prepare_corpus
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.judge_client import DeepSeekJudge

from fixtures.scripted_author import ScriptedAuthor

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


# Production scale on purpose: 64 sealed specs across 16 clusters, so the real split sizes
# (32 screen / 16 pilot / 16 reserve) and the real audit thresholds are what gets exercised.
async def _prepare(settings, *, total=64, clusters=16):
    author = ScriptedAuthor(clusters=clusters, per_cluster=total // clusters)
    judge = DeepSeekJudge(author, "deepseek-chat", "@SHAPEFLOW_PROVIDER@")
    return await prepare_corpus(
        settings, judge=judge, authored_at_utc="2026-07-24T00:00:00Z",
        target_model="Qwen3-14B-AWQ", total=total, clusters=clusters,
    )


async def test_prepare_seals_a_registry_and_both_task_views(settings):
    result = await _prepare(settings)

    assert result.registry_path.exists()
    assert result.manifest_path.exists()
    assert result.steward_task_count == 64
    assert result.runner_task_count == 64
    assert len(result.registry_sha256) == 64
    assert result.audit.ok


async def test_the_registry_is_write_once(settings):
    await _prepare(settings)
    with pytest.raises((RuntimeError, FileExistsError)):
        await _prepare(settings)


async def test_the_sealed_registry_digest_is_recomputed_not_trusted(settings):
    result = await _prepare(settings)
    body, digest = load_sealed_registry(settings)
    assert digest == result.registry_sha256
    assert body["claim_scope"] == "FORMATIVE_ONLY"
    assert body["corpus_tier"] == "FORMATIVE_MACHINE_AUTHORED"

    edited = json.loads(result.registry_path.read_text(encoding="utf-8"))
    edited["tasks"][0]["question"] = edited["tasks"][0]["question"] + " (edited)"
    result.registry_path.write_text(json.dumps(edited), encoding="utf-8")
    with pytest.raises(ValueError, match="has been edited"):
        load_sealed_registry(settings)


async def test_the_runner_never_sees_the_acquisition_spec(settings):
    """The single most important separation in the corpus.

    The steward's record carries the authored facets, the fixed queries and both probes; the
    runner's carries the question. A selector that could read the facets would be answering a
    decomposed question P0 was never given.
    """
    await _prepare(settings)
    steward_dir = settings.path("tasks")
    runner_dir = settings.path("frozen_corpus_for_runner") / "tasks"

    steward_files = sorted(p for p in steward_dir.glob("T*.json"))
    runner_files = sorted(p for p in runner_dir.glob("T*.json"))
    assert len(steward_files) == len(runner_files) == 64

    for path in runner_files:
        record = json.loads(path.read_text(encoding="utf-8"))
        assert set(record) == {"task_id", "split", "original_question", "corpus_tier",
                               "claim_scope"}
        assert "acquisition_spec" not in record
        assert "fixed_queries" not in json.dumps(record)
        assert "required_facets" not in json.dumps(record)

    full = json.loads(steward_files[0].read_text(encoding="utf-8"))
    assert full["acquisition_spec"]["fixed_queries"]
    assert full["acquisition_spec"]["authoring_method"] == "MACHINE_DECOMPOSED"


async def test_every_task_record_validates_against_its_schema(settings):
    from jsonschema import Draft202012Validator

    await _prepare(settings)
    task_schema = Draft202012Validator(
        json.loads((REPO / "schemas" / "task.schema.json").read_text(encoding="utf-8")))
    runner_schema = Draft202012Validator(
        json.loads((REPO / "schemas" / "runner_task.schema.json").read_text(encoding="utf-8")))

    for path in sorted(settings.path("tasks").glob("T*.json")):
        task_schema.validate(json.loads(path.read_text(encoding="utf-8")))
    for path in sorted((settings.path("frozen_corpus_for_runner") / "tasks").glob("T*.json")):
        runner_schema.validate(json.loads(path.read_text(encoding="utf-8")))


async def test_whole_clusters_land_in_one_split(settings):
    result = await _prepare(settings)
    by_cluster: dict[str, set[str]] = {}
    for task in result.registry.tasks:
        by_cluster.setdefault(task.topic_cluster, set()).add(
            result.registry.split_of[task.task_id])
    assert all(len(splits) == 1 for splits in by_cluster.values())


async def test_the_authoring_manifest_travels_with_the_registry(settings):
    result = await _prepare(settings)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["registry_sha256"] == result.registry_sha256
    assert manifest["claim_scope"] == "FORMATIVE_ONLY"
    assert manifest["fingerprint"]["provider"] == "deepseek"
    assert "Qwen" not in manifest["fingerprint"]["returned_model"]
