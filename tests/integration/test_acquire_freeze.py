"""Acquisition freezes one world per task, records what failed, and splits it by reader.

The assertions that carry the design: the search provider is called once per task and never again; the audit
occurrence graph does not reach the runner's tree; a dead query is part of the frozen world
rather than an absence; and the frozen pool reloads byte-identically for a treatment run.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.acquire.manifest import AcquisitionIntegrityError, load_task_manifest
from shapeflow_p1.acquire.exa_client import ExaCaptureClient
from shapeflow_p1.campaign.acquire import (
    acquire_all,
    acquired_task_ids,
    load_frozen_pool,
    queries_for_task,
    runner_pool_path,
    exa_params_from,
)
from shapeflow_p1.campaign.prepare import load_sealed_registry, prepare_corpus
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.judge_client import DeepSeekJudge

from fixtures.fake_exa import FakeExa
from fixtures.scripted_author import ScriptedAuthor

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


async def _prepared(settings, *, total=8, clusters=4):
    """A small sealed corpus. Audit thresholds are relaxed only for corpus size, not content."""
    author = ScriptedAuthor(clusters=clusters, per_cluster=total // clusters)
    judge = DeepSeekJudge(author, "deepseek-chat", "@SHAPEFLOW_PROVIDER@")
    relaxed = dict(settings.configs["task_source"])
    relaxed["audit"] = {**relaxed["audit"], "require_distinct_topic_clusters": clusters}
    relaxed["splits"] = {"FORMATIVE_SCREEN": total // 2,
                         "FORMATIVE_POWER_PILOT": total // 4,
                         "RESERVE": total - total // 2 - total // 4}
    relaxed["strata_min_counts"] = {"source_conflict": 1}
    settings.configs["task_source"] = relaxed
    return await prepare_corpus(
        settings, judge=judge, authored_at_utc="2026-07-24T00:00:00Z",
        target_model="Qwen3-14B-AWQ", total=total, clusters=clusters,
    )


def _factory(settings, fake):
    params = exa_params_from(settings)

    def make(task_id: str) -> ExaCaptureClient:
        return ExaCaptureClient(fake, params)

    return make


async def test_acquisition_freezes_one_world_per_task(settings):
    await _prepared(settings)
    fake = FakeExa()
    outcome = await acquire_all(settings, client_factory=_factory(settings, fake),
                                fetched_at_utc="2026-07-24T01:00:00Z")

    assert outcome.tasks_acquired == 6      # screen (4) + pilot (2); reserve is not acquired
    assert outcome.tasks_skipped == 0
    assert outcome.queries_ok > 0
    assert outcome.campaign_manifest_path.exists()
    assert len(outcome.campaign_sha256) == 64

    campaign = json.loads(outcome.campaign_manifest_path.read_text(encoding="utf-8"))
    assert campaign["claim_scope"] == "FORMATIVE_ONLY"
    assert campaign["corpus_tier"] == "FORMATIVE_MACHINE_AUTHORED"
    assert len(campaign["acquisition_merkle_root"]) == 64


async def test_reserve_worlds_are_not_fetched(settings):
    """Acquiring 16 worlds that may never be used would spend the cap on nothing."""
    result = await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")
    reserve = [t.task_id for t in result.registry.tasks
               if result.registry.split_of[t.task_id] == "RESERVE"]
    assert reserve
    for task_id in reserve:
        assert not runner_pool_path(settings, task_id).exists()


async def test_a_second_acquisition_makes_no_request(settings):
    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")
    first_calls = len(fake.calls)
    assert first_calls > 0

    again = await acquire_all(settings, client_factory=_factory(settings, fake),
                              fetched_at_utc="2026-07-24T02:00:00Z")
    assert len(fake.calls) == first_calls, "the frozen world was re-fetched"
    assert again.tasks_acquired == 0
    assert again.tasks_skipped == 6


async def test_the_runner_tree_has_no_audit_occurrence_graph(settings):
    """Vendor's dedup is what P0 saw; the duplicates it discarded are evaluator material."""
    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")

    task_id = acquired_task_ids(settings)[0]
    runner_body = json.loads(runner_pool_path(settings, task_id).read_text(encoding="utf-8"))
    assert all(o["vendor_visible_order"] is not None for o in runner_body["occurrences"])
    text = json.dumps(runner_body)
    assert "AUDIT_ONLY" not in text
    assert "duplicate_of_occurrence_id" not in text
    assert "rank" not in text and "score" not in text, (
        "rank and score are the audit graph's diversity signal; P0 never saw them"
    )

    steward_body = load_task_manifest(settings.path("acquisition") / f"{task_id}.json")
    assert any(o["visibility"] == "VENDOR_VISIBLE" for o in steward_body["occurrences"])


async def test_a_dead_query_is_part_of_the_frozen_world(settings):
    """A task with three dead queries must not look like one with three rich ones."""
    result = await _prepared(settings)
    first = sorted(
        (t for t in result.registry.tasks
         if result.registry.split_of[t.task_id] == "FORMATIVE_SCREEN"),
        key=lambda t: t.task_id,
    )[0]
    fake = FakeExa(fail_queries={first.fixed_queries[0]},
                      empty_queries={first.fixed_queries[1]})
    outcome = await acquire_all(settings, client_factory=_factory(settings, fake),
                                fetched_at_utc="2026-07-24T01:00:00Z")
    assert outcome.queries_failed >= 1
    assert outcome.queries_empty >= 1

    manifest = load_task_manifest(settings.path("acquisition") / f"{first.task_id}.json")
    statuses = {q["status"] for q in manifest["queries"]}
    assert "FAILED" in statuses
    assert "EMPTY" in statuses


async def test_the_frozen_pool_reloads_for_a_treatment_run(settings):
    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")

    task_id = acquired_task_ids(settings)[0]
    pool, store = load_frozen_pool(settings, task_id)
    assert pool.vendor_visible
    assert [o.vendor_visible_order for o in pool.vendor_visible] == list(
        range(len(pool.vendor_visible)))
    with_content = [o for o in pool.vendor_visible if o.content_hash]
    assert with_content
    text = store.read_text(pool.snapshots[with_content[0].content_hash])
    assert text.startswith("#")


async def test_an_edited_manifest_is_detected(settings):
    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")
    task_id = acquired_task_ids(settings)[0]
    path = settings.path("acquisition") / f"{task_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["occurrences"][0]["url"] = "https://swapped.example/doc"
    # The published pool is 0444; an edit has to defeat that first, which is the point.
    path.chmod(0o644)
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(AcquisitionIntegrityError, match="has been edited"):
        load_task_manifest(path)


async def test_an_edited_runner_pool_is_detected(settings):
    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")
    task_id = acquired_task_ids(settings)[0]
    path = runner_pool_path(settings, task_id)
    body = json.loads(path.read_text(encoding="utf-8"))
    body["occurrences"][0]["url"] = "https://swapped.example/doc"
    # The published pool is 0444; an edit has to defeat that first, which is the point.
    path.chmod(0o644)
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ValueError, match="has been edited"):
        load_frozen_pool(settings, task_id)


def test_the_query_set_keeps_the_question_and_facets_when_capped():
    task = {
        "acquisition_spec": {
            "fixed_queries": ["whole question", "facet one", "facet two", "facet three"],
            "conflict_probe": "conflict", "negative_or_gap_probe": "negative",
        }
    }
    assert queries_for_task(task, max_total=3) == ["whole question", "facet one", "facet two"]
    assert queries_for_task(task, max_total=99) == [
        "whole question", "facet one", "facet two", "facet three", "conflict", "negative",
    ]


# --- a crash is a gap to fill, not a task to skip forever ------------------------------------


async def test_a_world_whose_pool_was_never_published_is_rebuilt(settings):
    """The old skip predicate was "the manifest exists". The write order was manifest first,
    so a crash between the two left a task every later run skipped and whose pool was never
    written -- unfixable without deleting the write-once manifest that exists to prevent
    exactly that."""
    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")
    task_id = acquired_task_ids(settings)[0]
    pool = runner_pool_path(settings, task_id)
    pool.chmod(0o644)
    pool.unlink()

    fake2 = FakeExa()
    outcome = await acquire_all(settings, client_factory=_factory(settings, fake2),
                                fetched_at_utc="2026-07-24T02:00:00Z")
    assert fake2.calls, "the interrupted task was skipped instead of rebuilt"
    assert any(task_id in note for note in outcome.partial_worlds)
    assert runner_pool_path(settings, task_id).exists()
    # The interrupted attempt's manifest is kept, not deleted: it is the only record of
    # what that attempt saw.
    superseded = list(settings.path("acquisition").glob(f"{task_id}.json.superseded-*"))
    assert superseded, "the stale manifest was destroyed rather than set aside"


async def test_a_missing_snapshot_blob_is_not_a_frozen_world(settings):
    """`put_bytes` trusts an existing file. A world whose content is gone still had a
    manifest, a pool, and every hash in both agreeing with each other."""
    from shapeflow_p1.campaign.acquire import task_world_is_complete

    await _prepared(settings)
    fake = FakeExa()
    await acquire_all(settings, client_factory=_factory(settings, fake),
                      fetched_at_utc="2026-07-24T01:00:00Z")
    task_id = acquired_task_ids(settings)[0]
    body = json.loads(runner_pool_path(settings, task_id).read_text(encoding="utf-8"))
    ref = next(iter(body["snapshots"].values()))["object_ref"]
    objects = settings.path("frozen_corpus_for_runner") / "objects"
    blobs = [b for b in objects.rglob("*") if b.is_file() and b.stem == ref]
    assert blobs, "the snapshot blob should exist before we remove it"
    for blob in blobs:
        blob.unlink()

    complete, _digest, reason = task_world_is_complete(settings, task_id)
    assert not complete and "snapshot" in reason


async def test_a_rejected_credential_stops_instead_of_burning_the_cap(settings):
    """262 consecutive 401s is what happens when acquisition treats auth like a blip."""
    await _prepared(settings)
    registry, _ = load_sealed_registry(settings)
    first = queries_for_task(registry["tasks"][0], max_total=8)[0]
    fake = FakeExa(status_for={q: 401 for q in [first]})

    outcome = await acquire_all(settings, client_factory=_factory(settings, fake),
                                fetched_at_utc="2026-07-24T01:00:00Z")
    assert outcome.blocked_credential
    assert len(fake.calls) == 1, f"kept going after a rejected credential: {fake.calls}"
    assert outcome.tasks_acquired == 0


async def test_no_campaign_manifest_is_written_for_a_partial_world(settings):
    """It used to be rewritten on every invocation, including one that broke out early, so
    its Merkle root could silently describe half an acquisition."""
    await _prepared(settings)
    registry, _ = load_sealed_registry(settings)
    first = queries_for_task(registry["tasks"][0], max_total=8)[0]

    # A run that stops on a rejected credential has frozen nothing.
    stopped = await acquire_all(
        settings, client_factory=_factory(settings, FakeExa(status_for={first: 401})),
        fetched_at_utc="2026-07-24T01:00:00Z")
    assert stopped.incomplete, "a partial acquisition reported itself complete"
    assert stopped.campaign_manifest_path is None
    assert not (settings.path("acquisition") / "campaign_acquisition.json").exists()

    # The same call, once the world is whole, writes it.
    finished = await acquire_all(settings, client_factory=_factory(settings, FakeExa()),
                                 fetched_at_utc="2026-07-24T02:00:00Z")
    assert not finished.incomplete
    assert finished.campaign_manifest_path is not None


async def test_only_tasks_narrows_within_the_split_and_cannot_widen_it(settings):
    """Pulling a few worlds must not mean acquiring every world in the split.

    RESERVE worlds are deliberately not fetched up front -- acquiring 16 that may never be used
    would spend the cap on nothing. But a substituted task, or a small held-out set for judge
    calibration, needs exactly one or two of them. That is a task filter, not a split change.
    """
    await _prepared(settings)
    registry, _ = load_sealed_registry(settings)
    screen = [t["task_id"] for t in registry["tasks"] if t["split"] == "FORMATIVE_SCREEN"]
    reserve = [t["task_id"] for t in registry["tasks"] if t["split"] == "RESERVE"]
    assert reserve, "fixture must seal a RESERVE split for this to mean anything"

    fake = FakeExa()
    outcome = await acquire_all(
        settings, client_factory=_factory(settings, fake),
        fetched_at_utc="2026-07-24T01:00:00Z", only_tasks=screen[:1],
    )
    assert outcome.tasks_acquired == 1

    # A partial acquisition must not publish a campaign manifest describing a complete world.
    assert outcome.campaign_manifest_path is None

    # The filter narrows; it cannot reach a split the config excludes.
    with pytest.raises(ValueError, match="outside the acquired splits"):
        await acquire_all(
            settings, client_factory=_factory(settings, fake),
            fetched_at_utc="2026-07-24T01:00:00Z", only_tasks=reserve[:1],
        )
