"""Clusters and strata derived from the acquired world, not from labels."""

from __future__ import annotations

from shapeflow_p1.acquire.source_clusters import (
    TaskWorld,
    build_source_clusters,
    cross_split_overlap,
    realized_strata,
    strata_disagreements,
    worlds_from_pools,
)


def _world(task_id, split, urls=(), hashes=(), *, occurrences=None, missing=0, declared=()):
    return TaskWorld(
        task_id=task_id, split=split,
        urls=frozenset(urls), content_hashes=frozenset(hashes),
        occurrence_count=occurrences if occurrences is not None else len(urls),
        missing_content=missing, declared_strata=tuple(declared),
    )


def test_tasks_that_share_a_page_are_one_cluster():
    """Two tasks drawing on the same page are one observation, whatever topic label the
    author gave them."""
    worlds = [
        _world("T1", "SCREEN", urls=["https://a.example/x"], hashes=["h1"]),
        _world("T2", "SCREEN", urls=["https://b.example/y"], hashes=["h1"]),
        _world("T3", "SCREEN", urls=["https://c.example/z"], hashes=["h9"]),
    ]
    clusters = build_source_clusters(worlds)
    grouped = {c.cluster_id: c.task_ids for c in clusters}
    assert sorted(grouped.values()) == [["T1", "T2"], ["T3"]]


def test_the_same_page_at_two_urls_still_links_the_tasks():
    worlds = [
        _world("T1", "SCREEN", urls=["https://a.example/x?utm=1"], hashes=["same"]),
        _world("T2", "HOLDOUT", urls=["https://mirror.example/x"], hashes=["same"]),
    ]
    clusters = build_source_clusters(worlds)
    assert len(clusters) == 1
    assert clusters[0].shared_content == {"same"}


def test_a_cluster_spanning_two_splits_is_a_leak():
    """The check that existed compared topic labels, and the splitter assigned by topic
    label -- so it could not fail. This one compares sources."""
    worlds = [
        _world("T1", "FORMATIVE_SCREEN", urls=["https://shared.example/p"], hashes=["h"]),
        _world("T2", "HONEST_HOLDOUT", urls=["https://shared.example/p"], hashes=["h"]),
    ]
    leaks = cross_split_overlap(build_source_clusters(worlds))
    assert len(leaks) == 1
    assert leaks[0].splits == {"FORMATIVE_SCREEN", "HONEST_HOLDOUT"}


def test_unrelated_tasks_do_not_leak():
    worlds = [
        _world("T1", "FORMATIVE_SCREEN", urls=["https://a.example/p"], hashes=["h1"]),
        _world("T2", "HONEST_HOLDOUT", urls=["https://b.example/q"], hashes=["h2"]),
    ]
    assert cross_split_overlap(build_source_clusters(worlds)) == []


def test_evidence_volume_comes_from_the_pool():
    small = _world("T1", "S", urls=["https://a/1", "https://a/2"], hashes=["h1", "h2"])
    big = _world("T2", "S", urls=[f"https://a/{i}" for i in range(20)],
                 hashes=[f"h{i}" for i in range(20)])
    assert "evidence_volume_low" in realized_strata(small)
    assert "evidence_volume_high" in realized_strata(big)


def test_missing_page_content_is_measured_not_declared():
    world = _world("T1", "S", urls=["https://a/1", "https://a/2"],
                   hashes=["h1"], occurrences=2, missing=1)
    assert "raw_content_missing" in realized_strata(world)


def test_redundancy_is_measured_from_duplicate_content():
    world = _world("T1", "S", urls=[f"https://a/{i}" for i in range(10)],
                   hashes=["h1", "h2"], occurrences=10)
    assert "high_redundancy" in realized_strata(world)


def test_a_declared_stratum_the_world_does_not_support_is_a_finding():
    """The label was stamped on from the plan the author was asked to satisfy, then verified
    by counting the same stamps."""
    world = _world("T1", "S", urls=["https://a/1"], hashes=["h1"],
                   declared=("evidence_volume_high", "raw_content_missing"))
    found = strata_disagreements([world])
    assert found["T1"] == ["evidence_volume_high", "raw_content_missing"]


def test_a_declared_stratum_the_world_supports_is_not_a_finding():
    world = _world("T1", "S", urls=[f"https://a/{i}" for i in range(20)],
                   hashes=[f"h{i}" for i in range(20)],
                   declared=("evidence_volume_high", "citation_dense"))
    assert strata_disagreements([world]) == {}


def test_unmeasurable_strata_are_not_claimed_to_be_verified():
    """source_conflict is a property of content, not of a URL count. Reporting it as
    checked would be the same substitution this module exists to catch."""
    world = _world("T1", "S", urls=["https://a/1"], hashes=["h1"],
                   declared=("source_conflict", "natural_research_complete"))
    assert strata_disagreements([world]) == {}


def test_worlds_are_built_from_the_published_pools():
    pools = {
        "T1": {"occurrences": [
            {"url": "https://a/1", "content_hash": "h1"},
            {"url": "https://a/2", "content_hash": None},
        ]},
    }
    worlds = worlds_from_pools(pools, {"T1": "FORMATIVE_SCREEN"},
                               declared={"T1": ["evidence_volume_low"]})
    assert worlds[0].occurrence_count == 2
    assert worlds[0].missing_content == 1
    assert worlds[0].declared_strata == ("evidence_volume_low",)
