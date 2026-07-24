"""Authoring plans coverage before anything is written; the audit re-verifies it independently.

Every test here corresponds to a corpus defect that is invisible once results exist -- an
unretrievable question reads as a P1 failure, a paraphrase pair overstates n, a split cluster
leaks, a missing stratum makes the eligibility envelope unestimable.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.acquire.registry_audit import AuditFailed, audit_registry, audit_tasks
from shapeflow_p1.acquire.task_author import (
    AuthoringError,
    author_prompt_sha256,
    author_tasks,
    authoring_manifest,
    plan_profiles,
)
from shapeflow_p1.acquire.task_registry import STRATA, AuthoringFingerprint, TaskSpec, build_registry
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.judge_client import DeepSeekJudge

from fixtures.scripted_author import ScriptedAuthor

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def task_config():
    return Settings.load(REPO, data_root=Path("/tmp/x")).configs["task_source"]


# --- the frozen coverage plan ---------------------------------------------------------------


def test_the_plan_reaches_every_configured_stratum_minimum(task_config):
    profiles = plan_profiles(total=64, clusters=16,
                             min_counts=task_config["strata_min_counts"])
    assert len(profiles) == 64
    counts = {s: 0 for s in STRATA}
    for profile in profiles:
        for stratum in profile.strata:
            counts[stratum] += 1
    for stratum, minimum in task_config["strata_min_counts"].items():
        assert counts[stratum] >= minimum, f"{stratum}: {counts[stratum]} < {minimum}"


def test_the_plan_is_deterministic(task_config):
    a = plan_profiles(total=64, clusters=16, min_counts=task_config["strata_min_counts"])
    b = plan_profiles(total=64, clusters=16, min_counts=task_config["strata_min_counts"])
    assert [p.strata for p in a] == [p.strata for p in b]


def test_clusters_must_divide_the_corpus_evenly(task_config):
    with pytest.raises(AuthoringError, match="do not divide evenly"):
        plan_profiles(total=64, clusters=7, min_counts=task_config["strata_min_counts"])


def test_a_corpus_too_small_for_its_minimums_is_refused():
    with pytest.raises(AuthoringError, match="cannot reach the configured minimum"):
        plan_profiles(total=4, clusters=2, min_counts={"source_conflict": 99})


def test_the_prompt_digest_covers_every_prompt_byte():
    assert len(author_prompt_sha256()) == 64


# --- authoring against a scripted author ------------------------------------------------------


def _judge(author: ScriptedAuthor) -> DeepSeekJudge:
    return DeepSeekJudge(author, "deepseek-chat", "@SHAPEFLOW_PROVIDER@")


async def test_authoring_produces_the_planned_corpus(task_config):
    author = ScriptedAuthor(clusters=4, per_cluster=2)
    specs, fingerprint, responses = await author_tasks(
        _judge(author), total=8, clusters=4,
        min_counts={"source_conflict": 1}, requested_model="deepseek-chat", seed=7,
        authored_at_utc="2026-07-24T00:00:00Z", target_model="Qwen3-14B-AWQ",
    )
    assert len(specs) == 8
    assert len({s.task_id for s in specs}) == 8
    assert len({s.topic_cluster for s in specs}) == 4
    assert fingerprint.returned_model == "deepseek-chat"
    assert fingerprint.prompt_sha256 == author_prompt_sha256()
    assert all(s.corpus_tier == "FORMATIVE_MACHINE_AUTHORED" for s in specs)
    assert all(s.claim_scope == "FORMATIVE_ONLY" for s in specs)
    # One cluster call plus one per cluster: the call that decided the topics is part of how
    # the corpus came to exist and belongs in the manifest too.
    assert responses[0]["label"] == "clusters"
    assert len(responses) == 5
    assert all(r["prompt"] for r in responses), "the exact prompt bytes must be kept"
    assert all(r["attempts"] for r in responses)


async def test_a_skipped_profile_is_fatal_not_silently_dropped():
    """A dropped profile takes its stratum with it, and nothing downstream would notice."""
    author = ScriptedAuthor(clusters=2, per_cluster=2, skip_profile="P001")
    with pytest.raises(AuthoringError, match="skipped profile"):
        await author_tasks(
            _judge(author), total=4, clusters=2, min_counts={}, requested_model="deepseek-chat",
            seed=1, authored_at_utc="2026-07-24T00:00:00Z", target_model="Qwen3-14B-AWQ",
        )


async def test_a_corpus_authored_by_the_target_model_is_refused():
    author = ScriptedAuthor(clusters=2, per_cluster=2)
    with pytest.raises(AuthoringError, match="authored by the target model"):
        await author_tasks(
            _judge(author), total=4, clusters=2, min_counts={},
            requested_model="deepseek-chat", seed=1,
            authored_at_utc="2026-07-24T00:00:00Z", target_model="deepseek",
        )


async def test_the_authoring_manifest_records_how_the_corpus_came_to_exist():
    author = ScriptedAuthor(clusters=2, per_cluster=2)
    specs, fingerprint, responses = await author_tasks(
        _judge(author), total=4, clusters=2, min_counts={}, requested_model="deepseek-chat",
        seed=1, authored_at_utc="2026-07-24T00:00:00Z", target_model="Qwen3-14B-AWQ",
    )
    manifest = authoring_manifest(fingerprint, responses,
                                  plan_profiles(total=4, clusters=2, min_counts={}))
    assert manifest["prompt_sha256"] == author_prompt_sha256()
    assert len(manifest["manifest_sha256"]) == 64
    assert len(manifest["profiles"]) == 4
    assert specs


# --- the audit --------------------------------------------------------------------------------


def _spec(task_id="T1", question=None, queries=None, facets=("alpha",), cluster="c1",
          strata=STRATA) -> TaskSpec:
    return TaskSpec(
        task_id=task_id,
        topic="Topic",
        question=question or (
            "Which national statistics agencies published quarterly housing completion figures "
            "for 2025, and where do their published totals disagree?"
        ),
        required_facets=tuple(facets),
        fixed_queries=tuple(queries or [
            "national statistics housing completions 2025",
            "quarterly housing completion totals disagreement",
            "housing completions revision methodology",
        ]),
        strata=tuple(strata),
        topic_cluster=cluster,
    )


def test_an_unretrievable_query_is_a_finding(task_config):
    report = audit_tasks([_spec(queries=["the it of", "and or but", "to be"])],
                         config=task_config)
    assert any(f.check == "retrievability" for f in report.findings)


def test_a_query_set_unrelated_to_its_question_is_a_finding(task_config):
    report = audit_tasks(
        [_spec(queries=["marmalade recipes victorian", "citrus preserving jars",
                        "orange peel candying"])],
        config=task_config)
    assert any(f.check == "query_alignment" for f in report.findings)


def test_two_paraphrases_are_caught_as_one_observation(task_config):
    a = _spec(task_id="T1")
    b = _spec(task_id="T2", question=a.question + " Please be specific.")
    report = audit_tasks([a, b], config=task_config)
    assert any(f.check == "near_duplicate" for f in report.findings)


def test_a_short_question_is_a_finding(task_config):
    report = audit_tasks([_spec(question="Too short?")], config=task_config)
    assert any(f.check == "question_length" for f in report.findings)


def test_a_stratum_shortfall_is_a_finding(task_config):
    report = audit_tasks([_spec(strata=("evidence_volume_low",))], config=task_config)
    assert any(f.check == "stratum_shortfall" for f in report.findings)


def test_a_failing_audit_refuses_to_seal(task_config):
    report = audit_tasks([_spec(question="Too short?")], config=task_config)
    with pytest.raises(AuditFailed, match="not sealed"):
        report.raise_if_failed()


def test_a_cluster_split_across_splits_is_a_finding(task_config):
    """Tasks sharing sources are one observation; splitting them leaks."""
    tasks = [
        _spec(task_id=f"T{i}", cluster="shared" if i < 2 else f"c{i}",
              question=(
                  f"Which regulators published enforcement notices about topic {i} during 2025, "
                  f"and how did the published penalty totals differ between them?"
              ))
        for i in range(4)
    ]
    registry = build_registry(
        tasks=tasks,
        fingerprint=AuthoringFingerprint(
            provider="deepseek", requested_model="deepseek-chat", returned_model="deepseek-chat",
            system_fingerprint="fp", prompt_sha256="x" * 64, seed=1,
            authored_at_utc="2026-07-24T00:00:00Z"),
        screen_n=1, pilot_n=1, target_model="Qwen3-14B-AWQ",
    )
    # Force the leak the audit must catch.
    registry.split_of[tasks[0].task_id] = "FORMATIVE_SCREEN"
    registry.split_of[tasks[1].task_id] = "FORMATIVE_POWER_PILOT"
    report = audit_registry(registry, config={**task_config,
                                              "splits": {"FORMATIVE_SCREEN": 1,
                                                         "FORMATIVE_POWER_PILOT": 1,
                                                         "RESERVE": 2}})
    assert any(f.check == "cluster_split" for f in report.findings)
