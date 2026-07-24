"""The seal's preconditions, which are the ones that make the corpus usable at all."""

from __future__ import annotations

import pytest

from shapeflow_p1.acquire.task_registry import (
    CLAIM_SCOPE,
    CORPUS_TIER,
    STRATA,
    AuthoringFingerprint,
    SealError,
    TaskSpec,
    build_registry,
    merkle_root,
)

FP = AuthoringFingerprint(
    provider="deepseek", requested_model="deepseek-chat", returned_model="deepseek-chat",
    system_fingerprint="fp_x", prompt_sha256="a" * 64, seed=7,
    authored_at_utc="2026-07-24T18:00:00Z",
)


def _task(i: int, cluster: str, strata) -> TaskSpec:
    return TaskSpec(
        task_id=f"T{i:03d}", topic=f"topic {cluster}", question=f"question {i}?",
        required_facets=("f1", "f2"), fixed_queries=(f"q{i}a", f"q{i}b"),
        strata=tuple(strata), topic_cluster=cluster,
    )


def _covering_set(n: int):
    """n tasks whose union covers every stratum, one cluster each."""
    tasks = []
    for i in range(n):
        assigned = [STRATA[i % len(STRATA)]]
        if i < len(STRATA):
            assigned = [STRATA[i]]
        tasks.append(_task(i, f"c{i}", assigned))
    return tasks


def test_the_corpus_is_labelled_formative_and_the_label_travels():
    reg = build_registry(tasks=_covering_set(20), fingerprint=FP, screen_n=10, pilot_n=5,
                         target_model="Qwen/Qwen3-14B-AWQ")
    assert reg.corpus_tier == CORPUS_TIER == "FORMATIVE_MACHINE_AUTHORED"
    assert reg.claim_scope == CLAIM_SCOPE == "FORMATIVE_ONLY"
    for task in reg.tasks:
        assert task.corpus_tier == CORPUS_TIER
        assert task.claim_scope == CLAIM_SCOPE
    assert reg.content()["claim_scope"] == "FORMATIVE_ONLY"


def test_the_target_model_may_not_author_the_corpus():
    """A corpus written by the model under test is selected for what that model finds easy."""
    bad = AuthoringFingerprint(**{**FP.__dict__, "requested_model": "Qwen/Qwen3-14B-AWQ"})
    with pytest.raises(SealError, match="authored by the target model"):
        build_registry(tasks=_covering_set(20), fingerprint=bad, screen_n=10, pilot_n=5,
                       target_model="Qwen/Qwen3-14B-AWQ")


def test_every_stratum_must_be_represented():
    with pytest.raises(SealError, match="unrepresented"):
        build_registry(tasks=[_task(0, "c0", ["facets_1_2"])], fingerprint=FP,
                       screen_n=1, pilot_n=0, target_model="Q")


def test_a_topic_cluster_never_straddles_two_splits():
    """Two paraphrases, or two tasks sharing most sources, are one observation.

    Splitting them would leak between SCREEN and POWER_PILOT; counting them separately would
    overstate n.
    """
    tasks = _covering_set(len(STRATA))
    # Put three tasks in one cluster.
    tasks += [_task(100 + k, "shared", ["facets_1_2"]) for k in range(3)]
    reg = build_registry(tasks=tasks, fingerprint=FP, screen_n=8, pilot_n=6, target_model="Q")
    shared = [t for t in reg.tasks if t.topic_cluster == "shared"]
    assert len({reg.split_of[t.task_id] for t in shared}) == 1


def test_the_acquisition_digest_is_separate_from_the_task_digest():
    """A reworded question is a different change from a different world to search."""
    a = _task(1, "c", ["facets_1_2"])
    reworded = TaskSpec(**{**a.__dict__, "question": "a different question?"})
    requeried = TaskSpec(**{**a.__dict__, "fixed_queries": ("other",)})
    assert reworded.spec_sha256 != a.spec_sha256
    assert reworded.acquisition_spec_sha256 == a.acquisition_spec_sha256
    assert requeried.acquisition_spec_sha256 != a.acquisition_spec_sha256


def test_a_sealed_registry_is_write_once(tmp_path):
    reg = build_registry(tasks=_covering_set(20), fingerprint=FP, screen_n=10, pilot_n=5,
                         target_model="Q")
    path = tmp_path / "registry.json"
    digest = reg.seal(path)
    assert len(digest) == 64
    with pytest.raises(RuntimeError, match="write-once"):
        reg.seal(path)


def test_the_merkle_root_follows_any_task_change():
    a = _covering_set(20)
    reg_a = build_registry(tasks=a, fingerprint=FP, screen_n=10, pilot_n=5, target_model="Q")
    b = list(a)
    b[0] = TaskSpec(**{**b[0].__dict__, "question": "edited after the fact?"})
    reg_b = build_registry(tasks=b, fingerprint=FP, screen_n=10, pilot_n=5, target_model="Q")
    assert reg_a.merkle_root != reg_b.merkle_root
    assert merkle_root([]) == merkle_root([])


def test_reserve_is_preordered_so_substitution_cannot_be_chosen_later():
    """If a task proves unusable, the replacement is already decided.

    Picking a replacement after seeing results would let the corpus be tuned to the outcome.
    """
    tasks = _covering_set(24)
    reg = build_registry(tasks=tasks, fingerprint=FP, screen_n=8, pilot_n=6, target_model="Q")
    assert reg.reserve_order
    assert all(reg.split_of[t] == "RESERVE" for t in reg.reserve_order)
    # Deterministic, not sorted-by-id: the order follows cluster assignment, and what matters is
    # that the same inputs always yield the same order so the substitution is fixed in advance.
    again = build_registry(tasks=list(tasks), fingerprint=FP, screen_n=8, pilot_n=6,
                           target_model="Q")
    assert again.reserve_order == reg.reserve_order
