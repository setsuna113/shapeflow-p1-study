"""The frozen world: snapshot identity, vendor-visible dedup, and deterministic search.

These are the properties every arm depends on being identical. If two arms could see a
different snapshot for one URL, a different vendor-visible order, or a different ranking for
one query, the comparison between them would be measuring the corpus rather than the form.
"""

from __future__ import annotations

import pytest

from shapeflow.world.search_backend import (
    Bm25Index,
    ExactSnapshotReplayBackend,
    FrozenTaskCorpusBackend,
    ReplayMiss,
    SearchRecord,
)
from shapeflow.world.snapshot_store import SnapshotStore, normalize_v1
from shapeflow.world.source_pool import QueryResponse, RawResult, build_source_pool
from shapeflow.object_store import ObjectStore
from shapeflow.providers.retry import RetryDecision, capped_backoff, classify_status


# --- snapshot identity ----------------------------------------------------------------


def test_normalize_unifies_newlines_and_nfc():
    assert normalize_v1("a\r\nb\rc") == b"a\nb\nc"


def test_snapshot_dedup_identical_content(tmp_path):
    store = SnapshotStore(ObjectStore(tmp_path))
    a = store.freeze("page text", raw_content_format="markdown", fetched_at_utc="2026-07-24T00:00:00Z")
    b = store.freeze("page text", raw_content_format="markdown", fetched_at_utc="2026-07-24T01:00:00Z")
    assert a.content_hash == b.content_hash  # identity is content, not fetch time
    assert store.read_text(a) == "page text"


# --- the fairness-critical dedup ------------------------------------------------------


def _store(tmp_path):
    return SnapshotStore(ObjectStore(tmp_path))


def test_vendor_visible_reproduces_first_occurrence_order(tmp_path):
    # URL B appears in query 1 (rank 2) and again in query 2 (rank 1). Vendor keeps the
    # first occurrence and first-occurrence order: A, B, C.
    responses = [
        QueryResponse("qs1", "q1", (
            RawResult("https://a", "A", 1, "sa", "raw A"),
            RawResult("https://b", "B", 2, "sb", "raw B"),
        )),
        QueryResponse("qs2", "q2", (
            RawResult("https://b", "B", 1, "sb2", "raw B"),  # duplicate URL
            RawResult("https://c", "C", 2, "sc", "raw C"),
        )),
    ]
    pool = build_source_pool("t1", responses, _store(tmp_path), fetched_at_utc="2026-07-24T00:00:00Z")

    visible = pool.vendor_visible
    assert [o.url for o in visible] == ["https://a", "https://b", "https://c"]
    assert [o.vendor_visible_order for o in visible] == [0, 1, 2]

    # The second B is audit-only and points at the first B as its winner.
    dupes = [o for o in pool.occurrences if o.visibility == "AUDIT_ONLY"]
    assert len(dupes) == 1
    b_first = next(o for o in visible if o.url == "https://b")
    assert dupes[0].duplicate_of_occurrence_id == b_first.occurrence_id
    assert dupes[0].vendor_visible_order is None


def test_audit_graph_keeps_every_occurrence(tmp_path):
    responses = [
        QueryResponse("qs1", "q1", (RawResult("https://b", "B", 1, "s", "raw"),)),
        QueryResponse("qs2", "q2", (RawResult("https://b", "B", 1, "s", "raw"),)),
    ]
    pool = build_source_pool("t1", responses, _store(tmp_path), fetched_at_utc="2026-07-24T00:00:00Z")
    # Same content dedups to one snapshot, but both occurrences survive for lineage.
    assert len(pool.snapshots) == 1
    assert len(pool.audit_graph) == 2
    assert len(pool.vendor_visible) == 1


def test_missing_raw_content_keeps_snippet_and_no_snapshot(tmp_path):
    responses = [
        QueryResponse("qs1", "q1", (RawResult("https://a", "A", 1, "snippet only", None),)),
    ]
    pool = build_source_pool("t1", responses, _store(tmp_path), fetched_at_utc="2026-07-24T00:00:00Z")
    occ = pool.vendor_visible[0]
    assert occ.content_hash is None  # not WEBPAGE_P1-eligible
    assert occ.snippet_content == "snippet only"
    assert pool.snapshots == {}


# --- deterministic BM25 ---------------------------------------------------------------


def test_bm25_is_deterministic_and_ranks_relevance():
    docs = [
        ("d1", "the cat sat on the mat"),
        ("d2", "quantum chromodynamics and the strong force"),
        ("d3", "a cat and a dog"),
    ]
    idx = Bm25Index(docs)
    r1 = idx.search("cat", top_k=3)
    r2 = idx.search("cat", top_k=3)
    assert r1 == r2  # deterministic
    assert {d for d, _ in r1} == {"d1", "d3"}  # only cat docs
    assert "d2" not in {d for d, _ in r1}


def test_bm25_tie_break_by_doc_id():
    # Two identical docs -> identical scores; the tie must break by doc_id ascending.
    idx = Bm25Index([("zeta", "same words here"), ("alpha", "same words here")])
    result = idx.search("same words", top_k=2)
    assert [d for d, _ in result] == ["alpha", "zeta"]


def test_frozen_corpus_backend_returns_tavily_shaped_records(tmp_path):
    responses = [
        QueryResponse("qs1", "q1", (
            RawResult("https://cats", "Cats", 1, "about cats", "cats are feline animals"),
            RawResult("https://dogs", "Dogs", 2, "about dogs", "dogs are canine animals"),
        )),
    ]
    ss = _store(tmp_path)
    pool = build_source_pool("t1", responses, ss, fetched_at_utc="2026-07-24T00:00:00Z")
    backend = FrozenTaskCorpusBackend(pool, ss)
    hits = backend.search("feline", max_results=5)
    assert hits and hits[0].url == "https://cats"
    assert isinstance(hits[0], SearchRecord)
    assert hits[0].raw_content == "cats are feline animals"
    # Same query, identical results.
    assert backend.search("feline", max_results=5) == hits


def test_replay_miss_is_fatal_not_a_live_fallback():
    backend = ExactSnapshotReplayBackend({"qs1": [SearchRecord("u", "t", "c", None, 1.0, "o1")]})
    assert backend.search_by_snapshot("qs1")
    with pytest.raises(ReplayMiss):
        backend.search_by_snapshot("qs_unknown")


# --- retry classification -------------------------------------------------------------


@pytest.mark.parametrize("status,decision", [
    (429, RetryDecision.RETRY_AFTER),
    (500, RetryDecision.BACKOFF),
    (503, RetryDecision.BACKOFF),
    (400, RetryDecision.FAIL_FAST),
    (401, RetryDecision.FAIL_FAST),
    (402, RetryDecision.FAIL_FAST),
    (422, RetryDecision.FAIL_FAST),
])
def test_retry_classification(status, decision):
    assert classify_status(status) is decision


def test_backoff_is_capped_and_monotone_ceiling():
    ceilings = [capped_backoff(a, base=0.5, cap=10.0) for a in range(8)]
    assert ceilings[0] == 0.5
    assert ceilings[-1] == 10.0  # capped
    assert all(b <= 10.0 for b in ceilings)
