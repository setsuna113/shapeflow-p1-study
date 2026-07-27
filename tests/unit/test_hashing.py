"""Identity derivation: stable, domain-separated, and lineage-preserving."""

from __future__ import annotations

from shapeflow.hashing import (
    content_id,
    derive_id,
    occurrence_id,
    query_snapshot_id,
    sha256_hex,
)


def test_derive_id_is_stable_across_key_order():
    assert derive_id("x", {"a": 1, "b": 2}) == derive_id("x", {"b": 2, "a": 1})


def test_domain_separation_prevents_cross_type_aliasing():
    fields = {"a": 1}
    assert derive_id("occurrence", fields) != derive_id("evidence_span", fields)


def test_identical_bytes_share_a_content_id():
    assert content_id(b"page bytes") == content_id(b"page bytes")
    assert content_id(b"page bytes") != content_id(b"page byte")


def test_query_snapshot_commits_to_parameters_not_just_text():
    base = dict(query="q", api_version="1", schema_version="1")
    shallow = query_snapshot_id(**base, params={"search_depth": "basic"})
    deep = query_snapshot_id(**base, params={"search_depth": "advanced"})
    # Same words, different slice of the web: must not replay as one another.
    assert shallow != deep


def test_occurrence_keeps_lineage_distinct_for_same_page():
    qs = query_snapshot_id(query="q", params={}, api_version="1", schema_version="1")
    first = occurrence_id(query_snapshot_id=qs, rank=1, url="https://e.com/a")
    third = occurrence_id(query_snapshot_id=qs, rank=3, url="https://e.com/a")
    other_query = occurrence_id(query_snapshot_id="other", rank=1, url="https://e.com/a")
    # One page found at two ranks, or via two queries, is several occurrences.
    # Collapsing them would erase the citation lineage quality metrics rely on.
    assert len({first, third, other_query}) == 3


def test_occurrence_is_independent_of_content():
    # Two URLs serving identical bytes dedupe in storage but keep separate citations.
    qs = query_snapshot_id(query="q", params={}, api_version="1", schema_version="1")
    a = occurrence_id(query_snapshot_id=qs, rank=1, url="https://a.com/p")
    b = occurrence_id(query_snapshot_id=qs, rank=2, url="https://b.com/p")
    assert a != b
    assert content_id(b"same") == content_id(b"same")


def test_sha256_hex_rejects_str():
    import pytest

    with pytest.raises(TypeError):
        sha256_hex("not bytes")  # type: ignore[arg-type]
