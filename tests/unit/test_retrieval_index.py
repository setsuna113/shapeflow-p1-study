"""The dense index: exactness, determinism, and refusing a misaligned or unnormalised input.

Runs on synthetic shards. The real index lives on the run host and is checked by the re-encode
conformance test; what is checked here is the search itself, which must behave identically on
both.
"""

from __future__ import annotations

import pickle

import numpy as np
import pytest

from shapeflow.retrieval.index import DenseIndex, load_index


def _unit(rows: int, dim: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    v = rng.standard_normal((rows, dim)).astype(np.float32)
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def _write_shards(tmp_path, shards):
    for i, (vectors, docids) in enumerate(shards, start=1):
        with open(tmp_path / f"corpus.shard{i}_of_{len(shards)}.pkl", "wb") as fh:
            pickle.dump((vectors, docids), fh)
    return tmp_path


def test_a_document_retrieves_itself_at_rank_one(tmp_path):
    vectors = _unit(64, 32)
    docids = [f"d{i}" for i in range(64)]
    index = DenseIndex(vectors, docids)
    for i in (0, 17, 63):
        hits = index.search(vectors[i], top_k=5)
        assert hits[0].docid == docids[i]
        assert hits[0].rank == 1
        assert hits[0].score == pytest.approx(1.0, abs=1e-5)


def test_shards_concatenate_in_sorted_filename_order(tmp_path):
    a, b = _unit(8, 16, seed=1), _unit(8, 16, seed=2)
    _write_shards(tmp_path, [(a, [f"a{i}" for i in range(8)]),
                             (b, [f"b{i}" for i in range(8)])])
    index = load_index(tmp_path)
    assert index.num_docs == 16 and index.dim == 16
    assert len(index.shards) == 2
    assert all(s.sha256 for s in index.shards), "each shard must record its digest"
    assert index.search(a[3], top_k=1)[0].docid == "a3"
    assert index.search(b[5], top_k=1)[0].docid == "b5"


def test_ties_are_broken_by_docid_so_results_are_reproducible():
    """Identical vectors under different ids must always come back in the same order.

    With normalised float32 vectors, exact score ties are common. Left to argpartition they would
    resolve by whatever order the array happened to be in, and two runs of one query would
    retrieve different documents -- so two arms would see different worlds for reasons no
    artifact would record.
    """
    row = _unit(1, 8)[0]
    vectors = np.vstack([row] * 5)
    index = DenseIndex(vectors, ["z", "m", "a", "q", "b"])
    ranked = [h.docid for h in index.search(row, top_k=5)]
    assert ranked == ["a", "b", "m", "q", "z"]
    for _ in range(5):
        assert [h.docid for h in index.search(row, top_k=5)] == ranked


def test_an_unnormalised_query_is_refused():
    """An unnormalised query preserves rank order, so the damage shows only in the scores."""
    index = DenseIndex(_unit(8, 16), [f"d{i}" for i in range(8)])
    with pytest.raises(ValueError, match="not L2-normalised"):
        index.search(np.ones(16, dtype=np.float32) * 3.0, top_k=2)


def test_a_misaligned_index_is_refused_at_construction():
    with pytest.raises(ValueError, match="misaligned"):
        DenseIndex(_unit(4, 8), ["a", "b"])


def test_duplicate_docids_are_refused():
    with pytest.raises(ValueError, match="duplicate docids"):
        DenseIndex(_unit(3, 8), ["a", "a", "b"])


def test_a_dimension_mismatch_is_refused():
    index = DenseIndex(_unit(4, 8), list("abcd"))
    with pytest.raises(ValueError, match="dim"):
        index.search(_unit(1, 16)[0], top_k=1)


def test_shards_of_differing_dimension_are_refused(tmp_path):
    _write_shards(tmp_path, [(_unit(4, 8), list("abcd")), (_unit(4, 16), list("efgh"))])
    with pytest.raises(ValueError, match="disagree on dimension"):
        load_index(tmp_path)


def test_top_k_larger_than_the_corpus_returns_the_whole_corpus():
    index = DenseIndex(_unit(3, 8), list("abc"))
    assert len(index.search(_unit(1, 8)[0], top_k=99)) == 3


def test_an_empty_directory_is_an_error_not_an_empty_index(tmp_path):
    """An empty index would answer every query with nothing, which reads exactly like a corpus
    that genuinely contains nothing relevant."""
    with pytest.raises(FileNotFoundError, match="no index shards"):
        load_index(tmp_path)
