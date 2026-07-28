"""The conformance check must pass a correct encoder and fail every wrong one.

Uses a fake encoder rather than a real model: what is under test here is the *check*, and a check
that has only ever been run against a correct encoder is an assumption. The real model is
exercised by `scripts/retrieval_conformance.py` on the run host, against the shipped index.
"""

from __future__ import annotations

import numpy as np
import pytest

from shapeflow.retrieval.conformance import DEFAULT_TOLERANCE, check_encoder_matches_index
from shapeflow.retrieval.encoder import EncoderSpec
from shapeflow.retrieval.index import DenseIndex


class FakeEncoder:
    """Embeds text by a fixed random projection of its bytes. Deterministic, and 'correct'
    in the sense that the index below was built with the same projection."""

    def __init__(self, dim: int = 32, *, perturbation: float = 0.0, seed: int = 7) -> None:
        self.spec = EncoderSpec(model="fake/encoder", dtype="float32")
        self.dim = dim
        self.perturbation = perturbation
        self._basis = np.random.default_rng(seed).standard_normal((256, dim)).astype(np.float32)

    def _embed(self, text: str) -> np.ndarray:
        counts = np.zeros(256, dtype=np.float32)
        for byte in text.encode("utf-8"):
            counts[byte] += 1.0
        v = counts @ self._basis
        if self.perturbation:
            v = v + self.perturbation * np.random.default_rng(abs(hash(text)) % 2**32
                                                              ).standard_normal(self.dim)
        return (v / np.linalg.norm(v)).astype(np.float32)

    def encode(self, texts, *, is_query: bool):
        return np.vstack([self._embed(t) for t in texts])


DOCS = {f"d{i}": f"document number {i} about measurements and units observed in 2025"
        for i in range(12)}


def _index_from(encoder, docs) -> DenseIndex:
    vectors = encoder.encode(list(docs.values()), is_query=False)
    return DenseIndex(vectors, list(docs))


def test_a_correct_encoder_passes():
    encoder = FakeEncoder()
    result = check_encoder_matches_index(encoder, _index_from(encoder, DOCS), DOCS)
    assert result.ok
    assert result.checked == len(DOCS)
    assert result.min_cosine >= DEFAULT_TOLERANCE
    assert "reproduces" in result.diagnosis()


def test_a_wrong_recipe_fails_loudly_and_names_the_likely_cause():
    """A different projection stands in for the real failure -- the query instruction applied to
    passages, or the wrong max length -- all of which produce confidently wrong vectors."""
    index = _index_from(FakeEncoder(seed=7), DOCS)
    result = check_encoder_matches_index(FakeEncoder(seed=99), index, DOCS)
    assert not result.ok
    assert result.min_cosine < 0.5
    assert "passage_prefix is empty" in result.diagnosis()
    assert "passage_max_len is 4096" in result.diagnosis()


def test_small_numerical_drift_is_tolerated():
    """fp16-on-GPU versus fp32-on-CPU must not fail; only a wrong recipe should."""
    index = _index_from(FakeEncoder(seed=7), DOCS)
    result = check_encoder_matches_index(FakeEncoder(seed=7, perturbation=1e-4), index, DOCS)
    assert result.ok, f"tiny drift was rejected (min cosine {result.min_cosine:.6f})"


def test_drift_beyond_tolerance_is_not_tolerated():
    index = _index_from(FakeEncoder(seed=7), DOCS)
    result = check_encoder_matches_index(FakeEncoder(seed=7, perturbation=5.0), index, DOCS)
    assert not result.ok


def test_a_misaligned_index_fails_self_retrieval_even_with_perfect_cosines():
    """Cosine alone cannot catch a docid list rotated against its vectors.

    The vectors are right, so every cosine is 1.0 -- and every document retrieves its neighbour.
    Without the self-retrieval assertion this passes while retrieval returns the wrong document
    for every query.
    """
    encoder = FakeEncoder()
    vectors = encoder.encode(list(DOCS.values()), is_query=False)
    rotated = list(DOCS)[1:] + list(DOCS)[:1]
    index = DenseIndex(vectors, rotated)

    result = check_encoder_matches_index(encoder, index, {k: DOCS[k] for k in rotated})
    assert not result.ok
    assert result.min_cosine < DEFAULT_TOLERANCE or any(
        d.self_retrieved_rank != 1 for d in result.docs)


def test_an_empty_check_is_an_error_not_a_pass():
    encoder = FakeEncoder()
    with pytest.raises(ValueError, match="passes vacuously"):
        check_encoder_matches_index(encoder, _index_from(encoder, DOCS), {})


def test_the_result_records_what_it_checked_against():
    """A pass has to be re-derivable: which index, which encoder settings."""
    encoder = FakeEncoder()
    result = check_encoder_matches_index(encoder, _index_from(encoder, DOCS), DOCS)
    body = result.content()
    assert body["encoder_spec"]["model"] == "fake/encoder"
    assert body["encoder_spec"]["query_prefix_sha256"], "the prefix must be pinned by digest"
    assert body["encoder_spec"]["passage_prefix_sha256"] != \
        body["encoder_spec"]["query_prefix_sha256"], (
        "query and passage prefixes must differ; treating them as one is the failure this "
        "whole check exists to catch")
