"""Prove our encoder path reproduces the vectors the shipped index was built from.

This is the cheapest check in the program and guards the most expensive failure. Every way of
getting the encoder subtly wrong -- the query instruction applied to passages, the wrong maximum
length, mean pooling instead of last-token, a missing normalise, right padding, a different model
revision -- produces an encoder that runs, returns plausible vectors, and retrieves badly.

Retrieval then collapses toward BM25's 6% gold recall. The agent stops finding evidence, P0
accuracy falls under the competence floor, and the honest-looking conclusion is "the retriever is
not good enough for this benchmark". The actual cause is a one-line mismatch, and nothing
downstream distinguishes the two.

So: re-encode documents whose vectors are already in the index, and require them back. Two
assertions, because they fail differently:

- **cosine against the stored vector** catches a systematically wrong recipe;
- **rank-1 self-retrieval through our own search path** catches everything between the encoder
  and the ranking -- a transposed matrix, a misaligned docid list, a broken tie-break.

The tolerance is 0.999 rather than exact. The shipped vectors were produced in fp16 on a GPU and
ours in fp32 on a CPU, so bit-equality is not available and demanding it would fail for the one
reason that does not matter.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

__all__ = ["ConformanceResult", "DocCheck", "check_encoder_matches_index", "DEFAULT_TOLERANCE"]

#: Cosine floor. Anything below this is a recipe mismatch, not numerical drift.
DEFAULT_TOLERANCE = 0.999


@dataclass(frozen=True)
class DocCheck:
    docid: str
    cosine: float
    self_retrieved_rank: int | None

    @property
    def ok(self) -> bool:
        return self.cosine >= DEFAULT_TOLERANCE and self.self_retrieved_rank == 1


@dataclass
class ConformanceResult:
    checked: int = 0
    tolerance: float = DEFAULT_TOLERANCE
    docs: list[DocCheck] = field(default_factory=list)
    spec: Mapping = field(default_factory=dict)
    index_digests: Sequence[str] = ()

    @property
    def failures(self) -> list[DocCheck]:
        return [d for d in self.docs if not d.ok]

    @property
    def ok(self) -> bool:
        return bool(self.docs) and not self.failures

    @property
    def min_cosine(self) -> float:
        return min((d.cosine for d in self.docs), default=0.0)

    def content(self) -> dict:
        return {
            "ok": self.ok,
            "checked": self.checked,
            "tolerance": self.tolerance,
            "min_cosine": round(self.min_cosine, 6),
            "rank_1_count": sum(1 for d in self.docs if d.self_retrieved_rank == 1),
            "failures": [
                {"docid": d.docid, "cosine": round(d.cosine, 6),
                 "self_retrieved_rank": d.self_retrieved_rank}
                for d in self.failures[:20]
            ],
            "encoder_spec": dict(self.spec),
            "index_shard_sha256": list(self.index_digests),
        }

    def diagnosis(self) -> str:
        """Name the likely cause, because the symptom is the same for all of them."""
        if self.ok:
            return "encoder reproduces the shipped index"
        worst = min(self.docs, key=lambda d: d.cosine)
        if worst.cosine < 0.5:
            return (
                f"min cosine {worst.cosine:.4f} -- the recipe is wrong, not drifting. Check, in "
                "order: the query instruction must NOT be applied to passages (passage_prefix is "
                "empty); passage_max_len is 4096 not 512; pooling is last-token; normalise is on; "
                "padding_side is left; the model revision matches the one the index was built with."
            )
        if worst.cosine < DEFAULT_TOLERANCE:
            return (
                f"min cosine {worst.cosine:.4f} is close but under {DEFAULT_TOLERANCE}. Likely a "
                "dtype or revision difference rather than a wrong recipe; confirm the model "
                "revision before relaxing anything."
            )
        return ("cosines pass but self-retrieval does not: the encoder is right and something "
                "between it and the ranking is wrong -- docid alignment, or the tie-break.")


def check_encoder_matches_index(encoder, index, docs: Mapping[str, str], *,
                                tolerance: float = DEFAULT_TOLERANCE) -> ConformanceResult:
    """Re-encode ``docs`` (docid -> text) and compare against the index.

    ``docs`` must be documents the index already contains, so the comparison is against a stored
    vector rather than against another run of the same code -- which would agree with itself no
    matter how wrong it was.
    """
    import numpy as np

    result = ConformanceResult(tolerance=tolerance, spec=encoder.spec.content(),
                               index_digests=[s.sha256 for s in index.shards])
    docids = list(docs)
    if not docids:
        raise ValueError("no documents to check; a conformance run over nothing passes vacuously")

    vectors = encoder.encode([docs[d] for d in docids], is_query=False)
    for docid, vector in zip(docids, vectors):
        stored = index.vector_for(docid)
        cosine = float(np.dot(vector, stored) /
                       (np.linalg.norm(vector) * np.linalg.norm(stored)))
        hits = index.search(vector, top_k=1)
        rank = hits[0].rank if hits and hits[0].docid == docid else None
        result.docs.append(DocCheck(docid=docid, cosine=cosine, self_retrieved_rank=rank))
    result.checked = len(result.docs)
    return result
