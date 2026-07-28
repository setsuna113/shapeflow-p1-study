"""The dense index: shard files in, ranked docids out.

The index is a set of pickled ``(float32[N, dim], list[docid])`` shards published with the
benchmark. Its vectors are already L2-normalised, so the inner product *is* cosine similarity and
search is one matrix-vector product. There is no approximate structure and none is wanted: exact
search over ~100k vectors costs milliseconds, and an ANN index would introduce a recall knob that
every downstream comparison would then silently depend on.

Two properties this module exists to guarantee:

**Determinism.** Ties are broken by docid, always. Score ties are common with normalised vectors
at float32, and ``argpartition`` alone does not order them reproducibly -- two runs of the same
query would return different documents in a different order, and every arm would retrieve a
slightly different world.

**Verified identity.** The shard digests are recorded, so the index a run used is a fact rather
than a filename.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from ..hashing import sha256_hex

__all__ = ["DenseIndex", "IndexShard", "SearchHit", "load_index", "shard_digests"]


@dataclass(frozen=True)
class SearchHit:
    docid: str
    score: float
    rank: int


@dataclass(frozen=True)
class IndexShard:
    path: Path
    num_docs: int
    dim: int
    sha256: str


class DenseIndex:
    """Exact inner-product search over the concatenated shards."""

    def __init__(self, vectors: np.ndarray, docids: Sequence[str],
                 shards: Sequence[IndexShard] = ()) -> None:
        if vectors.ndim != 2:
            raise ValueError(f"expected a 2-D matrix, got shape {vectors.shape}")
        if len(docids) != vectors.shape[0]:
            raise ValueError(
                f"{len(docids)} docids for {vectors.shape[0]} vectors; the index is misaligned "
                "and every result would name the wrong document")
        if len(set(docids)) != len(docids):
            raise ValueError("duplicate docids in the index; a hit would be ambiguous")
        self._vectors = np.ascontiguousarray(vectors, dtype=np.float32)
        self._docids = list(docids)
        self._order = np.argsort(np.asarray(self._docids))  # stable tie-break, precomputed
        self.shards = tuple(shards)

    @property
    def num_docs(self) -> int:
        return self._vectors.shape[0]

    @property
    def dim(self) -> int:
        return int(self._vectors.shape[1])

    def search(self, query_vector: np.ndarray, *, top_k: int) -> list[SearchHit]:
        """Rank every document against ``query_vector`` and return the best ``top_k``.

        The query vector must be L2-normalised, like the stored ones. Not normalising it would
        still produce a ranking -- an unnormalised query preserves the *order* of an inner
        product -- so the failure would be invisible in the ranks and visible only in the scores,
        which is why it is checked rather than assumed.
        """
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if q.shape[0] != self.dim:
            raise ValueError(f"query has dim {q.shape[0]}, index has dim {self.dim}")
        norm = float(np.linalg.norm(q))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ValueError(
                f"query vector is not L2-normalised (norm {norm:.4f}); the stored vectors are, "
                "so scores would not be cosine similarities and any threshold on them would be "
                "meaningless")
        if top_k <= 0:
            raise ValueError("top_k must be positive")

        scores = self._vectors @ q
        k = min(top_k, self.num_docs)
        # Sort by (-score, docid): argsort over the docid order first, then a stable sort by
        # score, so equal scores come back in docid order on every run and every machine.
        candidate = self._order[np.argsort(-scores[self._order], kind="stable")][:k]
        return [
            SearchHit(docid=self._docids[i], score=float(scores[i]), rank=rank)
            for rank, i in enumerate(candidate, start=1)
        ]

    def has(self, docid: str) -> bool:
        return docid in set(self._docids)

    def vector_for(self, docid: str) -> np.ndarray:
        """The stored vector for one docid. Used by the re-encode conformance check."""
        try:
            return self._vectors[self._docids.index(docid)]
        except ValueError:
            raise KeyError(f"{docid!r} is not in this index") from None


def shard_digests(paths: Sequence[Path]) -> list[IndexShard]:
    """Digest and describe each shard without keeping its vectors in memory."""
    out = []
    for path in paths:
        with open(path, "rb") as handle:
            vectors, docids = pickle.load(handle)
        out.append(IndexShard(path=Path(path), num_docs=len(docids),
                              dim=int(vectors.shape[1]),
                              sha256=sha256_hex(Path(path).read_bytes())))
    return out


def load_index(directory: Path, *, pattern: str = "corpus.shard*.pkl") -> DenseIndex:
    """Load every shard in ``directory``, in sorted filename order.

    Sorted rather than glob order so the concatenation is identical on every filesystem; the
    docid list is what identifies a row, but a stable order keeps digests reproducible.
    """
    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no index shards matching {pattern!r} under {directory}")

    blocks: list[np.ndarray] = []
    docids: list[str] = []
    shards: list[IndexShard] = []
    for path in paths:
        with open(path, "rb") as handle:
            vectors, ids = pickle.load(handle)
        vectors = np.asarray(vectors, dtype=np.float32)
        blocks.append(vectors)
        docids.extend(str(d) for d in ids)
        shards.append(IndexShard(path=path, num_docs=len(ids), dim=int(vectors.shape[1]),
                                 sha256=sha256_hex(path.read_bytes())))

    dims = {s.dim for s in shards}
    if len(dims) != 1:
        raise ValueError(f"shards disagree on dimension: {sorted(dims)}")
    return DenseIndex(np.vstack(blocks), docids, shards)
