"""The document store: docid to text and url, from the benchmark's parquet shards.

Treatment-visible by design. This holds the corpus every arm retrieves from and nothing about
which documents are relevant -- the qrels, the graded answer and the negative sets live in the
evaluator tree, behind the leakage firewall.

Loaded eagerly. The corpus is ~1.7 GB of text over ~100k documents, and the alternative -- a
row-offset index with a read per hit -- would put filesystem latency inside the agent's search
path, where it would be indistinguishable from model latency in every measurement that matters.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping

from ..hashing import sha256_hex

__all__ = ["Document", "CorpusStore", "load_corpus"]


@dataclass(frozen=True)
class Document:
    docid: str
    text: str
    url: str

    @property
    def content_sha256(self) -> str:
        return sha256_hex(self.text.encode("utf-8"))


class CorpusStore:
    """docid -> Document, with the shard digests that identify which corpus this is."""

    def __init__(self, documents: Mapping[str, Document], *, shard_sha256: tuple[str, ...] = ()):
        self._docs = dict(documents)
        self.shard_sha256 = tuple(shard_sha256)

    def __len__(self) -> int:
        return len(self._docs)

    def __contains__(self, docid: object) -> bool:
        return docid in self._docs

    def __iter__(self) -> Iterator[str]:
        return iter(self._docs)

    def get(self, docid: str) -> Document:
        try:
            return self._docs[docid]
        except KeyError:
            # Never a silent miss: a retrieved docid with no document means the index and the
            # corpus are different vintages, and every result built from them is unattributable.
            raise KeyError(
                f"docid {docid!r} is in the index but not in the corpus; the index and corpus "
                "are from different builds") from None

    def texts(self, docids) -> dict[str, str]:
        return {d: self.get(d).text for d in docids}


def load_corpus(directory: Path, *, pattern: str = "*.parquet") -> CorpusStore:
    """Read every parquet shard in ``directory`` into memory."""
    import pyarrow.parquet as pq

    paths = sorted(Path(directory).glob(pattern))
    if not paths:
        raise FileNotFoundError(f"no corpus shards matching {pattern!r} under {directory}")

    documents: dict[str, Document] = {}
    digests: list[str] = []
    for path in paths:
        table = pq.read_table(path, columns=["docid", "text", "url"])
        for docid, text, url in zip(table.column("docid").to_pylist(),
                                    table.column("text").to_pylist(),
                                    table.column("url").to_pylist()):
            documents[str(docid)] = Document(docid=str(docid), text=text or "", url=url or "")
        # Digest the file rather than its contents-in-memory: it identifies the artifact on disk,
        # which is what a manifest can be checked against later.
        digests.append(sha256_hex(path.read_bytes()))
    return CorpusStore(documents, shard_sha256=tuple(digests))
