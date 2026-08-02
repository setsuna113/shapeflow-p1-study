"""The BrowseComp-Plus search backend: the frozen world every arm retrieves from.

Implements :class:`~shapeflow.world.search_backend.SearchBackend`, so the patched graph reaches
it through the same seam the Week-1 frozen pool used and nothing downstream of the search call
knows the difference.

Full documents are returned, not snippets. That is the declared deviation from the benchmark's
official setting: the H boundary exists to compress long pages, and a snippet workload would
delete the phenomenon under study. Truncation to the shared budget happens where it happens for
every other backend -- once, identically for P0 and P1.

The query encoder is reached through an injected callable rather than constructed here, because
it lives in another process with a torch dependency this package deliberately does not have.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Mapping

from ..hashing import derive_id
from ..world.search_backend import SearchRecord
from .corpus import CorpusStore
from .index import DenseIndex

__all__ = ["BrowseCompPlusBackend", "EncodeFn"]

#: query text -> L2-normalised vector. Supplied by the caller; see the module docstring.
EncodeFn = Callable[[str], "object"]


@dataclass
class BrowseCompPlusBackend:
    """Dense retrieval over the frozen BC+ corpus, returning whole documents."""

    index: DenseIndex
    corpus: CorpusStore
    encode: EncodeFn
    #: Cache of query text -> vector. Safe for correctness: an embedding is a pure function of
    #: the text and the pinned encoder. It is NOT arm-salted, deliberately -- salting it per arm
    #: would give whichever arm ran second a cold cache and a latency penalty that has nothing to
    #: do with its compression form, which is the very asymmetry the APC salt exists to prevent.
    cache: dict = field(default_factory=dict)
    queries_seen: list = field(default_factory=list)

    def search(self, query: str, *, max_results: int) -> list[SearchRecord]:
        vector = self.cache.get(query)
        if vector is None:
            vector = self.encode(query)
            self.cache[query] = vector
        self.queries_seen.append(query)

        records: list[SearchRecord] = []
        for hit in self.index.search(vector, top_k=max_results):
            document = self.corpus.get(hit.docid)
            records.append(SearchRecord(
                url=document.url,
                title=_title_of(document.text, document.url),
                content=_snippet_of(document.text),
                raw_content=document.text,
                score=hit.score,
                # Derived the same way the acquisition path derived it, so an occurrence is
                # identified by (query, rank, url) rather than by position in a list.
                occurrence_id=derive_id("occurrence", {
                    "query_snapshot_id": derive_id("query_snapshot", {
                        "query": query,
                        "params": {"top_k": max_results},
                        "api_version": "bcplus-dense-v1",
                        "schema_version": "v1",
                    }),
                    "rank": hit.rank,
                    "url": document.url,
                }),
            ))
        return records

    @property
    def cache_hit_rate(self) -> float:
        """Reported per arm in S1. A spread between arms is a timing asymmetry, not a detail."""
        if not self.queries_seen:
            return 0.0
        return 1.0 - (len(self.cache) / len(self.queries_seen))

    def stats(self) -> Mapping[str, object]:
        return {
            "queries": len(self.queries_seen),
            "distinct_queries": len(self.cache),
            "cache_hit_rate": round(self.cache_hit_rate, 4),
            "index_num_docs": self.index.num_docs,
            "index_dim": self.index.dim,
        }


def _title_of(text: str, url: str) -> str:
    """The document's own title if its front matter carries one, else the url.

    The corpus stores a ``--- title: ... ---`` header on most documents. Falling back to the url
    rather than to a placeholder keeps the field meaningful: a title is what a researcher reads
    when deciding whether to open a page, and "Untitled" would change that decision.
    """
    for line in text.splitlines()[:6]:
        if line.lower().startswith("title:"):
            return line.split(":", 1)[1].strip() or url
    return url


def _snippet_of(text: str, *, chars: int = 320) -> str:
    """A short excerpt, standing where a search vendor's snippet would.

    Taken from the body rather than the front matter, so the snippet says something about the
    document instead of repeating its title.
    """
    body = text
    if body.startswith("---"):
        _, _, rest = body.partition("---")
        _, _, rest = rest.partition("---")
        body = rest or text
    return " ".join(body.split())[:chars]
