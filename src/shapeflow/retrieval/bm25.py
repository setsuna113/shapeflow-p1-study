"""BM25 over the BrowseComp-Plus corpus, reached over loopback: a lighter world than the dense one.

The dense retriever is the study's frozen world. This one exists for mechanism runs that need a
real agent searching a real corpus but not the dense encoder's 16 GB and CPU encode time. It is
a different world -- BM25 was rejected for the study on recall -- so results from it are never
comparable to the dense world's, and its occurrence ids carry their own ``api_version`` so the
two can never be confused.

Everything else matches the dense backend: whole documents are returned, the title and snippet
come from the same helpers, and every query's docids are kept in ``trace`` because a
``SearchRecord`` deliberately has no docid field.

The searcher (pyserini + Java) runs in ``scripts/bm25_service.py`` under its own environment;
this client is stdlib only.
"""

from __future__ import annotations

import json
import math
import threading
import urllib.error
import urllib.parse
import urllib.request

from ..hashing import derive_id
from ..world.search_backend import SearchRecord

__all__ = ["Bm25Client", "Bm25ServiceError", "API_VERSION"]

API_VERSION = "bcplus-bm25-v1"
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})
_ROW_FIELDS = frozenset({"docid", "rank", "score", "url", "title", "snippet", "text"})
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class Bm25ServiceError(RuntimeError):
    pass


class Bm25Client:
    def __init__(self, base_url: str, *, timeout_seconds: float = 60.0) -> None:
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http" or parsed.hostname not in LOOPBACK_HOSTS:
            raise Bm25ServiceError(f"refusing {base_url!r}: the BM25 world is loopback only")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self._lock = threading.Lock()
        #: One entry per query answered, in order: {"query", "top_k", "docids"}.
        self.trace: list[dict] = []

    def _post(self, path: str, body: dict) -> dict:
        request = urllib.request.Request(  # noqa: S310 - loopback only, checked above
            self.base_url + path, data=json.dumps(body).encode("utf-8"), method="POST",
            headers={"Content-Type": "application/json"})
        try:
            with _OPENER.open(request, timeout=self.timeout_seconds) as response:
                payload = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            raise Bm25ServiceError(f"{path} answered {e.code}: {e.read()[:200]!r}") from e
        except OSError as e:
            raise Bm25ServiceError(f"{path} is unreachable: {e}") from e
        if not isinstance(payload, dict):
            raise Bm25ServiceError(f"{path} returned {type(payload).__name__}")
        return payload

    def search_rows(self, query: str, *, max_results: int) -> list[dict]:
        payload = self._post("/v1/search", {"query": query, "top_k": int(max_results)})
        rows = payload.get("results")
        if payload.get("query") != query or not isinstance(rows, list):
            raise Bm25ServiceError("malformed search response")
        if len(rows) > max_results:
            raise Bm25ServiceError(f"asked for {max_results} results and got {len(rows)}")
        seen = set()
        for position, row in enumerate(rows, start=1):
            if not isinstance(row, dict) or set(row) != _ROW_FIELDS:
                raise Bm25ServiceError(f"result {position} does not have exactly {_ROW_FIELDS}")
            if row["rank"] != position or row["docid"] in seen or not row["docid"]:
                raise Bm25ServiceError(f"result {position} has a bad rank or docid")
            if not isinstance(row["score"], int | float) or not math.isfinite(row["score"]):
                raise Bm25ServiceError(f"result {position} has a non-finite score")
            seen.add(row["docid"])
        with self._lock:
            self.trace.append({"query": query, "top_k": int(max_results),
                               "docids": [row["docid"] for row in rows]})
        return rows

    def search(self, query: str, *, max_results: int) -> list[SearchRecord]:
        snapshot = derive_id("query_snapshot", {
            "query": query, "params": {"top_k": max_results},
            "api_version": API_VERSION, "schema_version": "v1",
        })
        return [
            SearchRecord(
                url=row["url"], title=row["title"], content=row["snippet"],
                raw_content=row["text"], score=float(row["score"]),
                occurrence_id=derive_id("occurrence", {
                    "query_snapshot_id": snapshot, "rank": row["rank"], "url": row["url"]}),
            )
            for row in self.search_rows(query, max_results=max_results)
        ]
