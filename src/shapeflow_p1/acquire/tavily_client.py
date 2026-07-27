"""The Tavily capture client -- the one place that spends Tavily budget.

Used only during acquisition. It is written around an injected async transport so it can be
driven by a fake in tests and never touches the network there; the live transport is wired
only at bootstrap. Every call is threaded through the external-call FSM: reserve worst-case
budget, mark sent, store the *redacted* request/response, then settle at the reported usage.

Explicit parameters, never ``auto_parameters``: the same query must address the same slice
of the web on every acquisition, so the retrieval parameters are pinned and recorded. The
full raw response hash, request id, response time, usage and failed-results are all captured;
we never keep only the convenience ``content`` field.

The credential is read from a file path (or, in local dev, an env var), registered with the
redactor, and placed in the request body -- never logged, never in argv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from ..hashing import query_snapshot_id, sha256_hex
from ..secrets import SecretRedactor
from ..world.source_pool import QueryResponse, RawResult

__all__ = ["TavilyParams", "TavilyCaptureClient", "load_tavily_key"]

TAVILY_API_VERSION = "tavily-search-v1"
TAVILY_SCHEMA_VERSION = "v1"


def load_tavily_key(redactor: SecretRedactor) -> str:
    """Read the Tavily key from TAVILY_API_KEY_FILE (preferred) or TAVILY_API_KEY (dev), and
    register it for redaction. Never returns it via logs; callers put it only in the body."""
    path = os.environ.get("TAVILY_API_KEY_FILE")
    if path:
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    else:
        key = (os.environ.get("TAVILY_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "no Tavily credential: set TAVILY_API_KEY_FILE (production) or TAVILY_API_KEY (dev)"
        )
    redactor.register(key, label="tavily")
    return key


@dataclass(frozen=True)
class TavilyParams:
    """The pinned acquisition parameters. auto_parameters is deliberately absent."""

    search_depth: str = "advanced"
    include_raw_content: str = "markdown"
    include_answer: bool = False
    include_usage: bool = True
    max_results: int = 8
    topic: str = "general"

    def as_request(self, query: str, api_key: str) -> dict[str, Any]:
        return {
            "api_key": api_key,
            "query": query,
            "search_depth": self.search_depth,
            "include_raw_content": self.include_raw_content,
            "include_answer": self.include_answer,
            "include_usage": self.include_usage,
            "max_results": self.max_results,
            "topic": self.topic,
        }

    def snapshot_params(self) -> dict[str, Any]:
        """The parameters that define the query's identity -- everything except the key."""
        return {
            "search_depth": self.search_depth,
            "include_raw_content": self.include_raw_content,
            "include_answer": self.include_answer,
            "include_usage": self.include_usage,
            "max_results": self.max_results,
            "topic": self.topic,
        }


@dataclass
class CapturedQuery:
    """The frozen record of one acquisition query."""

    query_snapshot_id: str
    query_text: str
    request_id: str
    response_time: Optional[float]
    usage: dict[str, Any]
    failed_results: list[Any]
    raw_response_sha256: str
    response: QueryResponse


# transport(url, json_body) -> (status_code, response_dict). Injected so tests use a fake.
Transport = Callable[[str, dict], Awaitable[tuple[int, dict]]]


class TavilyCaptureClient:
    def __init__(
        self,
        transport: Transport,
        params: TavilyParams,
        api_key: str,
        *,
        endpoint: str = "https://api.tavily.com/search",
    ) -> None:
        self._transport = transport
        self._params = params
        self._api_key = api_key
        self._endpoint = endpoint

    def query_snapshot_id(self, query: str) -> str:
        return query_snapshot_id(
            query=query,
            params=self._params.snapshot_params(),
            api_version=TAVILY_API_VERSION,
            schema_version=TAVILY_SCHEMA_VERSION,
        )

    async def search(self, task_id: str, query: str) -> CapturedQuery:
        """Execute one acquisition query and capture everything about it.

        Budget/FSM integration is the caller's responsibility (it holds the ledger); this
        method performs the request and freezes the response. The returned raw hash and usage
        let the caller settle the reservation at actual cost.
        """
        qsid = self.query_snapshot_id(query)
        body = self._params.as_request(query, self._api_key)
        status, payload = await self._transport(self._endpoint, body)
        if status != 200:
            raise TavilyHTTPError(status, payload)

        results = tuple(
            RawResult(
                url=r["url"],
                title=r.get("title", ""),
                rank=i + 1,
                snippet=r.get("content", ""),
                raw_content=r.get("raw_content"),
                score=r.get("score"),
                published_date=r.get("published_date"),
            )
            for i, r in enumerate(payload.get("results", []))
        )
        # Hash the response with the key-free body echoed, so the raw hash is reproducible and
        # never depends on the secret.
        raw_hash = sha256_hex(
            _stable_response_bytes(query, self._params.snapshot_params(), payload)
        )
        return CapturedQuery(
            query_snapshot_id=qsid,
            query_text=query,
            request_id=str(payload.get("request_id", "")),
            response_time=payload.get("response_time"),
            usage=payload.get("usage", {}) or {},
            failed_results=payload.get("failed_results", []) or [],
            raw_response_sha256=raw_hash,
            response=QueryResponse(
                query_snapshot_id=qsid, query_text=query, results=results
            ),
        )


class TavilyHTTPError(RuntimeError):
    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(f"tavily returned HTTP {status}")
        self.status = status
        self.payload = payload


def _stable_response_bytes(query: str, params: dict, payload: dict) -> bytes:
    from ..canonical import canonical_json

    return canonical_json({"query": query, "params": params, "payload": payload})
