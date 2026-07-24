"""The Exa capture client -- the one place that spends search budget.

Replaces the Tavily client for acquisition. The switch is not cosmetic: the identity of a
frozen query, the record shape a snapshot is built from, and the unit the budget is settled
in all change with the vendor, so each is pinned explicitly here rather than inherited.

**Every parameter is pinned, and ``type: "auto"`` is refused.** Exa's ``auto`` mode picks a
search strategy per query, which is the same defect as Tavily's ``auto_parameters``: the
same query would address a different slice of the web on a later acquisition, and a frozen
world that cannot be re-derived is not frozen. The pinned type, result count and content
options all enter the query snapshot id.

**The mapping into the vendor-neutral record is versioned.** Exa's ``text`` is the cleaned
page text and becomes ``raw_content``; its ``highlights`` are the query-relevant excerpts
and become the short ``content`` snippet. That keeps the node named for what it is --
``PAGE_RAW_CONTENT``, cleaned text, never browser HTML -- and keeps P0 and P1 reading the
same bytes through the pinned ODR truncation. Exa returns no relevance score, so the
occurrence records ``None`` rather than a fabricated one.

**Cost is settled in dollars the vendor reported.** Exa returns ``costDollars`` per
response, so the ledger holds the real amount instead of a credit count converted by
assumption.

The credential is read from a file path (or, in local dev, an env var), registered with the
redactor, and sent in the ``x-api-key`` header -- never logged, never in argv.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..hashing import query_snapshot_id, sha256_hex
from ..secrets import SecretRedactor
from .source_pool import QueryResponse, RawResult

__all__ = [
    "ExaParams",
    "ExaCaptureClient",
    "ExaHTTPError",
    "load_exa_key",
    "EXA_API_VERSION",
    "EXA_RECORD_MAPPING_VERSION",
]

EXA_API_VERSION = "exa-search-2026-03"
EXA_SCHEMA_VERSION = "v1"
#: Bumped whenever the Exa field -> vendor-neutral record mapping below changes. It enters
#: the snapshot identity, so a remapped world can never be mistaken for the original one.
EXA_RECORD_MAPPING_VERSION = "exa_text_v1"

#: Exa search types. `auto` is excluded on purpose -- see the module docstring.
PINNABLE_TYPES = frozenset({"instant", "fast", "deep-lite", "deep", "deep-reasoning"})


def load_exa_key(redactor: SecretRedactor) -> str:
    """Read the Exa key from EXA_API_KEY_FILE (preferred) or EXA_API_KEY (dev), and register
    it for redaction. Never returned via logs; the caller puts it only in a header."""
    path = os.environ.get("EXA_API_KEY_FILE")
    if path:
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    else:
        key = (os.environ.get("EXA_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "no Exa credential: set EXA_API_KEY_FILE (production) or EXA_API_KEY (dev)"
        )
    redactor.register(key, label="exa")
    return key


@dataclass(frozen=True)
class ExaParams:
    """The pinned acquisition parameters. ``type: auto`` is rejected in __post_init__."""

    type: str = "fast"
    num_results: int = 8
    text_max_characters: int = 100_000
    include_html_tags: bool = False
    highlights: bool = True
    category: Optional[str] = None

    def __post_init__(self) -> None:
        if self.type not in PINNABLE_TYPES:
            raise ValueError(
                f"exa type {self.type!r} is not pinnable (allowed: {sorted(PINNABLE_TYPES)}). "
                "'auto' picks a strategy per query, so the same query would address a "
                "different slice of the web on a later acquisition"
            )
        if not 1 <= self.num_results <= 100:
            raise ValueError(f"num_results {self.num_results} is outside Exa's 1-100 range")

    def as_request(self, query: str) -> dict[str, Any]:
        body: dict[str, Any] = {
            "query": query,
            "type": self.type,
            "numResults": self.num_results,
            "contents": {
                "text": {
                    "maxCharacters": self.text_max_characters,
                    "includeHtmlTags": self.include_html_tags,
                },
                "highlights": self.highlights,
            },
        }
        if self.category:
            body["category"] = self.category
        return body

    def snapshot_params(self) -> dict[str, Any]:
        """Everything that defines the query's identity. No credential appears here."""
        return {
            "provider": "exa",
            "type": self.type,
            "num_results": self.num_results,
            "text_max_characters": self.text_max_characters,
            "include_html_tags": self.include_html_tags,
            "highlights": self.highlights,
            "category": self.category,
            "record_mapping": EXA_RECORD_MAPPING_VERSION,
        }

    def worst_case_usd(self, pricing: "ExaPricing") -> float:
        """The most one search can cost, from the pinned result count and the price list."""
        extra = max(0, self.num_results - pricing.results_included)
        return pricing.usd_per_request + extra * pricing.usd_per_extra_result


@dataclass(frozen=True)
class ExaPricing:
    """A snapshot of the published price list, carried in configs/acquisition.yaml.

    Kept as data rather than constants so the reservation is derived from the same numbers
    the report cites, and a price change is a config change with a new protocol SHA.
    """

    usd_per_request: float = 0.007
    usd_per_extra_result: float = 0.001
    results_included: int = 10
    source: str = "https://exa.ai/docs/changelog/pricing-update"
    retrieved_utc: str = "2026-07-24"


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
    cost_usd: float = 0.0


# transport(url, json_body) -> (status_code, response_dict). Injected so tests use a fake.
Transport = Callable[[str, dict], Awaitable[tuple[int, dict]]]


class ExaCaptureClient:
    def __init__(
        self,
        transport: Transport,
        params: ExaParams,
        *,
        endpoint: str = "https://api.exa.ai/search",
    ) -> None:
        self._transport = transport
        self._params = params
        self._endpoint = endpoint

    def query_snapshot_id(self, query: str) -> str:
        """The identity of this exact query under these exact parameters.

        The provider name and the record-mapping version are inside the digest, so an Exa
        snapshot can never collide with a Tavily one taken for the same query text -- two
        different worlds under one id would be indistinguishable afterwards.
        """
        return query_snapshot_id(
            query=query,
            params=self._params.snapshot_params(),
            api_version=EXA_API_VERSION,
            schema_version=EXA_SCHEMA_VERSION,
        )

    async def search(self, task_id: str, query: str) -> CapturedQuery:
        """Execute one acquisition query and capture everything about it."""
        qsid = self.query_snapshot_id(query)
        body = self._params.as_request(query)
        status, payload = await self._transport(self._endpoint, body)
        if status != 200:
            raise ExaHTTPError(status, payload)

        results = tuple(
            RawResult(
                url=r["url"],
                title=r.get("title") or "",
                rank=i + 1,
                # Exa's highlights are the query-relevant excerpts, which is what Tavily's
                # short `content` field was; the full page text is `text`.
                snippet=_join_highlights(r.get("highlights")),
                raw_content=r.get("text"),
                # Exa does not return a relevance score. Recording None keeps the absence
                # visible instead of inventing a number the audit graph would then rank by.
                score=None,
                published_date=r.get("publishedDate"),
            )
            for i, r in enumerate(payload.get("results", []))
        )
        raw_hash = sha256_hex(
            _stable_response_bytes(query, self._params.snapshot_params(), payload)
        )
        cost = payload.get("costDollars") or {}
        return CapturedQuery(
            query_snapshot_id=qsid,
            query_text=query,
            request_id=str(payload.get("requestId", "")),
            response_time=payload.get("response_time"),
            usage={"cost_dollars": cost, "result_count": len(results)},
            failed_results=payload.get("failed_results", []) or [],
            raw_response_sha256=raw_hash,
            response=QueryResponse(
                query_snapshot_id=qsid, query_text=query, results=results
            ),
            cost_usd=float(cost.get("total") or 0.0),
        )


class ExaHTTPError(RuntimeError):
    def __init__(self, status: int, payload: Any) -> None:
        super().__init__(f"exa returned HTTP {status}")
        self.status = status
        self.payload = payload


def _join_highlights(highlights: Any) -> str:
    if not highlights:
        return ""
    if isinstance(highlights, str):
        return highlights
    return " … ".join(str(h) for h in highlights if str(h).strip())


def _stable_response_bytes(query: str, params: dict, payload: dict) -> bytes:
    from ..canonical import canonical_json

    return canonical_json({"query": query, "params": params, "payload": payload})
