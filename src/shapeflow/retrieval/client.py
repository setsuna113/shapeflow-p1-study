"""The study package's end of the retrieval hop: a search backend that holds no model.

Drop-in for :class:`~shapeflow.world.search_backend.SearchBackend` -- ``search(query, *,
max_results) -> list[SearchRecord]`` -- so the patched graph reaches the frozen BrowseComp-Plus
corpus through the same seam the Week-1 frozen pool used, and nothing downstream of the search
call knows there is a process boundary in the middle.

Stdlib only, on purpose. This module is imported into the interpreter that runs the pinned vendor
agent, whose dependency closure is the system under measurement; it may not add a transport
library, let alone torch. One ``urllib`` request per search is the whole implementation.

Everything else here is refusal. Three failures are worth naming because each produces results
that look fine:

**A non-200 is not an empty result.** If a failed search returned ``[]``, "the service is down"
and "the corpus has nothing relevant" would be the same observation, and a run could complete
with a researcher that quietly saw nothing.

**A response that does not fit the schema is not partially usable.** The schema is closed in both
directions: an unenumerated field is a refusal, not something to ignore. That is the client half
of the leakage firewall -- a field carrying a qrel judgement or a gold flag has nowhere to land,
whatever a mis-configured service on the other end decides to send.

**A different retriever is not the frozen one.** The identity digest is checked once at
connection and again on every single response, because a service restarted against another index
between the two would otherwise be invisible, and the arms would be compared across two worlds.
The digest is also recomputed from the identity block it is supposed to summarise, so pinning a
freeze digest pins the object rather than a label the service printed beside it.

There is deliberately no retry. A retrieval service that is not answering is a stopped run, not a
slow one; retrying would convert a configuration error into a long silence.
"""

from __future__ import annotations

import json
import math
import re
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Mapping, Optional

from ..canonical import CanonicalizationError, canonical_json
from ..hashing import sha256_hex
from ..world.search_backend import SearchRecord

__all__ = [
    "RetrievalServiceError",
    "RetrievalClient",
    "connect",
]

#: Kept in step with :data:`shapeflow.retrieval.service.LOOPBACK_HOSTS` by a test rather than by
#: an import, so this module stays free of the service's numpy/pyarrow import graph.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

_RESPONSE_FIELDS = frozenset({"query", "top_k", "count", "retriever_id", "results"})
_RESULT_FIELDS = frozenset({
    "docid", "rank", "score", "url", "title", "snippet", "text", "occurrence_id"})
_TEXT_FIELDS = ("docid", "url", "title", "snippet", "text", "occurrence_id")
_HEX64 = re.compile(r"^[0-9a-f]{64}$")


class _RefuseRedirect(urllib.request.HTTPRedirectHandler):
    """Returning ``None`` here makes urllib raise the 3xx instead of chasing its ``Location``."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):  # noqa: D102
        return None


def _build_opener() -> urllib.request.OpenerDirector:
    """An opener that cannot leave the address ``__init__`` verified.

    Two stdlib defaults walk straight through the loopback check, and both were reachable before
    this existed:

    **Proxies.** ``urllib.request.urlopen`` builds its opener with a ``ProxyHandler`` populated
    from ``$http_proxy``. The run host's network is filtered and has one configured, so every
    "loopback" retrieval request was being handed to the proxy instead -- and whatever the proxy
    answered would have become the frozen world every arm searched. ``ProxyHandler({})`` takes the
    environment out of the decision entirely; ``no_proxy`` is not a substitute, because it is one
    more piece of host configuration this run's validity would silently depend on.

    **Redirects.** The default handler follows a 3xx anywhere, including to another machine and
    to a scheme this client refused to accept as a ``base_url``. One ``Location`` header would
    relocate the corpus after the check had passed, which is precisely the failure the loopback
    rule exists to prevent.
    """
    return urllib.request.build_opener(urllib.request.ProxyHandler({}), _RefuseRedirect)


#: Built once, at import: the handler set must not depend on the environment at call time.
_OPENER = _build_opener()


class RetrievalServiceError(RuntimeError):
    """The retrieval service refused, was unreachable, or answered something unusable."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(f"retrieval service ({status}): {message}")
        self.status = status


def _require_id_summarises_identity(retriever_id: str, identity: Mapping[str, Any]) -> None:
    """Require the advertised id to be the digest of the advertised identity.

    Without this the two halves of the identity check are pinned to different claims. The
    per-response check compares ``retriever_id`` -- an opaque string the service prints -- while
    the freeze check reads ``identity["freeze"]["digest"]``, and nothing binds them together. A
    caller that pins only the freeze digest (which is what a runner naturally has, since the
    freeze is a file in the repo and the retriever id is not) would then be checking a label
    beside the world rather than the world.

    Recomputing it makes the whole block content-addressed: the index shard digests, the corpus
    shard digests, the encoder recipe, the conformance evidence and the freeze all have to be the
    ones whose digest the responses carry. It is the same derivation the service uses, deliberately
    duplicated rather than imported, because importing the service would drag numpy and pyarrow
    into the interpreter running the pinned vendor agent. A test asserts the two agree.
    """
    try:
        recomputed = sha256_hex(canonical_json(dict(identity)))
    except CanonicalizationError as e:
        raise RetrievalServiceError(
            502, f"the retriever identity is not canonically encodable ({e}), so it cannot be "
                 "the block this service's digest was taken over") from e
    if recomputed != retriever_id:
        raise RetrievalServiceError(
            502,
            f"retriever_id {retriever_id[:12]} is not the digest of the identity this service "
            f"advertises (that hashes to {recomputed[:12]}). An id that does not summarise the "
            "block beside it pins nothing: the freeze, index and encoder it names could belong to "
            "some other retriever")


class RetrievalClient:
    """Search the frozen corpus over loopback, or fail. Never anything in between."""

    def __init__(
        self,
        base_url: str,
        *,
        expected_retriever_id: str = "",
        expected_freeze_digest: str = "",
        require_effective_freeze: bool = True,
        timeout_seconds: float = 120.0,
    ) -> None:
        """
        ``expected_retriever_id`` and ``expected_freeze_digest`` are the pins. Either one, when
        set, makes "am I talking to the retriever this run was designed against" a checked fact
        rather than a deployment assumption.

        ``require_effective_freeze`` defaults to **True**: a freeze whose ``effective_after`` is
        empty has not been validated by the P0 competence pilot, and letting an unvalidated
        retriever serve a campaign by default is precisely what that field exists to prevent. The
        pilot itself is the one caller that legitimately passes ``False``, and passing it is an
        explicit, greppable act rather than a silent default.
        """
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme != "http":
            raise RetrievalServiceError(
                500, f"base_url {base_url!r} must be http; the retrieval hop is loopback")
        if parsed.hostname not in LOOPBACK_HOSTS:
            raise RetrievalServiceError(
                500,
                f"refusing base_url {base_url!r}: the frozen world is served on loopback only, "
                "and a retrieval endpoint on another host is not the world this run froze "
                "against")
        if timeout_seconds <= 0:
            raise RetrievalServiceError(500, "timeout_seconds must be positive")
        self.base_url = base_url.rstrip("/")
        self.timeout_seconds = float(timeout_seconds)
        self._expected_retriever_id = expected_retriever_id
        self._expected_freeze_digest = expected_freeze_digest
        self._require_effective_freeze = require_effective_freeze
        self._lock = threading.Lock()
        self._verified: Optional[dict] = None
        self._retriever_id = ""

    # --- transport ----------------------------------------------------------------------

    def _request(self, path: str, body: Optional[Mapping[str, Any]]) -> dict:
        url = self.base_url + path
        data = None if body is None else json.dumps(body).encode("utf-8")
        # The urllib scheme audit (ruff S310) is answered in __init__, which refuses anything but
        # http on a loopback host before a request can be built here.
        request = urllib.request.Request(  # noqa: S310
            url, data=data, method="GET" if data is None else "POST",
            headers={"Content-Type": "application/json"} if data is not None else {})
        try:
            with _OPENER.open(request, timeout=self.timeout_seconds) as response:
                status, raw = response.status, response.read()
        except urllib.error.HTTPError as e:
            if 300 <= e.code < 400:
                # Reported here rather than left to fail as "not JSON": a redirect is not a
                # malformed answer, it is the frozen world being moved to an address this client
                # never verified, and the operator needs to be told which one.
                location = e.headers.get("Location", "(unnamed)") if e.headers else "(unnamed)"
                raise RetrievalServiceError(
                    502,
                    f"{url} answered {e.code} redirecting to {location!r}. The retrieval hop is "
                    "not followed off the address it was verified against: whatever answers there "
                    "is a different retriever, and results from it belong to a different world"
                ) from e
            status, raw = e.code, e.read()
        except OSError as e:
            # Covers URLError, connection refused and socket timeout. Reported as a failure, not
            # as "no results": a stopped service must not look like an empty corpus.
            raise RetrievalServiceError(
                503, f"{url} is unreachable ({type(e).__name__}: {e})") from e
        try:
            payload = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RetrievalServiceError(
                502, f"{url} returned a body that is not JSON ({e})") from e
        if not isinstance(payload, dict):
            raise RetrievalServiceError(502, f"{url} returned {type(payload).__name__}, not an "
                                             "object")
        if status != 200:
            raise RetrievalServiceError(status, str(payload.get("error", payload)))
        return payload

    # --- identity -----------------------------------------------------------------------

    def health(self) -> dict:
        """The service's live health block. Not validated beyond being a JSON object."""
        return self._request("/healthz", None)

    def verify(self) -> dict:
        """Check identity once and remember it. Returns the retriever identity block.

        Idempotent and thread-safe, so ``search`` can call it on every use without turning the
        first search of each researcher into a second round trip.
        """
        with self._lock:
            if self._verified is not None:
                return self._verified
            health = self.health()
            identity = health.get("retriever")
            retriever_id = health.get("retriever_id")
            if not isinstance(identity, dict) or not isinstance(retriever_id, str):
                raise RetrievalServiceError(
                    502, "health response carries no retriever identity; a service that cannot "
                         "say which index it holds cannot be verified against a freeze")
            if not _HEX64.match(retriever_id):
                raise RetrievalServiceError(502, f"retriever_id {retriever_id!r} is not a digest")
            _require_id_summarises_identity(retriever_id, identity)
            if self._expected_retriever_id and retriever_id != self._expected_retriever_id:
                raise RetrievalServiceError(
                    409,
                    f"retriever_id is {retriever_id[:12]}, expected "
                    f"{self._expected_retriever_id[:12]}; this is a different retriever and its "
                    "results belong to a different world")
            freeze = identity.get("freeze")
            if self._expected_freeze_digest:
                if not isinstance(freeze, dict):
                    raise RetrievalServiceError(
                        409, "the service declares no retrieval freeze, so it cannot be the one "
                             f"named by {self._expected_freeze_digest[:12]}")
                if freeze.get("digest") != self._expected_freeze_digest:
                    raise RetrievalServiceError(
                        409,
                        f"the service serves freeze {str(freeze.get('digest'))[:12]}, expected "
                        f"{self._expected_freeze_digest[:12]}")
            if self._require_effective_freeze:
                if not isinstance(freeze, dict):
                    raise RetrievalServiceError(
                        409,
                        "the service declares no retrieval freeze; refusing to retrieve under an "
                        "unfrozen retriever. Pass require_effective_freeze=False only from the "
                        "P0 competence pilot, which is what makes a freeze effective.")
                if not freeze.get("effective"):
                    raise RetrievalServiceError(
                        409,
                        f"retrieval freeze {str(freeze.get('digest'))[:12]} has no competence "
                        "pilot digest, so it is not in force. An unvalidated retriever becoming "
                        "the frozen one by default is what effective_after prevents.")
            self._verified = identity
            self._retriever_id = retriever_id
            return identity

    def stats(self) -> dict:
        """The service's cache accounting, for the per-arm retrieval report."""
        health = self.health()
        cache = health.get("cache")
        if not isinstance(cache, dict):
            raise RetrievalServiceError(502, "health response carries no cache block")
        return cache

    # --- search -------------------------------------------------------------------------

    def search_rows(self, query: str, *, max_results: int) -> list[dict]:
        """The validated wire rows, docid included.

        ``SearchRecord`` is shaped like a vendor search result and has no docid field, but a
        docid is what joins a retrieved document to the benchmark's own gold and evidence sets --
        on the *evaluator* side. Callers that need it use this method; the treatment path uses
        :meth:`search`.
        """
        if not isinstance(query, str) or not query.strip():
            raise RetrievalServiceError(400, "query must be a non-empty string")
        if isinstance(max_results, bool) or not isinstance(max_results, int) or max_results < 1:
            raise RetrievalServiceError(
                400, f"max_results must be a positive integer, got {max_results!r}")
        self.verify()
        payload = self._request("/v1/search", {"query": query, "top_k": max_results})
        return self._validate_response(payload, query=query, max_results=max_results)

    def search(self, query: str, *, max_results: int) -> list[SearchRecord]:
        """The :class:`SearchBackend` seam. Ranked, whole documents, or an exception."""
        return [
            SearchRecord(
                url=row["url"],
                title=row["title"],
                content=row["snippet"],
                raw_content=row["text"],
                score=row["score"],
                occurrence_id=row["occurrence_id"],
            )
            for row in self.search_rows(query, max_results=max_results)
        ]

    # --- response validation ------------------------------------------------------------

    def _validate_response(self, payload: dict, *, query: str, max_results: int) -> list[dict]:
        unknown = sorted(set(payload) - _RESPONSE_FIELDS)
        if unknown:
            raise RetrievalServiceError(
                502,
                f"search response carries unenumerated field(s) {unknown}; the schema is closed "
                "in both directions so that nothing evaluator-side has a field to travel in")
        missing = sorted(_RESPONSE_FIELDS - set(payload))
        if missing:
            raise RetrievalServiceError(502, f"search response is missing {missing}")
        if payload["query"] != query:
            raise RetrievalServiceError(
                502, "the response echoes a different query than the one sent; these results "
                     "belong to some other request")
        if payload["top_k"] != max_results:
            raise RetrievalServiceError(
                502, f"asked for {max_results} results, response says top_k={payload['top_k']!r}")
        if payload["retriever_id"] != self._retriever_id:
            raise RetrievalServiceError(
                409,
                f"this response came from retriever {str(payload['retriever_id'])[:12]}, not the "
                f"verified {self._retriever_id[:12]}; the service changed identity mid-run")
        results = payload["results"]
        if not isinstance(results, list):
            raise RetrievalServiceError(502, "results must be a list")
        if payload["count"] != len(results):
            raise RetrievalServiceError(
                502, f"response says count={payload['count']!r} with {len(results)} results")
        if len(results) > max_results:
            raise RetrievalServiceError(
                502, f"asked for {max_results} results and got {len(results)}; a backend that "
                     "returns more than it was asked lets one arm read more than the other")

        seen: set[str] = set()
        for position, row in enumerate(results, start=1):
            if not isinstance(row, dict):
                raise RetrievalServiceError(502, f"result {position} is not an object")
            extra = sorted(set(row) - _RESULT_FIELDS)
            if extra:
                raise RetrievalServiceError(
                    502, f"result {position} carries unenumerated field(s) {extra}")
            absent = sorted(_RESULT_FIELDS - set(row))
            if absent:
                raise RetrievalServiceError(502, f"result {position} is missing {absent}")
            for field_name in _TEXT_FIELDS:
                if not isinstance(row[field_name], str):
                    raise RetrievalServiceError(
                        502, f"result {position} field {field_name!r} is "
                             f"{type(row[field_name]).__name__}, not a string")
            if not row["docid"]:
                raise RetrievalServiceError(502, f"result {position} has an empty docid")
            if not _HEX64.match(row["occurrence_id"]):
                raise RetrievalServiceError(
                    502, f"result {position} has occurrence_id {row['occurrence_id']!r}, which is "
                         "not a derived id; citation lineage joins on it")
            if row["rank"] != position:
                raise RetrievalServiceError(
                    502,
                    f"result {position} claims rank {row['rank']!r}; ranks must be the positions "
                    "they occupy or the ordering the arm saw is not the ordering recorded")
            score = row["score"]
            if isinstance(score, bool) or not isinstance(score, (int, float)):
                raise RetrievalServiceError(
                    502, f"result {position} has a non-numeric score {score!r}")
            if not math.isfinite(score):
                raise RetrievalServiceError(
                    502, f"result {position} has a non-finite score {score!r}; the ranking that "
                         "produced it is meaningless")
            if row["docid"] in seen:
                raise RetrievalServiceError(
                    502, f"docid {row['docid']!r} appears twice; the index guarantees unique "
                         "docids, so a duplicate means the results were merged or reordered")
            seen.add(row["docid"])
            row["score"] = float(score)
        return results


def connect(base_url: str, **kwargs) -> RetrievalClient:
    """Build a client and verify its identity now, rather than on the first search.

    Used by a runner's preflight: a retriever mismatch should stop a run before a researcher has
    started, not halfway through one.
    """
    client = RetrievalClient(base_url, **kwargs)
    client.verify()
    return client
