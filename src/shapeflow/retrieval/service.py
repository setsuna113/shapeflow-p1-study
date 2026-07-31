"""The retrieval service: the frozen world, behind one loopback socket.

Why this is a separate process at all. The query encoder needs torch and transformers; the study
package's dependency closure is pinned to the vendor's own lock (``[tool.uv]
constraint-dependencies``) precisely so the agent framework under measurement is not perturbed by
our tooling. Pulling a deep-learning stack into that closure to embed a search query would be the
largest uncontrolled change in the repository. So the encoder lives here, in its own interpreter,
and the study package reaches it through :mod:`shapeflow.retrieval.client` over loopback.

That split only buys anything if the thing on this side is provably the frozen retriever, so this
module is mostly gates:

**It will not start against an unvalidated encoder.** The conformance report is not advice, it is
the admission condition. The encoder *recipe* is read out of the report rather than off the
command line, so "the encoder that was checked" and "the encoder that is serving" are the same
object by construction rather than by an operator retyping four flags. Its serving dtype, its
prefix digests and its index shard digests are all required to line up with the index actually
loaded here, and its ``ok`` flag is re-derived from the evidence beside it rather than believed:
a flag written by the process being gated is a gate that can be satisfied by asserting it has
been satisfied. Every wrong recipe -- the query instruction applied to
passages, mean pooling, a different revision -- yields an encoder that runs, returns plausible
vectors and retrieves badly, and the failure downstream looks like "this benchmark is too hard".

**It will not start against a corpus of a different vintage.** Every docid the index can return
must resolve to a document, checked exhaustively at startup rather than discovered on the first
query that happens to hit the gap.

**It never touches a GPU.** Each device already runs a vLLM engine at high memory utilisation and
S1's headline number is the sustainable arrival rate on that engine. An encoder sharing the
device spends SM time the measurement would attribute to serving, and no amount of care
afterwards can subtract it. ``CUDA_VISIBLE_DEVICES`` is emptied before torch is ever imported,
and a non-CPU device is refused rather than honoured.

**It fails closed.** A missing file, an unparseable report, an unknown field, a query it cannot
encode: every one of these is an explicit refusal with a status. An empty result list from this
service means "the corpus has nothing" and nothing else -- it is never what a failure degrades to.

On the leakage firewall, this process's position is structural rather than declarative. It opens
exactly two kinds of artifact -- the index shard pickles and the three columns ``docid``,
``text``, ``url`` of the corpus parquet -- and it has no reader for any other shape, so the
benchmark's relevance judgements, graded answers and hard-negative sets cannot be loaded here even
by accident: pointed at them, the corpus loader finds no shards and refuses to start. The response
schema is closed in the same way, so a label that somehow existed in this process would still have
no field to travel in.

Transitive dependency note for whoever installs the encoder interpreter: this module imports
:mod:`shapeflow.retrieval.backend` to build search records, which reaches
:mod:`shapeflow.world.search_backend` and from there ``zstandard``. That is deliberate -- one
implementation of "what a search result looks like" serves both the in-process backend and this
service, so an occurrence id cannot depend on which deployment produced it -- but it means the
encoder venv needs ``numpy``, ``pyarrow`` and ``zstandard``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import sys
import threading
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Callable, Mapping, Optional, Sequence

import numpy as np

from ..canonical import canonical_json
from ..hashing import sha256_hex
from .backend import BrowseCompPlusBackend
from .conformance import DEFAULT_TOLERANCE
from .corpus import CorpusStore, load_corpus
from .encoder import EncoderSpec
from .index import DenseIndex, SearchHit, load_index

__all__ = [
    "SERVICE_VERSION",
    "ServiceConfig",
    "ServiceError",
    "ServiceStartupError",
    "RetrievalService",
    "build_service",
    "load_conformance_report",
    "encoder_spec_from_report",
    "harden_cpu_only",
    "parse_cpuset",
    "apply_cpu_affinity",
    "make_handler",
    "make_server",
    "serve_forever",
    "main",
]

#: Wire contract version. It is inside the identity digest, so a client pinned to one
#: retriever cannot be silently answered by a service speaking a different response shape.
SERVICE_VERSION = "shapeflow-retrieval-service-v1"

#: Encoded once at startup to prove the live model produces index-dimension unit vectors before
#: any real query depends on it. Fixed text, so the check is the same on every host.
WARMUP_QUERY = "shapeflow retrieval service warmup probe"

#: A search body is a query and a count. A megabyte is already three orders of magnitude more
#: than any BrowseComp-Plus query, so anything larger is a mistake, not a long question.
MAX_BODY_BYTES = 1 << 20

#: Loopback only. This service holds no credential, but it does hold the frozen world: a
#: retriever reachable off-host is a retriever someone else's process can be pointed at, and
#: "same visible world" stops being a property of the deployment.
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "::1", "localhost"})

_SEARCH_BODY_FIELDS = frozenset({"query", "top_k"})


class ServiceError(RuntimeError):
    """A request the service refuses, carrying the HTTP status it should produce."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status


class ServiceStartupError(RuntimeError):
    """The service refuses to exist. Never downgraded to a warning: see the module docstring."""


# --- configuration ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ServiceConfig:
    """Everything resolved before the first request, and then never changed.

    Note what is *absent*: the encoder repo, revision, dtype, pooling and lengths. Those come
    out of the conformance report, so the served recipe is the validated recipe by construction.
    A flag for each of them would be four more chances to serve something nobody checked.
    """

    index_dir: Path
    corpus_dir: Path
    conformance_path: Path
    freeze_path: Optional[Path] = None

    bind_host: str = "127.0.0.1"
    bind_port: int = 8710

    device: str = "cpu"
    threads: int = 8
    cpuset: str = ""

    index_pattern: str = "corpus.shard*.pkl"
    corpus_pattern: str = "*.parquet"

    #: Ceiling on a single request's ``top_k``. The frozen campaign value is 5; this is a
    #: sanity bound so one malformed request cannot ask the service to rank and serialize the
    #: whole corpus. Requests above it are refused, never quietly clamped -- a clamp would mean
    #: an arm saw fewer documents than it asked for with nothing recording that it had.
    max_top_k: int = 100
    #: Refused rather than truncated, for the same reason: truncating would mean the service
    #: searched for something other than what was asked, and the response would not say so.
    max_query_chars: int = 32_768
    #: Distinct query vectors held. Bounded because this process outlives a campaign phase and
    #: an 8B vector is 16 KiB; unbounded, four lanes would leak steadily for the whole run.
    cache_capacity: int = 4096

    def validate(self) -> None:
        if self.bind_host not in LOOPBACK_HOSTS:
            raise ServiceStartupError(
                f"refusing to bind {self.bind_host!r}: the retrieval service serves the frozen "
                "world and is loopback-only, so no other host can be answered by it")
        if self.device != "cpu":
            raise ServiceStartupError(
                f"refusing device {self.device!r}: every GPU on this host runs a vLLM engine and "
                "S1's headline metric is that engine's sustainable arrival rate. An encoder "
                "sharing the device spends SM time the measurement would attribute to serving.")
        for name, value in (("threads", self.threads), ("max_top_k", self.max_top_k),
                            ("max_query_chars", self.max_query_chars),
                            ("cache_capacity", self.cache_capacity)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ServiceStartupError(f"{name} must be a positive integer, got {value!r}")
        # 0 means "let the OS choose", which is how a launcher (and every test here) gets a port
        # without racing another lane for a fixed one. The bound port is read back off the server.
        if not 0 <= self.bind_port <= 65535:
            raise ServiceStartupError(f"bind_port {self.bind_port} is out of range")


# --- the conformance gate --------------------------------------------------------------------


@dataclass(frozen=True)
class ConformanceRecord:
    """The conformance report, parsed and checked. Construction is the gate."""

    path: Path
    sha256: str
    ok: bool
    dtype: str
    checked: int
    tolerance: float
    min_cosine: float
    rank_1_count: int
    index_shard_sha256: tuple[str, ...]
    encoder_spec: Mapping[str, Any]

    def content(self) -> dict:
        return {
            "sha256": self.sha256,
            "ok": self.ok,
            "dtype": self.dtype,
            "checked": self.checked,
            "tolerance": self.tolerance,
            "min_cosine": self.min_cosine,
            "rank_1_count": self.rank_1_count,
        }


def _report_number(body: Mapping[str, Any], path: Path, field_name: str, *, integral: bool):
    """One numeric field of the report, or a refusal.

    Read through a checker rather than through ``int()``/``float()`` because those two coerce:
    ``int("forty")`` raises a ``ValueError`` that is not a startup refusal and never reaches the
    entry point's ``REFUSED`` path, and ``json.loads`` accepts the bare tokens ``NaN`` and
    ``Infinity``, either of which would sail through ``float()`` and then blow up much later
    inside the identity digest. A report that does not parse *as what it claims to be* is a
    failure, not an absence, and it has to say so here.
    """
    value = body[field_name]
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ServiceStartupError(
            f"conformance report at {path} has {field_name}={value!r}, which is "
            f"{type(value).__name__} and not a number; the evidence a report carries has to be "
            "readable as evidence")
    if isinstance(value, float) and not math.isfinite(value):
        raise ServiceStartupError(
            f"conformance report at {path} has a non-finite {field_name} ({value!r})")
    if integral and not isinstance(value, int):
        raise ServiceStartupError(
            f"conformance report at {path} has {field_name}={value!r}, which is not a whole "
            "number of documents")
    return value


def _require_the_evidence_supports_ok(body: Mapping[str, Any], path: Path, *, checked: int,
                                      tolerance: float, min_cosine: float,
                                      rank_1_count: int) -> None:
    """Re-derive ``ok`` from the numbers beside it instead of believing the flag.

    ``ok`` is one boolean written by the process being gated. On its own it is a gate that can be
    satisfied by asserting it has been satisfied, which AGENTS.md section 8 rules out: every
    criterion has to be re-derivable from the artifact it read. The report already carries the
    whole derivation -- how many documents were re-encoded, the floor they were held to, the worst
    cosine seen, how many self-retrieved at rank 1, and the failure list -- so the flag is checked
    against it here.

    Each clause below is a report that ``ok`` alone would have licensed:

    * ``checked: 0`` -- a run over no documents. :class:`ConformanceResult` would never emit it
      with ``ok`` true, but a hand-written or truncated file is exactly the artifact that reaches
      a service at 2am, and it would have started a campaign on an encoder nothing had compared
      to anything.
    * ``min_cosine`` under ``tolerance`` -- the encoder does not reproduce the index and the file
      says both things at once.
    * a ``tolerance`` below :data:`~shapeflow.retrieval.conformance.DEFAULT_TOLERANCE` -- the
      check re-run with the bar lowered until it passed, which is the substitution AGENTS.md
      section 9 forbids, arriving as evidence rather than as a decision.
    * ``rank_1_count`` short of ``checked``, or a non-empty ``failures`` list -- documents that
      did not retrieve themselves, which is the half of the check that catches a misaligned docid
      list or a broken tie-break rather than a wrong recipe.
    """
    if checked < 1:
        raise ServiceStartupError(
            f"conformance report at {path} records ok=true over {checked} documents; a "
            "conformance run over nothing passes vacuously and licenses nothing")
    if tolerance < DEFAULT_TOLERANCE:
        raise ServiceStartupError(
            f"conformance report at {path} passed at tolerance {tolerance}, below this repo's "
            f"floor of {DEFAULT_TOLERANCE}. A check re-run with the bar lowered until it passed "
            "is not evidence that the encoder reproduces the index.")
    if min_cosine < tolerance:
        raise ServiceStartupError(
            f"conformance report at {path} records ok=true with min_cosine {min_cosine} under "
            f"its own tolerance {tolerance}; the flag and the numbers beside it disagree, and the "
            "numbers are the measurement")
    failures = body["failures"]
    if not isinstance(failures, list):
        raise ServiceStartupError(
            f"conformance report at {path} has a malformed failures list")
    if failures:
        raise ServiceStartupError(
            f"conformance report at {path} records ok=true with {len(failures)} failing "
            f"document(s) (e.g. {failures[0]!r}); the flag and the numbers beside it disagree")
    if rank_1_count != checked:
        raise ServiceStartupError(
            f"conformance report at {path} records ok=true but only {rank_1_count} of {checked} "
            "documents retrieved themselves at rank 1; cosines can pass while the docid "
            "alignment or the tie-break is broken, which is why that count is checked separately")


def load_conformance_report(path: Path) -> ConformanceRecord:
    """Read and gate the conformance report produced by ``scripts/retrieval_conformance.py``.

    Every failure here is an explicit refusal. In particular a *missing* report and a report that
    records a *failed* check are different messages, because they call for different actions: one
    means run the check, the other means the encoder does not reproduce the index and nothing
    downstream should spend a GPU hour.

    The ``ok`` flag is necessary and not sufficient: see
    :func:`_require_the_evidence_supports_ok`.
    """
    path = Path(path)
    try:
        raw = path.read_bytes()
    except FileNotFoundError:
        raise ServiceStartupError(
            f"no conformance report at {path}. The service will not serve an encoder that was "
            "never checked against its index: run scripts/retrieval_conformance.py first."
        ) from None
    except OSError as e:
        raise ServiceStartupError(f"conformance report at {path} is unreadable: {e}") from e

    try:
        body = json.loads(raw.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ServiceStartupError(
            f"conformance report at {path} does not parse ({e}); an artifact that does not parse "
            "is a failure, not an absence") from e
    if not isinstance(body, dict):
        raise ServiceStartupError(f"conformance report at {path} is not a JSON object")

    for field_name in ("ok", "dtype", "checked", "tolerance", "min_cosine", "rank_1_count",
                       "failures", "index_shard_sha256", "encoder_spec"):
        if field_name not in body:
            raise ServiceStartupError(
                f"conformance report at {path} has no {field_name!r} field; a report missing the "
                "evidence it is supposed to carry cannot license a service")
    if body["ok"] is not True:
        raise ServiceStartupError(
            f"conformance report at {path} records ok={body['ok']!r}: this encoder does not "
            "reproduce this index. Serving anyway is the failure the check exists to prevent -- "
            f"diagnosis: {body.get('diagnosis', '(none recorded)')}")

    checked = _report_number(body, path, "checked", integral=True)
    tolerance = float(_report_number(body, path, "tolerance", integral=False))
    min_cosine = float(_report_number(body, path, "min_cosine", integral=False))
    rank_1_count = _report_number(body, path, "rank_1_count", integral=True)
    _require_the_evidence_supports_ok(body, path, checked=checked, tolerance=tolerance,
                                      min_cosine=min_cosine, rank_1_count=rank_1_count)

    spec = body["encoder_spec"]
    if not isinstance(spec, dict):
        raise ServiceStartupError(f"conformance report at {path} has a malformed encoder_spec")
    if body["dtype"] != spec.get("dtype"):
        raise ServiceStartupError(
            f"conformance report at {path} was run as dtype {body['dtype']!r} but its encoder "
            f"spec says {spec.get('dtype')!r}. A check passed in one dtype says nothing about a "
            "service running another.")
    shards = body["index_shard_sha256"]
    if not isinstance(shards, list) or not all(isinstance(s, str) for s in shards):
        raise ServiceStartupError(
            f"conformance report at {path} has a malformed index_shard_sha256 list")

    return ConformanceRecord(
        path=path,
        sha256=sha256_hex(raw),
        ok=True,
        dtype=str(body["dtype"]),
        checked=checked,
        tolerance=tolerance,
        min_cosine=min_cosine,
        rank_1_count=rank_1_count,
        index_shard_sha256=tuple(shards),
        encoder_spec=dict(spec),
    )


def encoder_spec_from_report(record: ConformanceRecord) -> EncoderSpec:
    """Rebuild the validated encoder recipe from the report, and prove it is still expressible.

    The rebuilt spec is re-rendered and compared field by field against the recorded one. That
    comparison covers the two prefix digests, which are *not* stored fields -- they are hashes of
    the module constants in :mod:`shapeflow.retrieval.encoder`. So editing the query instruction
    in this repo makes every older conformance report stop matching, instead of leaving a service
    that serves one recipe under the evidence of another.
    """
    spec_fields = ("model", "revision", "query_max_len", "passage_max_len", "pooling",
                   "normalize", "dtype", "padding_side")
    missing = [f for f in spec_fields if f not in record.encoder_spec]
    if missing:
        raise ServiceStartupError(
            f"conformance report at {record.path} has no {missing} in its encoder_spec; the "
            "serving recipe is read from the report and cannot be defaulted")
    spec = EncoderSpec(
        model=str(record.encoder_spec["model"]),
        revision=str(record.encoder_spec["revision"]),
        query_max_len=int(record.encoder_spec["query_max_len"]),
        passage_max_len=int(record.encoder_spec["passage_max_len"]),
        pooling=str(record.encoder_spec["pooling"]),
        normalize=bool(record.encoder_spec["normalize"]),
        dtype=str(record.encoder_spec["dtype"]),
        padding_side=str(record.encoder_spec["padding_side"]),
    )
    rendered = spec.content()
    differing = sorted(k for k, v in rendered.items() if record.encoder_spec.get(k) != v)
    if differing:
        details = ", ".join(
            f"{k}: report {record.encoder_spec.get(k)!r} vs code {rendered[k]!r}"
            for k in differing)
        raise ServiceStartupError(
            f"the encoder recipe in {record.path} is not the one this code implements "
            f"({details}). Serving under a report that describes a different encoder is exactly "
            "the unvalidated-encoder failure the conformance check exists to prevent.")
    return spec


# --- CPU-only hardening ------------------------------------------------------------------------


def harden_cpu_only(env: dict, *, threads: int) -> dict:
    """Mutate ``env`` so torch cannot see a GPU and cannot oversubscribe the pinned cores.

    Must run before torch is imported: ``CUDA_VISIBLE_DEVICES`` is read once at initialisation,
    so emptying it afterwards has no effect and the encoder would quietly land on a device that
    is serving the engine being measured.

    The thread caps matter for the same measurement: BLAS defaults to one thread per visible
    core, which on a pinned cpuset means every encode oversubscribes its own allocation and the
    measured p95 encode latency stops being a property of the configuration.
    """
    if isinstance(threads, bool) or not isinstance(threads, int) or threads < 1:
        raise ServiceStartupError(f"threads must be a positive integer, got {threads!r}")
    env["CUDA_VISIBLE_DEVICES"] = ""
    env["HIP_VISIBLE_DEVICES"] = ""
    env["OMP_NUM_THREADS"] = str(threads)
    env["MKL_NUM_THREADS"] = str(threads)
    env["OPENBLAS_NUM_THREADS"] = str(threads)
    env["NUMEXPR_NUM_THREADS"] = str(threads)
    # The tokenizers fork-parallelism warning is not the point: parallel tokenization spawns
    # threads outside the cap above, which is the same oversubscription by another route.
    env["TOKENIZERS_PARALLELISM"] = "false"
    return env


def parse_cpuset(spec: str) -> list[int]:
    """Parse a Linux cpuset string (``"0-3,8"``) into a sorted list of cpu ids.

    Explicitly parsed rather than passed to a shell, and a malformed spec is an error rather
    than an empty set: an empty affinity mask would either raise deep inside libc or silently
    leave the process free to roam every core, which is the opposite of what was asked.
    """
    cpus: set[int] = set()
    for part in spec.split(","):
        chunk = part.strip()
        if not chunk:
            raise ServiceStartupError(f"cpuset {spec!r} has an empty component")
        if "-" in chunk:
            low, _, high = chunk.partition("-")
            try:
                start, end = int(low), int(high)
            except ValueError:
                raise ServiceStartupError(f"cpuset {spec!r} has a malformed range {chunk!r}"
                                          ) from None
            if start > end:
                raise ServiceStartupError(f"cpuset {spec!r} has an inverted range {chunk!r}")
            cpus.update(range(start, end + 1))
        else:
            try:
                cpus.add(int(chunk))
            except ValueError:
                raise ServiceStartupError(f"cpuset {spec!r} has a malformed cpu id {chunk!r}"
                                          ) from None
    if not cpus:
        raise ServiceStartupError(f"cpuset {spec!r} names no cpus")
    if any(c < 0 for c in cpus):
        raise ServiceStartupError(f"cpuset {spec!r} names a negative cpu id")
    return sorted(cpus)


def apply_cpu_affinity(spec: str) -> list[int]:
    """Pin this process to ``spec``. A cpuset that cannot be applied is fatal, not ignored."""
    cpus = parse_cpuset(spec)
    setter = getattr(os, "sched_setaffinity", None)
    if setter is None:
        raise ServiceStartupError(
            "this platform has no sched_setaffinity, so the requested cpuset cannot be honoured; "
            "refusing rather than running unpinned, because an unpinned encoder competes with "
            "the engine whose arrival rate is being measured")
    try:
        setter(0, set(cpus))
    except OSError as e:
        raise ServiceStartupError(f"cpuset {spec!r} could not be applied: {e}") from e
    return cpus


# --- the query-vector cache --------------------------------------------------------------------


class BoundedVectorCache(dict):
    """Query text -> vector, bounded, evicting in insertion order.

    Correctness is unaffected by caching at all: an embedding is a pure function of the text and
    the pinned encoder. The bound exists because this process outlives a campaign phase, and the
    eviction order is insertion rather than recency because insertion order is a property of the
    request sequence alone -- an LRU would depend on how concurrent requests interleaved, so two
    replays of the same trace could evict different entries.

    Deliberately **not** salted per arm: salting would hand whichever arm ran second a cold cache
    and a latency penalty that has nothing to do with its compression form.
    """

    def __init__(self, capacity: int) -> None:
        if isinstance(capacity, bool) or not isinstance(capacity, int) or capacity < 1:
            raise ValueError(f"cache capacity must be a positive integer, got {capacity!r}")
        super().__init__()
        self._capacity = capacity
        self._lock = threading.Lock()

    @property
    def capacity(self) -> int:
        return self._capacity

    def __setitem__(self, key, value) -> None:
        with self._lock:
            super().__setitem__(key, value)
            while len(self) > self._capacity:
                super().__delitem__(next(iter(self)))


class _HitCapture:
    """A :class:`DenseIndex` façade that remembers the hits of the search it just forwarded.

    :class:`~shapeflow.world.search_backend.SearchRecord` carries no docid -- it is shaped like a
    vendor search result, and those have none. The wire format needs one, because a docid is what
    joins a retrieved document to the benchmark's own evaluation. Re-running the search to
    recover it would double the cost of every query, so the hits are captured on the way through
    instead. One instance per request, so nothing is shared between threads.
    """

    def __init__(self, index: DenseIndex) -> None:
        self._index = index
        self.hits: list[SearchHit] = []

    @property
    def num_docs(self) -> int:
        return self._index.num_docs

    @property
    def dim(self) -> int:
        return self._index.dim

    def search(self, query_vector, *, top_k: int) -> list[SearchHit]:
        self.hits = self._index.search(query_vector, top_k=top_k)
        return self.hits


# --- the service ---------------------------------------------------------------------------------


class RetrievalService:
    """Index + corpus + encoder, loaded once. No HTTP here, so it is directly testable."""

    def __init__(self, config: ServiceConfig, *, index: DenseIndex, corpus: CorpusStore,
                 spec: EncoderSpec, conformance: ConformanceRecord,
                 encode: Callable[[str], Any], freeze: Optional[Any] = None) -> None:
        self._config = config
        self._index = index
        self._corpus = corpus
        self._spec = spec
        self._conformance = conformance
        self._raw_encode = encode
        self._freeze = freeze
        self._cache = BoundedVectorCache(config.cache_capacity)
        # One encode at a time. Concurrent encodes would each spawn the configured thread count
        # inside the same pinned cpuset, so measured latency would depend on how many requests
        # happened to overlap. Health stays answerable meanwhile, which is why the whole handler
        # is not simply serialized.
        self._encode_lock = threading.Lock()
        self._counter_lock = threading.Lock()
        self._searches = 0
        self._encodes = 0
        self._errors = 0
        self._warm = False
        self._identity = self._build_identity()
        self._retriever_id = sha256_hex(canonical_json(self._identity))

    # --- identity ---------------------------------------------------------------------------

    def _build_identity(self) -> dict:
        """The block a caller compares to decide it is talking to the frozen retriever.

        Deliberately excludes anything that moves while the service runs. A digest that changed
        with the cache hit rate could not be pinned by a client, and a client that cannot pin it
        cannot tell this retriever from another one.
        """
        freeze_block = None
        if self._freeze is not None:
            freeze_block = {
                "digest": self._freeze.digest,
                "effective": bool(self._freeze.effective),
                "top_k": int(self._freeze.top_k),
                "full_document": bool(self._freeze.full_document),
            }
        return {
            "service_version": SERVICE_VERSION,
            "index": {
                "num_docs": self._index.num_docs,
                "dim": self._index.dim,
                "shard_sha256": [s.sha256 for s in self._index.shards],
            },
            "corpus": {
                "num_docs": len(self._corpus),
                "shard_sha256": list(self._corpus.shard_sha256),
            },
            "encoder": dict(self._spec.content()),
            "conformance": self._conformance.content(),
            "freeze": freeze_block,
        }

    @property
    def retriever_id(self) -> str:
        return self._retriever_id

    @property
    def identity(self) -> dict:
        return json.loads(json.dumps(self._identity))  # a copy; callers must not mutate it

    def health(self) -> dict:
        with self._counter_lock:
            searches, encodes, errors = self._searches, self._encodes, self._errors
        hits = searches - encodes
        return {
            "status": "ok",
            "warm": self._warm,
            "retriever_id": self._retriever_id,
            "retriever": self.identity,
            "cache": {
                "searches": searches,
                "encodes": encodes,
                "errors": errors,
                "hits": hits,
                # Reported to be compared *between arms*: a spread is a timing asymmetry, not a
                # detail. Computed from exact counters rather than from the cache's size, which
                # stops being an answer once anything has been evicted.
                "hit_rate": round(hits / searches, 4) if searches else 0.0,
                "size": len(self._cache),
                "capacity": self._cache.capacity,
            },
            "service": {
                "device": self._config.device,
                "threads": self._config.threads,
                "cpuset": self._config.cpuset,
                "max_top_k": self._config.max_top_k,
                "max_query_chars": self._config.max_query_chars,
            },
        }

    # --- encoding ---------------------------------------------------------------------------

    def _encode_checked(self, text: str):
        """Encode and validate, so nothing invalid is ever returned to the caller *or* cached.

        The validation is here rather than left to the index's own guard because of where the
        cache write happens: the backend stores whatever comes back before the index ever sees
        it, so a wrong-dimension vector -- the 0.6B encoder started against the 8B index, say --
        would be cached once and then fail identically on every retry, including retries that
        would otherwise have succeeded after the operator fixed the model.
        """
        vector = self._raw_encode(text)
        if vector is None:
            raise ServiceError(500, "the encoder returned no vector")
        array = np.asarray(vector, dtype=np.float32).reshape(-1)
        if array.shape[0] != self._index.dim:
            raise ServiceError(
                500,
                f"the encoder produced a {array.shape[0]}-dimensional vector but the index has "
                f"dimension {self._index.dim}; the running model is not the one this index was "
                "built with")
        if not bool(np.all(np.isfinite(array))):
            raise ServiceError(500, "the encoder produced a non-finite vector")
        norm = float(np.linalg.norm(array))
        if not np.isclose(norm, 1.0, atol=1e-3):
            raise ServiceError(
                500,
                f"the encoder produced a vector of norm {norm:.4f}; the stored vectors are "
                "L2-normalised, so scores would not be cosine similarities")
        return array

    def _encode_for_backend(self, text: str):
        """The callable handed to the backend. Called only on a cache miss, and counted there.

        The miss is counted *before* the encode, not after: an encode that raises is still a
        cache miss, and counting only successes would report a failing encoder as a service with
        a perfect hit rate.
        """
        with self._encode_lock:
            cached = self._cache.get(text)
            if cached is not None:
                # Another request encoded this text while we waited for the lock.
                return cached
            with self._counter_lock:
                self._encodes += 1
            return self._encode_checked(text)

    def warm(self) -> None:
        """Load the model and prove it produces index-dimension unit vectors, before serving.

        Without this, the first real query pays a minute of model load and the first evidence
        that the wrong model was configured arrives inside an agent's search call, where it is
        indistinguishable from a slow or unhelpful retriever.
        """
        self._encode_checked(WARMUP_QUERY)
        self._warm = True

    # --- search -----------------------------------------------------------------------------

    def search(self, query: Any, top_k: Any) -> dict:
        if not isinstance(query, str):
            raise ServiceError(400, "query must be a string")
        if not query.strip():
            raise ServiceError(
                400,
                "query is empty; an empty query would still return the corpus's nearest "
                "neighbours to nothing in particular, which is indistinguishable from a result")
        if len(query) > self._config.max_query_chars:
            raise ServiceError(
                400,
                f"query is {len(query)} characters, above the {self._config.max_query_chars} "
                "ceiling; refused rather than truncated, because a truncated query is a "
                "different query and the response would not say so")
        if isinstance(top_k, bool) or not isinstance(top_k, int):
            raise ServiceError(400, "top_k must be an integer")
        if top_k < 1:
            raise ServiceError(400, f"top_k must be positive, got {top_k}")
        if top_k > self._config.max_top_k:
            raise ServiceError(
                400,
                f"top_k {top_k} is above this service's ceiling of {self._config.max_top_k}; "
                "refused rather than clamped, so no arm can be served fewer documents than it "
                "asked for without that being visible")

        with self._counter_lock:
            self._searches += 1
        try:
            return self._ranked(query, top_k)
        except Exception:
            with self._counter_lock:
                self._errors += 1
            raise

    def _ranked(self, query: str, top_k: int) -> dict:
        capture = _HitCapture(self._index)
        # Constructed per request: the backend's own ``queries_seen`` log and cache-hit-rate
        # accounting assume a short-lived, single-threaded object with an unevicted cache, and
        # this process is none of those. The shared state is exactly the bounded vector cache.
        backend = BrowseCompPlusBackend(
            index=capture, corpus=self._corpus, encode=self._encode_for_backend,
            cache=self._cache, queries_seen=[])
        try:
            records = backend.search(query, max_results=top_k)
        except KeyError as e:
            # CorpusStore.get raises when a retrieved docid has no document. That means the index
            # and the corpus are different builds, and every result assembled from them would be
            # unattributable.
            raise ServiceError(500, f"index/corpus mismatch: {e}") from e
        except ValueError as e:
            raise ServiceError(500, f"the query vector was refused by the index: {e}") from e

        hits = capture.hits
        if len(hits) != len(records):
            raise ServiceError(
                500, f"{len(hits)} index hits produced {len(records)} records; refusing to guess "
                     "which document each result came from")

        results = []
        for position, (hit, record) in enumerate(zip(hits, records, strict=True), start=1):
            score = float(hit.score)
            if not math.isfinite(score):
                # A NaN sorts wherever the sort put it and would then travel as valid JSON
                # through json.loads on the client side, so it has to stop here.
                raise ServiceError(
                    500,
                    f"document {hit.docid!r} scored {score!r}; a non-finite score means the "
                    "index or the query vector is corrupt and the ranking is meaningless")
            if hit.rank != position:
                raise ServiceError(
                    500, f"index returned rank {hit.rank} at position {position}")
            results.append({
                "docid": hit.docid,
                "rank": position,
                "score": score,
                "url": record.url,
                "title": record.title,
                "snippet": record.content,
                "text": record.raw_content if record.raw_content is not None else "",
                "occurrence_id": record.occurrence_id,
            })
        return {
            "query": query,
            "top_k": top_k,
            "count": len(results),
            # Echoed on every response, not only on /healthz: a service restarted against another
            # index between a client's health check and its next search would otherwise be
            # invisible.
            "retriever_id": self._retriever_id,
            "results": results,
        }


# --- startup ---------------------------------------------------------------------------------


def crosscheck_index_against_corpus(index: DenseIndex, corpus: CorpusStore) -> None:
    """Require every docid the index can return to have a document.

    Exhaustive, not sampled, and done at startup. The alternative is discovering the gap on
    whichever query first ranks the missing document, in the middle of a run, as a 500 that looks
    like a transient. One full-corpus scan costs a second and turns a vintage mismatch into a
    refusal to start.

    Enumerated through the public search path with a synthetic unit vector rather than by
    reaching into the index's private docid list.
    """
    if index.num_docs == 0:
        raise ServiceStartupError("the index is empty; it would answer every query with nothing")
    probe = np.zeros(index.dim, dtype=np.float32)
    probe[0] = 1.0
    missing = [h.docid for h in index.search(probe, top_k=index.num_docs)
               if h.docid not in corpus]
    if missing:
        raise ServiceStartupError(
            f"{len(missing)} of {index.num_docs} indexed documents have no text in the corpus "
            f"(e.g. {sorted(missing)[:5]}); the index and the corpus are different builds")


def _require_freeze_agrees(freeze, *, index: DenseIndex, corpus: CorpusStore,
                           spec: EncoderSpec, conformance: ConformanceRecord) -> None:
    """The freeze names one retriever. Refuse to serve anything else under its digest."""
    if tuple(freeze.index_shard_sha256) != tuple(s.sha256 for s in index.shards):
        raise ServiceStartupError(
            "the retrieval freeze names different index shards than the ones loaded; two "
            "retrievers under one freeze cannot both be the one every arm searched")
    if freeze.corpus_shard_sha256 and tuple(freeze.corpus_shard_sha256) != tuple(
            corpus.shard_sha256):
        raise ServiceStartupError(
            "the retrieval freeze names different corpus shards than the ones loaded")
    if freeze.conformance_sha256 and freeze.conformance_sha256 != conformance.sha256:
        raise ServiceStartupError(
            f"the retrieval freeze names conformance report {freeze.conformance_sha256[:12]} but "
            f"this service was given {conformance.sha256[:12]}; the freeze is only evidence "
            "about the encoder it was built from")
    rendered = spec.content()
    frozen_encoder = {
        "model": freeze.encoder_repo,
        "revision": freeze.encoder_revision,
        "dtype": freeze.encoder_dtype,
        "pooling": freeze.pooling,
        "normalize": freeze.normalize,
        "query_prefix_sha256": freeze.query_prefix_sha256,
        "passage_prefix_sha256": freeze.passage_prefix_sha256,
        "query_max_len": freeze.query_max_len,
        "passage_max_len": freeze.passage_max_len,
    }
    differing = sorted(k for k, v in frozen_encoder.items() if rendered.get(k) != v)
    if differing:
        raise ServiceStartupError(
            f"the retrieval freeze describes a different encoder ({differing}); serving under "
            "its digest would attribute this retriever's results to that one")


def build_service(config: ServiceConfig, *, encode: Optional[Callable[[str], Any]] = None
                  ) -> RetrievalService:
    """Run every startup gate and return a service, or raise :class:`ServiceStartupError`.

    ``encode`` is injected by tests. Left ``None``, the real
    :class:`~shapeflow.retrieval.encoder.QueryEncoder` is constructed from the validated spec --
    which imports torch lazily, on the first encode, so this function stays importable and
    runnable in an interpreter that has none.
    """
    config.validate()
    conformance = load_conformance_report(config.conformance_path)
    spec = encoder_spec_from_report(conformance)

    try:
        index = load_index(config.index_dir, pattern=config.index_pattern)
    except (OSError, ValueError, EOFError, pickle.UnpicklingError) as e:
        # Wrapped so the entry point reports one refusal shape rather than a traceback: a
        # missing shard, a misaligned docid list and a corrupt pickle are all "this index cannot
        # be served", and none of them may become an empty index that answers every query with
        # nothing.
        raise ServiceStartupError(f"the index at {config.index_dir} cannot be loaded: {e}") from e
    loaded_shards = tuple(s.sha256 for s in index.shards)
    if conformance.index_shard_sha256 != loaded_shards:
        raise ServiceStartupError(
            f"the conformance report at {config.conformance_path} checked index shards "
            f"{[s[:12] for s in conformance.index_shard_sha256]} but {config.index_dir} holds "
            f"{[s[:12] for s in loaded_shards]}; the encoder was validated against a different "
            "index")

    try:
        corpus = load_corpus(config.corpus_dir, pattern=config.corpus_pattern)
    except (OSError, ValueError, KeyError) as e:
        raise ServiceStartupError(
            f"the corpus at {config.corpus_dir} cannot be loaded: {e}") from e
    crosscheck_index_against_corpus(index, corpus)

    freeze = None
    if config.freeze_path is not None:
        from .freeze import load_freeze  # local: keeps the import cost off the no-freeze path

        try:
            freeze = load_freeze(config.freeze_path)
        except FileNotFoundError:
            raise ServiceStartupError(
                f"no retrieval freeze at {config.freeze_path}") from None
        except (json.JSONDecodeError, KeyError, RuntimeError) as e:
            raise ServiceStartupError(
                f"the retrieval freeze at {config.freeze_path} is unusable: {e}") from e
        _require_freeze_agrees(freeze, index=index, corpus=corpus, spec=spec,
                               conformance=conformance)

    if encode is None:
        from .encoder import QueryEncoder

        encoder = QueryEncoder(spec, device=config.device, threads=config.threads)
        encode = encoder.encode_query

    return RetrievalService(config, index=index, corpus=corpus, spec=spec,
                            conformance=conformance, encode=encode, freeze=freeze)


# --- HTTP ---------------------------------------------------------------------------------------


_ROUTES = {
    "/healthz": "healthz",
    "/v1/search": "search",
}


def resolve_route(path: str) -> str:
    """Map a request path to a route name. An unknown path is 404, never guessed."""
    clean = path.split("?", 1)[0].rstrip("/") or "/"
    try:
        return _ROUTES[clean]
    except KeyError:
        raise ServiceError(404, f"no such route {clean!r}") from None


def _parse_search_body(body: Any) -> tuple[Any, Any]:
    """Closed schema. An unenumerated field is refused by name rather than ignored.

    Ignoring one would let a caller believe it had asked for something it had not -- ``topk``
    for ``top_k`` would silently take the default -- and it is the same rule that keeps an
    evaluator-side field from having anywhere to travel.
    """
    if not isinstance(body, dict):
        raise ServiceError(400, "request body must be a JSON object")
    unknown = sorted(set(body) - _SEARCH_BODY_FIELDS)
    if unknown:
        raise ServiceError(400, f"unknown field(s) in search body: {unknown}")
    for field_name in sorted(_SEARCH_BODY_FIELDS):
        if field_name not in body:
            raise ServiceError(
                400, f"search body has no {field_name!r}; it has no default, because a default "
                     "would silently decide what an arm saw")
    return body["query"], body["top_k"]


def make_handler(service: RetrievalService):
    """Build the request handler class bound to one service instance."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "shapeflow-retrieval"
        sys_version = ""

        def log_message(self, fmt: str, *args) -> None:  # noqa: A003
            # Field allowlist: the request line only. Query text is retrieved-page-adjacent
            # untrusted data and has no business in a log file, and a body dump here would be
            # the easiest way for it to get there.
            sys.stderr.write(f"retrieval {self.address_string()} {fmt % args}\n")

        def _send(self, status: int, payload: dict, *, close: bool = False) -> None:
            raw = json.dumps(payload, sort_keys=True, allow_nan=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            if close:
                # Set by the two refusals that answer *without* consuming the body they refused.
                # Keeping such a connection alive leaves it desynchronized: the next read for a
                # request line lands in the middle of the undrained body, where it blocks until a
                # newline that is never coming. The handler thread is then parked exactly as it
                # would have been by the read it declined to make.
                self.close_connection = True
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self) -> None:  # noqa: N802
            self._dispatch(b"")

        def do_POST(self) -> None:  # noqa: N802
            try:
                length = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                self._send(400, {"error": "bad Content-Length"}, close=True)
                return
            if length < 0:
                # Not pedantry: ``rfile.read(-1)`` reads until EOF, and on an HTTP/1.1 keep-alive
                # connection EOF never comes. The handler thread parks for the life of the socket,
                # so one malformed header silently retires a serving thread and the retrieval hop
                # for a whole lane goes quiet without a single error being recorded anywhere.
                self._send(400, {"error": "negative Content-Length"}, close=True)
                return
            if length > MAX_BODY_BYTES:
                self._send(413, {"error": f"body above {MAX_BODY_BYTES} bytes"}, close=True)
                return
            self._dispatch(self.rfile.read(length) if length else b"")

        def _dispatch(self, raw: bytes) -> None:
            try:
                route = resolve_route(self.path)
                if route == "healthz":
                    self._send(200, service.health())
                    return
                if self.command != "POST":
                    raise ServiceError(405, "search is POST only")
                try:
                    body = json.loads(raw.decode("utf-8")) if raw else None
                except (json.JSONDecodeError, UnicodeDecodeError) as e:
                    raise ServiceError(400, f"body is not valid JSON: {e}") from e
                query, top_k = _parse_search_body(body)
                self._send(200, service.search(query, top_k))
            except ServiceError as e:
                self._send(e.status, {"error": str(e)})
            except Exception as e:  # noqa: BLE001 - an unexpected failure is still a failure
                self._send(500, {"error": f"{type(e).__name__}: {e}"})

    return Handler


class _RetrievalHTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def make_server(service: RetrievalService, config: ServiceConfig) -> _RetrievalHTTPServer:
    """Bind the listener. Refuses any address that is not loopback."""
    if config.bind_host not in LOOPBACK_HOSTS:
        raise ServiceStartupError(
            f"refusing to bind {config.bind_host!r}: the retrieval service is loopback-only")
    return _RetrievalHTTPServer((config.bind_host, config.bind_port), make_handler(service))


def serve_forever(service: RetrievalService, config: ServiceConfig) -> _RetrievalHTTPServer:
    """Start serving on a background thread and return the server, for a caller that can stop it."""
    server = make_server(service, config)
    threading.Thread(target=server.serve_forever, name="retrieval-http", daemon=True).start()
    return server


# --- entry point ---------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m shapeflow.retrieval.service",
        description="Serve the frozen BrowseComp-Plus retriever on loopback.")
    parser.add_argument("--index-dir", required=True, type=Path)
    parser.add_argument("--corpus-dir", required=True, type=Path)
    parser.add_argument(
        "--conformance", required=True, type=Path,
        help="the report proving this encoder reproduces this index. Required: the encoder "
             "recipe is read from it, so the served recipe is the validated one.")
    parser.add_argument(
        "--freeze", type=Path, default=None,
        help="protocol/retrieval_freeze.json. When given, the service refuses to start unless "
             "the freeze names this index, this corpus and this conformance report.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--cpuset", default="", help='e.g. "0-15"; applied before torch loads')
    parser.add_argument("--max-top-k", type=int, default=100)
    parser.add_argument("--max-query-chars", type=int, default=32_768)
    parser.add_argument("--cache-capacity", type=int, default=4096)
    parser.add_argument("--index-pattern", default="corpus.shard*.pkl")
    parser.add_argument("--corpus-pattern", default="*.parquet")
    parser.add_argument(
        "--check-artifacts-only", action="store_true",
        help="run the startup gates, print the identity, and exit without loading the model. "
             "Checks the recorded evidence, NOT the live encoder -- health prints warm=false.")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _build_parser().parse_args(argv)
    config = ServiceConfig(
        index_dir=args.index_dir, corpus_dir=args.corpus_dir,
        conformance_path=args.conformance, freeze_path=args.freeze,
        bind_host=args.host, bind_port=args.port, threads=args.threads, cpuset=args.cpuset,
        index_pattern=args.index_pattern, corpus_pattern=args.corpus_pattern,
        max_top_k=args.max_top_k, max_query_chars=args.max_query_chars,
        cache_capacity=args.cache_capacity,
    )
    try:
        # Before build_service, because build_service constructs the encoder and the encoder
        # imports torch on its first use. CUDA_VISIBLE_DEVICES is read at torch initialisation.
        harden_cpu_only(os.environ, threads=config.threads)
        if config.cpuset:
            apply_cpu_affinity(config.cpuset)
        service = build_service(config)
        if not args.check_artifacts_only:
            # A ServiceError here is the live encoder failing its first vector, which is a
            # startup refusal like any other: the process must not go on to serve.
            service.warm()
    except (ServiceStartupError, ServiceError) as e:
        print(f"REFUSED: {e}", file=sys.stderr)
        return 2

    if args.check_artifacts_only:
        print(json.dumps(service.health(), indent=2, sort_keys=True))
        return 0

    server = make_server(service, config)
    host, port = server.server_address[0], server.server_address[1]
    print(f"retrieval service on http://{host}:{port}", flush=True)
    print(f"  retriever_id {service.retriever_id}", flush=True)
    print(f"  index        {config.index_dir} "
          f"({service.identity['index']['num_docs']} docs, dim "
          f"{service.identity['index']['dim']})", flush=True)
    print(f"  encoder      {service.identity['encoder']['model']} "
          f"@ {service.identity['encoder']['revision'][:12]} "
          f"({service.identity['encoder']['dtype']}, cpu, {config.threads} threads)", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
