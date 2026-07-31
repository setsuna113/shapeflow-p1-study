"""The retrieval service and its client: identity, fail-closed refusals, and the loopback hop.

Runs against a tiny synthetic index and corpus with an injected encode function, so it needs
neither torch, nor a GPU, nor the 1.6 GB shipped index. What it exercises is everything *except*
the model: the startup gates that decide whether a service may exist at all, the closed wire
schema in both directions, and the property that matters most downstream -- a record fetched
through the socket is the same record the in-process backend would have produced.

Every guard here has a test that feeds it the bad input. A refusal that has only ever seen
correct input is an assumption.
"""

from __future__ import annotations

import contextlib
import dataclasses
import hashlib
import json
import os
import pickle
import re
import subprocess
import sys
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import pytest

from shapeflow.canonical import canonical_json
from shapeflow.hashing import sha256_hex
from shapeflow.retrieval import client as client_module
from shapeflow.retrieval.backend import BrowseCompPlusBackend
from shapeflow.retrieval.client import RetrievalClient, RetrievalServiceError, connect
from shapeflow.retrieval.corpus import load_corpus
from shapeflow.retrieval.encoder import EncoderSpec
from shapeflow.retrieval.freeze import RetrievalFreeze, write_freeze
from shapeflow.retrieval.index import load_index
from shapeflow.retrieval.service import (
    LOOPBACK_HOSTS,
    RetrievalService,
    ServiceConfig,
    ServiceError,
    ServiceStartupError,
    apply_cpu_affinity,
    build_service,
    encoder_spec_from_report,
    harden_cpu_only,
    load_conformance_report,
    main,
    make_server,
    parse_cpuset,
)
from shapeflow.world.search_backend import SearchBackend, SearchRecord

DIM = 16
DOCIDS = ["d0", "d1", "d2", "d3", "d4", "d5"]


# --- synthetic world ----------------------------------------------------------------------


def unit_from_text(text: str, dim: int = DIM) -> np.ndarray:
    """A deterministic unit vector for ``text``, derived from its digest.

    Digest bytes rather than a seeded PRNG: the fixture must be identical on every machine and
    every numpy version, and a PRNG implementation is neither.
    """
    material = hashlib.sha256(text.encode("utf-8")).digest()
    while len(material) < dim:
        material += hashlib.sha256(material).digest()
    vector = np.array([b - 127.5 for b in material[:dim]], dtype=np.float32)
    return vector / np.linalg.norm(vector)


class FakeEncoder:
    """Stands in for QueryEncoder. ``like:<docid>`` returns that document's own vector."""

    def __init__(self, vectors: dict[str, np.ndarray]) -> None:
        self.vectors = vectors
        self.calls: list[str] = []

    def __call__(self, text: str) -> np.ndarray:
        self.calls.append(text)
        if text.startswith("like:"):
            return self.vectors[text[len("like:"):]]
        return unit_from_text(text)


def _document_text(docid: str) -> str:
    return (f"---\ntitle: Document {docid}\n---\n"
            f"The body of {docid} discusses the frozen corpus at length. " * 3)


def _write_index(directory: Path, vectors: np.ndarray, docids: list[str], shards: int = 2) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    per = len(docids) // shards
    for i in range(shards):
        lo = i * per
        hi = len(docids) if i == shards - 1 else lo + per
        with open(directory / f"corpus.shard{i + 1}_of_{shards}.pkl", "wb") as handle:
            pickle.dump((vectors[lo:hi], docids[lo:hi]), handle)


def _write_corpus(directory: Path, docids: list[str]) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    directory.mkdir(parents=True, exist_ok=True)
    table = pa.table({
        "docid": docids,
        "text": [_document_text(d) for d in docids],
        "url": [f"https://corpus.invalid/{d}" for d in docids],
    })
    pq.write_table(table, directory / "part-0.parquet")


def _conformance_body(index_dir: Path, spec: EncoderSpec) -> dict:
    digests = [sha256_hex(p.read_bytes()) for p in sorted(index_dir.glob("corpus.shard*.pkl"))]
    return {
        "ok": True,
        "checked": 40,
        "tolerance": 0.999,
        "min_cosine": 0.999781,
        "rank_1_count": 40,
        "failures": [],
        "encoder_spec": dict(spec.content()),
        "index_shard_sha256": digests,
        "model": spec.model,
        "index_dir": str(index_dir),
        "dtype": spec.dtype,
        "diagnosis": "encoder reproduces the shipped index",
    }


@dataclasses.dataclass
class World:
    root: Path
    index_dir: Path
    corpus_dir: Path
    conformance_path: Path
    spec: EncoderSpec
    encoder: FakeEncoder
    vectors: dict[str, np.ndarray]

    def config(self, **overrides) -> ServiceConfig:
        base = ServiceConfig(
            index_dir=self.index_dir, corpus_dir=self.corpus_dir,
            conformance_path=self.conformance_path, bind_port=0)
        return dataclasses.replace(base, **overrides) if overrides else base

    def service(self, **overrides) -> RetrievalService:
        return build_service(self.config(**overrides), encode=self.encoder)

    def write_conformance(self, body: dict) -> None:
        self.conformance_path.write_text(json.dumps(body, indent=2, sort_keys=True),
                                         encoding="utf-8")

    def conformance(self) -> dict:
        return json.loads(self.conformance_path.read_text(encoding="utf-8"))

    def write_freeze(self, *, effective_after: str = "", path: Path | None = None,
                     **overrides) -> RetrievalFreeze:
        """A freeze naming exactly this world, unless an override says otherwise."""
        index = load_index(self.index_dir)
        corpus = load_corpus(self.corpus_dir)
        spec = self.spec.content()
        fields = dict(
            encoder_repo=spec["model"], encoder_revision=spec["revision"],
            encoder_dtype=spec["dtype"], pooling=spec["pooling"], normalize=spec["normalize"],
            query_prefix_sha256=spec["query_prefix_sha256"],
            passage_prefix_sha256=spec["passage_prefix_sha256"],
            query_max_len=spec["query_max_len"], passage_max_len=spec["passage_max_len"],
            index_subset=self.index_dir.name, index_dim=index.dim, index_num_docs=index.num_docs,
            index_shard_sha256=tuple(s.sha256 for s in index.shards),
            top_k=5, corpus_shard_sha256=tuple(corpus.shard_sha256),
            conformance_sha256=sha256_hex(self.conformance_path.read_bytes()),
            effective_after=effective_after,
        )
        fields.update(overrides)
        freeze = RetrievalFreeze(**fields)
        write_freeze(freeze, path or (self.root / "freeze.json"))
        return freeze


@pytest.fixture
def world(tmp_path: Path) -> World:
    vectors = {d: unit_from_text(f"doc:{d}") for d in DOCIDS}
    matrix = np.vstack([vectors[d] for d in DOCIDS]).astype(np.float32)
    index_dir, corpus_dir = tmp_path / "index", tmp_path / "corpus"
    _write_index(index_dir, matrix, DOCIDS)
    _write_corpus(corpus_dir, DOCIDS)
    spec = EncoderSpec()
    conformance_path = tmp_path / "conformance.json"
    conformance_path.write_text(
        json.dumps(_conformance_body(index_dir, spec), indent=2, sort_keys=True), encoding="utf-8")
    return World(root=tmp_path, index_dir=index_dir, corpus_dir=corpus_dir,
                 conformance_path=conformance_path, spec=spec,
                 encoder=FakeEncoder(vectors), vectors=vectors)


@contextlib.contextmanager
def running(service: RetrievalService, config: ServiceConfig):
    """Serve on an ephemeral loopback port for the duration of the block."""
    server = make_server(service, config)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


@contextlib.contextmanager
def stub_service(responder):
    """A hand-written HTTP stub, for responses a correct service would never produce."""

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *_args) -> None:  # keep the test output readable
            return

        def _serve(self) -> None:
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length).decode()) if length else None
            status, payload = responder(self.path, body)
            raw = json.dumps(payload).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        do_GET = _serve
        do_POST = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}"
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)


def post(base_url: str, path: str, body: dict) -> tuple[int, dict]:
    # ruff S310 (unconstrained url scheme) is answered by construction here: every base_url in
    # this file comes from a server this test just bound on loopback.
    request = urllib.request.Request(  # noqa: S310
        base_url + path, data=json.dumps(body).encode("utf-8"), method="POST",
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=10) as response:  # noqa: S310
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read().decode())


# --- the happy path -----------------------------------------------------------------------


def test_health_reports_the_identity_a_caller_can_pin(world: World):
    service = world.service()
    health = service.health()
    identity = health["retriever"]
    assert identity["index"]["num_docs"] == len(DOCIDS)
    assert identity["index"]["dim"] == DIM
    assert len(identity["index"]["shard_sha256"]) == 2
    assert identity["corpus"]["shard_sha256"], "the corpus must name the shards it was built from"
    assert identity["encoder"]["model"] == world.spec.model
    assert identity["encoder"]["dtype"] == world.spec.dtype
    assert identity["conformance"]["ok"] is True
    assert identity["conformance"]["sha256"] == sha256_hex(world.conformance_path.read_bytes())
    assert re.fullmatch(r"[0-9a-f]{64}", health["retriever_id"])
    # Two services over the same artifacts are the same retriever; identity must not depend on
    # process state, or no client could pin it.
    assert world.service().retriever_id == service.retriever_id


def test_a_search_returns_ranked_whole_documents(world: World):
    service = world.service()
    with running(service, world.config()) as base:
        status, payload = post(base, "/v1/search", {"query": "like:d3", "top_k": 3})
    assert status == 200
    assert [r["rank"] for r in payload["results"]] == [1, 2, 3]
    assert payload["results"][0]["docid"] == "d3"
    assert payload["results"][0]["score"] == pytest.approx(1.0, abs=1e-5)
    assert payload["results"][0]["title"] == "Document d3"
    # Full documents, not snippets: the declared workload deviation. The whole text is on the
    # wire, and the snippet is a separate, shorter field.
    assert payload["results"][0]["text"] == _document_text("d3")
    assert len(payload["results"][0]["snippet"]) < len(payload["results"][0]["text"])
    assert payload["retriever_id"] == service.retriever_id


def test_a_client_record_is_what_the_in_process_backend_would_have_produced(world: World):
    """The socket must be transparent, or the process boundary is itself a treatment.

    Occurrence ids, titles and snippets are all derived; if the service derived them differently
    from the in-process backend, the same query would produce different citation lineage
    depending on how the run happened to be deployed.
    """
    def through_the_seam(backend: SearchBackend) -> list[SearchRecord]:
        """Typed as the protocol, so the client is used exactly as the graph uses a backend."""
        return backend.search("like:d1", max_results=4)

    index, corpus = load_index(world.index_dir), load_corpus(world.corpus_dir)
    expected = through_the_seam(
        BrowseCompPlusBackend(index=index, corpus=corpus, encode=world.encoder))

    with running(world.service(), world.config()) as base:
        got = through_the_seam(RetrievalClient(base, require_effective_freeze=False))
    assert got == expected
    assert all(isinstance(r, SearchRecord) for r in got)


def test_a_repeated_query_is_a_cache_hit_and_the_rate_is_reported(world: World):
    service = world.service()
    with running(service, world.config()) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        client.search("like:d0", max_results=2)
        client.search("like:d0", max_results=2)
        client.search("like:d1", max_results=2)
        stats = client.stats()
    assert world.encoder.calls == ["like:d0", "like:d1"], "the second identical query must not " \
                                                          "reach the encoder"
    assert stats == {"searches": 3, "encodes": 2, "errors": 0, "hits": 1, "hit_rate": 0.3333,
                     "size": 2, "capacity": 4096}
    # The same arithmetic the in-process backend reports, so a per-arm number means one thing.
    assert service.health()["cache"]["hit_rate"] == pytest.approx(1.0 - 2 / 3, abs=1e-4)


def test_repeated_searches_are_byte_identical(world: World):
    with running(world.service(), world.config()) as base:
        first = post(base, "/v1/search", {"query": "reproducible", "top_k": 5})[1]
        second = post(base, "/v1/search", {"query": "reproducible", "top_k": 5})[1]
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)


def test_the_cache_is_bounded_and_evicts_in_insertion_order(world: World):
    service = world.service(cache_capacity=2)
    with running(service, world.config(cache_capacity=2)) as base:
        for query in ("one", "two", "three"):
            post(base, "/v1/search", {"query": query, "top_k": 1})
        health = json.loads(
            urllib.request.urlopen(base + "/healthz", timeout=10).read())  # noqa: S310
    assert health["cache"]["size"] == 2
    assert "one" not in service._cache and "three" in service._cache


# --- startup gates: the conformance report ------------------------------------------------


def test_a_missing_conformance_report_refuses_the_service(world: World):
    world.conformance_path.unlink()
    with pytest.raises(ServiceStartupError, match="no conformance report"):
        world.service()


def test_an_unparseable_conformance_report_is_a_failure_not_an_absence(world: World):
    world.conformance_path.write_text("{not json", encoding="utf-8")
    with pytest.raises(ServiceStartupError, match="does not parse"):
        world.service()


def test_a_conformance_report_recording_failure_refuses_the_service(world: World):
    body = world.conformance()
    body["ok"] = False
    body["diagnosis"] = "min cosine 0.2 -- the recipe is wrong"
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="does not reproduce this index"):
        world.service()


def test_a_truthy_but_non_true_ok_flag_is_still_a_refusal(world: World):
    """``ok: "yes"`` must not pass. A string is what a hand-edited report looks like."""
    body = world.conformance()
    body["ok"] = "yes"
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="records ok="):
        world.service()


def test_a_report_missing_its_evidence_refuses_the_service(world: World):
    body = world.conformance()
    del body["min_cosine"]
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="no 'min_cosine' field"):
        world.service()


def test_a_report_checked_in_another_dtype_than_it_serves_is_refused(world: World):
    body = world.conformance()
    body["dtype"] = "bfloat16"          # top level says bf16, the spec still says fp32
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="says nothing about a service running another"):
        world.service()


def test_a_report_for_another_index_refuses_the_service(world: World):
    body = world.conformance()
    body["index_shard_sha256"] = ["0" * 64, "1" * 64]
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="validated against a different index"):
        world.service()


def test_a_changed_query_instruction_invalidates_an_older_report(world: World):
    """The prefix digests are hashes of this repo's constants, not stored fields.

    So a report produced before someone edited the query instruction no longer describes the
    encoder this code implements, and the service refuses rather than serving one recipe under
    the evidence of another.
    """
    body = world.conformance()
    body["encoder_spec"]["query_prefix_sha256"] = "f" * 64
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="not the one this code implements"):
        world.service()


def test_a_report_with_no_encoder_recipe_cannot_be_defaulted(world: World):
    body = world.conformance()
    del body["encoder_spec"]["pooling"]
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="cannot be defaulted"):
        world.service()


@pytest.mark.parametrize("edit,expected", [
    # A run over no documents. `ok` is one boolean written by the process being gated; on its own
    # it is a gate satisfiable by asserting it has been satisfied.
    (dict(checked=0, min_cosine=0.0, rank_1_count=0), "passes vacuously"),
    # The flag and the numbers beside it disagree, and the numbers are the measurement.
    (dict(min_cosine=0.02), "under its own tolerance"),
    (dict(failures=[{"docid": "d0", "cosine": 0.1, "self_retrieved_rank": None}]), "failing"),
    # Cosines can pass while the docid alignment or the tie-break is broken.
    (dict(rank_1_count=0), "retrieved themselves at rank 1"),
    # The check re-run with the bar lowered until it passed, arriving as evidence.
    (dict(tolerance=0.5, min_cosine=0.6), "below this repo's floor"),
])
def test_a_report_whose_numbers_contradict_its_ok_flag_is_refused(world: World, edit, expected):
    """The gate is the evidence, not the summary bit written by the thing being gated."""
    body = world.conformance()
    body.update(edit)
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match=expected):
        world.service()


@pytest.mark.parametrize("field,value", [
    ("checked", "forty"),
    ("checked", 40.5),
    ("min_cosine", "high"),
    ("min_cosine", None),
    ("rank_1_count", [40]),
    ("tolerance", True),
])
def test_an_unreadable_evidence_field_is_a_refusal_not_a_traceback(world: World, field, value):
    """``int("forty")`` raises a ValueError that never reaches the entry point's REFUSED path."""
    body = world.conformance()
    body[field] = value
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError):
        world.service()


def test_a_non_finite_number_in_the_report_is_refused_at_the_gate(world: World):
    """``json.loads`` accepts the bare token ``NaN``; ``float()`` passes it straight through.

    Unstopped it reaches the identity digest, where canonical_json rejects it -- as a
    CanonicalizationError from inside build_service, which is not a startup refusal and does not
    make the entry point print REFUSED.
    """
    raw = json.dumps(world.conformance()).replace('"min_cosine": 0.999781', '"min_cosine": NaN')
    world.conformance_path.write_text(raw, encoding="utf-8")
    with pytest.raises(ServiceStartupError, match="non-finite"):
        world.service()


def test_the_recipe_is_taken_from_the_report_not_from_the_command_line(world: World):
    body = world.conformance()
    body["encoder_spec"]["model"] = "Qwen/Qwen3-Embedding-8B"
    body["encoder_spec"]["revision"] = "1d8ad4ca9b3dd8059ad90a75d4983776a23d44af"
    world.write_conformance(body)
    spec = encoder_spec_from_report(load_conformance_report(world.conformance_path))
    assert spec.model == "Qwen/Qwen3-Embedding-8B"
    assert world.service().identity["encoder"]["revision"].startswith("1d8ad4ca")


# --- startup gates: index, corpus and freeze ----------------------------------------------


def test_a_corpus_missing_an_indexed_document_refuses_the_service(world: World):
    _write_corpus(world.corpus_dir, DOCIDS[:-1])
    with pytest.raises(ServiceStartupError, match="different builds"):
        world.service()


def test_an_absent_index_is_an_error_not_an_empty_index(world: World):
    """An empty index answers every query with nothing, which reads like an empty corpus."""
    for shard in world.index_dir.glob("*.pkl"):
        shard.unlink()
    with pytest.raises(ServiceStartupError, match="no index shards"):
        world.service()


def test_a_corrupt_index_shard_is_a_refusal_not_a_traceback(world: World):
    next(iter(sorted(world.index_dir.glob("*.pkl")))).write_bytes(b"not a pickle")
    with pytest.raises(ServiceStartupError, match="cannot be loaded"):
        world.service()


def test_an_absent_corpus_is_an_error_not_an_empty_corpus(world: World):
    for shard in world.corpus_dir.glob("*.parquet"):
        shard.unlink()
    with pytest.raises(ServiceStartupError, match="cannot be loaded"):
        world.service()


def test_a_freeze_naming_another_conformance_report_refuses_the_service(world: World):
    world.write_freeze()
    body = world.conformance()
    # Any edit changes the file digest the freeze recorded. Edited here in a field the report's
    # own consistency rules do not constrain, so what this test proves is the freeze/report
    # binding and not, accidentally, one of the evidence checks above.
    body["diagnosis"] = "re-run on another host"
    world.write_conformance(body)
    with pytest.raises(ServiceStartupError, match="names conformance report"):
        world.service(freeze_path=world.root / "freeze.json")


def test_a_freeze_naming_another_index_refuses_the_service(world: World):
    world.write_freeze(index_shard_sha256=("a" * 64,))
    with pytest.raises(ServiceStartupError, match="different index shards"):
        world.service(freeze_path=world.root / "freeze.json")


def test_a_freeze_describing_another_encoder_refuses_the_service(world: World):
    world.write_freeze(encoder_dtype="bfloat16")
    with pytest.raises(ServiceStartupError, match="different encoder"):
        world.service(freeze_path=world.root / "freeze.json")


def test_an_edited_freeze_refuses_the_service(world: World):
    freeze_path = world.root / "freeze.json"
    world.write_freeze()
    body = json.loads(freeze_path.read_text())
    body["results"]["top_k"] = 50
    freeze_path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(ServiceStartupError, match="unusable"):
        world.service(freeze_path=freeze_path)


def test_a_matching_freeze_is_published_in_the_identity(world: World):
    freeze = world.write_freeze(effective_after="c" * 64)
    identity = world.service(freeze_path=world.root / "freeze.json").identity
    assert identity["freeze"] == {"digest": freeze.digest, "effective": True, "top_k": 5,
                                  "full_document": True}


# --- startup gates: deployment shape ------------------------------------------------------


def test_a_non_loopback_bind_is_refused(world: World):
    with pytest.raises(ServiceStartupError, match="loopback-only"):
        world.service(bind_host="0.0.0.0")  # noqa: S104 - refusing this is the point


def test_a_gpu_device_is_refused(world: World):
    with pytest.raises(ServiceStartupError, match="sustainable arrival rate"):
        world.service(device="cuda:0")


def test_the_evaluator_tree_cannot_be_loaded_as_a_corpus(world: World):
    """The leakage firewall's structural half: this process has no reader for that material.

    Pointed at the benchmark's evaluator-only artifacts, the corpus loader finds no parquet
    shards and the service refuses to start. There is no path by which relevance judgements or
    graded answers become documents here, so nothing on the treatment path can read them.
    """
    evaluator_dir = world.root / "evaluator"
    evaluator_dir.mkdir()
    (evaluator_dir / "judgements.txt").write_text("q1 Q0 d0 1\n", encoding="utf-8")
    (evaluator_dir / "answers.jsonl").write_text('{"query_id": "q1", "answer": "42"}\n',
                                                 encoding="utf-8")
    with pytest.raises(ServiceStartupError, match="cannot be loaded"):
        world.service(corpus_dir=evaluator_dir)


def test_a_nonsense_thread_count_is_refused(world: World):
    with pytest.raises(ServiceStartupError, match="threads must be a positive integer"):
        world.service(threads=0)


def test_cpu_hardening_hides_every_device_before_torch_loads():
    env = {"CUDA_VISIBLE_DEVICES": "0,1,2,3"}
    harden_cpu_only(env, threads=12)
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert env["OMP_NUM_THREADS"] == "12" and env["MKL_NUM_THREADS"] == "12"
    assert env["TOKENIZERS_PARALLELISM"] == "false"
    with pytest.raises(ServiceStartupError, match="positive integer"):
        harden_cpu_only({}, threads=0)


@pytest.mark.parametrize("spec", ["", "0-", "3-1", "a", "0,,2", "-1"])
def test_a_malformed_cpuset_is_an_error_not_an_empty_mask(spec: str):
    with pytest.raises(ServiceStartupError):
        parse_cpuset(spec)


def test_a_cpuset_is_parsed_and_applied():
    assert parse_cpuset("0-3,8") == [0, 1, 2, 3, 8]
    current = sorted(os.sched_getaffinity(0))
    # Re-applying this process's own affinity is a no-op that still proves the call path works.
    assert apply_cpu_affinity(",".join(str(c) for c in current)) == current


def test_a_cpuset_that_cannot_be_applied_is_fatal():
    with pytest.raises(ServiceStartupError, match="could not be applied"):
        apply_cpu_affinity("99999")


# --- request-level refusals ---------------------------------------------------------------


def test_an_empty_query_is_refused(world: World):
    with running(world.service(), world.config()) as base:
        status, payload = post(base, "/v1/search", {"query": "   ", "top_k": 3})
    assert status == 400 and "empty" in payload["error"]


def test_an_over_long_query_is_refused_rather_than_truncated(world: World):
    config = world.config(max_query_chars=64)
    with running(build_service(config, encode=world.encoder), config) as base:
        status, payload = post(base, "/v1/search", {"query": "x" * 65, "top_k": 3})
    assert status == 400 and "truncated" in payload["error"]


@pytest.mark.parametrize("top_k", [0, -1, 101, "5", True])
def test_a_top_k_outside_the_contract_is_refused(world: World, top_k):
    with running(world.service(), world.config()) as base:
        status, _ = post(base, "/v1/search", {"query": "anything", "top_k": top_k})
    assert status == 400


def test_an_unenumerated_request_field_is_refused_by_name(world: World):
    """``topk`` for ``top_k`` must not silently take a default and change what an arm saw."""
    with running(world.service(), world.config()) as base:
        status, payload = post(base, "/v1/search", {"query": "q", "top_k": 2, "topk": 9})
    assert status == 400 and "topk" in payload["error"]


def test_a_missing_top_k_has_no_default(world: World):
    with running(world.service(), world.config()) as base:
        status, payload = post(base, "/v1/search", {"query": "q"})
    assert status == 400 and "no default" in payload["error"]


def test_an_unknown_route_is_a_404(world: World):
    with running(world.service(), world.config()) as base:
        status, _ = post(base, "/v1/documents", {"docid": "d0"})
    assert status == 404


def _raw_exchange(port: int, request: bytes) -> bytes:
    """Send ``request`` verbatim and read until the server closes or stops talking."""
    import socket

    sock = socket.create_connection(("127.0.0.1", port), timeout=10)
    sock.settimeout(10)
    received = b""
    try:
        sock.sendall(request)
        while True:
            chunk = sock.recv(4096)
            if not chunk:
                return received
            received += chunk
    except TimeoutError:
        return received + b"<STILL OPEN AND SILENT>"
    finally:
        sock.close()


def test_a_negative_content_length_is_refused_rather_than_parking_the_thread(world: World):
    """``rfile.read(-1)`` reads to EOF, and on a keep-alive connection EOF never comes.

    Unfixed, this test hangs for its whole timeout: the handler thread is retired for the life of
    the socket and the retrieval hop for that lane goes quiet without anything being recorded.
    """
    with running(world.service(), world.config()) as base:
        got = _raw_exchange(int(base.rsplit(":", 1)[1]),
                            b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: -1\r\n\r\n")
    assert got.startswith(b"HTTP/1.1 400"), got[:120]
    assert b"<STILL OPEN AND SILENT>" not in got, "the connection was left desynchronized"


def test_an_over_large_body_is_refused_and_the_connection_is_not_left_desynchronized(
        world: World):
    """The refusal answers without draining the body, so the connection has to be closed.

    Left open, the next read for a request line lands in the middle of the undrained body and
    blocks on a newline that is never coming -- the same parked thread the size check was
    supposed to avoid, reached by the path the size check itself opens.
    """
    body = b'{"query":"x","top_k":1}' + b" " * 64      # no newline anywhere in it
    with running(world.service(), world.config()) as base:
        got = _raw_exchange(
            int(base.rsplit(":", 1)[1]),
            b"POST /v1/search HTTP/1.1\r\nHost: x\r\nContent-Length: 99999999\r\n\r\n" + body)
    assert got.startswith(b"HTTP/1.1 413"), got[:120]
    assert b"<STILL OPEN AND SILENT>" not in got, "the connection was left desynchronized"


def test_a_body_that_is_not_json_is_refused(world: World):
    with running(world.service(), world.config()) as base:
        request = urllib.request.Request(base + "/v1/search", data=b"{oops",  # noqa: S310
                                         method="POST")
        try:
            urllib.request.urlopen(request, timeout=10)  # noqa: S310
            status = 200
        except urllib.error.HTTPError as e:
            status, payload = e.code, json.loads(e.read().decode())
    assert status == 400 and "not valid JSON" in payload["error"]


# --- encoder faults -----------------------------------------------------------------------


def test_a_wrong_dimension_encoder_is_caught_at_warmup(world: World):
    service = build_service(world.config(), encode=lambda text: np.ones(8) / np.sqrt(8))
    with pytest.raises(ServiceError, match="not the one this index was built with"):
        service.warm()


def test_a_bad_vector_is_never_written_to_the_cache(world: World):
    """A cached wrong-dimension vector would fail identically forever, including after a fix."""
    service = build_service(world.config(), encode=lambda text: np.ones(8) / np.sqrt(8))
    with running(service, world.config()) as base:
        status, payload = post(base, "/v1/search", {"query": "q", "top_k": 2})
    assert status == 500 and "dimension" in payload["error"]
    assert service.health()["cache"] == {"searches": 1, "encodes": 1, "errors": 1, "hits": 0,
                                         "hit_rate": 0.0, "size": 0, "capacity": 4096}


def test_an_unnormalised_vector_is_refused(world: World):
    service = build_service(world.config(), encode=lambda text: np.ones(DIM))
    with running(service, world.config()) as base:
        status, payload = post(base, "/v1/search", {"query": "q", "top_k": 2})
    assert status == 500 and "L2-normalised" in payload["error"]


def test_an_encoder_returning_nothing_is_refused(world: World):
    service = build_service(world.config(), encode=lambda text: None)
    with pytest.raises(ServiceError, match="no vector"):
        service.warm()


def test_a_non_finite_score_is_refused_rather_than_served(tmp_path: Path):
    """A NaN score survives json.loads on the client side, so it has to stop at the service."""
    vectors = {d: unit_from_text(f"doc:{d}") for d in DOCIDS}
    matrix = np.vstack([vectors[d] for d in DOCIDS]).astype(np.float32)
    matrix[2][0] = np.nan
    index_dir, corpus_dir = tmp_path / "index", tmp_path / "corpus"
    _write_index(index_dir, matrix, DOCIDS)
    _write_corpus(corpus_dir, DOCIDS)
    conformance_path = tmp_path / "conformance.json"
    conformance_path.write_text(json.dumps(_conformance_body(index_dir, EncoderSpec())),
                                encoding="utf-8")
    config = ServiceConfig(index_dir=index_dir, corpus_dir=corpus_dir,
                           conformance_path=conformance_path, bind_port=0)
    service = build_service(config, encode=FakeEncoder(vectors))
    with running(service, config) as base:
        status, payload = post(base, "/v1/search", {"query": "probe", "top_k": len(DOCIDS)})
    assert status == 500 and "non-finite score" in payload["error"]


# --- the client -----------------------------------------------------------------------------


def test_the_client_and_the_service_agree_on_what_loopback_means():
    assert client_module.LOOPBACK_HOSTS == LOOPBACK_HOSTS


def test_the_client_refuses_a_retriever_off_this_host():
    with pytest.raises(RetrievalServiceError, match="loopback only"):
        RetrievalClient("http://10.0.0.4:8710")


def test_an_unreachable_service_is_a_failure_not_an_empty_result():
    """Closed port. If this returned [], "down" and "nothing relevant" would be one observation."""
    client = RetrievalClient("http://127.0.0.1:1", require_effective_freeze=False)
    with pytest.raises(RetrievalServiceError, match="unreachable") as caught:
        client.search("anything", max_results=3)
    assert caught.value.status == 503


def test_the_client_refuses_a_freeze_that_is_not_effective(world: World):
    world.write_freeze(effective_after="")
    config = world.config(freeze_path=world.root / "freeze.json")
    with running(build_service(config, encode=world.encoder), config) as base:
        with pytest.raises(RetrievalServiceError, match="not in force"):
            connect(base)
        # The competence pilot is the caller that legitimately opts out, and does so explicitly.
        assert connect(base, require_effective_freeze=False).search("q", max_results=1)


def test_the_client_accepts_an_effective_freeze_it_was_pinned_to(world: World):
    freeze = world.write_freeze(effective_after="c" * 64)
    config = world.config(freeze_path=world.root / "freeze.json")
    with running(build_service(config, encode=world.encoder), config) as base:
        client = connect(base, expected_freeze_digest=freeze.digest)
        assert len(client.search("q", max_results=2)) == 2
        with pytest.raises(RetrievalServiceError, match="expected"):
            connect(base, expected_freeze_digest="b" * 64)


def test_the_client_refuses_a_service_with_no_declared_freeze(world: World):
    with running(world.service(), world.config()) as base:
        with pytest.raises(RetrievalServiceError, match="declares no retrieval freeze"):
            connect(base)


def test_the_client_refuses_a_retriever_id_it_was_not_pinned_to(world: World):
    with running(world.service(), world.config()) as base:
        with pytest.raises(RetrievalServiceError, match="different retriever"):
            connect(base, expected_retriever_id="a" * 64, require_effective_freeze=False)


def _health_of(world: World) -> dict:
    with running(world.service(), world.config()) as base:
        return json.loads(urllib.request.urlopen(base + "/healthz", timeout=10).read())  # noqa: S310


def test_the_client_refuses_results_from_an_identity_that_changed_mid_run(world: World):
    """A service restarted against another index between the health check and the next search."""
    health = _health_of(world)

    def responder(path, _body):
        if path == "/healthz":
            return 200, health
        return 200, {"query": "q", "top_k": 1, "count": 0, "results": [],
                     "retriever_id": "d" * 64}

    with stub_service(responder) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        with pytest.raises(RetrievalServiceError, match="changed identity mid-run"):
            client.search("q", max_results=1)


@pytest.mark.parametrize("mangle,expected", [
    (lambda r: r.__setitem__("relevance_label", 1), "unenumerated field"),
    (lambda r: r.__setitem__("rank", 7), "ranks must be the positions"),
    (lambda r: r.__setitem__("score", float("nan")), "non-finite score"),
    (lambda r: r.__setitem__("occurrence_id", "short"), "not a derived id"),
    (lambda r: r.__setitem__("text", None), "not a string"),
    (lambda r: r.pop("snippet"), "missing"),
])
def test_the_client_refuses_a_result_that_does_not_fit_the_closed_schema(
        world: World, mangle, expected):
    """An unenumerated field is refused, not ignored: it is where a qrel judgement would ride."""
    health = _health_of(world)
    with running(world.service(), world.config()) as base:
        truth = post(base, "/v1/search", {"query": "q", "top_k": 1})[1]
    truth["retriever_id"] = health["retriever_id"]
    mangle(truth["results"][0])

    def responder(path, _body):
        return (200, health) if path == "/healthz" else (200, truth)

    with stub_service(responder) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        with pytest.raises(RetrievalServiceError, match=expected):
            client.search("q", max_results=1)


@pytest.mark.parametrize("mangle,expected", [
    (lambda p: p.__setitem__("query", "a different question"), "different query"),
    (lambda p: p.__setitem__("top_k", 9), "response says top_k"),
    (lambda p: p.__setitem__("count", 5), "count="),
    (lambda p: p.__setitem__("debug_qrels", {}), "unenumerated field"),
    (lambda p: p.pop("results"), "missing"),
])
def test_the_client_refuses_a_response_that_does_not_fit_the_closed_schema(
        world: World, mangle, expected):
    health = _health_of(world)
    with running(world.service(), world.config()) as base:
        truth = post(base, "/v1/search", {"query": "q", "top_k": 1})[1]
    truth["retriever_id"] = health["retriever_id"]
    mangle(truth)

    def responder(path, _body):
        return (200, health) if path == "/healthz" else (200, truth)

    with stub_service(responder) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        with pytest.raises(RetrievalServiceError, match=expected):
            client.search("q", max_results=1)


def test_the_client_refuses_more_results_than_it_asked_for(world: World):
    """One arm reading more than the other is the fairness invariant, not a formatting detail."""
    health = _health_of(world)
    with running(world.service(), world.config()) as base:
        truth = post(base, "/v1/search", {"query": "q", "top_k": 3})[1]
    truth["retriever_id"] = health["retriever_id"]
    truth["top_k"] = 1
    truth["count"] = 3

    def responder(path, _body):
        return (200, health) if path == "/healthz" else (200, truth)

    with stub_service(responder) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        with pytest.raises(RetrievalServiceError, match="returns more than it was asked"):
            client.search("q", max_results=1)


def test_the_client_refuses_duplicate_docids(world: World):
    health = _health_of(world)
    with running(world.service(), world.config()) as base:
        truth = post(base, "/v1/search", {"query": "q", "top_k": 2})[1]
    truth["retriever_id"] = health["retriever_id"]
    truth["results"][1] = dict(truth["results"][1], docid=truth["results"][0]["docid"])

    def responder(path, _body):
        return (200, health) if path == "/healthz" else (200, truth)

    with stub_service(responder) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        with pytest.raises(RetrievalServiceError, match="appears twice"):
            client.search("q", max_results=2)


def test_the_client_reports_a_service_error_with_its_status(world: World):
    with running(world.service(), world.config()) as base:
        client = RetrievalClient(base, require_effective_freeze=False)
        with pytest.raises(RetrievalServiceError) as caught:
            client.search("q", max_results=10_000)
    assert caught.value.status == 400


def test_the_client_rejects_a_bad_request_before_the_round_trip(world: World):
    client = RetrievalClient("http://127.0.0.1:1", require_effective_freeze=False)
    with pytest.raises(RetrievalServiceError, match="non-empty string"):
        client.search("", max_results=3)
    with pytest.raises(RetrievalServiceError, match="positive integer"):
        client.search("q", max_results=0)


def test_search_rows_carry_the_docid_the_search_record_cannot(world: World):
    with running(world.service(), world.config()) as base:
        rows = RetrievalClient(base, require_effective_freeze=False).search_rows(
            "like:d4", max_results=2)
    assert rows[0]["docid"] == "d4"
    assert [r["rank"] for r in rows] == [1, 2]


def test_the_client_does_not_follow_a_redirect_off_the_verified_address(world: World):
    """A 302 relocates the frozen world *after* the loopback check has passed.

    urllib's default opener chases ``Location`` anywhere, including to another machine, so the
    base_url check was decorative: one header and the arms are reading a different index. The
    second stub here stands for that other index, and must never be contacted.
    """
    elsewhere = {"contacted": 0}

    def other_index(_path, _body):
        elsewhere["contacted"] += 1
        return 200, {"status": "ok", "retriever_id": "e" * 64, "retriever": {"freeze": None}}

    with stub_service(other_index) as away:
        # stub_service writes a JSON body and cannot set Location, so this one is hand-rolled.
        class Redirect(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_args) -> None:
                return

            def _serve(self) -> None:
                self.send_response(302)
                self.send_header("Location", away + self.path)
                self.send_header("Content-Length", "0")
                self.end_headers()

            do_GET = _serve
            do_POST = _serve

        server = ThreadingHTTPServer(("127.0.0.1", 0), Redirect)
        server.daemon_threads = True
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = RetrievalClient(f"http://127.0.0.1:{server.server_address[1]}",
                                     require_effective_freeze=False)
            with pytest.raises(RetrievalServiceError, match="not followed off the address"):
                client.search("q", max_results=1)
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)
    assert elsewhere["contacted"] == 0, "the client read an index it never verified"


def test_the_client_ignores_an_http_proxy_in_the_environment(world: World, monkeypatch):
    """``urlopen`` routes through ``$http_proxy`` even for 127.0.0.1.

    The run host's network is filtered and has one configured, so unfixed this sends every
    retrieval request to the proxy and treats whatever comes back as the frozen world.
    """
    seen: list[str] = []

    def proxy(path, _body):
        seen.append(path)
        return 200, {"status": "ok"}

    with stub_service(proxy) as proxy_url:
        monkeypatch.setenv("http_proxy", proxy_url)
        monkeypatch.setenv("HTTP_PROXY", proxy_url)
        monkeypatch.delenv("no_proxy", raising=False)
        monkeypatch.delenv("NO_PROXY", raising=False)
        with running(world.service(), world.config()) as base:
            rows = RetrievalClient(base, require_effective_freeze=False).search_rows(
                "like:d2", max_results=1)
    assert rows[0]["docid"] == "d2"
    assert seen == [], f"the loopback retrieval hop left this process through a proxy: {seen}"


def test_the_clients_opener_takes_the_environment_out_of_the_decision(monkeypatch):
    """Asserted structurally as well as behaviourally, and against a live counterexample.

    ``urlopen`` memoises its opener on first use, so a purely behavioural proxy test can pass for
    the accidental reason that some earlier test in the session already built one while no proxy
    was set. The handler set is the property that actually holds, in any order.
    """
    monkeypatch.setenv("http_proxy", "http://10.0.0.9:3128")
    monkeypatch.setenv("HTTP_PROXY", "http://10.0.0.9:3128")
    monkeypatch.delenv("no_proxy", raising=False)
    monkeypatch.delenv("NO_PROXY", raising=False)

    def proxied(opener):
        return [h.proxies for h in opener.handlers
                if isinstance(h, urllib.request.ProxyHandler) and h.proxies]

    # The counterexample: the stock opener, built right here, is proxied. Without this the
    # assertions below could be passing because the environment has no proxy to pick up.
    assert proxied(urllib.request.build_opener()), \
        "this environment must actually configure a proxy, or the rest of this test proves nothing"

    # ``_OPENER`` is built at import (so the environment at call time cannot reach it) and
    # ``_build_opener`` is re-run here (so the environment at build time cannot either).
    for opener in (client_module._OPENER, client_module._build_opener()):  # noqa: SLF001
        assert not proxied(opener), f"the client's opener carries proxies: {proxied(opener)}"
        redirectors = [h for h in opener.handlers
                       if isinstance(h, urllib.request.HTTPRedirectHandler)]
        assert redirectors and all(
            isinstance(h, client_module._RefuseRedirect) for h in redirectors), (  # noqa: SLF001
            "a stock HTTPRedirectHandler follows Location anywhere, including off this host")


def test_the_client_requires_the_retriever_id_to_be_the_digest_of_the_identity(world: World):
    """Otherwise pinning a freeze digest pins a label printed beside the world, not the world."""
    health = _health_of(world)
    assert health["retriever_id"] == sha256_hex(canonical_json(health["retriever"])), \
        "the client recomputes the service's derivation; the two must not drift apart"

    swapped = dict(health, retriever_id="a" * 64)

    with stub_service(lambda path, body: (200, swapped)) as base:
        with pytest.raises(RetrievalServiceError, match="is not the digest of the identity"):
            RetrievalClient(base, require_effective_freeze=False).verify()

    # And the other direction: an identity edited under an unchanged id. A service that swapped
    # its index but kept printing the old digest is exactly what a mid-run restart looks like.
    relabelled = json.loads(json.dumps(health))
    relabelled["retriever"]["index"]["shard_sha256"] = ["b" * 64]
    with stub_service(lambda path, body: (200, relabelled)) as base:
        with pytest.raises(RetrievalServiceError, match="is not the digest of the identity"):
            RetrievalClient(base, require_effective_freeze=False).verify()


def test_a_health_response_without_an_identity_is_refused():
    with stub_service(lambda path, body: (200, {"status": "ok"})) as base:
        with pytest.raises(RetrievalServiceError, match="no retriever identity"):
            RetrievalClient(base, require_effective_freeze=False).verify()


# --- the entry point ------------------------------------------------------------------------


def test_check_artifacts_only_runs_the_gates_and_reports_a_cold_encoder(world: World, capsys,
                                                                       monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    code = main(["--index-dir", str(world.index_dir), "--corpus-dir", str(world.corpus_dir),
                 "--conformance", str(world.conformance_path), "--check-artifacts-only"])
    assert code == 0
    health = json.loads(capsys.readouterr().out)
    assert health["warm"] is False, "the live encoder was not loaded and must not claim to be"
    assert health["retriever"]["index"]["num_docs"] == len(DOCIDS)
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_the_entry_point_refuses_rather_than_starting_degraded(world: World, capsys, monkeypatch):
    monkeypatch.setattr(os, "environ", dict(os.environ))
    world.conformance_path.unlink()
    code = main(["--index-dir", str(world.index_dir), "--corpus-dir", str(world.corpus_dir),
                 "--conformance", str(world.conformance_path), "--check-artifacts-only"])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err


def test_importing_the_client_pulls_neither_torch_nor_a_transport_library():
    """The whole point of the process split. Checked in a fresh interpreter, not this one.

    Asserting against ``sys.modules`` here would prove nothing: some other test in the same
    session may already have imported anything. A subprocess is the only place the claim "the
    client adds nothing to the interpreter that runs the pinned vendor agent" is actually
    testable.
    """
    probe = (
        "import sys; import shapeflow.retrieval.client as c;"
        "loaded = [m for m in ('torch', 'transformers', 'httpx', 'pyarrow') if m in sys.modules];"
        "print(loaded);"
        "assert not loaded, loaded;"
        "assert hasattr(c.RetrievalClient, 'search')"
    )
    result = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
