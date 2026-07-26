"""`evaluate` end to end, with a real ledger, a real object store and a scripted judge.

There was no test of this path at all, and it could not have passed one: the relation
classifier called ``get_event_loop().run_until_complete`` from inside the loop that was
already running it, so the first judged claim raised "this event loop is already running".
Every quality metric was 0.0 anyway, because ``citation_supports`` returned None for every
citation and ``covered`` requires a supporting one.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest

from shapeflow_p1.campaign.evaluate import (
    _citation_supports_for,
    _load_frozen_scope,
    _load_verified_truth_artifact,
    _ReadOnlyLedgerView,
    _verified_support_index,
    evaluate_frozen,
)
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.evaluation.runner import EvaluationError
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore

REPO = Path(__file__).resolve().parents[2]
EXECUTION_BINDING_SHA256 = "e" * 64
PROTOCOL_DOCUMENT_SHA256 = "a" * 64

_REPORT = """\
The reactor reached 41% efficiency in 2025 [1].

Sources:
[1] Efficiency review -- https://example.com/efficiency
"""

_PAGE = "The reactor reached 41% efficiency in 2025, the review reported."


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


@pytest.fixture(autouse=True)
def approved_execution_binding(monkeypatch):
    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module,
        "verified_execution_binding",
        lambda _repo, expected_digest=None: SimpleNamespace(
            digest=EXECUTION_BINDING_SHA256,
            protocol_sha=PROTOCOL_DOCUMENT_SHA256,
        ),
    )


async def _evaluate(settings, **kwargs):
    return await evaluate_frozen(
        settings,
        execution_binding_sha256=EXECUTION_BINDING_SHA256,
        protocol_document_sha256=PROTOCOL_DOCUMENT_SHA256,
        **kwargs,
    )


def _truth(task_id: str, settings, pool_sha256: str) -> dict:
    from shapeflow_p1.campaign.truth import (
        TRUTH_PROMPT_VERSION,
        _excerpt_spans,
        _h_candidate_spans,
        truth_prompt_sha256,
    )
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    truth_span = _excerpt_spans(settings, task_id)[0]
    candidate_spans = _h_candidate_spans(settings, task_id)
    packet = {
        "task_id": task_id,
        "required_facets": ["f1"],
        "atomic_evidence": [
            {"atom_id": "a1", "facet_id": "f1", "critical": False, "weight": 1.0,
             "known_unresolved": False,
             "supporting_span_ids": [truth_span["span_id"]]},
        ],
        "contradiction_pairs": [],
        "negative_evidence": [],
        "known_gaps": [],
        "critical_items": [],
        "authoring_method": "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT",
        "verifier_status": "PENDING",
        "content_sha256": "",
    }
    packet["content_sha256"] = sha256_hex(canonical_json(packet))
    support_index = {
        "version": "h_atom_support_index_v2",
        "chunk_max_tokens": 320,
        "semantic_checks": 0,
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            chunker: {
                span["span_id"]: sorted(span["source_occurrence_ids"])
                for span in spans
            }
            for chunker, spans in candidate_spans.items()
        },
        "chunkers": {
            chunker: {"a1": []} for chunker in candidate_spans
        },
    }
    support_index["content_sha256"] = sha256_hex(canonical_json(support_index))
    task_binding = sha256_hex(canonical_json({
        "task_id": task_id,
        "question": "What efficiency did the reactor reach?",
        "required_facets": ["f1"],
    }))
    acquisition = json.loads(
        (settings.path("acquisition") / f"{task_id}.json").read_text(encoding="utf-8")
    )
    attempts = [{
        "query_attempt_id": query["query_snapshot_id"],
        "query": query["query_text"],
        "status": query["status"],
    } for query in acquisition["queries"]]
    attempts_sha = sha256_hex(canonical_json(attempts))
    body = {
        "packet": packet,
        "atom_texts": {"a1": "41% efficiency in 2025"},
        "rejected": [],
        "provenance": {
            "config_sha256s": dict(sorted(settings.shas.items())),
            "claim_scope": settings.claim_scope,
            "source_pool_sha256": pool_sha256,
            "prompt_version": TRUTH_PROMPT_VERSION,
            "prompt_sha256": truth_prompt_sha256(),
            "task_question_facets_sha256": task_binding,
            "query_attempts_sha256": attempts_sha,
            "acquisition_digest": acquisition["acquisition_digest"],
            "h_atom_support_index_sha256": support_index["content_sha256"],
            "truth_source_binding_sha256": sha256_hex(canonical_json({
                "task_question_facets_sha256": task_binding,
                "source_pool_sha256": pool_sha256,
                "acquisition_digest": acquisition["acquisition_digest"],
                "query_attempts_sha256": attempts_sha,
            })),
        },
        "atom_support_index": support_index,
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _world(settings, task_id: str) -> None:
    """One frozen page, published exactly the way acquisition publishes it."""
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    ref = objects.put_bytes(_PAGE.encode("utf-8"))
    pool = {
        "task_id": task_id,
        "occurrences": [{"occurrence_id": "o1", "url": "https://example.com/efficiency",
                         "title": "Efficiency review", "snippet_content": "41%",
                         "content_hash": "h1", "vendor_visible_order": 0}],
        "snapshots": {"h1": {"object_ref": ref.key, "byte_len": len(_PAGE),
                             "raw_content_format": "exa_text",
                             "normalization_version": "v1",
                             "fetched_at_utc": "2026-07-25T00:00:00Z"}},
    }
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    pool["pool_sha256"] = sha256_hex(canonical_json(pool))
    path = settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool), encoding="utf-8")

    task_dir = settings.path("evaluator_root") / "tasks"
    task_dir.mkdir(parents=True, exist_ok=True)
    (task_dir / f"{task_id}.json").write_text(json.dumps({
        "task_id": task_id,
        "split": "FORMATIVE_SCREEN",
        "cluster_id": "cluster-1",
        "original_question": "What efficiency did the reactor reach?",
        "authored_facets": ["f1"],
        "fixed_queries": ["reactor efficiency"],
        "conflict_probe": "",
        "negative_or_gap_probe": "",
        "acquisition_spec_sha256": "s" * 64,
        "strata": [],
    }), encoding="utf-8")

    acquisition = {
        "task_id": task_id,
        "acquisition_spec_sha256": "s" * 64,
        "fetched_at_utc": "2026-07-25T00:00:00Z",
        "tavily_params": {},
        "queries": [{
            "query_snapshot_id": "q1",
            "query_text": "reactor efficiency",
            "status": "SUCCESS",
            "request_id": "request-1",
            "response_time": 0.1,
            "usage": {},
            "failed_results": [],
            "raw_response_sha256": "r" * 64,
            "result_count": 1,
        }],
        "occurrences": [{
            "occurrence_id": "o1",
            "query_snapshot_id": "q1",
            "url": "https://example.com/efficiency",
            "title": "Efficiency review",
            "rank": 0,
            "score": None,
            "published_date": None,
            "content_hash": "h1",
            "visibility": "VENDOR_VISIBLE",
            "duplicate_of_occurrence_id": None,
            "vendor_visible_order": 0,
        }],
        "snapshots": [{
            "content_hash": "h1",
            "raw_content_format": "exa_text",
            "byte_len": len(_PAGE),
            "object_ref": ref.key,
            "normalization_version": "v1",
            "fetched_at_utc": "2026-07-25T00:00:00Z",
        }],
    }
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    acquisition["merkle_root"] = "m" * 64
    acquisition["acquisition_digest"] = sha256_hex(canonical_json({
        key: value for key, value in acquisition.items()
        if key not in {"merkle_root", "acquisition_digest"}
    }))
    acquisition_path = settings.path("acquisition") / f"{task_id}.json"
    acquisition_path.parent.mkdir(parents=True, exist_ok=True)
    acquisition_path.write_text(json.dumps(acquisition), encoding="utf-8")

    truth_dir = settings.path("truth_packets")
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / f"{task_id}.json").write_text(
        json.dumps(_truth(task_id, settings, pool["pool_sha256"])),
        encoding="utf-8",
    )


def _pool(settings, task_id: str) -> dict:
    return json.loads(
        (settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json")
        .read_text(encoding="utf-8")
    )


def _rehash_truth(body: dict, *, packet: bool = False, support_index: bool = False) -> dict:
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    if packet:
        body["packet"]["content_sha256"] = ""
        body["packet"]["content_sha256"] = sha256_hex(canonical_json(body["packet"]))
    if support_index:
        body["atom_support_index"].pop("content_sha256", None)
        body["atom_support_index"]["content_sha256"] = sha256_hex(
            canonical_json(body["atom_support_index"]))
    body.pop("content_sha256", None)
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _verify_truth(settings, task_id: str) -> dict:
    path = settings.path("truth_packets") / f"{task_id}.json"
    truth = _load_verified_truth_artifact(
        path, settings=settings, expected_task_id=task_id)
    _verified_support_index(
        truth, path, settings=settings, task_id=task_id,
        pool=_pool(settings, task_id))
    return truth


def test_truth_rejects_post_authoring_task_facet_drift(settings):
    task_id = "T-truth-task-drift"
    _world(settings, task_id)
    task_path = settings.path("evaluator_root") / "tasks" / f"{task_id}.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    task["authored_facets"] = ["different-facet"]
    task_path.write_text(json.dumps(task), encoding="utf-8")

    with pytest.raises(EvaluationError, match="required facets|question/facet"):
        _verify_truth(settings, task_id)


def test_truth_rejects_resealed_acquisition_query_drift(settings):
    task_id = "T-truth-query-drift"
    _world(settings, task_id)
    path = settings.path("acquisition") / f"{task_id}.json"
    acquisition = json.loads(path.read_text(encoding="utf-8"))
    acquisition["queries"][0]["query_text"] = "post-outcome replacement query"
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    acquisition["acquisition_digest"] = sha256_hex(canonical_json({
        key: value for key, value in acquisition.items()
        if key not in {"merkle_root", "acquisition_digest"}
    }))
    path.write_text(json.dumps(acquisition), encoding="utf-8")

    with pytest.raises(EvaluationError, match="acquisition provenance|query-attempt"):
        _verify_truth(settings, task_id)


def test_truth_rejects_a_resealed_pool_that_is_not_the_acquisition_projection(settings):
    task_id = "T-truth-pool-projection"
    _world(settings, task_id)
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    pool_path = (
        settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json")
    pool = json.loads(pool_path.read_text(encoding="utf-8"))
    pool["occurrences"][0]["url"] = "https://post-outcome.invalid/replacement"
    pool.pop("pool_sha256")
    pool["pool_sha256"] = sha256_hex(canonical_json(pool))
    pool_path.write_text(json.dumps(pool), encoding="utf-8")

    truth_path = settings.path("truth_packets") / f"{task_id}.json"
    truth = json.loads(truth_path.read_text(encoding="utf-8"))
    truth["provenance"]["source_pool_sha256"] = pool["pool_sha256"]
    truth["provenance"]["truth_source_binding_sha256"] = sha256_hex(canonical_json({
        "task_question_facets_sha256":
            truth["provenance"]["task_question_facets_sha256"],
        "source_pool_sha256": pool["pool_sha256"],
        "acquisition_digest": truth["provenance"]["acquisition_digest"],
        "query_attempts_sha256": truth["provenance"]["query_attempts_sha256"],
    }))
    _rehash_truth(truth)
    truth_path.write_text(json.dumps(truth), encoding="utf-8")

    with pytest.raises(EvaluationError, match="vendor-visible projection"):
        _verify_truth(settings, task_id)


def test_truth_rejects_a_self_consistent_support_index_with_stale_pointer(settings):
    task_id = "T-truth-index-pointer"
    _world(settings, task_id)
    path = settings.path("truth_packets") / f"{task_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["atom_support_index"]["semantic_checks"] = 99
    _rehash_truth(body, support_index=True)
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(EvaluationError, match="provenance pointer"):
        _verify_truth(settings, task_id)


def test_truth_rejects_a_repointed_index_outside_occurrence_universe(settings):
    task_id = "T-truth-index-universe"
    _world(settings, task_id)
    path = settings.path("truth_packets") / f"{task_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    chunker = next(iter(body["atom_support_index"]["candidate_span_occurrence_ids"]))
    span_id = next(iter(
        body["atom_support_index"]["candidate_span_occurrence_ids"][chunker]))
    body["atom_support_index"]["candidate_span_occurrence_ids"][chunker][span_id] = [
        "foreign-occurrence"
    ]
    _rehash_truth(body, support_index=True)
    body["provenance"]["h_atom_support_index_sha256"] = (
        body["atom_support_index"]["content_sha256"])
    _rehash_truth(body)
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(EvaluationError, match="span/occurrence universe"):
        _verify_truth(settings, task_id)


def test_truth_rejects_a_rehashed_atom_span_outside_frozen_world(settings):
    task_id = "T-truth-span-universe"
    _world(settings, task_id)
    path = settings.path("truth_packets") / f"{task_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["packet"]["atomic_evidence"][0]["supporting_span_ids"] = ["f" * 64]
    _rehash_truth(body, packet=True)
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(EvaluationError, match="outside the frozen truth universe"):
        _verify_truth(settings, task_id)


def test_truth_rejects_a_rehashed_source_binding_claim(settings):
    task_id = "T-truth-source-binding"
    _world(settings, task_id)
    path = settings.path("truth_packets") / f"{task_id}.json"
    body = json.loads(path.read_text(encoding="utf-8"))
    body["provenance"]["truth_source_binding_sha256"] = "0" * 64
    _rehash_truth(body)
    path.write_text(json.dumps(body), encoding="utf-8")

    with pytest.raises(EvaluationError, match="truth-source binding"):
        _verify_truth(settings, task_id)


def test_citation_measurement_rejects_a_malformed_occurrence(settings):
    task_id = "T-citation-malformed"
    _world(settings, task_id)
    pool = _pool(settings, task_id)
    pool["occurrences"] = [None]
    with pytest.raises(EvaluationError, match="occurrence.*malformed"):
        _citation_supports_for(settings, task_id, pool, {}, lambda *_args: "entail")


def test_citation_measurement_rejects_an_occurrence_with_a_missing_snapshot(settings):
    task_id = "T-citation-missing-snapshot"
    _world(settings, task_id)
    pool = _pool(settings, task_id)
    pool["snapshots"] = {"other": next(iter(pool["snapshots"].values()))}
    with pytest.raises(EvaluationError, match="references missing snapshot"):
        _citation_supports_for(settings, task_id, pool, {}, lambda *_args: "entail")


def test_citation_measurement_rejects_a_corrupt_frozen_object(settings):
    task_id = "T-citation-corrupt-object"
    _world(settings, task_id)
    pool = _pool(settings, task_id)
    ref = str(pool["snapshots"]["h1"]["object_ref"])
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    objects._path_for(ref).write_bytes(b"not a zstd object")
    with pytest.raises(EvaluationError, match="snapshot.*unreadable"):
        _citation_supports_for(settings, task_id, pool, {}, lambda *_args: "entail")


def test_citation_in_pool_but_not_retrieved_by_this_arm_is_not_credited(settings):
    task_id = "T-citation-arm-lineage"
    _world(settings, task_id)
    resolver = _citation_supports_for(
        settings, task_id, _pool(settings, task_id), {},
        lambda *_args: "entail",
    )
    resolver.bind(_REPORT, {"claim-1": "41% efficiency"}, allowed_occurrence_ids=())
    assert resolver("claim-1", "[1]") is False

    resolver.bind(
        _REPORT, {"claim-1": "41% efficiency"}, allowed_occurrence_ids=("o1",))
    assert resolver("claim-1", "[1]") is True


def _committed_cell(settings, ledger: Ledger, task_id: str, arm_id: str,
                    *, run_id: str = "run-eval", block_id: str = "B1",
                    tamper_event_hash: bool = False,
                    omit_search_lineage: bool = False) -> str:
    store = ObjectStore(settings.path("object_store"))
    page_variant, close_variant = (
        ("P0", "P0") if arm_id == "P0" else (arm_id, "P0"))
    arm = {
        "arm_id": arm_id,
        "page_variant": page_variant,
        "close_variant": close_variant,
    }
    event = {
        "event_index": 0,
        "kind": "SEARCH_QUERY",
        "position": "PRE_TREATMENT",
        "query": "reactor efficiency",
        "result_count": 1,
        "source_occurrence_ids": ["o1"],
    }
    if omit_search_lineage:
        event.pop("source_occurrence_ids")
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    event["event_sha256"] = sha256_hex(canonical_json(event))
    if tamper_event_hash:
        event["result_count"] = 999
    record = {
        "cell": {
            "block_id": block_id, "arm": arm,
            "replicate_id": "0", "task_id": task_id, "seed": 1,
            "order_index": 0,
        },
        "run_id": run_id,
        "phase_id": "run-screen",
        "execution_binding_sha256": EXECUTION_BINDING_SHA256,
        "protocol_document_sha256": PROTOCOL_DOCUMENT_SHA256,
        "variant_id": f"{page_variant}+{close_variant}",
        "final_report": _REPORT,
        "checkpoints": [],
        "events": [event],
    }

    ref = store.put_bytes(canonical_json(record))
    key = ledger.ensure_work_item(
        protocol_sha=EXECUTION_BINDING_SHA256,
        split="FORMATIVE_SCREEN", phase_id="run-screen",
        task_id=task_id, arm_id=arm_id,
        variant_id=f"{page_variant}+{close_variant}",
        replicate_id="0", checkpoint_hash=block_id, stage_version="v1",
    )
    attempt = ledger.claim(key, "w", lease_seconds=60, run_id=run_id)
    ledger.advance(attempt.attempt_id, "MATERIALIZED")
    ledger.advance(attempt.attempt_id, "VALIDATED")
    ledger.commit(attempt.attempt_id, result_object_ref=ref.key)
    return ref.key


def _frozen_block(tmp_path: Path, *, task_id: str, output_ref: str,
                  arm_id: str = "P0", block_id: str = "B1",
                  run_id: str = "run-eval") -> Path:
    from shapeflow_p1.campaign.schedule import (
        ArmSpec,
        Block,
        Cell,
        ScheduleManifest,
        freeze_record,
        freeze_root_record,
    )
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    directory = tmp_path / "frozen-blocks"
    directory.mkdir(exist_ok=True)
    arm = ArmSpec(
        arm_id,
        "P0" if arm_id == "P0" else arm_id,
        "P0",
    )
    cell = Cell(
        block_id=block_id, task_id=task_id, arm=arm, seed=1,
        replicate_id="0", order_index=0,
    )
    block = Block(
        block_id=block_id, task_id=task_id, replicate_id="0", cells=(cell,))
    manifest = ScheduleManifest(
        execution_binding_sha256=EXECUTION_BINDING_SHA256,
        protocol_sha=PROTOCOL_DOCUMENT_SHA256,
        split="FORMATIVE_SCREEN", blocks=(block,), arms=(arm,),
        seeds=(1,), layer="causal", claim_scope="FORMATIVE_ONLY",
        notes={
            "tasks": 1,
            "blocks": 1,
            "cells": 1,
            "analysis_design_receipt_sha256": "c" * 64,
            "task_feature_registry_sha256": "d" * 64,
            "eligibility_spec_content_sha256": "e" * 64,
        },
    )
    key = f"{block_id}:{arm_id}:0"
    body = freeze_record(
        block, states={key: "COMMITTED"}, outputs={key: output_ref})
    body["cells"][0]["engine_epoch"] = "epoch-test"
    body["engine_epochs"] = ["epoch-test"]
    body["valid_for_paired_estimate"] = True
    body["execution_binding_sha256"] = EXECUTION_BINDING_SHA256
    body["protocol_document_sha256"] = PROTOCOL_DOCUMENT_SHA256
    body["freeze_sha256"] = sha256_hex(canonical_json({
        key: value for key, value in body.items() if key != "freeze_sha256"
    }))
    (directory / f"{block_id}.json").write_text(json.dumps(body), encoding="utf-8")
    root = freeze_root_record(
        manifest, run_id=run_id, phase_id="run-screen",
        split="FORMATIVE_SCREEN", block_records=[body])
    (directory / "FREEZE_ROOT.json").write_text(json.dumps(root), encoding="utf-8")
    return directory


class _NoClient:
    """The judge is monkeypatched, so this only has to exist to be called."""

    def deepseek_transport(self, **_kw):
        async def transport(_body):  # pragma: no cover - never dispatched
            raise AssertionError("the scripted judge should have intercepted this")

        return transport


class _ScriptedJudge:
    """Answers `entail` for anything mentioning 41%, `unrelated` otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, system, user, *, validate=None):
        from shapeflow_p1.evaluation.judge_client import JudgeResponse

        self.calls += 1
        relation = "entail" if "41" in user else "unrelated"
        return JudgeResponse(
            data={"relation": relation}, requested_model="deepseek-v4-flash",
            returned_model="deepseek-v4-flash", usage={}, request_id=f"r{self.calls}",
            system_fingerprint="fp_test",
        )


def test_evaluator_ledger_view_is_query_only(tmp_path):
    path = tmp_path / "runner-ledger.sqlite"
    Ledger(str(path)).close()

    view = _ReadOnlyLedgerView(path)
    try:
        assert view.raw_connection.execute(
            "SELECT COUNT(*) FROM work_items").fetchone()[0] == 0
        with pytest.raises(sqlite3.OperationalError, match="readonly"):
            view.raw_connection.execute("CREATE TABLE evaluator_must_not_write (id INTEGER)")
    finally:
        view.close()


@pytest.mark.parametrize(
    ("run_id", "phase_id"),
    [
        ("../../outside", "phase"),
        ("/outside-judgments", "phase"),
        ("run", "../../outside"),
        ("run", "/outside-judgments"),
    ],
)
async def test_evaluate_rejects_scope_escape_before_opening_runtime_state(
    settings, tmp_path, run_id, phase_id,
):
    with pytest.raises(ValueError, match="one safe path component"):
        await _evaluate(
            settings,
            repo=REPO,
            run_id=run_id,
            phase_id=phase_id,
            frozen_blocks_dir=tmp_path,
        )


async def test_evaluate_scores_a_frozen_task_end_to_end(settings, tmp_path, monkeypatch):
    task_id = "T-eval-1"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    ref = _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()
    frozen = _frozen_block(tmp_path, task_id=task_id, output_ref=ref)

    judge = _ScriptedJudge()
    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: judge)
    monkeypatch.setattr(evaluate_module, "provider_client_for",
                        lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    out = await _evaluate(
        settings, repo=REPO, run_id="run-eval", phase_id="run-screen",
        frozen_blocks_dir=frozen)

    assert out["scored"] == ["B1"], out
    assert judge.calls > 0, "no claim was ever judged"

    body = json.loads(
        (settings.path("judgments") / "run-eval" / "run-screen" / "B1.json")
        .read_text(encoding="utf-8"))
    arm = body["per_arm"]["P0:0"]
    # The number that was structurally 0.0 for every arm of every task.
    assert arm["weighted_required_atom_recall"] > 0.0, arm
    assert arm["citation_correctness"] > 0.0, arm
    assert arm["trajectory_metrics"]["status"] == "OK"
    assert arm["trajectory_metrics"]["query_count"] == 1.0
    assert arm["trajectory_metrics"]["unique_source_occurrence_count"] == 1.0
    assert len(arm["trajectory_metrics"]["trajectory_sha256"]) == 64
    receipt = json.loads(
        (settings.path("judgments") / "run-eval" / "run-screen"
         / "EVALUATION_SCOPE.json").read_text(encoding="utf-8"))
    assert receipt["all_offered_blocks"] == 1
    assert receipt["scores"][0]["block_id"] == "B1"
    assert len(receipt["freeze_root_sha256"]) == 64
    assert len(receipt["evaluation_scope_sha256"]) == 64
    from shapeflow_p1.analysis.estimands import load_scoped_scores

    scoped = load_scoped_scores(
        settings.path("judgments"), run_id="run-eval", phase_id="run-screen")
    assert [record["block_id"] for record in scoped] == ["B1"]


@pytest.mark.parametrize(
    ("cell_options", "message"),
    [
        ({"tamper_event_hash": True}, "does not verify"),
        ({"omit_search_lineage": True}, "source occurrence lineage"),
    ],
)
async def test_evaluate_structurally_rejects_invalid_frozen_trajectory(
    settings, tmp_path, monkeypatch, cell_options, message,
):
    task_id = "T-eval-invalid-trajectory"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    ref = _committed_cell(
        settings, ledger, task_id, "P0", **cell_options)
    ledger.close()
    frozen = _frozen_block(tmp_path, task_id=task_id, output_ref=ref)

    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module, "DeepSeekJudge", lambda *a, **kw: _ScriptedJudge())
    monkeypatch.setattr(
        evaluate_module, "provider_client_for", lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    with pytest.raises(EvaluationError, match=message):
        await _evaluate(
            settings, repo=REPO, run_id="run-eval", phase_id="run-screen",
            frozen_blocks_dir=frozen)
    assert not (
        settings.path("judgments") / "run-eval" / "run-screen" / "B1.json"
    ).exists()


async def test_corrupt_frozen_cell_output_aborts_measurement_instead_of_harming_arm(
    settings, tmp_path, monkeypatch
):
    """Storage damage is not a treatment outcome and must never enter the ITT score."""
    task_id = "T-eval-corrupt-cell"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    ref = _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()
    frozen = _frozen_block(tmp_path, task_id=task_id, output_ref=ref)

    # The root and ledger still point at this exact content address, but the stored bytes no
    # longer verify.  Treating the resulting read failure as an empty report would manufacture
    # a bad P0 outcome and bias the comparison toward P1.
    ObjectStore(settings.path("object_store"))._path_for(ref).write_bytes(
        b"corrupt frozen result"
    )

    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(
        evaluate_module, "provider_client_for", lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    with pytest.raises(
        EvaluationError, match="artifact loss is an evaluation structural failure"
    ):
        await _evaluate(
            settings, repo=REPO, run_id="run-eval", phase_id="run-screen",
            frozen_blocks_dir=frozen)

    assert not (
        settings.path("judgments") / "run-eval" / "run-screen" / "B1.json"
    ).exists()


async def test_the_judge_that_actually_answered_is_recorded(settings, tmp_path, monkeypatch):
    task_id = "T-eval-2"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    ref = _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()
    frozen = _frozen_block(tmp_path, task_id=task_id, output_ref=ref)

    judge = _ScriptedJudge()
    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: judge)
    monkeypatch.setattr(evaluate_module, "provider_client_for",
                        lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    await _evaluate(
        settings, repo=REPO, run_id="run-eval", phase_id="run-screen",
        frozen_blocks_dir=frozen)

    provenance = json.loads(
        (settings.path("judgments") / "run-eval" / "run-screen" / "B1.judge.json")
        .read_text(encoding="utf-8"))
    assert provenance["returned_models"] == ["deepseek-v4-flash"]
    assert provenance["system_fingerprints"] == ["fp_test"]
    assert provenance["judgments"] > 0
    assert len(provenance["relation_prompt_sha256"]) == 64
    # Every citation resolution is explicable rather than an unexplained zero.
    assert provenance["citation_resolutions"], provenance


async def test_a_task_whose_block_was_never_frozen_is_rejected(settings, tmp_path,
                                                                monkeypatch):
    """Half a paired comparison is not an observation (plan §16.3)."""
    task_id = "T-eval-3"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()

    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: _ScriptedJudge())
    monkeypatch.setattr(evaluate_module, "provider_client_for",
                        lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    empty = tmp_path / "frozen-blocks"
    empty.mkdir()
    with pytest.raises(ValueError, match="campaign root is missing"):
        await _evaluate(
            settings, repo=REPO, run_id="run-eval", phase_id="run-screen",
            frozen_blocks_dir=empty)


async def test_evaluate_cannot_mix_another_run_or_unfrozen_block(settings, tmp_path,
                                                                 monkeypatch):
    task_id = "T-eval-scope"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    wanted_ref = _committed_cell(
        settings, ledger, task_id, "P0", run_id="wanted", block_id="B-wanted")
    _committed_cell(
        settings, ledger, task_id, "H02", run_id="other", block_id="B-other")
    ledger.close()
    frozen = _frozen_block(
        tmp_path, task_id=task_id, output_ref=wanted_ref,
        arm_id="P0", block_id="B-wanted", run_id="wanted")

    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: _ScriptedJudge())
    monkeypatch.setattr(evaluate_module, "provider_client_for", lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    out = await _evaluate(
        settings, repo=REPO, run_id="wanted", phase_id="run-screen",
        frozen_blocks_dir=frozen)
    assert out["scored"] == ["B-wanted"]
    body = json.loads(
        (settings.path("judgments") / "wanted" / "run-screen" / "B-wanted.json")
        .read_text(encoding="utf-8"))
    assert set(body["per_arm"]) == {"P0:0"}


@pytest.mark.parametrize(
    "damage",
    ["missing_block", "extra_file", "tampered_block", "duplicate_root_block",
     "nonterminal_root"],
)
def test_frozen_campaign_root_rejects_incomplete_or_tampered_directories(
    tmp_path, damage
):
    directory = _frozen_block(
        tmp_path, task_id="T-integrity", output_ref="a" * 64)
    block_path = directory / "B1.json"
    root_path = directory / "FREEZE_ROOT.json"
    if damage == "missing_block":
        block_path.unlink()
    elif damage == "extra_file":
        (directory / "not-in-root.txt").write_text("extra", encoding="utf-8")
    elif damage == "tampered_block":
        body = json.loads(block_path.read_text(encoding="utf-8"))
        body["cells"][0]["state"] = "FAILED_FINAL"
        block_path.write_text(json.dumps(body), encoding="utf-8")
    else:
        from shapeflow_p1.canonical import canonical_json
        from shapeflow_p1.hashing import sha256_hex

        root = json.loads(root_path.read_text(encoding="utf-8"))
        if damage == "duplicate_root_block":
            root["blocks"].append(dict(root["blocks"][0]))
        else:
            root["terminal_frozen"] = False
        root["freeze_root_sha256"] = sha256_hex(canonical_json({
            key: value for key, value in root.items()
            if key != "freeze_root_sha256"
        }))
        root_path.write_text(json.dumps(root), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_frozen_scope(
            directory,
            run_id="run-eval",
            phase_id="run-screen",
            execution_binding_sha256=EXECUTION_BINDING_SHA256,
            protocol_document_sha256=PROTOCOL_DOCUMENT_SHA256,
        )


def test_frozen_campaign_root_cannot_be_evaluated_under_a_new_approval(tmp_path):
    directory = _frozen_block(
        tmp_path, task_id="T-binding", output_ref="a" * 64)
    with pytest.raises(ValueError, match="belongs to execution binding"):
        _load_frozen_scope(
            directory,
            run_id="run-eval",
            phase_id="run-screen",
            execution_binding_sha256="f" * 64,
            protocol_document_sha256=PROTOCOL_DOCUMENT_SHA256,
        )
