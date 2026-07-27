"""Human audit queue selection, the upgrade gate, and truth-packet assembly."""

from __future__ import annotations

import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from shapeflow_p1.campaign.truth import _excerpt_batches, _write_truth_artifact
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.evaluation.human_audit import (
    audit_gate_ready,
    select_audit_queue,
)
from shapeflow_p1.evaluation.truth_builder import CandidateAtom, assemble_truth_packet
from shapeflow_p1.hashing import sha256_hex

SCHEMA = json.loads(
    (Path(__file__).resolve().parents[2] / "schemas" / "truth_packet.schema.json").read_text()
)


# --- audit queue ----------------------------------------------------------------------


def test_queue_includes_sample_and_all_mandatory():
    tasks = {"lowvol": [f"l{i}" for i in range(10)], "highvol": [f"h{i}" for i in range(10)]}
    q = select_audit_queue(
        tasks_by_stratum=tasks, sample_fraction=0.2,
        critical_misses=["l0", "special_crit"], disagreements=["h3"], near_margin=["l9"],
        seed=1,
    )
    kinds = {(i.task_id, i.kind) for i in q}
    # every mandatory case present regardless of the sample
    assert ("special_crit", "CRITICAL_MISS") in kinds
    assert ("h3", "DISAGREEMENT") in kinds
    assert ("l9", "NEAR_MARGIN") in kinds
    # sample drawn from both strata
    sampled = [i for i in q if i.kind == "RANDOM_SAMPLE"]
    assert any(i.detail == "lowvol" for i in sampled)
    assert any(i.detail == "highvol" for i in sampled)


def test_queue_is_deterministic():
    tasks = {"s": [f"t{i}" for i in range(20)]}
    a = select_audit_queue(tasks_by_stratum=tasks, sample_fraction=0.25, seed=7)
    b = select_audit_queue(tasks_by_stratum=tasks, sample_fraction=0.25, seed=7)
    assert [i.task_id for i in a] == [i.task_id for i in b]


# --- upgrade gate ---------------------------------------------------------------------


def test_gate_ready_when_all_conditions_met():
    gate = audit_gate_ready(
        sampled_fraction=0.20, required_fraction=0.15, critical_errors=0,
        noncritical_accuracy=0.97, noncritical_accuracy_min=0.95, critical_harm_misses=0,
    )
    assert gate.ready and not gate.reasons


def test_gate_blocked_by_a_single_critical_error():
    gate = audit_gate_ready(
        sampled_fraction=0.20, required_fraction=0.15, critical_errors=1,
        noncritical_accuracy=0.99, noncritical_accuracy_min=0.95, critical_harm_misses=0,
    )
    assert not gate.ready
    assert any("critical audited error" in r for r in gate.reasons)


def test_gate_blocked_by_insufficient_sample():
    gate = audit_gate_ready(
        sampled_fraction=0.05, required_fraction=0.15, critical_errors=0,
        noncritical_accuracy=0.99, noncritical_accuracy_min=0.95, critical_harm_misses=0,
    )
    assert not gate.ready


# --- truth assembly -------------------------------------------------------------------


def _truth_artifact(marker: str) -> dict:
    body = {"schema_version": "truth-writer-race-test-v1", "marker": marker}
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def test_truth_artifact_creation_is_exclusive_under_concurrent_writers(tmp_path):
    path = tmp_path / "truth.json"
    barrier = threading.Barrier(2)

    def write(marker: str):
        barrier.wait()
        try:
            _write_truth_artifact(path, _truth_artifact(marker))
            return "WROTE"
        except ValueError:
            return "REFUSED"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(write, ("first", "second")))

    assert sorted(outcomes) == ["REFUSED", "WROTE"]
    assert json.loads(path.read_text(encoding="utf-8"))["marker"] in {"first", "second"}
    assert path.stat().st_mode & 0o222 == 0


def test_truth_artifact_existing_branch_rehashes_before_idempotent_return(tmp_path):
    path = tmp_path / "truth.json"
    artifact = _truth_artifact("original")
    _write_truth_artifact(path, artifact)
    _write_truth_artifact(path, artifact)

    os.chmod(path, 0o640)
    tampered = dict(artifact)
    tampered["marker"] = "edited-with-stale-hash"
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="was edited"):
        _write_truth_artifact(path, artifact)


def test_spanless_atom_is_rejected():
    cands = [
        CandidateAtom("a1", "f1", 1.0, False, supporting_span_ids=("s" * 1,)),
        CandidateAtom("a2", "f1", 1.0, True, supporting_span_ids=()),  # no span -> rejected
    ]
    packet, rejected = assemble_truth_packet("t1", cands, required_facets=["f1"])
    assert [a["atom_id"] for a in packet["atomic_evidence"]] == ["a1"]
    assert rejected and rejected[0].atom_id == "a2"


def test_contradiction_survives_only_if_both_sides_accepted():
    cands = [
        CandidateAtom("a", "f", 1.0, False, ("x",)),
        CandidateAtom("b", "f", 1.0, False, ()),  # dropped
    ]
    packet, _ = assemble_truth_packet("t", cands, required_facets=["f"],
                                     contradiction_pairs=[("a", "b")])
    assert packet["contradiction_pairs"] == []  # half-grounded pair not presented as truth


def test_assembled_packet_validates_against_schema_and_is_provisional():
    cands = [CandidateAtom("a1", "f1", 2.0, True, ("a" * 64,))]  # span id must be 64-hex
    packet, _ = assemble_truth_packet("t1", cands, required_facets=["f1"])
    Draft202012Validator(SCHEMA).validate(packet)
    assert packet["authoring_method"] == "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT"
    assert packet["verifier_status"] == "PENDING"
    assert packet["critical_items"] == ["a1"]


def test_exact_span_binding_requires_semantic_entailment():
    spans = {
        "a" * 64: "The filing count was 17 in 2025.",
        "b" * 64: "This paragraph discusses an unrelated city.",
    }
    candidate = CandidateAtom(
        "fact", "f", 1.0, True, tuple(spans),
        text="The filing count was 99 in 2025.",
    )
    packet, rejected = assemble_truth_packet(
        "t", [candidate], required_facets=["f"], span_texts=spans,
        semantic_verifier=lambda atom, span, _sid: "99" in span,
    )
    assert packet["atomic_evidence"] == []
    assert rejected[0].reason == "no semantically verified supporting span"


def test_negative_evidence_and_gaps_bind_to_real_query_attempts():
    candidate = CandidateAtom(
        "a", "f", 1.0, False, ("a" * 64,), text="A supported fact.")
    packet, _ = assemble_truth_packet(
        "t", [candidate], required_facets=["f"],
        negative_evidence=[
            {"atom_id": "a", "facet_id": "f", "query_attempt_id": "q-real"},
            {"atom_id": "a", "facet_id": "f", "query_attempt_id": "q-invented"},
            {"atom_id": "made-up", "facet_id": "f", "query_attempt_id": "q-real"},
        ],
        known_gaps=[
            {
                "query_attempt_id": "q-real",
                "query_text": "missing fact",
                "status": "TIMEOUT",
            },
            {
                "query_attempt_id": "q-invented",
                "query_text": "invented",
                "status": "FAILED",
            },
            {
                "query_attempt_id": "q-real-success",
                "query_text": "successful query",
                "status": "SUCCESS",
            },
        ],
        known_query_attempt_ids={"q-real"},
    )
    assert packet["negative_evidence"] == [
        {"atom_id": "a", "facet_id": "f", "query_attempt_id": "q-real"}]
    assert packet["known_gaps"] == [{
        "query_attempt_id": "q-real",
        "query_text": "missing fact",
        "status": "TIMEOUT",
    }]


def test_empty_query_registry_does_not_authorize_model_proposed_negative_or_gap_ids():
    candidate = CandidateAtom(
        "a", "f", 1.0, False, ("a" * 64,), text="No filing exists.")
    packet, _ = assemble_truth_packet(
        "t", [candidate], required_facets=["f"],
        negative_evidence=[
            {"atom_id": "a", "facet_id": "f", "query_attempt_id": "invented"},
        ],
        known_gaps=[{
            "query_attempt_id": "invented",
            "query_text": "made up query",
            "status": "TIMEOUT",
        }],
        known_query_attempt_ids=set(),
    )
    assert packet["negative_evidence"] == []
    assert packet["known_gaps"] == []


def test_truth_prompt_batching_keeps_every_complete_span():
    spans = [
        {"span_id": f"s{i}", "text": ("x" * 90) + str(i)}
        for i in range(75)
    ]
    batches = _excerpt_batches(spans, max_prompt_chars=500)
    flattened = [span for batch in batches for span in batch]
    assert flattened == spans
    assert all(span["text"].endswith(str(i)) for i, span in enumerate(flattened))


async def test_truth_builds_semantic_support_ids_for_each_h_chunker(monkeypatch):
    import shapeflow_p1.campaign.truth as truth_module

    monkeypatch.setattr(truth_module, "_h_candidate_spans", lambda _s, _t, **_kw: {
        "fixed_token_v1": [{
            "span_id": "fixed", "source_occurrence_ids": ["o1"],
            "char_start": 0, "char_end": 40, "_text": "The filing count was 17.",
        }],
        "paragraph_sentence_v1": [{
            "span_id": "paragraph", "source_occurrence_ids": ["o1"],
            "char_start": 0, "char_end": 40, "_text": "Unrelated material.",
        }],
    })
    candidate = CandidateAtom(
        "a", "f", 1.0, False, ("truth-span",),
        text="The filing count was 17.",
    )
    class _Settings:
        def get(self, *_args):
            return 1000

    index = await truth_module._atom_support_index(
        _Settings(), "t", candidates=[candidate],
        truth_spans=[{
            "span_id": "truth-span", "source_occurrence_ids": ["o1"],
            "char_start": 0, "char_end": 40,
        }],
        verifier=lambda atom, span, _sid: "17" in atom and "17" in span,
    )
    assert index["chunkers"]["fixed_token_v1"]["a"] == ["fixed"]
    assert index["chunkers"]["paragraph_sentence_v1"]["a"] == []


async def test_truth_artifact_records_extraction_exact_and_cross_chunker_judgments(
    tmp_path, monkeypatch,
):
    import shapeflow_p1.campaign.truth as truth_module
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.bench.grading.judge_client import JudgeAttempt, JudgeResponse
    from shapeflow_p1.hashing import sha256_hex

    span_id = "a" * 64
    attempts = [
        {"query_attempt_id": "q-success", "query": "filing search", "status": "SUCCESS"},
        {"query_attempt_id": "q-timeout", "query": "permit database", "status": "TIMEOUT"},
    ]
    monkeypatch.setattr(truth_module, "_excerpt_spans", lambda _settings, _task: [{
        "span_id": span_id,
        "text": "No public filing exists for Project Zephyr in 2025.",
        "source_occurrence_ids": ["o1"],
        "char_start": 0,
        "char_end": 52,
    }])
    monkeypatch.setattr(
        truth_module, "_query_attempts", lambda _settings, _task: attempts)

    async def support_index(_settings, _task_id, *, candidates, truth_spans, verifier):
        # Exercise the same judge-backed verifier a second time from the cross-chunker binding
        # stage, so provenance must distinguish it from exact truth-span binding.
        assert await truth_module._verified_relation(
            verifier, candidates[0].text,
            "No public filing exists for Project Zephyr in 2025.",
            "candidate-span",
        )
        body = {
            "version": "h_atom_support_index_v2",
            "tokenizer_sha256": "1" * 64,
            "prechunk_atom_occurrence_ids": {candidates[0].atom_id: ["o1"]},
            "candidate_span_occurrence_ids": {
                "markdown_structure_v1": {"candidate-span": ["o1"]},
            },
            "chunkers": {
                "markdown_structure_v1": {
                    candidates[0].atom_id: ["candidate-span"],
                },
            },
        }
        body["content_sha256"] = sha256_hex(canonical_json(body))
        return body

    monkeypatch.setattr(truth_module, "_atom_support_index", support_index)

    class ScriptedJudge:
        def __init__(self):
            self.calls = 0

        async def judge(self, system, user, *, validate=None):
            self.calls += 1
            is_binding = "verify one proposed atomic fact" in system
            data = (
                {"relation": "entail"}
                if is_binding else
                {
                    "atoms": [{
                        "atom_id": "absence",
                        "facet_id": "f1",
                        "text": "No public filing exists for Project Zephyr in 2025.",
                        "critical": False,
                        "supporting_span_ids": [span_id],
                    }],
                    "contradictions": [],
                    "negative_evidence": [{
                        "atom_id": "absence",
                        "facet_id": "f1",
                        "query_attempt_id": "q-success",
                    }],
                    # This suggestion must be ignored; q-success is not an operational gap.
                    "known_gaps": ["q-success"],
                }
            )
            if validate is not None:
                validate(data)
            attempt = JudgeAttempt(
                ordinal=0,
                prompt_sha256=f"{self.calls:064x}",
                sampling={"temperature": 0.0},
                status=200,
                outcome="accepted",
                returned_model=f"judge-returned-{self.calls}",
                system_fingerprint=f"fp-{self.calls}",
                request_id=f"attempt-{self.calls}",
                usage={"total_tokens": 10 + self.calls},
            )
            return JudgeResponse(
                data=data,
                requested_model="judge-requested",
                returned_model=f"judge-returned-{self.calls}",
                usage={"total_tokens": 10 + self.calls},
                request_id=f"response-{self.calls}",
                system_fingerprint=f"fp-{self.calls}",
                attempts=(attempt,),
            )

    class FakeSettings:
        shas = {"judge": "j" * 64}
        claim_scope = "FORMATIVE_ONLY"

        def path(self, name):
            return {
                "truth_packets": tmp_path / "truth",
                "frozen_corpus_for_runner": tmp_path / "frozen",
                "acquisition": tmp_path / "acquisition",
            }[name]

    settings = FakeSettings()
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    pool_dir.mkdir(parents=True)
    (pool_dir / "T.json").write_text(
        json.dumps({"pool_sha256": "p" * 64}), encoding="utf-8")
    acquisition_body = {
        "task_id": "T",
        "acquisition_spec_sha256": "s" * 64,
        "fetched_at_utc": "2026-07-25T00:00:00Z",
        "tavily_params": {},
        "queries": [],
        "occurrences": [],
        "snapshots": [],
    }
    acquisition_digest = sha256_hex(canonical_json(acquisition_body))
    acquisition_body.update({
        "merkle_root": "m" * 64,
        "acquisition_digest": acquisition_digest,
    })
    acquisition_path = settings.path("acquisition") / "T.json"
    acquisition_path.parent.mkdir(parents=True)
    acquisition_path.write_text(json.dumps(acquisition_body), encoding="utf-8")

    result = await truth_module.build_truth_for_task(
        settings,
        judge=ScriptedJudge(),
        task_id="T",
        question="Are Project Zephyr filings public?",
        required_facets=["f1"],
    )
    artifact = json.loads(result.path.read_text(encoding="utf-8"))
    provenance = artifact["provenance"]
    assert provenance["judge_calls_by_stage"] == {
        "truth_extraction": 1,
        "conflict_reconciliation": 0,
        "exact_span_binding": 1,
        "cross_chunker_binding": 1,
    }
    assert [entry["stage"] for entry in provenance["judge_attempts"]] == [
        "truth_extraction",
        "exact_span_binding",
        "cross_chunker_binding",
    ]
    assert all(entry["attempts"] for entry in provenance["judge_attempts"])
    assert all(entry["usage"] for entry in provenance["judge_attempts"])
    assert provenance["source_pool_sha256"] == "p" * 64
    assert provenance["acquisition_digest"] == acquisition_digest
    assert len(provenance["task_question_facets_sha256"]) == 64
    assert len(provenance["query_attempts_sha256"]) == 64
    assert len(provenance["truth_source_binding_sha256"]) == 64
    assert provenance["model_gap_suggestions_ignored"] == 1
    assert artifact["packet"]["known_gaps"] == [{
        "query_attempt_id": "q-timeout",
        "query_text": "permit database",
        "status": "TIMEOUT",
    }]


async def test_a_batch_whose_answer_overflows_the_cap_is_split_not_abandoned(
    tmp_path, monkeypatch,
):
    """A truncated answer means the batch was too big, so retrying it identically cannot help.

    Batching is sized by prompt characters, which bounds the input and says nothing about the
    output: a span-dense batch can sit well inside the prompt budget and still ask for more
    atoms than the completion cap holds. On the run host that abandoned a whole task after
    paying for four identical attempts, each of which truncated for the same reason.
    """
    import shapeflow_p1.campaign.truth as truth_module
    import json

    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.bench.grading.judge_client import (
        JudgeAttempt,
        JudgeResponse,
        JudgeTruncated,
    )
    from shapeflow_p1.hashing import sha256_hex

    spans = [
        {"span_id": f"{i:064x}", "text": f"Fact number {i} is established.",
         "source_occurrence_ids": ["o1"], "char_start": 0, "char_end": 30}
        for i in range(4)
    ]
    monkeypatch.setattr(truth_module, "_excerpt_spans", lambda _s, _t: spans)
    monkeypatch.setattr(truth_module, "_query_attempts", lambda _s, _t: [])

    class OverflowingJudge:
        """Truncates on any batch larger than two spans; answers otherwise."""

        def __init__(self):
            self.batch_sizes: list[int] = []

        async def judge(self, system, user, *, validate=None):
            if "verify one proposed atomic fact" in system:
                return JudgeResponse(
                    data={"relation": "entail"}, requested_model="m", returned_model="m",
                    usage={}, request_id="r", system_fingerprint="f",
                    attempts=(JudgeAttempt(0, "p", {}, 200, "accepted"),))
            if "identify genuine contradictions" in system:
                return JudgeResponse(
                    data={"contradictions": []}, requested_model="m", returned_model="m",
                    usage={}, request_id="r", system_fingerprint="f",
                    attempts=(JudgeAttempt(0, "p", {}, 200, "accepted"),))
            size = sum(1 for line in user.splitlines() if line.startswith("0"))
            self.batch_sizes.append(size)
            if size > 2:
                raise JudgeTruncated("every one of 4 attempts hit the 8000 token output cap")
            return JudgeResponse(
                data={"atoms": [{
                    "atom_id": f"a{size}", "facet_id": "f",
                    "text": "Fact number 0 is established.", "critical": False,
                    "supporting_span_ids": [spans[0]["span_id"]],
                }], "contradictions": [], "negative_evidence": [], "known_gaps": []},
                requested_model="m", returned_model="m", usage={}, request_id="r",
                system_fingerprint="f",
                attempts=(JudgeAttempt(0, "p", {}, 200, "accepted"),))

    class FakeSettings:
        shas = {"judge": "j" * 64}
        claim_scope = "FORMATIVE_ONLY"
        repo = Path(__file__).resolve().parents[2]

        def path(self, name):
            return {
                "truth_packets": tmp_path / "truth",
                "frozen_corpus_for_runner": tmp_path / "frozen",
                "acquisition": tmp_path / "acquisition",
            }[name]

        def get(self, *keys):
            return {("week1", "odr", "max_content_length"): 50_000}[keys]

    settings = FakeSettings()
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    pool_dir.mkdir(parents=True)
    # The pool digest is re-derived on load, so it has to be the real hash of the body.
    pool_body = {"occurrences": [], "snapshots": {}}
    pool_body["pool_sha256"] = sha256_hex(canonical_json(pool_body))
    (pool_dir / "T1.json").write_text(json.dumps(pool_body), encoding="utf-8")
    acquisition_body = {
        "task_id": "T1", "acquisition_spec_sha256": "s" * 64,
        "fetched_at_utc": "2026-07-25T00:00:00Z", "tavily_params": {},
        "queries": [], "occurrences": [], "snapshots": [],
    }
    acquisition_body["acquisition_digest"] = sha256_hex(canonical_json(acquisition_body))
    acquisition_body["merkle_root"] = "m" * 64
    acquisition_path = settings.path("acquisition") / "T1.json"
    acquisition_path.parent.mkdir(parents=True)
    acquisition_path.write_text(json.dumps(acquisition_body), encoding="utf-8")

    judge = OverflowingJudge()
    result = await truth_module.build_truth_for_task(
        settings, judge=judge, task_id="T1",
        question="How many facts are established?", required_facets=["f"],
        semantic_verifier=None, max_prompt_chars=100_000,
    )

    assert judge.batch_sizes[0] == 4, "the first attempt should use the packed batch"
    assert max(judge.batch_sizes[1:]) <= 2, "the oversized batch was not split"
    artifact = json.loads(result.path.read_text(encoding="utf-8"))
    assert artifact["provenance"]["span_batches_split_for_output_cap"] >= 1
    # And the packet is complete: splitting must not drop the spans it re-batched.
    assert artifact["provenance"]["span_batches"] > 1


def test_build_truth_skips_a_task_whose_packet_is_already_frozen(tmp_path, monkeypatch):
    """An interrupted truth build has to be resumable.

    A packet is write-once and the build is not reproducible run to run -- batching adapts to
    what truncates, a retry advances the judge's seed, and only byte-identical request bodies
    replay from the ledger. So re-deriving an existing packet yields a *different* artifact and
    the write-once guard refuses it, correctly. Before this, that meant every resumed run died
    on the first task that had already succeeded, leaving the rest permanently unbuilt.
    """
    import shapeflow_p1.cli as cli_module

    packets = tmp_path / "truth"
    evaluator_tasks = tmp_path / "evaluator" / "tasks"
    packets.mkdir(parents=True)
    evaluator_tasks.mkdir(parents=True)
    for task_id in ("T-done", "T-todo"):
        (evaluator_tasks / f"{task_id}.json").write_text(json.dumps({
            "original_question": "q", "authored_facets": ["f"]}), encoding="utf-8")
    (packets / "T-done.json").write_text("{}", encoding="utf-8")

    built: list[str] = []

    class FakeSettings:
        def path(self, name):
            return {"truth_packets": packets,
                    "evaluator_root": tmp_path / "evaluator"}[name]

        def judge_model(self):
            return "deepseek-v4-flash"

        def judge_max_retries(self):
            return 1

        def judge_sampling(self):
            return None

    async def fake_build(_settings, *, judge, task_id, question, required_facets):
        built.append(task_id)

    monkeypatch.setattr(cli_module, "_require_role", lambda _r: None)
    monkeypatch.setattr(cli_module, "_settings", lambda: FakeSettings())
    monkeypatch.setattr(cli_module, "_provider_client", lambda *_a, **_k: _FakeClient())
    monkeypatch.setattr(
        "shapeflow_p1.campaign.acquire.acquired_task_ids",
        lambda _s: ["T-done", "T-todo"])
    monkeypatch.setattr("shapeflow_p1.campaign.truth.build_truth_for_task", fake_build)
    monkeypatch.setattr(
        "shapeflow_p1.bench.grading.judge_client.DeepSeekJudge",
        lambda *_a, **_k: object())

    cli_module.build_truth(config=None)

    assert built == ["T-todo"], "the already-frozen packet was rebuilt instead of skipped"


class _FakeClient:
    def deepseek_transport(self, **_kw):
        return None


@pytest.mark.asyncio
async def test_a_contradiction_pass_that_overflows_is_reblocked_without_losing_cross_pairs(
    tmp_path, monkeypatch,
):
    """The conflict pass could not split at all, and build-truth died at 43 of 48 packets.

    Copying the extraction splitter would have been wrong. An atom belongs to one span, so
    halving an extraction batch loses nothing; a contradiction belongs to a *pair*, and halving a
    facet's fact list drops every pair that straddles the cut. A packet quietly missing
    contradictions is worse than no packet: it makes an arm that missed them look correct.

    So the facet is cut into four blocks and re-asked over each pair of blocks. This test puts
    the two contradicting facts in different blocks, which is exactly the case a naive halving
    would silently lose.
    """
    import json

    import shapeflow_p1.campaign.truth as truth_module
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.bench.grading.judge_client import (
        JudgeAttempt,
        JudgeResponse,
        JudgeTruncated,
    )
    from shapeflow_p1.hashing import sha256_hex

    spans = [
        {"span_id": f"{i:064x}", "text": f"Fact number {i} is established.",
         "source_occurrence_ids": ["o1"], "char_start": 0, "char_end": 30}
        for i in range(8)
    ]
    monkeypatch.setattr(truth_module, "_excerpt_spans", lambda _s, _t: spans)
    monkeypatch.setattr(truth_module, "_query_attempts", lambda _s, _t: [])

    def _ok(data):
        return JudgeResponse(
            data=data, requested_model="m", returned_model="m", usage={}, request_id="r",
            system_fingerprint="f", attempts=(JudgeAttempt(0, "p", {}, 200, "accepted"),))

    # The two facts that contradict each other. Their atom ids are content-derived, so they are
    # computed the same way the builder computes them.
    texts = [f"Claim {i} holds." for i in range(8)]
    ids = [truth_module._stable_atom_id("", "f", text) for text in texts]
    contradicting = {ids[0], ids[1]}

    class ConflictOverflowJudge:
        """Answers extraction and binding; truncates any conflict call over four facts."""

        def __init__(self):
            self.conflict_sizes: list[int] = []

        async def judge(self, system, user, *, validate=None):
            if "verify one proposed atomic fact" in system:
                return _ok({"relation": "entail"})
            if "identify genuine contradictions" in system:
                offered = [
                    line.split(":", 1)[0].strip()
                    for line in user.splitlines() if line.startswith("atom-")
                ]
                self.conflict_sizes.append(len(offered))
                if len(offered) > 4:
                    raise JudgeTruncated(
                        "every one of 4 attempts hit the 8000 token output cap")
                # Only report the pair when both sides are in this block pair -- a judge cannot
                # name a fact it was not shown, which is the whole reason halving loses pairs.
                if contradicting <= set(offered):
                    return _ok({"contradictions": [sorted(contradicting)]})
                return _ok({"contradictions": []})
            return _ok({
                "atoms": [
                    {"atom_id": f"a{i}", "facet_id": "f", "text": texts[i],
                     "critical": False, "supporting_span_ids": [spans[i]["span_id"]]}
                    for i in range(8)
                ],
                "contradictions": [], "negative_evidence": [], "known_gaps": [],
            })

    class FakeSettings:
        shas = {"judge": "j" * 64}
        claim_scope = "FORMATIVE_ONLY"
        repo = Path(__file__).resolve().parents[2]

        def path(self, name):
            return {
                "truth_packets": tmp_path / "truth",
                "frozen_corpus_for_runner": tmp_path / "frozen",
                "acquisition": tmp_path / "acquisition",
            }[name]

        def get(self, *keys):
            return {("week1", "odr", "max_content_length"): 50_000}[keys]

    settings = FakeSettings()
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    pool_dir.mkdir(parents=True)
    pool_body = {"occurrences": [], "snapshots": {}}
    pool_body["pool_sha256"] = sha256_hex(canonical_json(pool_body))
    (pool_dir / "T1.json").write_text(json.dumps(pool_body), encoding="utf-8")
    acquisition_body = {
        "task_id": "T1", "acquisition_spec_sha256": "s" * 64,
        "fetched_at_utc": "2026-07-25T00:00:00Z", "tavily_params": {},
        "queries": [], "occurrences": [], "snapshots": [],
    }
    acquisition_body["acquisition_digest"] = sha256_hex(canonical_json(acquisition_body))
    acquisition_body["merkle_root"] = "m" * 64
    acquisition_path = settings.path("acquisition") / "T1.json"
    acquisition_path.parent.mkdir(parents=True)
    acquisition_path.write_text(json.dumps(acquisition_body), encoding="utf-8")

    judge = ConflictOverflowJudge()
    result = await truth_module.build_truth_for_task(
        settings, judge=judge, task_id="T1",
        question="Which claims hold?", required_facets=["f"],
        semantic_verifier=None, max_prompt_chars=100_000,
    )

    assert judge.conflict_sizes[0] == 8, "the first conflict call should see the whole facet"
    assert max(judge.conflict_sizes[1:]) <= 4, "the overflowing facet was not re-blocked"
    artifact = json.loads(result.path.read_text(encoding="utf-8"))
    # The pair straddles two blocks and survives; a naive halving would have dropped it.
    packet = artifact["packet"] if "packet" in artifact else artifact
    assert packet["contradiction_pairs"], "a cross-block contradiction pair was lost"
    recorded = packet["contradiction_pairs"][0]
    assert {recorded["atom_id_a"], recorded["atom_id_b"]} == contradicting
    assert artifact["provenance"]["conflict_blocks"] > artifact["provenance"]["conflict_facets"]
