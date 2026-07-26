"""The component, H-total-effect, C-only and nested HxC estimands are not interchangeable."""

from __future__ import annotations

import asyncio
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from shapeflow_p1.campaign.fork import (
    ForkCapabilityError,
    ForkExecution,
    TrialKind,
    plan_c_only_e2e,
    plan_h_e2e,
    plan_nested_hxc,
    run_production_forks,
)
from shapeflow_p1.campaign.graph_driver import TrajectoryRecorder, first_divergence
from shapeflow_p1.campaign.screen import run_screening
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.canonical import canonical_json
from shapeflow_p1.evaluation.runner import score_direct_node_records
from shapeflow_p1.hashing import sha256_hex
from shapeflow_p1.odr.checkpoints import (
    CCheckpoint,
    CheckpointStore,
    EvidenceManifest,
    FrozenMessage,
    FrozenToolCall,
    HCheckpoint,
    SamplingEnvelope,
)
from shapeflow_p1.odr.vendor_hooks import (
    RunBinding,
    _emit_selection_records,
    bind_run,
)
from shapeflow_p1.strategies.pipeline import SelectionOutcome

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


def _sampling(seed=7):
    return SamplingEnvelope(
        model="m", temperature=0.3, top_p=1.0, max_tokens=512, seed=seed)


def _h():
    return HCheckpoint(
        task_id="T1", researcher_id="R1", assistant_turn_index=0,
        assistant_message=FrozenMessage(
            role="ai", content="", tool_calls=(
                FrozenToolCall(id="tc1", name="tavily_search",
                               args_canonical='{"queries":["q"]}'),
            )),
        sibling_tool_calls=(
            FrozenToolCall(id="tc1", name="tavily_search",
                           args_canonical='{"queries":["q"]}'),
        ),
        search_result_sets=(), non_search_outputs=(),
        researcher_state_hash="state", sampling=_sampling(),
    )


def _c(researcher_id="R1"):
    return CCheckpoint(
        task_id="T1", researcher_id=researcher_id,
        researcher_messages=(FrozenMessage(role="tool", content="evidence"),),
        evidence_manifest=EvidenceManifest(span_ids=("s1",)),
        query_attempt_ids=("q1",), close_reason="RESEARCH_COMPLETE",
        sampling=_sampling(),
    )


class _Backend:
    def __init__(self, *, upstream_calls=0):
        self.upstream_calls = upstream_calls

    def supports(self, trial_kind, boundary_kind):
        return True

    async def capture_boundaries(self, **_kwargs):  # pragma: no cover - not needed here
        return ()

    async def execute(self, spec, checkpoint):
        return ForkExecution(
            output=f"{spec.variant_id}:{checkpoint.digest}",
            start_checkpoint_digest=checkpoint.digest,
            first_treatment_checkpoint_digest=checkpoint.digest,
            seed_applied=True,
            upstream_research_calls=self.upstream_calls,
            trajectory_events=(
                {"kind": "TREATMENT", "checkpoint": checkpoint.digest},
                {"kind": "SEARCH_QUERY", "query": f"after-{spec.variant_id}"},
            ),
            terminal_close_checkpoint_digest=f"close-{spec.variant_id}",
        )


def test_h_total_effect_allows_post_treatment_trajectory_divergence(settings):
    store = CheckpointStore(settings.path("checkpoints"))
    digest = store.put(_h())
    plan = plan_h_e2e(
        settings, first_h_checkpoint_digest=digest, task_id="T1",
        page_variant_ids=["H02"], seed=7,
        execution_binding_sha256="e" * 64)

    # Resuming H necessarily performs downstream researcher work. It is an outcome, not a
    # violation, so production accepts it while still requiring the common first checkpoint.
    outcomes = asyncio.run(run_production_forks(
        settings, plan, backend=_Backend(upstream_calls=2), checkpoint_store=store))
    assert {o.state for o in outcomes} == {"COMMITTED"}
    assert {o.first_treatment_checkpoint_digest for o in outcomes} == {digest}
    assert len({o.trajectory_sha256 for o in outcomes}) == 2


def test_c_only_refuses_to_rerun_researcher_work(settings):
    store = CheckpointStore(settings.path("checkpoints"))
    digest = store.put(_c())
    plan = plan_c_only_e2e(
        settings, close_checkpoint_digest=digest, task_id="T1",
        close_variant_ids=["C01"], seed=7,
        execution_binding_sha256="e" * 64)
    with pytest.raises(ForkCapabilityError, match="upstream researcher calls"):
        asyncio.run(run_production_forks(
            settings, plan, backend=_Backend(upstream_calls=1), checkpoint_store=store))


def test_nested_hxc_shares_c_within_each_natural_h_trajectory(settings):
    h_digest = CheckpointStore(settings.path("checkpoints")).put(_h())
    h_plan = plan_h_e2e(
        settings, first_h_checkpoint_digest=h_digest, task_id="T1",
        page_variant_ids=["H02"], seed=7,
        execution_binding_sha256="e" * 64)
    c0 = _c("H0")
    c1 = _c("H1")
    store = CheckpointStore(settings.path("checkpoints"))
    d0, d1 = store.put(c0), store.put(c1)
    nested = plan_nested_hxc(
        h_plan,
        close_checkpoint_by_h_variant={"P0": d0, "H02": d1},
        close_variant_ids=["C01"],
        settings=settings,
        seed=7,
        execution_binding_sha256="e" * 64,
    )
    assert d0 != d1, "H is allowed to induce a different natural close state"
    assert set(nested.c_plans_by_h_variant) == {"P0", "H02"}
    for h_variant, c_plan in nested.c_plans_by_h_variant.items():
        expected = d0 if h_variant == "P0" else d1
        assert {f.checkpoint_digest for f in c_plan.forks} == {expected}
        assert c_plan.trial_kind is TrialKind.HXC_NESTED

    with pytest.raises(ValueError, match="parent H plan's execution binding"):
        plan_nested_hxc(
            h_plan,
            close_checkpoint_by_h_variant={"P0": d0, "H02": d1},
            close_variant_ids=["C01"],
            settings=settings,
            seed=7,
            execution_binding_sha256="f" * 64,
        )


def test_production_rejects_a_bare_payload_without_provenance(settings):
    store = CheckpointStore(settings.path("checkpoints"))
    digest = store.put(_c())
    plan = plan_c_only_e2e(
        settings, close_checkpoint_digest=digest, task_id="T1",
        close_variant_ids=["C01"], seed=7,
        execution_binding_sha256="e" * 64)

    class Bare(_Backend):
        async def execute(self, spec, checkpoint):
            return "looks plausible but proves nothing"

    with pytest.raises(ForkCapabilityError, match="ForkExecution provenance"):
        asyncio.run(run_production_forks(
            settings, plan, backend=Bare(), checkpoint_store=store))


def test_trajectory_records_pre_treatment_and_mediated_searches():
    recorder = TrajectoryRecorder(treatment_node="H")
    recorder.record("SEARCH_QUERY", {"query": "initial", "result_count": 3})
    recorder.record("H_CHECKPOINT", {"checkpoint": "h1"})
    recorder.record("PAGE_BATCH_REDUCED", {"checkpoint": "h1", "siblings": 1})
    recorder.record("SEARCH_QUERY", {"query": "gap exposed by H", "result_count": 2})
    assert [e["position"] for e in recorder.events] == [
        "PRE_TREATMENT", "PRE_TREATMENT", "TREATMENT", "POST_TREATMENT",
    ]
    assert recorder.summary()["first_treatment_checkpoint_digest"] == "h1"

    other = TrajectoryRecorder(treatment_node="H")
    other.record("SEARCH_QUERY", {"query": "initial", "result_count": 3})
    other.record("H_CHECKPOINT", {"checkpoint": "h1"})
    other.record("PAGE_BATCH_REDUCED", {"checkpoint": "h1", "siblings": 1})
    other.record("SEARCH_QUERY", {"query": "different downstream query", "result_count": 2})
    divergence = first_divergence(recorder.events, other.events)
    assert divergence["event_index"] == 3
    assert divergence["before_treatment"] is False


def test_direct_node_record_keeps_treatment_fidelity_fields():
    recorder = TrajectoryRecorder(treatment_node="H")
    recorder.record("NODE_SELECTION", {"direct_node_record": {
        "node": "H",
        "checkpoint_hash": "h1",
        "offered_span_ids": ["s1", "s2"],
        "selected_span_ids": ["s2"],
        "published_span_ids": ["s2"],
        "fell_back": False,
        "failure": None,
        "contract": "P1_TYPED",
        "aggregation": "coverage_budget_v1",
        "chunker": "fixed_token_v1",
        "stage": "global",
    }})
    record = recorder.direct_node_records[0]
    assert (record["contract"], record["aggregation"], record["stage"]) == (
        "P1_TYPED", "coverage_budget_v1", "global")
    assert record["chunker"] == "fixed_token_v1"
    assert record["offered_span_ids"] == ["s1", "s2"]
    assert recorder.events[0]["position"] == "TREATMENT"
    assert recorder.first_treatment_checkpoint_digest == "h1"


def test_hook_token_trace_survives_recorder_artifact_and_scores():
    """Exercise the production chain; a hook-only unit test missed the recorder whitelist."""
    tokenizer_digest = "a" * 64
    publication_map = (("H0_0_1", "s1"),)
    publication_costs = (("H0_0_1", 1),)
    publication_sha = sha256_hex(canonical_json({
        "handle_to_span": [list(item) for item in publication_map],
        "handle_token_counts": [list(item) for item in publication_costs],
    }))
    outcome = SelectionOutcome(
        text="rendered",
        view_sha256="b" * 64,
        selector_attempted=True,
        offered_span_ids=("s1",),
        offered_span_token_counts=(("s1", 10),),
        offered_material_tokens=10,
        offered_evidence_tokens=10,
        offered_context_tokens=0,
        publication_handle_map=publication_map,
        publication_handle_token_counts=publication_costs,
        publication_map_sha256=publication_sha,
        selected_span_ids=("s1",),
        staged_span_ids=("s1",),
        published_span_ids=("s1",),
        staged_rendered_tokens=4,
        published_rendered_tokens=4,
        offered_source_occurrence_ids=("o1",),
        normalization={
            "raw_count": 1,
            "unique_count": 1,
            "duplicate_count": 0,
            "semantic_conflict_count": 0,
            "rejected_reason": None,
        },
        contract="P1_ID",
        aggregation="stable_union_v1",
        chunker="markdown_structure_v1",
        tokenizer_sha256=tokenizer_digest,
        stage="single",
        checkpoint_hash="c" * 64,
    )
    strategy = SimpleNamespace(last_outcomes=[outcome])
    recorder = TrajectoryRecorder(treatment_node="H")
    binding = RunBinding(
        task_id="T1",
        researcher_id="R1",
        attempt_id="A1",
        task_ctx=SimpleNamespace(),
        on_event=recorder.record,
    )
    with bind_run(binding):
        _emit_selection_records(
            strategy, node="H", checkpoint="c" * 64
        )

    # This is the exact object the runner freezes and campaign/evaluate later consumes.
    artifact = {"direct_node_records": recorder.direct_node_records}
    record = artifact["direct_node_records"][0]
    assert record["materialization_trace_status"] == "OK"
    assert record["offered_span_token_counts"] == [["s1", 10]]
    assert record["published_rendered_tokens"] == 4
    assert record["tokenizer_sha256"] == tokenizer_digest

    truth = {
        "task_id": "T1",
        "required_facets": ["f"],
        "atomic_evidence": [{
            "atom_id": "a1",
            "facet_id": "f",
            "weight": 2.0,
            "supporting_span_ids": ["truth-span"],
        }],
        "contradiction_pairs": [],
    }
    support = {
        "tokenizer_sha256": tokenizer_digest,
        "chunkers": {"markdown_structure_v1": {"a1": ["s1"]}},
        "prechunk_atom_occurrence_ids": {"a1": ["o1"]},
        "candidate_span_occurrence_ids": {
            "markdown_structure_v1": {"s1": ["o1"]}
        },
    }
    scored = score_direct_node_records(
        truth,
        artifact["direct_node_records"],
        atom_support_index=support,
    )
    assert scored["status"] == "OK"
    assert scored["token_trace_complete"] is True
    assert scored["selected_token_precision"] == 1.0
    assert scored["materialization_ratio"] == 0.4


def test_direct_node_record_canonicalizes_normalization_and_recomputes_flags():
    recorder = TrajectoryRecorder(treatment_node="H")
    recorder.record("NODE_SELECTION", {"direct_node_record": {
        "node": "H",
        "checkpoint_hash": "h1",
        "selector_attempted": True,
        "normalization": {
            "raw_count": 2,
            "unique_count": 1,
            "duplicate_count": 1,
            "semantic_conflict_count": 0,
            "rejected_reason": None,
        },
    }})
    record = recorder.direct_node_records[0]
    assert record["normalization_trace_status"] == "OK"
    assert record["normalization"]["was_repaired"] is True
    assert record["normalization"]["strict_valid"] is False


def test_direct_node_record_missing_normalization_is_explicitly_fail_closed():
    recorder = TrajectoryRecorder(treatment_node="H")
    recorder.record("NODE_SELECTION", {"direct_node_record": {
        "node": "H",
        "checkpoint_hash": "h1",
        "selector_attempted": True,
        "normalization": None,
    }})
    record = recorder.direct_node_records[0]
    assert record["normalization"] is None
    assert record["normalization_trace_status"] == "MISSING"
    assert "no normalization trace" in record["normalization_trace_error"]


def test_direct_node_record_rejects_forged_derived_normalization_flags():
    recorder = TrajectoryRecorder(treatment_node="H")
    recorder.record("NODE_SELECTION", {"direct_node_record": {
        "node": "H",
        "checkpoint_hash": "h1",
        "selector_attempted": True,
        "normalization": {
            "raw_count": 1,
            "unique_count": 1,
            "duplicate_count": 0,
            "semantic_conflict_count": 0,
            "rejected_reason": None,
            "strict_valid": False,
        },
    }})
    record = recorder.direct_node_records[0]
    assert record["normalization"] is None
    assert record["normalization_trace_status"] == "INVALID"
    assert "not closed" in record["normalization_trace_error"]


def test_component_screen_stops_before_any_call_without_resume_backend(settings):
    body = asyncio.run(run_screening(settings, repo=REPO, component_only=True))
    assert body["ok"] is False
    assert "NO_BOUNDARY_FORK_BACKEND" in body["error"]
    assert not settings.path("runs").exists(), "fail-closed check mutated the run ledger"


def test_primary_screen_is_e2e_and_does_not_require_resume_backend(settings, monkeypatch):
    import shapeflow_p1.campaign.screen as screen

    observed = {}

    async def fake_leased(_settings, **kwargs):
        observed.update(kwargs)
        return {"ok": True, "execution_semantics": "COUPLED_SEED_E2E_ITT"}

    # A no-op lease, not None: production now refuses to run unleased, because a lease
    # that silently did not engage is how two workers end up on one card.
    monkeypatch.setattr(screen, "_gpu_lease", lambda _settings: nullcontext())
    monkeypatch.setattr(screen, "_run_screening_leased", fake_leased)
    monkeypatch.setattr(
        screen,
        "verified_execution_binding",
        lambda _repo, expected_digest=None: SimpleNamespace(
            digest="e" * 64, protocol_sha="d" * 64),
    )
    body = asyncio.run(run_screening(settings, repo=REPO))
    assert body["ok"] is True
    assert body["execution_semantics"] == "COUPLED_SEED_E2E_ITT"
    assert observed["component_only"] is False
    assert observed["fork_backend"] is None
