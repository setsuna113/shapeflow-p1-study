"""The langchain-free ODR adapter layer: close reasons, checkpoints, strategy binding,
parity comparison."""

from __future__ import annotations

import contextvars

from shapeflow_p1.odr.checkpoints import (
    CCheckpoint,
    EvidenceManifest,
    FrozenMessage,
    FrozenToolCall,
    HCheckpoint,
    SamplingEnvelope,
    VendorVisibleResult,
    fork_key,
)
from shapeflow_p1.odr.close_reason import CloseReason, classify_close
from shapeflow_p1.odr.hooks import (
    ResearcherHandoff,
    StrategyBundle,
    TaskContext,
    ToolObservation,
    bind_strategies,
    current_strategies,
)
from shapeflow_p1.odr.p0_parity import (
    ModelRequest,
    RunTrace,
    ToolMessageRecord,
    compare_traces,
)

# --- close reason classification (must mirror vendor precedence) ----------------------


def test_no_tool_call_is_early_exit():
    assert (
        classify_close(
            has_tool_calls=False,
            has_native_search=False,
            research_complete_called=False,
            tool_call_iterations=1,
            max_react_tool_calls=10,
        )
        is CloseReason.NO_TOOL_CALL
    )


def test_research_complete_takes_precedence_over_max_react():
    # Both conditions hold; the explicit signal wins, as it is the more informative label.
    assert (
        classify_close(
            has_tool_calls=True,
            has_native_search=False,
            research_complete_called=True,
            tool_call_iterations=10,
            max_react_tool_calls=10,
        )
        is CloseReason.RESEARCH_COMPLETE
    )


def test_max_react_when_budget_hit_without_complete():
    assert (
        classify_close(
            has_tool_calls=True,
            has_native_search=False,
            research_complete_called=False,
            tool_call_iterations=10,
            max_react_tool_calls=10,
        )
        is CloseReason.MAX_REACT_EXCEEDED
    )


def test_none_when_researcher_continues():
    assert (
        classify_close(
            has_tool_calls=True,
            has_native_search=False,
            research_complete_called=False,
            tool_call_iterations=2,
            max_react_tool_calls=10,
        )
        is None
    )


# --- checkpoint content addressing ----------------------------------------------------


def _sampling():
    return SamplingEnvelope(model="qwen", temperature=0.3, top_p=1.0, max_tokens=8192, seed=7)


def _h_checkpoint(order=0):
    msg = FrozenMessage(
        role="ai",
        content="I will search.",
        tool_calls=(FrozenToolCall(id="c1", name="tavily_search", args_canonical='{"q":"x"}'),),
    )
    result = VendorVisibleResult(
        vendor_visible_order=order, url="https://e.com", title="E",
        snippet="snip", raw_content_id="a" * 64,
    )
    return HCheckpoint(
        task_id="t1",
        researcher_id="r1",
        assistant_turn_index=0,
        assistant_message=msg,
        sibling_tool_calls=(FrozenToolCall(id="c1", name="tavily_search", args_canonical='{"q":"x"}'),),
        search_result_sets=(("c1", (result,)),),
        non_search_outputs=(),
        researcher_state_hash="b" * 64,
        sampling=_sampling(),
    )


def test_h_checkpoint_digest_is_deterministic():
    assert _h_checkpoint().digest == _h_checkpoint().digest


def test_h_checkpoint_digest_changes_with_content():
    assert _h_checkpoint(order=0).digest != _h_checkpoint(order=1).digest


def test_checkpoint_is_immutable():
    import dataclasses

    cp = _h_checkpoint()
    try:
        cp.task_id = "other"  # type: ignore[misc]
        assert False, "checkpoint must be frozen"
    except dataclasses.FrozenInstanceError:
        pass


def test_c_checkpoint_digest_distinguishes_close_reason():
    base = dict(
        task_id="t",
        researcher_id="r",
        researcher_messages=(FrozenMessage(role="tool", content="obs", tool_call_id="c1"),),
        evidence_manifest=EvidenceManifest(span_ids=("s1", "s2")),
        query_attempt_ids=("q1",),
        sampling=_sampling(),
    )
    complete = CCheckpoint(**base, close_reason="RESEARCH_COMPLETE")
    maxed = CCheckpoint(**base, close_reason="MAX_REACT_EXCEEDED")
    assert complete.digest != maxed.digest


def test_raw_content_is_referenced_not_inlined():
    # The checkpoint content must carry the raw content by hash, never the bytes.
    cp = _h_checkpoint()
    content = cp.content()
    ref = content["search_result_sets"][0][1][0]["raw_content_id"]
    assert ref == "a" * 64  # a content_id reference, not page bytes


def test_fork_key_depends_on_all_coordinates():
    base = dict(
        protocol_sha="p", boundary_id="b", variant_id="H02", seed=1,
        prompt_renderer_version="v1",
    )
    k = fork_key(**base)
    assert k != fork_key(**{**base, "variant_id": "H03"})
    assert k != fork_key(**{**base, "seed": 2})
    assert k == fork_key(**base)


# --- strategy binding is context-local ------------------------------------------------


class _NoopPage:
    def transform_tool_batch(self, *, task_ctx, checkpoint):
        return [ToolObservation(tool_call_id="c1", name="tavily_search", content="obs")]


class _NoopClose:
    def close_researcher(self, *, task_ctx, checkpoint):
        return ResearcherHandoff(compressed_research="done", raw_notes=())


def test_strategies_default_to_none():
    assert current_strategies() is None


def test_binding_is_isolated_per_context():
    bundle = StrategyBundle(variant_id="H02", page=_NoopPage(), close=_NoopClose())

    seen = {}

    def child():
        bind_strategies(bundle)
        seen["child"] = current_strategies()

    # Run in a copied context: the binding must NOT leak back to the parent, mirroring how
    # asyncio copies context per researcher task.
    ctx = contextvars.copy_context()
    ctx.run(child)
    assert seen["child"] is bundle
    assert current_strategies() is None  # parent context untouched


def test_taskcontext_carries_no_oracle_fields():
    # A structural check: TaskContext exposes only decision-time-visible fields.
    tc = TaskContext(
        task_id="t", protocol_sha="p", variant_id="H02", seed=1,
        research_topic="topic bytes", max_content_length=50000,
    )
    fields = set(tc.__dataclass_fields__)
    forbidden = {"acquisition_spec", "truth_packet", "gold_facets", "critical_items"}
    assert not (fields & forbidden)


# --- parity comparison ----------------------------------------------------------------


def _req(prompt="p", out="o"):
    return ModelRequest(
        op_class="RESEARCHER_REACT",
        prompt_sha256=prompt,
        tools_signature="tools",
        model="qwen",
        sampling_signature="s",
        output_sha256=out,
    )


def test_identical_traces_are_parity_ok():
    t = RunTrace(
        model_requests=(_req(),),
        tool_messages=(ToolMessageRecord("c1", "tavily_search", "h"),),
        close_reasons=("RESEARCH_COMPLETE",),
        final_report_sha256="f",
    )
    assert compare_traces(t, t, require_output_equality=True).ok


def test_differing_prompt_bytes_fail_parity():
    v = RunTrace(model_requests=(_req(prompt="a"),))
    p = RunTrace(model_requests=(_req(prompt="b"),))
    rep = compare_traces(v, p, require_output_equality=False)
    assert not rep.ok
    assert "envelope differs" in rep.diffs[0]


def test_output_difference_ignored_in_weak_mode_only():
    v = RunTrace(model_requests=(_req(out="x"),))
    p = RunTrace(model_requests=(_req(out="y"),))
    # Weak mode (real sampling): output difference is allowed.
    assert compare_traces(v, p, require_output_equality=False).ok
    # Strong mode (mock deterministic): it is a parity failure.
    assert not compare_traces(v, p, require_output_equality=True).ok


def test_tool_bytes_always_compared_even_in_weak_mode():
    v = RunTrace(tool_messages=(ToolMessageRecord("c1", "s", "h1"),))
    p = RunTrace(tool_messages=(ToolMessageRecord("c1", "s", "h2"),))
    assert not compare_traces(v, p, require_output_equality=False).ok


def test_close_reason_divergence_fails_parity():
    v = RunTrace(close_reasons=("RESEARCH_COMPLETE",))
    p = RunTrace(close_reasons=("MAX_REACT_EXCEEDED",))
    assert not compare_traces(v, p, require_output_equality=False).ok
