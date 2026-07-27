"""The langchain-free ODR adapter layer: close reasons, checkpoints, strategy binding,
parity comparison."""

from __future__ import annotations

import pytest

import contextvars

from shapeflow.odr.checkpoints import (
    CCheckpoint,
    EvidenceManifest,
    FrozenMessage,
    FrozenToolCall,
    HCheckpoint,
    SamplingEnvelope,
    VendorVisibleResult,
    fork_key,
)
from shapeflow.odr.close_reason import CloseReason, classify_close
from shapeflow.odr.hooks import (
    ResearcherHandoff,
    StrategyBundle,
    TaskContext,
    ToolObservation,
    bind_strategies,
    current_strategies,
)
from shapeflow.odr.p0_parity import (
    PublishBatch,
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


def _h_checkpoint(order=0, researcher_coordinate=None):
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
        researcher_coordinate=researcher_coordinate,
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


def test_deferred_page_addresses_exact_vendor_truncated_bytes_and_occurrence():
    from shapeflow.hashing import sha256_hex
    from shapeflow.odr.adapter import DeferredPageBatch

    raw = "abcdefghij"
    batch = DeferredPageBatch(
        tool_call_id="c1",
        tool_name="tavily_search",
        results=({
            "url": "https://e.invalid",
            "title": "E",
            "content": "snippet",
            "raw_content": raw,
            "_shapeflow_occurrence_id": "occ-1",
        },),
        render_vendor=lambda: None,
        max_content_length=5,
    )
    visible = batch.visible_results()[0]
    assert visible.raw_content_id == sha256_hex(raw[:5].encode("utf-8"))
    assert visible.source_occurrence_id == "occ-1"


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
        publish_batches=(PublishBatch((ToolMessageRecord("c1", "tavily_search", "h",)),),),
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


def test_an_atomic_publish_is_not_equal_to_two_partial_ones():
    """`[[A,B]]` vs `[[A],[B]]` is the distinction the whole H design turns on.

    A flat tool-message list cannot express it: both flatten to `[A, B]` and compared equal.
    But publishing half a batch is precisely what P1 must never do, and a P1 failure must take
    the whole batch to P0 rather than leaving a `[P1(A), P0(B)]` hybrid -- which is not an arm,
    and would be scored as one.
    """
    a = ToolMessageRecord("c1", "tavily_search", "a" * 64)
    b = ToolMessageRecord("c2", "tavily_search", "b" * 64)
    atomic = RunTrace(publish_batches=(PublishBatch((a, b)),))
    split = RunTrace(publish_batches=(PublishBatch((a,)), PublishBatch((b,))))
    assert atomic.tool_messages == split.tool_messages       # the old view saw no difference
    assert not compare_traces(atomic, split, require_output_equality=True).ok


def test_identical_bytes_with_different_routing_are_not_parity():
    a = ToolMessageRecord("c1", "tavily_search", "a" * 64)
    vendor = RunTrace(publish_batches=(PublishBatch((a,), goto="researcher"),))
    patched = RunTrace(publish_batches=(PublishBatch((a,), goto="compress_research"),))
    assert not compare_traces(vendor, patched, require_output_equality=True).ok


def test_a_hooks_off_run_that_reached_vendor_via_fallback_is_not_parity():
    """Falling back produces vendor's bytes, so byte comparison alone calls it identical.

    It is not: the hooks-off path must be inert, not merely recoverable. A patch that crashed
    and quietly fell back would otherwise pass the gate that exists to prove it is inert.
    """
    a = ToolMessageRecord("c1", "tavily_search", "a" * 64)
    vendor = RunTrace(publish_batches=(PublishBatch((a,)),))
    patched = RunTrace(publish_batches=(PublishBatch((a,), fallback="WHOLE_BATCH_P0"),))
    assert not compare_traces(vendor, patched, require_output_equality=True).ok


def test_a_swallowed_exception_is_a_parity_difference():
    a = ToolMessageRecord("c1", "tavily_search", "a" * 64)
    vendor = RunTrace(publish_batches=(PublishBatch((a,)),))
    patched = RunTrace(publish_batches=(PublishBatch((a,)),), exceptions=("TimeoutError",))
    assert not compare_traces(vendor, patched, require_output_equality=True).ok


def test_strategy_binding_is_released_even_when_an_arm_raises():
    """A binding that survived an exception would execute one arm's strategy under another
    arm's id -- a mislabelled observation, not a crash, and undetectable downstream."""
    from shapeflow.odr.hooks import StrategyBundle, current_strategies, strategies_bound

    bundle = StrategyBundle(variant_id="H02", page=object(), close=object())
    with pytest.raises(RuntimeError):
        with strategies_bound(bundle):
            assert current_strategies() is bundle
            raise RuntimeError("arm failed")
    assert current_strategies() is None


def test_both_strategy_protocols_are_async():
    """The vendor path is async and a strategy makes model calls. A sync strategy would block
    the event loop, changing the batching and timing this study measures."""
    import inspect

    from shapeflow.odr.hooks import PageTransformStrategy, ResearchCloseStrategy

    assert inspect.iscoroutinefunction(PageTransformStrategy.transform_tool_batch)
    assert inspect.iscoroutinefunction(ResearchCloseStrategy.close_researcher)


def test_frozen_message_keeps_the_fields_a_rendered_prompt_depends_on():
    """C_VISIBLE's claim is that the selector saw exactly what P0's compressor saw.

    The compressor sees the rendered message list, so a dropped field is a field the clone
    lacks -- the two prompts differ by however it renders, while every hash we compute agrees.
    """
    from shapeflow.odr.checkpoints import FrozenMessage

    keys = set(FrozenMessage(role="ai", content="x").content_dict())
    assert {"additional_kwargs", "response_metadata", "usage_metadata", "artifact",
            "invalid_tool_calls", "status"} <= keys
    # Structured content blocks survive as blocks rather than being flattened to text.
    blocks = FrozenMessage(role="ai", content=[{"type": "text", "text": "hi"}])
    assert blocks.content_dict()["content"] == [{"type": "text", "text": "hi"}]
    assert blocks.text_sha256 != FrozenMessage(role="ai", content="hi").text_sha256


def test_tool_bytes_always_compared_even_in_weak_mode():
    v = RunTrace(publish_batches=(PublishBatch((ToolMessageRecord("c1", "s", "h1"),)),))
    p = RunTrace(publish_batches=(PublishBatch((ToolMessageRecord("c1", "s", "h2"),)),))
    assert not compare_traces(v, p, require_output_equality=False).ok


def test_close_reason_divergence_fails_parity():
    v = RunTrace(close_reasons=("RESEARCH_COMPLETE",))
    p = RunTrace(close_reasons=("MAX_REACT_EXCEEDED",))
    assert not compare_traces(v, p, require_output_equality=False).ok


# --- a checkpoint has to survive the process that made it ------------------------------------


def _sampling():
    from shapeflow.odr.checkpoints import SamplingEnvelope

    return SamplingEnvelope(model="Qwen3-14B-AWQ", temperature=0.3, top_p=1.0,
                            max_tokens=4096, seed=11)


def _c_checkpoint():
    from shapeflow.odr.checkpoints import (
        CCheckpoint,
        EvidenceManifest,
        FrozenMessage,
        FrozenToolCall,
    )

    return CCheckpoint(
        task_id="T1", researcher_id="R1",
        researcher_messages=(
            FrozenMessage(role="system", content="you research"),
            FrozenMessage(
                role="ai", content=[{"type": "text", "text": "calling"}],
                tool_calls=(FrozenToolCall(id="c1", name="tavily_search",
                                           args_canonical='{"query":"q"}'),),
                additional_kwargs_canonical='{"refusal":null}',
                response_metadata_canonical='{"finish_reason":"tool_calls"}',
                usage_metadata_canonical='{"input_tokens":10}',
                message_id="m-2"),
            FrozenMessage(role="tool", content="page text", tool_call_id="c1",
                          name="tavily_search", artifact_canonical='{"n":1}',
                          status="success"),
        ),
        evidence_manifest=EvidenceManifest(span_ids=("s1", "s2")),
        query_attempt_ids=("qa1",), close_reason="RESEARCH_COMPLETE",
        sampling=_sampling(),
    )


def test_a_checkpoint_round_trips_byte_for_byte(tmp_path):
    """The forked-state design says every variant starts from byte-identical input. That is
    a claim about a document that can be read back, not about an object in memory."""
    from shapeflow.canonical import canonical_json
    from shapeflow.odr.checkpoints import CheckpointStore, to_document

    original = _c_checkpoint()
    store = CheckpointStore(tmp_path / "checkpoints")
    digest = store.put(original)

    restored = store.get(digest)
    assert restored == original
    assert canonical_json(to_document(restored)) == canonical_json(to_document(original))
    assert restored.researcher_messages[1].additional_kwargs_canonical == '{"refusal":null}'
    assert restored.researcher_messages[2].status == "success"


def test_an_h_checkpoint_round_trips_including_its_researcher_coordinate(tmp_path):
    """The C round trip above passed while every *child* H boundary was unreadable.

    ``researcher_coordinate`` is in ``content()`` and therefore in the digest, but
    ``from_document`` did not restore it, so a checkpoint written by a ConductResearch child
    rebuilt with ``None``, hashed differently, and tripped its own digest check on load. The
    store was effectively write-only for exactly the boundaries a fork has to start from.
    """
    from shapeflow.canonical import canonical_json
    from shapeflow.odr.checkpoints import CheckpointStore, to_document

    original = _h_checkpoint(researcher_coordinate=(2, 1, "call-abc"))
    store = CheckpointStore(tmp_path / "checkpoints")

    restored = store.get(store.put(original))

    assert restored.researcher_coordinate == (2, 1, "call-abc")
    assert restored == original
    assert canonical_json(to_document(restored)) == canonical_json(to_document(original))
    assert restored.digest == original.digest


def test_an_h_checkpoint_without_a_coordinate_still_round_trips(tmp_path):
    """``None`` is reserved for direct researcher-subgraph probes and must stay ``None``."""
    from shapeflow.odr.checkpoints import CheckpointStore

    original = _h_checkpoint()
    store = CheckpointStore(tmp_path / "checkpoints")

    restored = store.get(store.put(original))

    assert restored.researcher_coordinate is None
    assert restored == original


def test_a_coordinate_is_part_of_the_boundary_identity():
    """Two children of one supervisor turn are different boundaries, not one."""
    assert (
        _h_checkpoint(researcher_coordinate=(1, 0, "call-a")).digest
        != _h_checkpoint(researcher_coordinate=(1, 1, "call-b")).digest
    )
    assert (
        _h_checkpoint(researcher_coordinate=(1, 0, "call-a")).digest
        != _h_checkpoint().digest
    )


def test_a_checkpoint_document_that_does_not_rebuild_to_its_digest_is_refused(tmp_path):
    import json

    import pytest

    from shapeflow.odr.checkpoints import CheckpointStore, from_document, to_document

    body = to_document(_c_checkpoint())
    body["close_reason"] = "NO_TOOL_CALL"          # the state changed; the digest did not
    with pytest.raises(ValueError, match="not the one it claims"):
        from_document(body)


def test_the_stored_document_validates_against_the_schema(tmp_path):
    import json
    from pathlib import Path

    from jsonschema import Draft202012Validator

    from shapeflow.odr.checkpoints import to_document

    repo = Path(__file__).resolve().parents[2]
    schema = json.loads((repo / "schemas" / "checkpoint.schema.json").read_text())
    Draft202012Validator(schema).validate(to_document(_c_checkpoint()))


def test_the_summarize_timeout_binding_does_not_leak_out_of_a_cell():
    """It must be scoped, or one cell silently reconfigures everything after it.

    Assigning os.environ directly leaked the campaign's 300s ceiling into the whole process:
    once any test had run a cell, every later test inherited it instead of vendor's 60s, and the
    suite ran until it was killed. Vendor's fallback only means "byte-for-byte vendor by
    default" if the variable is genuinely absent outside the cell that set it.
    """
    import os

    from shapeflow.campaign.graph_driver import _summarize_timeout_applied

    os.environ.pop("SHAPEFLOW_SUMMARIZE_TIMEOUT_S", None)
    with _summarize_timeout_applied("300.0"):
        assert os.environ["SHAPEFLOW_SUMMARIZE_TIMEOUT_S"] == "300.0"
    assert "SHAPEFLOW_SUMMARIZE_TIMEOUT_S" not in os.environ

    os.environ["SHAPEFLOW_SUMMARIZE_TIMEOUT_S"] = "45.0"
    try:
        with _summarize_timeout_applied("300.0"):
            assert os.environ["SHAPEFLOW_SUMMARIZE_TIMEOUT_S"] == "300.0"
        assert os.environ["SHAPEFLOW_SUMMARIZE_TIMEOUT_S"] == "45.0"
    finally:
        os.environ.pop("SHAPEFLOW_SUMMARIZE_TIMEOUT_S", None)
