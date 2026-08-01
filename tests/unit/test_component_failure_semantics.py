"""Component failures are observations, never disguised vendor successes."""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from shapeflow.odr.adapter import (
    BatchOutcome,
    DeferredPageBatch,
    PageFailure,
    _resolve_failure,
)
from shapeflow.odr.checkpoints import SamplingEnvelope
from shapeflow.odr.hooks import StrategyBundle, strategies_bound
from shapeflow.odr.vendor_hooks import (
    ComponentTrialFailure,
    RunBinding,
    bind_run,
    record_tool_batch_publication,
    reduce_published_batch,
    run_close_strategy,
)


@pytest.mark.asyncio
async def test_e2e_whole_batch_fallback_preserves_vendor_sibling_concurrency():
    both_started = asyncio.Event()
    started: list[str] = []

    def deferred(name: str) -> DeferredPageBatch:
        async def render() -> str:
            started.append(name)
            if len(started) == 2:
                both_started.set()
            await asyncio.wait_for(both_started.wait(), timeout=0.5)
            return f"vendor-{name}"

        return DeferredPageBatch(
            tool_call_id=name,
            tool_name="tavily_search",
            results=(),
            render_vendor=render,
            max_content_length=50_000,
        )

    observations = (deferred("a"), deferred("b"))
    outcome = await _resolve_failure(
        PageFailure("TEST"),
        observations,
        [0, 1],
        SimpleNamespace(digest="checkpoint"),
        component_trial=False,
    )

    assert outcome.observations == ("vendor-a", "vendor-b")
    assert outcome.fell_back is True
    assert started == ["a", "b"]


@pytest.mark.asyncio
async def test_page_component_failure_aborts_before_toolmessage_publication(monkeypatch):
    import shapeflow.odr.vendor_hooks as vendor_hooks

    deferred = DeferredPageBatch(
        tool_call_id="",
        tool_name="tavily_search",
        results=(),
        render_vendor=lambda: None,
        max_content_length=50_000,
    )

    async def fail_batch(**_kwargs):
        return BatchOutcome(
            observations=(deferred,),
            checkpoint_digest="h-checkpoint",
            failure=PageFailure("INVALID_SELECTION", "bad contract"),
        )

    monkeypatch.setattr(vendor_hooks, "reduce_tool_batch", fail_batch)
    sampling = SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=8, seed=1)
    run = RunBinding(
        task_id="t",
        researcher_id="r",
        attempt_id="a",
        task_ctx=object(),
        component_trial=True,
        sampling=sampling,
    )
    bundle = StrategyBundle(variant_id="H", page=object(), close=object())
    assistant = SimpleNamespace(
        type="ai",
        content="",
        tool_calls=[{"id": "c1", "name": "tavily_search", "args": {}}],
        invalid_tool_calls=[],
        additional_kwargs={},
        response_metadata={},
        usage_metadata={},
    )

    with strategies_bound(bundle), bind_run(run):
        with pytest.raises(ComponentTrialFailure, match="INVALID_SELECTION"):
            await reduce_published_batch(
                assistant_message=assistant,
                tool_calls=[{"id": "c1", "name": "tavily_search", "args": {}}],
                observations=[deferred],
                researcher_state_hash="state",
                assistant_turn_index=0,
            )


@pytest.mark.asyncio
async def test_close_component_failure_does_not_publish_vendor_compression():
    class BrokenClose:
        async def close_researcher(self, **_kwargs):
            raise ValueError("invalid close selection")

    sampling = SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=8, seed=1)
    run = RunBinding(
        task_id="t",
        researcher_id="r",
        attempt_id="a",
        task_ctx=object(),
        component_trial=True,
        sampling=sampling,
    )
    bundle = StrategyBundle(variant_id="C", page=object(), close=BrokenClose())

    with strategies_bound(bundle), bind_run(run):
        with pytest.raises(ComponentTrialFailure, match="invalid close selection"):
            await run_close_strategy(
                researcher_messages=[],
                close_reason="NO_TOOL_CALL",
            )


@pytest.mark.asyncio
async def test_direct_selection_and_atomic_publish_are_emitted_at_their_real_boundaries(
    monkeypatch,
):
    import shapeflow.odr.vendor_hooks as vendor_hooks

    deferred = DeferredPageBatch(
        tool_call_id="",
        tool_name="tavily_search",
        results=(),
        render_vendor=lambda: None,
        max_content_length=50_000,
    )

    async def succeed_batch(**_kwargs):
        return BatchOutcome(
            observations=("selected evidence",),
            checkpoint_digest="h-checkpoint",
        )

    monkeypatch.setattr(vendor_hooks, "reduce_tool_batch", succeed_batch)
    outcome = SimpleNamespace(
        offered_span_ids=("s1", "s2"),
        selected_span_ids=("s2",),
        published_span_ids=("s2",),
        selector_attempted=True,
        normalization={
            "raw_count": 1,
            "unique_count": 1,
            "duplicate_count": 0,
            "semantic_conflict_count": 0,
            "rejected_reason": None,
        },
        failure=None,
        contract="P1_ID",
        aggregation="stable_union_v1",
        stage="single",
        checkpoint_hash="h-checkpoint",
    )
    page = SimpleNamespace(last_outcomes=[outcome])
    events: list[tuple[str, dict]] = []
    run = RunBinding(
        task_id="t",
        researcher_id="r",
        attempt_id="a",
        task_ctx=object(),
        sampling=SamplingEnvelope(
            model="m", temperature=0.0, top_p=1.0, max_tokens=8, seed=1
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )
    bundle = StrategyBundle(variant_id="H", page=page, close=object())
    tool_calls = [{"id": "c1", "name": "tavily_search", "args": {}}]
    assistant = SimpleNamespace(
        type="ai",
        content="",
        tool_calls=tool_calls,
        invalid_tool_calls=[],
        additional_kwargs={},
        response_metadata={},
        usage_metadata={},
    )

    with strategies_bound(bundle), bind_run(run):
        reduced = await reduce_published_batch(
            assistant_message=assistant,
            tool_calls=tool_calls,
            observations=[deferred],
            researcher_state_hash="state",
            assistant_turn_index=0,
        )
        assert reduced == ("selected evidence",)
        assert "NODE_SELECTION" in [kind for kind, _ in events]
        assert "TOOL_BATCH_PUBLISHED" not in [kind for kind, _ in events]

        record_tool_batch_publication(
            tool_calls=tool_calls,
            tool_outputs=[
                SimpleNamespace(
                    content="selected evidence",
                    name="tavily_search",
                    tool_call_id="c1",
                )
            ],
        )

    direct = next(payload["direct_node_record"] for kind, payload in events
                  if kind == "NODE_SELECTION")
    assert direct["offered_span_ids"] == ["s1", "s2"]
    assert direct["selected_span_ids"] == ["s2"]
    assert direct["published_span_ids"] == ["s2"]
    assert direct["selector_attempted"] is True
    assert direct["normalization"]["raw_count"] == 1
    published = next(payload for kind, payload in events
                     if kind == "TOOL_BATCH_PUBLISHED")
    assert published["checkpoint"] == "h-checkpoint"
    assert published["sibling_count"] == 1
    assert published["tool_call_ids"] == ["c1"]
    assert published["publish_calls"] == 1
    assert published["atomic_publish"] is True


@pytest.mark.asyncio
async def test_atomic_publish_rejects_a_reordered_or_partial_toolmessage_batch(monkeypatch):
    import shapeflow.odr.vendor_hooks as vendor_hooks

    deferred = DeferredPageBatch(
        tool_call_id="",
        tool_name="tavily_search",
        results=(),
        render_vendor=lambda: None,
        max_content_length=50_000,
    )

    async def succeed_batch(**_kwargs):
        return BatchOutcome(
            observations=("a", "b"),
            checkpoint_digest="h-checkpoint",
        )

    monkeypatch.setattr(vendor_hooks, "reduce_tool_batch", succeed_batch)
    page = SimpleNamespace(last_outcomes=[])
    events: list[tuple[str, dict]] = []
    run = RunBinding(
        task_id="t",
        researcher_id="r",
        attempt_id="a",
        task_ctx=object(),
        sampling=SamplingEnvelope(
            model="m", temperature=0.0, top_p=1.0, max_tokens=8, seed=1
        ),
        on_event=lambda kind, payload: events.append((kind, payload)),
    )
    bundle = StrategyBundle(variant_id="H", page=page, close=object())
    tool_calls = [
        {"id": "a", "name": "tavily_search", "args": {}},
        {"id": "b", "name": "tavily_search", "args": {}},
    ]
    assistant = SimpleNamespace(
        type="ai",
        content="",
        tool_calls=tool_calls,
        invalid_tool_calls=[],
        additional_kwargs={},
        response_metadata={},
        usage_metadata={},
    )

    with strategies_bound(bundle), bind_run(run):
        await reduce_published_batch(
            assistant_message=assistant,
            tool_calls=tool_calls,
            observations=[deferred, deferred],
            researcher_state_hash="state",
            assistant_turn_index=0,
        )
        with pytest.raises(RuntimeError, match="atomic publication integrity"):
            record_tool_batch_publication(
                tool_calls=tool_calls,
                tool_outputs=[
                    SimpleNamespace(content="a", name="tavily_search", tool_call_id="a")
                ],
            )

    kinds = [kind for kind, _ in events]
    assert "TOOL_BATCH_PUBLICATION_REJECTED" in kinds
    assert "TOOL_BATCH_PUBLISHED" not in kinds
