"""Capture-time C provenance and nested ConductResearch identity are fail-closed."""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace

import pytest
from langchain_core.messages import ToolMessage

from shapeflow.odr.adapter import (
    BatchOutcome,
    DeferredPageBatch,
    PageFailure,
    _resolve_failure,
)
from shapeflow.odr.checkpoints import (
    FrozenMessage,
    HCheckpoint,
    SamplingEnvelope,
    VendorVisibleResult,
)
from shapeflow.odr.hooks import (
    ResearcherHandoff,
    StrategyBundle,
    strategies_bound,
)
from shapeflow.odr.vendor_hooks import (
    RunBinding,
    bind_run,
    current_published_tool_provenance,
    current_run,
    invoke_child_researcher,
    record_tool_batch_publication,
    reduce_published_batch,
    refine_child_research_tasks,
    run_close_strategy,
)
from shapeflow.p1 import handle_codec
from shapeflow.p1.view import compact_publication_handle
from shapeflow.strategies.p0 import VendorPageStrategy
from shapeflow.strategies.page_h import PageSelectionStrategy
from shapeflow.strategies.visible_view import build_visible_view


SAMPLING = SamplingEnvelope(
    model="m", temperature=0.0, top_p=1.0, max_tokens=32, seed=1
)


def _result(occurrence_id: str) -> VendorVisibleResult:
    return VendorVisibleResult(
        vendor_visible_order=0,
        url=f"https://{occurrence_id}.example",
        title=occurrence_id,
        snippet=f"snippet-{occurrence_id}",
        raw_content_id=f"content-{occurrence_id}",
        source_occurrence_id=occurrence_id,
    )


def _checkpoint(*pairs: tuple[str, tuple[VendorVisibleResult, ...]]) -> HCheckpoint:
    return HCheckpoint(
        task_id="task",
        researcher_id="researcher",
        assistant_turn_index=0,
        assistant_message=FrozenMessage(role="ai", content="search"),
        sibling_tool_calls=(),
        search_result_sets=tuple(pairs),
        non_search_outputs=(),
        researcher_state_hash="state",
        sampling=SAMPLING,
    )


def _deferred(call_id: str, occurrence_ids: tuple[str, ...], text: str) -> DeferredPageBatch:
    async def render() -> str:
        return text

    return DeferredPageBatch(
        tool_call_id=call_id,
        tool_name="tavily_search",
        results=tuple(
            {
                "url": f"https://{occurrence_id}.example",
                "title": occurrence_id,
                "content": f"snippet-{occurrence_id}",
                "raw_content": f"raw-{occurrence_id}",
                "_shapeflow_occurrence_id": occurrence_id,
            }
            for occurrence_id in occurrence_ids
        ),
        render_vendor=render,
        max_content_length=50_000,
    )


@pytest.mark.asyncio
async def test_p0_and_fallback_lineage_come_from_each_frozen_call_not_reverse_lookup():
    checkpoint = _checkpoint(
        ("c1", (_result("o1"), _result("o2"))),
        ("c2", (_result("o3"),)),
    )
    page = VendorPageStrategy({
        "c1": _deferred("c1", ("o1", "o2"), "vendor-one"),
        "c2": _deferred("c2", ("o3",), "vendor-two"),
    })
    produced = await page.transform_tool_batch(
        task_ctx=SimpleNamespace(), checkpoint=checkpoint
    )
    assert [item.source_occurrence_ids for item in produced] == [
        ("o1", "o2"),
        ("o3",),
    ]

    fallback = await _resolve_failure(
        PageFailure("P1_FAILED"),
        (
            page._deferred["c1"],
            page._deferred["c2"],
        ),
        [0, 1],
        checkpoint,
        component_trial=False,
    )
    assert fallback.observations == ("vendor-one", "vendor-two")
    assert fallback.publication_provenance == (
        ("c1", ("o1", "o2")),
        ("c2", ("o3",)),
    )


def test_p1_lineage_contains_only_final_published_spans_and_snippet_passthrough():
    groups = [
        {
            "spans": [
                {"span_id": "s1", "source_occurrence_ids": ["o1"]},
                {"span_id": "s2", "source_occurrence_ids": ["o2"]},
            ],
            "passthrough_occurrence_ids": (),
        },
        {
            "spans": [],
            "passthrough_occurrence_ids": ("o3",),
        },
    ]
    # o1 was offered/local-selected but the final reducer published only s2. It must not be
    # smuggled into C_VISIBLE by using the broad offered-source set.
    final_outcome = SimpleNamespace(published_span_ids=("s2",))
    assert PageSelectionStrategy._published_occurrences(
        groups, [final_outcome]
    ) == ("o2", "o3")


@pytest.mark.asyncio
async def test_atomic_publish_sidecar_enriches_only_the_frozen_c_checkpoint(monkeypatch):
    import shapeflow.odr.vendor_hooks as vendor_hooks

    captured = []

    class CaptureClose:
        async def close_researcher(self, *, task_ctx, checkpoint):
            captured.append(checkpoint)
            return ResearcherHandoff("compressed", ("raw",))

    async def selected_batch(**_kwargs):
        return BatchOutcome(
            observations=("selected evidence",),
            checkpoint_digest="h-checkpoint",
            publication_provenance=(("c1", ("o-selected",)),),
        )

    monkeypatch.setattr(vendor_hooks, "reduce_tool_batch", selected_batch)
    page = SimpleNamespace(last_outcomes=[])
    bundle = StrategyBundle(variant_id="P1", page=page, close=CaptureClose())
    run = RunBinding(
        task_id="task",
        researcher_id="root",
        attempt_id="attempt",
        task_ctx=SimpleNamespace(),
        sampling=SAMPLING,
    )
    tool_calls = [{"id": "c1", "name": "tavily_search", "args": {}}]
    deferred = _deferred("c1", ("o-selected", "o-dropped"), "vendor")
    assistant = SimpleNamespace(
        type="ai",
        content="",
        tool_calls=tool_calls,
        invalid_tool_calls=[],
        additional_kwargs={},
        response_metadata={},
        usage_metadata={},
    )
    live_message = ToolMessage(
        content="selected evidence",
        name="tavily_search",
        tool_call_id="c1",
    )

    with strategies_bound(bundle), bind_run(run):
        await reduce_published_batch(
            assistant_message=assistant,
            tool_calls=tool_calls,
            observations=[deferred],
            researcher_state_hash="state",
            assistant_turn_index=0,
        )
        record_tool_batch_publication(
            tool_calls=tool_calls, tool_outputs=[live_message]
        )
        assert current_published_tool_provenance()["c1"][
            "source_occurrence_ids"
        ] == ("o-selected",)
        await run_close_strategy(
            researcher_messages=[live_message],
            close_reason="NO_TOOL_CALL",
        )

    # The graph-visible message remains vendor-shaped: no artifact was written into it.
    assert getattr(live_message, "artifact", None) is None
    frozen = captured[0].researcher_messages[0]
    artifact = json.loads(frozen.artifact_canonical)
    assert artifact["source_occurrence_ids"] == ["o-selected"]
    segment = build_visible_view((frozen,)).message_segments[0]
    assert segment["kind"] == "TOOL_EVIDENCE"
    assert segment["occurrence_ids"] == ["o-selected"]
    assert current_published_tool_provenance() == {}


@pytest.mark.asyncio
async def test_multi_turn_sidecar_accumulates_and_reused_id_conflict_fails_closed(monkeypatch):
    import shapeflow.odr.vendor_hooks as vendor_hooks

    outcomes = iter((
        BatchOutcome(
            observations=("turn-one",),
            checkpoint_digest="h1",
            publication_provenance=(("c1", ("o1",)),),
        ),
        BatchOutcome(
            observations=("turn-two",),
            checkpoint_digest="h2",
            publication_provenance=(("c2", ("o2",)),),
        ),
        BatchOutcome(
            observations=("different",),
            checkpoint_digest="h3",
            publication_provenance=(("c1", ("o3",)),),
        ),
    ))

    async def next_outcome(**_kwargs):
        return next(outcomes)

    monkeypatch.setattr(vendor_hooks, "reduce_tool_batch", next_outcome)
    run = RunBinding(
        task_id="task",
        researcher_id="root",
        attempt_id="attempt",
        task_ctx=SimpleNamespace(),
        sampling=SAMPLING,
    )
    bundle = StrategyBundle(
        variant_id="P1",
        page=SimpleNamespace(last_outcomes=[]),
        close=SimpleNamespace(),
    )

    def assistant(call_id: str):
        calls = [{"id": call_id, "name": "tavily_search", "args": {}}]
        return calls, SimpleNamespace(
            type="ai",
            content="",
            tool_calls=calls,
            invalid_tool_calls=[],
            additional_kwargs={},
            response_metadata={},
            usage_metadata={},
        )

    with strategies_bound(bundle), bind_run(run):
        for call_id, content in (("c1", "turn-one"), ("c2", "turn-two")):
            calls, ai = assistant(call_id)
            await reduce_published_batch(
                assistant_message=ai,
                tool_calls=calls,
                observations=[_deferred(call_id, (f"o-{call_id}",), "vendor")],
                researcher_state_hash=f"state-{call_id}",
                assistant_turn_index=0,
            )
            record_tool_batch_publication(
                tool_calls=calls,
                tool_outputs=[
                    ToolMessage(
                        content=content,
                        name="tavily_search",
                        tool_call_id=call_id,
                    )
                ],
            )
        assert set(current_published_tool_provenance()) == {"c1", "c2"}

        calls, ai = assistant("c1")
        await reduce_published_batch(
            assistant_message=ai,
            tool_calls=calls,
            observations=[_deferred("c1", ("o3",), "vendor")],
            researcher_state_hash="state-3",
            assistant_turn_index=1,
        )
        with pytest.raises(RuntimeError, match="reused"):
            record_tool_batch_publication(
                tool_calls=calls,
                tool_outputs=[
                    ToolMessage(
                        content="different",
                        name="tavily_search",
                        tool_call_id="c1",
                    )
                ],
            )

    # A new run in the same asyncio context starts empty; no sidecar or pending batch leaks.
    with bind_run(run):
        assert current_published_tool_provenance() == {}


@pytest.mark.asyncio
async def test_child_researchers_get_isolated_deterministic_coordinates_concurrently():
    entered = asyncio.Event()
    seen = []

    class ChildGraph:
        async def ainvoke(self, state, config):
            binding = current_run()
            seen.append((
                binding.researcher_id,
                binding.researcher_coordinate,
                state["research_topic"],
            ))
            if len(seen) == 2:
                entered.set()
            await asyncio.wait_for(entered.wait(), timeout=1)
            return {"coordinate": binding.researcher_coordinate}

    async def never_started():
        raise AssertionError("coarse parent-bound child coroutine ran")

    parent = RunBinding(
        task_id="task",
        researcher_id="task:0",
        attempt_id="attempt",
        task_ctx=SimpleNamespace(),
        sampling=SAMPLING,
    )
    calls = [
        {"id": "conduct-a", "args": {"research_topic": "alpha"}},
        {"id": "conduct-b", "args": {"research_topic": "beta"}},
    ]
    with bind_run(parent):
        tasks = refine_child_research_tasks(
            original_tasks=[never_started(), never_started()],
            researcher_subgraph=ChildGraph(),
            allowed_tool_calls=calls,
            config={"configurable": {}},
            supervisor_research_iteration=3,
        )
        results = await asyncio.gather(*tasks)
        assert current_run() == parent
        assert current_published_tool_provenance() == {}

    assert {coordinate for _rid, coordinate, _topic in seen} == {
        (3, 0, "conduct-a"),
        (3, 1, "conduct-b"),
    }
    assert len({rid for rid, _coordinate, _topic in seen}) == 2
    assert {result["coordinate"] for result in results} == {
        (3, 0, "conduct-a"),
        (3, 1, "conduct-b"),
    }
    assert current_run() is None


def test_child_structural_slot_conflict_and_compact_handle_aliasing_fail_closed():
    parent = RunBinding(
        task_id="task",
        researcher_id="task:0",
        attempt_id="attempt",
        task_ctx=SimpleNamespace(),
    )
    graph = SimpleNamespace(ainvoke=lambda *_args, **_kwargs: None)
    with bind_run(parent):
        first = invoke_child_researcher(
            researcher_subgraph=graph,
            input_state={},
            config={},
            supervisor_research_iteration=1,
            allowed_tool_call_ordinal=0,
            tool_call_id="call-a",
        )
        first.close()
        with pytest.raises(RuntimeError, match="reused"):
            invoke_child_researcher(
                researcher_subgraph=graph,
                input_state={},
                config={},
                supervisor_research_iteration=1,
                allowed_tool_call_ordinal=0,
                tool_call_id="call-b",
            )

    a = compact_publication_handle(
        0, 0, 1, researcher_coordinate=(1, 0)
    )
    b = compact_publication_handle(
        0, 0, 1, researcher_coordinate=(1, 1)
    )
    later = compact_publication_handle(
        0, 0, 1, researcher_coordinate=(2, 0)
    )
    assert len({a, b, later}) == 3
    # Absence of a researcher coordinate is its own structural position, not a shorter handle:
    # the predecessor dropped the field and changed the handle's arity, which is what the
    # trajectory validator's grammar then failed to match.
    unnested = compact_publication_handle(0, 0, 1)
    assert unnested == "Ha"
    assert unnested not in {a, b, later}
    for handle in (a, b, later, unnested):
        assert handle_codec.validate(handle), handle
