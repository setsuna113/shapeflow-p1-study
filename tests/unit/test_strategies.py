"""Every registered arm must be constructible, and the arms must stay separable.

configs/variants.yaml listed sixteen variants while production code implemented none of them.
A campaign would have found that out one arm at a time, mid-run, with budget already spent --
so the first test here is simply that the whole registry instantiates.

The rest defend the separations the design depends on. If C_VISIBLE can reach raw page bytes, or
a control quietly does what the treatment does, the numbers still come out; they just answer a
different question than the one asked.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer
from shapeflow_p1.odr.checkpoints import (
    CCheckpoint,
    EvidenceManifest,
    FrozenMessage,
    FrozenToolCall,
    HCheckpoint,
    SamplingEnvelope,
    VendorVisibleResult,
)
from shapeflow_p1.odr.hooks import TaskContext
from shapeflow_p1.strategies.close_visible import CloseSelectionStrategy, CloseStrategyConfig
from shapeflow_p1.strategies.factory import StrategyFactory, UnknownVariant, load_registry
from shapeflow_p1.strategies.visible_view import build_visible_view

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
TOK = WhitespaceTokenizer()
PAGE = "Cats are feline animals. Dogs are canine animals. Birds can fly high.\n\nMore text here."
TASK = TaskContext(task_id="t", protocol_sha="p", variant_id="v", seed=1,
                   research_topic="feline animals", max_content_length=200)


async def _fake_model(*, prompt, op_class, max_tokens, schema_name):
    """A selector that always picks E1, plus plausible usage."""
    if schema_name is None:
        return "A short prose summary of the evidence.", {"prompt_tokens": 10,
                                                          "completion_tokens": 5}
    return ({"contract": "P1_ID", "selected_ids": ["E1"]},
            {"prompt_tokens": 100, "completion_tokens": 8})


def _factory(**kw) -> StrategyFactory:
    return StrategyFactory(
        registry=load_registry(CONFIGS), model_call=_fake_model, tokenizer=TOK,
        token_budget=200, raw_text_for=lambda cid: PAGE,
        occurrence_for=lambda cid: "occ-" + cid[:8], **kw
    )


def test_every_registered_variant_is_constructible():
    """The registry listed arms that no code could build. That must fail at startup, not mid-run."""
    bundles = _factory().build_all()
    assert set(bundles) == set(load_registry(CONFIGS))
    assert len(bundles) == 16
    for vid, bundle in bundles.items():
        assert hasattr(bundle.page, "transform_tool_batch"), vid
        assert hasattr(bundle.close, "close_researcher"), vid


def test_an_unregistered_variant_is_refused():
    with pytest.raises(UnknownVariant, match="not in the frozen registry"):
        _factory().build("H99-INVENTED")


def test_the_primary_two_by_two_is_buildable():
    """P0, H, C_VISIBLE and H+C_VISIBLE are the primary design; all four must exist."""
    f = _factory()
    assert f.build("P0").variant_id == "P0"
    assert f.build("H02").variant_id == "H02"
    assert f.build("C02").variant_id == "C02"
    joint = f.build_joint("H02", "C02")
    assert joint.variant_id == "H02+C02"
    # The joint cell is composed of the same halves as the main-effect cells, so the
    # interaction cannot drift from the arms it is meant to combine.
    assert type(joint.page) is type(f.build("H02").page)
    assert type(joint.close) is type(f.build("C02").close)


def test_c_visible_has_no_means_to_read_raw_page_bytes():
    """Structural, not conventional. C_VISIBLE is constructed without a raw-span accessor.

    A flag saying "do not read raw spans" is a rule someone can get wrong; not holding the
    accessor is a property of the object.
    """
    f = _factory(raw_spans_for=lambda cp: [{"namespace": "RAW_SOURCE"}])
    c_visible = f.build("C02").close
    assert c_visible._raw_spans_for is None
    c_registry = f.build("C06-REG").close
    assert c_registry._raw_spans_for is not None


def test_constructing_c_visible_with_registry_access_is_an_error():
    with pytest.raises(ValueError, match="C_VISIBLE was constructed with a raw-span accessor"):
        CloseSelectionStrategy(
            CloseStrategyConfig(variant_id="C02", node="C_VISIBLE", contract="P1_TYPED",
                                aggregation="coverage_budget_v1",
                                close_mode="dedicated_selector", token_budget=100),
            selector=None, tokenizer=TOK, raw_spans_for=lambda cp: [],
        )


def test_controls_do_not_select_ids():
    """A control that did what the treatment does would control for nothing."""
    f = _factory()
    cpu = f.build("H00-CPU").page
    assert type(cpu._selector).__name__ == "CpuLexicalAsyncSelector"
    prose = f.build("H00-PROSE").page
    assert type(prose).__name__ == "ProsePageStrategy"
    assert getattr(prose._selector, "is_prose", False) is True


def test_a_close_only_arm_leaves_the_page_node_at_p0():
    """The two nodes are measured separately; a C arm that also changed H would confound them."""
    page = _factory().build("C02").page
    assert type(page).__name__ == "VendorPageStrategy"


def test_a_page_only_arm_leaves_the_close_node_at_p0():
    close = _factory().build("H02").close
    assert type(close).__name__ == "VendorCloseStrategy"


def test_the_fused_extension_is_not_implemented_and_says_so():
    """Two of three exits never call the fused tool, and the third does not either.

    C05-FUSED-EXT needs a P1-only tool because vendor's ResearchComplete has an empty
    schema, and that tool does not exist. Both branches run the fallback, so the arm differs
    from dedicated_selector by a label. The label now says which, because "FUSED" on a path
    that never fused is how an untested arm gets reported as a tested null.
    """
    from shapeflow_p1.strategies.fused import FusedCloseStrategy

    fused = _factory().build("C05-FUSED-EXT").close
    assert isinstance(fused, FusedCloseStrategy)

    async def run(reason):
        cp = CCheckpoint(
            task_id="t", researcher_id="r",
            researcher_messages=(FrozenMessage(role="tool", content="obs", tool_call_id="c1"),),
            evidence_manifest=EvidenceManifest(span_ids=()), query_attempt_ids=(),
            close_reason=reason,
            sampling=SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=10),
        )
        try:
            await fused.close_researcher(task_ctx=TASK, checkpoint=cp)
        except Exception:
            pass
        return fused.last_path

    assert asyncio.run(run("MAX_REACT_EXCEEDED")) == "DEDICATED_FALLBACK"
    assert asyncio.run(run("NO_TOOL_CALL")) == "DEDICATED_FALLBACK"
    assert asyncio.run(run("RESEARCH_COMPLETE")) == "FALLBACK_FUSED_NOT_IMPLEMENTED"


def _h_checkpoint() -> HCheckpoint:
    return HCheckpoint(
        task_id="t", researcher_id="r", assistant_turn_index=0,
        assistant_message=FrozenMessage(role="ai", content="searching"),
        sibling_tool_calls=(FrozenToolCall(id="c1", name="tavily_search", args_canonical="{}"),),
        search_result_sets=(("c1", (VendorVisibleResult(
            vendor_visible_order=0, url="https://a.example", title="A", snippet="snip",
            raw_content_id="a" * 64),)),),
        non_search_outputs=(), researcher_state_hash="s" * 64,
        sampling=SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=10),
    )


def test_the_h_arm_actually_publishes_selected_evidence_and_records_work():
    """A P1 arm that produced vendor's output would be inert -- exactly what the canary checks."""
    page = _factory().build("H02").page
    out = asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))
    assert len(out) == 1
    assert out[0].tool_call_id == "c1"
    assert "Selected evidence" in out[0].content
    assert "[E1]" in out[0].content
    # Work is recorded whether or not the output was usable: tokens spent on a failed selection
    # are still tokens spent, and recording them only on success flatters P1 when it goes wrong.
    assert page.last_work.selector_calls == 1
    assert page.last_work.prompt_tokens == 100


def test_the_short_prose_control_is_held_to_the_same_token_budget():
    """If the control could exceed the budget it would no longer be token-matched, and the
    comparison it exists to support would be void."""
    f = StrategyFactory(
        registry=load_registry(CONFIGS), model_call=_long_prose, tokenizer=TOK,
        token_budget=5, raw_text_for=lambda cid: PAGE, occurrence_for=lambda cid: "occ",
    )
    page = f.build("H00-PROSE").page
    out = asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))
    assert TOK.count(out[0].content) <= 5


async def _long_prose(*, prompt, op_class, max_tokens, schema_name):
    return " ".join(f"word{i}" for i in range(200)), {"prompt_tokens": 5, "completion_tokens": 200}


def test_the_visible_view_keeps_model_reasoning_out_of_evidence():
    """Selecting a span over AI reasoning must not make it TOOL_EVIDENCE.

    That line is the difference between C_VISIBLE reading what the compressor read and
    C_VISIBLE manufacturing provenance for a model's own words.
    """
    view = build_visible_view((
        FrozenMessage(role="ai", content="I think the answer is X", message_id="m1"),
        FrozenMessage(role="tool", content="SOURCE 1: evidence", tool_call_id="c1",
                      message_id="m2"),
    ))
    kinds = {seg["message_id"]: seg["kind"] for seg in view.message_segments}
    assert kinds["m1"] == "MODEL_DERIVED_CONTEXT"
    assert kinds["m2"] == "TOOL_EVIDENCE"
    # Each segment addresses real bytes of the rendered view.
    for seg in view.message_segments:
        body = view.view_bytes[seg["byte_start"]:seg["byte_end"]].decode()
        assert body in ("I think the answer is X", "SOURCE 1: evidence")


def test_a_tool_message_without_recorded_provenance_gets_none():
    """Provenance is not reconstructed after the fact.

    Reverse-mapping a model-written summary back to the page it described is precisely the
    smuggling that separates C_REGISTRY from C_VISIBLE.
    """
    view = build_visible_view((
        FrozenMessage(role="tool", content="summary of a page", tool_call_id="c1",
                      message_id="m1"),
    ))
    assert view.message_segments[0]["occurrence_ids"] == []
