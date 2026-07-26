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
import re
from dataclasses import replace
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator

from shapeflow_p1.campaign.selector_client import (
    ALIAS_BY_OP,
    SHORT_PROSE_OPS,
    STRUCTURED_SELECTOR_OPS,
    SelectorModelCall,
    SelectorResponseError,
    load_selector_schema,
    selector_schema_name,
)
from shapeflow_p1.evaluation.citation_support import parse_citation_map
from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer
from shapeflow_p1.hashing import sha256_hex
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
from shapeflow_p1.p1.aggregators import stable_union_v1
from shapeflow_p1.p1.contracts import parse_selection
from shapeflow_p1.p1.view import CandidateViewRecord
from shapeflow_p1.providers.provider_client import ProviderCallError
from shapeflow_p1.strategies.close_visible import (
    CloseSelectionError,
    CloseSelectionStrategy,
    CloseStrategyConfig,
)
from shapeflow_p1.strategies.factory import (
    StrategyFactory,
    UnknownVariant,
    VariantUnavailable,
    load_registry,
)
from shapeflow_p1.strategies.page_h import (
    PageSelectionError,
    PageSelectionStrategy,
    PageStrategyConfig,
)
from shapeflow_p1.strategies.pipeline import WorkRecord, spans_from_page
from shapeflow_p1.strategies.selectors_async import CpuLexicalAsyncSelector
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
    """Every runnable arm builds; planned arms are explicit rather than fallback-shaped."""
    registry = load_registry(CONFIGS)
    bundles = _factory().build_all()
    assert set(bundles) == {vid for vid, spec in registry.items() if spec.runnable}
    assert len(bundles) == 16
    assert {"H_TYPED_STABLE", "H_HIER_COVERAGE", "C_TYPED_STABLE"} <= set(bundles)
    for vid, bundle in bundles.items():
        assert hasattr(bundle.page, "transform_tool_batch"), vid
        assert hasattr(bundle.close, "close_researcher"), vid
    assert {
        vid for vid, spec in registry.items() if not spec.runnable
    } == {"C04", "C05-FUSED-EXT", "C06-REG"}


def test_an_unregistered_variant_is_refused():
    with pytest.raises(UnknownVariant, match="not in the frozen registry"):
        _factory().build("H99-INVENTED")


def test_guided_decode_schema_is_one_contract_not_the_union():
    id_schema = load_selector_schema(CONFIGS.parent, "P1_ID")
    typed_schema = load_selector_schema(CONFIGS.parent, "P1_TYPED")
    id_validator = Draft202012Validator(id_schema)
    typed_validator = Draft202012Validator(typed_schema)
    id_output = {"contract": "P1_ID", "selected_ids": ["E1"]}
    typed_output = {
        "contract": "P1_TYPED",
        "selections": [{"span_id": "E1", "role": "support"}],
    }
    assert id_validator.is_valid(id_output)
    assert typed_validator.is_valid(typed_output)
    assert not id_validator.is_valid(typed_output)
    assert not typed_validator.is_valid(id_output)
    assert selector_schema_name("P1_BRIDGE") == "selector_output_P1_BRIDGE"


def test_selector_seed_is_sent_to_the_engine_request():
    class CaptureClient:
        body = None

        async def chat_completions(self, body, *, cell_token):
            self.body = dict(body)
            return {
                "choices": [{"message": {
                    "content": '{"contract":"P1_ID","selected_ids":["E1"]}'
                }}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    client = CaptureClient()
    call = SelectorModelCall(
        client, cell_token="cell", repo=CONFIGS.parent,  # noqa: S106
        temperature=0.3, top_p=0.9, max_completion_tokens=128, seed=1977,
    )
    asyncio.run(call(
        prompt="x", op_class="PAGE_P1_SELECTOR_LOCAL", max_tokens=64,
        schema_name="selector_output_P1_ID",
    ))
    assert client.body["seed"] == 1977


def test_short_prose_completion_cap_is_not_raised_to_structured_json_floor():
    """A four-token prose remainder must reach the engine as four, not sixty-four."""

    class CaptureClient:
        bodies = []

        async def chat_completions(self, body, *, cell_token):
            self.bodies.append(dict(body))
            content = (
                "four token prose"
                if "response_format" not in body
                else '{"contract":"P1_ID","selected_ids":[]}'
            )
            return {
                "choices": [{"message": {"content": content}}],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1},
            }

    client = CaptureClient()
    call = SelectorModelCall(
        client, cell_token="cell", repo=CONFIGS.parent,  # noqa: S106
        temperature=0.0, top_p=1.0, max_completion_tokens=128, seed=7,
    )
    asyncio.run(call(
        prompt="prose",
        op_class="COMPRESSOR_SHORT_PROSE",
        max_tokens=4,
        schema_name=None,
    ))
    asyncio.run(call(
        prompt="structured",
        op_class="COMPRESSOR_P1_SELECTOR",
        max_tokens=4,
        schema_name="selector_output_P1_ID",
    ))

    assert client.bodies[0]["max_tokens"] == 4
    assert client.bodies[1]["max_tokens"] == 64


def test_selector_op_policy_is_an_exhaustive_disjoint_partition():
    assert not (STRUCTURED_SELECTOR_OPS & SHORT_PROSE_OPS)
    assert STRUCTURED_SELECTOR_OPS | SHORT_PROSE_OPS == set(ALIAS_BY_OP)


@pytest.mark.parametrize("guided", [True, False])
@pytest.mark.parametrize("op_class", sorted(STRUCTURED_SELECTOR_OPS))
def test_structured_ops_cannot_dispatch_without_a_contract_schema(op_class, guided):
    class NoDispatchClient:
        calls = 0

        async def chat_completions(self, body, *, cell_token):
            self.calls += 1
            raise AssertionError("invalid request reached provider")

    client = NoDispatchClient()
    call = SelectorModelCall(
        client, cell_token="cell", repo=CONFIGS.parent,  # noqa: S106
        temperature=0.0, top_p=1.0, max_completion_tokens=128, seed=7,
        guided_decoding=guided,
    )
    with pytest.raises(ProviderCallError, match="requires one contract-specific schema"):
        asyncio.run(call(
            prompt="structured",
            op_class=op_class,
            max_tokens=4,
            schema_name=None,
        ))
    assert client.calls == 0


@pytest.mark.parametrize("guided", [True, False])
@pytest.mark.parametrize("op_class", sorted(SHORT_PROSE_OPS))
def test_short_prose_ops_cannot_dispatch_with_a_structured_schema(op_class, guided):
    class NoDispatchClient:
        calls = 0

        async def chat_completions(self, body, *, cell_token):
            self.calls += 1
            raise AssertionError("invalid request reached provider")

    client = NoDispatchClient()
    call = SelectorModelCall(
        client, cell_token="cell", repo=CONFIGS.parent,  # noqa: S106
        temperature=0.0, top_p=1.0, max_completion_tokens=128, seed=7,
        guided_decoding=guided,
    )
    with pytest.raises(ProviderCallError, match="must not carry a structured schema"):
        asyncio.run(call(
            prompt="prose",
            op_class=op_class,
            max_tokens=4,
            schema_name="selector_output_P1_ID",
        ))
    assert client.calls == 0


def test_guided_decoding_off_still_validates_schema_and_parses_structured_json():
    class CaptureClient:
        body = None

        async def chat_completions(self, body, *, cell_token):
            self.body = dict(body)
            return {
                "choices": [{"message": {
                    "content": '{"contract":"P1_ID","selected_ids":[]}'
                }}],
                "usage": {},
            }

    client = CaptureClient()
    call = SelectorModelCall(
        client, cell_token="cell", repo=CONFIGS.parent,  # noqa: S106
        temperature=0.0, top_p=1.0, max_completion_tokens=128, seed=7,
        guided_decoding=False,
    )
    parsed, _ = asyncio.run(call(
        prompt="structured",
        op_class="COMPRESSOR_P1_SELECTOR",
        max_tokens=4,
        schema_name="selector_output_P1_ID",
    ))
    assert parsed == {"contract": "P1_ID", "selected_ids": []}
    assert "response_format" not in client.body
    assert client.body["max_tokens"] == 64

    with pytest.raises(ValueError, match="not contract-specific"):
        asyncio.run(call(
            prompt="structured",
            op_class="COMPRESSOR_P1_SELECTOR",
            max_tokens=4,
            schema_name="selector_output",
        ))


def test_structured_selector_rejects_a_configured_cap_below_grammar_floor():
    class NoDispatchClient:
        calls = 0

        async def chat_completions(self, body, *, cell_token):
            self.calls += 1
            raise AssertionError("invalid request reached provider")

    client = NoDispatchClient()
    call = SelectorModelCall(
        client, cell_token="cell", repo=CONFIGS.parent,  # noqa: S106
        temperature=0.0, top_p=1.0, max_completion_tokens=32, seed=7,
    )
    with pytest.raises(ProviderCallError, match="below its 64-token grammar floor"):
        asyncio.run(call(
            prompt="structured",
            op_class="PAGE_P1_SELECTOR_LOCAL",
            max_tokens=4,
            schema_name="selector_output_P1_ID",
        ))
    assert client.calls == 0


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
    with pytest.raises(VariantUnavailable, match="C_VISIBLE fallback is forbidden"):
        f.build("C06-REG")


def test_constructing_c_visible_with_registry_access_is_an_error():
    with pytest.raises(ValueError, match="C_VISIBLE was constructed with a raw-span accessor"):
        CloseSelectionStrategy(
            CloseStrategyConfig(variant_id="C02", node="C_VISIBLE", contract="P1_TYPED",
                                aggregation="coverage_budget_v1",
                                close_mode="dedicated_selector", token_budget=100),
            selector=None, tokenizer=TOK, raw_spans_for=lambda cp: [],
        )


def test_controls_isolate_selector_backend_and_generic_shortening():
    """CPU holds the structured path fixed; prose deliberately changes the whole path."""
    f = _factory()
    cpu = f.build("H00-CPU").page
    assert type(cpu._selector).__name__ == "CpuLexicalAsyncSelector"
    assert cpu.config.contract == "P1_ID"
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


def test_unimplemented_close_extensions_cannot_masquerade_as_runnable():
    """Two of three exits never call the fused tool, and the third does not either.

    C05-FUSED-EXT needs a P1-only tool because vendor's ResearchComplete has an empty
    schema, and that tool does not exist. Both branches run the fallback, so the arm differs
    from dedicated_selector by a label. The label now says which, because "FUSED" on a path
    that never fused is how an untested arm gets reported as a tested null.
    """
    f = _factory()
    with pytest.raises(VariantUnavailable, match="real ResearchCompleteWithSelection"):
        f.build("C05-FUSED-EXT")
    with pytest.raises(VariantUnavailable, match="selector input prefix"):
        f.build("C04")


def test_close_cpu_control_really_runs_cpu_selection():
    close = _factory().build("C00-CPU").close
    assert isinstance(close, CloseSelectionStrategy)
    assert type(close._selector).__name__ == "CpuLexicalAsyncSelector"


def test_cpu_control_uses_the_shared_rendered_token_budget_not_a_fixed_top_k():
    text = " ".join(
        f"feline evidence sentence number {index} has several supporting words."
        for index in range(30)
    )
    content_hash = sha256_hex(text.encode())
    spans = spans_from_page(
        text,
        content_hash=content_hash,
        occurrence_id="occ",
        chunker="fixed_token_v1",
        tokenizer=TOK,
        max_tokens=10,
    )
    view = CandidateViewRecord.build(
        spans=spans,
        tokenizer=TOK,
        namespace="RAW_SOURCE",
        topic="feline evidence",
        contract="P1_ID",
        token_budget=42,
        query_attempts=[],
        snapshot_texts={content_hash: text},
    )
    context = TaskContext(
        task_id="t",
        protocol_sha="p",
        variant_id="H00-CPU",
        seed=1,
        research_topic="feline evidence",
        max_content_length=1000,
        selected_token_budget=42,
    )

    raw, _work = asyncio.run(
        CpuLexicalAsyncSelector().select(task_ctx=context, view=view)
    )
    parsed = parse_selection(
        raw, view.candidate_set, expected_contract="P1_ID"
    )

    assert len(raw["selected_ids"]) != 8
    assert view.cost(stable_union_v1(parsed, view.registry)) <= 42


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


def _c_checkpoint() -> CCheckpoint:
    return CCheckpoint(
        task_id="t",
        researcher_id="r",
        researcher_messages=(
            FrozenMessage(role="system", content="Research feline animals."),
            FrozenMessage(
                role="tool",
                content="Cats are feline animals. Dogs are canine animals.",
                tool_call_id="c1",
                message_id="m1",
                artifact_canonical='{"source_occurrence_ids":["occ-a"]}',
                status="success",
            ),
        ),
        evidence_manifest=EvidenceManifest(span_ids=("source-span",)),
        query_attempt_ids=("qa1",),
        close_reason="RESEARCH_COMPLETE",
        sampling=SamplingEnvelope(
            model="m", temperature=0.0, top_p=1.0, max_tokens=10, seed=1
        ),
    )


def test_the_h_arm_actually_publishes_selected_evidence_and_records_work():
    """A P1 arm that produced vendor's output would be inert -- exactly what the canary checks."""
    page = _factory().build("H02").page
    out = asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))
    assert len(out) == 1
    assert out[0].tool_call_id == "c1"
    assert "Selected evidence" in out[0].content
    assert "[H0_0_" in out[0].content
    assert "[E1]" not in out[0].content
    # Work is recorded whether or not the output was usable: tokens spent on a failed selection
    # are still tokens spent, and recording them only on success flatters P1 when it goes wrong.
    assert page.last_work.selector_calls == 1
    assert page.last_work.prompt_tokens == 100
    outcome = page.last_outcomes[0]
    assert outcome.offered_evidence_tokens == sum(
        count for _span_id, count in outcome.offered_span_token_counts
    )
    assert outcome.published_rendered_tokens > 0
    assert outcome.published_rendered_tokens <= page.config.token_budget


def test_h_publication_handles_are_unique_across_pages_and_stable_across_turns():
    """Prompt-local E1 labels must not become ambiguous after page outputs are concatenated."""
    base = _h_checkpoint()
    results = (
        VendorVisibleResult(
            vendor_visible_order=0,
            url="https://a.example",
            title="A",
            snippet="a",
            raw_content_id="a" * 64,
            source_occurrence_id="occ-a",
        ),
        VendorVisibleResult(
            vendor_visible_order=1,
            url="https://b.example",
            title="B",
            snippet="b",
            raw_content_id="b" * 64,
            source_occurrence_id="occ-b",
        ),
    )
    first_checkpoint = replace(
        base,
        search_result_sets=(("c1", results),),
    )
    page = _factory().build("H02").page
    first = asyncio.run(
        page.transform_tool_batch(task_ctx=TASK, checkpoint=first_checkpoint)
    )[0].content
    handles = re.findall(r"\[(H[0-9a-z]+_[0-9a-z]+_[0-9a-z]+)\]", first)
    assert len(handles) == 2
    assert len(set(handles)) == 2
    assert "[E1]" not in first

    later_checkpoint = replace(
        first_checkpoint,
        assistant_turn_index=first_checkpoint.assistant_turn_index + 1,
        researcher_state_hash="t" * 64,
    )
    # An idempotent graph retry of the same immutable checkpoint preserves the mapping.
    replay = asyncio.run(
        page.transform_tool_batch(task_ctx=TASK, checkpoint=first_checkpoint)
    )[0].content
    assert re.findall(
        r"\[(H[0-9a-z]+_[0-9a-z]+_[0-9a-z]+)\]", replay
    ) == handles

    later = asyncio.run(
        page.transform_tool_batch(task_ctx=TASK, checkpoint=later_checkpoint)
    )[0].content
    later_handles = re.findall(
        r"\[(H[0-9a-z]+_[0-9a-z]+_[0-9a-z]+)\]", later
    )
    assert len(later_handles) == 2
    assert set(later_handles).isdisjoint(handles)

    conflicting_retry = replace(
        first_checkpoint,
        researcher_state_hash="u" * 64,
    )
    with pytest.raises(PageSelectionError, match="PUBLICATION_SCOPE_REUSE"):
        asyncio.run(
            page.transform_tool_batch(
                task_ctx=TASK,
                checkpoint=conflicting_retry,
            )
        )


def test_short_prose_records_the_exact_bounded_output_tokens():
    page = _factory().build("H00-PROSE").page
    out = asyncio.run(
        page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint())
    )
    # The treatment-specific materialization is capped before H's common outer tool-message
    # wrapper, exactly like H02.  Record both quantities rather than pretending the wrapper
    # did not reach the downstream researcher.
    assert page.last_rendered_token_counts == (
        page.last_control_records[0]["rendered_tokens"],
    )
    assert page.last_control_records[0]["prose_body_tokens"] == TOK.count(
        "A short prose summary of the evidence."
    )
    assert page.last_published_token_counts == (TOK.count(out[0].content),)
    assert page.last_published_token_counts[0] >= page.last_rendered_token_counts[0]
    assert 0 < page.last_rendered_token_counts[0] <= page.config.token_budget


def test_h_prose_and_id_offer_identical_per_page_views_and_preserve_result_order():
    """The prose contrast changes the output language, not its H input or source order."""
    base = _h_checkpoint()
    cp = HCheckpoint(
        task_id=base.task_id,
        researcher_id=base.researcher_id,
        assistant_turn_index=base.assistant_turn_index,
        assistant_message=base.assistant_message,
        sibling_tool_calls=base.sibling_tool_calls,
        search_result_sets=(("c1", (
            VendorVisibleResult(
                0, "https://a.example", "A", "raw", "a" * 64,
                source_occurrence_id="occ-a",
            ),
            VendorVisibleResult(
                1, "https://snippet.example", "Snippet Source",
                "The vendor-only snippet.", None,
                source_occurrence_id="occ-snippet",
            ),
        )),),
        non_search_outputs=base.non_search_outputs,
        researcher_state_hash=base.researcher_state_hash,
        sampling=base.sampling,
    )
    factory = _factory()
    p1 = factory.build("H02").page
    prose = factory.build("H00-PROSE").page
    p1_output = asyncio.run(
        p1.transform_tool_batch(task_ctx=TASK, checkpoint=cp)
    )[0].content
    prose_output = asyncio.run(
        prose.transform_tool_batch(task_ctx=TASK, checkpoint=cp)
    )[0].content

    selectable_outcomes = [
        outcome for outcome in p1.last_outcomes if outcome.offered_span_ids
    ]
    assert len(selectable_outcomes) == len(prose.last_control_records) == 1
    assert selectable_outcomes[0].view_sha256 == \
        prose.last_control_records[0]["candidate_view_sha256"]
    assert list(selectable_outcomes[0].offered_span_ids) == \
        prose.last_control_records[0]["offered_span_ids"]
    assert selectable_outcomes[0].offered_source_occurrence_ids == ("occ-a",)
    assert prose.last_control_records[0]["offered_source_occurrence_ids"] == [
        "occ-a"
    ]
    # asyncio.gather may finish in any order, but the published bytes retain vendor order.
    assert prose_output.index("A short prose summary") < \
        prose_output.index("The vendor-only snippet.")
    assert p1_output.index("https://a.example") < \
        p1_output.index("https://snippet.example")


@pytest.mark.parametrize("variant_id", ["H02", "H00-PROSE"])
def test_h_sibling_tool_calls_keep_vendor_outer_concurrency(variant_id):
    """Whole-batch atomicity must not serialize otherwise independent sibling reducers."""
    active = 0
    maximum = 0

    async def concurrent_model(*, prompt, op_class, max_tokens, schema_name):
        nonlocal active, maximum
        active += 1
        maximum = max(maximum, active)
        await asyncio.sleep(0.01)
        active -= 1
        if schema_name is None:
            return "bounded prose", {"prompt_tokens": 1, "completion_tokens": 2}
        return (
            {"contract": "P1_ID", "selected_ids": ["E1"]},
            {"prompt_tokens": 1, "completion_tokens": 2},
        )

    base = _h_checkpoint()
    checkpoint = HCheckpoint(
        task_id=base.task_id,
        researcher_id=base.researcher_id,
        assistant_turn_index=base.assistant_turn_index,
        assistant_message=base.assistant_message,
        sibling_tool_calls=(
            FrozenToolCall(id="c1", name="tavily_search", args_canonical="{}"),
            FrozenToolCall(id="c2", name="tavily_search", args_canonical="{}"),
        ),
        search_result_sets=(
            ("c1", (
                VendorVisibleResult(
                    0, "https://a.example", "A", "a", "a" * 64,
                    source_occurrence_id="occ-a",
                ),
            )),
            ("c2", (
                VendorVisibleResult(
                    0, "https://b.example", "B", "b", "b" * 64,
                    source_occurrence_id="occ-b",
                ),
            )),
        ),
        non_search_outputs=(),
        researcher_state_hash=base.researcher_state_hash,
        sampling=base.sampling,
    )
    factory = StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=concurrent_model,
        tokenizer=TOK,
        token_budget=200,
        raw_text_for=lambda _content_id: PAGE,
        occurrence_for=lambda content_id: "occ-" + content_id[:8],
    )
    outputs = asyncio.run(
        factory.build(variant_id).page.transform_tool_batch(
            task_ctx=TASK,
            checkpoint=checkpoint,
        )
    )

    assert maximum >= 2
    assert [output.tool_call_id for output in outputs] == ["c1", "c2"]


def test_h_prose_atomic_sibling_failure_retains_every_attempt_trace() -> None:
    async def one_fails(*, prompt, op_class, max_tokens, schema_name):
        if "BBBB_ONLY" in prompt:
            error = RuntimeError("second sibling failed")
            error.usage = {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "retries": 1,
            }
            raise error
        return "bounded prose", {"prompt_tokens": 2, "completion_tokens": 2}

    base = _h_checkpoint()
    checkpoint = HCheckpoint(
        task_id=base.task_id,
        researcher_id=base.researcher_id,
        assistant_turn_index=base.assistant_turn_index,
        assistant_message=base.assistant_message,
        sibling_tool_calls=(
            FrozenToolCall(id="c1", name="tavily_search", args_canonical="{}"),
            FrozenToolCall(id="c2", name="tavily_search", args_canonical="{}"),
        ),
        search_result_sets=(
            ("c1", (
                VendorVisibleResult(
                    0, "https://a.example", "A", "a", "a" * 64,
                    source_occurrence_id="occ-a",
                ),
            )),
            ("c2", (
                VendorVisibleResult(
                    0, "https://b.example", "B", "b", "b" * 64,
                    source_occurrence_id="occ-b",
                ),
            )),
        ),
        non_search_outputs=(),
        researcher_state_hash=base.researcher_state_hash,
        sampling=base.sampling,
    )
    raw_pages = {
        "a" * 64: "AAAA_ONLY source material",
        "b" * 64: "BBBB_ONLY source material",
    }
    page = StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=one_fails,
        tokenizer=TOK,
        token_budget=80,
        raw_text_for=raw_pages.__getitem__,
        occurrence_for=lambda content_id: "occ-" + content_id[0],
    ).build("H00-PROSE").page

    with pytest.raises(PageSelectionError, match="second sibling failed"):
        asyncio.run(
            page.transform_tool_batch(task_ctx=TASK, checkpoint=checkpoint)
        )

    assert len(page.last_control_records) == 2
    assert {record["attempt_status"] for record in page.last_control_records} == {
        "OUTPUT_OBSERVED",
        "CALL_FAILED",
    }
    assert all(
        record["batch_accepted"] is False
        and record["publication_status"] == "FALLBACK_DISCARDED"
        for record in page.last_control_records
    )
    assert page.last_work.selector_calls == 2
    assert page.last_work.completion_tokens == 3
    assert page.last_work.retries == 1


def test_c_prose_and_id_offer_identical_visible_views_and_chunker():
    """C's prose control must not silently use a different visible chunker."""
    factory = _factory()
    p1 = factory.build("C01").close
    prose = factory.build("C00-PROSE").close
    checkpoint = _c_checkpoint()

    asyncio.run(p1.close_researcher(task_ctx=TASK, checkpoint=checkpoint))
    prose_output = asyncio.run(
        prose.close_researcher(task_ctx=TASK, checkpoint=checkpoint)
    )

    assert p1.last_outcome is not None
    assert p1.last_outcome.offered_context_tokens > 0
    assert p1.last_outcome.offered_evidence_tokens > 0
    assert p1.last_outcome.offered_material_tokens == (
        p1.last_outcome.offered_evidence_tokens
        + p1.last_outcome.offered_context_tokens
    )
    control = prose.last_control_records[0]
    assert p1.last_outcome.view_sha256 == control["candidate_view_sha256"]
    assert list(p1.last_outcome.offered_span_ids) == control["offered_span_ids"]
    assert control["chunker"] == p1.config.chunker == "markdown_structure_v1"
    assert control["scope"] == p1.config.scope == "per_researcher"
    assert prose.last_published_token_counts == (
        TOK.count(prose_output.compressed_research),
    )


def test_c_id_and_prose_share_byte_bound_source_affordance_without_header_selection():
    """A selected body chunk inherits only its own frozen title/URL in both control arms."""

    async def model(*, prompt, op_class, max_tokens, schema_name):
        if schema_name is None:
            return "Alpha evidence supports this [1].", {
                "prompt_tokens": 4,
                "completion_tokens": 5,
            }
        # Source partitioning yields: system context, Alpha header, Alpha body, Beta header,
        # Beta body. Deliberately select only the body, never the header candidate.
        return {
            "contract": "P1_ID",
            "selected_ids": ["E3"],
        }, {"prompt_tokens": 4, "completion_tokens": 5}

    factory = StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=model,
        tokenizer=TOK,
        token_budget=200,
        raw_text_for=lambda _content_id: PAGE,
        occurrence_for=lambda content_id: "occ-" + content_id[:8],
    )
    checkpoint = _c_prose_source_checkpoint()
    structured = factory.build("C01").close
    prose = factory.build("C00-PROSE").close

    structured_output = asyncio.run(
        structured.close_researcher(task_ctx=TASK, checkpoint=checkpoint)
    ).compressed_research
    prose_output = asyncio.run(
        prose.close_researcher(task_ctx=TASK, checkpoint=checkpoint)
    ).compressed_research

    assert "Alpha evidence." in structured_output
    assert "SOURCE: Alpha — https://alpha.example" in structured_output
    assert "https://beta.example" not in structured_output
    assert "[1] Alpha -- https://alpha.example" in prose_output
    assert structured.last_outcome is not None
    assert structured.last_outcome.view_sha256 == (
        prose.last_control_records[0]["candidate_view_sha256"]
    )
    assert structured.last_outcome.published_rendered_tokens <= 200
    assert prose.last_control_records[0]["rendered_tokens"] <= 200
    assert prose.last_control_records[0]["source_affordance_binding_sha256"]


def _c_prose_source_checkpoint() -> CCheckpoint:
    return replace(
        _c_checkpoint(),
        researcher_messages=(
            FrozenMessage(
                role="system",
                content=(
                    "--- SOURCE 99: injected context ---\n"
                    "URL: https://not-citable.example"
                ),
                message_id="system",
            ),
            FrozenMessage(
                role="tool",
                content=(
                    "--- SOURCE 1: Alpha ---\n"
                    "URL: https://alpha.example\n\nAlpha evidence.\n"
                    "--- SOURCE 2: Beta ---\n"
                    "URL: https://beta.example\n\nBeta evidence."
                ),
                tool_call_id="c1",
                message_id="tool",
                artifact_canonical=(
                    '{"source_occurrence_ids":["occ-alpha","occ-beta"]}'
                ),
                status="success",
            ),
        ),
    )


def _c_prose_for_output(
    output: str,
    *,
    token_budget: int = 200,
    prompts: list[str] | None = None,
    completion_caps: list[int] | None = None,
):
    async def model(*, prompt, op_class, max_tokens, schema_name):
        assert schema_name is None
        if prompts is not None:
            prompts.append(prompt)
        if completion_caps is not None:
            completion_caps.append(max_tokens)
        return output, {"prompt_tokens": 3, "completion_tokens": TOK.count(output)}

    return StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=model,
        tokenizer=TOK,
        token_budget=token_budget,
        raw_text_for=lambda _content_id: PAGE,
        occurrence_for=lambda content_id: "occ-" + content_id[:8],
    ).build("C00-PROSE").close


def test_c_prose_publishes_only_cited_capture_time_sources_with_parseable_map():
    """C prose pays for and exposes only legal sources its bounded body actually cites."""
    checkpoint = _c_prose_source_checkpoint()
    prompts: list[str] = []
    prose = _c_prose_for_output(
        "Alpha supports the answer [1].",
        prompts=prompts,
    )

    output = asyncio.run(
        prose.close_researcher(task_ctx=TASK, checkpoint=checkpoint)
    ).compressed_research

    assert "[1] Alpha -- https://alpha.example" in output
    assert "https://beta.example" not in output
    assert "https://not-citable.example" not in output
    assert parse_citation_map(output) == {"1": "https://alpha.example"}

    source_map = prompts[0].split("CITABLE SOURCE MAP:\n", 1)[1].split(
        "\n\nHARD LIMIT:", 1
    )[0]
    assert source_map == (
        "[1] Alpha -- https://alpha.example\n"
        "[2] Beta -- https://beta.example"
    )
    assert "https://not-citable.example" not in source_map

    control = prose.last_control_records[0]
    assert control["visible_source_affordance_count"] == 2
    assert control["offered_source_occurrence_ids"] == [
        "occ-alpha", "occ-beta"
    ]
    assert control["used_source_occurrence_ids"] == ["occ-alpha"]
    assert control["used_citation_labels"] == ["1"]
    assert control["citation_affordance_policy"] == (
        "CAPTURE_TIME_BYTE_RANGE_SOURCE_REGISTRY_V3"
    )
    assert control["visible_source_affordances_sha256"] != \
        control["used_source_map_sha256"]
    assert TOK.count(output) <= prose.config.token_budget


def test_c_prose_small_budget_has_no_all_source_tax_and_drops_unused_urls():
    """Many offered sources do not consume output budget unless the bounded body uses them."""
    checkpoint = replace(
        _c_prose_source_checkpoint(),
        researcher_messages=(
            FrozenMessage(
                role="tool",
                content=(
                    "--- SOURCE 1: Alpha ---\n"
                    "URL: https://alpha.example\n\nAlpha evidence.\n"
                    "--- SOURCE 2: Beta ---\n"
                    "URL: https://beta.example\n\nBeta evidence.\n"
                    "--- SOURCE 3: Gamma ---\n"
                    "URL: https://gamma.example\n\nGamma evidence."
                ),
                tool_call_id="c1",
                message_id="tool",
                artifact_canonical=(
                    '{"source_occurrence_ids":'
                    '["occ-alpha","occ-beta","occ-gamma"]}'
                ),
                status="success",
            ),
        ),
    )
    caps: list[int] = []
    prose = _c_prose_for_output(
        "Alpha evidence [1]",
        token_budget=10,
        completion_caps=caps,
    )

    output = asyncio.run(
        prose.close_researcher(task_ctx=TASK, checkpoint=checkpoint)
    ).compressed_research

    assert parse_citation_map(output) == {"1": "https://alpha.example"}
    assert "https://beta.example" not in output
    assert "https://gamma.example" not in output
    assert TOK.count(output) <= 10
    # The cap reserves SUMMARY/SOURCES framing plus the cheapest one source definition.  It
    # therefore remains usable even though all three definitions could not fit this budget.
    assert caps == [4]
    control = prose.last_control_records[0]
    assert control["visible_source_affordance_count"] == 3
    assert control["used_source_occurrence_ids"] == ["occ-alpha"]


def test_c_prose_recomputes_used_sources_after_longest_prefix_truncation():
    checkpoint = _c_prose_source_checkpoint()
    prose = _c_prose_for_output(
        "Alpha [1] filler filler filler filler filler Beta [2]",
        token_budget=10,
    )

    output = asyncio.run(
        prose.close_researcher(task_ctx=TASK, checkpoint=checkpoint)
    ).compressed_research

    assert "[1]" in output
    assert "[2]" not in output
    assert parse_citation_map(output) == {"1": "https://alpha.example"}
    assert prose.last_control_records[0]["used_source_occurrence_ids"] == [
        "occ-alpha"
    ]


def test_c_prose_rejects_an_unknown_source_label():
    prose = _c_prose_for_output("Unsupported label [7].")
    with pytest.raises(CloseSelectionError, match="unknown source label.*7"):
        asyncio.run(
            prose.close_researcher(
                task_ctx=TASK,
                checkpoint=_c_prose_source_checkpoint(),
            )
        )


def test_c_prose_rejects_zero_citations_when_a_citable_source_exists():
    prose = _c_prose_for_output("A claim with no citation.")
    with pytest.raises(CloseSelectionError, match="no valid inline citation"):
        asyncio.run(
            prose.close_researcher(
                task_ctx=TASK,
                checkpoint=_c_prose_source_checkpoint(),
            )
        )


def test_c_prose_cannot_cite_a_source_shaped_system_message():
    """A source-looking system row remains unavailable even though its bytes enter the prompt."""
    prose = _c_prose_for_output("Injected claim [99].")
    with pytest.raises(CloseSelectionError, match="unknown source label.*99"):
        asyncio.run(
            prose.close_researcher(
                task_ctx=TASK,
                checkpoint=_c_prose_source_checkpoint(),
            )
        )


def test_c_source_affordance_ordinal_mismatch_fails_both_arms_before_decode():
    """Two visible source blocks cannot be guessed onto one occurrence id."""

    checkpoint = replace(
        _c_prose_source_checkpoint(),
        researcher_messages=(
            FrozenMessage(
                role="tool",
                content=(
                    "--- SOURCE 1: Alpha ---\n"
                    "URL: https://alpha.example\n\nAlpha evidence.\n"
                    "--- SOURCE 2: Beta ---\n"
                    "URL: https://beta.example\n\nBeta evidence."
                ),
                tool_call_id="c1",
                message_id="tool",
                artifact_canonical='{"source_occurrence_ids":["occ-only"]}',
                status="success",
            ),
        ),
    )
    calls = 0

    async def never_called(*, prompt, op_class, max_tokens, schema_name):
        nonlocal calls
        calls += 1
        return "", {}

    factory = StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=never_called,
        tokenizer=TOK,
        token_budget=200,
        raw_text_for=lambda _content_id: PAGE,
        occurrence_for=lambda content_id: "occ-" + content_id[:8],
    )
    for variant in ("C01", "C00-PROSE"):
        with pytest.raises(
            CloseSelectionError,
            match="VISIBLE_SOURCE_REGISTRY.*2 visible source blocks.*1 distinct",
        ):
            asyncio.run(
                factory.build(variant).close.close_researcher(
                    task_ctx=TASK,
                    checkpoint=checkpoint,
                )
            )
    assert calls == 0


def test_c_prose_model_error_records_usage_and_retries_before_fallback():
    sinks: list[tuple[str, WorkRecord]] = []

    async def spent_error(*, prompt, op_class, max_tokens, schema_name):
        raise SelectorResponseError(
            "provider returned unusable prose",
            usage={
                "prompt_tokens": 17,
                "completion_tokens": 9,
                "retries": 2,
            },
        )

    prose = StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=spent_error,
        tokenizer=TOK,
        token_budget=200,
        raw_text_for=lambda _content_id: PAGE,
        occurrence_for=lambda content_id: "occ-" + content_id[:8],
        work_sink=lambda variant, work: sinks.append((variant, work)),
    ).build("C00-PROSE").close

    with pytest.raises(CloseSelectionError, match="provider returned unusable prose"):
        asyncio.run(
            prose.close_researcher(
                task_ctx=TASK,
                checkpoint=_c_prose_source_checkpoint(),
            )
        )

    assert prose.last_work.selector_calls == 1
    assert prose.last_work.prompt_tokens == 17
    assert prose.last_work.completion_tokens == 9
    assert prose.last_work.retries == 2
    assert prose.last_work_incomplete is True
    assert sinks == [("C00-PROSE", prose.last_work)]


def test_c_prose_cancellation_records_partial_usage_then_propagates():
    sinks: list[tuple[str, WorkRecord]] = []

    async def cancelled(*, prompt, op_class, max_tokens, schema_name):
        error = asyncio.CancelledError("cancelled upstream")
        error.usage = {
            "prompt_tokens": 13,
            "completion_tokens": 3,
            "retries": 1,
        }
        raise error

    prose = StrategyFactory(
        registry=load_registry(CONFIGS),
        model_call=cancelled,
        tokenizer=TOK,
        token_budget=200,
        raw_text_for=lambda _content_id: PAGE,
        occurrence_for=lambda content_id: "occ-" + content_id[:8],
        work_sink=lambda variant, work: sinks.append((variant, work)),
    ).build("C00-PROSE").close

    with pytest.raises(asyncio.CancelledError, match="cancelled upstream"):
        asyncio.run(
            prose.close_researcher(
                task_ctx=TASK,
                checkpoint=_c_prose_source_checkpoint(),
            )
        )

    assert prose.last_work.selector_calls == 1
    assert prose.last_work.prompt_tokens == 13
    assert prose.last_work.completion_tokens == 3
    assert prose.last_work.retries == 1
    assert prose.last_work_incomplete is True
    assert sinks == [("C00-PROSE", prose.last_work)]


def test_c_prose_ignores_invalid_suffix_only_after_budget_prefix_is_frozen():
    prose = _c_prose_for_output(
        "Alpha [1] filler filler filler filler unknown [7]",
        token_budget=10,
    )

    output = asyncio.run(
        prose.close_researcher(
            task_ctx=TASK,
            checkpoint=_c_prose_source_checkpoint(),
        )
    ).compressed_research

    assert "[1]" in output
    assert "[7]" not in output
    normalization = prose.last_control_records[0]["normalization"]
    assert normalization["raw_semantically_valid"] is False
    assert normalization["published_policy_valid"] is True
    assert normalization["semantic_repair_applied"] is False
    assert normalization["invalid_raw_suffix_outside_policy_output"] is True
    assert (
        "INVALID_RAW_SUFFIX_OUTSIDE_POLICY_OUTPUT"
        in normalization["normalization_reasons"]
    )
    assert normalization["raw_contract_adherence_sensitivity_excludes"] is True


def test_c_prose_invalid_url_inside_budgeted_prefix_fails_without_shortening_again():
    prose = _c_prose_for_output(
        "Alpha https://invented.example supports this [1].",
        token_budget=200,
    )

    with pytest.raises(CloseSelectionError, match="body emitted a URL"):
        asyncio.run(
            prose.close_researcher(
                task_ctx=TASK,
                checkpoint=_c_prose_source_checkpoint(),
            )
        )

    record = prose.last_control_records[0]
    assert record["publication_status"] == "REJECTED"
    assert record["normalization"]["published_policy_valid"] is False
    assert (
        record["normalization"]["published_validation"]["body_url_count"]
        == 1
    )


def test_contract_is_locked_even_when_model_returns_another_valid_contract():
    """A union-schema response used to let H03 silently execute H02."""
    page = _factory().build("H03").page
    with pytest.raises(PageSelectionError, match="CONTRACT.*locked to 'P1_TYPED'"):
        asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))
    assert page.last_outcomes[0].contract == "P1_TYPED"
    assert page.last_outcomes[0].failure.reason == "CONTRACT"


def test_no_raw_snippet_survives_beside_selected_raw_evidence():
    """P0 passes a result's snippet through when raw_content is absent; P1 must not drop it."""
    base = _h_checkpoint()
    cp = HCheckpoint(
        task_id=base.task_id, researcher_id=base.researcher_id,
        assistant_turn_index=base.assistant_turn_index,
        assistant_message=base.assistant_message,
        sibling_tool_calls=base.sibling_tool_calls,
        search_result_sets=(("c1", (
            base.search_result_sets[0][1][0],
            VendorVisibleResult(
                vendor_visible_order=1, url="https://snippet.example", title="Snippet Source",
                snippet="The only vendor-visible fallback sentence.", raw_content_id=None,
            ),
        )),),
        non_search_outputs=base.non_search_outputs,
        researcher_state_hash=base.researcher_state_hash,
        sampling=base.sampling,
    )
    out = asyncio.run(
        _factory().build("H02").page.transform_tool_batch(task_ctx=TASK, checkpoint=cp)
    )
    assert "The only vendor-visible fallback sentence." in out[0].content
    assert "Snippet Source" in out[0].content
    assert "https://snippet.example" in out[0].content


def test_identical_bytes_at_two_urls_keep_two_occurrence_lineages():
    """A content digest is not a citation occurrence; duplicate bytes must not pick first URL."""
    base = _h_checkpoint()
    same_content_id = "a" * 64
    cp = HCheckpoint(
        task_id=base.task_id, researcher_id=base.researcher_id,
        assistant_turn_index=base.assistant_turn_index,
        assistant_message=base.assistant_message,
        sibling_tool_calls=base.sibling_tool_calls,
        search_result_sets=(("c1", (
            VendorVisibleResult(
                0, "https://a.example", "A", "a", same_content_id,
                source_occurrence_id="occ-url-a",
            ),
            VendorVisibleResult(
                1, "https://b.example", "B", "b", same_content_id,
                source_occurrence_id="occ-url-b",
            ),
        )),),
        non_search_outputs=base.non_search_outputs,
        researcher_state_hash=base.researcher_state_hash,
        sampling=base.sampling,
    )
    page = _factory().build("H02").page
    asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=cp))
    assert len(page.last_outcomes) == 2
    assert page.last_outcomes[0].published_span_ids != \
        page.last_outcomes[1].published_span_ids


class _PickFirstSelector:
    def __init__(self) -> None:
        self.calls = []

    async def select(self, *, task_ctx, view):
        self.calls.append(tuple((c.span_id, c.text) for c in view.candidates))
        return (
            {"contract": "P1_ID", "selected_ids": ["E1"]},
            WorkRecord(selector_calls=1, completion_tokens=1),
        )


def test_hierarchical_reduce_sees_only_map_survivors():
    """The local pass must constrain reduce, not merely add an unused model call."""
    pages = {
        "a" * 64: "alpha one two three four five six seven eight",
        "b" * 64: "beta one two three four five six seven eight",
    }
    local = _PickFirstSelector()
    global_selector = _PickFirstSelector()
    strategy = PageSelectionStrategy(
        PageStrategyConfig(
            variant_id="H-test", chunker="fixed_token_v1", scope="hierarchical",
            contract="P1_ID", aggregation="stable_union_v1", token_budget=200,
            chunk_max_tokens=3,
        ),
        selector=local, global_selector=global_selector, tokenizer=TOK,
        raw_text_for=pages.__getitem__, occurrence_for=lambda cid: "occ-" + cid[0],
    )
    base = _h_checkpoint()
    cp = HCheckpoint(
        task_id=base.task_id, researcher_id=base.researcher_id,
        assistant_turn_index=base.assistant_turn_index,
        assistant_message=base.assistant_message,
        sibling_tool_calls=base.sibling_tool_calls,
        search_result_sets=(("c1", (
            VendorVisibleResult(0, "https://a.example", "A", "a", "a" * 64),
            VendorVisibleResult(1, "https://b.example", "B", "b", "b" * 64),
        )),),
        non_search_outputs=base.non_search_outputs,
        researcher_state_hash=base.researcher_state_hash,
        sampling=base.sampling,
    )
    asyncio.run(strategy.transform_tool_batch(task_ctx=TASK, checkpoint=cp))

    assert len(local.calls) == 2
    assert all(len(call) > 1 for call in local.calls)
    assert len(global_selector.calls) == 1
    # One published map survivor per page, and no rejected map candidate reappears at reduce.
    assert len(global_selector.calls[0]) == 2
    local_survivors = {
        sid for outcome in strategy.last_outcomes if outcome.stage == "local"
        for sid in outcome.published_span_ids
    }
    assert {sid for sid, _ in global_selector.calls[0]} == local_survivors
    assert {
        outcome.offered_source_occurrence_ids
        for outcome in strategy.last_outcomes if outcome.stage == "local"
    } == {("occ-a",), ("occ-b",)}
    assert strategy.last_outcomes[-1].offered_source_occurrence_ids == (
        "occ-a", "occ-b")
    assert strategy.last_outcomes[-1].stage == "global"
    assert strategy.last_outcomes[-1].checkpoint_hash == cp.digest


class _SpentInvalidJsonSelector:
    async def select(self, *, task_ctx, view):
        raise SelectorResponseError(
            "selector returned invalid JSON",
            usage={"prompt_tokens": 71, "completion_tokens": 19, "retries": 1},
        )


def test_post_decode_failure_keeps_selector_work():
    page = PageSelectionStrategy(
        PageStrategyConfig(
            variant_id="H-error", chunker="markdown_structure_v1", scope="per_page",
            contract="P1_ID", aggregation="stable_union_v1", token_budget=200,
        ),
        selector=_SpentInvalidJsonSelector(), tokenizer=TOK,
        raw_text_for=lambda cid: PAGE, occurrence_for=lambda cid: "occ",
    )
    with pytest.raises(PageSelectionError, match="SELECTOR_ERROR"):
        asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))
    assert page.last_work.selector_calls == 1
    assert page.last_work.prompt_tokens == 71
    assert page.last_work.completion_tokens == 19
    assert page.last_work.retries == 1
    attempt = page.last_outcomes[-1]
    assert attempt.selector_attempted is True
    assert attempt.normalization["rejected_reason"] == "failure_before_parse"


class _LongBridgeSelector:
    async def select(self, *, task_ctx, view):
        return ({
            "contract": "P1_BRIDGE",
            "selections": [{"span_id": "E1", "role": "support"}],
            "bridges": [{
                "text": "This evidence therefore supports the answer in a bounded connection",
                "evidence_ids": ["E1"],
            }],
        }, WorkRecord(selector_calls=1, completion_tokens=12))


def test_bridge_caps_from_variant_reach_publish_preflight():
    page = PageSelectionStrategy(
        PageStrategyConfig(
            variant_id="H-bridge", chunker="markdown_structure_v1", scope="per_page",
            contract="P1_BRIDGE", aggregation="stable_union_v1", token_budget=200,
            bridge_token_cap_each=3, bridge_token_cap_total=4,
        ),
        selector=_LongBridgeSelector(), tokenizer=TOK,
        raw_text_for=lambda cid: PAGE, occurrence_for=lambda cid: "occ",
    )
    with pytest.raises(PageSelectionError, match="PREFLIGHT.*per-bridge token cap"):
        asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))


def test_the_short_prose_control_is_held_to_the_same_token_budget():
    """If the control could exceed the budget it would no longer be token-matched, and the
    comparison it exists to support would be void."""
    completion_caps = []

    async def long_prose(*, prompt, op_class, max_tokens, schema_name):
        completion_caps.append(max_tokens)
        return (
            " ".join(f"word{i}" for i in range(max_tokens)),
            {"prompt_tokens": 5, "completion_tokens": max_tokens},
        )

    f = StrategyFactory(
        registry=load_registry(CONFIGS), model_call=long_prose, tokenizer=TOK,
        token_budget=16, raw_text_for=lambda cid: PAGE,
        occurrence_for=lambda cid: "occ",
    )
    page = f.build("H00-PROSE").page
    out = asyncio.run(page.transform_tool_batch(task_ctx=TASK, checkpoint=_h_checkpoint()))
    # H02's preflight cap is the inner rendered selection; both arms then receive the same
    # outer ``Selected evidence`` wrapper. The inner count includes the immutable per-source
    # title/URL wrapper, as H02's renderer includes its own citation metadata.
    assert page.last_rendered_token_counts[0] <= 16
    assert "https://a.example" in out[0].content
    assert page.last_published_token_counts == (TOK.count(out[0].content),)
    assert completion_caps == [
        page.last_control_records[0]["completion_token_cap"]
    ]
    assert 0 < completion_caps[0] < page.config.token_budget


def test_the_visible_view_keeps_model_reasoning_out_of_evidence():
    """Selecting a span over AI reasoning must not make it TOOL_EVIDENCE.

    That line is the difference between C_VISIBLE reading what the compressor read and
    C_VISIBLE manufacturing provenance for a model's own words.
    """
    view = build_visible_view((
        FrozenMessage(role="ai", content="I think the answer is X", message_id="m1"),
        FrozenMessage(role="tool", content="SOURCE 1: evidence", tool_call_id="c1",
                      message_id="m2",
                      artifact_canonical='{"source_occurrence_ids":["occ-source-1"]}'),
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
    assert view.message_segments[0]["kind"] == "TOOL_UNATTRIBUTED_CONTEXT"
