"""The H unit is the gather batch, and `whole_batch` is the only scope that says so.

`HC_MECHANISM_v1.md` has always specified "One whole-batch selector request covering every page
in the batch", but every shipped Freeze-1 H variant carried `scope: per_page` and
`PageSelectionStrategy` fans out twice -- over siblings, then over pages within each sibling. So
the arms issued one request *per page*, each carrying its own `selected_token_budget`, and at a
mean 9.07 pages per batch the effective batch ceiling was nine times the frozen one.

Nothing caught it. The contract documents were written after the implementation and describe
themselves as pinning "the mechanism as implemented"; the conformance obligations test form
purity, atomicity and determinism, and none of them counts selector requests. These tests are
that missing count, plus the two properties that make the fix safe to adopt: at one sibling per
batch the handle domain does not move, and a batch that would need two requests is refused
rather than quietly served.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from shapeflow.evidence.model_tokenizer import WhitespaceTokenizer
from shapeflow.odr.checkpoints import (
    FrozenMessage,
    HCheckpoint,
    SamplingEnvelope,
    VendorVisibleResult,
)
from shapeflow.strategies.page_h import (
    PageSelectionError,
    PageSelectionStrategy,
    PageStrategyConfig,
)
from shapeflow.strategies.pipeline import WorkRecord

SAMPLING = SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=32, seed=1)

#: Long enough that the markdown chunker emits several spans per page, so a whole-batch view is
#: visibly the union of the pages rather than one page that happens to be first.
PAGE_TEXT = "\n\n".join(f"## heading {n}\n\nbody sentence {n} " + "filler " * 40
                        for n in range(4))


def _result(order: int, occurrence_id: str) -> VendorVisibleResult:
    return VendorVisibleResult(
        vendor_visible_order=order,
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


class RecordingSelector:
    """Selects the first candidate of every view and remembers each view it was shown."""

    def __init__(self) -> None:
        self.views: list = []

    async def select(self, *, task_ctx, view):
        self.views.append(view)
        labels = [view.candidates[0].label] if view.candidates else []
        return ({"contract": "P1_ID", "selected_ids": labels}, WorkRecord(selector_calls=1))


def _strategy(scope: str, selector) -> PageSelectionStrategy:
    return PageSelectionStrategy(
        PageStrategyConfig(
            variant_id=f"T-{scope}",
            chunker="markdown_structure_v1",
            scope=scope,
            contract="P1_ID",
            aggregation="stable_union_v1",
            token_budget=512,
        ),
        selector=selector,
        tokenizer=WhitespaceTokenizer(),
        raw_text_for=lambda content_id: PAGE_TEXT,
        occurrence_for=lambda content_id: content_id.replace("content-", ""),
    )


def _ctx():
    return SimpleNamespace(research_topic="a topic", selected_token_budget=512)


def _one_sibling(pages: int) -> HCheckpoint:
    return _checkpoint(
        ("call-1", tuple(_result(i, f"o{i}") for i in range(pages))),
    )


@pytest.mark.asyncio
async def test_a_whole_batch_arm_issues_exactly_one_selector_request_per_batch():
    """The count *is* the treatment. Freeze-1's H arms issued one per page."""
    selector = RecordingSelector()
    await _strategy("whole_batch", selector).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=5))

    assert len(selector.views) == 1, (
        f"whole_batch made {len(selector.views)} selector requests for a 5-page batch; "
        "the contract specifies one")


@pytest.mark.asyncio
async def test_per_page_issues_one_request_per_page_which_is_the_defect_being_fixed():
    """The comparison that makes the previous test mean something.

    Kept as a live assertion rather than a comment: `per_page` remains in the registry so the
    Freeze-1 arms stay reproducible, and if it ever stopped fanning out, every republished
    per-page number would silently describe a different mechanism.
    """
    selector = RecordingSelector()
    await _strategy("per_page", selector).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=5))

    assert len(selector.views) == 5


@pytest.mark.asyncio
async def test_the_whole_batch_view_carries_every_pages_candidates():
    """One request is only the contract's shape if that request can see the whole batch.

    A single call over one page's candidates would satisfy the request count and still be the
    per-page mechanism.
    """
    selector = RecordingSelector()
    await _strategy("whole_batch", selector).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=5))

    per_page = RecordingSelector()
    await _strategy("per_page", per_page).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=5))

    whole = {c.span_id for c in selector.views[0].candidates}
    split = {c.span_id for view in per_page.views for c in view.candidates}
    assert whole == split, "the whole-batch view is not the union of the per-page views"


def _stated_budgets(selector: "RecordingSelector") -> list[int]:
    """The rendered-token budget each selector call was actually held to.

    Read out of the prompt rather than off the config, because the budget is enforced once per
    `run_selection` -- so the ceiling a *batch* faces is this list summed, and no configured
    number anywhere states that sum. That is exactly why the per-page drift was invisible.
    """
    budgets = []
    for view in selector.views:
        for line in view.prompt_bytes.decode("utf-8").splitlines():
            if line.startswith("TOKEN BUDGET"):
                budgets.append(int(line.rsplit(":", 1)[1].strip()))
                break
    return budgets


@pytest.mark.asyncio
async def test_the_rendered_token_budget_covers_the_batch_not_each_page():
    """Five pages under one 512 budget, not five 512 budgets.

    This is the arithmetic the drift hid: `selected_token_budget` is applied once per selector
    call, so multiplying the calls multiplied the batch's real ceiling while every configured
    number stayed at 512. Measured over the recorded checkpoints the mean batch is 9.07 pages,
    which made the effective H ceiling ~4,644 tokens.
    """
    whole = RecordingSelector()
    await _strategy("whole_batch", whole).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=5))

    per_page = RecordingSelector()
    await _strategy("per_page", per_page).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=5))

    assert _stated_budgets(whole) == [512]
    assert sum(_stated_budgets(per_page)) == 5 * 512, (
        "per_page is expected to multiply the ceiling by the page count; if it no longer does, "
        "the Freeze-1 H numbers describe something other than what was published")


@pytest.mark.asyncio
async def test_a_multi_sibling_batch_is_refused_rather_than_split():
    """Two siblings cannot be one request, so the arm stops instead of becoming per-sibling.

    Refusing routes the batch down the ordinary P1-failure path, where end-to-end runs fall the
    whole batch back to P0 and component trials record the failure. Serving it would restore the
    exact defect under the fixed name.
    """
    selector = RecordingSelector()
    checkpoint = _checkpoint(
        ("call-1", (_result(0, "o0"), _result(1, "o1"))),
        ("call-2", (_result(0, "o2"),)),
    )

    with pytest.raises(PageSelectionError) as caught:
        await _strategy("whole_batch", selector).transform_tool_batch(
            task_ctx=_ctx(), checkpoint=checkpoint)

    assert caught.value.failure.reason == "WHOLE_BATCH_MULTI_SIBLING"
    assert not selector.views, "the arm spent a selector request before refusing"


@pytest.mark.asyncio
async def test_whole_batch_and_per_tool_call_agree_on_handles_at_one_sibling():
    """`whole_batch` must not move the publication handle domain.

    The two scopes coincide at one sibling per batch, which is what every recorded checkpoint
    shows. Asserting the handle map is identical is what lets `whole_batch` be adopted without
    re-minting the handle radices or invalidating a published handle's meaning -- and it is why
    `per_tool_call` is extended rather than renamed.
    """
    whole = RecordingSelector()
    await _strategy("whole_batch", whole).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=3))

    per_call = RecordingSelector()
    await _strategy("per_tool_call", per_call).transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=3))

    assert whole.views[0].publication_handle_map == per_call.views[0].publication_handle_map
    assert whole.views[0].publication_map_sha256 == per_call.views[0].publication_map_sha256


def test_a_whole_batch_arm_is_charged_to_its_own_op_class():
    """The op class follows the unit, not the boundary.

    Recording a whole-batch selector under ``PAGE_P1_SELECTOR_LOCAL`` would sum one-request-per-
    batch work with one-request-per-page work in the same ledger line, and the saving this
    rebuild exists to measure would be averaged with the saving of the mechanism it replaces.
    This repo has already shipped that exact bug once, with the SHORT_PROSE aliases pointing at
    the structured-selector op classes.
    """
    from shapeflow.campaign.selector_client import ALIAS_BY_OP, STRUCTURED_SELECTOR_OPS
    from shapeflow.runtime.provider_server import DEFAULT_MODEL_ALIASES
    from shapeflow.runtime.request_tags import TREATMENT_OPS, OpClass
    from shapeflow.strategies.factory import StrategyFactory, load_registry

    registry = load_registry(Path(__file__).resolve().parents[2] / "configs")
    factory = StrategyFactory(
        registry=registry, model_call=lambda **_kw: None, tokenizer=WhitespaceTokenizer())

    batch_arm = factory.build("HW02")
    per_page_arm = factory.build("H02")

    assert batch_arm.page._selector._op_class == "PAGE_P1_SELECTOR_BATCH"
    assert per_page_arm.page._selector._op_class == "PAGE_P1_SELECTOR_LOCAL"

    # And the class must be routable end to end, or the request is charged to nothing.
    alias = ALIAS_BY_OP[OpClass.PAGE_P1_SELECTOR_BATCH.value]
    assert DEFAULT_MODEL_ALIASES[alias] is OpClass.PAGE_P1_SELECTOR_BATCH
    assert OpClass.PAGE_P1_SELECTOR_BATCH in TREATMENT_OPS
    assert OpClass.PAGE_P1_SELECTOR_BATCH.value in STRUCTURED_SELECTOR_OPS


def test_only_an_llm_arm_without_admission_carries_a_window_ceiling():
    """The ceiling refuses a prompt the engine would reject. It belongs to exactly one shape.

    A CPU arm has no context window at all, so giving CPU-FULL a ceiling would mark 61.9% of
    batches PROMPT_INFEASIBLE for the one arm that is feasible on all of them -- the precise
    opposite of what it measures. An arm with admission cannot exceed the window by
    construction, so a ceiling there could only ever fire on a bug.
    """
    from shapeflow.strategies.factory import StrategyFactory, load_registry

    factory = StrategyFactory(
        registry=load_registry(Path(__file__).resolve().parents[2] / "configs"),
        model_call=lambda **_kw: None, tokenizer=WhitespaceTokenizer(),
        prompt_budget=20_000, prompt_window_ceiling=32_000,
    )
    ceilings = {
        arm: factory.build(arm).page.config.prompt_window_ceiling
        for arm in ("CPU-FULL", "CPU-PROMPTVIEW", "LLM-PROMPTVIEW", "LLM-FULL")
    }

    assert ceilings == {
        "CPU-FULL": 0, "CPU-PROMPTVIEW": 0, "LLM-PROMPTVIEW": 0, "LLM-FULL": 32_000}


def test_the_four_matched_arms_differ_in_exactly_one_thing_each():
    """The shootout's two contrasts only mean something if the arms are matched.

    CPU-FULL vs CPU-PROMPTVIEW must differ only in admission (that difference *is* the price of
    pruning); CPU-PROMPTVIEW vs LLM-PROMPTVIEW only in who ranks (that difference *is* the
    model's value). Any second difference makes both numbers unattributable.
    """
    from shapeflow.strategies.factory import load_registry

    registry = load_registry(Path(__file__).resolve().parents[2] / "configs")
    fields = ("node", "chunker", "scope", "contract", "aggregation", "close_mode",
              "publication_path", "output_representation", "selector_backend",
              "prompt_admission")

    def spec(arm):
        return {f: getattr(registry[arm], f) for f in fields}

    full, view = spec("CPU-FULL"), spec("CPU-PROMPTVIEW")
    assert [f for f in fields if full[f] != view[f]] == ["prompt_admission"]

    cpu, llm = spec("CPU-PROMPTVIEW"), spec("LLM-PROMPTVIEW")
    assert [f for f in fields if cpu[f] != llm[f]] == ["selector_backend"]


@pytest.mark.asyncio
async def test_whole_batch_prose_summarises_the_batch_once_and_keeps_every_source_citable():
    """The matched control must be one request under one cap, and still cite every page.

    Per-page prose names its one source in the header, which is right for a per-page summary.
    A batch summary that kept that shape would drop provenance for every page but the first,
    making the control a straw man: the downstream writer can cite P0 and structured P1 but
    would not be able to cite this. The source map, like the C-side close document, is
    immutable framing charged against the budget before a token is decoded.
    """
    from shapeflow.strategies.prose import ProsePageStrategy

    class OneShotProse:
        def __init__(self) -> None:
            self.calls = 0
            self.caps: list[int] = []

        def prompt_for(self, *, task_ctx, view, token_budget, source_entries=None):
            return "prompt"

        async def summarize(self, *, task_ctx, view, token_budget,
                            max_completion_tokens=None, source_entries=None):
            self.calls += 1
            self.caps.append(int(max_completion_tokens))
            return "a batch summary", WorkRecord(selector_calls=1)

    selector = OneShotProse()
    strategy = ProsePageStrategy(
        config=PageStrategyConfig(
            variant_id="SHORTPROSE-PROMPTVIEW", chunker="markdown_structure_v1",
            scope="whole_batch", contract="SHORT_PROSE", aggregation="stable_union_v1",
            token_budget=512),
        selector=selector, tokenizer=WhitespaceTokenizer(),
        raw_text_for=lambda _cid: PAGE_TEXT,
        occurrence_for=lambda cid: cid.replace("content-", ""),
    )

    observations = await strategy.transform_tool_batch(
        task_ctx=_ctx(), checkpoint=_one_sibling(pages=4))

    assert selector.calls == 1, f"{selector.calls} prose requests for one gather batch"
    content = "".join(str(o.content) for o in observations)
    for page in range(4):
        assert f"https://o{page}.example" in content, (
            f"page {page} lost its provenance; the control cannot be cited for it")
    assert selector.caps[0] < 512, "the source map must be charged before decoding, not after"
    assert WhitespaceTokenizer().count(content) <= 512
