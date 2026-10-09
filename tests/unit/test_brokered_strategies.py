"""The agent side: price both forms, send nothing, wait, then run exactly the chosen one.

These drive ``BrokeredPageStrategy`` and ``BrokeredCloseStrategy`` with a scripted broker client
and counting doubles for both forms, so each test can say precisely which model work happened:
none while pricing, and afterwards only the decided form's.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from shapeflow.broker.client import BROKER, FAIL_CLOSED, Resolution
from shapeflow.campaign.selector_client import SelectorModelCall
from shapeflow.contracts.offer import Form
from shapeflow.evidence.model_tokenizer import WhitespaceTokenizer
from shapeflow.odr.adapter import DeferredPageBatch, reduce_tool_batch
from shapeflow.odr.checkpoints import (
    CCheckpoint,
    EvidenceManifest,
    FrozenMessage,
    HCheckpoint,
    SamplingEnvelope,
)
from shapeflow.runtime.provider_server import provider_prompt_sha256
from shapeflow.strategies.brokered import (
    BrokeredCloseStrategy,
    BrokeredPageStrategy,
    BrokerSettings,
)
from shapeflow.strategies.offer_pricing import VendorPrompts, render_qwen3_chatml
from shapeflow.strategies.page_h import (
    PageSelectionError,
    PageSelectionStrategy,
    PageStrategyConfig,
)
from shapeflow.strategies.selectors_async import LlmAsyncSelector

REPO = Path(__file__).resolve().parents[2]
SAMPLING = SamplingEnvelope(model="m", temperature=0.0, top_p=1.0, max_tokens=32, seed=1)
PAGE_TEXT = "\n\n".join(f"## heading {n}\n\nbody sentence {n} " + "filler " * 40
                        for n in range(4))
PROMPTS = VendorPrompts(
    summarize_webpage="Summarize this page ({date}):\n{webpage_content}",
    compress_system="Clean up the findings. Today is {date}.",
    compress_human="Please clean up these findings.",
    today=lambda: "Fri Oct 9, 2026",
)
TOK = WhitespaceTokenizer()


# --- doubles ---------------------------------------------------------------------------------


class ScriptedBroker:
    def __init__(self, form: Form = Form.P1, *, source: str = BROKER, reason: str = "",
                 before_return=None, raise_exc: BaseException | None = None) -> None:
        self.form, self.source, self.reason = form, source, reason
        self.before_return = before_return
        self.raise_exc = raise_exc
        self.offers: list = []
        self.acks: list = []

    async def resolve(self, offer):
        self.offers.append(offer)
        if self.before_return is not None:
            self.before_return()
        if self.raise_exc is not None:
            raise self.raise_exc
        return Resolution(form=self.form, source=self.source, reason=self.reason,
                          ticket={"form": self.form.value})

    async def ack(self, item_id, outcome):
        self.acks.append((item_id, outcome))


class RecordingModelCall:
    """What the executing selector sends. Picks the first offered label; can be made to fail."""

    def __init__(self, *, fail: bool = False) -> None:
        self.prompts: list[str] = []
        self.fail = fail

    async def __call__(self, *, prompt, op_class, max_tokens, schema_name=None):
        self.prompts.append(prompt)
        if self.fail:
            raise RuntimeError("engine refused")
        label = re.search(r"^\[([^\]]+)\]", prompt, re.MULTILINE).group(1)
        return {"contract": "P1_ID", "selected_ids": [label]}, {"prompt_tokens": 1}


def _body_for():
    return SelectorModelCall(None, cell_token="c", repo=REPO, temperature=0.0, top_p=1.0,
                             max_completion_tokens=768, seed=1).request_body


def _page_strategy(model_call) -> PageSelectionStrategy:
    return PageSelectionStrategy(
        PageStrategyConfig(variant_id="T-WB", chunker="markdown_structure_v1",
                           scope="whole_batch", contract="P1_ID",
                           aggregation="stable_union_v1", token_budget=512),
        selector=LlmAsyncSelector(model_call, op_class="PAGE_P1_SELECTOR_BATCH",
                                  expected_contract="P1_ID"),
        tokenizer=TOK,
        raw_text_for=lambda content_id: PAGE_TEXT,
        occurrence_for=lambda content_id: content_id,
    )


def _settings(broker) -> BrokerSettings:
    return BrokerSettings(client=broker, cell_token="cell-1", tokenizer=TOK, prompts=PROMPTS,
                          p0_page_max_tokens=8192, p0_close_max_tokens=4096,
                          max_model_len=32768, deadline_ns=10**9)


def _brokered_page(broker, executor_call=None):
    executor_call = executor_call or RecordingModelCall()
    strategy = BrokeredPageStrategy(
        settings=_settings(broker), executor=_page_strategy(executor_call),
        probe_strategy_for=_page_strategy, body_for=_body_for())
    return strategy, executor_call


class CountingRender:
    def __init__(self, text: str) -> None:
        self.text, self.calls = text, 0

    async def __call__(self):
        self.calls += 1
        return self.text


def _deferred(call_id: str, pages: int, *, raw: bool = True) -> DeferredPageBatch:
    results = tuple({
        "url": f"https://{call_id}-{i}.example", "title": f"t{i}", "content": f"snip{i}",
        "raw_content": PAGE_TEXT if raw else None, "_shapeflow_occurrence_id": f"{call_id}-o{i}",
    } for i in range(pages))
    return DeferredPageBatch(tool_call_id=call_id, tool_name="tavily_search", results=results,
                             render_vendor=CountingRender(f"vendor summary of {call_id}"),
                             max_content_length=50_000)


def _checkpoint(batches) -> HCheckpoint:
    return HCheckpoint(
        task_id="task", researcher_id="r", assistant_turn_index=0,
        assistant_message=FrozenMessage(role="ai", content="search"), sibling_tool_calls=(),
        search_result_sets=tuple((b.tool_call_id, b.visible_results()) for b in batches),
        non_search_outputs=(), researcher_state_hash="state", sampling=SAMPLING)


def _ctx():
    return SimpleNamespace(research_topic="a topic", selected_token_budget=512)


async def _run_page(strategy, batches):
    strategy._deferred = {b.tool_call_id: b for b in batches}
    return await strategy.transform_tool_batch(task_ctx=_ctx(), checkpoint=_checkpoint(batches))


# --- H ---------------------------------------------------------------------------------------


async def test_pricing_sends_nothing_and_p1_then_runs_alone():
    broker = ScriptedBroker(Form.P1)
    strategy, executor_call = _brokered_page(broker)
    batch = _deferred("call-1", pages=3)
    observations = await _run_page(strategy, [batch])

    (offer,) = broker.offers
    assert len(offer.p0.requests) == 3 and offer.p1 is not None
    assert batch.render_vendor.calls == 0, "P0 ran although P1 was chosen"
    assert len(executor_call.prompts) == 1, "P1 must send exactly its one selector request"
    # The priced request is the one sent, byte for byte.
    sent = [{"role": "user", "content": executor_call.prompts[0]}]
    assert offer.p1.selector.prompt_sha256 == provider_prompt_sha256(sent)
    assert offer.p1.selector.est_uncached_prefill == TOK.count(render_qwen3_chatml(sent))
    assert observations[0].content.startswith("Selected evidence")
    assert broker.acks[-1][1]["outcome"] == "P1_OK"


async def test_p0_prices_vendors_own_summary_prompts():
    broker = ScriptedBroker(Form.P0)
    strategy, _ = _brokered_page(broker)
    batch = _deferred("call-1", pages=2)
    await _run_page(strategy, [batch])
    expected = PROMPTS.summarize_webpage.format(webpage_content=PAGE_TEXT, date=PROMPTS.today())
    sha = provider_prompt_sha256([{"role": "user", "content": expected}])
    assert [r.prompt_sha256 for r in broker.offers[0].p0.requests] == [sha, sha]
    assert {r.est_decode for r in broker.offers[0].p0.requests} == {8192}


async def test_p0_runs_vendor_and_never_the_selector():
    broker = ScriptedBroker(Form.P0)
    strategy, executor_call = _brokered_page(broker)
    batch = _deferred("call-1", pages=2)
    observations = await _run_page(strategy, [batch])
    assert batch.render_vendor.calls == 1 and executor_call.prompts == []
    assert observations[0].content == "vendor summary of call-1"
    assert strategy.last_outcomes == ()
    assert broker.acks[-1][1] == {"form": "P0", "source": BROKER, "outcome": "P0_OK",
                                  "detail": ""}


async def test_a_snippet_only_batch_has_no_choice_to_offer():
    broker = ScriptedBroker()
    strategy, _ = _brokered_page(broker)
    batch = _deferred("call-1", pages=2, raw=False)
    await _run_page(strategy, [batch])
    assert broker.offers == [] and batch.render_vendor.calls == 1


async def test_a_multi_sibling_batch_offers_p0_only():
    broker = ScriptedBroker(Form.P0)
    strategy, _ = _brokered_page(broker)
    await _run_page(strategy, [_deferred("call-1", 1), _deferred("call-2", 1)])
    (offer,) = broker.offers
    assert offer.p1 is None and len(offer.p0.requests) == 2


async def test_pricing_leaves_the_executing_instance_untouched():
    broker = ScriptedBroker(Form.P0)
    strategy, _ = _brokered_page(broker)
    before = dict(strategy._executor._publication_scope_digests)
    await _run_page(strategy, [_deferred("call-1", 2)])
    assert strategy._executor._publication_scope_digests == before
    assert strategy._executor.last_outcomes == []


async def test_a_new_batch_arriving_mid_wait_does_not_change_what_p0_runs():
    first, second = _deferred("call-1", 1), _deferred("call-9", 1)
    holder = {}
    broker = ScriptedBroker(Form.P0, before_return=lambda: setattr(
        holder["s"], "_deferred", {"call-9": second}))
    strategy, _ = _brokered_page(broker)
    holder["s"] = strategy
    observations = await _run_page(strategy, [first])
    assert first.render_vendor.calls == 1 and second.render_vendor.calls == 0
    assert observations[0].tool_call_id == "call-1"


async def test_a_p1_failure_falls_the_whole_batch_back_to_p0_and_tells_the_broker():
    broker = ScriptedBroker(Form.P1)
    strategy, _ = _brokered_page(broker, RecordingModelCall(fail=True))
    batch = _deferred("call-1", 2)
    strategy._deferred = {"call-1": batch}
    outcome = await reduce_tool_batch(
        strategy=strategy, task_ctx=_ctx(), checkpoint=_checkpoint([batch]),
        observations=[batch], tool_calls=[{"id": "call-1", "name": "tavily_search"}],
        component_trial=False, store_checkpoint=None)
    assert outcome.fell_back and batch.render_vendor.calls == 1
    assert broker.acks[-1][1]["outcome"] == "P1_FAILED"


async def test_without_reduce_a_p1_failure_is_reraised_not_swallowed():
    broker = ScriptedBroker(Form.P1)
    strategy, _ = _brokered_page(broker, RecordingModelCall(fail=True))
    with pytest.raises(PageSelectionError):
        await _run_page(strategy, [_deferred("call-1", 2)])


async def test_a_fail_closed_resolution_runs_p0():
    broker = ScriptedBroker(Form.P0, source=FAIL_CLOSED, reason="AGENT_WAIT_TIMEOUT")
    strategy, executor_call = _brokered_page(broker)
    batch = _deferred("call-1", 2)
    await _run_page(strategy, [batch])
    assert batch.render_vendor.calls == 1 and executor_call.prompts == []
    assert broker.acks[-1][1]["source"] == FAIL_CLOSED


async def test_cancellation_while_waiting_runs_nothing():
    broker = ScriptedBroker(raise_exc=asyncio.CancelledError())
    strategy, executor_call = _brokered_page(broker)
    batch = _deferred("call-1", 2)
    with pytest.raises(asyncio.CancelledError):
        await _run_page(strategy, [batch])
    assert batch.render_vendor.calls == 0 and executor_call.prompts == []


async def test_the_same_checkpoint_twice_is_two_items():
    broker = ScriptedBroker(Form.P0)
    strategy, _ = _brokered_page(broker)
    batch = _deferred("call-1", 1)
    await _run_page(strategy, [batch])
    await _run_page(strategy, [batch])
    first, second = (o.item_id for o in broker.offers)
    assert second == f"{first}:r1"


# --- C ---------------------------------------------------------------------------------------


def _close_checkpoint() -> CCheckpoint:
    return CCheckpoint(
        task_id="task", researcher_id="r",
        researcher_messages=(FrozenMessage(role="human", content="find things"),
                             FrozenMessage(role="ai", content="found them")),
        evidence_manifest=EvidenceManifest(span_ids=()), query_attempt_ids=(),
        close_reason="NO_TOOL_CALL", sampling=SAMPLING)


class _FailingProbeClose:
    """Stands in for a C_VISIBLE probe that cannot build a view for this history."""

    last_outcome = None

    async def close_researcher(self, *, task_ctx, checkpoint):
        from shapeflow.strategies.close_visible import CloseSelectionError
        from shapeflow.strategies.pipeline import SelectionFailure

        raise CloseSelectionError(SelectionFailure("VISIBLE_SOURCE_REGISTRY", "no sources"))


async def test_a_close_with_no_viable_p1_offers_p0_and_declines_to_vendor():
    from shapeflow.odr.vendor_hooks import _UseVendorCompression

    broker = ScriptedBroker(Form.P0)
    strategy = BrokeredCloseStrategy(
        settings=_settings(broker), executor=None,
        probe_strategy_for=lambda probe: _FailingProbeClose(), body_for=_body_for())
    with pytest.raises(_UseVendorCompression):
        await strategy.close_researcher(task_ctx=_ctx(), checkpoint=_close_checkpoint())
    (offer,) = broker.offers
    assert offer.p1 is None
    (p0,) = offer.p0.requests
    assert p0.op_class.value == "COMPRESSOR_P0" and p0.est_decode == 4096
    assert broker.acks[-1][1]["outcome"] == "P0_DECLINED_TO_VENDOR"
