"""Pricing both forms at a handoff point, without sending anything.

An offer is a description of what *would* be sent, so pricing must never send. Each plan is
priced from the same code that will later execute it, so the price describes the request the
agent would actually issue rather than a re-rendering of it:

- **P0 at H** -- vendor's own summarization prompt (``summarize_webpage_prompt``, the date,
  each result's ``raw_content[:max_content_length]``), one request per result with raw content,
  in vendor's order. Prompt bytes are exact.
- **P1 at H and C** -- the production selection strategy, built afresh from the same factory
  but with a :class:`PricingProbe` in place of the model call. The probe records the exact
  provider request body and declines, so the view, the prompt admission and the prompt bytes
  are the ones the executing strategy will build -- and the executing instance, which carries
  per-batch state, is never touched by pricing.
- **P0 at C** -- vendor's ``compress_research`` messages: its system prompt, the researcher's
  history from the checkpoint, and its closing instruction. Approximate: the multi-turn chat
  template is rendered here by hand, and vendor may retry with a trimmed history.

Every descriptor carries ``prompt_sha256`` computed exactly as the provider records it
(:func:`~shapeflow.runtime.provider_server.provider_prompt_sha256`), so each request the agent
later sends can be joined back to the offer that priced it. ``est_cached_prefill`` is 0 --
nothing here predicts prefix-cache hits -- and ``est_decode`` is the request's own
``max_tokens``: the plan's upper bound, not a prediction.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..contracts.offer import P1Plan, RenderDescriptor, RequestDescriptor
from ..evidence.chunkers import Tokenizer
from ..runtime.provider_server import provider_prompt_sha256
from ..runtime.request_tags import OpClass

__all__ = [
    "VendorPrompts", "PricingProbe", "PricingProbeDeclined", "CapturedRequest", "P1Price",
    "render_qwen3_chatml", "price_p0_h", "price_p0_c", "price_p1",
    "NOMINAL_RENDER_CPU_MS",
]

#: No CPU cost model is calibrated; ``plan_weight`` does not read the render estimate.
NOMINAL_RENDER_CPU_MS = 1.0


@dataclass(frozen=True)
class VendorPrompts:
    """The vendor strings P0 sends, loaded from the installed (patched) graph."""

    summarize_webpage: str
    compress_system: str
    compress_human: str
    today: Callable[[], str]

    @classmethod
    def load(cls) -> VendorPrompts:
        from open_deep_research.prompts import (
            compress_research_simple_human_message,
            compress_research_system_prompt,
            summarize_webpage_prompt,
        )
        from open_deep_research.utils import get_today_str

        return cls(summarize_webpage=summarize_webpage_prompt,
                   compress_system=compress_research_system_prompt,
                   compress_human=compress_research_simple_human_message,
                   today=get_today_str)


def _text_of(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                       for part in content)
    return "" if content is None else str(content)


def render_qwen3_chatml(messages: Sequence[Mapping]) -> str:
    """Qwen3 ChatML with thinking disabled, as the engine would see ``messages``.

    Exact for a single user turn (the case every P0@H and P1 request is). For multi-turn
    histories with tool calls it follows Qwen3's template shape -- tool calls as
    ``<tool_call>`` blocks, tool results as ``<tool_response>`` user turns -- which is an
    approximation, and P0@C is the only plan priced with it.
    """
    out = []
    for message in messages:
        role = message.get("role", "user")
        content = _text_of(message.get("content"))
        if role == "tool":
            out.append(f"<|im_start|>user\n<tool_response>\n{content}\n</tool_response>"
                       "<|im_end|>\n")
            continue
        if role == "assistant" and message.get("tool_calls"):
            for call in message["tool_calls"]:
                function = call.get("function", call)
                content += ("\n<tool_call>\n" + json.dumps(
                    {"name": function.get("name"), "arguments": function.get("arguments")},
                    ensure_ascii=False) + "\n</tool_call>")
        out.append(f"<|im_start|>{role}\n{content}<|im_end|>\n")
    out.append("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    return "".join(out)


def _count(tokenizer: Tokenizer, text: str) -> int:
    try:
        return tokenizer.count(text)
    except RuntimeError:
        # The frozen tokenizer refuses text whose offsets it cannot map back; a price is still
        # owed, so count ids directly from its backend when it has one.
        backend = getattr(tokenizer, "_backend", None)
        if backend is None:
            raise
        return len(backend.encode(text, add_special_tokens=False).ids)


def _descriptor(op: OpClass, messages: list, max_tokens: int,
                tokenizer: Tokenizer) -> RequestDescriptor:
    return RequestDescriptor(
        op_class=op,
        prompt_sha256=provider_prompt_sha256(messages),
        max_tokens=max_tokens,
        est_uncached_prefill=_count(tokenizer, render_qwen3_chatml(messages)),
        est_cached_prefill=0,
        est_decode=max_tokens,
    )


def price_p0_h(checkpoint, deferred: Mapping[str, Any], *, prompts: VendorPrompts,
               tokenizer: Tokenizer, max_tokens: int) -> tuple[RequestDescriptor, ...]:
    """One PAGE_P0_SUMMARY request per result with raw content, in vendor's order."""
    date = prompts.today()
    out = []
    for call_id, _ in checkpoint.search_result_sets:
        batch = deferred[call_id]
        for result in batch.results:
            raw = result.get("raw_content")
            if not raw:
                continue  # vendor's noop(): no request for a snippet-only result
            prompt = prompts.summarize_webpage.format(
                webpage_content=str(raw)[: batch.max_content_length], date=date)
            out.append(_descriptor(OpClass.PAGE_P0_SUMMARY,
                                   [{"role": "user", "content": prompt}], max_tokens, tokenizer))
    return tuple(out)


def _openai_dicts(messages: Sequence[Any]) -> list[dict]:
    try:
        from langchain_openai.chat_models.base import _convert_message_to_dict
    except ImportError:  # pragma: no cover - the run host always has langchain_openai
        return [{"role": getattr(m, "type", "user"), "content": _text_of(m.content)}
                for m in messages]
    return [_convert_message_to_dict(m) for m in messages]


def price_p0_c(checkpoint, *, prompts: VendorPrompts, tokenizer: Tokenizer,
               max_tokens: int) -> tuple[RequestDescriptor, ...]:
    """Vendor's first compression attempt: system prompt, history, closing instruction."""
    from langchain_core.messages import HumanMessage, SystemMessage

    from ..odr.adapter import thaw_message

    messages = [SystemMessage(content=prompts.compress_system.format(date=prompts.today())),
                *(thaw_message(m) for m in checkpoint.researcher_messages),
                HumanMessage(content=prompts.compress_human)]
    return (_descriptor(OpClass.COMPRESSOR_P0, _openai_dicts(messages), max_tokens, tokenizer),)


class PricingProbeDeclined(Exception):
    """Raised by the probe after recording. Pricing only; never a real selector failure."""


@dataclass(frozen=True)
class CapturedRequest:
    op_class: str
    body: dict


class PricingProbe:
    """Stands in for ``SelectorModelCall``: records the exact request body, then declines.

    ``body_for`` is the real call's
    :meth:`~shapeflow.campaign.selector_client.SelectorModelCall.request_body`, so the decode
    cap, the guided-decoding schema and the messages are the ones the executing strategy will
    send.
    """

    def __init__(self, body_for: Callable[..., dict]) -> None:
        self._body_for = body_for
        self.captured: list[CapturedRequest] = []

    async def __call__(self, *, prompt: str, op_class: str, max_tokens: int,
                       schema_name: str | None = None):
        body = self._body_for(prompt=prompt, op_class=op_class, max_tokens=max_tokens,
                              schema_name=schema_name)
        self.captured.append(CapturedRequest(op_class=op_class, body=body))
        raise PricingProbeDeclined()


@dataclass(frozen=True)
class P1Price:
    plan: P1Plan | None
    #: Why P1 is not offered, when it is not. Empty when ``plan`` is set.
    absent_reason: str = ""
    view_sha256: str = ""
    candidates: int = 0
    probe_failure: str = ""
    extra: dict = field(default_factory=dict)


def price_p1(probe: PricingProbe, *, outcomes: Sequence[Any], failure_reason: str,
             tokenizer: Tokenizer, max_model_len: int) -> P1Price:
    """Turn what one probe run captured into a single-selector P1 plan, or say why not."""
    view_sha = next((o.view_sha256 for o in outcomes if getattr(o, "view_sha256", "")), "")
    offered = next((o.offered for o in outcomes if getattr(o, "offered", 0)), 0)
    if not probe.captured:
        return P1Price(plan=None, absent_reason=failure_reason or "NO_SELECTOR_REQUEST",
                       view_sha256=view_sha, candidates=offered, probe_failure=failure_reason)
    if len(probe.captured) != 1:
        return P1Price(plan=None, absent_reason="MULTI_REQUEST_P1", view_sha256=view_sha,
                       candidates=offered)
    captured = probe.captured[0]
    messages = captured.body["messages"]
    descriptor = RequestDescriptor(
        op_class=OpClass(captured.op_class),
        prompt_sha256=provider_prompt_sha256(messages),
        max_tokens=int(captured.body["max_tokens"]),
        est_uncached_prefill=_count(tokenizer, render_qwen3_chatml(messages)),
        est_cached_prefill=0,
        est_decode=int(captured.body["max_tokens"]),
    )
    if descriptor.est_uncached_prefill + descriptor.max_tokens > max_model_len:
        # Sent anyway, the engine would refuse it; offering it would let a decision choose a
        # plan that can only fail into the whole-batch fallback.
        return P1Price(plan=None, absent_reason="WINDOW_INFEASIBLE", view_sha256=view_sha,
                       candidates=offered)
    return P1Price(
        plan=P1Plan(selector=descriptor,
                    render=RenderDescriptor(est_cpu_ms=NOMINAL_RENDER_CPU_MS,
                                            candidate_count=offered)),
        view_sha256=view_sha, candidates=offered)
