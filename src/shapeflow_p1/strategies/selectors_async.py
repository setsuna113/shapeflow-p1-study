"""The three things that can produce a selection, behind one async interface.

They exist to be told apart. Comparing the LLM selector to the CPU one separates "an LLM adds
value" from "pointing at spans instead of prose adds value"; comparing it to SHORT_PROSE
separates the pointer mechanism from simply decoding fewer tokens. Without both controls, a
token or latency saving has at least three explanations and the study cannot pick one.

Every selector reports its own work, and reports it whether or not the output was usable. The
tokens were spent either way.
"""

from __future__ import annotations

import re
import time
from collections import Counter
from typing import Any, Protocol

from .pipeline import WorkRecord

__all__ = ["AsyncSelector", "LlmAsyncSelector", "CpuLexicalAsyncSelector", "ShortProseSelector"]

_TOKEN = re.compile(r"[a-z0-9]+")


class AsyncSelector(Protocol):
    async def select(self, *, task_ctx: Any, view: Any) -> tuple[dict, WorkRecord]:
        """Return (raw selector output, work spent)."""


class LlmAsyncSelector:
    """The treatment selector: the same local target model P0 uses, through the proxy.

    ``model_call`` is injected so this class never reaches for a network of its own, and so the
    proxy -- which does admission, budgeting and telemetry -- is the only path to the engine.
    """

    def __init__(self, model_call, *, op_class: str) -> None:
        self._call = model_call
        self._op_class = op_class

    async def select(self, *, task_ctx: Any, view: Any) -> tuple[dict, WorkRecord]:
        prompt = view.prompt_bytes.decode("utf-8")
        started = time.perf_counter()
        parsed, usage = await self._call(
            prompt=prompt, op_class=self._op_class,
            max_tokens=max(64, view_token_ceiling(view)),
            schema_name="selector_output",
        )
        return parsed, WorkRecord(
            selector_calls=1,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cpu_seconds=time.perf_counter() - started,
            retries=usage.get("retries", 0),
        )


def view_token_ceiling(view) -> int:
    """A completion ceiling derived from the budget, so runaway output is cut at the engine.

    The contract's maxItems is a post-hoc validator: by the time it runs the tokens exist and
    have been charged. This is the half that actually bounds the cost.
    """
    return 8 * len(view.candidates) + 128


class CpuLexicalAsyncSelector:
    """Deterministic lexical selection, no model. A control, never a candidate.

    Its cost is CPU seconds and zero decode tokens, which is exactly the comparison that says
    whether the LLM is doing anything an ordinary ranker could not.
    """

    def __init__(self, *, max_selected: int = 8) -> None:
        self._max = max_selected

    async def select(self, *, task_ctx: Any, view: Any) -> tuple[dict, WorkRecord]:
        started = time.perf_counter()
        topic = getattr(task_ctx, "research_topic", "") or ""
        q_terms = Counter(_TOKEN.findall(topic.lower()))
        scored: list[tuple[float, str, str]] = []
        for cand in view.candidates:
            tf = Counter(_TOKEN.findall(cand.text.lower()))
            score = float(sum(q_terms[t] * tf[t] for t in q_terms))
            if score > 0:
                scored.append((score, cand.label, cand.span_id))
        scored.sort(key=lambda t: (-t[0], t[2]))
        labels = [label for _, label, _ in scored[: self._max]]
        return ({"contract": "P1_ID", "selected_ids": labels},
                WorkRecord(selector_calls=0, cpu_seconds=time.perf_counter() - started))


class ShortProseSelector:
    """A prose control, budgeted to the same rendered-token ceiling as the P1 arm.

    This is what separates "pointing at evidence helps" from "producing less text helps". It
    writes a summary rather than selecting ids, and it is held to the same token budget, so any
    remaining difference cannot be attributed to output length.

    It does not emit a selector contract -- there is no id selection to validate -- so it
    bypasses the parse/aggregate path and returns its prose directly.
    """

    is_prose = True

    def __init__(self, model_call, *, op_class: str) -> None:
        self._call = model_call
        self._op_class = op_class

    async def summarize(self, *, task_ctx: Any, view: Any, token_budget: int
                        ) -> tuple[str, WorkRecord]:
        from ..p1.prompts import render_short_prose_prompt

        prompt = render_short_prose_prompt(
            topic=getattr(task_ctx, "research_topic", ""),
            candidates=[(c.label, c.text, c.heading_path, c.context) for c in view.candidates],
            budget=token_budget,
        )
        started = time.perf_counter()
        text, usage = await self._call(
            prompt=prompt, op_class=self._op_class, max_tokens=token_budget, schema_name=None,
        )
        return text, WorkRecord(
            selector_calls=1,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cpu_seconds=time.perf_counter() - started,
            retries=usage.get("retries", 0),
        )
