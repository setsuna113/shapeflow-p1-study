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

from ..campaign.selector_client import selector_schema_name
from ..p1.contracts import P1_CONTRACTS
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

    def __init__(self, model_call, *, op_class: str, expected_contract: str) -> None:
        if expected_contract not in P1_CONTRACTS:
            raise ValueError(
                f"LLM selector expected_contract must be one of {sorted(P1_CONTRACTS)}, "
                f"got {expected_contract!r}"
            )
        self._call = model_call
        self._op_class = op_class
        self.expected_contract = expected_contract

    async def select(self, *, task_ctx: Any, view: Any) -> tuple[dict, WorkRecord]:
        prompt = view.prompt_bytes.decode("utf-8")
        started = time.perf_counter()
        parsed, usage = await self._call(
            prompt=prompt, op_class=self._op_class,
            max_tokens=max(64, view_token_ceiling(view)),
            schema_name=selector_schema_name(self.expected_contract),
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

    def __init__(self, *, max_selected: int = 64) -> None:
        # 64 is the frozen post-hoc schema ceiling.  It is not the effective control budget:
        # the greedy loop below charges the exact shared renderer and stops at the same
        # selected-token budget as the LLM arm.
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
        from ..p1.aggregators import stable_union_v1
        from ..p1.contracts import parse_selection

        token_budget = int(getattr(task_ctx, "selected_token_budget", 0) or 0)
        if token_budget <= 0:
            raise ValueError("CPU control requires the frozen positive selected-token budget")
        labels: list[str] = []
        for _score, label, _span_id in scored:
            if len(labels) >= self._max:
                break
            proposed = [*labels, label]
            parsed = parse_selection(
                {"contract": "P1_ID", "selected_ids": proposed},
                view.candidate_set,
                expected_contract="P1_ID",
            )
            rendered_tokens = view.cost(stable_union_v1(parsed, view.registry))
            if rendered_tokens <= token_budget:
                labels = proposed
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

    @staticmethod
    def prompt_for(
        *,
        task_ctx: Any,
        view: Any,
        token_budget: int,
        source_entries: tuple[tuple[str, str, str], ...] | None = None,
    ) -> str:
        from ..p1.prompts import render_short_prose_prompt

        return render_short_prose_prompt(
            topic=getattr(task_ctx, "research_topic", ""),
            candidates=[
                (
                    candidate.label,
                    candidate.text,
                    candidate.heading_path,
                    candidate.context,
                    candidate.origin_kind,
                    candidate.message_role,
                )
                for candidate in view.candidates
            ],
            budget=token_budget,
            source_entries=source_entries,
        )

    async def summarize(
        self,
        *,
        task_ctx: Any,
        view: Any,
        token_budget: int,
        max_completion_tokens: int | None = None,
        source_entries: tuple[tuple[str, str, str], ...] | None = None,
    ) -> tuple[str, WorkRecord]:
        prompt = self.prompt_for(
            task_ctx=task_ctx,
            view=view,
            token_budget=token_budget,
            source_entries=source_entries,
        )
        completion_cap = (
            token_budget
            if max_completion_tokens is None
            else int(max_completion_tokens)
        )
        if completion_cap <= 0 or completion_cap > token_budget:
            raise ValueError(
                "SHORT_PROSE completion cap must be positive and no larger than "
                "the rendered-token budget"
            )
        started = time.perf_counter()
        text, usage = await self._call(
            prompt=prompt,
            op_class=self._op_class,
            max_tokens=completion_cap,
            schema_name=None,
        )
        return text, WorkRecord(
            selector_calls=1,
            prompt_tokens=usage.get("prompt_tokens", 0),
            completion_tokens=usage.get("completion_tokens", 0),
            cpu_seconds=time.perf_counter() - started,
            retries=usage.get("retries", 0),
        )
