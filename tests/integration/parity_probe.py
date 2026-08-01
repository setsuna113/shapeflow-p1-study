"""Drive the real researcher subgraph and emit a RunTrace as JSON.

Run as a subprocess with ``PYTHONPATH`` pointing at one ODR source root, so the pristine and
patched trees genuinely cannot share module state. Both call themselves
``open_deep_research``; importing one after the other in a single interpreter would hand the
second run the first one's modules and produce a parity result that means nothing.

Everything non-deterministic is replaced before the graph is built: a scripted chat model and
canned Tavily responses. What remains is the graph's own behaviour, which is the thing under
comparison.

Usage:  python parity_probe.py <scenario> [--strategy p0]
Emits one JSON object on stdout.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import sys
from typing import Any


def sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# --- scenarios -------------------------------------------------------------------------
#
# Each returns (tavily_responses_by_query, scripted_assistant_turns). A turn is a list of tool
# calls; an empty list means the researcher stops without calling anything.

def _result(url: str, title: str, content: str, raw: str | None) -> dict:
    return {
        "url": url,
        "title": title,
        "content": content,
        "raw_content": raw,
        # Vendor ignores this capture-time sidecar field. The patched explicit-P0 path carries
        # it through H publication so the actual C checkpoint can prove source provenance
        # without changing the graph-visible ToolMessage.
        "_shapeflow_occurrence_id": "occ-" + sha(url)[:16],
    }


SCENARIOS: dict[str, dict] = {
    "single_search": {
        "tavily": {"cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)]},
        "turns": [[{"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"}], []],
    },
    "multi_query_one_call": {
        "tavily": {
            "cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)],
            "dogs": [_result("https://b.example", "B", "snip b", "RAW B " * 20)],
        },
        "turns": [[{"name": "tavily_search", "args": {"queries": ["cats", "dogs"]}, "id": "c1"}], []],
    },
    "two_search_siblings": {
        "tavily": {
            "cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)],
            "dogs": [_result("https://b.example", "B", "snip b", "RAW B " * 20)],
        },
        "turns": [[
            {"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"},
            {"name": "tavily_search", "args": {"queries": ["dogs"]}, "id": "c2"},
        ], []],
    },
    "mixed_search_and_think": {
        "tavily": {"cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)]},
        "turns": [[
            {"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"},
            {"name": "think_tool", "args": {"reflection": "considering"}, "id": "c2"},
        ], []],
    },
    "empty_raw_content": {
        "tavily": {"cats": [_result("https://a.example", "A", "snip only", None)]},
        "turns": [[{"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"}], []],
    },
    "duplicate_url": {
        "tavily": {
            "cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)],
            "felines": [_result("https://a.example", "A dup", "snip dup", "RAW DUP " * 20)],
        },
        "turns": [[{"name": "tavily_search",
                    "args": {"queries": ["cats", "felines"]}, "id": "c1"}], []],
    },
    "no_results": {
        "tavily": {"cats": []},
        "turns": [[{"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"}], []],
    },
    "research_complete": {
        "tavily": {"cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)]},
        "turns": [[
            {"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"},
        ], [{"name": "ResearchComplete", "args": {}, "id": "c9"}]],
    },
    "no_tool_call": {
        "tavily": {},
        "turns": [[]],
    },
    "max_react": {
        "tavily": {"cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)]},
        # max_react_tool_calls is set to 1 in the config below, so the first turn exceeds it.
        "turns": [[{"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"}]],
    },
    "tool_exception": {
        "tavily": {"boom": "RAISE"},
        "turns": [[{"name": "tavily_search", "args": {"queries": ["boom"]}, "id": "c1"}], []],
    },
    "summarization_timeout": {
        "tavily": {"cats": [_result("https://a.example", "A", "snip a", "RAW A " * 20)]},
        "turns": [[{"name": "tavily_search", "args": {"queries": ["cats"]}, "id": "c1"}], []],
        "summarize_timeout": True,
    },
}


# --- deterministic doubles ---------------------------------------------------------------


class ScriptedModel:
    """A chat model that replays a fixed script and records every request it received."""

    def __init__(self, turns: list, log: list, summary_fails: bool = False) -> None:
        self._turns = list(turns)
        self._log = log
        self._structured = None
        self._summary_fails = summary_fails
        self._config: dict = {}

    # langchain surface used by the graph
    def with_config(self, config=None, **kw):
        clone = ScriptedModel(self._turns, self._log, self._summary_fails)
        clone._turns = self._turns
        clone._structured = self._structured
        clone._config = {**self._config, **(config or {}), **kw}
        return clone

    def with_structured_output(self, schema, **kw):
        clone = self.with_config()
        clone._structured = schema
        return clone

    def with_retry(self, **kw):
        return self

    def bind_tools(self, tools, **kw):
        clone = self.with_config()
        clone._config = {**clone._config,
                         "tools_signature": ",".join(sorted(getattr(t, "name", str(t))
                                                            for t in tools))}
        return clone

    async def ainvoke(self, messages, config=None, **kw):
        from langchain_core.messages import AIMessage

        rendered = "\n".join(
            f"{getattr(m, 'type', '?')}::{m.content}" for m in messages
        )
        op = "SUMMARIZE" if self._structured is not None else "REACT"
        self._log.append({
            "op_class": op,
            "prompt_sha256": sha(rendered),
            "tools_signature": self._config.get("tools_signature", ""),
            "model": str(self._config.get("model", "scripted")),
            "sampling_signature": json.dumps(
                {k: v for k, v in sorted(self._config.items())
                 if k in ("max_tokens", "temperature")}, sort_keys=True),
        })
        if self._structured is not None:
            if self._summary_fails:
                await asyncio.sleep(3600)     # cancelled by vendor's 60s wait_for
            out = self._structured(summary="SUM", key_excerpts="EXC")
            self._log[-1]["output_sha256"] = sha("SUM|EXC")
            return out
        turn = self._turns.pop(0) if self._turns else []
        msg = AIMessage(content="thinking", tool_calls=[
            {"id": tc["id"], "name": tc["name"], "args": tc["args"], "type": "tool_call"}
            for tc in turn
        ])
        self._log[-1]["output_sha256"] = sha(json.dumps(turn, sort_keys=True))
        return msg


def install_doubles(scenario: dict, log: list) -> None:
    """Replace every source of nondeterminism, before the graph is imported."""
    import open_deep_research.utils as u

    async def fake_tavily_search_async(queries, max_results=5, topic="general",
                                       include_raw_content=True, config=None):
        out = []
        for q in queries:
            canned = scenario["tavily"].get(q, [])
            if canned == "RAISE":
                raise RuntimeError("tavily exploded")
            out.append({"query": q, "results": [dict(r) for r in canned]})
        return out

    u.tavily_search_async = fake_tavily_search_async

    model = ScriptedModel(scenario["turns"], log,
                          summary_fails=scenario.get("summarize_timeout", False))
    u.init_chat_model = lambda *a, **k: model
    u.get_api_key_for_model = lambda *a, **k: "FAKEFAKEFAKE"
    u.get_tavily_api_key = lambda *a, **k: "FAKEFAKEFAKE"

    import open_deep_research.deep_researcher as dr
    dr.configurable_model = model
    dr.get_api_key_for_model = lambda *a, **k: "FAKEFAKEFAKE"
    # Freeze the date so prompts are byte-stable across runs.
    u.get_today_str = lambda: "FROZEN DATE"
    dr.get_today_str = lambda: "FROZEN DATE"


# --- the run ------------------------------------------------------------------------------


async def run(
    scenario_name: str,
    strategy: str | None,
    *,
    capture_checkpoints: bool = False,
) -> dict:
    scenario = SCENARIOS[scenario_name]
    log: list = []
    install_doubles(scenario, log)

    import open_deep_research.deep_researcher as dr

    published: list = []
    original_tools = dr.researcher_tools

    async def traced_tools(state, config):
        command = await original_tools(state, config)
        update = (command.update or {}) if hasattr(command, "update") else {}
        messages = update.get("researcher_messages", []) or []
        published.append({
            "messages": [
                {"tool_call_id": getattr(m, "tool_call_id", ""), "name": getattr(m, "name", ""),
                 "content_sha256": sha(str(m.content))}
                for m in messages
            ],
            "goto": getattr(command, "goto", ""),
            "state_update_sha256": sha(json.dumps(
                sorted(update.keys()), sort_keys=True) + str(len(messages))),
        })
        return command

    dr.researcher_tools = traced_tools
    builder = dr.StateGraph(dr.ResearcherState, output_schema=dr.ResearcherOutputState,
                            config_schema=dr.Configuration)
    builder.add_node("researcher", dr.researcher)
    builder.add_node("researcher_tools", traced_tools)
    builder.add_node("compress_research", dr.compress_research)
    builder.add_edge(dr.START, "researcher")
    builder.add_edge("compress_research", dr.END)
    graph = builder.compile()

    from langchain_core.messages import HumanMessage

    # Gate B binds an explicit P0 strategy so the deferred/checkpoint/reduce/refill path
    # actually executes. Hooks-off never runs that code, so only this can show a bug inside it.
    binding_cm = None
    strategy_cm = None
    captured_checkpoints: list = []
    captured_events: list[dict] = []
    if strategy == "p0":
        from shapeflow.odr.hooks import StrategyBundle, strategies_bound
        from shapeflow.odr.vendor_hooks import RunBinding, bind_run
        from shapeflow.p1.selectors import Candidate  # noqa: F401  (import sanity)
        from shapeflow.strategies.p0 import VendorCloseStrategy, VendorPageStrategy

        page = VendorPageStrategy({})
        strategy_cm = strategies_bound(
            StrategyBundle(variant_id="P0", page=page, close=VendorCloseStrategy()))
        binding_cm = bind_run(RunBinding(
            task_id="parity", researcher_id="r0", attempt_id="a0",
            task_ctx=None, component_trial=False,
            store_checkpoint=(
                lambda checkpoint: captured_checkpoints.append(checkpoint)
                if capture_checkpoints else None
            ),
            on_event=(
                lambda kind, payload: captured_events.append(
                    {"kind": kind, "payload": payload}
                )
                if capture_checkpoints else None
            ),
        ))

    config = {"configurable": {
        "search_api": "tavily",
        "max_react_tool_calls": 1 if scenario_name == "max_react" else 5,
        "max_content_length": 200,
        "research_model": "scripted", "research_model_max_tokens": 100,
        "summarization_model": "scripted", "summarization_model_max_tokens": 100,
        "compression_model": "scripted", "compression_model_max_tokens": 100,
        "max_structured_output_retries": 1,
        "mcp_config": None,
    }}
    exceptions: list[str] = []
    result: dict = {}
    import contextlib
    stack = contextlib.ExitStack()
    if strategy_cm is not None:
        stack.enter_context(strategy_cm)
        stack.enter_context(binding_cm)
    try:
      with stack:
        result = await graph.ainvoke(
            {"researcher_messages": [HumanMessage(content="research cats")],
             "research_topic": "cats", "tool_call_iterations": 0},
            config,
        )
    except Exception as e:  # noqa: BLE001
        exceptions.append(type(e).__name__)

    trace = {
        "scenario": scenario_name,
        "strategy": strategy or "none",
        "model_requests": log,
        "publish_batches": published,
        "close_reasons": [],
        "exceptions": exceptions,
        "final_report_sha256": sha(str(result.get("compressed_research", ""))),
        "raw_notes_sha256": sha(json.dumps(result.get("raw_notes", []), sort_keys=True)),
        "odr_root": os.path.dirname(
            __import__("open_deep_research").__path__[0]
        ),
    }
    if capture_checkpoints:
        from shapeflow.strategies.visible_view import build_visible_view

        trace["captured_c_checkpoints"] = [
            {
                "digest": checkpoint.digest,
                "researcher_id": checkpoint.researcher_id,
                "tool_messages": [
                    {
                        "tool_call_id": message.tool_call_id,
                        "artifact_canonical": message.artifact_canonical,
                    }
                    for message in checkpoint.researcher_messages
                    if message.role == "tool"
                ],
                "visible_segments": list(
                    build_visible_view(checkpoint.researcher_messages).message_segments
                ),
            }
            for checkpoint in captured_checkpoints
            if type(checkpoint).__name__ == "CCheckpoint"
        ]
        trace["captured_hook_events"] = captured_events
    return trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("scenario")
    parser.add_argument("--strategy", default=None)
    parser.add_argument("--capture-checkpoints", action="store_true")
    args = parser.parse_args()
    trace = asyncio.run(run(
        args.scenario,
        args.strategy,
        capture_checkpoints=args.capture_checkpoints,
    ))
    sys.stdout.write("<<<TRACE>>>" + json.dumps(trace, sort_keys=True))


if __name__ == "__main__":
    main()
