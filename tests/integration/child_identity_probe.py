"""Exercise the materialized ODR supervisor_tools node with real patch dispatch."""

from __future__ import annotations

import argparse
import asyncio
import json


def _calls(items):
    from langchain_core.messages import AIMessage

    return AIMessage(
        content="delegate",
        tool_calls=[
            {
                "id": call_id,
                "name": "ConductResearch",
                "args": {"research_topic": topic},
                "type": "tool_call",
            }
            for call_id, topic in items
        ],
    )


def _normalise(command):
    update = dict(command.update or {})
    return {
        "goto": str(command.goto),
        "messages": [
            {
                "content": str(message.content),
                "name": str(message.name),
                "tool_call_id": str(message.tool_call_id),
            }
            for message in update.get("supervisor_messages", ())
        ],
        "raw_notes": list(update.get("raw_notes", ()) or ()),
    }


async def run(mode: str, scenario: str) -> dict:
    import open_deep_research.deep_researcher as dr

    observations = []
    expected_parallel = 2 if scenario == "concurrent" else 1
    both_entered = asyncio.Event()
    entered_count = 0

    class FakeResearcherSubgraph:
        async def ainvoke(self, state, config):
            nonlocal entered_count
            topic = state["research_topic"]
            coordinate = None
            researcher_id = None
            handle = None
            markers = []
            if mode == "bound":
                from shapeflow_p1.odr import vendor_hooks as hooks
                from shapeflow_p1.p1.view import compact_publication_handle

                binding = hooks.current_run()
                coordinate = binding.researcher_coordinate
                researcher_id = binding.researcher_id
                sidecar = hooks._PUBLISHED_TOOL_PROVENANCE.get()  # probe isolation itself
                sidecar[f"marker:{topic}"] = {
                    "name": "probe",
                    "content_sha256": topic,
                    "source_occurrence_ids": (),
                    "checkpoint": "probe",
                }
                if expected_parallel == 2:
                    entered_count += 1
                    if entered_count == 2:
                        both_entered.set()
                    await asyncio.wait_for(both_entered.wait(), timeout=1)
                markers = sorted(sidecar)
                handle = compact_publication_handle(
                    0,
                    0,
                    1,
                    researcher_coordinate=(coordinate[0], coordinate[1]),
                )
            observations.append({
                "topic": topic,
                "coordinate": list(coordinate) if coordinate else None,
                "researcher_id": researcher_id,
                "handle": handle,
                "markers": markers,
            })
            return {
                "compressed_research": f"report:{topic}",
                "raw_notes": [f"raw:{topic}"],
            }

    dr.researcher_subgraph = FakeResearcherSubgraph()
    config = {"configurable": {
        "max_researcher_iterations": 20,
        "max_concurrent_research_units": 2 if scenario == "concurrent" else 1,
        "research_model": "probe",
    }}

    async def invoke(iteration: int, items):
        return await dr.supervisor_tools({
            "supervisor_messages": [_calls(items)],
            "research_iterations": iteration,
            "research_brief": "probe",
        }, config)

    parent_sidecar = None
    if mode == "bound":
        from shapeflow_p1.odr.vendor_hooks import (
            RunBinding,
            bind_run,
            current_published_tool_provenance,
        )

        binding = RunBinding(
            task_id="task",
            researcher_id="task:replicate0",
            attempt_id="attempt",
            task_ctx=None,
        )
        with bind_run(binding):
            if scenario == "sequential":
                commands = [
                    await invoke(1, [("conduct-a", "alpha")]),
                    await invoke(2, [("conduct-b", "beta")]),
                ]
            else:
                commands = [
                    await invoke(
                        7,
                        [("conduct-a", "alpha"), ("conduct-b", "beta")],
                    )
                ]
            parent_sidecar = current_published_tool_provenance()
    else:
        if scenario == "sequential":
            commands = [
                await invoke(1, [("conduct-a", "alpha")]),
                await invoke(2, [("conduct-b", "beta")]),
            ]
        else:
            commands = [
                await invoke(
                    7,
                    [("conduct-a", "alpha"), ("conduct-b", "beta")],
                )
            ]

    return {
        "mode": mode,
        "scenario": scenario,
        "commands": [_normalise(command) for command in commands],
        "children": sorted(observations, key=lambda value: value["topic"]),
        "parent_sidecar": parent_sidecar,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("hooks-off", "bound"), required=True)
    parser.add_argument("--scenario", choices=("sequential", "concurrent"), required=True)
    args = parser.parse_args()
    print("<<<TRACE>>>" + json.dumps(
        asyncio.run(run(args.mode, args.scenario)), sort_keys=True
    ))


if __name__ == "__main__":
    main()
