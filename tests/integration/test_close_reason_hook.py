"""The close hook classifies the assistant decision, not the trailing tool result."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from langchain_core.messages import AIMessage, ToolMessage

REPO = Path(__file__).resolve().parents[2]
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"


def _reason(messages, *, iterations=1, maximum=5):
    """Invoke the exact materialized patch in an isolated interpreter.

    The development venv can contain yesterday's installed ODR tree, and a prior test can also
    cache that module name from the pristine parity tree.  Importing in this pytest process made
    the test nondeterministically inspect either tree.  The production parity harness already
    isolates them by subprocess; this hook-level check must do the same.
    """
    payload = {
        "messages": [
            {
                "type": getattr(message, "type", ""),
                "content": getattr(message, "content", ""),
                "tool_calls": list(getattr(message, "tool_calls", None) or ()),
                "tool_call_id": getattr(message, "tool_call_id", None),
            }
            for message in messages
        ],
        "iterations": iterations,
        "maximum": maximum,
    }
    script = r"""
import json, sys
from types import SimpleNamespace
from langchain_core.messages import AIMessage, ToolMessage
from open_deep_research.deep_researcher import _sf_close_reason

body = json.loads(sys.stdin.read())
messages = []
for raw in body["messages"]:
    if raw["type"] == "ai":
        messages.append(AIMessage(
            content=raw["content"], tool_calls=raw.get("tool_calls") or []))
    elif raw["type"] == "tool":
        messages.append(ToolMessage(
            content=raw["content"], tool_call_id=raw.get("tool_call_id") or "tool"))
result = _sf_close_reason(
    {"researcher_messages": messages, "tool_call_iterations": body["iterations"]},
    SimpleNamespace(max_react_tool_calls=body["maximum"]),
)
print(result)
"""
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join((
        str(PATCHED), str(REPO / "src"), env.get("PYTHONPATH", ""),
    ))
    completed = subprocess.run(
        [sys.executable, "-c", script],
        input=json.dumps(payload),
        text=True,
        capture_output=True,
        check=True,
        cwd=REPO,
        env=env,
    )
    return completed.stdout.strip().splitlines()[-1]


def test_research_complete_survives_trailing_toolmessage():
    decision = AIMessage(
        content="",
        tool_calls=[{"id": "c", "name": "ResearchComplete", "args": {}, "type": "tool_call"}],
    )
    result = ToolMessage(content="Research completed", tool_call_id="c")
    assert _reason([decision, result]) == "RESEARCH_COMPLETE"


def test_max_react_survives_trailing_toolmessage():
    decision = AIMessage(
        content="",
        tool_calls=[{
            "id": "s",
            "name": "tavily_search",
            "args": {"queries": ["x"]},
            "type": "tool_call",
        }],
    )
    result = ToolMessage(content="search result", tool_call_id="s")
    assert _reason([decision, result], iterations=5, maximum=5) == "MAX_REACT_EXCEEDED"


def test_no_tool_exit_uses_last_ai_message():
    assert _reason([AIMessage(content="done")]) == "NO_TOOL_CALL"
