"""A deterministic OpenAI-compatible engine, good enough to drive the real ODR graph.

It answers by *tool name*, which is how the graph actually distinguishes its call sites:
``ResearchQuestion`` for the brief, ``ConductResearch``/``ResearchComplete`` for the supervisor,
``tavily_search``/``ResearchComplete`` for the researcher, ``Summary`` for page summarization,
and plain content for compression and the final report.

Deterministic on purpose. A double that varied its answers would make an offline end-to-end
test flaky in a way that looks like a graph bug, and the thing under test here is that the run
completes with no network at all -- not what the model said.
"""

from __future__ import annotations

import json
from typing import Optional


class FakeEngine:
    """Records every request and replies with a scripted, tool-aware completion."""

    def __init__(self, *, searches_per_researcher: int = 1,
                 selector_ids: Optional[list] = None) -> None:
        self.searches_per_researcher = searches_per_researcher
        self.selector_ids = selector_ids
        self.requests: list[dict] = []
        self._search_counts: dict[str, int] = {}
        self._conduct_counts: dict[str, int] = {}

    # transport signature matches ProviderService's injected upstream
    def __call__(self, url, headers, body, timeout):
        self.requests.append({"url": url, "body": body})
        names = _tool_names(body)
        model = str(body.get("model", "fake"))

        # Structured output arrives two ways depending on the langchain version: as a bound
        # tool, or as response_format json_schema. Handling only one leaves the graph parsing a
        # prose answer as JSON, which fails in a way that looks like a graph bug.
        schema = _response_schema_name(body)
        if schema is not None:
            return self._content_reply(model, json.dumps(self._structured(schema, body)))

        if "ResearchQuestion" in names:
            return self._tool_reply(model, "ResearchQuestion", {
                "research_brief": _last_user(body)[:400] or "Research the question.",
            })
        if "ClarifyWithUser" in names:
            return self._tool_reply(model, "ClarifyWithUser", {
                "need_clarification": False, "question": "", "verification": "Starting research.",
            })
        if "Summary" in names:
            excerpt = _last_user(body)[:200]
            return self._tool_reply(model, "Summary", {
                "summary": f"Summary of the page: {excerpt[:120]}",
                "key_excerpts": excerpt,
            })
        if "ConductResearch" in names:
            key = _conversation_key(body)
            count = self._conduct_counts.get(key, 0)
            self._conduct_counts[key] = count + 1
            if count == 0:
                return self._tool_reply(model, "ConductResearch", {
                    "research_topic": _last_user(body)[:300] or "the question",
                })
            return self._tool_reply(model, "ResearchComplete", {})
        if "tavily_search" in names:
            key = _conversation_key(body)
            count = self._search_counts.get(key, 0)
            self._search_counts[key] = count + 1
            if count < self.searches_per_researcher:
                return self._tool_reply(model, "tavily_search", {
                    "queries": [_search_query(body)],
                    "max_results": 5,
                    "topic": "general",
                })
            return self._tool_reply(model, "ResearchComplete", {})
        if self.selector_ids is not None:
            # A P1 selector call: reply with the ids it was offered, as the contract requires.
            return self._content_reply(model, json.dumps({"selected_ids": self.selector_ids}))
        return self._content_reply(
            model,
            "## Findings\n\nThe frozen sources report measurements for 2025 [1].\n\n"
            "### Sources\n[1] https://example.invalid/doc\n",
        )

    def _structured(self, schema: str, body: dict) -> dict:
        text = _last_user(body)
        if schema == "ResearchQuestion":
            return {"research_brief": text[:400] or "Research the question."}
        if schema == "ClarifyWithUser":
            return {"need_clarification": False, "question": "",
                    "verification": "Starting research."}
        if schema == "Summary":
            return {"summary": f"Summary of the page: {text[:120]}",
                    "key_excerpts": text[:200]}
        if schema == "ConductResearch":
            return {"research_topic": text[:300] or "the question"}
        return {}

    # --- reply shapes ---------------------------------------------------------------

    def _tool_reply(self, model: str, name: str, args: dict):
        return 200, {
            "id": f"chatcmpl-{len(self.requests)}",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [{
                        "id": f"call_{len(self.requests)}",
                        "type": "function",
                        "function": {"name": name, "arguments": json.dumps(args)},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
            "usage": {"prompt_tokens": 128, "completion_tokens": 32},
        }, 0.01

    def _content_reply(self, model: str, content: str):
        return 200, {
            "id": f"chatcmpl-{len(self.requests)}",
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 256, "completion_tokens": 64},
        }, 0.01


def _response_schema_name(body: dict) -> Optional[str]:
    fmt = body.get("response_format")
    if isinstance(fmt, dict) and fmt.get("type") == "json_schema":
        schema = fmt.get("json_schema") or {}
        name = schema.get("name")
        if name:
            return str(name)
    return None


def _tool_names(body: dict) -> set:
    names = set()
    for spec in body.get("tools") or ():
        if isinstance(spec, dict):
            function = spec.get("function") or {}
            name = function.get("name") or spec.get("name")
            if name:
                names.add(str(name))
    return names


def _last_user(body: dict) -> str:
    for message in reversed(body.get("messages") or ()):
        if message.get("role") in ("user", "human"):
            return _text(message.get("content"))
    return ""


def _text(content) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content if isinstance(part, dict))
    return str(content or "")


def _conversation_key(body: dict) -> str:
    """A stable key per conversation, so two researchers count their own searches."""
    messages = body.get("messages") or ()
    return _text(messages[0].get("content"))[:80] if messages else "-"


def _search_query(body: dict) -> str:
    text = _last_user(body) or "frozen corpus"
    words = [w for w in text.split() if w.isalnum()]
    return " ".join(words[:6]) or "frozen corpus"
