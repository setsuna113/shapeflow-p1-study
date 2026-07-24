"""The selector's only route to the engine: through the provider, with a real decode cap.

Two things this exists to get right.

**The cap is a completion limit, not a schema bound.** ``selector_output.schema.json`` has a
``maxItems``; that bounds what may be *accepted* after the fact, by which point the tokens have
been generated and charged. The ceiling that actually bounds cost is ``max_tokens`` on the
request, plus guided decoding constraining the grammar as it is produced. Reporting a schema
bound as a decode cap would overstate the saving by exactly the tokens the model emitted and we
then threw away.

**The selector uses the same engine as P0.** Not a second model, not a remote one: the arm under
test is a different *representation*, and swapping the model as well would confound the two. The
op class is carried by the model alias so the ledger can separate P0's summarization cost from
P1's selection cost while the engine receives the same served model either way.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

from ..providers.provider_client import ProviderCallError, ProviderClient

__all__ = ["SelectorModelCall", "ALIAS_BY_OP", "load_selector_schema"]

#: op class -> the model alias that carries it. The provider rewrites every alias to the one
#: served model, so P0 and P1 issue byte-identical upstream requests.
ALIAS_BY_OP = {
    "PAGE_P1_SELECTOR_LOCAL": "qwen-selector-page",
    "PAGE_P1_SELECTOR_GLOBAL": "qwen-selector-page-global",
    "COMPRESSOR_P1_SELECTOR": "qwen-selector-close",
    "PAGE_P1_SHORT_PROSE": "qwen-prose-page",
    "COMPRESSOR_SHORT_PROSE": "qwen-prose-close",
}


def load_selector_schema(repo: Path) -> dict:
    return json.loads(
        (Path(repo) / "schemas" / "selector_output.schema.json").read_text(encoding="utf-8"))


class SelectorModelCall:
    """Callable matching the selector contract: ``(prompt, op_class, max_tokens, schema_name)``.

    Returns ``(parsed_or_text, usage)``. A selector call that produced unusable output still
    spent its tokens, so ``usage`` is returned on every path -- reporting cost only on success
    would make P1 look cheapest exactly when it failed.
    """

    def __init__(
        self,
        client: ProviderClient,
        *,
        cell_token: str,
        repo: Path,
        temperature: float,
        top_p: float,
        max_completion_tokens: int,
        guided_decoding: bool = True,
    ) -> None:
        self._client = client
        self._cell_token = cell_token
        self._schema = load_selector_schema(repo)
        self._temperature = temperature
        self._top_p = top_p
        self._cap = max_completion_tokens
        self._guided = guided_decoding

    async def __call__(
        self,
        *,
        prompt: str,
        op_class: str,
        max_tokens: int,
        schema_name: Optional[str] = None,
    ) -> tuple[Any, dict]:
        alias = ALIAS_BY_OP.get(op_class)
        if alias is None:
            raise ProviderCallError(
                400,
                f"op class {op_class!r} has no model alias; an untagged treatment request would "
                "land in the work total with no attribution",
            )
        # The engine's own ceiling, never above the configured cap: this is the limit that
        # actually stops decoding, and it is what the work accounting reports.
        capped = max(64, min(int(max_tokens), self._cap))
        body: dict = {
            "model": alias,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": capped,
        }
        if schema_name and self._guided:
            # Constrain the grammar while it is generated, rather than validating afterwards.
            # vLLM 0.24 honours `response_format: json_schema` but ignores a top-level
            # `guided_json`, so the former is what actually bounds the output; the latter passed
            # silently and the model emitted prose (and, with Qwen3, a <think> block) instead.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": self._schema},
            }

        payload = await self._client.chat_completions(body, cell_token=self._cell_token)
        usage = dict(payload.get("usage") or {})
        content = _content_of(payload)
        if schema_name is None:
            return content, usage
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # The tokens were spent. The caller records the failure with its cost attached.
            raise ValueError(f"selector returned invalid JSON: {e}") from e
        if not isinstance(parsed, dict):
            raise ValueError("selector output must be a JSON object")
        return parsed, usage


def _content_of(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""
