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
from copy import deepcopy
from pathlib import Path
from typing import Any, Optional

from ..p1.contracts import P1_CONTRACTS
from ..providers.provider_client import ProviderCallError, ProviderClient
from ..runtime.request_tags import OpClass

__all__ = [
    "SelectorModelCall",
    "SelectorResponseError",
    "ALIAS_BY_OP",
    "STRUCTURED_SELECTOR_OPS",
    "SHORT_PROSE_OPS",
    "load_selector_schema",
    "selector_schema_name",
]

#: op class -> the model alias that carries it. The provider rewrites every alias to the one
#: served model, so P0 and P1 issue byte-identical upstream requests.
#:
#: Keyed off :class:`OpClass` rather than bare strings. These two registries previously drifted:
#: the SHORT_PROSE op classes lived here but were absent from the enum, so the aliases for them
#: were pointed at the *selector* op classes instead. Nothing failed loudly -- the ledger simply
#: recorded prose-control work under the structured-selector label, which is exactly the
#: distinction H_ID_VS_PROSE and C_ID_VS_PROSE exist to measure.
ALIAS_BY_OP = {
    OpClass.PAGE_P1_SELECTOR_LOCAL.value: "qwen-selector-page",
    OpClass.PAGE_P1_SELECTOR_GLOBAL.value: "qwen-selector-page-global",
    OpClass.PAGE_P1_SELECTOR_BATCH.value: "qwen-selector-batch",
    OpClass.COMPRESSOR_P1_SELECTOR.value: "qwen-selector-close",
    OpClass.PAGE_P1_SHORT_PROSE.value: "qwen-prose-page",
    OpClass.COMPRESSOR_SHORT_PROSE.value: "qwen-prose-close",
}

STRUCTURED_SELECTOR_OPS = frozenset({
    OpClass.PAGE_P1_SELECTOR_LOCAL.value,
    OpClass.PAGE_P1_SELECTOR_GLOBAL.value,
    OpClass.PAGE_P1_SELECTOR_BATCH.value,
    OpClass.COMPRESSOR_P1_SELECTOR.value,
})
SHORT_PROSE_OPS = frozenset({
    OpClass.PAGE_P1_SHORT_PROSE.value,
    OpClass.COMPRESSOR_SHORT_PROSE.value,
})

if (
    STRUCTURED_SELECTOR_OPS & SHORT_PROSE_OPS
    or STRUCTURED_SELECTOR_OPS | SHORT_PROSE_OPS != frozenset(ALIAS_BY_OP)
):
    raise RuntimeError("selector op policy does not partition every treatment model alias")

#: Every alias this module can emit must name an op class the ledger knows, or the work lands in
#: ``unavailable`` (work_accounting rejects an unknown op_class) and silently leaves the totals.
if not frozenset(ALIAS_BY_OP) <= {o.value for o in OpClass}:
    raise RuntimeError("selector aliases reference op classes absent from OpClass")


_SCHEMA_PREFIX = "selector_output_"


def selector_schema_name(contract: str) -> str:
    """Return the guided-decoding schema name for exactly one treatment contract."""
    if contract not in P1_CONTRACTS:
        raise ValueError(f"unknown selector contract {contract!r}")
    return f"{_SCHEMA_PREFIX}{contract}"


def _contract_from_schema_name(name: str) -> str:
    if not name.startswith(_SCHEMA_PREFIX):
        raise ValueError(
            f"selector schema {name!r} is not contract-specific; the union schema must never "
            "be used for a treatment call"
        )
    contract = name[len(_SCHEMA_PREFIX):]
    if contract not in P1_CONTRACTS:
        raise ValueError(f"selector schema {name!r} names unknown contract {contract!r}")
    return contract


def load_selector_schema(repo: Path, contract: str) -> dict:
    """Load a closed schema whose root accepts exactly ``contract``.

    The source file intentionally contains all three definitions for schema maintenance.  A
    treatment request must not send that union to guided decoding: doing so lets an ID-only arm
    generate a typed or bridge object and silently become another arm.  Keeping the shared
    ``$defs`` but replacing the root branch gives vLLM a self-contained, one-contract grammar.
    """
    if contract not in P1_CONTRACTS:
        raise ValueError(f"unknown selector contract {contract!r}")
    root = json.loads(
        (Path(repo) / "schemas" / "selector_output.schema.json").read_text(encoding="utf-8"))
    schema = deepcopy(root)
    schema["title"] = f"SelectorOutput-{contract}"
    schema["oneOf"] = [{"$ref": f"#/$defs/{contract}"}]
    return schema


class SelectorResponseError(ValueError):
    """Post-dispatch selector failure carrying usage that has already been spent."""

    def __init__(self, message: str, *, usage: dict) -> None:
        super().__init__(message)
        self.usage = dict(usage)


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
        seed: int,
        guided_decoding: bool = True,
    ) -> None:
        if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
            raise ValueError(f"selector seed must be a non-negative integer, got {seed!r}")
        if (
            isinstance(max_completion_tokens, bool)
            or not isinstance(max_completion_tokens, int)
            or max_completion_tokens <= 0
        ):
            raise ValueError("selector max_completion_tokens must be a positive integer")
        self._client = client
        self._cell_token = cell_token
        self._repo = Path(repo)
        self._schemas = {
            contract: load_selector_schema(self._repo, contract)
            for contract in sorted(P1_CONTRACTS)
        }
        self._temperature = temperature
        self._top_p = top_p
        self._cap = max_completion_tokens
        self._seed = seed
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
        structured = op_class in STRUCTURED_SELECTOR_OPS
        prose = op_class in SHORT_PROSE_OPS
        if structured:
            if not isinstance(schema_name, str) or not schema_name:
                raise ProviderCallError(
                    400,
                    f"structured op {op_class!r} requires one contract-specific schema",
                )
            contract = _contract_from_schema_name(schema_name)
            if self._cap < 64:
                raise ProviderCallError(
                    400,
                    "structured selector configured max_completion_tokens is below its "
                    "64-token grammar floor",
                )
        elif prose:
            if schema_name is not None:
                raise ProviderCallError(
                    400,
                    f"SHORT_PROSE op {op_class!r} must not carry a structured schema",
                )
            contract = None
        else:  # Defensive even though ALIAS_BY_OP and the partition are checked at import.
            raise ProviderCallError(400, f"op class {op_class!r} has no frozen decode policy")
        if isinstance(max_tokens, bool) or not isinstance(max_tokens, int) or max_tokens <= 0:
            raise ValueError("selector request max_tokens must be a positive integer")
        # SHORT_PROSE is token-matched at a potentially tiny residual budget after its source
        # wrapper. Raising 4 to 64 both spends and generates treatment-only text that the
        # publisher must discard. Structured JSON selectors retain their explicit 64-token
        # grammar floor, but neither path may exceed the campaign-wide configured ceiling.
        if prose:
            capped = min(max_tokens, self._cap)
        else:
            capped = min(max(64, max_tokens), self._cap)
        body: dict = {
            "model": alias,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": self._temperature,
            "top_p": self._top_p,
            "max_tokens": capped,
            # Pairing a cell in metadata while sampling the selector unseeded does not make a
            # paired experiment. This is the actual engine request field.
            "seed": self._seed,
        }
        if structured and self._guided:
            # Constrain the grammar while it is generated, rather than validating afterwards.
            # vLLM 0.24 honours `response_format: json_schema` but ignores a top-level
            # `guided_json`, so the former is what actually bounds the output; the latter passed
            # silently and the model emitted prose (and, with Qwen3, a <think> block) instead.
            body["response_format"] = {
                "type": "json_schema",
                "json_schema": {"name": schema_name, "schema": self._schemas[contract]},
            }

        payload = await self._client.chat_completions(body, cell_token=self._cell_token)
        usage = dict(payload.get("usage") or {})
        content = _content_of(payload)
        if prose:
            return content, usage
        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as e:
            # The tokens were spent. The caller records the failure with its cost attached.
            raise SelectorResponseError(
                f"selector returned invalid JSON: {e}", usage=usage
            ) from e
        if not isinstance(parsed, dict):
            raise SelectorResponseError(
                "selector output must be a JSON object", usage=usage
            )
        return parsed, usage


def _content_of(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"].get("content") or ""
    except (KeyError, IndexError, TypeError):
        return ""
