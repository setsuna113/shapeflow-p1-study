"""The DeepSeek judge client -- the only place DeepSeek is called, and only for evaluation.

DeepSeek never touches the treatment path (no P0/P1 inference, no selector). It produces truth-
packet candidates and blind quality judgments. The one discipline that matters most here: on a
failure it cannot recover from, it returns :class:`JudgeUnavailable` -- it never imputes a 0, a
mean, or a "pass". A missing judgment must stay missing (and route to the human queue), because a
silently-imputed score would corrupt the very quality endpoints the study reports.

JSON mode is enforced (``response_format={"type":"json_object"}``) and every response is validated
against a local schema. Empty content, a truncated body, or invalid JSON triggers a bounded retry;
a 4xx fails fast; 429/5xx back off. The requested vs returned model and the system fingerprint are
captured so a mid-epoch model drift is detected rather than silently mixed.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Optional

from ..providers.retry import RetryDecision, classify_status
from ..secrets import SecretRedactor

__all__ = ["JudgeUnavailable", "JudgeResponse", "DeepSeekJudge", "load_deepseek_key"]


class JudgeUnavailable(RuntimeError):
    """A judgment could not be obtained. Never imputed -- the caller marks it missing."""


@dataclass(frozen=True)
class JudgeResponse:
    data: dict
    requested_model: str
    returned_model: str
    usage: dict
    request_id: str
    system_fingerprint: str


def load_deepseek_key(redactor: SecretRedactor) -> str:
    import os

    path = os.environ.get("DEEPSEEK_API_KEY_FILE")
    if path:
        with open(path, encoding="utf-8") as fh:
            key = fh.read().strip()
    else:
        key = (os.environ.get("DEEPSEEK_API_KEY") or "").strip()
    if not key:
        raise RuntimeError(
            "no DeepSeek credential: set DEEPSEEK_API_KEY_FILE (production) or DEEPSEEK_API_KEY (dev)"
        )
    redactor.register(key, label="deepseek")
    return key


# transport(body) -> (status_code, response_dict). Injected so tests need no network.
Transport = Callable[[dict], Awaitable[tuple[int, dict]]]
Sleeper = Callable[[float], Awaitable[None]]


async def _noop_sleep(_seconds: float) -> None:  # pragma: no cover - trivial
    return None


class DeepSeekJudge:
    def __init__(
        self,
        transport: Transport,
        model: str,
        api_key: str,
        *,
        max_retries: int = 3,
        sleeper: Sleeper = _noop_sleep,
    ) -> None:
        if not model:
            raise ValueError("judge model must be resolved and non-empty (frozen at launch)")
        self._transport = transport
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleeper

    def _request_body(self, system: str, user: str) -> dict:
        return {
            "model": self._model,
            "api_key": self._api_key,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
        }

    async def judge(
        self, system: str, user: str, *, validate: Optional[Callable[[dict], None]] = None
    ) -> JudgeResponse:
        """Return a validated JSON judgment, or raise JudgeUnavailable after bounded retries.

        ``validate`` is an optional local schema check (raises on invalid). It is applied on every
        attempt so an invalid body is retried rather than accepted.
        """
        body = self._request_body(system, user)
        last_reason = "unknown"
        for attempt in range(self._max_retries + 1):
            status, payload = await self._transport(body)

            if status != 200:
                decision = classify_status(status)
                if decision is RetryDecision.FAIL_FAST:
                    raise JudgeUnavailable(f"deepseek {status} (fail-fast)")
                last_reason = f"http {status}"
                await self._sleep(0.5 * (2 ** attempt))
                continue

            content = _extract_content(payload)
            if not content:
                last_reason = "empty content"
                await self._sleep(0.2)
                continue
            try:
                data = json.loads(content)
                if not isinstance(data, dict):
                    raise ValueError("top-level JSON is not an object")
                if validate is not None:
                    validate(data)
            except Exception as e:  # invalid JSON or schema violation -> retry
                last_reason = f"invalid body: {e}"
                await self._sleep(0.2)
                continue

            return JudgeResponse(
                data=data,
                requested_model=self._model,
                returned_model=str(payload.get("model", "")),
                usage=payload.get("usage", {}) or {},
                request_id=str(payload.get("id", "")),
                system_fingerprint=str(payload.get("system_fingerprint", "")),
            )

        raise JudgeUnavailable(f"exhausted {self._max_retries} retries: {last_reason}")


def _extract_content(payload: dict) -> str:
    try:
        choice = payload["choices"][0]
        # A length-truncated response is treated as empty so it retries rather than parsing junk.
        if choice.get("finish_reason") == "length":
            return ""
        return choice["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""
