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

Two things this client is careful about, because both were wrong and both are invisible in the
output:

**The sampling envelope is sent, not merely configured.** ``temperature``, ``top_p``, ``seed``,
``max_tokens`` and the thinking switch travel on the request and are recorded with the response.
A configured-but-unsent parameter is worse than no parameter: the corpus carries a fingerprint
claiming a decoding policy that never reached the model.

**Every attempt is kept.** A retry advances the seed, so it is a genuinely different request
rather than the same one bought twice, and each attempt records its own prompt hash, seed,
returned model, fingerprint, usage and failure reason. Keeping only the accepted response makes
the prompt that produced the corpus unrecoverable for any call that needed a second try.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Mapping, Optional

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..providers.retry import RetryDecision, classify_status
from ..secrets import SecretRedactor

__all__ = [
    "JudgeUnavailable",
    "JudgeTruncated",
    "JudgeResponse",
    "JudgeAttempt",
    "SamplingEnvelope",
    "DeepSeekJudge",
    "load_deepseek_key",
]


class JudgeUnavailable(RuntimeError):
    """A judgment could not be obtained. Never imputed -- the caller marks it missing."""


class JudgeTruncated(JudgeUnavailable):
    """Every attempt hit the output cap.

    Distinct from "unavailable" on purpose: the model answered every time, and the answer
    did not fit. Raising the cap is a protocol change; silently retrying an uncapped body
    until one happens to fit is not.
    """


@dataclass(frozen=True)
class SamplingEnvelope:
    """The decoding policy that is actually put on the wire, and frozen with the corpus."""

    temperature: float = 0.0
    top_p: float = 1.0
    seed: Optional[int] = None
    max_tokens: Optional[int] = None
    #: On vLLM the thinking switch is a request parameter (``chat_template_kwargs``). An earlier
    #: comment here claimed DeepSeek had no such knob and that thinking was a property of the
    #: model id; that is wrong, and ``reasoning_effort`` below is the correction. Note that with
    #: ``send_thinking_switch`` false this pair was never put on the wire at all, so declaring
    #: ``enable_thinking: false`` described nothing -- the model reasoned by default.
    enable_thinking: bool = False
    send_thinking_switch: bool = False
    #: DeepSeek's own control: one of low|medium|high|max|xhigh, validated server-side (an
    #: unknown variant is a 400). ``None`` sends nothing and takes the endpoint default.
    #: Reasoning tokens are billed *inside* ``completion_tokens``, so raising this eats into the
    #: same ``max_tokens`` the JSON answer needs -- the two must move together.
    reasoning_effort: Optional[str] = None

    def for_attempt(self, index: int) -> "SamplingEnvelope":
        """The envelope for retry ``index``.

        The seed advances so a retry is a different request. Re-sending byte-identical
        bytes lands on the same logical call, which either returns the same rejected body
        or pays for it twice; neither is a retry.
        """
        if self.seed is None:
            return self
        return SamplingEnvelope(
            temperature=self.temperature, top_p=self.top_p, seed=self.seed + index,
            max_tokens=self.max_tokens, enable_thinking=self.enable_thinking,
            send_thinking_switch=self.send_thinking_switch,
            reasoning_effort=self.reasoning_effort,
        )

    def request_fields(self) -> dict:
        fields: dict = {"temperature": self.temperature, "top_p": self.top_p}
        if self.seed is not None:
            fields["seed"] = self.seed
        if self.max_tokens is not None:
            fields["max_tokens"] = self.max_tokens
        if self.reasoning_effort is not None:
            fields["reasoning_effort"] = self.reasoning_effort
        return fields

    def content(self) -> dict:
        return {
            "temperature": self.temperature, "top_p": self.top_p, "seed": self.seed,
            "max_tokens": self.max_tokens, "enable_thinking": self.enable_thinking,
            "thinking_switch_sent": self.send_thinking_switch,
            "reasoning_effort": self.reasoning_effort,
        }


@dataclass(frozen=True)
class JudgeAttempt:
    """One dispatch. Kept whether it succeeded or not."""

    ordinal: int
    prompt_sha256: str
    sampling: dict
    status: int
    outcome: str
    returned_model: str = ""
    system_fingerprint: str = ""
    request_id: str = ""
    usage: dict = field(default_factory=dict)
    response_sha256: str = ""
    finish_reason: str = ""
    reason: str = ""
    #: Observed, not requested. Where the endpoint has no thinking switch this is the only
    #: honest statement about whether the model reasoned before answering.
    reasoning_tokens: int = 0

    def content(self) -> dict:
        return {
            "ordinal": self.ordinal, "prompt_sha256": self.prompt_sha256,
            "sampling": self.sampling, "status": self.status, "outcome": self.outcome,
            "returned_model": self.returned_model,
            "system_fingerprint": self.system_fingerprint,
            "request_id": self.request_id, "usage": self.usage,
            "response_sha256": self.response_sha256,
            "finish_reason": self.finish_reason, "reason": self.reason,
            "reasoning_tokens": self.reasoning_tokens,
        }


@dataclass(frozen=True)
class JudgeResponse:
    data: dict
    requested_model: str
    returned_model: str
    usage: dict
    request_id: str
    system_fingerprint: str
    attempts: tuple[JudgeAttempt, ...] = ()


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
        sampling: Optional[SamplingEnvelope] = None,
    ) -> None:
        if not model:
            raise ValueError("judge model must be resolved and non-empty (frozen at launch)")
        self._transport = transport
        self._model = model
        self._api_key = api_key
        self._max_retries = max_retries
        self._sleep = sleeper
        self._sampling = sampling or SamplingEnvelope()

    @property
    def sampling(self) -> SamplingEnvelope:
        return self._sampling

    def _request_body(self, system: str, user: str, sampling: SamplingEnvelope) -> dict:
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        body = {
            "model": self._model,
            "api_key": self._api_key,
            "messages": messages,
            "response_format": {"type": "json_object"},
            **sampling.request_fields(),
        }
        if sampling.send_thinking_switch:
            # Carried the same way the treatment path carries it, so "thinking is off" is a
            # property of the request rather than of a config file nobody sent.
            body["chat_template_kwargs"] = {"enable_thinking": sampling.enable_thinking}
        return body

    async def judge(
        self, system: str, user: str, *, validate: Optional[Callable[[dict], None]] = None
    ) -> JudgeResponse:
        """Return a validated JSON judgment, or raise JudgeUnavailable after bounded retries.

        ``validate`` is an optional local schema check (raises on invalid). It is applied on every
        attempt so an invalid body is retried rather than accepted.
        """
        attempts: list[JudgeAttempt] = []
        last_reason = "unknown"
        truncated_every_time = True
        for index in range(self._max_retries + 1):
            sampling = self._sampling.for_attempt(index)
            body = self._request_body(system, user, sampling)
            prompt_sha = sha256_hex(canonical_json(body["messages"]))
            status, payload = await self._transport(body)

            # This attempt's values are bound as defaults rather than captured. Every call is
            # made inside this iteration, so late binding is harmless today -- but a closure
            # over loop variables that records *which attempt* something happened on is one
            # refactor away from silently attributing every attempt to the last one.
            def record(
                outcome: str,
                *,
                reason: str = "",
                finish: str = "",
                index: int = index,
                prompt_sha: str = prompt_sha,
                sampling=sampling,
                status: int = status,
                payload: dict = payload,
            ) -> JudgeAttempt:
                return JudgeAttempt(
                    ordinal=index, prompt_sha256=prompt_sha, sampling=sampling.content(),
                    status=status, outcome=outcome,
                    returned_model=str(payload.get("model", "")),
                    system_fingerprint=str(payload.get("system_fingerprint", "")),
                    request_id=str(payload.get("id", "")),
                    usage=dict(payload.get("usage", {}) or {}),
                    response_sha256=sha256_hex(canonical_json(payload)),
                    finish_reason=finish, reason=reason,
                    reasoning_tokens=_reasoning_tokens(payload.get("usage") or {}),
                )

            if status != 200:
                truncated_every_time = False
                decision = classify_status(status)
                attempts.append(record("http_error", reason=f"http {status}"))
                if decision is RetryDecision.FAIL_FAST:
                    raise JudgeUnavailable(
                        f"deepseek {status} (fail-fast) after {len(attempts)} attempt(s)")
                last_reason = f"http {status}"
                await self._sleep(0.5 * (2 ** index))
                continue

            finish = _finish_reason(payload)
            if finish == "length":
                # The model answered and the answer did not fit. Recorded as its own
                # outcome rather than laundered into "empty", which is what made a
                # truncated corpus draft look like a flaky connection.
                attempts.append(record("truncated", reason="output cap reached",
                                       finish=finish))
                last_reason = "output cap reached"
                await self._sleep(0.2)
                continue

            truncated_every_time = False
            content = _extract_content(payload)
            if not content:
                attempts.append(record("empty", reason="empty content", finish=finish))
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
                attempts.append(record("invalid", reason=f"invalid body: {e}", finish=finish))
                last_reason = f"invalid body: {e}"
                await self._sleep(0.2)
                continue

            attempts.append(record("accepted", finish=finish))
            return JudgeResponse(
                data=data,
                requested_model=self._model,
                returned_model=str(payload.get("model", "")),
                usage=payload.get("usage", {}) or {},
                request_id=str(payload.get("id", "")),
                system_fingerprint=str(payload.get("system_fingerprint", "")),
                attempts=tuple(attempts),
            )

        if truncated_every_time and attempts:
            raise JudgeTruncated(
                f"every one of {len(attempts)} attempts hit the {self._sampling.max_tokens} "
                "token output cap"
            )
        raise JudgeUnavailable(f"exhausted {self._max_retries} retries: {last_reason}")


def _reasoning_tokens(usage: Mapping[str, Any]) -> int:
    details = usage.get("completion_tokens_details") or {}
    try:
        return int(details.get("reasoning_tokens") or 0)
    except (TypeError, ValueError):
        return 0


def _finish_reason(payload: Mapping[str, Any]) -> str:
    try:
        return str(payload["choices"][0].get("finish_reason") or "")
    except (KeyError, IndexError, TypeError):
        return ""


def _extract_content(payload: dict) -> str:
    try:
        return payload["choices"][0]["message"]["content"] or ""
    except (KeyError, IndexError, TypeError):
        return ""
