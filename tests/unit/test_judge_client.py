"""DeepSeek judge: JSON-mode, retries, and the fail-closed JUDGE_UNAVAILABLE discipline."""

from __future__ import annotations

import pytest

from shapeflow.bench.grading.judge_client import (
    DeepSeekJudge,
    JudgeTruncated,
    JudgeUnavailable,
    SamplingEnvelope,
)


def _ok_payload(content, *, finish="stop", model="deepseek-chat"):
    return {
        "id": "req-1", "model": model, "system_fingerprint": "fp-1",
        "usage": {"total_tokens": 10},
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }


def _transport_seq(*responses):
    calls = {"n": 0}
    bodies: list[dict] = []

    async def transport(body):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        bodies.append(dict(body))
        return responses[i]

    transport.calls = calls  # type: ignore[attr-defined]
    transport.bodies = bodies  # type: ignore[attr-defined]
    return transport


async def test_happy_json_judgment():
    t = _transport_seq((200, _ok_payload('{"verdict": "A", "score": 3}')))
    judge = DeepSeekJudge(t, "deepseek-chat", "sk-FAKEFAKEFAKEFAKEFAKE")
    resp = await judge.judge("sys", "user")
    assert resp.data == {"verdict": "A", "score": 3}
    assert resp.returned_model == "deepseek-chat"
    assert resp.system_fingerprint == "fp-1"


async def test_empty_then_success_retries():
    t = _transport_seq(
        (200, _ok_payload("")),                       # empty -> retry
        (200, _ok_payload('{"ok": true}')),
    )
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    resp = await judge.judge("s", "u")
    assert resp.data == {"ok": True}
    assert t.calls["n"] == 2


async def test_length_truncation_is_retried_and_recorded_as_truncation():
    t = _transport_seq(
        (200, _ok_payload('{"partial":', finish="length")),  # truncated -> retry
        (200, _ok_payload('{"done": 1}')),
    )
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    response = await judge.judge("s", "u")
    assert response.data == {"done": 1}
    assert [a.outcome for a in response.attempts] == ["truncated", "accepted"]
    assert response.attempts[0].finish_reason == "length"


async def test_a_body_that_never_fits_the_cap_is_its_own_failure():
    """The model answered every time. Calling that "unavailable" hides a cap that is too
    small behind what looks like a flaky connection."""
    t = _transport_seq(*[(200, _ok_payload('{"partial":', finish="length"))] * 4)
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE", max_retries=3,
                          sampling=SamplingEnvelope(max_tokens=64))
    with pytest.raises(JudgeTruncated, match="64"):
        await judge.judge("s", "u")


async def test_the_sampling_envelope_is_actually_sent():
    t = _transport_seq((200, _ok_payload('{"ok": 1}')))
    judge = DeepSeekJudge(
        t, "m", "sk-FAKEFAKEFAKEFAKEFAKE",
        sampling=SamplingEnvelope(temperature=0.4, top_p=0.9, seed=7, max_tokens=4096),
    )
    response = await judge.judge("s", "u")
    body = t.bodies[0]
    assert body["temperature"] == 0.4
    assert body["top_p"] == 0.9
    assert body["seed"] == 7
    assert body["max_tokens"] == 4096
    # DeepSeek has no thinking switch, so none is invented; the envelope records that it
    # was not sent, and the attempt records the reasoning tokens that came back.
    assert "chat_template_kwargs" not in body
    assert response.attempts[0].sampling["seed"] == 7
    assert response.attempts[0].sampling["thinking_switch_sent"] is False


async def test_the_thinking_switch_is_sent_only_where_it_exists():
    t = _transport_seq((200, _ok_payload('{"ok": 1}')))
    judge = DeepSeekJudge(
        t, "m", "sk-FAKEFAKEFAKEFAKEFAKE",
        sampling=SamplingEnvelope(enable_thinking=False, send_thinking_switch=True),
    )
    await judge.judge("s", "u")
    assert t.bodies[0]["chat_template_kwargs"] == {"enable_thinking": False}


async def test_reasoning_is_measured_rather_than_assumed():
    payload = _ok_payload('{"ok": 1}')
    payload["usage"] = {"prompt_tokens": 10, "completion_tokens": 900,
                        "completion_tokens_details": {"reasoning_tokens": 460}}
    t = _transport_seq((200, payload))
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    response = await judge.judge("s", "u")
    assert response.attempts[0].reasoning_tokens == 460


async def test_a_retry_advances_the_seed_so_it_is_a_different_request():
    """Re-sending byte-identical bytes is not a retry: it lands on the same logical call,
    which either returns the same rejected body or pays for it twice."""
    t = _transport_seq(
        (200, _ok_payload("not json")),
        (200, _ok_payload('{"ok": 1}')),
    )
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE",
                          sampling=SamplingEnvelope(seed=100))
    response = await judge.judge("s", "u")
    assert [b["seed"] for b in t.bodies] == [100, 101]
    assert [a.sampling["seed"] for a in response.attempts] == [100, 101]


async def test_every_attempt_is_kept_with_its_model_and_fingerprint():
    t = _transport_seq(
        (500, {"error": "boom"}),
        (200, _ok_payload('{"ok": 1}')),
    )
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    response = await judge.judge("s", "u")
    assert len(response.attempts) == 2
    assert response.attempts[0].outcome == "http_error"
    assert response.attempts[0].status == 500
    assert response.attempts[1].outcome == "accepted"
    assert response.attempts[1].returned_model == response.returned_model
    assert all(a.prompt_sha256 for a in response.attempts)


async def test_invalid_json_exhausts_to_unavailable_never_imputes():
    t = _transport_seq((200, _ok_payload("not json at all")))
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE", max_retries=2)
    with pytest.raises(JudgeUnavailable):
        await judge.judge("s", "u")
    # tried initial + 2 retries = 3 transport calls
    assert t.calls["n"] == 3


async def test_4xx_fails_fast():
    t = _transport_seq((401, {"error": "bad key"}))
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    with pytest.raises(JudgeUnavailable, match="fail-fast"):
        await judge.judge("s", "u")
    assert t.calls["n"] == 1  # no retry on a 4xx


async def test_5xx_retries_then_unavailable():
    t = _transport_seq((503, {}))
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE", max_retries=2)
    with pytest.raises(JudgeUnavailable):
        await judge.judge("s", "u")
    assert t.calls["n"] == 3  # retried the 5xx


async def test_schema_validation_rejects_and_retries():
    def require_verdict(d):
        if "verdict" not in d:
            raise ValueError("missing verdict")

    t = _transport_seq(
        (200, _ok_payload('{"nope": 1}')),                 # fails schema -> retry
        (200, _ok_payload('{"verdict": "B"}')),
    )
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    resp = await judge.judge("s", "u", validate=require_verdict)
    assert resp.data == {"verdict": "B"}
    assert t.calls["n"] == 2


async def test_empty_model_name_is_rejected():
    with pytest.raises(ValueError, match="frozen at launch"):
        DeepSeekJudge(_transport_seq((200, _ok_payload("{}"))), "", "sk-FAKEFAKEFAKEFAKEFAKE")
