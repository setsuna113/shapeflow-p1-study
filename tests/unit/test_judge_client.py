"""DeepSeek judge: JSON-mode, retries, and the fail-closed JUDGE_UNAVAILABLE discipline."""

from __future__ import annotations

import pytest

from shapeflow_p1.evaluation.judge_client import DeepSeekJudge, JudgeUnavailable


def _ok_payload(content, *, finish="stop", model="deepseek-chat"):
    return {
        "id": "req-1", "model": model, "system_fingerprint": "fp-1",
        "usage": {"total_tokens": 10},
        "choices": [{"finish_reason": finish, "message": {"content": content}}],
    }


def _transport_seq(*responses):
    calls = {"n": 0}

    async def transport(body):
        i = min(calls["n"], len(responses) - 1)
        calls["n"] += 1
        return responses[i]

    transport.calls = calls  # type: ignore[attr-defined]
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


async def test_length_truncation_treated_as_empty():
    t = _transport_seq(
        (200, _ok_payload('{"partial":', finish="length")),  # truncated -> retry
        (200, _ok_payload('{"done": 1}')),
    )
    judge = DeepSeekJudge(t, "m", "sk-FAKEFAKEFAKEFAKEFAKE")
    assert (await judge.judge("s", "u")).data == {"done": 1}


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
