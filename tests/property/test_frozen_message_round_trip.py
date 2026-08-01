"""Property: freeze -> thaw -> freeze is the identity, for any message shape.

This is the single assumption the C-boundary fork rests on, and until now it had neither a
caller nor a test. The fork's honesty claim is *proved*, not asserted: it reconstructs a
researcher state from a stored ``CCheckpoint`` via :func:`thaw_message`, re-enters
``compress_research``, and requires the close hook to re-derive **the same digest** as the
boundary it planned to fork from. Every field that fails to survive the round trip changes
that digest, so the fork fails closed on a checkpoint that is in fact perfectly good.

Two shapes made this fail in practice, both from testing a value for truthiness rather than
for presence: a message whose ``name`` or ``id`` is ``""`` thawed to ``None``, and a
``ToolMessage`` whose ``status`` is ``""`` lost it. Hand-picked examples miss exactly this
class of bug, which is why the check is a property over generated shapes.

The stronger of the two assertions is the rendering one: ``get_buffer_string`` is what the
compressor prompt is actually built from, so two states that render identically are the same
input to the model even if some non-rendered field drifted. Both are required -- the digest
protects the pairing, the rendering protects the treatment.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from shapeflow.odr.adapter import _canon, freeze_messages, thaw_message
from shapeflow.odr.checkpoints import (
    CCheckpoint,
    EvidenceManifest,
    FrozenMessage,
    FrozenToolCall,
    SamplingEnvelope,
)

# `None` and `""` are deliberately both in range: they are different values and the round trip
# has to keep them apart.
_optional_text = st.one_of(st.none(), st.just(""), st.text(min_size=1, max_size=12))

_json_scalar = st.one_of(
    st.none(), st.booleans(), st.integers(-1000, 1000), st.text(max_size=8)
)
_json_object = st.dictionaries(st.text(min_size=1, max_size=6), _json_scalar, max_size=3)

_text_content = st.text(max_size=40)
_block_content = st.lists(
    st.fixed_dictionaries({"type": st.just("text"), "text": st.text(max_size=20)}),
    min_size=1,
    max_size=3,
)
_content = st.one_of(_text_content, _block_content)

_tool_call = st.builds(
    FrozenToolCall,
    id=st.text(min_size=1, max_size=8),
    name=st.sampled_from(["tavily_search", "ConductResearch", "think_tool"]),
    # Canonicalize with the same function the code uses. Hand-rolled json.dumps differs on
    # non-ASCII escaping, which would fail the round trip for a reason that is purely an
    # artefact of how the *test* built its input.
    args_canonical=_json_object.map(_canon),
)


@st.composite
def _frozen_message(draw) -> FrozenMessage:
    role = draw(st.sampled_from(["ai", "human", "system", "tool"]))
    tool_calls = tuple(draw(st.lists(_tool_call, max_size=2))) if role == "ai" else ()
    return FrozenMessage(
        role=role,
        content=draw(_content),
        tool_calls=tool_calls,
        name=draw(_optional_text),
        # langchain requires tool_call_id on a ToolMessage, so a frozen tool message always
        # has one (possibly ""); None is unreachable and is refused rather than coerced.
        tool_call_id=(
            draw(st.one_of(st.just(""), st.text(min_size=1, max_size=8)))
            if role == "tool"
            else None
        ),
        message_id=draw(_optional_text),
        additional_kwargs_canonical=draw(_json_object.map(_canon)),
        response_metadata_canonical=draw(_json_object.map(_canon)),
        # Only AIMessage carries usage_metadata; every other role freezes to "{}" because
        # `freeze_message` reads it with getattr(..., None). Verified against langchain 1.4.8.
        usage_metadata_canonical=(
            draw(st.sampled_from([
                "{}",
                _canon({"input_tokens": 10, "output_tokens": 2, "total_tokens": 12}),
            ]))
            if role == "ai"
            else "{}"
        ),
        artifact_canonical=(
            draw(st.one_of(st.none(), _json_object.map(_canon))) if role == "tool" else None
        ),
        invalid_tool_calls_canonical="[]",
        # Only the reachable space. `freeze_message` reads `getattr(message, "status", None)`,
        # and langchain's ToolMessage.status is a Literal["success", "error"] defaulting to
        # "success" -- so a frozen tool message always carries one of those two, and a
        # non-tool message always carries None. Generating "" or None-on-a-tool here would
        # test a state no freeze can produce; `thaw_message` rejects those loudly instead,
        # which is asserted separately below.
        status=draw(st.sampled_from(["success", "error"])) if role == "tool" else None,
    )


def _c_checkpoint(messages: tuple[FrozenMessage, ...]) -> CCheckpoint:
    return CCheckpoint(
        task_id="T1",
        researcher_id="R1",
        researcher_messages=messages,
        evidence_manifest=EvidenceManifest(span_ids=()),
        query_attempt_ids=(),
        close_reason="RESEARCH_COMPLETE",
        sampling=SamplingEnvelope(
            model="Qwen3-14B-AWQ", temperature=0.3, top_p=1.0, max_tokens=8192, seed=1
        ),
    )


@settings(max_examples=200, suppress_health_check=[HealthCheck.too_slow])
@given(_frozen_message())
def test_freeze_thaw_freeze_is_the_identity_on_one_message(frozen: FrozenMessage):
    assert freeze_messages([thaw_message(frozen)])[0] == frozen


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_frozen_message(), min_size=1, max_size=5))
def test_freeze_thaw_freeze_preserves_the_c_checkpoint_digest(messages: list[FrozenMessage]):
    """The fork's proof: the reconstructed state must hash to the boundary it forked from."""
    original = _c_checkpoint(tuple(messages))
    rebuilt = _c_checkpoint(freeze_messages([thaw_message(m) for m in messages]))
    assert rebuilt.digest == original.digest


@settings(max_examples=150, suppress_health_check=[HealthCheck.too_slow])
@given(st.lists(_frozen_message(), min_size=1, max_size=5))
def test_the_thawed_state_renders_exactly_as_the_original(messages: list[FrozenMessage]):
    """What the compressor sees is the rendered message list, so that must match too."""
    from langchain_core.messages import get_buffer_string

    thawed = [thaw_message(m) for m in messages]
    twice = [thaw_message(m) for m in freeze_messages(thawed)]
    assert get_buffer_string(thawed) == get_buffer_string(twice)


def test_an_empty_name_is_not_silently_dropped():
    """The concrete regression: `if frozen.name:` turned "" into None.

    Kept as an explicit case next to the property so the failure it describes stays readable
    when someone changes this code again.
    """
    frozen = FrozenMessage(role="ai", content="hi", name="", message_id="")
    round_tripped = freeze_messages([thaw_message(frozen)])[0]
    assert round_tripped.name == ""
    assert round_tripped.message_id == ""
    assert round_tripped == frozen


def test_a_tool_status_that_no_freeze_could_produce_is_refused():
    """Silently defaulting it would forge a state that re-freezes to a different digest.

    A fork proves it started from the planned boundary by re-deriving the checkpoint digest.
    If thawing quietly turned an impossible status into "success", a tampered checkpoint would
    reconstruct into a *valid-looking* state, which is precisely the substitution the digest
    proof exists to catch.
    """
    import pytest

    for bad in ("", None, "SUCCESS", "ok"):
        frozen = FrozenMessage(
            role="tool", content="page text", tool_call_id="c1", name="tavily_search", status=bad
        )
        with pytest.raises(ValueError, match="did not come from a real message"):
            thaw_message(frozen)


def test_a_tool_message_without_a_tool_call_id_is_refused():
    """`or ""` silently rewrote None to "", which moves the digest the fork proof compares."""
    import pytest

    frozen = FrozenMessage(role="tool", content="page text", tool_call_id=None, status="success")
    with pytest.raises(ValueError, match="did not come from a real message"):
        thaw_message(frozen)
