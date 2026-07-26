"""The C fork's honesty claims, each with the counter-example that would violate it.

The fork's whole value is that P0 and every C variant demonstrably start from *one* boundary
and do *no* upstream work. Neither is asserted by the backend -- both are proved: the close
hook rebuilds a ``CCheckpoint`` from whatever state it actually received, and the arm is
refused unless that re-derived digest is the planned one.

These tests drive vendor's real hook (``run_close_strategy``) through a stand-in node shaped
exactly like the patched ``compress_research``, so the digest really is re-derived rather than
echoed back by a fake. Only the graph wrapper is a double; the proof is the production path.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from shapeflow_p1.campaign.c_fork_backend import (
    CLOSE_OP_ALLOWLIST,
    CForkBackend,
    UpstreamCallDuringFork,
    reconstruct_researcher_state,
)
from shapeflow_p1.campaign.fork import ForkCapabilityError, ForkSpec, TrialKind
from shapeflow_p1.odr.checkpoints import (
    CCheckpoint,
    EvidenceManifest,
    FrozenMessage,
    SamplingEnvelope,
)
from shapeflow_p1.odr.continuation import (
    ContinuationEnvelope,
    child_slot_key,
    note_slots_from_supervisor_messages,
)
from shapeflow_p1.odr.hooks import StrategyBundle
from shapeflow_p1.odr.vendor_hooks import RunBinding

ANCHOR_DATE = "Sat Jul 26, 2026"
ANCHOR_EPOCH = "epoch-anchor"
CALL_ID = "call-child-2"
BINDING = "b" * 64


# --- the world -----------------------------------------------------------------------------


def _sampling() -> SamplingEnvelope:
    return SamplingEnvelope(
        model="Qwen3-14B-AWQ", temperature=0.3, top_p=1.0, max_tokens=8192, seed=7
    )


def _checkpoint(close_reason: str = "RESEARCH_COMPLETE") -> CCheckpoint:
    return CCheckpoint(
        task_id="T1",
        researcher_id=f"T1:r0/child-0-1-{child_slot_key(CALL_ID)}",
        researcher_messages=(
            FrozenMessage(role="human", content="research the thing"),
            FrozenMessage(role="ai", content="done researching"),
        ),
        evidence_manifest=EvidenceManifest(span_ids=()),
        query_attempt_ids=(),
        close_reason=close_reason,
        sampling=_sampling(),
    )


def _envelope(**overrides) -> ContinuationEnvelope:
    from langchain_core.messages import AIMessage, ToolMessage

    supervisor_messages = [
        AIMessage(content="plan"),
        ToolMessage(content="note from sibling", tool_call_id="call-child-1",
                    name="ConductResearch"),
        ToolMessage(content="a thought", tool_call_id="think-1", name="think_tool"),
        ToolMessage(content="ANCHOR NOTE", tool_call_id=CALL_ID, name="ConductResearch"),
    ]
    notes = ["note from sibling", "a thought", "ANCHOR NOTE"]
    defaults = dict(
        task_id="T1",
        seed=7,
        research_brief="the brief",
        root_messages=(FrozenMessage(role="human", content="the question"),),
        notes=note_slots_from_supervisor_messages(supervisor_messages, notes),
        anchor_today_str=ANCHOR_DATE,
        anchor_engine_epoch=ANCHOR_EPOCH,
        anchor_run_ref="anchor-run-1",
    )
    defaults.update(overrides)
    return ContinuationEnvelope(**defaults)


class _P1Close:
    """A P1 close strategy: returns a handoff, like close_visible does."""

    def __init__(self) -> None:
        self.seen_digests: list[str] = []
        self.last_outcome = None
        self.last_outcomes = ()

    async def close_researcher(self, *, task_ctx, checkpoint):
        self.seen_digests.append(checkpoint.digest)
        from shapeflow_p1.odr.hooks import ResearcherHandoff

        return ResearcherHandoff(compressed_research="P1 SELECTED IDS", raw_notes=())


class _P0Close:
    """The explicit-P0 close: declines so vendor's own compression runs."""

    def __init__(self) -> None:
        self.seen_digests: list[str] = []
        self.last_outcome = None
        self.last_outcomes = ()

    async def close_researcher(self, *, task_ctx, checkpoint):
        from shapeflow_p1.odr.vendor_hooks import _UseVendorCompression

        self.seen_digests.append(checkpoint.digest)
        raise _UseVendorCompression()


def _bundle(close) -> StrategyBundle:
    return StrategyBundle(variant_id="V", page=None, close=close)


def _fake_close_graph(vendor_output: str = "VENDOR P0 COMPRESSION"):
    """A graph shaped exactly like the patched ``compress_research``.

    The hook call, the ``return`` of its handoff, and the vendor fallback below it mirror
    ``patches/odr_p1_hooks.patch`` so the real derivation path is exercised.
    """
    from langgraph.graph import END, START, StateGraph

    async def compress_research(state, config=None):
        from shapeflow_p1.odr import vendor_hooks as _sf

        handoff = await _sf.run_close_strategy(
            researcher_messages=list(state["researcher_messages"]),
            close_reason="RESEARCH_COMPLETE",
        )
        if handoff is not None:
            return handoff
        return {"compressed_research": vendor_output, "raw_notes": []}

    builder = StateGraph(dict)
    builder.add_node("compress_research", compress_research)
    builder.add_edge(START, "compress_research")
    builder.add_edge("compress_research", END)
    return builder.compile()


def _fake_report_graph(recorder: list):
    from langgraph.graph import END, START, StateGraph

    async def final_report_generation(state, config=None):
        recorder.append(dict(state))
        notes = state.get("notes") or []
        if isinstance(notes, dict):
            notes = notes.get("value", [])
        return {"final_report": "REPORT OVER " + " | ".join(notes)}

    builder = StateGraph(dict)
    builder.add_node("final_report_generation", final_report_generation)
    builder.add_edge(START, "final_report_generation")
    builder.add_edge("final_report_generation", END)
    return builder.compile()


def _backend(close, *, envelope=None, epochs=None, today=None, terminal_mode="REPORT",
             reports=None, vendor_output="VENDOR P0 COMPRESSION"):
    envelope = envelope if envelope is not None else _envelope()
    epoch_values = list(epochs or [ANCHOR_EPOCH, ANCHOR_EPOCH])

    def engine_epoch() -> str:
        return epoch_values.pop(0) if len(epoch_values) > 1 else epoch_values[0]

    def run_binding_for(spec, checkpoint):
        return RunBinding(
            task_id=checkpoint.task_id,
            researcher_id=checkpoint.researcher_id,
            attempt_id=spec.fork_id,
            task_ctx=None,
            component_trial=True,
            sampling=checkpoint.sampling,
        )

    return CForkBackend(
        envelope=envelope,
        strategy_for=lambda variant_id: _bundle(close),
        run_binding_for=run_binding_for,
        odr_config_for=lambda spec: {},
        today_str=lambda: today or ANCHOR_DATE,
        engine_epoch=engine_epoch,
        terminal_mode=terminal_mode,
        close_graph=_fake_close_graph(vendor_output),
        report_graph=_fake_report_graph(reports if reports is not None else []),
    )


def _spec(digest: str, variant_id: str = "C01",
          trial_kind: TrialKind = TrialKind.C_FROZEN_CONTINUATION) -> ForkSpec:
    return ForkSpec(
        fork_id=f"fork-{variant_id}",
        checkpoint_digest=digest,
        boundary_kind="C",
        task_id="T1",
        variant_id=variant_id,
        seed=7,
        execution_binding_sha256=BINDING,
        trial_kind=trial_kind,
    )


# --- what the backend will and will not claim ----------------------------------------------


def test_the_backend_refuses_estimands_it_cannot_perform():
    """H_E2E and nested HxC need mid-trajectory resume; substituting a re-run would be a lie."""
    backend = _backend(_P1Close())
    assert backend.supports(TrialKind.C_FROZEN_CONTINUATION, "C")
    assert backend.supports(TrialKind.COMPONENT, "C")
    assert not backend.supports(TrialKind.H_E2E, "H")
    assert not backend.supports(TrialKind.HXC_NESTED, "C")
    assert not backend.supports(TrialKind.C_FROZEN_CONTINUATION, "H")


def test_capture_is_refused_rather_than_re_running_the_upstream():
    backend = _backend(_P1Close())
    with pytest.raises(ForkCapabilityError, match="does not capture boundaries"):
        asyncio.run(backend.capture_boundaries(task_id="T1", question="q", seed=7))


# --- the digest proof ----------------------------------------------------------------------


def test_p0_and_every_c_variant_start_from_the_same_proved_digest():
    checkpoint = _checkpoint()
    starts = set()
    for close, variant in ((_P0Close(), "P0"), (_P1Close(), "C01"), (_P1Close(), "C02")):
        execution = asyncio.run(
            _backend(close).execute(_spec(checkpoint.digest, variant), checkpoint)
        )
        starts.add(execution.start_checkpoint_digest)
        assert execution.upstream_research_calls == 0
        assert execution.seed_applied is True

    assert starts == {checkpoint.digest}, "the arms did not share one boundary"


def test_the_strategy_receives_the_boundary_that_was_planned():
    """The digest is re-derived from the reconstructed state, not copied from the spec."""
    checkpoint = _checkpoint()
    close = _P1Close()
    asyncio.run(_backend(close).execute(_spec(checkpoint.digest), checkpoint))
    assert close.seen_digests == [checkpoint.digest]


def test_a_checkpoint_that_rebuilds_to_another_digest_is_refused():
    """A tampered boundary reconstructs into a valid-looking state; the proof is what catches it."""
    checkpoint = _checkpoint()
    tampered = _spec("f" * 64)
    with pytest.raises(ForkCapabilityError, match="did not start where it claims"):
        asyncio.run(_backend(_P1Close()).execute(tampered, checkpoint))


def test_a_mutated_message_moves_the_digest_and_is_refused():
    original = _checkpoint()
    mutated = CCheckpoint(
        task_id=original.task_id,
        researcher_id=original.researcher_id,
        researcher_messages=(
            FrozenMessage(role="human", content="research something else"),
            FrozenMessage(role="ai", content="done researching"),
        ),
        evidence_manifest=original.evidence_manifest,
        query_attempt_ids=original.query_attempt_ids,
        close_reason=original.close_reason,
        sampling=original.sampling,
    )
    with pytest.raises(ForkCapabilityError, match="did not start where it claims"):
        asyncio.run(_backend(_P1Close()).execute(_spec(original.digest), mutated))


# --- zero upstream ---------------------------------------------------------------------------


def test_a_search_during_a_fork_is_refused():
    """The frozen search never reaches the provider, so only the in-process poison sees it.

    The allowlist refuses forbidden op classes before dispatch, but a frozen-corpus lookup is
    served locally and would never appear there -- yet it is still upstream work one arm did
    and its pair did not.
    """
    import open_deep_research.utils as vendor_utils

    class _SearchingClose(_P1Close):
        async def close_researcher(self, *, task_ctx, checkpoint):
            await vendor_utils.tavily_search_async(["q"])
            return await super().close_researcher(task_ctx=task_ctx, checkpoint=checkpoint)

    from shapeflow_p1.odr.vendor_hooks import ComponentTrialFailure

    checkpoint = _checkpoint()
    # It surfaces as a ComponentTrialFailure because forks run with component_trial=True: a
    # failed C treatment must be recorded as a failure, never quietly replaced by vendor's
    # compression, which would turn "this variant broke" into "it behaved exactly like P0".
    with pytest.raises(
        (ComponentTrialFailure, UpstreamCallDuringFork, ForkCapabilityError)
    ) as excinfo:
        asyncio.run(_backend(_SearchingClose()).execute(_spec(checkpoint.digest), checkpoint))
    assert "C fork attempted a search" in str(excinfo.value)
    assert "STRATEGY_ERROR" in str(excinfo.value), "the arm was not recorded as a failure"


def test_the_search_poison_is_removed_afterwards():
    """A fork must not leave the vendor module poisoned for the rest of the process."""
    import open_deep_research.utils as vendor_utils

    before = vendor_utils.tavily_search_async
    checkpoint = _checkpoint()
    asyncio.run(_backend(_P1Close()).execute(_spec(checkpoint.digest), checkpoint))
    assert vendor_utils.tavily_search_async is before


def test_the_close_allowlist_names_only_post_boundary_ops():
    assert set(CLOSE_OP_ALLOWLIST) == {
        "COMPRESSOR_P0", "COMPRESSOR_P1_SELECTOR", "COMPRESSOR_SHORT_PROSE", "FINAL_WRITER",
    }


def test_every_configured_fork_arm_is_admitted_by_the_close_allowlist():
    """The allowlist must cover every arm the config actually lists.

    The literal above pins today's set; this pins the *invariant*. C00-PROSE was a listed fork
    arm whose op class was missing from the allowlist, so the arm would have been refused at its
    own boundary -- silently removing C_ID_VS_PROSE from the Holm family. A test comparing the
    allowlist only against a hand-written literal cannot see that; this one can.
    """
    import yaml

    from shapeflow_p1.strategies.factory import load_registry

    arms = yaml.safe_load(Path("configs/week1.yaml").read_text())["c_fork"]["arms"]
    registry = load_registry(Path("configs"))

    for arm in arms:
        variant = registry.get(str(arm))
        assert variant is not None, f"fork arm {arm!r} is not in the variant registry"
        if variant.node == "P0" or variant.selector_backend != "LLM":
            continue  # vendor prose / CPU_LEXICAL issue no selector-family call
        op = ("COMPRESSOR_SHORT_PROSE" if variant.contract == "SHORT_PROSE"
              else "COMPRESSOR_P1_SELECTOR")
        assert op in CLOSE_OP_ALLOWLIST, (
            f"fork arm {arm!r} dispatches {op}, which its own allowlist would refuse"
        )


# --- the frozen continuation -----------------------------------------------------------------


def test_exactly_one_note_slot_is_substituted():
    """Every sibling note is byte-identical; only the forking child's own note changes."""
    checkpoint = _checkpoint()
    reports: list = []
    execution = asyncio.run(
        _backend(_P1Close(), reports=reports).execute(_spec(checkpoint.digest), checkpoint)
    )

    assert len(reports) == 1
    notes = reports[0]["notes"]["value"]
    assert notes == ["note from sibling", "a thought", "P1 SELECTED IDS"]
    assert "P1 SELECTED IDS" in execution.output


def test_the_note_vector_is_overridden_not_appended():
    """A bare list is *added* by override_reducer, leaving the anchor's note alongside the arm's."""
    checkpoint = _checkpoint()
    reports: list = []
    asyncio.run(_backend(_P1Close(), reports=reports).execute(_spec(checkpoint.digest), checkpoint))
    assert reports[0]["notes"]["type"] == "override"


def test_p0_writes_its_report_over_vendors_own_compression():
    checkpoint = _checkpoint()
    reports: list = []
    asyncio.run(
        _backend(_P0Close(), reports=reports, vendor_output="VENDOR TEXT")
        .execute(_spec(checkpoint.digest, "P0"), checkpoint)
    )
    assert reports[0]["notes"]["value"][2] == "VENDOR TEXT"


def test_a_boundary_whose_note_slot_is_ambiguous_is_refused():
    """Excluded before any arm is offered, so the ITT denominator is untouched."""
    checkpoint = _checkpoint()
    orphan = CCheckpoint(
        task_id=checkpoint.task_id,
        researcher_id="T1:r0/child-0-9-" + child_slot_key("a-call-nobody-recorded"),
        researcher_messages=checkpoint.researcher_messages,
        evidence_manifest=checkpoint.evidence_manifest,
        query_attempt_ids=checkpoint.query_attempt_ids,
        close_reason=checkpoint.close_reason,
        sampling=checkpoint.sampling,
    )
    with pytest.raises(ForkCapabilityError, match="no unique note slot"):
        asyncio.run(_backend(_P1Close()).execute(_spec(orphan.digest), orphan))


def test_close_only_mode_does_not_claim_a_terminal_report():
    """The fail-closed degradation must be visible, not a quietly shorter run."""
    checkpoint = _checkpoint()
    execution = asyncio.run(
        _backend(_P1Close(), terminal_mode="CLOSE_ONLY")
        .execute(_spec(checkpoint.digest), checkpoint)
    )
    assert execution.output == "P1 SELECTED IDS"
    assert execution.terminal_close_checkpoint_digest == ""


# --- pairing guards ---------------------------------------------------------------------------


def test_an_engine_restart_between_anchor_and_arm_voids_the_pair():
    checkpoint = _checkpoint()
    backend = _backend(_P1Close(), epochs=["epoch-different"])
    with pytest.raises(ForkCapabilityError, match="came from a different boot"):
        asyncio.run(backend.execute(_spec(checkpoint.digest), checkpoint))


def test_an_engine_restart_during_the_arm_voids_it():
    checkpoint = _checkpoint()
    backend = _backend(_P1Close(), epochs=[ANCHOR_EPOCH, "epoch-restarted"])
    with pytest.raises(ForkCapabilityError, match="engine epoch changed during the arm"):
        asyncio.run(backend.execute(_spec(checkpoint.digest), checkpoint))


def test_a_boundary_straddling_utc_midnight_is_refused():
    """get_today_str is formatted into both prompts, so the arms would differ beyond treatment."""
    checkpoint = _checkpoint()
    backend = _backend(_P1Close(), today="Sun Jul 27, 2026")
    with pytest.raises(ForkCapabilityError, match="differ by more than the treatment"):
        asyncio.run(backend.execute(_spec(checkpoint.digest), checkpoint))


# --- state reconstruction -----------------------------------------------------------------------


def test_tool_call_iterations_is_reconstructed_from_the_messages():
    """Defaulting it would change the close reason, hence the digest, hence fail the proof."""
    checkpoint = CCheckpoint(
        task_id="T1",
        researcher_id="T1:r0",
        researcher_messages=(
            FrozenMessage(role="human", content="go"),
            FrozenMessage(role="ai", content="searching", tool_calls=(
                __import__("shapeflow_p1.odr.checkpoints", fromlist=["FrozenToolCall"])
                .FrozenToolCall(id="c1", name="tavily_search", args_canonical="{}"),
            )),
            FrozenMessage(role="tool", content="result", tool_call_id="c1", status="success"),
            FrozenMessage(role="ai", content="done"),
        ),
        evidence_manifest=EvidenceManifest(span_ids=()),
        query_attempt_ids=(),
        close_reason="RESEARCH_COMPLETE",
        sampling=_sampling(),
    )
    state = reconstruct_researcher_state(checkpoint, research_topic="topic")
    assert state["tool_call_iterations"] == 1
    assert state["research_topic"] == "topic"
    assert len(state["researcher_messages"]) == 4
