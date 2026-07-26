"""Forked-state component trials: one boundary, many variants, one set of input bytes."""

from __future__ import annotations

from pathlib import Path

import pytest

from shapeflow_p1.campaign.fork import (
    ForkOutcome,
    inert_variants,
    plan_forks,
    run_forks,
    write_fork_record,
)
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.odr.checkpoints import (
    CCheckpoint,
    CheckpointStore,
    EvidenceManifest,
    FrozenMessage,
    SamplingEnvelope,
)

REPO = Path(__file__).resolve().parents[2]
BINDING = "e" * 64


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


def _checkpoint(task_id="T1"):
    return CCheckpoint(
        task_id=task_id, researcher_id="R1",
        researcher_messages=(
            FrozenMessage(role="system", content="research"),
            FrozenMessage(role="tool", content="page bytes", tool_call_id="c1"),
        ),
        evidence_manifest=EvidenceManifest(span_ids=("s1",)),
        query_attempt_ids=("qa1",), close_reason="RESEARCH_COMPLETE",
        sampling=SamplingEnvelope(model="m", temperature=0.3, top_p=1.0, max_tokens=512),
    )


def _stored(settings):
    store = CheckpointStore(settings.path("checkpoints"))
    return store, store.put(_checkpoint())


def test_p0_is_always_in_the_fork_set(settings):
    _store, digest = _stored(settings)
    plan = plan_forks(settings, checkpoint_digest=digest, boundary_kind="C",
                      task_id="T1", variant_ids=["C01", "C02"], seed=7,
                      execution_binding_sha256=BINDING)
    assert plan.variant_ids == ["P0", "C01", "C02"]


def test_a_boundary_with_no_p1_variants_is_refused(settings):
    """P0 alone measures nothing. The freeze gate that was supposed to enforce this only
    checked that the assignment was non-empty."""
    _store, digest = _stored(settings)
    with pytest.raises(ValueError, match="no P1 variants"):
        plan_forks(settings, checkpoint_digest=digest, boundary_kind="C",
                   task_id="T1", variant_ids=[], seed=7,
                   execution_binding_sha256=BINDING)


def test_every_fork_of_a_boundary_gets_the_same_bytes(settings):
    """The whole claim of a forked-state trial. It was never true: the screen re-ran the
    full graph per arm, so each arm reached its own boundary with its own history."""
    _store, digest = _stored(settings)
    plan = plan_forks(settings, checkpoint_digest=digest, boundary_kind="C",
                      task_id="T1", variant_ids=["C01", "C02"], seed=7,
                      execution_binding_sha256=BINDING)

    seen = []

    def execute(spec, checkpoint):
        seen.append((spec.variant_id, checkpoint.digest,
                     tuple(m.content for m in checkpoint.researcher_messages)))
        return f"output-{spec.variant_id}"

    import asyncio

    outcomes = asyncio.run(run_forks(settings, plan, execute=execute))
    assert [s[1] for s in seen] == [digest] * 3
    assert len({s[2] for s in seen}) == 1, "two forks saw different input bytes"
    assert [o.state for o in outcomes] == ["COMMITTED"] * 3


def test_fork_ids_differ_by_variant_and_seed(settings):
    _store, digest = _stored(settings)
    a = plan_forks(settings, checkpoint_digest=digest, boundary_kind="C", task_id="T1",
                   variant_ids=["C01"], seed=7,
                   execution_binding_sha256=BINDING)
    b = plan_forks(settings, checkpoint_digest=digest, boundary_kind="C", task_id="T1",
                   variant_ids=["C01"], seed=8,
                   execution_binding_sha256=BINDING)
    assert {f.fork_id for f in a.forks}.isdisjoint({f.fork_id for f in b.forks})
    assert len({f.fork_id for f in a.forks}) == len(a.forks)


def test_a_failing_variant_is_recorded_as_failed_not_as_p0(settings):
    """Falling back is right end to end and wrong here: it turns "this variant broke" into
    "this variant behaved exactly like P0"."""
    _store, digest = _stored(settings)
    plan = plan_forks(settings, checkpoint_digest=digest, boundary_kind="C",
                      task_id="T1", variant_ids=["C01"], seed=7,
                      execution_binding_sha256=BINDING)

    def execute(spec, checkpoint):
        if spec.variant_id == "C01":
            raise RuntimeError("selector returned a dangling id")
        return "output-P0"

    import asyncio

    outcomes = asyncio.run(run_forks(settings, plan, execute=execute))
    by_variant = {o.variant_id: o for o in outcomes}
    assert by_variant["C01"].state == "FAILED_FINAL"
    assert "dangling id" in by_variant["C01"].error
    assert by_variant["C01"].output_sha256 == ""


def test_a_variant_identical_to_p0_from_one_boundary_is_inert(settings):
    """Same input, same output means the variant did nothing. Across separately-run graphs
    this comparison meant nothing, which is part of why an inert arm went undetected."""
    outcomes = [
        ForkOutcome("f0", "P0", "d", state="COMMITTED", output_sha256="aaa"),
        ForkOutcome("f1", "C01", "d", state="COMMITTED", output_sha256="aaa"),
        ForkOutcome("f2", "C02", "d", state="COMMITTED", output_sha256="bbb"),
    ]
    assert inert_variants(outcomes) == ["C01"]


def test_the_fork_record_names_the_shared_boundary(settings, tmp_path):
    _store, digest = _stored(settings)
    plan = plan_forks(settings, checkpoint_digest=digest, boundary_kind="C",
                      task_id="T1", variant_ids=["C01"], seed=7,
                      execution_binding_sha256=BINDING)
    outcomes = [
        ForkOutcome(plan.forks[0].fork_id, "P0", digest, state="COMMITTED",
                    output_sha256="aaa"),
        ForkOutcome(plan.forks[1].fork_id, "C01", digest, state="COMMITTED",
                    output_sha256="aaa"),
    ]
    path = write_fork_record(tmp_path / "forks", plan, outcomes)

    import json

    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["component_trial"] is True
    assert body["checkpoint_digest"] == digest
    assert {f["checkpoint_digest"] for f in body["forks"]} == {digest}
    assert body["inert_variants"] == ["C01"]


def test_a_checkpoint_written_in_one_process_forks_in_another(settings):
    """The point of persisting it. Nothing could fork before, because the stored record was
    two fields with no state and no reader."""
    store, digest = _stored(settings)
    reopened = CheckpointStore(settings.path("checkpoints"))
    assert reopened.get(digest) == store.get(digest)
    assert digest in reopened.digests()
