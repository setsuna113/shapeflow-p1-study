"""A one-arm campaign is a design, not an unimplemented case.

The P0 competence pilot (protocol §5.3) runs a single arm, deliberately: it decides whether the
frozen retriever supports the task at all, and putting a P1 arm in it would make the retriever's
admission condition depend on treatment data.

Williams squares are undefined below two treatments, so ``build_blocks`` used to die three frames
down with "need at least 2 treatments" -- the shape of a case nobody had thought about rather than
one that was decided. It is decided here: one arm has exactly one order, so carryover balance is
vacuously satisfied, and the manifest says which of the two it is.
"""

from __future__ import annotations

import pytest

from shapeflow.campaign.schedule import ArmSpec, build_blocks

P0 = ArmSpec(arm_id="P0", page_variant="P0", close_variant="P0")
H = ArmSpec(arm_id="H_MARKDOWN_ID", page_variant="H02", close_variant="P0")


def _blocks(arms, task_ids=("1", "2", "3"), second_seed_fraction=0.0):
    return build_blocks(
        execution_binding_sha256="a" * 64, protocol_sha="b" * 64, split="b1_select",
        task_ids=list(task_ids), arms=list(arms), seeds=[1, 2], layer="causal_native",
        claim_scope="test", second_seed_fraction=second_seed_fraction)


def test_a_single_arm_campaign_schedules():
    manifest = _blocks([P0])
    assert len(manifest.blocks) == 3
    assert [c.arm.arm_id for b in manifest.blocks for c in b.cells] == ["P0"] * 3


def test_the_manifest_says_the_balance_is_vacuous_not_achieved():
    """The distinction is the point: an unbalanced design and a design with nothing to balance
    look identical in a manifest that only records 'balanced'."""
    notes = _blocks([P0]).notes
    assert notes["randomization"] == "single_arm_no_ordering"
    assert notes["carryover_balance"] == "VACUOUS_SINGLE_ARM"


def test_a_single_arm_second_seed_still_gets_its_replicate():
    manifest = _blocks([P0], second_seed_fraction=0.34)
    replicates = sorted((b.task_id, b.replicate_id) for b in manifest.blocks)
    assert len(manifest.blocks) == 4
    assert len({task for task, _ in replicates}) == 3
    assert any(replicate == "1" for _, replicate in replicates)


def test_two_arms_still_take_the_williams_path():
    """The single-arm branch must not swallow the real design."""
    notes = _blocks([P0, H]).notes
    assert notes.get("randomization") != "single_arm_no_ordering"
    assert notes.get("carryover_balance") != "VACUOUS_SINGLE_ARM"
    assert {c.arm.arm_id for b in _blocks([P0, H]).blocks for c in b.cells} == {"P0", H.arm_id}


def test_zero_arms_is_still_refused():
    with pytest.raises(ValueError, match="no arms"):
        _blocks([])
