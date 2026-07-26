"""The C-fork block is internally coherent, and honestly declared as secondary and unwired.

This phase implements protocol section 15.1's *direct node effect* -- a controlled direct effect
at one close boundary. It is deliberately not decision-facing this round, and these tests pin
that: the block must declare itself secondary and unwired, C must carry no per-node estimand
override, and the single decision-facing estimand stays the full-graph COUPLED_SEED_E2E_ITT.

The rest still applies: a block that names arms which are not runnable, declares an estimand
nothing implements, or lists an allowlist that would refuse its own arms is broken whether or
not anything reads it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from shapeflow_p1.campaign.c_fork import c_fork_arms, terminal_mode
from shapeflow_p1.campaign.c_fork_backend import CLOSE_OP_ALLOWLIST, CForkBackend
from shapeflow_p1.campaign.fork import TrialKind

REPO = Path(__file__).resolve().parents[2]


@pytest.fixture()
def week1() -> dict:
    return yaml.safe_load((REPO / "configs" / "week1.yaml").read_text(encoding="utf-8"))


@pytest.fixture()
def decision() -> dict:
    return yaml.safe_load((REPO / "configs" / "decision.yaml").read_text(encoding="utf-8"))


def test_the_phase_declares_the_estimand_the_backend_implements(week1):
    """A config naming an estimand nothing can execute is worse than no config."""
    block = week1["c_fork"]
    assert block["estimand"] == TrialKind.C_FROZEN_CONTINUATION.value
    assert CForkBackend.SUPPORTED == (TrialKind.COMPONENT, TrialKind.C_FROZEN_CONTINUATION)


def test_the_arm_set_includes_the_same_boundary_baseline(week1):
    arms = week1["c_fork"]["arms"]
    assert arms[0] == "P0", "P0 is the baseline every other arm is compared against"
    assert len(set(arms)) == len(arms)
    assert len(arms) >= 2, "a fork set of P0 alone measures nothing"


def test_every_forked_arm_is_a_runnable_variant(week1):
    """An arm that cannot be built would fail per boundary, after the anchor was already paid."""
    variants = yaml.safe_load(
        (REPO / "configs" / "variants.yaml").read_text(encoding="utf-8"))
    runnable = {
        str(entry["variant_id"]) for entry in variants["variants"]
        if entry.get("runnable", True)
    }
    for arm in week1["c_fork"]["arms"]:
        if arm == "P0":
            continue
        assert arm in runnable, f"c_fork arm {arm} is not a runnable variant"


def test_the_allowlist_matches_what_the_backend_enforces(week1):
    """Config and code disagreeing here means the pre-dispatch refusal is not the stated one."""
    assert tuple(week1["c_fork"]["allowed_op_classes"]) == CLOSE_OP_ALLOWLIST


def test_the_terminal_mode_is_config_visible_and_validated(week1):
    assert week1["c_fork"]["terminal_mode"] in ("REPORT", "CLOSE_ONLY")


def test_c_carries_no_per_node_estimand_override(decision):
    """C's decision-facing effect is the full-graph E2E ITT, exactly as H's is.

    An earlier revision made the fork C's primary effect source, on the premise that C fires
    after all research so upstream differences are noise. The vendor graph refutes that:
    supervisor_tools feeds the compressed note back into supervisor_messages and returns
    Command(goto="supervisor"), so C's output conditions whether more research happens.
    Splitting the estimand also makes the core 2x2 incoherent -- the H main effect, the C main
    effect and the HxC interaction have to come from one estimand.
    """
    by_node = decision["structured_increment"]["by_node"]
    c_node = by_node["C_VISIBLE"]
    for banned in (
        "primary_effect_source",
        "secondary_effect_source",
        "secondary_effect_use",
        "mediation_gap_reporting_required",
        "work_fraction_basis",
    ):
        assert banned not in c_node, (
            f"C_VISIBLE re-acquired {banned!r}: C must not carry a per-node estimand override"
        )
    # Structurally the same shape as the other two nodes -- that symmetry is the guarantee.
    assert set(c_node) == set(by_node["WEBPAGE_P1"])


def test_the_fork_is_declared_secondary_and_unwired(week1):
    """Secondary must be a frozen config property, not a claim in a commit message."""
    block = week1["c_fork"]
    assert block["decision_facing"] is False
    assert block["wired"] is False
    assert block["estimand_class"] == "DIRECT_NODE_EFFECT"
    # The single decision-facing estimand, for every node alike.
    assert week1["screen"]["primary_estimand"] == "COUPLED_SEED_E2E_ITT"


def test_the_block_enters_the_execution_binding(tmp_path, monkeypatch):
    """Changing the phase must invalidate the approval, or it is not bound at all."""
    import shutil

    from shapeflow_p1.protocol import compute_binding

    work = tmp_path / "repo"
    shutil.copytree(REPO / "configs", work / "configs")
    shutil.copytree(REPO / "protocol", work / "protocol")
    (work / "patches").mkdir()
    (work / "patches" / "patched_tree.sha256").write_text("a" * 64)

    monkeypatch.setattr("shapeflow_p1.protocol.read_vendor_pin", lambda _repo: "v" * 40)
    before = compute_binding(work, approved_commit="c" * 40).digest

    config = work / "configs" / "week1.yaml"
    body = config.read_text(encoding="utf-8").replace(
        "terminal_mode: REPORT", "terminal_mode: CLOSE_ONLY")
    config.write_text(body, encoding="utf-8")
    after = compute_binding(work, approved_commit="c" * 40).digest

    assert before != after, "the c_fork block does not enter the execution binding"


def test_the_helpers_read_the_live_config():
    """The code path and the file must be the same source of truth."""
    from shapeflow_p1.campaign.settings import Settings

    settings = Settings.load(REPO, data_root=Path("/tmp/c-fork-config-check"))
    assert c_fork_arms(settings)[0] == "P0"
    assert terminal_mode(settings) in ("REPORT", "CLOSE_ONLY")
