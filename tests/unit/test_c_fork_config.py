"""The C-fork phase is wired and bound, not dormant code behind a default-off flag.

The point of landing the fork before the freeze was that it become a real experimental path
inside the approval hash. A block that exists but changes no digest, names arms that are not
runnable, or declares an estimand nothing implements would be the same dormancy with extra
YAML -- so each of those is asserted here rather than assumed.
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


def test_the_full_graph_c_arms_are_demoted_to_sensitivity(decision):
    """The headline C effect must come from the fork, with the blocked path stated."""
    node = decision["structured_increment"]["by_node"]["C_VISIBLE"]
    assert node["primary_effect_source"] == "C_FROZEN_CONTINUATION_FORK"
    assert node["secondary_effect_source"] == "FULL_GRAPH_COUPLED_SEED_E2E"
    assert node["secondary_effect_use"].startswith("SENSITIVITY_ONLY")
    assert node["mediation_gap_reporting_required"] is True


def test_the_work_fraction_basis_forbids_the_fabricated_saving(decision):
    """Post-boundary alone reports a report produced by two model calls."""
    node = decision["structured_increment"]["by_node"]["C_VISIBLE"]
    assert node["work_fraction_basis"] == "TOTAL_WITH_SHARED_UPSTREAM"


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
