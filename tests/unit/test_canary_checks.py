"""The canary's own assertions, tested against synthetic runs.

The canary is the gate that decides whether screening starts, so its checks have to fail when
they should. Each test below constructs the exact shape of a P1 arm that *looks* like it worked
-- no selector decode, a relabelled P0 report, a CPU control that quietly called the model --
and requires the corresponding check to go red.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shapeflow_p1.campaign.canary import CANARY_CHECKS, _p1_differs_from_p0, _verify
from shapeflow_p1.campaign.schedule import ArmSpec, build_blocks, cell_key
from shapeflow_p1.campaign.settings import Settings

REPO = Path(__file__).resolve().parents[2]

ARMS = [
    ArmSpec("P0", "P0", "P0"),
    ArmSpec("H_ID", "H02", "P0"),
    ArmSpec("CPU_LEXICAL", "H00-CPU", "P0"),
    ArmSpec("SHORT_PROSE", "H00-PROSE", "P0"),
]


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


def _manifest():
    return build_blocks(protocol_sha="a" * 64, split="FORMATIVE_SCREEN", task_ids=["T1"],
                        arms=ARMS, seeds=[1], layer="causal", claim_scope="FORMATIVE_ONLY")


def _outputs(manifest, *, p1_report="P1 report", p0_report="P0 report", counts=None):
    records = {}
    for cell in manifest.cells:
        arm = cell.arm.arm_id
        records[cell_key(cell)] = {
            "cell": cell.content(),
            "work_key": f"WK-{arm}",
            "final_report": p0_report if arm == "P0" else p1_report,
            "events": [{"kind": "PAGE_BATCH_REDUCED", "fell_back": False}],
            "checkpoints": [{"kind": "HCheckpoint", "digest": "d"},
                            {"kind": "CCheckpoint", "digest": "e"}],
            "counts": counts or {"page_batches_deferred": 1, "page_batches_reduced": 1,
                                 "page_fallbacks": 0, "close_reduced": 1, "close_failed": 0},
        }
    return records


def _states(manifest, value="COMMITTED"):
    return {cell_key(c): value for c in manifest.cells}


def _run(settings, monkeypatch, inference):
    import shapeflow_p1.campaign.canary as canary

    manifest = _manifest()
    monkeypatch.setattr(canary, "_inference_events", lambda _s: inference)
    monkeypatch.setattr(canary, "_open_external_calls", lambda _s: 0)
    monkeypatch.setattr(canary, "_patched_graph_check",
                        lambda _r: {"name": "patched_graph_invoked", "status": "PASS",
                                    "detail": "stubbed"})
    return manifest, canary


def _good_inference():
    return [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-H_ID",
         "completion_tokens": 120, "prompt_tokens": 900},
        {"op_class": "PAGE_P0_SUMMARY", "work_key": "WK-P0",
         "completion_tokens": 400, "prompt_tokens": 3000},
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-SHORT_PROSE",
         "completion_tokens": 200, "prompt_tokens": 900},
    ]


def test_a_healthy_run_passes_every_check(settings, monkeypatch):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=REPO)
    failed = [c for c in checks if c["status"] != "PASS"]
    assert failed == [], failed
    assert {c["name"] for c in checks} >= set(CANARY_CHECKS)


def test_a_selector_that_never_decoded_fails(settings, monkeypatch):
    """A P1 arm with zero selector decode did not select anything, however green it looks."""
    inference = [e for e in _good_inference() if not e["op_class"].startswith("PAGE_P1")]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=REPO)
    assert _status(checks, "selector_decode") == "FAIL"


def test_p1_bytes_identical_to_p0_fails(settings, monkeypatch):
    """The failure mode this exists for: copy P0's output and change the label."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest, p1_report="same", p0_report="same"), repo=REPO)
    assert _status(checks, "p1_bytes_differ_from_p0") == "FAIL"


def test_a_cpu_control_that_called_the_model_fails(settings, monkeypatch):
    """If it decoded, it is not a CPU control, and the comparison it anchors is meaningless."""
    inference = _good_inference() + [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-CPU_LEXICAL",
         "completion_tokens": 64, "prompt_tokens": 500},
    ]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=REPO)
    assert _status(checks, "cpu_control_zero_decode") == "FAIL"


def test_a_prose_control_over_its_cap_fails(settings, monkeypatch):
    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    inference = _good_inference() + [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-SHORT_PROSE",
         "completion_tokens": cap + 1, "prompt_tokens": 500},
    ]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=REPO)
    assert _status(checks, "short_prose_same_budget") == "FAIL"


def test_a_partial_batch_publication_fails(settings, monkeypatch):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    counts = {"page_batches_deferred": 2, "page_batches_reduced": 1, "page_fallbacks": 0,
              "close_reduced": 1, "close_failed": 0}
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest, counts=counts), repo=REPO)
    assert _status(checks, "atomic_publication") == "FAIL"


def test_an_incomplete_canary_fails(settings, monkeypatch):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    states = _states(manifest)
    states[next(iter(states))] = "FAILED_FINAL"
    checks = canary._verify(settings, manifest, states, _outputs(manifest), repo=REPO)
    assert _status(checks, "cells_committed") == "FAIL"


def test_an_open_reservation_fails(settings, monkeypatch):
    import shapeflow_p1.campaign.canary as canary

    manifest, canary = _run(settings, monkeypatch, _good_inference())
    monkeypatch.setattr(canary, "_open_external_calls", lambda _s: 3)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=REPO)
    assert _status(checks, "reservations_closed") == "FAIL"


def test_a_replay_miss_fails(settings, monkeypatch):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs))
    outputs[first]["events"].append({"kind": "PAGE_BATCH_REDUCED",
                                     "detail": "ReplayMiss: query not captured"})
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=REPO)
    assert _status(checks, "no_live_search_miss") == "FAIL"


def test_a_readable_evaluator_tree_fails(settings, monkeypatch, tmp_path):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    (tmp_path / "evaluator").mkdir(parents=True, exist_ok=True)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=REPO)
    assert _status(checks, "truth_invisible_to_treatment") == "FAIL"


def test_the_p0_comparison_is_per_task():
    p0 = {"k": {"cell": {"task_id": "T1"}, "final_report": "A"}}
    p1 = {"j": {"cell": {"task_id": "T1"}, "final_report": "B"},
          "l": {"cell": {"task_id": "T2"}, "final_report": "C"}}
    differing, compared = _p1_differs_from_p0(p0, p1)
    assert (differing, compared) == (1, 1), "a P1 cell with no P0 on its task was compared"


def _status(checks: list[dict], name: str) -> str:
    for check in checks:
        if check["name"] == name:
            return check["status"]
    raise AssertionError(f"check {name!r} was not produced; produced {[c['name'] for c in checks]}")


def test_verify_is_the_public_entry_point():
    assert callable(_verify)
