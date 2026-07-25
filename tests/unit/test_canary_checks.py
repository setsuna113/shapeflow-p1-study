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


@pytest.fixture()
def proved_repo(tmp_path, settings):
    """A repo whose installer already proved the runner is shut out of both trees.

    Cross-uid isolation cannot be demonstrated from inside one process, so the canary reads
    the proof install_host.sh generates. Staging it here is what makes the healthy case
    healthy; without it the check fails, which is the point.
    """
    import json

    for name in ("evaluator_root", "steward_root"):
        settings.path(name).mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "reports" / "CREDENTIAL_ISOLATION.json").write_text(json.dumps({
        "denied": {"sfrunner": True, "sfinfer": True, "sfsteward": True,
                   "sfevaluator": True},
        "provider_can_read": True,
        "runner_cannot_list": ["evaluator", "steward"],
    }), encoding="utf-8")
    return repo


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


def _provider_ledger(settings, *, max_tokens=768):
    """A real provider ledger with one committed inference attempt.

    The ledger-derived checks read it rather than a stub, because their whole failure mode
    was reading an absent ledger as "nothing was dispatched" and passing.
    """
    import json

    from shapeflow_p1.experiment.ledger import Ledger
    from shapeflow_p1.object_store import ObjectStore

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(path))
    store = ObjectStore(
        settings.data_root / str(settings.get("week1", "paths", "provider_root")) / "objects")
    ref = store.put_bytes(json.dumps({"model": "m", "max_tokens": max_tokens}).encode())
    with ledger.transaction() as cur:
        cur.execute(
            "INSERT INTO external_calls(call_id, provider, op_class, call_key, work_key,"
            " state, attempt_count, created_at, updated_at)"
            " VALUES ('c1','vllm','PAGE_P1_SELECTOR_LOCAL','k','WK-H_ID','COMMITTED',1,1,1)")
        cur.execute(
            "INSERT INTO external_call_attempts(attempt_id, call_id, attempt_ordinal, state,"
            " request_object_ref, usage_json, opened_at)"
            " VALUES ('a1','c1',0,'COMMITTED',?,'{\"completion_tokens\": 120}',1)",
            (ref.key,))
    ledger.close()


def _run(settings, monkeypatch, inference):
    import shapeflow_p1.campaign.canary as canary

    manifest = _manifest()
    _provider_ledger(settings)
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


def test_a_healthy_run_passes_every_check(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    failed = [c for c in checks if c["status"] != "PASS"]
    assert failed == [], failed
    assert {c["name"] for c in checks} >= set(CANARY_CHECKS)


def test_a_selector_that_never_decoded_fails(settings, monkeypatch, proved_repo):
    """A P1 arm with zero selector decode did not select anything, however green it looks."""
    inference = [e for e in _good_inference() if not e["op_class"].startswith("PAGE_P1")]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "selector_decode") == "FAIL"


def test_p1_bytes_identical_to_p0_fails(settings, monkeypatch, proved_repo):
    """The failure mode this exists for: copy P0's output and change the label."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest, p1_report="same", p0_report="same"), repo=proved_repo)
    assert _status(checks, "p1_bytes_differ_from_p0") == "FAIL"


def test_a_cpu_control_that_called_the_model_fails(settings, monkeypatch, proved_repo):
    """If it decoded, it is not a CPU control, and the comparison it anchors is meaningless."""
    inference = _good_inference() + [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-CPU_LEXICAL",
         "completion_tokens": 64, "prompt_tokens": 500},
    ]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "cpu_control_zero_decode") == "FAIL"


def test_a_prose_control_over_its_cap_fails(settings, monkeypatch, proved_repo):
    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    inference = _good_inference() + [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-SHORT_PROSE",
         "completion_tokens": cap + 1, "prompt_tokens": 500},
    ]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "short_prose_same_budget") == "FAIL"


def test_a_partial_batch_publication_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    counts = {"page_batches_deferred": 2, "page_batches_reduced": 1, "page_fallbacks": 0,
              "close_reduced": 1, "close_failed": 0}
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest, counts=counts), repo=proved_repo)
    assert _status(checks, "atomic_publication") == "FAIL"


def test_an_incomplete_canary_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    states = _states(manifest)
    states[next(iter(states))] = "FAILED_FINAL"
    checks = canary._verify(settings, manifest, states, _outputs(manifest), repo=proved_repo)
    assert _status(checks, "cells_committed") == "FAIL"


def test_an_open_reservation_fails(settings, monkeypatch, proved_repo):
    import shapeflow_p1.campaign.canary as canary

    manifest, canary = _run(settings, monkeypatch, _good_inference())
    monkeypatch.setattr(canary, "_open_external_calls", lambda _s: 3)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "reservations_closed") == "FAIL"


def test_a_replay_miss_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs))
    outputs[first]["events"].append({"kind": "PAGE_BATCH_REDUCED",
                                     "detail": "ReplayMiss: query not captured"})
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "no_live_search_miss") == "FAIL"


def test_a_readable_evaluator_tree_fails(settings, monkeypatch, proved_repo, tmp_path):
    """The in-process probe is evidence in one direction only, so it is applied in that
    direction: if the *runner* can list the tree, no proof outweighs it."""
    monkeypatch.setenv("USER", "runner")
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    (tmp_path / "evaluator").mkdir(parents=True, exist_ok=True)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "truth_invisible_to_treatment") == "FAIL"


def test_isolation_is_not_proved_by_an_absent_tree(settings, monkeypatch, proved_repo):
    """It passed for exactly as long as there was nothing to protect."""
    import shutil

    manifest, canary = _run(settings, monkeypatch, _good_inference())
    shutil.rmtree(settings.path("evaluator_root"))
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "truth_invisible_to_treatment") == "FAIL"


def test_isolation_without_a_cross_uid_proof_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    (proved_repo / "reports" / "CREDENTIAL_ISOLATION.json").unlink()
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
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


# --- the checks that used to pass no matter what ---------------------------------------------


def test_the_fallback_check_can_actually_fail(settings, monkeypatch, proved_repo):
    """It was `_check(..., True, ...)`: it formatted counts into a sentence and passed."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs.values()))
    first["counts"]["page_fallbacks"] = 3          # a fallback the ledger knows nothing about
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "fallback_and_failures_accounted") == "FAIL"


def test_the_generation_cap_check_reads_requests_not_config(settings, monkeypatch,
                                                            proved_repo):
    """It multiplied two config values into a constant True and never saw a request."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    # Re-write the recorded request without a cap.
    import json

    from shapeflow_p1.object_store import ObjectStore

    store = ObjectStore(settings.data_root
                        / str(settings.get("week1", "paths", "provider_root")) / "objects")
    ref = store.put_bytes(json.dumps({"model": "m"}).encode())
    from shapeflow_p1.experiment.ledger import Ledger

    ledger = Ledger(str(settings.data_root
                        / str(settings.get("week1", "paths", "provider_ledger"))))
    with ledger.transaction() as cur:
        cur.execute("UPDATE external_call_attempts SET request_object_ref=?", (ref.key,))
    ledger.close()

    checks = canary._verify(settings, manifest, _states(manifest), _outputs(manifest),
                            repo=proved_repo)
    assert _status(checks, "generation_cap_is_a_completion_limit") == "FAIL"


def test_one_checkpoint_kind_is_not_both(settings, monkeypatch, proved_repo):
    """It was a set intersection, so an H-only run passed a check named 'both boundaries'."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    for record in outputs.values():
        record["checkpoints"] = [c for c in record["checkpoints"]
                                 if c["kind"] != "CCheckpoint"]
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "checkpoints_present") == "FAIL"


def test_an_arm_that_is_byte_identical_to_p0_fails(settings, monkeypatch, proved_repo):
    """One differing cell out of six used to be enough for the whole check."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    p0 = next(v for v in outputs.values() if v["cell"]["arm"]["arm_id"] == "P0")
    for record in outputs.values():
        if record["cell"]["arm"]["arm_id"] != "P0":
            record["final_report"] = p0["final_report"]
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "p1_bytes_differ_from_p0") == "FAIL"


def test_an_arm_that_reduced_nothing_fails_even_beside_one_that_did(settings, monkeypatch,
                                                                    proved_repo):
    """reduce_published_batch fires for any bound bundle, so one working control satisfied
    the old total while a completely inert LLM arm passed behind it."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    for record in outputs.values():
        if record["cell"]["arm"]["arm_id"] not in ("P0", "CPU_LEXICAL"):
            record["counts"]["page_batches_reduced"] = 0
            record["counts"]["close_reduced"] = 0
            record["counts"]["page_batches_deferred"] = 0
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "p1_strategy_invocations") == "FAIL"


def test_an_unreadable_ledger_is_a_failure_not_a_zero(settings, monkeypatch, proved_repo):
    import shapeflow_p1.campaign.canary as canary_module

    manifest, canary = _run(settings, monkeypatch, _good_inference())
    monkeypatch.setattr(canary_module, "_inference_events",
                        lambda _s: (_ for _ in ()).throw(
                            canary_module.LedgerUnreadable("gone")))
    checks = canary._verify(settings, manifest, _states(manifest), _outputs(manifest),
                            repo=proved_repo)
    assert _status(checks, "ledger_readable") == "FAIL"
