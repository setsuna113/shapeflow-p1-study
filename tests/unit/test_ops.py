"""GPU lease (flock) and the watchdog drift/auto-stop decisions."""

from __future__ import annotations

import pytest

from shapeflow_p1.ops.gpu_lease import GpuLease, LeaseHeld
from shapeflow_p1.ops.watchdog import DriftGuard, check_drift, evaluate_auto_stop


# --- GPU lease ------------------------------------------------------------------------


def test_lease_is_exclusive_on_same_host(tmp_path):
    uuid = "GPU-1234"
    a = GpuLease(uuid, lock_dir=tmp_path).acquire()
    try:
        # A second lease on the same UUID must fail while the first is held.
        with pytest.raises(LeaseHeld):
            GpuLease(uuid, lock_dir=tmp_path).acquire()
    finally:
        a.release()
    # After release, it can be re-acquired.
    b = GpuLease(uuid, lock_dir=tmp_path).acquire()
    b.release()


def test_lease_context_manager_releases(tmp_path):
    uuid = "GPU-abcd"
    with GpuLease(uuid, lock_dir=tmp_path):
        pass
    # released -> re-acquirable
    with GpuLease(uuid, lock_dir=tmp_path):
        pass


def test_lease_requires_uuid(tmp_path):
    with pytest.raises(ValueError):
        GpuLease("", lock_dir=tmp_path)


def test_distinct_uuids_do_not_conflict(tmp_path):
    with GpuLease("GPU-A", lock_dir=tmp_path):
        with GpuLease("GPU-B", lock_dir=tmp_path):
            pass  # different devices, both leasable


# --- drift ----------------------------------------------------------------------------


def test_no_drift_within_guard():
    assert check_drift(baseline_latency=1.0, current_latency=1.1, baseline_tps=100,
                       current_tps=95, guard=DriftGuard()) == []


def test_latency_drift_flagged():
    reasons = check_drift(baseline_latency=1.0, current_latency=1.5, baseline_tps=100,
                          current_tps=100, guard=DriftGuard())
    assert any("latency drift" in r for r in reasons)


def test_throughput_drift_flagged():
    reasons = check_drift(baseline_latency=1.0, current_latency=1.0, baseline_tps=100,
                          current_tps=70, guard=DriftGuard())
    assert any("throughput drift" in r for r in reasons)


# --- auto-stop ------------------------------------------------------------------------


def test_no_stop_when_healthy():
    assert evaluate_auto_stop().stop is False


def test_each_condition_stops():
    assert evaluate_auto_stop(gpu_hardware_error=True).stop
    assert evaluate_auto_stop(secret_leak_detected=True).stop
    assert evaluate_auto_stop(p0_parity_failed=True).stop
    assert evaluate_auto_stop(holdout_read_before_release=True).stop
    assert evaluate_auto_stop(consecutive_oom=3).stop
    # below the OOM threshold does not stop
    assert not evaluate_auto_stop(consecutive_oom=2).stop


def test_stop_reasons_listed():
    stop = evaluate_auto_stop(budget_exhausted=True, freeze_hash_changed=True)
    assert stop.stop
    assert len(stop.reasons) == 2


# --- reports may not claim what no artifact supports ----------------------------------------


def test_a_report_that_claims_a_finished_block_fails_the_gate(tmp_path):
    from shapeflow_p1.ops.acceptance import check_report_claims

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "WEEK1_P1_FINAL.md").write_text(
        "# Week 1\n\nBlock C/D 完成; ran on a leased GPU with 662 tests all green.\n",
        encoding="utf-8",
    )
    gate = check_report_claims(tmp_path)
    assert gate.status == "FAIL"
    assert "Block C/D 完成" in gate.detail
    assert "leased GPU" in gate.detail


def test_a_report_of_what_actually_happened_passes(tmp_path):
    from shapeflow_p1.ops.acceptance import check_report_claims

    (tmp_path / "reports").mkdir()
    (tmp_path / "reports" / "FAILED_AUTHORING_ATTEMPT.md").write_text(
        "Authoring stopped: the search credential was rejected 262 times.\n"
        "Spend kept on the books: $1.339 DeepSeek, 990 GPU-seconds.\n",
        encoding="utf-8",
    )
    assert check_report_claims(tmp_path).status == "PASS"


def test_the_gate_is_vacuous_only_when_there_are_no_reports(tmp_path):
    from shapeflow_p1.ops.acceptance import check_report_claims

    gate = check_report_claims(tmp_path)
    assert gate.status == "PASS" and "no reports yet" in gate.detail
