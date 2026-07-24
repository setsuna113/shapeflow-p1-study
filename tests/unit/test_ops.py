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
