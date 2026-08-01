"""Choosing a GPU from the permitted pool, and refusing rather than guessing."""

from __future__ import annotations

import pytest

from shapeflow.ops.gpu_pool import GpuDevice, GpuPoolError, select_devices

# Real UUID shape: _require_uuid rejects anything else before it reaches an argv.
POOL = [
    "GPU-ef013951-e496-78da-da70-a5a289dcc634",
    "GPU-d1f5d8c4-2f0f-45f7-0574-ac90018440db",
    "GPU-11b9da9d-6149-3c1b-79c8-011d74a180d2",
    "GPU-99847987-522a-5889-1066-a21512dff220",
]
AAA, BBB, CCC, DDD = POOL


def _device(uuid: str, *, used: int = 0, procs: int = 0, index: int = 0) -> GpuDevice:
    return GpuDevice(
        uuid=uuid, name="NVIDIA GeForce RTX 4090", index=index,
        memory_total_mib=24564, memory_used_mib=used, compute_process_count=procs,
    )


def test_the_first_free_device_in_pool_order_is_chosen():
    """Pool order, not lowest utilisation: the choice has to be reproducible."""
    devices = [
        _device(AAA, used=22129, procs=1),   # our own stale engine
        _device(BBB, used=12),
        _device(CCC, used=4),                # emptier, deliberately not preferred
        _device(DDD, used=12),
    ]
    assert select_devices(POOL, devices=devices) == [BBB]
    assert select_devices(POOL, count=2, devices=devices) == [BBB, CCC]


def test_a_card_with_a_compute_process_is_never_selected():
    devices = [_device(AAA, used=0, procs=1), _device(BBB, used=12)]
    assert select_devices(POOL[:2], devices=devices) == [BBB]


def test_a_leaked_allocation_with_no_live_process_still_blocks():
    """Zero processes but gigabytes held is a dead job's memory, not an idle card."""
    devices = [_device(AAA, used=8000, procs=0), _device(BBB, used=12)]
    assert select_devices(POOL[:2], devices=devices) == [BBB]


def test_a_few_mib_of_display_overhead_does_not_disqualify():
    devices = [_device(AAA, used=300)]
    assert select_devices([AAA], devices=devices) == [AAA]


def test_too_few_free_devices_is_a_hard_error_naming_each_one():
    devices = [
        _device(AAA, used=22129, procs=1),
        _device(BBB, used=9000, procs=2),
    ]
    with pytest.raises(GpuPoolError) as excinfo:
        select_devices(POOL[:2], count=1, devices=devices)
    message = str(excinfo.value)
    assert AAA in message and BBB in message
    assert "1 compute process(es)" in message
    assert "Refusing to start over another process" in message


def test_a_pool_device_absent_from_the_host_is_reported_not_assumed():
    devices = [_device(AAA, used=22129, procs=1)]
    with pytest.raises(GpuPoolError, match="absent from this host"):
        select_devices([AAA, "GPU-00000000-0000-0000-0000-000000000000"], count=2,
                       devices=devices)


def test_an_empty_pool_is_refused():
    with pytest.raises(GpuPoolError, match="pool is empty"):
        select_devices([], devices=[])
