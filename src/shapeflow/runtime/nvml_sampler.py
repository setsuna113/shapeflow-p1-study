"""NVML power/energy sampling for the per-task energy guard.

Energy is reported alongside (never instead of) service time and tokens, because a work saving that
merely moves cost onto the GPU's power draw is not a real saving. The integration -- turning a
series of instantaneous power readings into joules -- is pure and tested here; the NVML reads that
produce the samples are a thin wrapper supplied on the run host, so the energy math is validated
without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = [
    "PowerSample",
    "integrate_energy_joules",
    "NvmlSampler",
    "EnergyCounter",
    "read_total_energy_joules",
]


@dataclass(frozen=True)
class PowerSample:
    ts: float          # seconds
    power_w: float     # instantaneous power draw, watts


def integrate_energy_joules(samples: Sequence[PowerSample]) -> float:
    """Trapezoidal integral of power over time -> joules. Needs >= 2 samples; returns 0 otherwise."""
    ordered = sorted(samples, key=lambda s: s.ts)
    energy = 0.0
    for a, b in zip(ordered, ordered[1:]):
        dt = b.ts - a.ts
        if dt < 0:
            raise ValueError("power samples must be time-ordered")
        energy += 0.5 * (a.power_w + b.power_w) * dt
    return energy


class NvmlSampler:
    """Thin wrapper around pynvml; instantiated only on the run host. Left importable everywhere so
    the module (and integrate_energy_joules) is available without a GPU."""

    def __init__(self, gpu_uuid: str) -> None:
        self._uuid = gpu_uuid
        self._handle = None

    def start(self) -> None:  # pragma: no cover - requires NVML/GPU
        import pynvml

        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByUUID(self._uuid.encode())

    def sample(self, ts: float) -> PowerSample:  # pragma: no cover - requires NVML/GPU
        import pynvml

        if self._handle is None:
            raise RuntimeError("call start() first (run host only)")
        milliwatts = pynvml.nvmlDeviceGetPowerUsage(self._handle)
        return PowerSample(ts=ts, power_w=milliwatts / 1000.0)


def read_total_energy_joules(gpu_uuid: str) -> float | None:
    """The driver's own monotonic energy counter for one GPU, in joules.

    Preferred over integrating power samples: the counter is accumulated by the driver at its
    own rate, so it cannot miss a burst between two of our reads, and it needs no sampling
    thread competing with the run. Returns None when NVML, the device, or the counter is
    unavailable -- energy is reported alongside work, never as a gate, so an absent reading must
    degrade to "not measured" rather than to a fabricated zero.
    """
    try:  # pragma: no cover - requires NVML/GPU
        import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid.encode())
        return float(pynvml.nvmlDeviceGetTotalEnergyConsumption(handle)) / 1000.0
    except Exception:  # noqa: BLE001 - any NVML failure means "not measured"
        return None


@dataclass
class EnergyCounter:
    """Energy drawn between two reads of the driver's counter.

    A counter difference, not an integral, so the value is the GPU's own accounting rather than
    ours. ``joules()`` returns None if either end is unavailable or if the counter went
    backwards, which is what a driver reset looks like; a negative or invented energy figure
    next to a work saving is worse than no figure, because it is the number that would decide
    whether a saving is real or merely moved onto the power bill.
    """

    gpu_uuid: str
    start_joules: float | None = None
    end_joules: float | None = None

    def start(self) -> None:
        self.start_joules = read_total_energy_joules(self.gpu_uuid)

    def stop(self) -> None:
        self.end_joules = read_total_energy_joules(self.gpu_uuid)

    def joules(self) -> float | None:
        if self.start_joules is None or self.end_joules is None:
            return None
        delta = self.end_joules - self.start_joules
        return delta if delta >= 0 else None
