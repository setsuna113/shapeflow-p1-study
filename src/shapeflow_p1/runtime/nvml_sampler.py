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

__all__ = ["PowerSample", "integrate_energy_joules", "NvmlSampler"]


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
