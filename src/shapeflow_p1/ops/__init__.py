"""Operational safety: GPU leasing, the drift/auto-stop watchdog, preflight, service status.

The pure decision logic here (lease acquisition via flock, drift thresholds, auto-stop conditions)
is validated in the dev environment; the parts that read real NVML/GPU state are thin wrappers
supplied on the run host. Keeping the *decisions* testable means the guard that pauses on a
foreign GPU process or a period drift is exercised long before it must fire on hardware.
"""
