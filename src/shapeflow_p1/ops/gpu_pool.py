"""Choose which of the permitted GPUs to lease, from what is actually idle right now.

``configs/stack.yaml`` has listed a four-device ``gpu_uuid_pool`` since the plan was written,
and nothing read it: the device was hard-pinned to a single ``host.gpu_uuid``. So the campaign
could only ever run on one specific card, and would refuse to start whenever that card was
busy -- including when it was busy with our *own* previous engine.

The split is deliberate, and mirrors how ``attention_backend`` is already handled:

* the **pool** is hash-locked in ``stack.yaml`` and enters the approval binding, because which
  devices a campaign is permitted to use is a protocol statement;
* the **selection** is an operational observation resolved by ``freeze-stack`` into
  ``protocol/stack_manifest.json`` -- whose own sha is in the binding -- so the card that
  actually served a run is still bound, still recorded, and still re-verified by ``doctor``,
  without pretending it was decided in advance.

Selection is deterministic (first free device *in pool order*), never "emptiest". A
lowest-utilisation rule would make the choice depend on transient state, so the same frozen
config could resolve to a different device on a re-run and nothing downstream would notice.
"""

from __future__ import annotations

import re
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_MAX_USED_MEMORY_MIB",
    "GpuDevice",
    "GpuPoolError",
    "observe_pool",
    "select_devices",
]

#: A device is "free" only below this. Not zero: a display or ECC reservation legitimately
#: holds a few MiB on an otherwise idle card, and requiring exactly 0 would refuse to run on
#: hardware that is genuinely available.
DEFAULT_MAX_USED_MEMORY_MIB = 512


class GpuPoolError(RuntimeError):
    """The permitted pool cannot supply the devices this campaign needs."""


@dataclass(frozen=True)
class GpuDevice:
    uuid: str
    name: str
    index: int
    memory_total_mib: int
    memory_used_mib: int
    compute_process_count: int

    def is_free(self, *, max_used_memory_mib: int = DEFAULT_MAX_USED_MEMORY_MIB) -> bool:
        """Both conditions, because either alone misses a real case.

        A leaked allocation whose process already exited shows zero compute processes while
        holding gigabytes; a just-started process holds almost no memory yet. Starting on
        either would put two engines on one card.
        """
        return self.compute_process_count == 0 and self.memory_used_mib <= max_used_memory_mib

    def why_busy(self) -> str:
        reasons = []
        if self.compute_process_count:
            reasons.append(f"{self.compute_process_count} compute process(es)")
        if self.memory_used_mib:
            reasons.append(f"{self.memory_used_mib} MiB in use")
        return ", ".join(reasons) or "idle"


#: ``GPU-`` followed by a canonical UUID. The only value interpolated into an nvidia-smi
#: argument list is a pool UUID, so validating its shape means a malformed config cannot turn
#: into an extra command-line argument.
_UUID_RE = re.compile(r"^GPU-[0-9a-fA-F]{8}(-[0-9a-fA-F]{4}){3}-[0-9a-fA-F]{12}$")


def _require_uuid(uuid: str) -> str:
    if not _UUID_RE.match(uuid):
        raise GpuPoolError(f"not a GPU UUID: {uuid!r} (devices are leased by UUID, never index)")
    return uuid


def _run(args: Sequence[str]) -> str | None:
    """Return stdout, or None when the tool is missing or fails.

    None rather than a guess: an unreadable GPU state must not resolve to "looks free".
    """
    if not shutil.which(args[0]):
        return None
    try:
        # No shell, fixed argv[0], and every interpolated value is a shape-checked UUID.
        done = subprocess.run(  # noqa: S603
            args, capture_output=True, text=True, timeout=30, check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return done.stdout if done.returncode == 0 else None


def observe_pool(pool_uuids: Sequence[str]) -> list[GpuDevice]:
    """Observe every pool device present on this host, in pool order.

    Devices in the pool but absent from the host are omitted rather than invented; the caller
    decides whether the survivors are enough.
    """
    listing = _run([
        "nvidia-smi",
        "--query-gpu=uuid,name,index,memory.total,memory.used",
        "--format=csv,noheader,nounits",
    ])
    if listing is None:
        raise GpuPoolError("nvidia-smi is unavailable; the GPU pool cannot be observed")

    present: dict[str, tuple[str, int, int, int]] = {}
    for line in listing.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 5:
            continue
        uuid, name, index, total, used = fields
        try:
            present[uuid] = (name, int(index), int(total), int(used))
        except ValueError:
            continue

    # Per-device rather than one host-wide query: the host-wide form cannot say *which* card a
    # process is on, and treating any busy GPU as disqualifying is what made a neighbour's job
    # -- or our own engine on another card -- block the whole campaign.
    devices: list[GpuDevice] = []
    for uuid in pool_uuids:
        if uuid not in present:
            continue
        name, index, total, used = present[uuid]
        apps = _run([
            "nvidia-smi", "-i", _require_uuid(uuid),
            "--query-compute-apps=pid", "--format=csv,noheader",
        ])
        if apps is None:
            raise GpuPoolError(f"cannot read the compute processes on {uuid}")
        devices.append(GpuDevice(
            uuid=uuid, name=name, index=index,
            memory_total_mib=total, memory_used_mib=used,
            compute_process_count=len([ln for ln in apps.splitlines() if ln.strip()]),
        ))
    return devices


def select_devices(
    pool_uuids: Sequence[str],
    *,
    count: int = 1,
    max_used_memory_mib: int = DEFAULT_MAX_USED_MEMORY_MIB,
    devices: Sequence[GpuDevice] | None = None,
) -> list[str]:
    """The first ``count`` free devices in pool order, or a hard error naming what is busy."""
    if count < 1:
        raise GpuPoolError(f"count must be at least 1, got {count}")
    if not pool_uuids:
        raise GpuPoolError("the permitted GPU pool is empty")
    observed = list(devices) if devices is not None else observe_pool(pool_uuids)

    missing = [uuid for uuid in pool_uuids if uuid not in {d.uuid for d in observed}]
    free = [d for d in observed if d.is_free(max_used_memory_mib=max_used_memory_mib)]
    if len(free) < count:
        detail = "; ".join(f"{d.uuid} ({d.why_busy()})" for d in observed) or "no pool device"
        absent = f"; absent from this host: {missing}" if missing else ""
        raise GpuPoolError(
            f"need {count} free GPU(s) from the permitted pool but only {len(free)} are idle: "
            f"{detail}{absent}. Refusing to start over another process rather than killing it."
        )
    return [d.uuid for d in free[:count]]
