"""What any campaign process needs before it can do work: a ledger, a provider, and a device.

These three are shared by every entry point that touches the GPU -- the campaign driver, the
canary, and anything else that has to prove it is the only occupant of a card. They were private
helpers of the Week-1 screen; they are here because they belong to the *session*, not to any one
campaign.

The GPU lease is the load-bearing one. Refusing to run unleased is deliberate: an unleasable run
that proceeds anyway is how two workers end up on one device, and nothing downstream would show
it -- the numbers would simply be wrong in a way that looks like variance.
"""

from __future__ import annotations

import os

from ..experiment.ledger import Ledger
from ..object_store import ObjectStore
from ..providers.provider_client import ProviderClient, load_role_token
from .settings import Settings

__all__ = ["open_run_ledger", "provider_client_for", "gpu_lease"]


def open_run_ledger(settings: Settings) -> tuple[Ledger, ObjectStore]:
    runs = settings.path("runs")
    runs.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(runs / "ledger.sqlite"))
    store = ObjectStore(settings.path("object_store"))
    return ledger, store


def provider_client_for(settings: Settings, role: str = "runner") -> ProviderClient:
    host = settings.get("week1", "provider", "bind_host")
    # settings.provider_port, not the configured base: a lane that dialled the base port would
    # send every request to lane 0's provider while believing it was talking to its own.
    port = settings.provider_port
    token_dir = str(settings.get("week1", "provider", "token_dir"))
    return ProviderClient(base_url=f"http://{host}:{port}",
                          token=load_role_token(token_dir, role))


def gpu_lease(settings: Settings):
    """The lease for the device this run will use.

    The UUID comes from the environment the engine was started with (CUDA_VISIBLE_DEVICES is
    set to a UUID by start_engine.sh, never an index -- an index is not a device identity).

    Returning None when nothing was pinned -- which is what this did -- meant the lease was
    silently skipped exactly when it mattered: neither ``sfsupervise`` nor ``bootstrap``'s
    privilege-drop helper passed these variables through, so in production the mutual
    exclusion never engaged at all. On a host with several cards and more than one worker,
    that is how two runs end up on one device, and nothing downstream would show it. An
    unleasable run is refused instead.
    """
    from ..ops.gpu_lease import GpuLease

    uuid = (os.environ.get("SHAPEFLOW_GPU_UUID")
            or os.environ.get("CUDA_VISIBLE_DEVICES") or "").strip()
    if not uuid.startswith("GPU-"):
        raise RuntimeError(
            "no GPU UUID in the environment: set SHAPEFLOW_GPU_UUID (or CUDA_VISIBLE_DEVICES) "
            "to the leased device UUID. Refusing to run unleased, because the GPU lease is "
            f"what stops a second worker using the same card (saw {uuid!r})"
        )
    lock_file = settings.data_root / str(settings.get("week1", "runtime", "gpu_lease_file"))
    return GpuLease(uuid, lock_dir=lock_file.parent)
