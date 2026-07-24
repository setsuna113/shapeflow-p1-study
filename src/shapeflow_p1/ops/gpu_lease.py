"""A cross-process GPU lease via flock.

The campaign leases GPUs by UUID, not index (indices renumber; a UUID is the device). The lease is
an advisory ``flock`` on a per-UUID lock file, so a second coordinator on the same host cannot grab
a GPU this run already holds -- and a crash releases the lock automatically when the fd closes. This
is the mutual-exclusion half of GPU safety; the watchdog handles the "someone else's process
appeared on my GPU" half by waiting, never killing.
"""

from __future__ import annotations

import errno
import fcntl
import os
from pathlib import Path
from typing import Optional

__all__ = ["GpuLease", "LeaseHeld"]


class LeaseHeld(RuntimeError):
    """The GPU is already leased by another process on this host."""


class GpuLease:
    """A context manager holding an flock on a per-UUID lock file."""

    def __init__(self, gpu_uuid: str, *, lock_dir: str | os.PathLike[str]) -> None:
        if not gpu_uuid:
            raise ValueError("gpu_uuid required (lease by UUID, never index)")
        self._uuid = gpu_uuid
        self._dir = Path(lock_dir)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._path = self._dir / f"gpu-{gpu_uuid}.lock"
        self._fd: Optional[int] = None

    @property
    def path(self) -> Path:
        return self._path

    def acquire(self) -> "GpuLease":
        fd = os.open(self._path, os.O_CREAT | os.O_RDWR, 0o600)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as e:
            os.close(fd)
            if e.errno in (errno.EAGAIN, errno.EACCES):
                raise LeaseHeld(f"GPU {self._uuid} already leased") from e
            raise
        os.ftruncate(fd, 0)
        os.write(fd, f"pid={os.getpid()} uuid={self._uuid}\n".encode())
        os.fsync(fd)
        self._fd = fd
        return self

    def release(self) -> None:
        if self._fd is not None:
            fcntl.flock(self._fd, fcntl.LOCK_UN)
            os.close(self._fd)
            self._fd = None

    @property
    def held(self) -> bool:
        return self._fd is not None

    def __enter__(self) -> "GpuLease":
        return self.acquire()

    def __exit__(self, *exc) -> None:
        self.release()
