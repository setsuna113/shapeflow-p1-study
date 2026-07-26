"""Content-addressed blob store.

Every large artifact in the study -- frozen page bytes, checkpoint states, raw model
outputs, metric dumps -- lives here, keyed by the SHA-256 of its exact stored bytes.
SQLite holds only identities, sizes and references; the bytes never go in the database.

Three properties this store must guarantee, because the whole audit trail rests on them:

- **Deduplication.** Byte-identical content is stored once. Two URLs serving the same
  page, or the same checkpoint re-materialized, cost one blob.
- **Verified reads.** ``get`` re-hashes what it read and compares to the key. A silent
  bit-flip on disk becomes a loud :class:`CorruptObject`, never a wrong answer fed into
  an experiment. The plan makes a hash mismatch a hard stop, not something to repair in
  place.
- **Crash-atomic writes.** A blob appears at its final path only after its bytes are
  fully written and fsync'd. A process killed mid-write leaves a ``.tmp`` file that the
  reader never sees, not a half-blob at the real key. This is what lets resume trust
  that "the key exists" means "the bytes are whole".

The key is a *plain* SHA-256 of the stored bytes, deliberately not the domain-separated
``content_id`` from :mod:`shapeflow_p1.hashing`. The key's only job is integrity ("do
these bytes match what I asked for"); semantic identities live in the domain tables and
point here via an object reference. Keeping them separate means a snapshot's
``content_hash`` and its ``object_ref`` are computed for different purposes and neither
constrains the other.
"""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import zstandard as zstd

from .fsmode import chmod_shared
from .hashing import sha256_hex

__all__ = ["ObjectRef", "ObjectStore", "CorruptObject"]


class CorruptObject(RuntimeError):
    """Stored bytes no longer hash to their key. Fatal: never silently repaired."""


@dataclass(frozen=True)
class ObjectRef:
    """A stored blob's identity and footprint."""

    key: str  # sha256 hex of the raw (decompressed) bytes
    raw_size: int
    stored_size: int  # bytes actually on disk after compression


class ObjectStore:
    """A zstd-compressed content-addressed store rooted at one directory."""

    def __init__(self, root: str | os.PathLike[str], *, level: int = 10) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._level = level

    @property
    def root(self) -> Path:
        return self._root

    def _path_for(self, key: str) -> Path:
        # Shard by the first two bytes so no single directory holds the whole campaign.
        if len(key) != 64 or any(c not in "0123456789abcdef" for c in key):
            raise ValueError(f"not a sha256 hex key: {key!r}")
        return self._root / key[:2] / key[2:4] / f"{key}.zst"

    def has(self, key: str) -> bool:
        return self._path_for(key).exists()

    def put_bytes(self, raw: bytes) -> ObjectRef:
        """Store ``raw`` and return its reference. Idempotent for identical bytes."""
        if not isinstance(raw, (bytes, bytearray)):
            raise TypeError(f"put_bytes expects bytes, got {type(raw).__name__}")
        raw = bytes(raw)
        key = sha256_hex(raw)
        final = self._path_for(key)

        if final.exists():
            # Already stored. Trust the earlier verified write rather than rewriting,
            # so a concurrent writer can't truncate a blob a reader is mid-read on.
            return ObjectRef(key=key, raw_size=len(raw), stored_size=final.stat().st_size)

        final.parent.mkdir(parents=True, exist_ok=True)
        # zstandard compressor/decompressor contexts are not safe for simultaneous use.
        # Provider request handlers write telemetry from multiple threads, so sharing one
        # context can produce intermittent ``Src size is incorrect`` failures. A fresh context
        # per operation keeps the store thread-safe; unique temp files below already make
        # concurrent same-key publication crash-atomic.
        compressed = zstd.ZstdCompressor(level=self._level).compress(raw)

        # temp -> fsync -> atomic rename, all on the same directory (same filesystem)
        # so os.replace is a true atomic swap.
        fd, tmp_name = tempfile.mkstemp(dir=final.parent, suffix=".tmp")
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as fh:
                fh.write(compressed)
                fh.flush()
                os.fsync(fh.fileno())
            # Widen from mkstemp's 0600 *before* the rename, so the blob is never visible at
            # its final key with an ACL mask that would deny the readers it was written for:
            # the steward publishes page bytes the runner must read, and the runner publishes
            # treatment outputs the evaluator must score. See shapeflow_p1.fsmode.
            chmod_shared(tmp)
            os.replace(tmp, final)
        except BaseException:
            tmp.unlink(missing_ok=True)
            raise

        # Best-effort durability of the rename itself.
        self._fsync_dir(final.parent)
        return ObjectRef(key=key, raw_size=len(raw), stored_size=len(compressed))

    def get_bytes(self, key: str) -> bytes:
        """Return the stored bytes for ``key``, verifying integrity."""
        path = self._path_for(key)
        if not path.exists():
            raise KeyError(key)
        try:
            raw = zstd.ZstdDecompressor().decompress(path.read_bytes())
        except zstd.ZstdError as e:
            # A blob that will not even decompress is corrupt just as surely as one whose
            # bytes hash wrong; both are a hard stop, never a silent skip.
            raise CorruptObject(f"object {key} failed to decompress: {e}") from e
        actual = sha256_hex(raw)
        if actual != key:
            raise CorruptObject(
                f"object {key} decompressed to bytes hashing to {actual}; disk corruption"
            )
        return raw

    def verify(self, key: str) -> bool:
        """True iff the blob exists and still hashes to its key. Never raises for a
        mismatch -- resume uses this to decide whether a COMMITTED artifact is trustworthy."""
        try:
            self.get_bytes(key)
            return True
        except (KeyError, CorruptObject, zstd.ZstdError):
            return False

    @staticmethod
    def _fsync_dir(directory: Path) -> None:
        # Directory fsync is POSIX-only; on platforms without it the rename durability
        # is weaker but the atomicity (which is what matters for correctness) still holds.
        try:
            dfd = os.open(directory, os.O_RDONLY)
        except (OSError, AttributeError):
            return
        try:
            os.fsync(dfd)
        except OSError:
            pass
        finally:
            os.close(dfd)
