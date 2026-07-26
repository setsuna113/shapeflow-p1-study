#!/usr/bin/env python3
"""Mint the identity of one concrete vLLM boot.

The frozen stack hash identifies bytes and flags; it deliberately stays constant when the same
service restarts.  Paired cells must additionally know whether they crossed such a restart.
systemd supplies a fresh ``INVOCATION_ID`` for every service invocation.  Manual/supervised
launches get a cryptographically random id instead.  The file is atomically replaced because
replacing it is exactly what a new engine boot means.
"""

from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: write_engine_epoch.py PATH", file=sys.stderr)
        return 2
    path = Path(sys.argv[1])
    invocation = os.environ.get("INVOCATION_ID", "").strip().lower()
    if len(invocation) != 32 or any(c not in "0123456789abcdef" for c in invocation):
        invocation = secrets.token_hex(16)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    descriptor = os.open(
        temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
    try:
        with os.fdopen(descriptor, "w", encoding="ascii") as handle:
            handle.write(invocation + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
