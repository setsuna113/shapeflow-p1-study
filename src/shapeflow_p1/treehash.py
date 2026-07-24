"""Content hashing for an installed or materialized source tree.

Used to pin three things the launch gate cannot take on trust: the patched vendor tree, the
*installed* `open_deep_research` package, and the reused vLLM installation.

The hash is over ``(relative path, file bytes)`` pairs, sorted. Two details matter:

- **Paths are relative.** Hashing absolute paths makes two identical trees in different
  directories hash differently, which is exactly the comparison this exists to make.
- **Byte-compiled and metadata files are skipped.** ``__pycache__``, ``*.pyc``, ``*.dist-info``
  and ``*.egg-info`` differ between an sdist build and an install of the same source without
  the source differing, so including them would report drift that is not there.

`open_deep_research` installs as a *copy* into site-packages (and is a namespace package, so
``__file__`` is None). Comparing the installed tree's hash to the patched tree's hash is
therefore both possible and stronger than the path check the plan originally described: it
proves the running bytes are the patched bytes rather than that some path resolves somewhere.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Iterable, Iterator

__all__ = ["tree_sha256", "tree_files"]

_SKIP_DIRS = {"__pycache__", ".git", ".pytest_cache", ".ruff_cache", ".mypy_cache"}
_SKIP_SUFFIXES = (".pyc", ".pyo")
_SKIP_DIR_SUFFIXES = (".dist-info", ".egg-info")


def tree_files(root: Path) -> Iterator[Path]:
    """Yield every hashable file under ``root``, in no particular order."""
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        parts = set(path.relative_to(root).parts)
        if parts & _SKIP_DIRS:
            continue
        if any(p.endswith(_SKIP_DIR_SUFFIXES) for p in path.relative_to(root).parts):
            continue
        if path.suffix in _SKIP_SUFFIXES:
            continue
        yield path


def tree_sha256(root: Path, *, extra: Iterable[tuple[str, bytes]] = ()) -> str:
    """Digest a source tree by relative path and content.

    ``extra`` lets a caller fold in facts that are not files -- a version string, a commit --
    so the digest can commit to them without inventing a file to hold them.
    """
    root = Path(root)
    digest = hashlib.sha256()
    entries = sorted(
        (str(p.relative_to(root)), p) for p in tree_files(root)
    )
    for rel, path in entries:
        digest.update(rel.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(path.read_bytes()).digest())
    for key, value in sorted(extra):
        digest.update(key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(hashlib.sha256(value).digest())
    return digest.hexdigest()
