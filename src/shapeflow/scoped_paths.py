"""Safe paths for evaluator artifacts scoped by one run and one phase.

``run_id`` and ``phase_id`` cross a CLI/API trust boundary.  Treating them as ordinary path
fragments lets an absolute path or ``../`` redirect a score read or analysis write outside the
evaluator judgment tree.  This module gives every such caller the same narrow grammar and then
verifies the fully resolved path is still below the intended root (including existing symlinks).
"""

from __future__ import annotations

import re
from pathlib import Path

__all__ = ["safe_scope_component", "resolve_scoped_path"]

_SAFE_COMPONENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")


def safe_scope_component(value: str, *, name: str) -> str:
    """Return one portable path component or reject it.

    The ASCII grammar is deliberate: IDs are protocol coordinates, not user-facing labels.
    It excludes both POSIX and Windows separators, ``.``/``..``, drive prefixes, whitespace,
    control characters, and empty or unbounded names.
    """
    if not isinstance(value, str) or not _SAFE_COMPONENT.fullmatch(value):
        raise ValueError(
            f"{name} must be one safe path component "
            "(1-128 ASCII letters, digits, '.', '_' or '-', starting alphanumeric)"
        )
    return value


def resolve_scoped_path(
    root: Path,
    *,
    run_id: str,
    phase_id: str,
    tail: tuple[str, ...] = (),
) -> Path:
    """Resolve a run/phase artifact path and prove containment under ``root``.

    ``tail`` is reserved for fixed program-owned names such as
    ``("analysis", "ITT_VERDICT_INPUTS.json")``.  Containment is checked after resolving the
    complete path, so an existing symlink in the run, phase, or analysis directory cannot
    redirect the operation.
    """
    run = safe_scope_component(run_id, name="run_id")
    phase = safe_scope_component(phase_id, name="phase_id")
    root_resolved = Path(root).resolve(strict=False)
    candidate = root_resolved.joinpath(run, phase, *tail).resolve(strict=False)
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ValueError(
            f"resolved run/phase path escapes the configured root {root_resolved}"
        ) from exc
    return candidate
