"""Canonical JSON: one byte string per logical value, on every machine.

Every identity in this study (span_id, occurrence_id, protocol_sha, freeze hashes) is a
digest over the output of :func:`canonical_json`. If two hosts could serialize the same
logical value differently, a resumed run would compute different IDs for identical
content and the ledger's idempotency would silently break. So this module is
deliberately strict: it rejects anything whose JSON encoding is ambiguous rather than
guessing a normalization.

Rejected on purpose:

- ``NaN`` / ``Infinity`` — not JSON, and not comparable, so they cannot appear in an ID.
- non-``str`` mapping keys — ``json`` coerces ``{1: "a"}`` and ``{"1": "a"}`` to the same
  bytes, so two distinct objects would collide on one hash.
- ``tuple`` / ``set`` — a tuple silently becomes a list (so a producer switching types
  changes nothing, hiding a real schema change) and a set has no defined order.
- ``bytes``, ``datetime``, and arbitrary objects — each has several plausible encodings;
  the caller must choose one explicitly and pass a ``str``.
- unpaired surrogates — these cannot be UTF-8 encoded, and would raise deep inside a
  digest call instead of at the boundary.
"""

from __future__ import annotations

import json
import math
from typing import Any

__all__ = ["CanonicalizationError", "canonical_json", "canonical_str"]


class CanonicalizationError(TypeError):
    """A value has no unambiguous canonical JSON encoding."""


def _reject(path: str, why: str) -> None:
    where = path or "<root>"
    raise CanonicalizationError(f"{where}: {why}")


def _validate(value: Any, path: str = "") -> None:
    """Walk the structure and reject every ambiguous encoding before serializing.

    Done as a separate pass so the error names the exact path (``events[3].usage``)
    rather than surfacing as an opaque failure inside ``json.dumps``.
    """
    if value is None or isinstance(value, str):
        if isinstance(value, str):
            try:
                value.encode("utf-8")
            except UnicodeEncodeError:
                _reject(path, "string contains unpaired surrogates and is not encodable")
        return

    # bool must be checked before int: bool is a subclass of int.
    if isinstance(value, bool):
        return

    if isinstance(value, int):
        return

    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            _reject(path, f"non-finite float {value!r} cannot appear in a canonical value")
        return

    if isinstance(value, dict):
        for key, sub in value.items():
            if not isinstance(key, str):
                _reject(
                    path,
                    f"mapping key {key!r} is {type(key).__name__}, not str; "
                    "non-string keys collide once JSON coerces them",
                )
            _validate(sub, f"{path}.{key}" if path else key)
        return

    if isinstance(value, list):
        for i, sub in enumerate(value):
            _validate(sub, f"{path}[{i}]")
        return

    if isinstance(value, (tuple, set, frozenset)):
        _reject(
            path,
            f"{type(value).__name__} is not allowed; use a list (tuple would silently "
            "encode as one, hiding a type change; set has no defined order)",
        )

    if isinstance(value, (bytes, bytearray)):
        _reject(path, "bytes have no single canonical JSON form; encode explicitly (e.g. hex)")

    _reject(path, f"{type(value).__name__} is not canonically serializable")


def canonical_json(value: Any) -> bytes:
    """Return the canonical UTF-8 JSON encoding of ``value``.

    Sorted keys, no insignificant whitespace, non-ASCII kept as UTF-8 (escaping it would
    make the bytes depend on the encoder's escaping policy rather than the content).
    """
    _validate(value)
    text = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
        check_circular=True,
    )
    return text.encode("utf-8")


def canonical_str(value: Any) -> str:
    """:func:`canonical_json` as ``str``, for embedding in text artifacts."""
    return canonical_json(value).decode("utf-8")
