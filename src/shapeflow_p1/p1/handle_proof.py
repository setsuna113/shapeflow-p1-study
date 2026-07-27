"""Prove, by enumeration, that no reachable publication handle can exceed the token cap.

The cap on a published handle is a protocol bound, and the previous encoding violated it for
every handle it ever produced -- in all 146 canary cells, across all 19 arms, H fell back to P0
before its first selector call and nothing failed loudly. The lesson is not "pick a smaller
encoding"; it is that a bound on a *rendered* cost cannot be argued from the shape of the
encoding, because the tokenizer's merge table is the only thing that decides it.

So this enumerates the entire reachable domain against the exact frozen ``tokenizer.json`` and
reports the worst case. The domain is finite and small by construction -- the radices are frozen
protocol -- so "every handle this codec can emit" is a few tens of millions of strings, about a
minute and a half of one host's CPU. A sampled check would not do: the previous cap held for
most handles and failed for the ones production actually produced.

The result is cached as a receipt keyed by (format version, radices, tokenizer digest). Any of
the three moving invalidates it, which is the point: a re-tokenized model or a widened radix is
exactly when the proof has to be redone.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..canonical import canonical_json
from ..evidence.chunkers import Tokenizer
from ..evidence.model_tokenizer import tokenizer_sha256
from ..hashing import sha256_hex
from . import handle_codec
from .handle_codec import FROZEN_RADICES, HandleRadices

__all__ = ["prove_handle_domain", "load_or_prove", "HandleProofError"]

_BATCH = 200_000


class HandleProofError(RuntimeError):
    """The frozen handle domain does not fit the frozen token cap."""


def _identity(radices: HandleRadices, tokenizer: Tokenizer) -> dict:
    return {
        "format_version": handle_codec.FORMAT_VERSION,
        "radices": radices.content(),
        "capacity": radices.capacity,
        "max_publication_handle_tokens": handle_codec.MAX_PUBLICATION_HANDLE_TOKENS,
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
    }


def prove_handle_domain(
    tokenizer: Tokenizer,
    *,
    radices: HandleRadices = FROZEN_RADICES,
) -> dict:
    """Enumerate every handle the domain can produce and cost each one exactly.

    Returns the identity of what was proven plus the full width histogram, so a later reader can
    see the distribution and not merely the verdict. Raises when the cap is exceeded: a proof
    that quietly returns a failing maximum would be a report, not a gate.
    """
    identity = _identity(radices, tokenizer)
    histogram: dict[int, int] = {}
    worst: list[tuple[int, str]] = []
    # encode_batch when the backend offers it: the pure-Python path costs minutes at this size,
    # and the whole point is that the proof is cheap enough to actually run before every launch.
    backend = getattr(tokenizer, "_backend", None)
    batch_encode = getattr(backend, "encode_batch", None)

    def costs(batch: list[str]) -> list[int]:
        if batch_encode is not None:
            return [
                len(encoded.ids)
                for encoded in batch_encode(batch, add_special_tokens=False)
            ]
        return [tokenizer.count(text) for text in batch]

    batch: list[str] = []

    def flush() -> None:
        for text, count in zip(batch, costs(batch)):
            histogram[count] = histogram.get(count, 0) + 1
            if count > handle_codec.MAX_PUBLICATION_HANDLE_TOKENS and len(worst) < 8:
                worst.append((count, text))
        batch.clear()

    for handle in handle_codec.iter_domain(radices):
        batch.append(f"[{handle}]")
        if len(batch) >= _BATCH:
            flush()
    if batch:
        flush()

    max_tokens = max(histogram) if histogram else 0
    result = {
        **identity,
        "max_encoded_width": handle_codec.max_encoded_width(radices),
        "max_tokens": max_tokens,
        "token_histogram": {str(k): histogram[k] for k in sorted(histogram)},
    }
    result["content_sha256"] = sha256_hex(canonical_json(result))
    if max_tokens > handle_codec.MAX_PUBLICATION_HANDLE_TOKENS:
        raise HandleProofError(
            f"{len(worst)}+ handles exceed the frozen cap "
            f"{handle_codec.MAX_PUBLICATION_HANDLE_TOKENS}; worst {worst[:3]}. Publication "
            "would fall back to P0 for every H arm, exactly as it silently did before."
        )
    return result


def load_or_prove(
    tokenizer: Tokenizer,
    *,
    receipt_path: Path,
    radices: HandleRadices = FROZEN_RADICES,
    allow_write: bool = True,
) -> dict:
    """Return a matching cached proof, or run one and cache it.

    A cached proof is honoured only when the format version, every radix, the cap and the
    tokenizer digest all match. Anything else and it is re-proven -- a stale receipt asserting a
    bound for a domain that has since widened is worse than no receipt.
    """
    identity = _identity(radices, tokenizer)
    existing: Optional[dict] = None
    try:
        existing = json.loads(Path(receipt_path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        existing = None
    if isinstance(existing, dict) and all(
        existing.get(key) == value for key, value in identity.items()
    ):
        recorded = str(existing.get("content_sha256") or "")
        recomputed = sha256_hex(canonical_json(
            {k: v for k, v in existing.items() if k != "content_sha256"}
        ))
        if recorded and recorded == recomputed:
            return existing

    result = prove_handle_domain(tokenizer, radices=radices)
    if allow_write:
        try:
            path = Path(receipt_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
        except OSError:
            # A read-only tree is not a reason to refuse to launch; the proof itself already
            # ran and passed. Only its cache is unavailable.
            pass
    return result
