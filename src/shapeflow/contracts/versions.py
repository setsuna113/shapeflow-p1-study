"""Digests of the contract documents, and the assertion that binds code to them.

A host implementation declares the contract digest it was written against and calls
:func:`assert_contract` at import. If the document moves, every host fails loudly at import time
rather than continuing to run against a contract nobody re-read.

The failure is deliberately not repairable by copying the new digest into the module. Changing a
contract changes ``contracts_sha`` in the execution binding, so the approval has to be re-minted
either way; pasting the digest without re-reading the text would produce a green import and a
false claim.
"""

from __future__ import annotations

from pathlib import Path

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = ["DOCS", "contract_digest", "contracts_digest", "assert_contract", "ContractDrift"]

DOCS = ("HC_MECHANISM_v1.md", "OFFER_INTERFACE_v1.md", "OBSERVABILITY_v1.md")

_DOC_DIR = Path(__file__).resolve().parent / "docs"


class ContractDrift(RuntimeError):
    """A module was written against a contract text that is no longer the one on disk."""


def contract_digest(name: str) -> str:
    """Digest of one contract document, by exact bytes."""
    if name not in DOCS:
        raise ValueError(f"{name!r} is not a contract document; known: {DOCS}")
    return sha256_hex((_DOC_DIR / name).read_bytes())


def contracts_digest() -> str:
    """Digest over every contract document, by name and bytes.

    Matches what :func:`shapeflow.protocol.compute_binding` records as ``contracts_sha``, so the
    binding and this module cannot disagree about which texts are in force.
    """
    return sha256_hex(canonical_json({name: contract_digest(name) for name in DOCS}))


def assert_contract(name: str, declared: str) -> None:
    """Assert ``declared`` is the current digest of contract ``name``.

    Called at import by every module claiming conformance. ``declared`` is a literal in the
    calling module, so the claim is visible in the diff when the contract moves.
    """
    actual = contract_digest(name)
    if declared != actual:
        raise ContractDrift(
            f"{name} has changed: this module was written against {declared[:12]}, the document "
            f"now hashes to {actual[:12]}. Re-read the document and re-derive the implementation; "
            "updating the literal alone would assert a conformance nobody checked."
        )
