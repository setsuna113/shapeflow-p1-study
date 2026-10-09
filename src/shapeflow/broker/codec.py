"""Offers on the wire: the exact inverse of ``Offer.content()``.

An offer crosses a process boundary when the broker serves several agents, and the broker must
decide on the offer the agent priced, not on a near copy of it. Decoding therefore reconstructs
the frozen types field by field and checks that re-encoding gives back the same document, so a
field this module forgot to carry fails loudly here instead of vanishing from the decision.
"""

from __future__ import annotations

from collections.abc import Mapping

from ..canonical import canonical_json
from ..contracts.offer import (
    Boundary,
    ContractViolation,
    Offer,
    P0Plan,
    P1Plan,
    RenderDescriptor,
    RequestDescriptor,
)
from ..contracts.versions import assert_contract
from ..hashing import sha256_hex
from ..runtime.request_tags import OpClass

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["offer_from_content", "offer_digest"]


def _request(doc: Mapping) -> RequestDescriptor:
    return RequestDescriptor(
        op_class=OpClass(doc["op_class"]),
        prompt_sha256=str(doc["prompt_sha256"]),
        max_tokens=int(doc["max_tokens"]),
        est_uncached_prefill=int(doc["est_uncached_prefill"]),
        est_cached_prefill=int(doc["est_cached_prefill"]),
        est_decode=int(doc["est_decode"]),
    )


def offer_from_content(doc: Mapping) -> Offer:
    p1_doc: Mapping | None = doc.get("p1")
    offer = Offer(
        item_id=str(doc["item_id"]),
        boundary=Boundary(doc["boundary"]),
        unit_id=str(doc["unit_id"]),
        p0=P0Plan(requests=tuple(_request(r) for r in doc["p0"]["requests"])),
        p1=None if p1_doc is None else P1Plan(
            selector=_request(p1_doc["selector"]),
            render=RenderDescriptor(
                est_cpu_ms=float(p1_doc["render"]["est_cpu_ms"]),
                candidate_count=int(p1_doc["render"]["candidate_count"]),
            ),
        ),
        submitted_at_ns=int(doc["submitted_at_ns"]),
        deadline_ns=None if doc.get("deadline_ns") is None else int(doc["deadline_ns"]),
    )
    if canonical_json(offer.content()) != canonical_json(dict(doc)):
        raise ContractViolation(
            f"offer {offer.item_id} does not round-trip; a field the codec does not carry "
            "would be missing from the plan the broker decides on")
    return offer


def offer_digest(offer: Offer) -> str:
    """Identity of an offer's content. Two submissions of one item must carry the same digest."""
    return sha256_hex(canonical_json(offer.content()))


assert_contract(CONTRACT, CONTRACT_SHA256)
