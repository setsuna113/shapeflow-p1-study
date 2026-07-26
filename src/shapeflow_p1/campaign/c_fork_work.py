"""Attribute work to a C fork without inventing a saving that does not exist.

A fork arm physically performs only two model calls -- its compressor and one final writer --
because the whole upstream trajectory ran once, in the anchor. The tempting bookkeeping is to
record each arm's work as those two calls. That would be a fabrication: it says the arm
produced a report having done almost no work, and a P1 arm compared against a P0 arm on that
basis shows an enormous saving that is really just the upstream nobody counted.

So there are two different numbers, and they answer different questions:

``POST_BOUNDARY_INCREMENTAL``
    Work strictly after the boundary. This is the *direct* C effect -- what the reducer itself
    costs -- and it is the honest denominator for "is this reducer cheaper than that one".

``TOTAL_WITH_SHARED_UPSTREAM``
    The same identical upstream constant added to every arm of the boundary, plus that arm's
    own post-fork work. This is what a deployed system would actually spend, and it is the
    number any decision-facing *fraction* must be computed on. A fractional saving computed on
    post-boundary work alone is precisely the fabrication above.

The upstream constant is derived, not assumed: the anchor's events are partitioned by op class
and ordinal (never by wall clock, so the receipt is stable), and the shared prefix for the k-th
boundary is every anchor event dispatched strictly before the k-th compressor call. If the
anchor's compressor count does not match its captured C boundary count, the anchor is unusable
and says so rather than truncating to whichever is shorter.

Zero upstream is refused in three independent places -- writer, cross-arm, reader -- because
this is the one arithmetic error in the whole study that would look like a spectacular result.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = [
    "COMPRESSOR_OPS",
    "ForkAccountingError",
    "SharedUpstream",
    "attribution_for_arm",
    "partition_anchor_work",
    "total_work_endpoint",
]

#: Op classes that mark a close boundary in the anchor's request stream.
COMPRESSOR_OPS = ("COMPRESSOR_P0", "COMPRESSOR_P1_SELECTOR")

#: Op classes a fork arm may issue at all. Anything else means upstream work leaked in.
POST_BOUNDARY_OPS = ("COMPRESSOR_P0", "COMPRESSOR_P1_SELECTOR", "FINAL_WRITER")


class ForkAccountingError(RuntimeError):
    """Work cannot be attributed honestly, so it is not attributed at all."""


@dataclass(frozen=True)
class SharedUpstream:
    """The upstream constant every arm of one boundary carries, identically."""

    anchor_work_key: str
    boundary_ordinal: int
    call_ids: tuple[str, ...]
    service_seconds: float
    prompt_tokens: int
    completion_tokens: int
    by_op: Mapping[str, int]

    @property
    def call_count(self) -> int:
        return len(self.call_ids)

    def content(self) -> dict:
        return {
            "schema_version": "anchor_work_partition_v1",
            "anchor_work_key": self.anchor_work_key,
            "boundary_ordinal": self.boundary_ordinal,
            "shared_call_ids": list(self.call_ids),
            "shared_call_count": self.call_count,
            "shared_service_seconds": self.service_seconds,
            "shared_prompt_tokens": self.prompt_tokens,
            "shared_completion_tokens": self.completion_tokens,
            "shared_by_op": dict(sorted(self.by_op.items())),
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def partition_anchor_work(
    events: Sequence, *, anchor_work_key: str, boundary_count: int
) -> list[SharedUpstream]:
    """Split the anchor's request stream into one shared prefix per close boundary.

    Ordered by dispatch timestamp, which is well defined because the causal layer serialises
    upstream dispatch. The partition itself is by op class and ordinal, so the receipt does not
    change if the same run is re-read.
    """
    ordered = sorted(events, key=lambda e: (e.upstream_dispatch_ts, e.call_id))
    compressor_positions = [
        index for index, event in enumerate(ordered)
        if str(getattr(event.op_class, "value", event.op_class)) in COMPRESSOR_OPS
    ]
    if len(compressor_positions) != boundary_count:
        raise ForkAccountingError(
            f"anchor {anchor_work_key} issued {len(compressor_positions)} compressor call(s) "
            f"but captured {boundary_count} close boundary(ies); the shared upstream cannot be "
            "attributed and this anchor is unusable"
        )

    shared: list[SharedUpstream] = []
    for ordinal, position in enumerate(compressor_positions):
        prefix = ordered[:position]
        if not prefix:
            raise ForkAccountingError(
                f"boundary {ordinal} of anchor {anchor_work_key} has no upstream work before "
                "it; a close boundary is always preceded by the research that produced it"
            )
        by_op: dict[str, int] = {}
        for event in prefix:
            key = str(getattr(event.op_class, "value", event.op_class))
            by_op[key] = by_op.get(key, 0) + 1
        shared.append(SharedUpstream(
            anchor_work_key=anchor_work_key,
            boundary_ordinal=ordinal,
            call_ids=tuple(event.call_id for event in prefix),
            service_seconds=sum(event.service_seconds for event in prefix),
            prompt_tokens=sum(int(event.prompt_tokens) for event in prefix),
            completion_tokens=sum(int(event.completion_tokens) for event in prefix),
            by_op=by_op,
        ))
    return shared


def attribution_for_arm(shared: SharedUpstream, post_boundary: Mapping[str, float]) -> dict:
    """The arm's work record, refusing to write one that claims no upstream.

    This is the *writer* layer. A record without a positive shared constant must never reach
    the store, because everything downstream would then be computing a saving against an
    upstream of zero.
    """
    if shared.call_count < 1:
        raise ForkAccountingError(
            "refusing to write a fork work record with no shared upstream calls"
        )
    if shared.service_seconds <= 0.0 or shared.prompt_tokens <= 0:
        raise ForkAccountingError(
            f"shared upstream is non-positive (service={shared.service_seconds}s, "
            f"prompt_tokens={shared.prompt_tokens}); a boundary is always preceded by real work"
        )

    post_service = float(post_boundary.get("service_seconds", 0.0))
    post_prompt = int(post_boundary.get("prompt_tokens", 0))
    post_completion = int(post_boundary.get("completion_tokens", 0))
    forbidden = sorted(set(post_boundary.get("by_op", {})) - set(POST_BOUNDARY_OPS))
    if forbidden:
        raise ForkAccountingError(
            f"a fork arm issued op classes outside the close boundary: {forbidden}"
        )

    total_service = shared.service_seconds + post_service
    total_prompt = shared.prompt_tokens + post_prompt
    total_completion = shared.completion_tokens + post_completion
    # Cheap, but it is the invariant the whole scheme rests on: the total can never be below
    # the constant every arm carries.
    for label, total, constant in (
        ("service_seconds", total_service, shared.service_seconds),
        ("prompt_tokens", total_prompt, shared.prompt_tokens),
        ("completion_tokens", total_completion, shared.completion_tokens),
    ):
        if total < constant:
            raise ForkAccountingError(
                f"total {label} {total} is below the shared upstream constant {constant}"
            )

    return {
        "schema_version": "c_fork_work_attribution_v1",
        "upstream_attribution": {
            "mode": "SHARED_FROZEN_ANCHOR",
            "anchor_work_key": shared.anchor_work_key,
            "anchor_work_partition_sha256": shared.digest,
            "boundary_ordinal": shared.boundary_ordinal,
            "shared_call_count": shared.call_count,
            "shared_service_seconds": shared.service_seconds,
            "shared_prompt_tokens": shared.prompt_tokens,
            "shared_completion_tokens": shared.completion_tokens,
        },
        # The direct C effect. Labelled, because a fraction computed on this is the
        # fabricated saving.
        "post_boundary_incremental": {
            "basis": "POST_BOUNDARY_INCREMENTAL",
            "service_seconds": post_service,
            "prompt_tokens": post_prompt,
            "completion_tokens": post_completion,
            "by_op": dict(sorted(post_boundary.get("by_op", {}).items())),
        },
        # What a deployed system spends, and the only basis for a decision-facing fraction.
        "total_with_shared_upstream": {
            "basis": "TOTAL_WITH_SHARED_UPSTREAM",
            "service_seconds": total_service,
            "prompt_tokens": total_prompt,
            "completion_tokens": total_completion,
        },
    }


def assert_arms_share_one_upstream(records: Sequence[Mapping]) -> None:
    """The *cross-arm* layer: one boundary, one upstream constant, for every arm.

    Two arms of a pair that disagree here are not comparable, however plausible each looks on
    its own -- their difference would include a difference in what was counted.
    """
    if not records:
        raise ForkAccountingError("a boundary with no arms cannot be checked for agreement")
    digests = {str(r["upstream_attribution"]["anchor_work_partition_sha256"]) for r in records}
    if len(digests) != 1:
        raise ForkAccountingError(
            f"arms of one boundary carry {len(digests)} different upstream partitions; "
            "their work is not comparable"
        )
    for field in ("shared_service_seconds", "shared_prompt_tokens",
                  "shared_completion_tokens", "shared_call_count"):
        values = {r["upstream_attribution"][field] for r in records}
        if len(values) != 1:
            raise ForkAccountingError(
                f"arms of one boundary disagree on {field}: {sorted(values)}"
            )


def total_work_endpoint(record: Mapping | None) -> dict:
    """The *reader* layer: produce a number, or say why there is none. Never zero.

    ``estimands._work`` already refuses non-positive service seconds, but the failure this
    guards against is subtler -- a record with the upstream silently missing would present a
    small, positive, entirely wrong total. NOT_ESTIMABLE is the honest answer.
    """
    if not record:
        return {"status": "NOT_ESTIMABLE", "reason": "SHARED_UPSTREAM_WORK_MISSING"}
    attribution = record.get("upstream_attribution") or {}
    if attribution.get("mode") != "SHARED_FROZEN_ANCHOR":
        return {"status": "NOT_ESTIMABLE", "reason": "SHARED_UPSTREAM_WORK_MISSING"}
    for field in ("shared_service_seconds", "shared_prompt_tokens", "shared_call_count"):
        value = attribution.get(field)
        if not isinstance(value, (int | float)) or value <= 0:
            return {"status": "NOT_ESTIMABLE", "reason": "SHARED_UPSTREAM_WORK_MISSING"}
    total = record.get("total_with_shared_upstream") or {}
    service = total.get("service_seconds")
    if not isinstance(service, (int | float)) or service <= 0:
        return {"status": "NOT_ESTIMABLE", "reason": "NON_POSITIVE_TOTAL_SERVICE"}
    return {
        "status": "OK",
        "basis": "TOTAL_WITH_SHARED_UPSTREAM",
        "service_seconds": float(service),
        "prompt_tokens": int(total.get("prompt_tokens", 0)),
        "completion_tokens": int(total.get("completion_tokens", 0)),
    }
