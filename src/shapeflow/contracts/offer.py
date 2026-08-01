"""The offer interface, as types.

`OFFER_INTERFACE_v1.md` is the contract; this is its type surface. The shapes here are chosen so
that the contract's obligations are hard to violate rather than merely documented:

- a plan is a **description of what would be sent**, and carries no way to send it;
- an :class:`Offer` holds both plans, so "no form is started at submission" is the only thing it
  *can* mean;
- a :class:`Decision` names one form per admitted item, so a mixed-form item is unconstructible
  rather than merely forbidden;
- every type is frozen and canonically serializable, so byte-identical determinism (C10) is a
  property of the data rather than of the code that happens to build it.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Optional

from ..canonical import canonical_json
from ..hashing import sha256_hex
from ..runtime.request_tags import OpClass
from .observability import ClosedFeatureMap
from .versions import assert_contract

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = [
    "Form", "Boundary", "RequestDescriptor", "RenderDescriptor", "P0Plan", "P1Plan", "Offer",
    "SnapshotVersion", "SchedulerSnapshot", "Decision", "MaterializeResult",
    "StaleSnapshot", "ContractViolation",
]


class ContractViolation(RuntimeError):
    """A construction the offer contract forbids."""


class StaleSnapshot(RuntimeError):
    """The snapshot a decision was computed against is no longer current.

    Raised by ``materialize`` having materialized **nothing**. Carrying the version makes the
    retry attributable to a specific tick rather than to a generic failure.
    """

    def __init__(self, version: "SnapshotVersion") -> None:
        super().__init__(f"snapshot {version.epoch}#{version.tick} is stale; nothing materialized")
        self.version = version


class Form(Enum):
    P0 = "P0"
    P1 = "P1"


class Boundary(Enum):
    H = "H"
    C = "C"


@dataclass(frozen=True)
class RequestDescriptor:
    """What *would* be sent. Deliberately not a request.

    There is no client, no transport and no send method here, so an implementation cannot
    accidentally dispatch a plan it was only supposed to price. The demand estimates are the
    broker's cost-model input and are the only numbers a decision may use about this request.
    """

    op_class: OpClass
    prompt_sha256: str
    max_tokens: int
    est_uncached_prefill: int
    est_cached_prefill: int
    est_decode: int

    def content(self) -> dict:
        return {
            "op_class": self.op_class.value,
            "prompt_sha256": self.prompt_sha256,
            "max_tokens": self.max_tokens,
            "est_uncached_prefill": self.est_uncached_prefill,
            "est_cached_prefill": self.est_cached_prefill,
            "est_decode": self.est_decode,
        }


@dataclass(frozen=True)
class RenderDescriptor:
    """The CPU half of a P1 plan: validate ids, merge spans, order, render.

    Costed separately and reported separately. It is never merged into W, which counts model
    work; a renderer that got folded into the token total would let CPU time masquerade as a
    GPU saving.
    """

    est_cpu_ms: float
    candidate_count: int

    def content(self) -> dict:
        return {"est_cpu_ms": self.est_cpu_ms, "candidate_count": self.candidate_count}


@dataclass(frozen=True)
class P0Plan:
    """N summarization/compression requests, scheduled independently once materialized."""

    requests: tuple[RequestDescriptor, ...]

    def __post_init__(self) -> None:
        if not self.requests:
            raise ContractViolation("a P0 plan with no requests is not a plan")

    def content(self) -> dict:
        return {"form": "P0", "requests": [r.content() for r in self.requests]}


@dataclass(frozen=True)
class P1Plan:
    """One selector request plus a CPU render."""

    selector: RequestDescriptor
    render: RenderDescriptor

    def content(self) -> dict:
        return {"form": "P1", "selector": self.selector.content(), "render": self.render.content()}


@dataclass(frozen=True)
class Offer:
    """One unit of pending work, with both futures and neither started.

    ``unit_id`` is the boundary's own unit -- the assistant turn's sibling batch at H, the close
    at C -- so an offer can always be traced back to the decision unit the results are counted in.

    ``p1`` is optional: a job whose P1 plan is structurally impossible (nothing chunkable, a
    namespace-illegal candidate set) offers only P0. That is not a fallback; it is an offer with
    one alternative, and it is the honest way to say so.
    """

    item_id: str
    boundary: Boundary
    unit_id: str
    p0: P0Plan
    submitted_at_ns: int
    p1: Optional[P1Plan] = None
    deadline_ns: Optional[int] = None

    def forms(self) -> tuple[Form, ...]:
        return (Form.P0,) if self.p1 is None else (Form.P0, Form.P1)

    def plan_for(self, form: Form):
        if form is Form.P0:
            return self.p0
        if self.p1 is None:
            raise ContractViolation(
                f"item {self.item_id} offers no P1 plan; admitting it as P1 would materialize "
                "a form that was never offered"
            )
        return self.p1

    def content(self) -> dict:
        return {
            "item_id": self.item_id,
            "boundary": self.boundary.value,
            "unit_id": self.unit_id,
            "p0": self.p0.content(),
            "p1": self.p1.content() if self.p1 else None,
            "submitted_at_ns": self.submitted_at_ns,
            "deadline_ns": self.deadline_ns,
        }


@dataclass(frozen=True)
class SnapshotVersion:
    """Identifies a tick *within an engine epoch*.

    The epoch is part of the identity because a restarted engine resets its tick counter. Without
    it, a snapshot from before a restart could compare equal to a current one, and step 6 would
    silently never fire -- the failure mode being that staleness detection appears to work
    perfectly and has in fact been disabled.
    """

    epoch: str
    tick: int

    def content(self) -> dict:
        return {"epoch": self.epoch, "tick": self.tick}


@dataclass(frozen=True)
class SchedulerSnapshot:
    """Engine state at one tick, plus how old it is.

    ``taken_at_ns`` exists so staleness is measurable rather than assumed. A host that cannot sit
    inside the scheduler reports a real age here, and the pre-registered ceiling on that age is a
    validity criterion for the broker itself.
    """

    version: SnapshotVersion
    taken_at_ns: int
    features: ClosedFeatureMap

    def age_ns(self, now_ns: int) -> int:
        return max(0, now_ns - self.taken_at_ns)


@dataclass(frozen=True)
class Decision:
    """One tick's atomic answer to both questions at once.

    ``admitted`` and ``form`` are produced together by construction: a decision that named
    admissions without forms, or forms without admissions, would be the sequential baseline this
    study exists to beat, and the type refuses to express it.
    """

    admitted: tuple[str, ...]
    form: Mapping[str, Form]
    deferred: tuple[str, ...]
    snapshot_version: SnapshotVersion
    solve_ns: int
    fail_closed: bool = False
    reason: str = ""

    def __post_init__(self) -> None:
        missing = [i for i in self.admitted if i not in self.form]
        if missing:
            raise ContractViolation(
                f"admitted without a form: {missing}. Admission and form choice are one decision."
            )
        extra = [i for i in self.form if i not in self.admitted]
        if extra:
            raise ContractViolation(
                f"form chosen for an item that was not admitted: {extra}. A form assigned to a "
                "deferred item is a materialization nobody authorized."
            )
        overlap = sorted(set(self.admitted) & set(self.deferred))
        if overlap:
            raise ContractViolation(f"both admitted and deferred: {overlap}")

    def content(self) -> dict:
        return {
            "admitted": list(self.admitted),
            "form": {k: v.value for k, v in sorted(self.form.items())},
            "deferred": list(self.deferred),
            "snapshot_version": self.snapshot_version.content(),
            "fail_closed": self.fail_closed,
            "reason": self.reason,
        }

    @property
    def digest(self) -> str:
        """Identity of the decision itself.

        Excludes ``solve_ns``: two runs of a deterministic broker on identical input must compare
        equal (C10), and how long the solver took is a measurement of the machine, not part of
        the answer.
        """
        return sha256_hex(canonical_json(self.content()))


@dataclass(frozen=True)
class MaterializeResult:
    """What materialization actually produced.

    ``dispatched`` counts sub-requests per item so a caller can assert that a P0 item produced N
    of them and a P1 item produced one -- the observable difference between the two forms.
    """

    materialized: tuple[str, ...]
    dispatched: Mapping[str, int]
    degraded_to_p0: tuple[str, ...] = ()

    def content(self) -> dict:
        return {
            "materialized": list(self.materialized),
            "dispatched": dict(sorted(self.dispatched.items())),
            "degraded_to_p0": list(self.degraded_to_p0),
        }


assert_contract(CONTRACT, CONTRACT_SHA256)
