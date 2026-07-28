"""A reference broker: correct against the contract, deliberately simple as a policy.

This exists so the contract has something conformant to test against, and so B5 has a baseline.
It is **not** the ShapeFlow policy: no risk predictor, no 2-swap packing, no cost model beyond
the demand estimates on the plans themselves. Those arrive later and must pass the same suite.

Its one real property is that admission and form are decided **together**, in a single pass over
a jointly-ordered candidate list, rather than in two stages. The sequential baselines that
Freeze-1 §7 arms ④ and ⑤ measure against are separate implementations precisely so the difference
is a difference in code and not in configuration.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..contracts.offer import Decision, Form, Offer, SchedulerSnapshot
from ..contracts.versions import assert_contract

CONTRACT = "OFFER_INTERFACE_v1.md"
CONTRACT_SHA256 = "f9a400457f00ed7cf918fd202ac0307a9cf9c196c4ce7f22a94f3ce575c93084"

__all__ = ["ReferenceBroker", "GreedyJointBroker", "plan_weight"]


def plan_weight(offer: Offer, form: Form) -> int:
    """A plan's cost in decode-token equivalents, from its own estimates only.

    Uses the descriptors and nothing else. A weight that consulted engine state would make the
    ordering depend on the snapshot, and C10's determinism would then be a property of the
    snapshot rather than of the broker.
    """
    plan = offer.plan_for(form)
    requests = plan.requests if form is Form.P0 else (plan.selector,)
    return sum(r.est_uncached_prefill + r.est_cached_prefill + r.est_decode for r in requests)


@dataclass(frozen=True)
class GreedyJointBroker:
    """Admit by joint (item, form) value until the round's capacity is used.

    The candidate list is *jointly* ordered over items and forms: every admissible (item, form)
    pair competes in one ranking, so a cheap P1 for one item can displace an expensive P0 for
    another. A two-stage policy could not express that, which is the whole of claim C2b.

    Capacity here is a fixed token allowance per tick, which is a placeholder for the real cost
    model. Deliberately crude: a plausible-looking cost model that had never been calibrated
    would make this reference broker look like a result.
    """

    capacity_tokens: int = 8192
    #: When false, P1 is never chosen. Used to build the P0-only baseline from the same code
    #: path, so the baseline and the treatment cannot differ by anything except the choice.
    allow_p1: bool = True

    def decide(self, offers, snapshot: SchedulerSnapshot, budget_ns: int) -> Decision:
        candidates = []
        for offer in offers:
            for form in offer.forms():
                if form is Form.P1 and not self.allow_p1:
                    continue
                candidates.append((plan_weight(offer, form), offer.item_id, form))
        # Cheapest first, then by item id and form so ties resolve identically on every run --
        # C10 is a property of this sort key.
        candidates.sort(key=lambda c: (c[0], c[1], c[2].value))

        admitted: list[str] = []
        form: dict[str, Form] = {}
        used = 0
        for weight, item_id, chosen in candidates:
            if item_id in form:
                continue
            if used + weight > self.capacity_tokens:
                continue
            form[item_id] = chosen
            admitted.append(item_id)
            used += weight

        deferred = tuple(o.item_id for o in offers if o.item_id not in form)
        return Decision(
            admitted=tuple(admitted),
            form=form,
            deferred=deferred,
            snapshot_version=snapshot.version,
            solve_ns=0,
            reason=f"greedy joint: {used}/{self.capacity_tokens} tokens",
        )


#: The default conformant broker.
ReferenceBroker = GreedyJointBroker

assert_contract(CONTRACT, CONTRACT_SHA256)
