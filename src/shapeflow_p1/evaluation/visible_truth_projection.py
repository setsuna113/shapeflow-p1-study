"""What the compressor could possibly have kept: the denominator C_VISIBLE is scored against.

``C_VISIBLE`` is the compressor-only experiment. Its claim is that the selector saw exactly what
P0's compressor saw -- no more -- so the only fair question to ask of it is "of the information
that had already reached the compressor's input, how much survived?".

That makes the denominator specific: a truth atom counts against the C reducer only if it is
``EXPLICITLY_VISIBLE`` in the exact ``researcher_messages`` bytes the compressor was handed.
An atom that never made it into those bytes -- because an upstream page summary dropped it, or
because the researcher never retrieved it -- is ``NOT_VISIBLE``, and penalising the reducer for
it would attribute an upstream loss to the compressor. That is the single most likely way to
manufacture a false C_VISIBLE failure.

Three rules the projection follows, each protecting a different boundary:

**Exact bytes only.** Visibility is decided over the frozen message bytes, addressed by
:class:`~shapeflow_p1.evidence.identity` visible-message spans. The raw registry is never
consulted: reverse-mapping a model-written page summary back to the page it summarised would
credit the compressor with text it never received, which is the ``C_REGISTRY`` treatment and a
different experiment.

**Only capture-time-attributed tool output is evidence.** Spans whose message is
assistant-authored stay ``MODEL_DERIVED_CONTEXT``. Tool messages with no capture-time source
occurrence stay ``TOOL_UNATTRIBUTED_CONTEXT``. Both may carry useful context, but neither can
make a truth atom count as visible: the model asserting something is not the source saying it,
and a tool-shaped string is not a citable source when its provenance was never captured.

**Evaluator-only.** This module runs after treatment output is frozen, under the evaluator
identity, and nothing it produces is ever handed back to a selector, aggregator or preflight.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = [
    "EXPLICITLY_VISIBLE",
    "NOT_VISIBLE",
    "AMBIGUOUS",
    "ProjectedAtom",
    "VisibleTruthProjection",
    "project_visible_truth",
    "visible_recall",
]

EXPLICITLY_VISIBLE = "EXPLICITLY_VISIBLE"
NOT_VISIBLE = "NOT_VISIBLE"
AMBIGUOUS = "AMBIGUOUS"

#: Only tool output with capture-time source provenance can carry evidence.
_EVIDENCE_KINDS = frozenset({"TOOL_EVIDENCE"})


@dataclass(frozen=True)
class ProjectedAtom:
    truth_atom_id: str
    supporting_visible_span_ids: tuple[str, ...]
    projection_status: str

    def content(self) -> dict:
        return {
            "truth_atom_id": self.truth_atom_id,
            "supporting_visible_span_ids": list(self.supporting_visible_span_ids),
            "projection_status": self.projection_status,
        }


@dataclass
class VisibleTruthProjection:
    checkpoint_hash: str
    visible_compressor_view_hash: str
    projected_atoms: tuple[ProjectedAtom, ...]
    projection_model_prompt_hash: str = ""
    audit_status: str = "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT"
    notes: dict = field(default_factory=dict)

    def content(self) -> dict:
        return {
            "checkpoint_hash": self.checkpoint_hash,
            "visible_compressor_view_hash": self.visible_compressor_view_hash,
            "projected_atoms": [a.content() for a in self.projected_atoms],
            "projection_model_prompt_hash": self.projection_model_prompt_hash,
            "audit_status": self.audit_status,
            "notes": dict(sorted(self.notes.items())),
        }

    @property
    def content_sha256(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    def to_json(self) -> dict:
        body = self.content()
        body["content_sha256"] = self.content_sha256
        return body

    @property
    def explicitly_visible_ids(self) -> tuple[str, ...]:
        return tuple(a.truth_atom_id for a in self.projected_atoms
                     if a.projection_status == EXPLICITLY_VISIBLE)


#: decide(atom_text, span_text) -> "entail" | "unrelated" | "uncertain".
#: Injected so the projection is testable deterministically and so the judge -- when one is used
#: -- reaches this module only through an explicit, recorded contract.
VisibilityDecider = Callable[[str, str], str]


def _lexical_decider(min_shared_anchors: int = 1) -> VisibilityDecider:
    """A deterministic fallback: an atom is visible if its anchors appear in the span.

    Used when no judge is available. It is conservative in the right direction -- it can only
    say "entail" when the literal tokens are present -- so it under-counts visibility rather
    than crediting the compressor with information it never received.
    """
    import re

    token = re.compile(r"[A-Za-z0-9][A-Za-z0-9.\-/%]*")

    def decide(atom_text: str, span_text: str) -> str:
        anchors = [t for t in token.findall(atom_text)
                   if any(ch.isdigit() for ch in t) or t[:1].isupper()]
        if not anchors:
            return "uncertain"
        hay = span_text.lower()
        hits = sum(1 for a in anchors if a.lower() in hay)
        if hits >= max(min_shared_anchors, len(anchors) // 2 + 1):
            return "entail"
        return "unrelated" if hits == 0 else "uncertain"

    return decide


def project_visible_truth(
    *,
    checkpoint_hash: str,
    visible_view_hash: str,
    spans: Sequence[dict],
    span_texts: dict,
    atoms: Iterable[tuple[str, str]],
    decider: VisibilityDecider | None = None,
    projection_prompt_hash: str = "",
) -> VisibleTruthProjection:
    """Decide, for each truth atom, whether the compressor's input carried it.

    ``spans`` are visible-message spans; ``span_texts`` maps span id to its exact bytes decoded
    as text; ``atoms`` is ``(atom_id, atom_text)``.

    Only ``TOOL_EVIDENCE`` spans can make an atom visible. A
    ``MODEL_DERIVED_CONTEXT`` span -- the researcher's own reasoning -- may mention the same
    fact, and a ``TOOL_UNATTRIBUTED_CONTEXT`` span may look like a search result, but neither
    has source provenance. Treating either as visible evidence would credit the compressor
    with evidence the capture-time record cannot establish.
    """
    decide = decider or _lexical_decider()
    evidence_spans = [s for s in spans if s.get("kind") in _EVIDENCE_KINDS]

    projected: list[ProjectedAtom] = []
    for atom_id, atom_text in atoms:
        supporting: list[str] = []
        uncertain = False
        for span in evidence_spans:
            span_id = span.get("visible_span_id") or span.get("span_id") or ""
            text = span_texts.get(span_id, "")
            if not text:
                continue
            verdict = decide(atom_text, text)
            if verdict == "entail":
                supporting.append(span_id)
            elif verdict == "uncertain":
                uncertain = True
        if supporting:
            status = EXPLICITLY_VISIBLE
        elif uncertain:
            # Neither visible nor provably absent. Excluded from the denominator rather than
            # guessed: an atom counted as visible on a maybe would penalise the reducer for
            # something that may never have been there.
            status = AMBIGUOUS
        else:
            status = NOT_VISIBLE
        projected.append(ProjectedAtom(
            truth_atom_id=atom_id,
            supporting_visible_span_ids=tuple(sorted(supporting)),
            projection_status=status,
        ))

    return VisibleTruthProjection(
        checkpoint_hash=checkpoint_hash,
        visible_compressor_view_hash=visible_view_hash,
        projected_atoms=tuple(projected),
        projection_model_prompt_hash=projection_prompt_hash,
        notes={
            "evidence_spans": len(evidence_spans),
            "model_derived_spans": sum(
                1 for span in spans
                if span.get("kind") == "MODEL_DERIVED_CONTEXT"
            ),
            "unattributed_tool_spans": sum(
                1 for span in spans
                if span.get("kind") == "TOOL_UNATTRIBUTED_CONTEXT"
            ),
            "user_context_spans": sum(
                1 for span in spans
                if span.get("kind") == "USER_CONTEXT"
            ),
            "non_citable_context_spans": sum(
                1 for span in spans if span.get("kind") not in _EVIDENCE_KINDS
            ),
        },
    )


def visible_recall(
    projection: VisibleTruthProjection,
    retained_atom_ids: Iterable[str],
    *,
    weights: dict | None = None,
) -> float | None:
    """Weighted recall over the ``EXPLICITLY_VISIBLE`` atoms only.

    Returns ``None`` when nothing was visible: with an empty denominator there is no question to
    answer, and reporting 0.0 would record a failure the reducer could not have avoided.
    """
    visible = projection.explicitly_visible_ids
    if not visible:
        return None
    retained = set(retained_atom_ids)
    if weights is None:
        resolved_weights = {atom_id: 1.0 for atom_id in visible}
    else:
        missing = set(visible) - set(weights)
        if missing:
            raise ValueError(
                "visible truth atoms lack frozen weights: "
                f"{sorted(missing)}"
            )
        resolved_weights: dict[str, float] = {}
        for atom_id in visible:
            raw = weights[atom_id]
            if isinstance(raw, bool):
                raise ValueError(
                    f"visible truth atom {atom_id!r} has a boolean weight")
            try:
                weight = float(raw)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"visible truth atom {atom_id!r} has a non-numeric weight"
                ) from exc
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(
                    f"visible truth atom {atom_id!r} has an invalid weight {raw!r}")
            resolved_weights[atom_id] = weight
    total = sum(resolved_weights[a] for a in visible)
    kept = sum(resolved_weights[a] for a in visible if a in retained)
    return kept / total if total else None
