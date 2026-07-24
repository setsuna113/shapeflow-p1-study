"""Publish-time preflight: structural gates, and nothing about truth.

Preflight is the last check before P1 output is published downstream. It verifies *form*:
ids resolve, offsets reconstruct, citations close, budgets hold, bridges stay bounded, and a
declared contradiction keeps both sides. It must never consult a TruthPacket, gold facet, or
critical-item list -- those are evaluator-only, and letting any of them gate the treatment
would let the selector be tuned against the answer key it is later scored on. Truth-critical
recall is computed *after* publish, by the isolated evaluator.

Two design rules make the check mean what it says:

**Preflight resolves the bytes itself.** It takes ``snapshot_texts`` and ``visible_views``, not
a ``source_text_for`` callable. A caller-supplied body supplier sat squarely on the publication
path: hashing a span against the snapshot and then rendering whatever the callable returned let
"INJECTED BODY" pass with ok=True. What is verified and what is published are now the same
bytes because they come from the same place.

**Preflight never raises.** Every failure is a recorded error and a fail-closed result with
``rendered=None``. The adapter's whole-batch P0 fallback needs a decision object; an exception
escaping here would make the batch's failure path whatever the caller's ``except`` happens to
be. A failure is recorded, not hidden: in a component trial the sample is marked failed; in an
end-to-end run the policy falls back to P0 and the P1 cost already spent is still charged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..evidence.chunkers import Tokenizer
from ..evidence.lineage import lineage_closure_errors, reconstruction_errors
from ..hashing import sha256_hex
from .aggregators import AggregatedEvidence
from .contracts import OVERALL_FACET, ParsedSelection
from .renderer import RenderResult, context_text_for, render

__all__ = [
    "PreflightConfig", "PreflightResult", "preflight",
    "bridge_novelty_errors", "ref_errors", "candidate_view_sha",
]


REF_FIELDS = ("context_refs", "heading_refs")


def ref_errors(spans: list[dict], snapshot_texts: dict[str, str]) -> list[str]:
    """Return an error per context/heading ref that does not reproduce its recorded bytes.

    Both reach the selector prompt and the downstream prompt, so both are evidence and get
    evidence's treatment: addressed by offset into a frozen snapshot and re-hashed here. As free
    strings they were unbound injection channels -- the span id was unaffected, RAW_SOURCE
    reconstruction passed, preflight returned ok, and the injected text was rendered anyway.

    Note what this alone does NOT catch: a ref *repointed* to different but genuine bytes with a
    correctly recomputed hash. Per-field integrity holds there; only ``candidate_view_sha``,
    taken over the whole offered set before the selector ran, detects it.
    """
    errors: list[str] = []
    for span in spans:
        sid = (span.get("span_id") or span.get("visible_span_id") or "?")[:12]
        for field_name in REF_FIELDS:
            kind = field_name.removesuffix("_refs")
            for ref in span.get(field_name) or []:
                text = snapshot_texts.get(ref["content_hash"])
                if text is None:
                    errors.append(
                        f"span {sid}: {kind} snapshot {ref['content_hash'][:12]} missing"
                    )
                    continue
                start, end = ref["char_start"], ref["char_end"]
                if not (0 <= start <= end <= len(text)):
                    errors.append(
                        f"span {sid}: {kind} offsets [{start},{end}] out of bounds "
                        f"(len {len(text)})"
                    )
                    continue
                if sha256_hex(text[start:end].encode("utf-8")) != ref["text_sha256"]:
                    errors.append(
                        f"span {sid}: {kind} hash mismatch -- the rendered {kind} is not the "
                        "frozen bytes it claims to address"
                    )
    return errors


def _candidate_view_sha(
    registry: dict, offered_span_ids: list[str], *, namespace: str,
    prompt_bundle_version: str = "", renderer_grouping_version: str = "",
) -> str:
    """Digest of the exact candidate view a selector was shown.

    Per-field hashes cannot detect a ref *repointed* to other legitimate bytes, because the new
    bytes hash correctly. What identifies the view is the whole offered set together: which
    spans, in which order (the order allocates the E-labels), under which namespace, with which
    body/context/heading identities, rendered by which prompt and grouping version.

    It covers **every offered candidate, not only the chosen ones**. An unselected candidate
    still shapes what the model picked -- a forged one can steer a selection it never appears
    in -- so the view is verified before the selector call and again at publish.
    """
    from ..canonical import canonical_json

    entries = []
    for label_index, span_id in enumerate(offered_span_ids, start=1):
        span = registry[span_id]
        entries.append({
            "label": f"E{label_index}",
            "span_id": span_id,
            "namespace": span.get("namespace"),
            # Body identity, by namespace. Both are the recorded digest of the exact bytes.
            "body_sha256": span.get("text_sha256") or span.get("exact_text_sha256"),
            "context_refs": [
                {k: r[k] for k in ("content_hash", "char_start", "char_end", "text_sha256")}
                for r in span.get("context_refs") or []
            ],
            "heading_refs": [
                {k: r[k] for k in ("content_hash", "char_start", "char_end", "text_sha256")}
                for r in span.get("heading_refs") or []
            ],
        })
    return sha256_hex(canonical_json({
        "namespace": namespace,
        "prompt_bundle_version": prompt_bundle_version,
        "renderer_grouping_version": renderer_grouping_version,
        "candidates": entries,
    }))

# A number, or a capitalized word that is not sentence-initial boilerplate. Deliberately
# coarse: this is a *structural* guard against a bridge asserting a fact its cited spans do
# not contain, not a semantic entailment check (that is the evaluator's job, after publish).
_NUMBER = re.compile(r"\d[\d,.]*")
_ENTITY = re.compile(r"\b[A-Z][A-Za-z0-9-]{2,}\b")
# Connectives and framing words a bridge may legitimately open with.
_ALLOWED_CAPS = frozenset({
    "The", "This", "These", "Those", "Both", "Neither", "However", "Although", "While",
    "Whereas", "They", "It", "There", "Some", "Other", "Sources", "Evidence", "Together",
    "One", "Two", "Three", "First", "Second", "Third", "Overall", "Across", "Within",
})


def bridge_novelty_errors(
    bridge_text: str, cited_texts: list[str]
) -> list[str]:
    """Return an error per number/entity in ``bridge_text`` that no cited span contains.

    A bridge is bounded connective text whose whole justification is that it only *links*
    evidence. The moment it introduces a figure or a name that is nowhere in the spans it
    cites, it has become unsourced generation charged to the P1 arm -- and the citation makes
    it look sourced. Comparison is against the cited spans' raw text, so a number written the
    same way anywhere in them passes.
    """
    haystack = " ".join(cited_texts)
    errors: list[str] = []
    for number in set(_NUMBER.findall(bridge_text)):
        normalized = number.rstrip(".,")
        if normalized and normalized not in haystack:
            errors.append(f"introduces number {normalized!r} absent from its cited spans")
    for entity in set(_ENTITY.findall(bridge_text)):
        if entity in _ALLOWED_CAPS:
            continue
        if entity not in haystack:
            errors.append(f"introduces entity {entity!r} absent from its cited spans")
    return sorted(errors)


@dataclass(frozen=True)
class PreflightConfig:
    selected_token_budget: int
    bridge_token_cap_total: int | None = None
    bridge_token_cap_each: int | None = None
    # The namespace this variant is allowed to publish from. C_VISIBLE is "VISIBLE_MESSAGE":
    # a RAW_SOURCE span reaching its output means the selector read bytes P0's compressor never
    # had, which is C_REGISTRY provenance wearing a C_VISIBLE label. Left None only for tests
    # and for variants that legitimately span both (C_REGISTRY).
    expected_namespace: str | None = None


@dataclass
class PreflightResult:
    errors: list[str] = field(default_factory=list)
    # The render preflight actually verified. The caller publishes THIS, not something it
    # rendered separately -- otherwise the budget gate and the published bytes can diverge.
    rendered: RenderResult | None = None

    @property
    def ok(self) -> bool:
        return not self.errors


def preflight(
    *,
    selection: ParsedSelection,
    aggregated: AggregatedEvidence,
    registry: dict,
    snapshot_texts: dict[str, str],
    known_occurrence_ids: set[str],
    tokenizer: Tokenizer,
    label_for: Callable[[str], str],
    config: PreflightConfig,
    visible_views: dict[str, bytes] | None = None,
    source_meta: dict | None = None,
    query_status: dict[str, str] | None = None,
    candidate_view_sha: str | None = None,
    offered_span_ids: list[str] | None = None,
) -> PreflightResult:
    """Structurally verify a P1 publication and return the exact bytes to publish.

    Renders the evidence itself, from bytes it resolves itself, in canonical order -- so "what
    was checked" and "what is published" are the same object. Never raises: an integrity failure
    returns a result with ``rendered=None`` so the adapter can run its whole-batch P0 fallback
    on a decision rather than on an exception.
    """
    visible_views = visible_views or {}
    result = PreflightResult()
    body_for = _body_resolver(snapshot_texts, visible_views, result)

    # 1. Every aggregated span must be a real candidate.
    spans = []
    for item in aggregated.items:
        span = registry.get(item.span_id)
        if span is None:
            result.errors.append(f"published span {item.span_id[:12]} not in candidate registry")
        else:
            spans.append(span)

    # 1b. And must be something the selector actually chose, carrying the annotations the
    #     selector gave it. An aggregator may DROP; it may never rewrite or invent. Otherwise a
    #     `support/efficacy` becomes `background/safety` on the way out and every downstream
    #     metric attributes the rewrite to the model.
    result.errors.extend(_provenance_errors(selection, aggregated))

    # 1c. The whole offered candidate view must be the one the selector was shown. Per-field
    #     hashes miss a ref repointed to other genuine bytes; this digest does not. Unselected
    #     candidates are covered too, because a forged one steers selections it never joins.
    if candidate_view_sha is not None:
        if offered_span_ids is None:
            offered_span_ids = sorted(registry)
        from .renderer import RENDERER_GROUPING_VERSION

        actual = _candidate_view_sha(
            registry, offered_span_ids,
            namespace=config.expected_namespace or "RAW_SOURCE",
            renderer_grouping_version=RENDERER_GROUPING_VERSION,
        )
        if actual != candidate_view_sha:
            result.errors.append(
                f"candidate view digest changed since the selector call "
                f"({candidate_view_sha[:12]} -> {actual[:12]}); the bytes offered to the model "
                "are not the bytes being published from"
            )

    # 2. Every published span is in the namespace this variant may publish from.
    if config.expected_namespace is not None:
        for span in spans:
            got = span.get("namespace")
            if got != config.expected_namespace:
                sid = span.get("span_id") or span.get("visible_span_id") or "?"
                result.errors.append(
                    f"published span {sid[:12]} is in namespace {got!r}, but this variant may "
                    f"only publish {config.expected_namespace!r}"
                )

    # 3. Offsets still reconstruct the recorded bytes -- in BOTH namespaces. A visible-message
    #    span is reconstructed against the exact compressor bytes, which is what makes the
    #    compressor-only claim checkable rather than asserted.
    result.errors.extend(
        reconstruction_errors(spans, snapshot_texts, visible_views=visible_views)
    )

    # 4. Every citation resolves to a real occurrence in the frozen world.
    result.errors.extend(lineage_closure_errors(spans, known_occurrence_ids))

    # 4b. Interpretive context and heading breadcrumbs re-derive from the same frozen snapshot.
    #     Carried as free text they left the span id unchanged and every other check passing
    #     while the injected string was rendered into the downstream prompt.
    result.errors.extend(ref_errors(spans, snapshot_texts))

    # 5. A declared contradiction keeps both sides. The roles are read from what was PUBLISHED,
    #    not from the pre-aggregation selection: checking the selection lets an aggregator drop
    #    one side while preflight still sees both and passes.
    kept_ids = {it.span_id for it in aggregated.items}
    declared: dict[str, set[str]] = {}
    for item in selection.items:
        for facet, role in item.relations:
            if role:
                declared.setdefault(facet, set()).add(role)
    published: dict[str, set[str]] = {}
    for item in aggregated.items:
        if item.role:
            for facet in item.facet_ids or (OVERALL_FACET,):
                published.setdefault(facet, set()).add(item.role)
    for facet, roles in declared.items():
        if {"support", "contradict"} <= roles:
            survived = published.get(facet, set())
            for role in ("support", "contradict"):
                if role not in survived:
                    result.errors.append(
                        f"contradiction on facet {facet!r} was one-sided: no {role} span "
                        "survived into the published output"
                    )

    # 6. No bridge may cite a span that did not survive. A bridge whose support was dropped for
    #    budget is an unsourced assertion with a citation that resolves to nothing, and the
    #    renderer would have to label a span that is not in the output.
    for bridge in aggregated.bridges:
        dangling = [sid for sid in bridge.evidence_span_ids if sid not in kept_ids]
        if dangling:
            result.errors.append(
                f"bridge {bridge.text[:40]!r} cites "
                f"{', '.join(s[:12] for s in dangling)}, which is not in the published output"
            )
            continue
        # 6b. A bridge may connect its cited evidence; it may not add to it.
        cited = [body_for(registry[sid]) for sid in bridge.evidence_span_ids]
        for problem in bridge_novelty_errors(bridge.text, cited):
            result.errors.append(f"bridge {bridge.text[:40]!r} {problem}")

    # 7. Render it here, from the bytes resolved here, in canonical order -- and gate on THAT.
    #    The result is returned so the caller publishes exactly what preflight verified.
    #
    #    Rendering is skipped entirely once any integrity error exists. Rendering anyway is how
    #    a missing snapshot became a KeyError escaping preflight instead of a fail-closed
    #    result, leaving the batch's failure path to whatever the caller's `except` was.
    if not result.errors:
        rendered = render(
            aggregated, registry,
            source_text_for=body_for, label_for=label_for, tokenizer=tokenizer,
            source_meta=source_meta, visible_views=visible_views, query_status=query_status,
            context_text_for=context_text_for(snapshot_texts),
        )
        if not result.errors:      # body_for records rather than raises; re-check
            result.rendered = rendered
            if rendered.token_count > config.selected_token_budget:
                result.errors.append(
                    f"rendered {rendered.token_count} tokens exceeds selected_token_budget "
                    f"{config.selected_token_budget}"
                )

    # 8. Bridge caps: an unbounded bridge is just P0 in disguise.
    if aggregated.bridges:
        total = 0
        for bridge in aggregated.bridges:
            each = tokenizer.count(bridge.text)
            total += each
            if config.bridge_token_cap_each is not None and each > config.bridge_token_cap_each:
                result.errors.append(
                    f"bridge exceeds per-bridge token cap ({each} > {config.bridge_token_cap_each})"
                )
        if config.bridge_token_cap_total is not None and total > config.bridge_token_cap_total:
            result.errors.append(
                f"bridges exceed total token cap ({total} > {config.bridge_token_cap_total})"
            )

    if result.errors:
        # Fail closed: never hand back a render alongside errors. The adapter must not be able
        # to publish "the bytes preflight produced" from a run preflight rejected.
        result.rendered = None
    return result


def _body_resolver(
    snapshot_texts: dict[str, str], visible_views: dict[str, bytes], result: "PreflightResult"
) -> Callable[[dict], str]:
    """Resolve a span's exact bytes from frozen storage, by namespace.

    This replaces the caller-supplied ``source_text_for``. That callable sat on the publication
    path: the span was hashed against the snapshot and then whatever the callable returned was
    rendered, so "INJECTED BODY" passed with ok=True. Records a failure instead of raising, so
    a missing snapshot is a fail-closed result rather than a KeyError.
    """

    def resolve(span: dict) -> str:
        namespace = span.get("namespace")
        sid = (span.get("span_id") or span.get("visible_span_id") or "?")[:12]
        try:
            if namespace == "RAW_SOURCE":
                return snapshot_texts[span["content_hash"]][span["char_start"]:span["char_end"]]
            if namespace == "VISIBLE_MESSAGE":
                view = visible_views[span["visible_compressor_view_hash"]]
                return view[span["byte_start"]:span["byte_end"]].decode("utf-8")
        except (KeyError, UnicodeDecodeError) as e:
            result.errors.append(f"span {sid}: cannot resolve body from frozen bytes ({e!r})")
            return ""
        result.errors.append(f"span {sid}: unknown namespace {namespace!r}; cannot resolve body")
        return ""

    return resolve


def _provenance_errors(
    selection: ParsedSelection, aggregated: AggregatedEvidence
) -> list[str]:
    """An aggregator may drop. It may never rewrite an annotation or invent an element.

    Preflight compared only span *membership*, so `support/efficacy` could be published as
    `background/safety` and every downstream metric would attribute the rewrite to the model.
    Gaps and bridges could likewise be introduced after the fact -- unsourced output charged to
    the selector.
    """
    errors: list[str] = []
    by_span = {item.span_id: item for item in selection.items}

    for item in aggregated.items:
        chosen = by_span.get(item.span_id)
        if chosen is None:
            errors.append(f"published span {item.span_id[:12]} is not in the resolved selection")
            continue
        if item.role != chosen.role or tuple(item.facet_ids) != chosen.facet_ids:
            errors.append(
                f"published span {item.span_id[:12]} carries annotation "
                f"({item.role!r}, {tuple(item.facet_ids)!r}) but the selector chose "
                f"({chosen.role!r}, {chosen.facet_ids!r}); an aggregator may drop, never rewrite"
            )

    if selection.contract == "P1_ID":
        # P1_ID isolates the pointer mechanism; anything else in its output is a different arm.
        for item in aggregated.items:
            if item.role or item.facet_ids:
                errors.append(
                    f"P1_ID output published span {item.span_id[:12]} with a role/facet; the "
                    "contract has no way to express one, so it was added downstream"
                )
        if aggregated.gaps or aggregated.bridges:
            errors.append("P1_ID output published gaps or bridges, which the contract cannot emit")

    declared_gaps = set(selection.gaps)
    for gap in aggregated.gaps:
        if gap not in declared_gaps:
            errors.append(f"published gap on facet {gap.facet_id!r} was not in the selection")
    declared_bridges = set(selection.bridges)
    for bridge in aggregated.bridges:
        if bridge not in declared_bridges:
            errors.append(f"published bridge {bridge.text[:40]!r} was not in the selection")

    published_ids = {item.span_id for item in aggregated.items}
    expected_dropped = set(selection.selected_span_ids) - published_ids
    if set(aggregated.dropped_for_budget) != expected_dropped:
        errors.append(
            f"dropped_for_budget records {sorted(s[:12] for s in aggregated.dropped_for_budget)} "
            f"but {sorted(s[:12] for s in expected_dropped)} actually vanished; the work "
            "accounting and the published output disagree about what was cut"
        )
    return errors


# Public alias: the adapter computes this BEFORE the selector call and passes it back in.
candidate_view_sha = _candidate_view_sha
