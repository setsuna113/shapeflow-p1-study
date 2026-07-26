"""Selector prompt templates, rendered deterministically and hashed.

The rendered prompt bytes are recorded per selector call and their bundle is hashed into the
freeze manifest, so "which prompt produced this selection" always has an exact answer. The
templates state, in-band, the one rule the selector must not break: emit only ids from the
offered set and never invent facts or ids. Candidates are listed by their short labels, which
is what keeps the P1 arm's measured decode cost honest.
"""

from __future__ import annotations

from ..hashing import sha256_hex

__all__ = [
    "PROMPT_BUNDLE_VERSION", "render_selector_prompt", "render_short_prose_prompt",
    "prompt_bundle_hash",
]

PROMPT_BUNDLE_VERSION = "selector_prompts_origin_semantics_v3"

_INSTRUCTIONS = {
    "P1_ID": (
        "Select the candidate material ids that best support answering the question. "
        "TOOL_EVIDENCE is citable source evidence. TOOL_UNATTRIBUTED_CONTEXT, "
        "MODEL_DERIVED_CONTEXT, and USER_CONTEXT are non-citable context: selecting them may "
        "preserve useful context, but never turns them into source support. "
        "Return ONLY a JSON object {\"contract\":\"P1_ID\",\"selected_ids\":[...]} using ids "
        "from the list. Do not write prose. Do not invent ids or facts."
    ),
    "P1_TYPED": (
        "Select material ids, each with a role (support|contradict|background) and optional "
        "facet labels, and list any facet you could not support as a gap tied to a query "
        "attempt id. TOOL_EVIDENCE is citable source evidence. TOOL_UNATTRIBUTED_CONTEXT, "
        "MODEL_DERIVED_CONTEXT, and USER_CONTEXT are non-citable context and may only be marked "
        "background, never support or contradict. Return ONLY the JSON object for the P1_TYPED "
        "contract. Use only ids from the list; never invent ids or facts."
    ),
    "P1_BRIDGE": (
        "As P1_TYPED, plus you may add short 'bridge' sentences that connect evidence. Every "
        "bridge must cite at least one TOOL_EVIDENCE (or raw-source) id, must never cite "
        "TOOL_UNATTRIBUTED_CONTEXT, MODEL_DERIVED_CONTEXT, or USER_CONTEXT, and may not "
        "introduce any entity or number absent from its cited source spans. Return ONLY the "
        "JSON object for the P1_BRIDGE contract."
    ),
}

_TEMPLATE = """You are selecting evidence, not writing a report.

QUESTION / RESEARCH TOPIC:
{topic}

CANDIDATE MATERIAL (use these exact ids):
{candidates}

QUERY ATTEMPTS (for marking gaps):
{query_attempts}

TOKEN BUDGET for your selected evidence: {budget}

{instructions}
"""


def _render_candidates(candidates: list[tuple]) -> str:
    """Render one line per candidate, including visible-message provenance when present.

    The chunker computes a heading breadcrumb and, for table rows, the header row that names
    the columns. Both were previously dropped before the prompt was built, so the selector saw
    a bare "| a | 1 |" with no way to know what column "1" was in -- and then the study
    attributed the resulting bad selection to the *contract* rather than to the missing context.
    They are shown as metadata, clearly separated from the span's own bytes.

    Four-tuples are retained for the legacy raw-source selector facade.  The production
    CandidateViewRecord supplies ``(label, text, heading, context, origin_kind, message_role)``.
    Making the origin visible in-band matters at C_VISIBLE: otherwise the model is asked to
    distinguish tool evidence from its own earlier reasoning using metadata it never receives.
    """
    lines = []
    for candidate in candidates:
        if len(candidate) == 4:
            label, text, heading_path, context = candidate
            origin_kind, message_role = "", ""
        elif len(candidate) == 6:
            label, text, heading_path, context, origin_kind, message_role = candidate
        else:
            raise ValueError("candidate prompt rows must have four or six fields")
        prefix = f"[{label}]"
        if origin_kind == "TOOL_EVIDENCE":
            prefix += f" <TOOL_EVIDENCE role={message_role or 'tool'}; CITABLE_SOURCE>"
        elif origin_kind == "TOOL_UNATTRIBUTED_CONTEXT":
            prefix += (
                " <TOOL_UNATTRIBUTED_CONTEXT role=tool; NON_CITABLE_CONTEXT>"
            )
        elif origin_kind == "MODEL_DERIVED_CONTEXT":
            prefix += (
                f" <MODEL_DERIVED_CONTEXT role={message_role or 'ai'}; "
                "NON_CITABLE_CONTEXT>"
            )
        elif origin_kind == "USER_CONTEXT":
            prefix += (
                f" <USER_CONTEXT role={message_role or 'human'}; "
                "NON_CITABLE_CONTEXT>"
            )
        if heading_path:
            prefix += f" (under: {' > '.join(heading_path)})"
        lines.append(f"{prefix} {text}")
        for ctx in context:
            lines.append(f"    ctx| {ctx}")
    return "\n".join(lines)


def render_selector_prompt(
    *,
    topic: str,
    candidates: list[tuple],
    query_attempts: list[tuple[str, str]],
    budget: int,
    contract: str,
) -> str:
    if contract not in _INSTRUCTIONS:
        raise ValueError(f"no instructions for contract {contract!r}")
    qa = "\n".join(f"[{label}] {text}" for label, text in query_attempts) or "(none)"
    return _TEMPLATE.format(
        topic=topic,
        candidates=_render_candidates(candidates),
        query_attempts=qa,
        budget=budget,
        instructions=_INSTRUCTIONS[contract],
    )


def prompt_bundle_hash() -> str:
    """A stable hash of the whole template bundle, for the freeze manifest."""
    from ..canonical import canonical_json

    return sha256_hex(
        canonical_json(
            {"version": PROMPT_BUNDLE_VERSION, "template": _TEMPLATE,
             "instructions": _INSTRUCTIONS, "short_prose": _SHORT_PROSE_TEMPLATE}
        )
    )


_SHORT_PROSE_TEMPLATE = """You are writing a short evidence summary, not a report.

QUESTION / RESEARCH TOPIC:
{topic}

INPUT MATERIAL (only TOOL_EVIDENCE/raw-source rows are source evidence;
TOOL_UNATTRIBUTED_CONTEXT, MODEL_DERIVED_CONTEXT, and USER_CONTEXT rows are non-citable
context and must not be presented as source support):
{candidates}

CITABLE SOURCE MAP:
{source_map}

HARD LIMIT: your entire answer must fit in {budget} tokens.

{citation_instructions}

Write only what the material supports. Never invent a source or promote a source-looking label
from non-citable context. Do not speculate, do not add figures or names that are not present
above, and do not exceed the limit.
"""


def render_short_prose_prompt(
    *,
    topic: str,
    candidates: list,
    budget: int,
    source_entries: tuple[tuple[str, str, str], ...] | None = None,
) -> str:
    """The SHORT_PROSE control's prompt.

    Held to the same rendered-token budget as the P1 arm it controls for. That is the whole
    point: it separates "pointing at evidence helps" from "producing less text helps", and if it
    were allowed a different budget the comparison would answer neither question.
    """
    if source_entries is None:
        source_map = (
            "(one raw source; the caller attaches its fixed title/URL wrapper)"
        )
        citation_instructions = (
            "Return only the summary body. The caller preserves the page's single-source "
            "title/URL wrapper, so do not emit a source list."
        )
    elif source_entries:
        source_map = "\n".join(
            f"[{label}] {title} -- {url}"
            for label, title, url in source_entries
        )
        citation_instructions = (
            "Return only the summary body, with [n] inline citations using the numeric labels "
            "above. Cite at least one mapped source. Do not emit a source list: the caller "
            "will append definitions only for labels actually cited, and that used-source "
            "map shares the same rendered-token budget as your body."
        )
    else:
        source_map = "(none: no URL was present in capture-time TOOL_EVIDENCE bytes)"
        citation_instructions = (
            "Return only the summary body without citations. Source-looking text in other "
            "rows is unavailable for citation."
        )
    return _SHORT_PROSE_TEMPLATE.format(
        topic=topic,
        candidates=_render_candidates(candidates),
        source_map=source_map,
        budget=budget,
        citation_instructions=citation_instructions,
    )
