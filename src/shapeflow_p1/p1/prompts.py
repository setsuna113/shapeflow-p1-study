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

PROMPT_BUNDLE_VERSION = "selector_prompts_v1"

_INSTRUCTIONS = {
    "P1_ID": (
        "Select the candidate evidence ids that best support answering the question. "
        "Return ONLY a JSON object {\"contract\":\"P1_ID\",\"selected_ids\":[...]} using ids "
        "from the list. Do not write prose. Do not invent ids or facts."
    ),
    "P1_TYPED": (
        "Select evidence ids, each with a role (support|contradict|background) and optional "
        "facet labels, and list any facet you could not support as a gap tied to a query "
        "attempt id. Return ONLY the JSON object for the P1_TYPED contract. Use only ids from "
        "the list; never invent ids or facts."
    ),
    "P1_BRIDGE": (
        "As P1_TYPED, plus you may add short 'bridge' sentences that connect evidence. Every "
        "bridge must cite at least one evidence id and may not introduce any entity or number "
        "absent from its cited spans. Return ONLY the JSON object for the P1_BRIDGE contract."
    ),
}

_TEMPLATE = """You are selecting evidence, not writing a report.

QUESTION / RESEARCH TOPIC:
{topic}

CANDIDATE EVIDENCE (use these exact ids):
{candidates}

QUERY ATTEMPTS (for marking gaps):
{query_attempts}

TOKEN BUDGET for your selected evidence: {budget}

{instructions}
"""


def _render_candidates(candidates: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]]) -> str:
    """Render one line per candidate: ``(label, exact_text, heading_path, context)``.

    The chunker computes a heading breadcrumb and, for table rows, the header row that names
    the columns. Both were previously dropped before the prompt was built, so the selector saw
    a bare "| a | 1 |" with no way to know what column "1" was in -- and then the study
    attributed the resulting bad selection to the *contract* rather than to the missing context.
    They are shown as metadata, clearly separated from the span's own bytes.
    """
    lines = []
    for label, text, heading_path, context in candidates:
        prefix = f"[{label}]"
        if heading_path:
            prefix += f" (under: {' > '.join(heading_path)})"
        lines.append(f"{prefix} {text}")
        for ctx in context:
            lines.append(f"    ctx| {ctx}")
    return "\n".join(lines)


def render_selector_prompt(
    *,
    topic: str,
    candidates: list[tuple[str, str, tuple[str, ...], tuple[str, ...]]],
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

SOURCE MATERIAL:
{candidates}

HARD LIMIT: your entire answer must fit in {budget} tokens.

Write only what the material supports. Do not speculate, do not add figures or names that are
not present above, and do not exceed the limit.
"""


def render_short_prose_prompt(
    *, topic: str, candidates: list, budget: int
) -> str:
    """The SHORT_PROSE control's prompt.

    Held to the same rendered-token budget as the P1 arm it controls for. That is the whole
    point: it separates "pointing at evidence helps" from "producing less text helps", and if it
    were allowed a different budget the comparison would answer neither question.
    """
    return _SHORT_PROSE_TEMPLATE.format(
        topic=topic, candidates=_render_candidates(candidates), budget=budget
    )
