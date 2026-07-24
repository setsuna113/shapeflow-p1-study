"""Selector prompt templates, rendered deterministically and hashed.

The rendered prompt bytes are recorded per selector call and their bundle is hashed into the
freeze manifest, so "which prompt produced this selection" always has an exact answer. The
templates state, in-band, the one rule the selector must not break: emit only ids from the
offered set and never invent facts or ids. Candidates are listed by their short labels, which
is what keeps the P1 arm's measured decode cost honest.
"""

from __future__ import annotations

from ..hashing import sha256_hex

__all__ = ["PROMPT_BUNDLE_VERSION", "render_selector_prompt", "prompt_bundle_hash"]

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


def _render_candidates(candidates: list[tuple[str, str]]) -> str:
    # candidates: list of (label, exact_text). Text is shown so the model can judge relevance.
    return "\n".join(f"[{label}] {text}" for label, text in candidates)


def render_selector_prompt(
    *,
    topic: str,
    candidates: list[tuple[str, str]],
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
            {"version": PROMPT_BUNDLE_VERSION, "template": _TEMPLATE, "instructions": _INSTRUCTIONS}
        )
    )
