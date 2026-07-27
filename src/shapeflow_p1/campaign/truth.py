"""Building TruthPackets from the frozen world, under the evaluator identity.

DeepSeek proposes candidate atoms; the assembly rule decides what becomes truth, and it is
strict: an atom enters accepted truth only if it binds to at least one exact span of a frozen
page. A fact the model asserted without a span is dropped, a contradiction needs a span on each
side, and a gap binds to a real query attempt rather than to the model's opinion.

Two things this is careful about.

**Truth is derived from sources, never from an arm's output.** The candidate generator reads the
frozen source pool only. Deriving truth from what P0 produced would make P0 the gold standard,
and every arm would then be scored on how closely it reproduced P0 rather than on whether it was
right (plan §13.1).

**Nothing here is presented as human-verified.** Every packet is
``MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT`` with ``verifier_status: PENDING``, and that label
travels into every score computed against it. An unaudited machine packet supporting a
paper-grade claim is exactly the failure this study is built to avoid.
"""

from __future__ import annotations

import inspect
import json
import os
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

from ..canonical import canonical_json
from ..evaluation.judge_client import DeepSeekJudge, JudgeTruncated, JudgeUnavailable
from ..evaluation.truth_builder import CandidateAtom, assemble_truth_packet
from ..evidence.chunkers import (
    fixed_token_v1,
    markdown_structure_v1,
    paragraph_sentence_v1,
)
from ..evidence.model_tokenizer import load_frozen_tokenizer, tokenizer_sha256
from ..evidence.identity import build_evidence_span
from ..hashing import sha256_hex
from .acquire import load_frozen_pool
from .settings import Settings

__all__ = ["TRUTH_PROMPT_VERSION", "TruthResult", "build_truth_for_task", "truth_prompt_sha256"]

TRUTH_PROMPT_VERSION = "judge_truth_v2_full_world"
BIND_PROMPT_VERSION = "exact_span_entailment_v1"

_TRUTH_SYSTEM = """\
You extract verifiable atomic facts from source excerpts. You never add anything the excerpts do
not state, and you never answer from your own knowledge.

Every fact you return must be traceable to at least one excerpt id you were given. A fact you
cannot bind to an excerpt must be omitted, not guessed.

Return only JSON matching the requested schema.
"""

_TRUTH_PROMPT = """\
Research question:
{question}

Required facets:
{facets}

Source excerpts (id: text):
{excerpts}

Frozen query attempts (id | status | query):
{query_attempts}

For each atomic fact the excerpts establish, return:
- "atom_id": a short unique slug
- "facet_id": the required facet it answers, or "other"
- "text": the fact, stated in one sentence, using the excerpts' own numbers and names
- "critical": true only if a report that got this wrong would mislead a reader materially
- "supporting_span_ids": the excerpt ids that state it (at least one, never invented)

Also return "contradictions": pairs of atom_ids that the excerpts genuinely disagree about.
Return "negative_evidence" only when one of the returned, span-grounded atoms explicitly
establishes absence. Name that atom and the frozen query attempt that looked. Return
an empty "known_gaps" list: gap membership is derived mechanically from the frozen
acquisition status, never from this model. Never invent an atom or query-attempt id.

Return JSON:
{{"atoms": [...], "contradictions": [["atom_id_a", "atom_id_b"]],
  "negative_evidence": [{{"atom_id": "...", "facet_id": "...",
                           "query_attempt_id": "..."}}],
  "known_gaps": []}}
"""

_TRUTH_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["atoms"],
    "properties": {
        "atoms": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["atom_id", "facet_id", "text", "critical", "supporting_span_ids"],
            "properties": {
                "atom_id": {"type": "string", "minLength": 1},
                "facet_id": {"type": "string"},
                "text": {"type": "string", "minLength": 5},
                "critical": {"type": "boolean"},
                "supporting_span_ids": {"type": "array", "items": {"type": "string"}},
            },
        }},
        "contradictions": {"type": "array", "items": {
            "type": "array", "minItems": 2, "maxItems": 2, "items": {"type": "string"},
        }},
        "negative_evidence": {"type": "array", "items": {
            "type": "object", "additionalProperties": False,
            "required": ["atom_id", "facet_id", "query_attempt_id"],
            "properties": {
                "atom_id": {"type": "string", "minLength": 1},
                "facet_id": {"type": "string"},
                "query_attempt_id": {"type": "string"},
            },
        }},
        "known_gaps": {"type": "array", "items": {"type": "string"}},
    },
}

_BIND_SYSTEM = """\
You verify one proposed atomic fact against one exact frozen source span. Use only the supplied
span. Return entail only when the span itself establishes the complete fact, including names,
quantities, dates and qualifications. Partial overlap is not entailment.
"""

_BIND_PROMPT = """\
Proposed fact:
{atom}

Exact frozen span:
{span}

Return JSON: {{"relation": "entail" | "not_entail" | "uncertain"}}
"""

_BIND_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["relation"],
    "properties": {"relation": {"enum": ["entail", "not_entail", "uncertain"]}},
}

_CONFLICT_SYSTEM = """\
You identify genuine contradictions among source-grounded atomic facts. Two facts contradict
only when they make incompatible claims about the same entity, measure, time and scope.
Differences in date, population or definition are not contradictions.
"""

_CONFLICT_PROMPT = """\
Facet: {facet}

Grounded facts (id: fact):
{facts}

Return JSON: {{"contradictions": [["atom_id_a", "atom_id_b"]]}}
"""

_CONFLICT_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["contradictions"],
    "properties": {
        "contradictions": {
            "type": "array",
            "items": {
                "type": "array", "minItems": 2, "maxItems": 2,
                "items": {"type": "string"},
            },
        },
    },
}

SemanticVerifier = Callable[[str, str, str], object]


def truth_prompt_sha256() -> str:
    return sha256_hex(canonical_json({
        "version": TRUTH_PROMPT_VERSION,
        "system": _TRUTH_SYSTEM,
        "user": _TRUTH_PROMPT,
        "truth_schema": _TRUTH_SCHEMA,
        "binding": {
            "version": BIND_PROMPT_VERSION,
            "system": _BIND_SYSTEM,
            "user": _BIND_PROMPT,
            "schema": _BIND_SCHEMA,
        },
        "cross_batch_contradiction": {
            "system": _CONFLICT_SYSTEM,
            "user": _CONFLICT_PROMPT,
            "schema": _CONFLICT_SCHEMA,
        },
    }))


@dataclass
class TruthResult:
    task_id: str
    packet: dict
    atom_texts: dict
    rejected: list = field(default_factory=list)
    span_count: int = 0
    path: Path | None = None


def _write_truth_artifact(path: Path, artifact: dict) -> None:
    """Create one immutable truth artifact, or verify an identical existing artifact."""
    path = Path(path)
    expected = str(artifact.get("content_sha256") or "")
    actual = sha256_hex(canonical_json({
        key: value for key, value in artifact.items() if key != "content_sha256"
    }))
    if not expected or expected != actual:
        raise ValueError("truth artifact content_sha256 does not match its content")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o440)
        return
    except FileExistsError:
        pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"existing truth artifact {path} is unreadable: {exc}") from exc
    recorded = str(existing.get("content_sha256") or "")
    existing_actual = sha256_hex(canonical_json({
        key: value for key, value in existing.items() if key != "content_sha256"
    }))
    if recorded != existing_actual:
        raise ValueError(
            f"{path} existing truth artifact was edited: records {recorded!r}, "
            f"hashes to {existing_actual!r}"
        )
    if recorded != expected or canonical_json(existing) != canonical_json(artifact):
        raise ValueError(
            f"{path} already contains a different truth artifact; corrected truth "
            "requires a new version, never an in-place overwrite"
        )


def _checker():
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(_TRUTH_SCHEMA)
    return validator.validate


def _excerpt_spans(settings: Settings, task_id: str) -> list[dict]:
    """Chunk the task's frozen pages into addressable excerpts.

    Truth is bound to these ids, so an atom's provenance is a span of a page that was actually
    frozen -- not a page the model remembers.
    """
    pool, store = load_frozen_pool(settings, task_id)
    tokenizer = load_frozen_tokenizer(settings)
    spans: list[dict] = []
    for occurrence in pool.vendor_visible:
        if not occurrence.content_hash:
            continue
        snapshot = pool.snapshots.get(occurrence.content_hash)
        if snapshot is None:
            continue
        text = store.read_text(snapshot)
        for chunk in markdown_structure_v1(text, tokenizer=tokenizer, max_tokens=320):
            span = build_evidence_span(
                chunk, text, content_hash=occurrence.content_hash,
                source_occurrence_ids=[occurrence.occurrence_id],
                chunker_version="markdown_structure_v1",
            )
            # ``build_evidence_span`` intentionally stores identity/provenance, not duplicated
            # payload bytes. The evaluator prompt still needs the exact addressed bytes.
            span["text"] = text[chunk.char_start:chunk.char_end]
            spans.append(span)
    return spans


def _h_candidate_spans(
    settings: Settings, task_id: str, *, tokenizer=None
) -> dict[str, list[dict]]:
    """Rebuild each runnable H chunker over the exact vendor-truncated treatment bytes."""
    from ..strategies.factory import load_registry

    registry = load_registry(settings.repo / "configs")
    chunkers = sorted({
        spec.chunker for spec in registry.values()
        if spec.node == "WEBPAGE_P1" and spec.runnable
    })
    pool, store = load_frozen_pool(settings, task_id)
    tokenizer = tokenizer or load_frozen_tokenizer(settings)
    limit = int(settings.get("week1", "odr", "max_content_length"))
    out: dict[str, list[dict]] = {name: [] for name in chunkers}
    for occurrence in pool.vendor_visible:
        if not occurrence.content_hash:
            continue
        snapshot = pool.snapshots.get(occurrence.content_hash)
        if snapshot is None:
            continue
        text = store.read_text(snapshot)[:limit]
        content_hash = sha256_hex(text.encode("utf-8"))
        for chunker in chunkers:
            if chunker == "fixed_token_v1":
                chunks = fixed_token_v1(
                    text, tokenizer=tokenizer, window=320, overlap=0)
            elif chunker == "paragraph_sentence_v1":
                chunks = paragraph_sentence_v1(
                    text, tokenizer=tokenizer, max_tokens=320)
            elif chunker == "markdown_structure_v1":
                chunks = markdown_structure_v1(
                    text, tokenizer=tokenizer, max_tokens=320)
            else:
                # Registry validation should make this impossible. Refusing is safer than an
                # empty support set that would make an implemented arm look unmeasurable.
                raise ValueError(f"truth support index has no builder for H chunker {chunker!r}")
            for chunk in chunks:
                span = build_evidence_span(
                    chunk, text, content_hash=content_hash,
                    source_occurrence_ids=[occurrence.occurrence_id],
                    chunker_version=chunker,
                )
                out[chunker].append({
                    **span,
                    "_text": text[chunk.char_start:chunk.char_end],
                })
    return out


async def _atom_support_index(
    settings: Settings,
    task_id: str,
    *,
    candidates: Sequence[CandidateAtom],
    truth_spans: Sequence[dict],
    verifier: SemanticVerifier,
) -> dict:
    """Bind grounded atoms to every runnable H chunker's exact treatment span ids.

    Candidate spans are prefiltered by source occurrence and character overlap with an already
    verified truth span. The semantic verifier then confirms the complete fact survives that
    chunk boundary. This avoids an infeasible all-atoms x all-spans judge cross product while
    still refusing overlap as proof of entailment.
    """
    truth_by_id = {str(span["span_id"]): span for span in truth_spans}
    tokenizer = load_frozen_tokenizer(settings)
    candidate_spans = _h_candidate_spans(
        settings, task_id, tokenizer=tokenizer
    )
    supports: dict[str, dict[str, list[str]]] = {}
    candidate_span_occurrences: dict[str, dict[str, list[str]]] = {}
    cache: dict[tuple[str, str], bool] = {}
    vendor_limit = int(settings.get("week1", "odr", "max_content_length"))
    # Treatment-independent H denominator: which frozen source occurrences contain each
    # grounded atom within the exact vendor-visible prefix. This survives a poor candidate
    # chunker splitting the fact across chunks; such a split lowers candidate coverage instead
    # of making the fact disappear from its own denominator.
    prechunk_occurrences: dict[str, list[str]] = {}
    for atom in candidates:
        occurrences: set[str] = set()
        for truth_span_id in atom.supporting_span_ids:
            truth_span = truth_by_id.get(truth_span_id)
            if truth_span is None or int(truth_span["char_end"]) > vendor_limit:
                continue
            occurrences.update(map(
                str, truth_span.get("source_occurrence_ids") or ()))
        prechunk_occurrences[atom.atom_id] = sorted(occurrences)
    for chunker, spans in candidate_spans.items():
        candidate_span_occurrences[chunker] = {
            str(span["span_id"]): sorted(map(
                str, span.get("source_occurrence_ids") or ()))
            for span in spans
        }
        by_occurrence: dict[str, list[dict]] = {}
        for span in spans:
            for occurrence_id in span.get("source_occurrence_ids") or ():
                by_occurrence.setdefault(str(occurrence_id), []).append(span)
        per_atom: dict[str, list[str]] = {}
        for atom in candidates:
            candidates_to_check: dict[str, dict] = {}
            for truth_span_id in atom.supporting_span_ids:
                truth_span = truth_by_id.get(truth_span_id)
                if truth_span is None:
                    continue
                for occurrence_id in truth_span.get("source_occurrence_ids") or ():
                    for candidate_span in by_occurrence.get(str(occurrence_id), ()):
                        if (
                            int(candidate_span["char_start"]) < int(truth_span["char_end"])
                            and int(truth_span["char_start"]) < int(candidate_span["char_end"])
                        ):
                            candidates_to_check[str(candidate_span["span_id"])] = candidate_span
            bound: list[str] = []
            for span_id, span in sorted(candidates_to_check.items()):
                key = (atom.text, span_id)
                if key not in cache:
                    cache[key] = await _verified_relation(
                        verifier, atom.text, str(span["_text"]), span_id)
                if cache[key]:
                    bound.append(span_id)
            per_atom[atom.atom_id] = bound
        supports[chunker] = per_atom
    body = {
        "version": "h_atom_support_index_v2",
        "chunk_max_tokens": 320,
        "tokenizer_sha256": tokenizer_sha256(tokenizer),
        "semantic_checks": len(cache),
        "prechunk_atom_occurrence_ids": prechunk_occurrences,
        "candidate_span_occurrence_ids": candidate_span_occurrences,
        "chunkers": supports,
    }
    body["content_sha256"] = sha256_hex(canonical_json(body))
    return body


def _excerpt_batches(spans: Sequence[dict], *, max_prompt_chars: int) -> list[list[dict]]:
    """Pack every complete span exactly once without slicing or dropping it.

    The old 60-span/400-character caps silently removed the long tail from the answer key.
    Batching controls judge context size without changing the truth world: a span too large
    for the target still forms a one-span batch in full.
    """
    if max_prompt_chars <= 0:
        raise ValueError("max_prompt_chars must be positive")
    batches: list[list[dict]] = []
    current: list[dict] = []
    chars = 0
    for span in spans:
        size = len(str(span["span_id"])) + len(str(span["text"])) + 4
        if current and chars + size > max_prompt_chars:
            batches.append(current)
            current = []
            chars = 0
        current.append(span)
        chars += size
    if current:
        batches.append(current)
    return batches


def _query_attempts(settings: Settings, task_id: str) -> list[dict]:
    """Read the evaluator-visible frozen acquisition attempts, including failures."""
    from ..acquire.manifest import load_task_manifest

    path = settings.path("acquisition") / f"{task_id}.json"
    if not path.exists():
        return []
    body = load_task_manifest(path)
    return [
        {
            "query_attempt_id": str(q.get("query_snapshot_id") or ""),
            "query": str(q.get("query_text") or ""),
            "status": str(q.get("status") or ""),
        }
        for q in (body.get("queries") or [])
        if q.get("query_snapshot_id")
    ]


def _format_query_attempts(attempts: Sequence[dict]) -> str:
    return "\n".join(
        f"{q['query_attempt_id']} | {q['status']} | {q['query']}" for q in attempts
    ) or "(no query attempts were recorded)"


#: How many blocks an over-long facet is cut into when its contradiction answer will not fit.
#: Four, not two. Contradiction is a property of a *pair*, so a split has to keep every pair
#: co-resident in at least one call: with two halves the only call that holds a cross-half pair
#: is the union, which is the request that just truncated. With four blocks, the six unordered
#: block pairs each carry half the facts and together cover every pair of facts exactly.
_CONFLICT_BLOCKS = 4


async def _facet_contradictions(
    judge,
    task_id: str,
    facet: str,
    facts: Sequence["CandidateAtom"],
    *,
    responses: list,
) -> tuple[set[tuple[str, str]], int]:
    """Contradiction pairs within one facet, split by blocked pairs when the answer overflows.

    The extraction pass above splits its batches when the output cap is hit, and that is safe
    there because an atom belongs to one span. Contradiction does not decompose that way: naively
    halving a facet's fact list silently drops every pair that straddles the cut, and a truth
    packet that quietly lost contradictions makes an arm that missed them look correct. So this
    splits into blocks and re-asks over each *pair of blocks*, which bounds both the prompt and
    the answer while leaving no pair unexamined.

    The judge's sampling envelope is frozen protocol and is deliberately not widened for this
    pass: raising ``max_tokens`` for one call would make this stage's judge a different judge
    from the one that authored everything else.
    """
    prompt = _CONFLICT_PROMPT.format(
        facet=facet,
        facts="\n".join(f"{fact.atom_id}: {fact.text}" for fact in facts),
    )
    try:
        response = await judge.judge(
            _CONFLICT_SYSTEM, prompt, validate=_checker_for(_CONFLICT_SCHEMA)
        )
    except JudgeTruncated as e:
        if len(facts) <= _CONFLICT_BLOCKS:
            # Below this the blocks stop shrinking and the recursion would not terminate.
            raise JudgeUnavailable(
                f"truth for {task_id} unavailable: facet {facet!r} still overflows the output "
                f"cap at {len(facts)} facts ({e}); it cannot be split without dropping pairs, "
                "and a packet missing contradictions makes an arm that missed them look right"
            ) from e
        blocks = [list(facts[index::_CONFLICT_BLOCKS]) for index in range(_CONFLICT_BLOCKS)]
        pairs: set[tuple[str, str]] = set()
        used = 0
        for i in range(_CONFLICT_BLOCKS):
            for j in range(i + 1, _CONFLICT_BLOCKS):
                found, sub_used = await _facet_contradictions(
                    judge, task_id, facet, blocks[i] + blocks[j], responses=responses
                )
                pairs |= found
                used += sub_used
        return pairs, used
    except JudgeUnavailable as e:
        raise JudgeUnavailable(f"truth for {task_id} unavailable: {e}") from e

    responses.append(response)
    known_atom_ids = {fact.atom_id for fact in facts}
    pairs = set()
    for pair in response.data.get("contradictions") or ():
        canonical_pair = tuple(sorted(map(str, pair))) if len(pair) == 2 else ()
        if (
            len(canonical_pair) == 2
            and canonical_pair[0] != canonical_pair[1]
            and set(canonical_pair) <= known_atom_ids
        ):
            pairs.add(canonical_pair)
    return pairs, 1


def _stable_atom_id(proposed: str, facet_id: str, text: str) -> str:
    # The judge's slug is presentation, not identity: two batches may name the same fact
    # differently. Identity comes from the normalized fact and facet.
    del proposed
    return "atom-" + sha256_hex(canonical_json({"facet": facet_id, "text": text}))[:20]


async def _verified_relation(
    verifier: SemanticVerifier,
    atom_text: str,
    span_text: str,
    span_id: str,
) -> bool:
    result = verifier(atom_text, span_text, span_id)
    if inspect.isawaitable(result):
        result = await result
    return result is True or str(result).lower() in {
        "entail", "entailed", "support", "supported",
    }


def _judge_verifier(
    judge: DeepSeekJudge, *, response_sink: list | None = None
) -> SemanticVerifier:
    async def verify(atom_text: str, span_text: str, _span_id: str) -> str:
        response = await judge.judge(
            _BIND_SYSTEM,
            _BIND_PROMPT.format(atom=atom_text, span=span_text),
            validate=lambda body: _checker_for(_BIND_SCHEMA)(body),
        )
        if response_sink is not None:
            response_sink.append(response)
        return str(response.data.get("relation") or "uncertain")

    return verify


def _checker_for(schema: dict):
    from jsonschema import Draft202012Validator

    return Draft202012Validator(schema).validate


async def build_truth_for_task(
    settings: Settings,
    *,
    judge: DeepSeekJudge,
    task_id: str,
    question: str,
    required_facets: Sequence[str],
    max_spans: int | None = None,
    max_prompt_chars: int = 80_000,
    semantic_verifier: SemanticVerifier | None = None,
) -> TruthResult:
    """Propose, bind and assemble one task's TruthPacket. Write-once under the evaluator tree."""
    if max_spans is not None:
        raise ValueError(
            "max_spans truncates the frozen truth world and is no longer supported; "
            "use max_prompt_chars to batch all spans instead"
        )
    spans = _excerpt_spans(settings, task_id)
    if not spans:
        raise ValueError(
            f"truth for {task_id} has no frozen source spans; refusing to create an empty "
            "answer key"
        )
    span_text = {s["span_id"]: s["text"] for s in spans}
    attempts = _query_attempts(settings, task_id)
    query_attempt_ids = {q["query_attempt_id"] for q in attempts}
    query_attempt_text = _format_query_attempts(attempts)
    binding_responses: list = []
    verifier = semantic_verifier or _judge_verifier(
        judge, response_sink=binding_responses)

    # key=(facet,text) deduplicates the same fact proposed from overlapping chunks/batches.
    accumulated: dict[tuple[str, str], dict] = {}
    pairs: set[tuple[str, str]] = set()
    locally_proposed_pairs = 0
    negative_evidence: list[dict] = []
    known_gaps: list[dict] = [{
        "query_attempt_id": q["query_attempt_id"],
        "query_text": q["query"],
        "status": q["status"],
    } for q in attempts
        if q["status"] in {"FAILED", "TIMEOUT", "BLOCKED_BUDGET"}
    ]
    model_gap_suggestions_ignored = 0
    responses = []

    # A stack, not a list, so a batch whose *answer* does not fit can be split and its halves
    # pushed back. Batching is sized by prompt characters, which bounds the input but says
    # nothing about the output: a span-dense batch can be well under the prompt budget and
    # still ask for more atoms than the completion cap can hold. Retrying the identical request
    # cannot fix that -- the same spans ask for the same answer -- so every attempt truncated
    # and the whole task was abandoned after paying for four of them.
    pending = list(reversed(_excerpt_batches(spans, max_prompt_chars=max_prompt_chars)))
    split_batches = 0
    while pending:
        batch = pending.pop()
        excerpts = "\n".join(f"{s['span_id']}: {s['text']}" for s in batch)
        try:
            response = await judge.judge(
                _TRUTH_SYSTEM,
                _TRUTH_PROMPT.format(
                    question=question,
                    facets="\n".join(f"- {f}" for f in required_facets) or "- (none stated)",
                    excerpts=excerpts or "(no excerpts)",
                    query_attempts=query_attempt_text,
                ),
                validate=_checker(),
            )
        except JudgeTruncated as e:
            if len(batch) < 2:
                # One span whose atoms do not fit is a real limit, not a batching mistake.
                # Splitting further is impossible and dropping it would silently shorten the
                # answer key, so the task stops here.
                raise JudgeUnavailable(
                    f"truth for {task_id} unavailable: a single span still overflows the "
                    f"output cap ({e}); it cannot be split further and omitting it would "
                    "make an omission outside the surviving spans look correct"
                ) from e
            middle = len(batch) // 2
            pending.append(batch[middle:])
            pending.append(batch[:middle])
            split_batches += 1
            continue
        except JudgeUnavailable as e:
            # No packet rather than an incomplete one: an answer key that silently omitted one
            # batch would make omissions outside the surviving batches look correct.
            raise JudgeUnavailable(f"truth for {task_id} unavailable: {e}") from e
        responses.append(response)
        local_ids: dict[str, str] = {}
        batch_ids = {s["span_id"] for s in batch}
        for atom in response.data["atoms"]:
            text = str(atom["text"])
            facet_id = str(atom["facet_id"])
            atom_id = _stable_atom_id(str(atom["atom_id"]), facet_id, text)
            local_ids[str(atom["atom_id"])] = atom_id
            key = (facet_id, text)
            entry = accumulated.setdefault(key, {
                "atom_id": atom_id, "facet_id": facet_id, "text": text,
                "critical": False, "supporting_span_ids": set(),
            })
            entry["critical"] = bool(entry["critical"] or atom["critical"])
            entry["supporting_span_ids"].update(
                s for s in atom["supporting_span_ids"] if s in batch_ids
            )
        # Batch extraction may propose conflicts, but those proposals are not adjudications.
        # Only the dedicated, all-grounded-facts conflict pass below can enter the packet.
        locally_proposed_pairs += len(response.data.get("contradictions") or ())
        for item in response.data.get("negative_evidence") or ():
            local_atom_id = str(item.get("atom_id") or "")
            stable_atom_id = local_ids.get(local_atom_id)
            if stable_atom_id is None:
                continue
            negative_evidence.append({
                "atom_id": stable_atom_id,
                "facet_id": str(item.get("facet_id") or ""),
                "query_attempt_id": str(item.get("query_attempt_id") or ""),
            })
        # A real SUCCESS id suggested by the model still cannot become an unresolved gap.
        model_gap_suggestions_ignored += len(response.data.get("known_gaps") or ())

    # How many responses came from extraction, fixed before the conflict pass appends
    # its own. Adaptive splitting means this is no longer the initial batch count.
    extraction_batches = len(responses)

    candidates: list[CandidateAtom] = []
    for entry in accumulated.values():
        verified_ids: list[str] = []
        for span_id in sorted(entry["supporting_span_ids"]):
            if await _verified_relation(verifier, entry["text"], span_text[span_id], span_id):
                verified_ids.append(span_id)
        candidates.append(CandidateAtom(
            atom_id=entry["atom_id"], facet_id=entry["facet_id"], weight=1.0,
            critical=entry["critical"], supporting_span_ids=tuple(verified_ids),
            text=entry["text"],
        ))
    exact_span_binding_response_count = len(binding_responses)

    # Extraction is batched so no source span is dropped. Reconcile conflicts over the
    # resulting grounded facts by facet, otherwise facts on opposite sides of a batch boundary
    # could never form a contradiction pair.
    verified_by_facet: dict[str, list[CandidateAtom]] = {}
    for candidate in candidates:
        if candidate.supporting_span_ids:
            verified_by_facet.setdefault(candidate.facet_id, []).append(candidate)
    conflict_blocks = 0
    for facet, facts in sorted(verified_by_facet.items()):
        if len(facts) < 2:
            continue
        facet_pairs, blocks = await _facet_contradictions(
            judge, task_id, facet, facts, responses=responses
        )
        conflict_blocks += blocks
        pairs |= facet_pairs

    support_index = await _atom_support_index(
        settings, task_id,
        candidates=[c for c in candidates if c.supporting_span_ids],
        truth_spans=spans,
        verifier=verifier,
    )
    cross_chunker_binding_response_count = (
        len(binding_responses) - exact_span_binding_response_count
    )

    packet, rejected = assemble_truth_packet(
        task_id, candidates, required_facets=list(required_facets),
        contradiction_pairs=sorted(pairs), span_texts=span_text,
        negative_evidence=negative_evidence, known_gaps=known_gaps,
        known_query_attempt_ids=query_attempt_ids,
    )
    accepted_ids = {a["atom_id"] for a in packet["atomic_evidence"]}
    atom_texts = {c.atom_id: c.text for c in candidates if c.atom_id in accepted_ids}
    response_records = [
        *[
            (
                "truth_extraction"
                if index < extraction_batches else "conflict_reconciliation",
                response,
            )
            for index, response in enumerate(responses)
        ],
        *[
            (
                "exact_span_binding"
                if index < exact_span_binding_response_count else "cross_chunker_binding",
                response,
            )
            for index, response in enumerate(binding_responses)
        ],
    ]
    all_responses = [response for _stage, response in response_records]
    returned_models = sorted({r.returned_model for r in all_responses if r.returned_model})
    fingerprints = sorted({
        r.system_fingerprint for r in all_responses if r.system_fingerprint
    })
    injected_identity = None
    if semantic_verifier is not None:
        injected_identity = {
            "module": str(getattr(semantic_verifier, "__module__", "")),
            "qualname": str(getattr(
                semantic_verifier, "__qualname__",
                getattr(semantic_verifier, "__name__", type(semantic_verifier).__name__),
            )),
        }
    provenance = {
        "prompt_version": TRUTH_PROMPT_VERSION,
        "prompt_sha256": truth_prompt_sha256(),
        "requested_model": all_responses[0].requested_model if all_responses else "",
        "returned_model": ",".join(returned_models),
        "system_fingerprint": ",".join(fingerprints),
        "span_count": len(spans),
        # Batches actually judged, which exceeds the initial packing when a batch had to be
        # split because its answer overflowed the completion cap. Recorded with the split count
        # so a packet built from adapted batching is distinguishable from one that was not.
        "span_batches": len(responses),
        "span_batches_split_for_output_cap": split_batches,
        # Contradiction calls actually made. It exceeds the facet count exactly when a facet's
        # answer overflowed and had to be re-asked as blocked pairs, so a packet whose
        # contradictions came from a split is distinguishable from one whose did not.
        "conflict_blocks": conflict_blocks,
        "conflict_facets": sum(
            1 for facts in verified_by_facet.values() if len(facts) >= 2
        ),
        "judge_calls_total": len(all_responses),
        "judge_calls_by_stage": {
            "truth_extraction": extraction_batches,
            "conflict_reconciliation": len(responses) - extraction_batches,
            "exact_span_binding": exact_span_binding_response_count,
            "cross_chunker_binding": cross_chunker_binding_response_count,
        },
        "judge_attempts": [
            {
                "stage": stage,
                "request_id": str(response.request_id or ""),
                "requested_model": str(response.requested_model or ""),
                "returned_model": str(response.returned_model or ""),
                "system_fingerprint": str(response.system_fingerprint or ""),
                "usage": dict(response.usage or {}),
                "attempts": [
                    attempt.content() for attempt in (response.attempts or ())
                ],
            }
            for stage, response in response_records
        ],
        "extractor_contradiction_proposals_ignored": locally_proposed_pairs,
        "model_gap_suggestions_ignored": model_gap_suggestions_ignored,
        "span_truncation": False,
        "max_prompt_chars_per_batch": max_prompt_chars,
        "truth_chunker": "markdown_structure_v1",
        "truth_chunk_max_tokens": 320,
        "tokenizer_sha256": support_index["tokenizer_sha256"],
        "h_atom_support_index_sha256": support_index["content_sha256"],
        "semantic_binding_verifier": (
            "injected" if semantic_verifier is not None else "judge_exact_span_v1"),
        "semantic_binding_verifier_identity": injected_identity,
        "semantic_binding_audit_status": (
            "INJECTED_REQUIRES_EXPLICIT_AUDIT"
            if semantic_verifier is not None
            else "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT"
        ),
        "h_atom_support_index_audit_status":
            "MACHINE_CANDIDATE_PENDING_HUMAN_AUDIT",
        "claim_scope": settings.claim_scope,
        "config_sha256s": dict(sorted(settings.shas.items())),
        "task_question_facets_sha256": sha256_hex(canonical_json({
            "task_id": task_id,
            "question": question,
            "required_facets": list(required_facets),
        })),
    }
    runner_pool = json.loads(
        (settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json")
        .read_text(encoding="utf-8")
    )
    provenance["source_pool_sha256"] = str(runner_pool.get("pool_sha256") or "")
    provenance["query_attempts_sha256"] = sha256_hex(canonical_json(attempts))
    provenance["acquisition_digest"] = ""
    acquisition_path = settings.path("acquisition") / f"{task_id}.json"
    if acquisition_path.exists():
        from ..acquire.manifest import load_task_manifest

        provenance["acquisition_digest"] = str(
            load_task_manifest(acquisition_path).get("acquisition_digest") or "")
    provenance["truth_source_binding_sha256"] = sha256_hex(canonical_json({
        "task_question_facets_sha256": provenance["task_question_facets_sha256"],
        "source_pool_sha256": provenance["source_pool_sha256"],
        "acquisition_digest": provenance["acquisition_digest"],
        "query_attempts_sha256": provenance["query_attempts_sha256"],
    }))
    directory = settings.path("truth_packets")
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{task_id}.json"
    artifact = {
        "packet": packet,
        "atom_texts": atom_texts,
        "rejected": [{"atom_id": r.atom_id, "reason": r.reason} for r in rejected],
        "provenance": provenance,
        "atom_support_index": support_index,
    }
    artifact["content_sha256"] = sha256_hex(canonical_json(artifact))
    _write_truth_artifact(path, artifact)
    return TruthResult(task_id=task_id, packet=packet, atom_texts=atom_texts,
                       rejected=list(rejected), span_count=len(spans), path=path)
