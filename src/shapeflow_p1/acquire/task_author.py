"""Authoring the corpus with DeepSeek, under a frozen prompt, before any outcome exists.

Three properties this module is built to guarantee, each defending against a way a corpus can
be quietly wrong.

**The author is never the subject.** DeepSeek writes the tasks; the Qwen model under test never
does. A corpus written by the model being measured is selected for what that model finds easy,
and every arm inherits the bias while the comparison silently absorbs it. The seal in
:func:`~shapeflow_p1.acquire.task_registry.build_registry` refuses if the two ever coincide.

**Coverage is by construction, not by luck.** The stratum profile of every task is computed
here from ``configs/task_source.yaml`` and handed to the model as a requirement. Asking for
"a varied set" and hoping produces a corpus whose gaps are invisible until the eligibility
envelope cannot be estimated -- and by then the Tavily budget is spent. The audit then
re-verifies coverage independently, so a model that ignored the instruction is caught rather
than trusted.

**Everything about the authoring is frozen.** Prompt bytes, requested and returned model,
sampling, seed and the response hash all travel in the
:class:`~shapeflow_p1.acquire.task_registry.AuthoringFingerprint`, so the corpus is
reproducible and can never later be confused with a human-curated one.

The output is always ``FORMATIVE_MACHINE_AUTHORED`` with claim scope ``FORMATIVE_ONLY``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Sequence

from ..canonical import canonical_json
from ..evaluation.judge_client import DeepSeekJudge, JudgeUnavailable
from ..hashing import derive_id, sha256_hex
from .task_registry import STRATA, AuthoringFingerprint, TaskSpec

__all__ = [
    "PROMPT_VERSION",
    "AUTHOR_SYSTEM_PROMPT",
    "StratumProfile",
    "plan_profiles",
    "author_prompt_sha256",
    "AuthoringError",
    "author_clusters",
    "author_tasks",
    "authoring_manifest",
]

PROMPT_VERSION = "task_author_v1"

AUTHOR_SYSTEM_PROMPT = """\
You write research tasks for an evaluation of a web research agent. You never answer them.

A good task here is one a competent researcher could resolve from public web sources found by
a general web search, using between two and ten pages, in a single session. It must have a
verifiable answer: concrete entities, numbers, dates or documented positions. Do not invent
facts, do not assert what the answer is, and do not reference private, paywalled or
login-required sources.

Return only JSON matching the requested schema. No prose outside the JSON.
"""

_CLUSTER_PROMPT = """\
Propose {n} distinct research TOPIC CLUSTERS for a web-research benchmark.

Clusters must be mutually unrelated: two clusters must not share sources, entities or a
substantive subject area. Prefer areas with a stable public documentary record.

Return JSON:
{{"clusters": [{{"cluster_id": "kebab-case-slug", "title": "...", "scope": "one sentence"}}]}}
"""

_TASK_PROMPT = """\
Write {n} research tasks inside the topic cluster below. Each task must match the stratum
profile assigned to it exactly -- the profile is a requirement, not a suggestion.

Cluster: {cluster_title}
Scope: {cluster_scope}

Assigned profiles:
{profiles}

Stratum meanings:
- evidence_volume low/medium/high: roughly how much source text a researcher must read.
- facets_1_2 / facets_3_4 / facets_5_plus: how many distinct sub-questions the task requires.
- single_source_fact: one authoritative source settles it.
- multi_source_synthesis: the answer only exists by combining several sources.
- source_conflict: published sources genuinely disagree, and the task must surface that.
- negative_evidence: at least one sub-question has no public answer, and saying so is correct.
- citation_dense: the answer needs many distinct attributions.
- table_list_heavy: the evidence lives in tables, lists or specification sheets.
- high_redundancy: many sources repeat the same content, so selection matters.
- raw_content_missing: sources likely to resist text extraction (PDF-heavy, JS-heavy).
- close_research_complete / close_max_calls / close_no_tool: how much searching the task
  should demand -- a quick decisive finish, a long multi-round search, or a question that is
  answerable with almost no searching.

For every task give:
- "question": the research question, 60-600 characters, self-contained, no answer implied.
- "required_facets": 1-8 short noun phrases naming the sub-questions that must be covered.
- "fixed_queries": 3-6 web search queries. The first must address the whole question; the rest
  must each address one required facet. Write them as a person would type into a search box.
- "conflict_probe": a query that would surface disagreement, or null.
- "negative_probe": a query whose emptiness would be informative, or null.

Return JSON:
{{"tasks": [{{"profile_id": "...", "question": "...", "required_facets": [...],
  "fixed_queries": [...], "conflict_probe": "..." , "negative_probe": "..."}}]}}
"""


class AuthoringError(RuntimeError):
    """The author could not produce a corpus that satisfies the frozen plan."""


def author_prompt_sha256() -> str:
    """Digest of every prompt byte the authoring depends on.

    Recorded in the fingerprint so a reworded instruction is a different corpus rather than the
    same one with a different history.
    """
    return sha256_hex(canonical_json({
        "version": PROMPT_VERSION,
        "system": AUTHOR_SYSTEM_PROMPT,
        "clusters": _CLUSTER_PROMPT,
        "tasks": _TASK_PROMPT,
    }))


# --- the frozen coverage plan ----------------------------------------------------------


@dataclass(frozen=True)
class StratumProfile:
    """The strata one task must exhibit. Computed before authoring, verified after."""

    profile_id: str
    cluster_index: int
    strata: tuple[str, ...]

    def describe(self) -> str:
        return f"- {self.profile_id}: {', '.join(self.strata)}"


_EVIDENCE = ("evidence_volume_low", "evidence_volume_medium", "evidence_volume_high")
_FACETS = ("facets_1_2", "facets_3_4", "facets_5_plus")
_CLOSE = ("close_research_complete", "close_max_calls", "close_no_tool")
_OPTIONAL = (
    "single_source_fact", "multi_source_synthesis", "source_conflict", "negative_evidence",
    "citation_dense", "table_list_heavy", "high_redundancy", "raw_content_missing",
)


def plan_profiles(*, total: int, clusters: int, min_counts: dict[str, int]) -> list[StratumProfile]:
    """Deterministically assign a stratum profile to every task before anything is written.

    Round-robin over the mutually exclusive dimensions, then spread the optional strata so each
    reaches its configured minimum. Deterministic on purpose: the corpus plan is part of the
    pre-registration, so it must be reproducible from the config alone.
    """
    if total % clusters:
        raise AuthoringError(
            f"{total} tasks do not divide evenly into {clusters} clusters; an uneven cluster "
            "would make one topic carry more weight than another in a cluster-weighted analysis"
        )
    per_cluster = total // clusters
    profiles: list[StratumProfile] = []
    optional_cursor = 0
    for index in range(total):
        strata = [
            _EVIDENCE[index % len(_EVIDENCE)],
            _FACETS[(index // len(_EVIDENCE)) % len(_FACETS)],
            _CLOSE[index % len(_CLOSE)],
        ]
        # Two optional strata each, walked in a fixed order so every one is reached.
        for _ in range(2):
            candidate = _OPTIONAL[optional_cursor % len(_OPTIONAL)]
            optional_cursor += 1
            if candidate not in strata:
                strata.append(candidate)
        profiles.append(StratumProfile(
            profile_id=f"P{index:03d}",
            cluster_index=index // per_cluster,
            strata=tuple(sorted(strata)),
        ))

    shortfalls = {
        name: min_counts.get(name, 0) - sum(1 for p in profiles if name in p.strata)
        for name in STRATA
    }
    missing = {name: n for name, n in shortfalls.items() if n > 0}
    if missing:
        # Top up deterministically from the front rather than leaving a stratum short: a
        # stratum with no tasks means "where does P1 work" is unanswerable for that condition.
        topped = [list(p.strata) for p in profiles]
        for name, needed in sorted(missing.items()):
            added = 0
            for i in range(len(topped)):
                if added >= needed:
                    break
                if name not in topped[i]:
                    topped[i].append(name)
                    added += 1
            if added < needed:
                raise AuthoringError(
                    f"cannot reach the configured minimum of {min_counts[name]} tasks for "
                    f"stratum {name!r} with only {total} tasks"
                )
        profiles = [
            StratumProfile(profile_id=p.profile_id, cluster_index=p.cluster_index,
                           strata=tuple(sorted(s)))
            for p, s in zip(profiles, topped)
        ]
    return profiles


# --- calling the author -------------------------------------------------------------------


_CLUSTER_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["clusters"],
    "properties": {"clusters": {
        "type": "array", "minItems": 1,
        "items": {
            "type": "object", "additionalProperties": False,
            "required": ["cluster_id", "title", "scope"],
            "properties": {
                "cluster_id": {"type": "string", "pattern": "^[a-z0-9][a-z0-9-]{2,48}$"},
                "title": {"type": "string", "minLength": 3},
                "scope": {"type": "string", "minLength": 10},
            },
        },
    }},
}

_TASK_SCHEMA = {
    "type": "object", "additionalProperties": False, "required": ["tasks"],
    "properties": {"tasks": {
        "type": "array", "minItems": 1,
        "items": {
            "type": "object", "additionalProperties": False,
            "required": ["profile_id", "question", "required_facets", "fixed_queries"],
            "properties": {
                "profile_id": {"type": "string"},
                "question": {"type": "string", "minLength": 40, "maxLength": 900},
                "required_facets": {"type": "array", "minItems": 1, "maxItems": 8,
                                    "items": {"type": "string", "minLength": 2}},
                "fixed_queries": {"type": "array", "minItems": 3, "maxItems": 8,
                                  "items": {"type": "string", "minLength": 5}},
                "conflict_probe": {"type": ["string", "null"]},
                "negative_probe": {"type": ["string", "null"]},
            },
        },
    }},
}


def _checker(schema: dict):
    from jsonschema import Draft202012Validator

    validator = Draft202012Validator(schema)

    def check(data: dict) -> None:
        validator.validate(data)

    return check


async def author_clusters(judge: DeepSeekJudge, *, n: int) -> list[dict]:
    """Ask for ``n`` mutually unrelated topic clusters, and refuse a short or duplicated set."""
    response = await judge.judge(
        AUTHOR_SYSTEM_PROMPT, _CLUSTER_PROMPT.format(n=n), validate=_checker(_CLUSTER_SCHEMA),
    )
    clusters = response.data["clusters"][:n]
    ids = [c["cluster_id"] for c in clusters]
    if len(set(ids)) != len(ids):
        raise AuthoringError("the author returned duplicate cluster ids")
    if len(clusters) < n:
        raise AuthoringError(
            f"asked for {n} clusters and got {len(clusters)}; a short cluster set would make "
            "the split unit smaller than the design assumes"
        )
    return clusters


async def author_tasks(
    judge: DeepSeekJudge,
    *,
    total: int,
    clusters: int,
    min_counts: dict[str, int],
    requested_model: str,
    seed: int,
    authored_at_utc: str,
    target_model: str,
    min_question_chars: int = 60,
    max_question_chars: int = 600,
    min_fixed_queries: int = 3,
    attempts_per_cluster: int = 3,
) -> tuple[list[TaskSpec], AuthoringFingerprint, list[dict]]:
    """Author the whole corpus. Returns (tasks, fingerprint, raw responses).

    Raises rather than returning a short corpus: a registry that quietly came up eight tasks
    short would change the design's power without changing its pre-registration.
    """
    profiles = plan_profiles(total=total, clusters=clusters, min_counts=min_counts)
    cluster_records = await author_clusters(judge, n=clusters)
    raw_responses: list[dict] = []
    specs: list[TaskSpec] = []
    returned_model = ""
    system_fingerprint = ""

    for cluster_index, cluster in enumerate(cluster_records):
        wanted = [p for p in profiles if p.cluster_index == cluster_index]
        prompt = _TASK_PROMPT.format(
            n=len(wanted),
            cluster_title=cluster["title"],
            cluster_scope=cluster["scope"],
            profiles="\n".join(p.describe() for p in wanted),
        )
        # A bounded retry on a *format* rule, not on anything about the content: a question
        # two characters under the declared floor is a malformed draft, and re-requesting it is
        # not corpus selection. Retrying on anything outcome-related would be.
        response = None
        correction = ""
        for attempt in range(attempts_per_cluster):
            try:
                response = await judge.judge(
                    AUTHOR_SYSTEM_PROMPT, prompt + correction,
                    validate=_checker(_TASK_SCHEMA))
            except JudgeUnavailable as e:
                raise AuthoringError(
                    f"cluster {cluster['cluster_id']!r} could not be authored: {e}. A partially "
                    "authored corpus is not a corpus; nothing is sealed."
                ) from e
            short = [
                t["profile_id"] for t in response.data["tasks"]
                if not (min_question_chars <= len(" ".join(str(t["question"]).split()))
                        <= max_question_chars)
            ]
            thin = [
                t["profile_id"] for t in response.data["tasks"]
                if len([q for q in t["fixed_queries"] if str(q).strip()]) < min_fixed_queries
            ]
            if not short and not thin:
                break
            parts = []
            if short:
                parts.append(
                    f"these profiles had a question outside the {min_question_chars}-"
                    f"{max_question_chars} character range and must be rewritten longer or "
                    f"shorter: {', '.join(short)}")
            if thin:
                parts.append(
                    f"these profiles had fewer than {min_fixed_queries} fixed_queries and must "
                    f"be given more: {', '.join(thin)}")
            correction = (
                f"\n\nATTEMPT {attempt + 2}: " + "; ".join(parts) + "\n"
            )
        else:
            raise AuthoringError(
                f"cluster {cluster['cluster_id']!r} still violates the declared question "
                f"length or query-count bounds after {attempts_per_cluster} attempts"
            )
        returned_model = response.returned_model or returned_model
        system_fingerprint = response.system_fingerprint or system_fingerprint
        raw_responses.append({
            "cluster_id": cluster["cluster_id"],
            "response_sha256": sha256_hex(canonical_json(response.data)),
            "usage": response.usage,
            "request_id": response.request_id,
        })

        by_profile = {t["profile_id"]: t for t in response.data["tasks"]}
        for profile in wanted:
            draft = by_profile.get(profile.profile_id)
            if draft is None:
                raise AuthoringError(
                    f"the author skipped profile {profile.profile_id} in cluster "
                    f"{cluster['cluster_id']!r}; the stratum it carries would be unrepresented"
                )
            specs.append(_to_spec(draft, profile, cluster))

    if len(specs) != total:
        raise AuthoringError(f"authored {len(specs)} tasks, expected {total}")

    fingerprint = AuthoringFingerprint(
        provider="deepseek",
        requested_model=requested_model,
        returned_model=returned_model or requested_model,
        system_fingerprint=system_fingerprint,
        prompt_sha256=author_prompt_sha256(),
        seed=seed,
        authored_at_utc=authored_at_utc,
    )
    if target_model and target_model.lower() in (fingerprint.returned_model or "").lower():
        raise AuthoringError(
            f"the corpus was authored by the target model {target_model!r}; a corpus written by "
            "the model under test is selected for what that model finds easy"
        )
    return specs, fingerprint, raw_responses


def _to_spec(draft: dict, profile: StratumProfile, cluster: dict) -> TaskSpec:
    question = " ".join(str(draft["question"]).split())
    facets = tuple(str(f).strip() for f in draft["required_facets"] if str(f).strip())
    queries = tuple(str(q).strip() for q in draft["fixed_queries"] if str(q).strip())
    conflict = _clean(draft.get("conflict_probe"))
    negative = _clean(draft.get("negative_probe"))
    # Derived from content, not from a counter: re-authoring the same question in the same
    # cluster yields the same id, and two different questions never collide.
    task_id = "T" + derive_id("task", {
        "cluster": cluster["cluster_id"], "question": question,
    })[:14]
    return TaskSpec(
        task_id=task_id,
        topic=str(cluster["title"]),
        question=question,
        required_facets=facets,
        fixed_queries=queries,
        strata=profile.strata,
        topic_cluster=str(cluster["cluster_id"]),
        conflict_probe=conflict,
        negative_probe=negative,
    )


def _clean(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = " ".join(str(value).split())
    return text or None


def authoring_manifest(
    fingerprint: AuthoringFingerprint,
    raw_responses: Sequence[dict],
    profiles: Sequence[StratumProfile],
) -> dict:
    """The record of how the corpus came to exist, hashed alongside the registry."""
    return {
        "prompt_version": PROMPT_VERSION,
        "prompt_sha256": author_prompt_sha256(),
        "fingerprint": fingerprint.content(),
        "responses": list(raw_responses),
        "profiles": [
            {"profile_id": p.profile_id, "cluster_index": p.cluster_index,
             "strata": list(p.strata)}
            for p in profiles
        ],
        "manifest_sha256": sha256_hex(canonical_json({
            "fingerprint": fingerprint.content(),
            "responses": list(raw_responses),
        })),
    }
