"""The task corpus: sealed before any P1 output exists, and honest about what it is.

Protocol section 7.1 gives three sources in order of preference. There is no user-supplied
sealed registry and no pre-existing Tavily-compatible pool, so this builds the third:
``FORMATIVE_MACHINE_AUTHORED``, with ``claim_scope = FORMATIVE_ONLY``. That label is load-bearing
and travels with every downstream artifact -- a machine-authored corpus cannot support a
confirmatory claim, and the protocol says so explicitly. Presenting these results as
paper-confirmatory would be the same category of error as any of the fabrications this study
guards against.

Three rules the sealing enforces:

**The author is not the subject.** Tasks are written by DeepSeek, never by the Qwen target
model. A corpus authored by the model under test would be selected for what that model finds
easy, and every arm shares that bias while the comparison silently inherits it.

**Sealed before acquisition, and before any outcome exists.** Task specs are frozen and hashed
first; only then is Tavily called. A task cannot be edited, dropped or reworded after any P0 or
P1 result is known -- that is outcome-dependent corpus selection, and it invalidates everything
computed on it.

**Split by topic and source cluster, not by task.** Two paraphrases of one question, or two
tasks sharing most of their sources, are one observation. Splitting them across SCREEN and
POWER_PILOT would leak, and counting them as independent would overstate n.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Optional

from ..canonical import canonical_json
from ..hashing import derive_id, merkle_root, sha256_hex

__all__ = [
    "CORPUS_TIER",
    "CLAIM_SCOPE",
    "STRATA",
    "TaskSpec",
    "AuthoringFingerprint",
    "TaskRegistry",
    "SplitName",
    "build_registry",
    "merkle_root",
]

CORPUS_TIER = "FORMATIVE_MACHINE_AUTHORED"
CLAIM_SCOPE = "FORMATIVE_ONLY"

# Protocol section 7.2. Every stratum must be represented, because "P1 helps" without knowing
# where it helps is not an answer -- the eligibility envelope is one of the five questions.
STRATA = (
    "evidence_volume_low", "evidence_volume_medium", "evidence_volume_high",
    "facets_1_2", "facets_3_4", "facets_5_plus",
    "single_source_fact", "multi_source_synthesis",
    "source_conflict", "negative_evidence", "citation_dense",
    "table_list_heavy", "high_redundancy", "raw_content_missing",
    "close_research_complete", "close_max_calls", "close_no_tool",
)

SplitName = str
SPLITS: tuple[SplitName, ...] = ("FORMATIVE_SCREEN", "FORMATIVE_POWER_PILOT", "RESERVE")


@dataclass(frozen=True)
class AuthoringFingerprint:
    """Who wrote the tasks, with what, and under which frozen prompt.

    Recorded so the corpus is reproducible and so its provenance cannot later be confused with
    a human-curated one. ``target_model`` is asserted absent from the authoring path.
    """

    provider: str
    requested_model: str
    returned_model: str
    system_fingerprint: str
    prompt_sha256: str
    seed: int
    authored_at_utc: str
    #: The decoding policy that reached the model. Not the one the config declares -- those
    #: were the same value only by coincidence, and for most of them not at all.
    sampling: Mapping[str, Any] = field(default_factory=dict)

    def content(self) -> dict:
        return {
            "provider": self.provider,
            "requested_model": self.requested_model,
            "returned_model": self.returned_model,
            "system_fingerprint": self.system_fingerprint,
            "prompt_sha256": self.prompt_sha256,
            "sampling": dict(self.sampling),
            "seed": self.seed,
            "authored_at_utc": self.authored_at_utc,
        }


@dataclass(frozen=True)
class TaskSpec:
    """One sealed research task and the acquisition it authorises.

    ``fixed_queries`` is part of the seal: the frozen source pool is built from exactly these,
    so every arm of this task searches the same world. Letting an arm issue its own queries
    would give the arms different worlds and make the comparison meaningless.
    """

    task_id: str
    topic: str
    question: str
    required_facets: tuple[str, ...]
    fixed_queries: tuple[str, ...]
    strata: tuple[str, ...]
    topic_cluster: str
    conflict_probe: Optional[str] = None
    negative_probe: Optional[str] = None
    corpus_tier: str = CORPUS_TIER
    claim_scope: str = CLAIM_SCOPE

    def content(self) -> dict:
        return {
            "task_id": self.task_id,
            "topic": self.topic,
            "question": self.question,
            "required_facets": list(self.required_facets),
            "fixed_queries": list(self.fixed_queries),
            "strata": sorted(self.strata),
            "topic_cluster": self.topic_cluster,
            "conflict_probe": self.conflict_probe,
            "negative_probe": self.negative_probe,
            "corpus_tier": self.corpus_tier,
            "claim_scope": self.claim_scope,
        }

    @property
    def spec_sha256(self) -> str:
        return sha256_hex(canonical_json(self.content()))

    @property
    def acquisition_spec_sha256(self) -> str:
        """Digest of just the part that drives Tavily.

        Separate from the task digest so a change to the question is distinguishable from a
        change to the world the arms will search.
        """
        return sha256_hex(canonical_json({
            "task_id": self.task_id,
            "fixed_queries": list(self.fixed_queries),
            "conflict_probe": self.conflict_probe,
            "negative_probe": self.negative_probe,
        }))


@dataclass
class TaskRegistry:
    """The sealed corpus: tasks, their split, and the provenance of the whole thing."""

    tasks: tuple[TaskSpec, ...]
    split_of: dict[str, SplitName]
    reserve_order: tuple[str, ...]
    fingerprint: AuthoringFingerprint
    corpus_tier: str = CORPUS_TIER
    claim_scope: str = CLAIM_SCOPE

    def of_split(self, split: SplitName) -> tuple[TaskSpec, ...]:
        return tuple(t for t in self.tasks if self.split_of[t.task_id] == split)

    @property
    def merkle_root(self) -> str:
        return merkle_root([t.spec_sha256 for t in self.tasks])

    def strata_coverage(self) -> dict[str, int]:
        counts = {s: 0 for s in STRATA}
        for task in self.tasks:
            for s in task.strata:
                if s in counts:
                    counts[s] += 1
        return counts

    def missing_strata(self) -> tuple[str, ...]:
        return tuple(s for s, n in self.strata_coverage().items() if n == 0)

    def content(self) -> dict:
        return {
            "corpus_tier": self.corpus_tier,
            "claim_scope": self.claim_scope,
            "fingerprint": self.fingerprint.content(),
            "merkle_root": self.merkle_root,
            "tasks": [
                {**t.content(), "spec_sha256": t.spec_sha256,
                 "acquisition_spec_sha256": t.acquisition_spec_sha256,
                 "split": self.split_of[t.task_id]}
                for t in sorted(self.tasks, key=lambda t: t.task_id)
            ],
            "reserve_order": list(self.reserve_order),
        }

    def seal(self, path: Path) -> str:
        """Write the registry and return its digest. Refuses to overwrite.

        A sealed registry that could be rewritten is not sealed. Re-authoring after any outcome
        exists is outcome-dependent corpus selection, so the file is write-once and a new corpus
        gets a new path and a new approval.
        """
        path = Path(path)
        if path.exists():
            raise RuntimeError(
                f"{path} already exists. A sealed registry is write-once: rewriting it after "
                "any result exists would be outcome-dependent corpus selection."
            )
        body = self.content()
        digest = sha256_hex(canonical_json(body))
        body["registry_sha256"] = digest
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(body, indent=2, sort_keys=True), encoding="utf-8")
        return digest


class SealError(RuntimeError):
    """The registry cannot be sealed as specified."""


def build_registry(
    *,
    tasks: Iterable[TaskSpec],
    fingerprint: AuthoringFingerprint,
    screen_n: int,
    pilot_n: int,
    target_model: str,
) -> TaskRegistry:
    """Assign splits and validate the seal's preconditions.

    Splitting is by ``topic_cluster``: an entire cluster goes to one split. Two paraphrases of a
    question, or two tasks sharing most of their sources, are one observation -- splitting them
    would leak between SCREEN and POWER_PILOT and counting them separately would overstate n.
    """
    tasks = sorted(tasks, key=lambda t: t.task_id)
    if fingerprint.requested_model == target_model or fingerprint.returned_model == target_model:
        raise SealError(
            f"tasks were authored by the target model {target_model!r}. A corpus written by the "
            "model under test is selected for what that model finds easy, and every arm "
            "inherits the bias while the comparison silently absorbs it."
        )
    ids = [t.task_id for t in tasks]
    if len(set(ids)) != len(ids):
        raise SealError("duplicate task ids")

    clusters: dict[str, list[TaskSpec]] = {}
    for task in tasks:
        clusters.setdefault(task.topic_cluster, []).append(task)

    split_of: dict[str, SplitName] = {}
    assigned = {"FORMATIVE_SCREEN": 0, "FORMATIVE_POWER_PILOT": 0}
    reserve: list[str] = []
    for cluster in sorted(clusters):
        members = clusters[cluster]
        if assigned["FORMATIVE_SCREEN"] + len(members) <= screen_n:
            target = "FORMATIVE_SCREEN"
        elif assigned["FORMATIVE_POWER_PILOT"] + len(members) <= pilot_n:
            target = "FORMATIVE_POWER_PILOT"
        else:
            target = "RESERVE"
        for task in members:
            split_of[task.task_id] = target
            if target == "RESERVE":
                reserve.append(task.task_id)
        if target in assigned:
            assigned[target] += len(members)

    registry = TaskRegistry(
        tasks=tuple(tasks), split_of=split_of, reserve_order=tuple(reserve),
        fingerprint=fingerprint,
    )
    missing = registry.missing_strata()
    if missing:
        raise SealError(
            f"{len(missing)} stratum/strata unrepresented: {list(missing)}. Protocol 7.2 "
            "requires all of them -- 'P1 helps' without knowing where it helps is not an answer."
        )
    return registry
