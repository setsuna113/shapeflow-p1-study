"""`prepare`: author, audit, split, seal -- and split the corpus by who may read it.

The whole phase runs before a single Tavily credit is spent and before any P0 or P1 output
exists. That ordering is the point: a task cannot be edited, dropped or reworded once an outcome
is known, because that is outcome-dependent corpus selection and it invalidates everything
computed on the corpus.

Two files come out of every task, on purpose:

``steward/tasks/<id>.json``   the whole record, including the authored facets, the fixed queries
                              and both probes. Steward-readable only. This is what drives Tavily.
``runner/frozen_corpus/tasks/<id>.json``
                              the question, and nothing else.

Handing a selector the authored facets would give P1 a decomposition of the question that P0
never received, and the measured advantage would be an artifact of the harness. Writing two
files with different owners makes that a filesystem fact rather than a rule someone has to keep.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Sequence

from ..acquire.registry_audit import AuditReport, audit_registry, audit_tasks
from ..acquire.task_author import author_tasks, authoring_manifest, plan_profiles
from ..acquire.task_registry import TaskRegistry, TaskSpec, build_registry
from ..canonical import canonical_json
from ..hashing import sha256_hex
from .settings import Settings

__all__ = ["PrepareResult", "prepare_corpus", "write_task_views", "load_sealed_registry"]

_STRATUM_TO_VOLUME = {
    "evidence_volume_low": "low",
    "evidence_volume_medium": "medium",
    "evidence_volume_high": "high",
}
_STRATUM_TO_FACETS = {"facets_1_2": "1-2", "facets_3_4": "3-4", "facets_5_plus": "5+"}
_BOOLEAN_STRATA = {
    "single_source_fact": "single_source_fact",
    "multi_source_synthesis": "multi_source_synthesis",
    "source_conflict": "source_conflict",
    "negative_evidence": "negative_evidence",
    "citation_dense": "citation_dense",
    "table_list_heavy": "table_list_heavy",
    "high_redundancy": "high_redundancy",
    "raw_content_missing": "raw_content_missing_expected",
}


@dataclass
class PrepareResult:
    registry: TaskRegistry
    registry_sha256: str
    audit: AuditReport
    manifest_path: Path
    registry_path: Path
    steward_task_count: int
    runner_task_count: int


def _task_record(task: TaskSpec, split: str) -> dict:
    """The steward's full record, in the shape ``schemas/task.schema.json`` describes."""
    acquisition = {
        "authored_facets": list(task.required_facets),
        "fixed_queries": list(task.fixed_queries),
        "conflict_probe": task.conflict_probe,
        "negative_or_gap_probe": task.negative_probe,
        "table_list_numeric_probe": None,
        "authoring_method": "MACHINE_DECOMPOSED",
        "authoring_model_prompt_fingerprint": None,
        "content_sha256": task.acquisition_spec_sha256,
    }
    strata: dict = {"evidence_volume": "medium", "facet_count": "3-4"}
    for stratum in task.strata:
        if stratum in _STRATUM_TO_VOLUME:
            strata["evidence_volume"] = _STRATUM_TO_VOLUME[stratum]
        elif stratum in _STRATUM_TO_FACETS:
            strata["facet_count"] = _STRATUM_TO_FACETS[stratum]
        elif stratum in _BOOLEAN_STRATA:
            strata[_BOOLEAN_STRATA[stratum]] = True
    return {
        "task_id": task.task_id,
        "split": split,
        "cluster_id": task.topic_cluster,
        "corpus_tier": task.corpus_tier,
        "claim_scope": task.claim_scope,
        "treatment_visible": {"original_question": task.question},
        "acquisition_spec": acquisition,
        "strata": strata,
    }


def _runner_record(task: TaskSpec, split: str) -> dict:
    """What the treatment identity may see: the question, and nothing else."""
    return {
        "task_id": task.task_id,
        "split": split,
        "original_question": task.question,
        "corpus_tier": task.corpus_tier,
        "claim_scope": task.claim_scope,
    }


def _evaluator_record(task: TaskSpec, split: str) -> dict:
    """What the evaluator needs to build truth, and where it is allowed to live.

    The authored facets and the acquisition spec are the answer key's raw material. They
    were reachable from the steward tree, which the runner could traverse -- so the identity
    under measurement could read the facets its output was about to be scored against. This
    view puts them where only the evaluator can read them.
    """
    return {
        "task_id": task.task_id,
        "split": split,
        "cluster_id": task.topic_cluster,
        "original_question": task.question,
        "authored_facets": list(task.required_facets),
        "fixed_queries": list(task.fixed_queries),
        "conflict_probe": task.conflict_probe,
        "negative_or_gap_probe": task.negative_probe,
        "acquisition_spec_sha256": task.acquisition_spec_sha256,
        "strata": list(task.strata),
    }


def _validator(repo: Path, schema_name: str):
    from jsonschema import Draft202012Validator

    schema = json.loads((repo / "schemas" / schema_name).read_text(encoding="utf-8"))
    return Draft202012Validator(schema)


def write_task_views(settings: Settings, registry: TaskRegistry) -> tuple[int, int]:
    """Materialize both views, each validated against its own schema. Write-once.

    Refusing to overwrite matters as much here as it does for the registry: a task record that
    could be rewritten after acquisition would let the corpus be reshaped around a result.
    """
    steward_dir = settings.path("tasks")
    runner_dir = settings.path("frozen_corpus_for_runner") / "tasks"
    evaluator_dir = settings.path("evaluator_root") / "tasks"
    steward_dir.mkdir(parents=True, exist_ok=True)
    runner_dir.mkdir(parents=True, exist_ok=True)
    evaluator_dir.mkdir(parents=True, exist_ok=True)

    task_schema = _validator(settings.repo, "task.schema.json")
    runner_schema = _validator(settings.repo, "runner_task.schema.json")

    written_steward = written_runner = 0
    for task in registry.tasks:
        split = registry.split_of[task.task_id]
        full = _task_record(task, split)
        task_schema.validate(full)
        path = steward_dir / f"{task.task_id}.json"
        if not path.exists():
            path.write_text(json.dumps(full, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            os.chmod(path, 0o440)
            written_steward += 1

        visible = _runner_record(task, split)
        runner_schema.validate(visible)
        rpath = runner_dir / f"{task.task_id}.json"
        if not rpath.exists():
            rpath.write_text(json.dumps(visible, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")
            os.chmod(rpath, 0o444)
            written_runner += 1

        epath = evaluator_dir / f"{task.task_id}.json"
        if not epath.exists():
            epath.write_text(
                json.dumps(_evaluator_record(task, split), indent=2, sort_keys=True) + "\n",
                encoding="utf-8")
            os.chmod(epath, 0o440)
    return written_steward, written_runner


async def prepare_corpus(
    settings: Settings,
    *,
    judge,
    authored_at_utc: str,
    target_model: str,
    total: Optional[int] = None,
    clusters: Optional[int] = None,
) -> PrepareResult:
    """Author, audit, split and seal. Anything less than a complete corpus raises.

    The audit runs twice on purpose: once on the authored tasks, and once on the *sealed*
    structure. The first catches an unretrievable or duplicated question; the second catches a
    cluster split across two splits, a wrong split size, or a reserve order that does not
    enumerate the reserve -- defects that only exist once the tasks have been assigned.
    """
    task_config = settings.configs["task_source"]

    # The seal is write-once, and that check used to happen *after* authoring: a re-run against
    # an existing registry paid DeepSeek to author a full corpus and only then refused to write
    # it. Observed on the run host -- $2.07 and 64 authored tasks, discarded. Nothing about the
    # refusal needs the corpus to exist, so it moves ahead of the spend.
    settings.ensure_paths("tasks", "steward_root")
    registry_path = settings.path("tasks") / str(task_config["source"]["registry_path"])
    if registry_path.exists():
        raise RuntimeError(
            f"{registry_path} already exists. A sealed registry is write-once: rewriting it "
            "after any result exists would be outcome-dependent corpus selection. Nothing was "
            "authored and nothing was charged. To continue the existing corpus, run the later "
            "steps against it; to start a new one, give it a new data root and a new approval."
        )

    splits = task_config["splits"]
    total = total or sum(splits.values())
    minimum = clusters or int(task_config["audit"]["require_distinct_topic_clusters"])
    # Whole clusters move to one split, so the cluster count has to divide the corpus evenly --
    # and it must still meet the configured minimum. Searching *downwards* from the minimum
    # satisfies the first constraint by breaking the second, which is how the first attempt
    # produced eight clusters against a declared floor of twelve and the audit refused to seal.
    clusters = next((n for n in range(minimum, total + 1) if total % n == 0), 0)
    if not clusters:
        raise ValueError(
            f"no cluster count divides {total} tasks evenly at or above the configured minimum "
            f"of {minimum}; change the split sizes rather than the floor"
        )

    specs, fingerprint, responses = await author_tasks(
        judge,
        total=total,
        clusters=clusters,
        min_counts=task_config["strata_min_counts"],
        requested_model=settings.judge_model(),
        seed=int(task_config["authoring"]["seed"]),
        authored_at_utc=authored_at_utc,
        target_model=target_model,
        min_question_chars=int(task_config["audit"]["min_question_chars"]),
        max_question_chars=int(task_config["audit"]["max_question_chars"]),
        min_fixed_queries=int(task_config["audit"]["min_fixed_queries"]),
    )

    audit = audit_tasks(specs, config=task_config)
    audit.raise_if_failed()

    registry = build_registry(
        tasks=specs,
        fingerprint=fingerprint,
        screen_n=int(splits["FORMATIVE_SCREEN"]),
        pilot_n=int(splits["FORMATIVE_POWER_PILOT"]),
        target_model=target_model,
    )
    sealed_audit = audit_registry(registry, config=task_config)
    sealed_audit.raise_if_failed()

    # Re-checked by seal() itself: the pre-flight check above saves the spend, but only the
    # write-once check at the moment of writing is a guarantee against a concurrent author.
    registry_sha = registry.seal(registry_path)

    manifest = authoring_manifest(
        fingerprint, responses,
        plan_profiles(total=total, clusters=clusters,
                      min_counts=task_config["strata_min_counts"]),
    )
    manifest["registry_sha256"] = registry_sha
    manifest["corpus_tier"] = registry.corpus_tier
    manifest["claim_scope"] = registry.claim_scope
    manifest["audit_stats"] = audit.stats
    manifest_path = settings.path("tasks") / "authoring_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(
            f"{manifest_path} already exists; the authoring record is write-once so a re-authored "
            "corpus cannot be presented under the provenance of the first one"
        )
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                             encoding="utf-8")

    steward_count, runner_count = write_task_views(settings, registry)
    return PrepareResult(
        registry=registry, registry_sha256=registry_sha, audit=sealed_audit,
        manifest_path=manifest_path, registry_path=registry_path,
        steward_task_count=steward_count, runner_task_count=runner_count,
    )


def load_sealed_registry(settings: Settings) -> tuple[dict, str]:
    """Read the sealed registry and re-derive its digest.

    The digest is recomputed rather than trusted: a sealed file whose recorded hash no longer
    matches its contents has been edited, and every result computed against it is suspect.
    """
    task_config = settings.configs["task_source"]
    path = settings.path("tasks") / str(task_config["source"]["registry_path"])
    body = json.loads(path.read_text(encoding="utf-8"))
    recorded = body.pop("registry_sha256", "")
    actual = sha256_hex(canonical_json(body))
    if recorded != actual:
        raise ValueError(
            f"{path} records registry_sha256 {recorded!r} but its contents hash to {actual!r}; "
            "the sealed corpus has been edited"
        )
    body["registry_sha256"] = recorded
    return body, actual


def tasks_of_split(registry_body: dict, split: str) -> Sequence[dict]:
    return [t for t in registry_body["tasks"] if t["split"] == split]
