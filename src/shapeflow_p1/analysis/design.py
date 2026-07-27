"""Freeze the pre-treatment feature registry and eligibility specification.

The question "when is P1 useful?" cannot be answered by choosing features after outcomes are
visible.  This module runs after acquisition but before the first treatment cell.  It derives a
small, deterministic task-level feature table solely from the frozen vendor-visible world and
binds that table into the already configured shallow eligibility learner.

The artifacts live in the evaluator tree.  The runner never reads them, and no task answer,
TruthPacket, P0 output, selector trace, fallback, token reuse result or treatment outcome enters
the features.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path

from ..canonical import canonical_json
from ..evidence.chunkers import Tokenizer, markdown_structure_v1
from ..evidence.model_tokenizer import load_frozen_tokenizer, tokenizer_sha256
from ..hashing import sha256_hex
from ..object_store import ObjectStore
from .e2e_effects import (
    ALLOWED_PRETREATMENT_FEATURES,
    ELIGIBILITY_TARGET_FAMILY,
    QUALITY_GUARD_DIRECTIONS,
    task_feature_registry_sha256,
)

__all__ = [
    "FEATURE_EXTRACTOR_VERSION",
    "build_eligibility_spec",
    "build_task_feature_registry",
    "freeze_analysis_design",
    "load_runner_analysis_design_receipt",
]


FEATURE_EXTRACTOR_VERSION = "frozen_task_features_v3_pretreatment_task_and_pool"
REFERENCE_CHUNKER_VERSION = "markdown_structure_v1:max_tokens=320:model_tokenizer"
REGISTRY_FILENAME = "TASK_FEATURE_REGISTRY.json"
ELIGIBILITY_SPEC_FILENAME = "ELIGIBILITY_SPEC.json"
RUNNER_RECEIPT_FILENAME = "ANALYSIS_DESIGN_RECEIPT.json"
EVALUATOR_RECEIPT_FILENAME = "ANALYSIS_DESIGN_RECEIPT.json"

_DECLARED_STRATA_FEATURES = (
    "evidence_volume_low",
    "evidence_volume_medium",
    "evidence_volume_high",
    "facets_1_2",
    "facets_3_4",
    "facets_5_plus",
    "single_source_fact",
    "multi_source_synthesis",
    "source_conflict",
    "negative_evidence",
    "citation_dense",
    "table_list_heavy",
    "high_redundancy",
    "raw_content_missing",
)


def _unsigned_sha(body: Mapping[str, object], field: str = "content_sha256") -> str:
    return sha256_hex(canonical_json({
        key: value for key, value in body.items() if key != field
    }))


def _quantile(values: Sequence[int], q: float) -> float:
    """Deterministic linear quantile, defined here rather than delegated to library defaults."""
    if not values:
        return 0.0
    ordered = sorted(float(value) for value in values)
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_verified_json(path: Path, *, hash_field: str) -> dict:
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"pre-treatment input {path} is unreadable: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"pre-treatment input {path} is not a JSON object")
    recorded = str(body.get(hash_field) or "")
    actual = _unsigned_sha(body, hash_field)
    if recorded != actual:
        raise ValueError(
            f"pre-treatment input {path} was edited: records {recorded!r}, hashes to {actual}")
    return body


def _task_features(
    *,
    pool: Mapping[str, object],
    task_view: Mapping[str, object],
    objects: ObjectStore,
    tokenizer: Tokenizer,
) -> dict[str, float]:
    occurrences = pool.get("occurrences")
    snapshots = pool.get("snapshots")
    if not isinstance(occurrences, list) or not isinstance(snapshots, dict):
        raise ValueError("frozen source pool lacks occurrences or snapshots")
    all_chunks = []
    candidate_tokens = 0
    raw_available = 0
    evidence_identities: list[str] = []

    for index, raw_occurrence in enumerate(occurrences):
        if not isinstance(raw_occurrence, dict):
            raise ValueError(f"source occurrence {index} is malformed")
        content_hash = str(raw_occurrence.get("content_hash") or "")
        text = ""
        if content_hash:
            snapshot = snapshots.get(content_hash)
            if not isinstance(snapshot, dict):
                raise ValueError(
                    f"occurrence {index} references missing snapshot {content_hash!r}")
            object_ref = str(snapshot.get("object_ref") or "")
            if not object_ref:
                raise ValueError(f"snapshot {content_hash!r} has no object_ref")
            try:
                text = objects.get_bytes(object_ref).decode("utf-8")
            except (KeyError, UnicodeDecodeError, RuntimeError) as exc:
                raise ValueError(
                    f"snapshot {content_hash!r} is unavailable or corrupt") from exc
            raw_available += 1
            evidence_identities.append(f"raw:{content_hash}")
        else:
            text = str(raw_occurrence.get("snippet_content") or "")
            evidence_identities.append(f"snippet:{sha256_hex(text.encode('utf-8'))}")
        candidate_tokens += tokenizer.count(text)
        all_chunks.extend(
            markdown_structure_v1(text, tokenizer=tokenizer, max_tokens=320))

    source_count = len(occurrences)
    lengths = [chunk.token_len for chunk in all_chunks]
    structural = sum(
        chunk.kind.startswith("table_") or chunk.kind == "list_item"
        for chunk in all_chunks
    )
    unique_evidence = len(set(evidence_identities))
    question = task_view.get("original_question")
    facets = task_view.get("authored_facets")
    queries = task_view.get("fixed_queries")
    strata = task_view.get("strata")
    if not isinstance(question, str) or not question.strip():
        raise ValueError("evaluator task view lacks a non-empty original_question")
    if (
        not isinstance(facets, list)
        or not facets
        or any(not isinstance(item, str) or not item.strip() for item in facets)
    ):
        raise ValueError("evaluator task view lacks valid authored_facets")
    if (
        not isinstance(queries, list)
        or not queries
        or any(not isinstance(item, str) or not item.strip() for item in queries)
    ):
        raise ValueError("evaluator task view lacks valid fixed_queries")
    if (
        not isinstance(strata, list)
        or any(not isinstance(item, str) or not item for item in strata)
        or len(strata) != len(set(strata))
    ):
        raise ValueError("evaluator task view lacks a valid unique strata list")
    declared_strata = set(strata)
    features = {
        "candidate_evidence_tokens": float(candidate_tokens),
        "source_count": float(source_count),
        "span_count": float(len(all_chunks)),
        "span_length_q25": _quantile(lengths, 0.25),
        "span_length_q50": _quantile(lengths, 0.50),
        "span_length_q75": _quantile(lengths, 0.75),
        "span_length_q90": _quantile(lengths, 0.90),
        "table_list_fraction": (
            float(structural) / len(all_chunks) if all_chunks else 0.0
        ),
        "redundancy": (
            1.0 - float(unique_evidence) / source_count if source_count else 0.0
        ),
        "raw_content_available_fraction": (
            float(raw_available) / source_count if source_count else 0.0
        ),
        "question_token_count": float(tokenizer.count(question)),
        "authored_facet_count": float(len(facets)),
        "fixed_query_count": float(len(queries)),
    }
    features.update({
        f"declared_stratum_{name}": float(name in declared_strata)
        for name in _DECLARED_STRATA_FEATURES
    })
    if set(features) - ALLOWED_PRETREATMENT_FEATURES:
        raise AssertionError("feature extractor emitted an unregistered feature")
    return features


def build_task_feature_registry(settings) -> dict:
    """Build a content-addressed registry over every acquired pre-treatment task."""
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    task_dir = settings.path("evaluator_root") / "tasks"
    pool_paths = sorted(pool_dir.glob("*.json"))
    if not pool_paths:
        raise ValueError("no acquired frozen source pools; analysis design cannot be frozen")
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    tokenizer = load_frozen_tokenizer(settings)
    tokenizer_digest = tokenizer_sha256(tokenizer)
    records: dict[str, dict] = {}
    for pool_path in pool_paths:
        pool = _load_verified_json(pool_path, hash_field="pool_sha256")
        task_id = str(pool.get("task_id") or "")
        if not task_id or task_id != pool_path.stem or task_id in records:
            raise ValueError(f"pool {pool_path} has duplicate or inconsistent task identity")
        task_path = task_dir / f"{task_id}.json"
        try:
            task_view = json.loads(task_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"evaluator task view {task_path} is unreadable: {exc}") from exc
        if (
            not isinstance(task_view, dict)
            or str(task_view.get("task_id") or "") != task_id
            or not task_view.get("split")
            or not task_view.get("cluster_id")
        ):
            raise ValueError(f"evaluator task view {task_path} lacks frozen coordinates")
        record = {
            "schema_version": "frozen_task_features_v1",
            "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
            "reference_chunker_version": REFERENCE_CHUNKER_VERSION,
            "tokenizer_sha256": tokenizer_digest,
            "task_id": task_id,
            "split": str(task_view["split"]),
            "cluster_id": str(task_view["cluster_id"]),
            "source_pool_sha256": str(pool["pool_sha256"]),
            "task_view_sha256": sha256_hex(canonical_json(task_view)),
            "features": _task_features(
                pool=pool,
                task_view=task_view,
                objects=objects,
                tokenizer=tokenizer,
            ),
        }
        record["content_sha256"] = _unsigned_sha(record)
        records[task_id] = record

    registry = {
        "schema_version": "frozen_task_feature_registry_v1",
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "reference_chunker_version": REFERENCE_CHUNKER_VERSION,
        "tokenizer_sha256": tokenizer_digest,
        "records": dict(sorted(records.items())),
        "task_feature_registry_sha256": task_feature_registry_sha256(records),
    }
    registry["content_sha256"] = _unsigned_sha(registry)
    return registry


def build_eligibility_spec(settings, registry: Mapping[str, object]) -> dict:
    """Derive the one allowed spec from the frozen registry and hash-locked configs."""
    config = settings.get("decision", "eligibility_learner")
    feature_names = list(map(str, config["feature_names"]))
    if (
        str(config.get("analysis_scope") or "")
        != "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY"
        or
        str(config["feature_extractor_version"]) != FEATURE_EXTRACTOR_VERSION
        or not feature_names
        or len(feature_names) != len(set(feature_names))
        or any(name not in ALLOWED_PRETREATMENT_FEATURES for name in feature_names)
    ):
        raise ValueError("decision.yaml declares an unsupported eligibility feature design")
    declared_targets = config.get("targets")
    if not isinstance(declared_targets, Mapping) or set(declared_targets) != set(
        ELIGIBILITY_TARGET_FAMILY
    ):
        raise ValueError("decision.yaml must declare the exact eligibility target family")
    core = settings.get("decision", "e2e_analysis", "core_factorial")
    if not isinstance(core, Mapping) or set(core) != {"p0", "h", "c", "hc"}:
        raise ValueError("decision.yaml lacks the exact core factorial for eligibility")
    arm_map = {
        semantic: str(raw.get("arm_id") or "")
        for semantic, raw in core.items()
        if isinstance(raw, Mapping)
    }
    if set(arm_map) != {"p0", "h", "c", "hc"} or len(set(arm_map.values())) != 4:
        raise ValueError("core factorial arm identities are invalid")
    target_rows: list[dict] = []
    for target_id, (treatment, comparators) in ELIGIBILITY_TARGET_FAMILY.items():
        raw = declared_targets[target_id]
        if (
            not isinstance(raw, Mapping)
            or str(raw.get("treatment_semantic") or "") != treatment
            or tuple(map(str, raw.get("comparator_semantics") or ())) != comparators
            or str(raw.get("comparator_logic") or "") != "ALL"
        ):
            raise ValueError(
                f"decision.yaml eligibility target {target_id} differs from the frozen family"
            )
        target_rows.append({
            "target_id": target_id,
            "treatment_semantic": treatment,
            "treatment_arm_id": arm_map[treatment],
            "comparator_semantics": list(comparators),
            "comparator_arm_ids": [arm_map[item] for item in comparators],
            "comparator_logic": "ALL",
        })
    structured = settings.get(
        "decision", "structured_increment", "strict_quality_noninferiority"
    )
    if not isinstance(structured, Mapping):
        raise ValueError("decision.yaml lacks strict eligibility quality guards")
    higher = {
        str(metric): float(margin)
        for metric, margin in dict(
            structured.get("higher_is_better_min_effect") or {}
        ).items()
    }
    lower = {
        str(metric): float(margin)
        for metric, margin in dict(
            structured.get("lower_is_better_max_effect") or {}
        ).items()
    }
    if (
        not higher
        or not lower
        or set(higher) & set(lower)
        or any(QUALITY_GUARD_DIRECTIONS.get(metric) != "higher_is_better" for metric in higher)
        or any(QUALITY_GUARD_DIRECTIONS.get(metric) != "lower_is_better" for metric in lower)
    ):
        raise ValueError("decision.yaml eligibility quality guards have invalid directions")
    absolute = config.get("absolute_treatment_requirements")
    if (
        not isinstance(absolute, Mapping)
        or set(absolute) != {"qualified_report_min"}
        or float(absolute["qualified_report_min"]) != 1.0
    ):
        raise ValueError(
            "decision.yaml eligibility must require an always-qualified strict treatment report"
        )
    body = {
        "schema_version": "e2e_eligibility_spec_v2",
        "analysis_scope": "EXPLORATORY_TASK_LEVEL_PRETREATMENT_FORMATIVE_ONLY",
        "targets": target_rows,
        "quality_view": str(config["quality_view"]),
        "quality_guards": {
            "higher_is_better_min_effect": higher,
            "lower_is_better_max_effect": lower,
            "inapplicability_policy": "BOTH_ARMS_INAPPLICABLE_PASS_OTHERWISE_FAIL_CLOSED",
        },
        "absolute_treatment_requirements": {
            "qualified_report_min": 1.0,
            "replicate_policy": "ALL_REPLICATES_MUST_QUALIFY",
        },
        "minimum_work_saving": float(settings.get(
            "decision", "utility", "minimum_meaningful_work_reduction")),
        "feature_names": feature_names,
        "task_feature_registry_sha256":
            str(registry["task_feature_registry_sha256"]),
        "training_splits": list(map(str, config["training_splits"])),
        "heldout_splits": list(map(str, config["heldout_splits"])),
        "excluded_splits": list(map(str, config["excluded_splits"])),
        "max_depth": int(config["max_depth"]),
        "min_tasks_per_leaf": int(config["min_independent_tasks_per_leaf"]),
        "eligible_threshold": float(config["eligible_success_threshold"]),
        "cv_folds": int(config["cluster_cv_folds"]),
        "stability_bootstraps": int(config["stability_bootstraps"]),
        "minimum_root_stability": float(config["bootstrap_rule_stability_min"]),
    }
    body["content_sha256"] = _unsigned_sha(body)
    return body


#: Directories the runner creates only while executing treatment. ``ledger.sqlite`` is
#: deliberately absent: ``open_run_ledger`` creates that file merely by *opening* it, which
#: several read-only gates in bootstrap_and_run.sh do before any cell has run. Treating its
#: existence as treatment state made the guard fire on a ledger holding zero runs, zero work
#: items and zero attempts -- refusing a pre-registration that was in fact perfectly timed.
_TREATMENT_ARTIFACT_DIRS = (
    "schedules",
    "e2e_blocks",
    "component_forks",
    "trajectory_diagnostics",
)


def _treatment_state_exists(runs: Path) -> bool:
    """Whether the runner has begun treatment, refusing to guess when it cannot be observed.

    ``Path.exists()`` answers *False* for EACCES, and this guard runs as the steward, who has
    traverse-only access to the runner tree -- the ``runner/runs`` ACL names the evaluator, not
    the steward. So the whole pre-registration guarantee ("no analysis design is authored after
    treatment state may have been observed") silently evaluated to "no treatment state exists",
    every time, on the run host and nowhere else.

    An unobservable path is not an absent one, so a refused stat still raises. Traverse alone is
    enough to ask whether a *named* path is there, which is all this needs -- it never lists the
    directory and never reads a block.
    """
    for name in _TREATMENT_ARTIFACT_DIRS:
        path = runs / name
        try:
            path.stat()
        except FileNotFoundError:
            continue
        except PermissionError as e:
            raise ValueError(
                f"cannot determine whether treatment state exists at {path}: {e}. The "
                "pre-registration guard must not be satisfied by an unreadable path; grant "
                "this identity traverse access to the runner runs directory, or run the "
                "freeze before the runner tree is created."
            ) from e
        except OSError as e:
            raise ValueError(f"cannot stat {path} to check for treatment state: {e}") from e
        return True
    return False


def _write_once(path: Path, body: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = _load_verified_json(path, hash_field="content_sha256")
        if existing != body:
            raise ValueError(
                f"{path} already freezes a different pre-treatment analysis design")
        return
    # Exclusive creation makes "before outcomes" a property of the artifact, not a last-writer
    # convention.  A partially written file fails verification and is never silently replaced.
    with path.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.chmod(path, 0o440)


def freeze_analysis_design(settings) -> dict:
    """Write the feature registry and eligibility spec exactly once, before treatment."""
    directory = settings.path("evaluator_root") / "analysis_design"
    registry_path = directory / REGISTRY_FILENAME
    spec_path = directory / ELIGIBILITY_SPEC_FILENAME
    receipt_path = (
        settings.path("frozen_corpus_for_runner") / RUNNER_RECEIPT_FILENAME)
    evaluator_receipt_path = directory / EVALUATOR_RECEIPT_FILENAME
    if registry_path.exists() or spec_path.exists():
        if not registry_path.exists() or not spec_path.exists():
            raise ValueError("pre-treatment analysis design is only partially frozen")
        registry = _load_verified_json(
            registry_path, hash_field="content_sha256")
        spec = _load_verified_json(spec_path, hash_field="content_sha256")
        expected_registry = build_task_feature_registry(settings)
        if registry != expected_registry:
            raise ValueError(
                "existing task-feature registry differs from the current pre-treatment "
                "extractor or frozen inputs; use a new campaign data root"
            )
        expected_spec = build_eligibility_spec(settings, registry)
        if spec != expected_spec:
            raise ValueError(
                "existing eligibility spec differs from the current hash-locked design; "
                "use a new campaign data root"
            )
        if (
            str(spec.get("task_feature_registry_sha256") or "")
            != str(registry.get("task_feature_registry_sha256") or "")
        ):
            raise ValueError("frozen eligibility spec does not bind its feature registry")
        if not receipt_path.exists() or not evaluator_receipt_path.exists():
            if _treatment_state_exists(settings.path("runs")):
                raise ValueError(
                    "one or more analysis design receipts are absent after treatment state "
                    "exists")
            receipt = _analysis_design_receipt(settings, registry, spec)
            _write_once(receipt_path, receipt)
            _write_once(evaluator_receipt_path, receipt)
        else:
            receipt = _load_verified_json(
                receipt_path, hash_field="content_sha256")
            evaluator_receipt = _load_verified_json(
                evaluator_receipt_path, hash_field="content_sha256")
            if evaluator_receipt != receipt:
                raise ValueError(
                    "runner and evaluator analysis-design receipts differ")
        _verify_receipt_binding(settings, receipt, registry, spec)
        return {
            "registry": registry,
            "eligibility_spec": spec,
            "receipt": receipt,
            "registry_path": str(registry_path),
            "eligibility_spec_path": str(spec_path),
            "receipt_path": str(receipt_path),
            "evaluator_receipt_path": str(evaluator_receipt_path),
        }
    if _treatment_state_exists(settings.path("runs")):
        raise ValueError(
            "runner treatment artifacts already exist; pre-treatment analysis design cannot be "
            "authored after treatment state may have been observed")
    registry = build_task_feature_registry(settings)
    spec = build_eligibility_spec(settings, registry)
    receipt = _analysis_design_receipt(settings, registry, spec)
    _write_once(registry_path, registry)
    _write_once(spec_path, spec)
    _write_once(receipt_path, receipt)
    _write_once(evaluator_receipt_path, receipt)
    return {
        "registry": registry,
        "eligibility_spec": spec,
        "receipt": receipt,
        "registry_path": str(registry_path),
        "eligibility_spec_path": str(spec_path),
        "receipt_path": str(receipt_path),
        "evaluator_receipt_path": str(evaluator_receipt_path),
    }


def _analysis_design_receipt(
    settings, registry: Mapping[str, object], spec: Mapping[str, object]
) -> dict:
    campaign_path = settings.path("acquisition") / "campaign_acquisition.json"
    campaign = _load_verified_json(
        campaign_path, hash_field="campaign_acquisition_sha256")
    records = registry.get("records")
    if not isinstance(records, Mapping):
        raise ValueError("feature registry has no task records")
    receipt = {
        "schema_version": "pre_treatment_analysis_design_receipt_v1",
        "feature_extractor_version": FEATURE_EXTRACTOR_VERSION,
        "task_feature_registry_sha256":
            str(registry["task_feature_registry_sha256"]),
        "feature_registry_content_sha256": str(registry["content_sha256"]),
        "eligibility_spec_content_sha256": str(spec["content_sha256"]),
        "campaign_acquisition_sha256":
            str(campaign["campaign_acquisition_sha256"]),
        "sealed_registry_sha256": str(campaign["registry_sha256"]),
        # Runner-safe coordinates only.  No authored facets, truth labels, audit-only ranks or
        # eligibility feature values cross the UID boundary.
        "task_pools": [
            {
                "task_id": str(task_id),
                "split": str(record["split"]),
                "source_pool_sha256": str(record["source_pool_sha256"]),
            }
            for task_id, record in sorted(records.items())
        ],
        "decision_config_sha256": str(settings.shas["decision"]),
        "variants_config_sha256": str(settings.shas["variants"]),
        "week1_config_sha256": str(settings.shas["week1"]),
    }
    receipt["content_sha256"] = _unsigned_sha(receipt)
    return receipt


def _verify_receipt_binding(
    settings,
    receipt: Mapping[str, object],
    registry: Mapping[str, object],
    spec: Mapping[str, object],
) -> None:
    expected = _analysis_design_receipt(settings, registry, spec)
    if dict(receipt) != expected:
        raise ValueError(
            "runner analysis-design receipt does not bind the frozen evaluator artifacts")


def load_runner_analysis_design_receipt(settings) -> dict:
    """Read and verify the non-secret receipt available to preflight and the runner."""
    path = settings.path("frozen_corpus_for_runner") / RUNNER_RECEIPT_FILENAME
    receipt = _load_verified_json(path, hash_field="content_sha256")
    if receipt.get("schema_version") != "pre_treatment_analysis_design_receipt_v1":
        raise ValueError("analysis-design receipt has an unsupported schema")
    expected_configs = {
        "decision_config_sha256": settings.shas["decision"],
        "variants_config_sha256": settings.shas["variants"],
        "week1_config_sha256": settings.shas["week1"],
    }
    for field, expected in expected_configs.items():
        if str(receipt.get(field) or "") != expected:
            raise ValueError(f"analysis-design receipt {field} does not match live config")
    for field in (
        "task_feature_registry_sha256",
        "feature_registry_content_sha256",
        "eligibility_spec_content_sha256",
        "campaign_acquisition_sha256",
        "sealed_registry_sha256",
    ):
        if len(str(receipt.get(field) or "")) != 64:
            raise ValueError(f"analysis-design receipt lacks {field}")
    tasks = receipt.get("task_pools")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("analysis-design receipt has no task pool index")
    task_ids: set[str] = set()
    for row in tasks:
        task_id = str((row or {}).get("task_id") or "")
        if (
            not task_id
            or task_id in task_ids
            or not (row or {}).get("split")
            or len(str((row or {}).get("source_pool_sha256") or "")) != 64
        ):
            raise ValueError("analysis-design receipt has invalid task pool coordinates")
        task_ids.add(task_id)
    return receipt
