"""`evaluate`: score the frozen outputs, as the evaluator, once they cannot change.

Reads committed cell outputs from the runner's ledger, pairs each task's arms with that task's
TruthPacket, and writes one score file per task under the evaluator tree. Nothing here can feed
back into a treatment: it runs after the fact, under a different identity, and its outputs live
in a directory the runner cannot read.
"""

from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from ..canonical import canonical_json
from ..evaluation.citation_support import parse_citation_map
from ..bench.grading.judge_client import DeepSeekJudge
from ..evaluation.runner import (
    ArmOutput,
    EvaluationError,
    judge_policy_digest,
    score_task,
    selector_normalization_metrics,
    write_scores,
)
from ..hashing import sha256_hex
from ..object_store import CorruptObject, ObjectStore
from ..protocol import ApprovalError, verified_execution_binding
from ..providers.provider_client import PROVIDER_KEY_PLACEHOLDER
from ..scoped_paths import resolve_scoped_path
from .schedule import FROZEN_ROOT_FILENAME
from .screen import provider_client_for
from .settings import Settings

__all__ = ["evaluate_frozen", "judge_relation_via"]

_RELATION_SYSTEM = """\
You decide whether a claim is entailed by, contradicted by, or unrelated to a fact.

You are given one claim and one fact. Judge only their relationship. You do not know which
system produced the claim, and you must not guess.

Answer JSON: {"relation": "entail" | "contradict" | "unrelated" | "uncertain"}
"""

_RELATION_PROMPT = """\
Claim: {claim}

Fact: {fact}
"""


#: Versioned with the prompt bytes it hashes. judge_policy_digest covers configs/judge.yaml
#: and never covered this prompt at all, so a reworded relation instruction changed every
#: score without changing the policy sha that is supposed to identify the rater.
RELATION_PROMPT_VERSION = "relation_v1"
EVALUATION_SCOPE_FILENAME = "EVALUATION_SCOPE.json"


class _ReadOnlyLedgerView:
    """The evaluator's query-only view of the runner ledger.

    Constructing ``experiment.Ledger`` executes migrations and schema DDL, so giving that class
    a runner-owned path silently requires evaluator write access.  The frozen evaluator only
    performs one SELECT; SQLite URI ``mode=ro`` plus ``query_only`` makes that contract real.

    The ledger runs in WAL mode (``experiment/ledger.py`` sets ``journal_mode=WAL``), and a WAL
    reader classically needs *write* access to the ``-shm`` sidecar, which the evaluator does
    not have.  Verified on the pinned SQLite 3.50.4 that this is not a problem here: from a
    separate process, with ``-shm`` and ``-wal`` mode 0444, and with the WAL checkpointed away
    entirely, ``mode=ro`` opens and reads correctly -- SQLite falls back to a heap-backed shm.
    Recorded because the failure mode is version-dependent: if the pinned SQLite ever moves,
    re-check this before assuming the evaluator can still open a live runner ledger.
    """

    def __init__(self, path: Path) -> None:
        resolved = Path(path).resolve(strict=True)
        self._connection = sqlite3.connect(
            f"{resolved.as_uri()}?mode=ro",
            uri=True,
            check_same_thread=False,
            isolation_level=None,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA query_only=ON")

    @property
    def raw_connection(self) -> sqlite3.Connection:
        return self._connection

    def close(self) -> None:
        self._connection.close()


def open_run_ledger(settings: Settings):
    """Open the runner ledger without schema creation or any write capability.

    The local name is retained as a test seam, but unlike ``campaign.screen.open_run_ledger``
    this evaluator-side implementation is intentionally read-only.
    """
    return _ReadOnlyLedgerView(settings.path("runs") / "ledger.sqlite"), None


def relation_prompt_sha256() -> str:
    from ..canonical import canonical_json
    from ..hashing import sha256_hex

    return sha256_hex(canonical_json({
        "version": RELATION_PROMPT_VERSION,
        "system": _RELATION_SYSTEM,
        "user": _RELATION_PROMPT,
        "schema": {"relation": ["entail", "contradict", "unrelated", "uncertain"]},
    }))


def _first_boundary_input_receipt(events: list[dict]) -> dict:
    """Freeze the complete ordered input set at the first atomic H/C boundary.

    An H checkpoint can contain several sibling pages. Equality of page one alone does not
    establish a one-factor control: every candidate-bearing page in that first atomic batch is
    bound in order. Empty/snippet/non-attempt records are skipped until the first eligible
    boundary. Later checkpoints remain post-treatment mediated outcomes and are never compared.
    """

    def seal(body: dict) -> dict:
        body["content_sha256"] = sha256_hex(canonical_json(body))
        return body

    def malformed(event: dict, ordinal: int, kind: str, reason: str) -> dict:
        return seal(
            {
                "schema_version": "first_boundary_input_receipt_v1",
                "status": "MALFORMED",
                "reason": reason,
                "event_index": int(event.get("event_index", ordinal)),
                "event_kind": kind,
            }
        )

    def normalized_candidate(raw: object) -> tuple[dict | None, str | None]:
        if not isinstance(raw, dict):
            return None, "FIRST_BOUNDARY_RECORD_IS_NOT_AN_OBJECT"
        digest = str(raw.get("candidate_view_sha256") or "")
        offered = raw.get("offered_span_ids")
        # Explicit non-attempt, snippet, or empty-view records do not define an eligible
        # selector/compressor input. They cannot cause us to bind a later page in the same
        # otherwise-empty group as if it were the first atomic boundary.
        if offered in (None, []) and not digest:
            return None, None
        if offered == [] and digest:
            return None, None
        try:
            valid_digest = (
                len(digest) == 64
                and digest == digest.lower()
                and int(digest, 16) >= 0
            )
        except ValueError:
            valid_digest = False
        if (
            not valid_digest
            or not isinstance(offered, list)
            or not offered
            or any(not isinstance(value, str) or not value for value in offered)
            or len(offered) != len(set(offered))
        ):
            return None, "FIRST_ELIGIBLE_BOUNDARY_VIEW_IS_MALFORMED"
        occurrence_ids = raw.get("offered_source_occurrence_ids")
        if not isinstance(occurrence_ids, list) or any(
            not isinstance(value, str) or not value for value in occurrence_ids
        ):
            return None, "FIRST_ELIGIBLE_BOUNDARY_OCCURRENCE_LINEAGE_IS_MALFORMED"
        return (
            {
                "candidate_view_sha256": digest,
                "offered_span_ids": list(offered),
                "offered_source_occurrence_ids": list(occurrence_ids),
            },
            None,
        )

    for ordinal, event in enumerate(events):
        if not isinstance(event, dict):
            continue
        kind = str(event.get("kind") or "")
        if kind == "PROSE_CONTROL_OUTPUT":
            controls = event.get("control_records")
            if controls is None:
                continue
            if not isinstance(controls, list):
                return malformed(
                    event,
                    ordinal,
                    kind,
                    "FIRST_PROSE_BOUNDARY_CONTROL_RECORDS_ARE_NOT_A_LIST",
                )
            candidates: list[dict] = []
            node = str(event.get("node") or "")
            for raw in controls:
                candidate, error = normalized_candidate(raw)
                if error:
                    return malformed(event, ordinal, kind, error)
                if candidate is not None:
                    candidates.append(candidate)
                    if isinstance(raw, dict):
                        node = node or str(raw.get("node") or "")
            if not candidates:
                continue
            checkpoint = str(event.get("checkpoint") or "")
            return seal(
                {
                    "schema_version": "first_boundary_input_receipt_v1",
                    "status": "OK",
                    "node": node,
                    "candidate_input_count": len(candidates),
                    "candidate_inputs": candidates,
                    "boundary_checkpoint_hash": checkpoint,
                    "first_event_index": int(event.get("event_index", ordinal)),
                    "event_kind": kind,
                    "comparison_scope": "COMPLETE_ORDERED_FIRST_ATOMIC_H_OR_C_BOUNDARY_INPUT_SET",
                    "later_boundary_policy": "MEDIATED_E2E_OUTCOME_NOT_PAIRING_FILTER",
                }
            )
        if kind != "NODE_SELECTION":
            continue
        raw = event.get("direct_node_record") or event
        first, error = normalized_candidate(raw)
        if error:
            return malformed(event, ordinal, kind, error)
        if first is None:
            continue
        if not isinstance(raw, dict):
            return malformed(event, ordinal, kind, "FIRST_BOUNDARY_RECORD_IS_NOT_AN_OBJECT")
        checkpoint = str(
            raw.get("checkpoint_hash") or event.get("checkpoint") or ""
        )
        node = str(raw.get("node") or event.get("node") or "")
        if not checkpoint or not node:
            return malformed(
                event,
                ordinal,
                kind,
                "FIRST_ELIGIBLE_BOUNDARY_LACKS_NODE_OR_CHECKPOINT_IDENTITY",
            )
        candidates = [first]
        # Collect every candidate-bearing sibling in this exact atomic checkpoint. Ignore
        # explicit no-view records; stop at the first eligible later checkpoint.
        for later_ordinal, later in enumerate(events[ordinal + 1 :], start=ordinal + 1):
            if not isinstance(later, dict) or later.get("kind") != "NODE_SELECTION":
                continue
            later_raw = later.get("direct_node_record") or later
            if isinstance(later_raw, dict):
                later_checkpoint = str(
                    later_raw.get("checkpoint_hash") or later.get("checkpoint") or ""
                )
                later_node = str(later_raw.get("node") or later.get("node") or "")
                if (
                    (later_checkpoint and later_checkpoint != checkpoint)
                    or (later_node and later_node != node)
                ):
                    break
            candidate, later_error = normalized_candidate(later_raw)
            if later_error:
                return malformed(later, later_ordinal, "NODE_SELECTION", later_error)
            if candidate is None:
                continue
            if not isinstance(later_raw, dict):
                return malformed(
                    later,
                    later_ordinal,
                    "NODE_SELECTION",
                    "FIRST_BOUNDARY_RECORD_IS_NOT_AN_OBJECT",
                )
            later_checkpoint = str(
                later_raw.get("checkpoint_hash") or later.get("checkpoint") or ""
            )
            later_node = str(later_raw.get("node") or later.get("node") or "")
            if later_checkpoint != checkpoint or later_node != node:
                break
            candidates.append(candidate)
        return seal(
            {
                "schema_version": "first_boundary_input_receipt_v1",
                "status": "OK",
                "node": node,
                "candidate_input_count": len(candidates),
                "candidate_inputs": candidates,
                "boundary_checkpoint_hash": checkpoint,
                "first_event_index": int(event.get("event_index", ordinal)),
                "event_kind": kind,
                "comparison_scope": "COMPLETE_ORDERED_FIRST_ATOMIC_H_OR_C_BOUNDARY_INPUT_SET",
                "later_boundary_policy": "MEDIATED_E2E_OUTCOME_NOT_PAIRING_FILTER",
            }
        )
    return seal(
        {
            "schema_version": "first_boundary_input_receipt_v1",
            "status": "MISSING",
            "reason": "NO_ELIGIBLE_ATOMIC_BOUNDARY_INPUT_SET_IN_FROZEN_EVENT_STREAM",
            "comparison_scope": "COMPLETE_ORDERED_FIRST_ATOMIC_H_OR_C_BOUNDARY_INPUT_SET",
            "later_boundary_policy": "MEDIATED_E2E_OUTCOME_NOT_PAIRING_FILTER",
        }
    )


def _prose_control_records(events: list[dict]) -> tuple[dict, ...]:
    """Freeze every prose attempt exactly once, retaining event/node provenance."""

    records: list[dict] = []
    for ordinal, event in enumerate(events):
        if not isinstance(event, dict) or event.get("kind") != "PROSE_CONTROL_OUTPUT":
            continue
        raw_records = event.get("control_records")
        if not isinstance(raw_records, list):
            raise EvaluationError(
                f"PROSE_CONTROL_OUTPUT event {ordinal} has no control-record list"
            )
        event_node = str(event.get("node") or "").upper()
        if event_node.startswith("H"):
            event_node = "H"
        elif event_node.startswith("C"):
            event_node = "C"
        else:
            raise EvaluationError(
                f"PROSE_CONTROL_OUTPUT event {ordinal} has no valid H/C node"
            )
        checkpoint = str(event.get("checkpoint") or "")
        for index, raw in enumerate(raw_records):
            if not isinstance(raw, dict):
                raise EvaluationError(
                    f"PROSE_CONTROL_OUTPUT event {ordinal} record {index} is not an object"
                )
            recorded_node = str(raw.get("node") or "").upper()
            if recorded_node and not recorded_node.startswith(event_node):
                raise EvaluationError(
                    f"PROSE_CONTROL_OUTPUT event {ordinal} record {index} changes its node"
                )
            item = dict(raw)
            item["node"] = event_node
            item["checkpoint_hash"] = str(
                item.get("checkpoint_hash") or checkpoint
            )
            item["event_index"] = int(event.get("event_index", ordinal))
            records.append(item)
    return tuple(records)


def _prose_expected_nodes(variant_id: str, registry: dict) -> tuple[str, ...]:
    nodes: list[str] = []
    for part in str(variant_id).split("+"):
        spec = registry.get(part)
        if spec is None or spec.publication_path != "DIRECT_PROSE":
            continue
        node = "H" if spec.node == "WEBPAGE_P1" else "C"
        if node not in nodes:
            nodes.append(node)
    return tuple(nodes)


def judge_relation_via(judge: DeepSeekJudge, loop=None, *, observed=None):
    """A blind relation classifier. Uncertainty stays uncertain rather than becoming a pass.

    The synchronous signature is what ``build_assessment`` takes, and the judge is async, so
    the call is handed to the caller's running loop from the worker thread the scoring runs
    on. It used to be ``get_event_loop().run_until_complete`` executed *inside* that same
    loop, which raises "this event loop is already running" -- so ``shapeflow-p1 evaluate``
    could not score a single claim, and no test covered it.
    """
    import asyncio

    def relate(claim: str, fact: str) -> str:
        coro = judge.judge(_RELATION_SYSTEM,
                           _RELATION_PROMPT.format(claim=claim, fact=fact))
        if loop is None:                       # already off-loop (tests, sync callers)
            response = asyncio.run(coro)
        else:
            response = asyncio.run_coroutine_threadsafe(coro, loop).result()
        if observed is not None:
            # Recorded per judgment: judge_policy_digest was computed once, before the loop,
            # with requested_model == returned_model by construction, so a mid-campaign model
            # change was invisible in the scores.
            observed.append({
                "requested_model": response.requested_model,
                "returned_model": response.returned_model,
                "system_fingerprint": response.system_fingerprint,
            })
        relation = str(response.data.get("relation", "uncertain"))
        return relation if relation in ("entail", "contradict", "unrelated") else "uncertain"

    return relate


async def evaluate_frozen(settings: Settings, *, repo: Path,
                          task_limit: int | None = None,
                          run_id: str,
                          phase_id: str,
                          frozen_blocks_dir: Path,
                          execution_binding_sha256: str,
                          protocol_document_sha256: str) -> dict:
    """Score the frozen outputs of one run and phase.

    Scoped on purpose. It used to score every COMMITTED row in the ledger regardless of which
    run or phase produced it, so a diagnostic and a screen ended up in one pooled number, and
    a block that was never frozen -- half a paired comparison -- was scored as if it were.
    """
    import asyncio

    try:
        binding = verified_execution_binding(
            repo, expected_digest=execution_binding_sha256)
    except ApprovalError as exc:
        raise EvaluationError(
            f"cannot evaluate outside the approved execution binding: {exc}") from exc
    if protocol_document_sha256 != binding.protocol_sha:
        raise EvaluationError(
            "protocol_document_sha256 does not match the verified live approval binding")
    scores_dir = resolve_scoped_path(
        settings.path("judgments"), run_id=run_id, phase_id=phase_id)
    if frozen_blocks_dir is None:
        raise ValueError(
            "evaluate requires a frozen block directory; pooled ledger scans are forbidden"
        )
    if task_limit is not None:
        raise ValueError(
            "task_limit would create a partial all-offered ITT result; use a separately "
            "identified diagnostic run instead"
        )
    scope = _load_frozen_scope(
        Path(frozen_blocks_dir),
        run_id=run_id,
        phase_id=phase_id,
        execution_binding_sha256=binding.digest,
        protocol_document_sha256=binding.protocol_sha,
    )
    from ..strategies.factory import load_registry

    variant_registry = load_registry(Path(repo) / "configs")

    loop = asyncio.get_running_loop()
    ledger, _store = open_run_ledger(settings)
    store = ObjectStore(settings.path("object_store"))
    client = provider_client_for(settings, "evaluator")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_REPORT"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
        max_retries=settings.judge_max_retries(),
        sampling=settings.judge_sampling(),
    )

    rows = ledger.raw_connection.execute(
        "SELECT w.*, a.run_id, a.state AS attempt_state, a.result_object_ref,"
        " a.attempt_ordinal, a.error_class FROM work_items w"
        " JOIN attempts a ON a.work_key=w.work_key"
        " WHERE w.phase_id=? AND a.run_id=?"
        " ORDER BY w.work_key, a.attempt_ordinal",
        (phase_id, run_id),
    ).fetchall()
    attempts: dict[tuple[str, str, str, str], object] = {}
    for row in rows:
        if str(row["protocol_sha"] or "") != scope.execution_binding_sha256:
            raise EvaluationError(
                f"ledger row {row['work_key']!r} belongs to execution binding "
                f"{row['protocol_sha']!r}, not {scope.execution_binding_sha256!r}"
            )
        key = (row["checkpoint_hash"], row["task_id"], row["arm_id"], row["replicate_id"])
        attempts[key] = row  # ordered: latest attempt wins, deterministically

    by_block: dict[tuple[str, str, str], list[tuple[_FrozenCell, object, dict]]] = {}
    for cell in scope.cells:
        key = (cell.block_id, cell.task_id, cell.arm_id, cell.replicate_id)
        row = attempts.get(key)
        if row is None:
            raise EvaluationError(
                f"frozen assignment {key} has no attempt in run={run_id} phase={phase_id}; "
                "refusing to turn an unrelated/missing row into an outcome"
            )
        expected_variant = f"{cell.page_variant}+{cell.close_variant}"
        if str(row["variant_id"] or "") != expected_variant:
            raise EvaluationError(
                f"frozen assignment {key} expects variant {expected_variant!r}, "
                f"ledger records {row['variant_id']!r}"
            )
        ledger_state = str(row["attempt_state"] or "")
        if cell.state and cell.state != ledger_state:
            raise EvaluationError(
                f"frozen assignment {key} says {cell.state}, ledger says {ledger_state}"
            )
        ledger_ref = str(row["result_object_ref"] or "")
        if cell.output_ref and ledger_ref and cell.output_ref != ledger_ref:
            raise EvaluationError(
                f"frozen assignment {key} points at {cell.output_ref}, ledger at {ledger_ref}"
            )
        ref = cell.output_ref or ledger_ref
        if not ref:
            raise EvaluationError(
                f"frozen assignment {key} has no immutable output/tombstone reference")
        if not store.verify(ref):
            raise EvaluationError(
                f"frozen assignment {key} output {ref} is unavailable or corrupt; "
                "artifact loss is an evaluation structural failure, not model harm")
        try:
            record = json.loads(store.get_bytes(ref).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise EvaluationError(
                f"frozen assignment {key} output {ref} is not valid JSON; "
                "measurement corruption cannot be scored as an arm failure") from exc
        if not isinstance(record, dict):
            raise EvaluationError(
                f"frozen assignment {key} output {ref} is not an object")
        by_block.setdefault(
            (cell.block_id, cell.task_id, cell.replicate_id), []
        ).append((cell, row, record))

    truth_dir = settings.path("truth_packets")
    scored: list[str] = []
    score_records: list[dict] = []
    skipped: list[str] = []
    observed_all: list[dict] = []

    for (block_id, task_id, replicate_id), scoped in sorted(by_block.items()):
        truth_path = truth_dir / f"{task_id}.json"
        if not truth_path.exists():
            ledger.close()
            raise EvaluationError(
                f"{task_id} has no truth packet; dropping its offered block would bias ITT"
            )
        truth = _load_verified_truth_artifact(
            truth_path, settings=settings, expected_task_id=task_id)
        support_index = _verified_support_index(
            truth,
            truth_path,
            settings=settings,
            task_id=task_id,
            pool=_frozen_pool_for(settings, task_id),
        )
        outputs = []
        for cell, row, record in scoped:
            _assert_record_coordinates(
                record,
                cell=cell,
                run_id=run_id,
                phase_id=phase_id,
                execution_binding_sha256=scope.execution_binding_sha256,
                protocol_document_sha256=scope.protocol_document_sha256,
            )
            ledger_state = str(row["attempt_state"])
            assignment_state = "COMMITTED" if ledger_state == "COMMITTED" else ledger_state
            traces = record.get("direct_node_records")
            if traces is None:
                traces = [
                    {k: v for k, v in event.items() if k != "kind"}
                    for event in (record.get("events") or [])
                    if event.get("kind") == "NODE_SELECTION"
                ]
            counts = record.get("counts") or {}
            raw_events = record.get("events")
            if not isinstance(raw_events, list):
                if assignment_state == "COMMITTED":
                    raise EvaluationError(
                        f"frozen assignment {(block_id, task_id, cell.arm_id, replicate_id)} "
                        "has no ordered event stream; trajectory measurement is structurally "
                        "unavailable"
                    )
                trajectory_metrics = {
                    "schema_version": "frozen_trajectory_metrics_v1",
                    "status": "UNAVAILABLE_TERMINAL_FAILURE",
                    "reason": "terminal failure has no committed frozen event stream",
                }
                raw_events = []
            else:
                from ..evaluation.trajectory_eval import summarize_frozen_events

                try:
                    trajectory_metrics = summarize_frozen_events(raw_events)
                except ValueError as exc:
                    raise EvaluationError(
                        f"frozen assignment "
                        f"{(block_id, task_id, cell.arm_id, replicate_id)} has an invalid "
                        f"trajectory event stream: {exc}"
                    ) from exc
            query_events = [
                event for event in raw_events
                if isinstance(event, dict) and event.get("kind") == "SEARCH_QUERY"
            ]
            if any("source_occurrence_ids" not in event for event in query_events):
                raise EvaluationError(
                    f"frozen assignment {(block_id, task_id, cell.arm_id, replicate_id)} "
                    "has search events without source-occurrence lineage; citation measurement "
                    "is structurally unavailable, not an arm failure"
                )
            retrieved_occurrence_ids = tuple(sorted({
                str(occurrence_id)
                for event in query_events
                for occurrence_id in (event.get("source_occurrence_ids") or ())
                if str(occurrence_id)
            }))
            fell_back = bool(
                record.get("fell_back")
                or counts.get("page_fallbacks")
                or any(t.get("fell_back") for t in (traces or []))
            )
            observed_variant = str(
                record.get("variant_id")
                or (row["variant_id"] if row is not None else "")
            )
            outputs.append(ArmOutput(
                task_id=task_id,
                arm_id=cell.arm_id,
                variant_id=observed_variant,
                replicate_id=replicate_id,
                final_report=str(record.get("final_report") or ""),
                frozen=True,  # the assignment, including its failure, is frozen
                terminal_failure=bool(record.get("error")) or assignment_state != "COMMITTED",
                block_id=block_id,
                assignment_state=assignment_state,
                fell_back=fell_back,
                direct_node_records=tuple(traces or ()),
                prose_control_records=_prose_control_records(raw_events),
                prose_control_expected_nodes=_prose_expected_nodes(
                    observed_variant, variant_registry
                ),
                first_boundary_input=_first_boundary_input_receipt(raw_events),
                work_summary=dict(record.get("work_summary") or {}),
                retrieved_source_occurrence_ids=retrieved_occurrence_ids,
                trajectory_metrics=trajectory_metrics,
            ))
        observed: list[dict] = []
        relate = judge_relation_via(judge, loop, observed=observed)
        pool = _frozen_pool_for(settings, task_id)
        supports = _citation_supports_for(settings, task_id, pool, truth, relate)

        score = await asyncio.to_thread(
            score_task,
            truth_body=truth["packet"], outputs=outputs, atom_texts=truth["atom_texts"],
            judge_relation=relate,
            citation_supports=supports,
            # Filled with the actually observed model/fingerprint immediately below.
            judge_policy_sha256="PENDING_OBSERVED_JUDGE_POLICY",
            claim_scope=settings.claim_scope,
            run_id=run_id, phase_id=phase_id, block_id=block_id,
            replicate_id=replicate_id,
            execution_binding_sha256=scope.execution_binding_sha256,
            protocol_document_sha256=scope.protocol_document_sha256,
            truth_chunker=str(
                (truth.get("provenance") or {}).get("truth_chunker")
                or "markdown_structure_v1"),
            atom_support_index=support_index,
        )
        await asyncio.to_thread(
            _add_direct_checkpoint_metrics,
            settings, score, outputs, truth, relate,
        )
        score.judge_policy_sha256 = _policy_sha(settings, observed)
        score.judge_provenance = {
            "judgments": len(observed),
            "requested_model": settings.judge_model(),
            "returned_models": sorted({
                o["returned_model"] for o in observed if o.get("returned_model")
            }),
            "system_fingerprints": sorted({
                o["system_fingerprint"] for o in observed if o.get("system_fingerprint")
            }),
        }
        frozen_block = scope.blocks[block_id]
        for arm_score in score.per_arm.values():
            arm_score["frozen_scope"] = {
                "freeze_root_sha256": scope.freeze_root_sha256,
                "schedule_sha256": scope.schedule_sha256,
                "block_freeze_sha256": frozen_block.freeze_sha256,
                "block_digest": frozen_block.block_digest,
                "valid_for_paired_estimate":
                    frozen_block.valid_for_paired_estimate,
                "invalid_reason": frozen_block.invalid_reason,
                "engine_epochs": list(frozen_block.engine_epochs),
                "engine_epoch_by_arm": dict(
                    sorted(frozen_block.engine_epoch_by_arm.items())),
            }
        observed_all.extend(observed)
        score_sha = write_scores(score, scores_dir / f"{block_id}.json")
        _write_judge_provenance(scores_dir, block_id, settings, observed, supports,
                                task_id=task_id, run_id=run_id, phase_id=phase_id)
        scored.append(block_id)
        score_records.append({
            "block_id": block_id,
            "task_id": task_id,
            "replicate_id": replicate_id,
            "execution_binding_sha256": scope.execution_binding_sha256,
            "protocol_document_sha256": scope.protocol_document_sha256,
            "score_content_sha256": score_sha,
            "block_freeze_sha256": frozen_block.freeze_sha256,
            "block_digest": frozen_block.block_digest,
            "valid_for_paired_estimate": frozen_block.valid_for_paired_estimate,
            "invalid_reason": frozen_block.invalid_reason,
            "engine_epochs": list(frozen_block.engine_epochs),
            "engine_epoch_by_arm": dict(
                sorted(frozen_block.engine_epoch_by_arm.items())),
        })

    evaluation_scope = _write_evaluation_scope_receipt(
        scores_dir,
        scope=scope,
        run_id=run_id,
        phase_id=phase_id,
        score_records=score_records,
    )
    ledger.close()
    return {"scored": scored, "skipped": skipped, "claim_scope": settings.claim_scope,
            "run_id": run_id, "phase_id": phase_id, "blocks_offered": len(by_block),
            "execution_binding_sha256": scope.execution_binding_sha256,
            "protocol_document_sha256": scope.protocol_document_sha256,
            "freeze_root_sha256": scope.freeze_root_sha256,
            "schedule_sha256": scope.schedule_sha256,
            "evaluation_scope_sha256": evaluation_scope,
            "judge_policy_sha256": _policy_sha(settings, observed_all),
            "relation_prompt_sha256": relation_prompt_sha256()}


@dataclass(frozen=True)
class _FrozenCell:
    block_id: str
    task_id: str
    replicate_id: str
    arm_id: str
    page_variant: str
    close_variant: str
    seed: int
    order_index: int
    engine_epoch: str
    state: str
    output_ref: str


@dataclass(frozen=True)
class _FrozenBlock:
    block_id: str
    task_id: str
    replicate_id: str
    block_digest: str
    freeze_sha256: str
    valid_for_paired_estimate: bool
    invalid_reason: str
    engine_epochs: tuple[str, ...]
    engine_epoch_by_arm: dict[str, str]


@dataclass(frozen=True)
class _FrozenScope:
    run_id: str
    phase_id: str
    execution_binding_sha256: str
    protocol_document_sha256: str
    schedule_sha256: str
    freeze_root_sha256: str
    analysis_design_receipt_sha256: str
    task_feature_registry_sha256: str
    eligibility_spec_content_sha256: str
    cells: tuple[_FrozenCell, ...]
    blocks: dict[str, _FrozenBlock]


def _load_verified_truth_artifact(
    path: Path, *, settings: Settings, expected_task_id: str
) -> dict:
    """Verify the complete answer-key artifact, not only its optional support index."""
    from jsonschema import Draft202012Validator

    from ..canonical import canonical_json
    from ..hashing import sha256_hex

    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid truth artifact {path}: {exc}") from exc
    recorded_wrapper = str(body.get("content_sha256") or "")
    actual_wrapper = sha256_hex(canonical_json(
        {key: value for key, value in body.items() if key != "content_sha256"}))
    if recorded_wrapper != actual_wrapper:
        raise EvaluationError(
            f"{path} truth wrapper was edited: records {recorded_wrapper}, "
            f"hashes to {actual_wrapper}"
        )

    packet = body.get("packet")
    if not isinstance(packet, dict):
        raise EvaluationError(f"{path} has no truth packet object")
    schema = json.loads(
        (settings.repo / "schemas" / "truth_packet.schema.json")
        .read_text(encoding="utf-8")
    )
    errors = sorted(Draft202012Validator(schema).iter_errors(packet), key=lambda e: e.json_path)
    if errors:
        raise EvaluationError(
            f"{path} violates truth_packet schema at {errors[0].json_path}: "
            f"{errors[0].message}"
        )
    recorded_packet = str(packet.get("content_sha256") or "")
    unsigned_packet = {**packet, "content_sha256": ""}
    actual_packet = sha256_hex(canonical_json(unsigned_packet))
    if recorded_packet != actual_packet:
        raise EvaluationError(
            f"{path} packet was edited: records {recorded_packet}, hashes to {actual_packet}"
        )
    if str(packet.get("task_id") or "") != expected_task_id:
        raise EvaluationError(
            f"{path} belongs to task {packet.get('task_id')!r}, "
            f"not {expected_task_id!r}"
        )
    atoms = list(packet.get("atomic_evidence") or ())
    accepted_ids = {str(atom["atom_id"]) for atom in atoms}
    if len(accepted_ids) != len(atoms):
        raise EvaluationError(f"{path} contains duplicate truth atom ids")
    atom_texts = body.get("atom_texts")
    if not isinstance(atom_texts, dict) or set(map(str, atom_texts)) != accepted_ids:
        raise EvaluationError(
            f"{path} atom_texts keys do not exactly match accepted truth atoms")
    if any(not isinstance(text, str) or not text.strip() for text in atom_texts.values()):
        raise EvaluationError(f"{path} contains an empty/non-string accepted atom text")

    provenance = body.get("provenance")
    if not isinstance(provenance, dict):
        raise EvaluationError(f"{path} has no truth provenance")
    if provenance.get("config_sha256s") != dict(sorted(settings.shas.items())):
        raise EvaluationError(f"{path} was authored under a different frozen configuration")
    if str(provenance.get("claim_scope") or "") != settings.claim_scope:
        raise EvaluationError(f"{path} claim scope differs from the evaluation campaign")
    pool = _frozen_pool_for(settings, expected_task_id)
    if (
        not pool
        or str(pool.get("task_id") or "") != expected_task_id
        or str(provenance.get("source_pool_sha256") or "")
        != str(pool.get("pool_sha256") or "")
    ):
        raise EvaluationError(f"{path} is not bound to the current frozen source pool")

    task_path = (
        settings.path("evaluator_root") / "tasks" / f"{expected_task_id}.json")
    try:
        task_view = json.loads(task_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid sealed evaluator task view {task_path}: {exc}") from exc
    if (
        not isinstance(task_view, dict)
        or str(task_view.get("task_id") or "") != expected_task_id
        or not isinstance(task_view.get("original_question"), str)
        or not str(task_view.get("original_question") or "").strip()
        or not isinstance(task_view.get("authored_facets"), list)
        or any(not isinstance(facet, str) for facet in task_view.get("authored_facets") or ())
    ):
        raise EvaluationError(
            f"{task_path} lacks the sealed question/facet coordinates for {expected_task_id}")
    expected_facets = list(task_view["authored_facets"])
    if list(packet.get("required_facets") or ()) != expected_facets:
        raise EvaluationError(
            f"{path} required facets differ from the sealed evaluator task view")
    task_binding = sha256_hex(canonical_json({
        "task_id": expected_task_id,
        "question": task_view["original_question"],
        "required_facets": expected_facets,
    }))
    if str(provenance.get("task_question_facets_sha256") or "") != task_binding:
        raise EvaluationError(
            f"{path} question/facet provenance differs from the sealed evaluator task view")

    from ..acquire.manifest import AcquisitionIntegrityError, load_task_manifest

    acquisition_path = settings.path("acquisition") / f"{expected_task_id}.json"
    try:
        acquisition = load_task_manifest(acquisition_path)
    except (AcquisitionIntegrityError, OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(
            f"invalid frozen acquisition manifest {acquisition_path}: {exc}") from exc
    if str(acquisition.get("task_id") or "") != expected_task_id:
        raise EvaluationError(f"{acquisition_path} belongs to a different task")
    if (
        str(task_view.get("acquisition_spec_sha256") or "")
        != str(acquisition.get("acquisition_spec_sha256") or "")
    ):
        raise EvaluationError(
            f"{path} task view and acquisition manifest disagree on acquisition spec")
    acquisition_digest = str(acquisition.get("acquisition_digest") or "")
    if str(provenance.get("acquisition_digest") or "") != acquisition_digest:
        raise EvaluationError(
            f"{path} acquisition provenance differs from the frozen manifest")

    raw_queries = acquisition.get("queries")
    if not isinstance(raw_queries, list) or any(
        not isinstance(query, dict) for query in raw_queries
    ):
        raise EvaluationError(f"{acquisition_path} has a malformed query-attempt list")
    attempts = [
        {
            "query_attempt_id": str(query.get("query_snapshot_id") or ""),
            "query": str(query.get("query_text") or ""),
            "status": str(query.get("status") or ""),
        }
        for query in raw_queries
    ]
    attempt_ids = [attempt["query_attempt_id"] for attempt in attempts]
    if any(not attempt_id for attempt_id in attempt_ids) or len(set(attempt_ids)) != len(
        attempt_ids
    ):
        raise EvaluationError(
            f"{acquisition_path} has missing or duplicate query-attempt identities")
    if any(
        attempt["status"]
        not in {"SUCCESS", "EMPTY", "FAILED", "TIMEOUT", "BLOCKED_BUDGET"}
        for attempt in attempts
    ):
        raise EvaluationError(f"{acquisition_path} has an invalid query-attempt status")
    attempts_sha = sha256_hex(canonical_json(attempts))
    if str(provenance.get("query_attempts_sha256") or "") != attempts_sha:
        raise EvaluationError(
            f"{path} query-attempt provenance differs from the frozen acquisition")
    attempt_by_id = {attempt["query_attempt_id"]: attempt for attempt in attempts}
    for gap in packet.get("known_gaps") or ():
        frozen = attempt_by_id.get(str(gap.get("query_attempt_id") or ""))
        if (
            frozen is None
            or str(gap.get("query_text") or "") != frozen["query"]
            or str(gap.get("status") or "") != frozen["status"]
            or frozen["status"] not in {"FAILED", "TIMEOUT", "BLOCKED_BUDGET"}
        ):
            raise EvaluationError(
                f"{path} contains a known gap outside the frozen query-attempt universe")
    for negative in packet.get("negative_evidence") or ():
        if str(negative.get("query_attempt_id") or "") not in attempt_by_id:
            raise EvaluationError(
                f"{path} contains negative evidence outside the frozen query-attempt universe")

    from .truth import TRUTH_PROMPT_VERSION, truth_prompt_sha256

    if (
        str(provenance.get("prompt_version") or "") != TRUTH_PROMPT_VERSION
        or str(provenance.get("prompt_sha256") or "") != truth_prompt_sha256()
    ):
        raise EvaluationError(f"{path} was authored with a different truth prompt")
    expected_source_binding = sha256_hex(canonical_json({
        "task_question_facets_sha256": task_binding,
        "source_pool_sha256": str(pool["pool_sha256"]),
        "acquisition_digest": acquisition_digest,
        "query_attempts_sha256": attempts_sha,
    }))
    if str(provenance.get("truth_source_binding_sha256") or "") != expected_source_binding:
        raise EvaluationError(
            f"{path} truth-source binding does not match its sealed inputs")

    _verify_runner_pool_matches_acquisition(
        pool, acquisition, path=path, task_id=expected_task_id)
    return body


def _verify_runner_pool_matches_acquisition(
    pool: dict, acquisition: dict, *, path: Path, task_id: str
) -> None:
    """Prove the runner-visible pool is the projection of the sealed acquisition."""
    raw_manifest_occurrences = acquisition.get("occurrences")
    if not isinstance(raw_manifest_occurrences, list) or any(
        not isinstance(occurrence, dict) for occurrence in raw_manifest_occurrences
    ):
        raise EvaluationError(f"{path} acquisition has a malformed occurrence universe")
    manifest_occurrences = [
        occurrence
        for occurrence in raw_manifest_occurrences
        if str(occurrence.get("visibility") or "") == "VENDOR_VISIBLE"
    ]
    visible_orders = [
        occurrence.get("vendor_visible_order") for occurrence in manifest_occurrences
    ]
    if (
        any(
            not isinstance(order, int) or isinstance(order, bool) or order < 0
            for order in visible_orders
        )
        or sorted(visible_orders) != list(range(len(visible_orders)))
    ):
        raise EvaluationError(
            f"{path} acquisition has an invalid vendor-visible occurrence order")
    manifest_occurrences.sort(key=lambda occurrence: occurrence["vendor_visible_order"])
    occurrence_fields = (
        "occurrence_id", "url", "title", "content_hash", "vendor_visible_order")
    expected_occurrences = [
        {field: occurrence.get(field) for field in occurrence_fields}
        for occurrence in manifest_occurrences
    ]
    actual_occurrences = [
        {field: occurrence.get(field) for field in occurrence_fields}
        for occurrence in (pool.get("occurrences") or ())
        if isinstance(occurrence, dict)
    ]
    if (
        len(actual_occurrences) != len(pool.get("occurrences") or ())
        or actual_occurrences != expected_occurrences
    ):
        raise EvaluationError(
            f"{path} source pool is not the vendor-visible projection of "
            f"{task_id}'s acquisition")

    snapshot_fields = (
        "object_ref", "byte_len", "raw_content_format",
        "normalization_version", "fetched_at_utc",
    )
    manifest_snapshots: dict[str, dict] = {}
    for snapshot in acquisition.get("snapshots") or ():
        if not isinstance(snapshot, dict):
            raise EvaluationError(
                f"{path} acquisition contains a malformed source snapshot")
        content_hash = str(snapshot.get("content_hash") or "")
        if not content_hash or content_hash in manifest_snapshots:
            raise EvaluationError(
                f"{path} acquisition contains a missing/duplicate source snapshot identity")
        manifest_snapshots[content_hash] = {
            field: snapshot.get(field) for field in snapshot_fields
        }
    pool_snapshots = pool.get("snapshots")
    if not isinstance(pool_snapshots, dict):
        raise EvaluationError(f"{path} frozen source pool has no snapshot mapping")
    actual_snapshots = {
        str(content_hash): {
            field: snapshot.get(field) for field in snapshot_fields
        }
        for content_hash, snapshot in pool_snapshots.items()
        if isinstance(snapshot, dict)
    }
    if len(actual_snapshots) != len(pool_snapshots) or actual_snapshots != manifest_snapshots:
        raise EvaluationError(
            f"{path} source-pool snapshots differ from {task_id}'s acquisition")


def _verified_support_index(
    truth: dict,
    path: Path,
    *,
    settings: Settings,
    task_id: str,
    pool: dict,
) -> dict:
    index = truth.get("atom_support_index")
    if not isinstance(index, dict):
        raise EvaluationError(f"{path} atom_support_index is absent or not an object")
    from ..canonical import canonical_json
    from ..hashing import sha256_hex

    recorded = str(index.get("content_sha256") or "")
    actual = sha256_hex(canonical_json(
        {k: v for k, v in index.items() if k != "content_sha256"}))
    if recorded != actual:
        raise EvaluationError(
            f"{path} atom_support_index was edited: records {recorded}, hashes to {actual}")
    provenance = truth.get("provenance") or {}
    if str(provenance.get("h_atom_support_index_sha256") or "") != recorded:
        raise EvaluationError(
            f"{path} atom_support_index does not match its provenance pointer")

    packet = truth.get("packet") or {}
    atoms = list(packet.get("atomic_evidence") or ())
    atom_ids = {str(atom.get("atom_id") or "") for atom in atoms}
    if not atom_ids or "" in atom_ids or len(atom_ids) != len(atoms):
        raise EvaluationError(f"{path} support index has no unambiguous truth-atom universe")

    pool_occurrences = list(pool.get("occurrences") or ())
    occurrence_id_list = [
        str(occurrence.get("occurrence_id") or "")
        for occurrence in pool_occurrences
        if isinstance(occurrence, dict)
    ]
    occurrence_ids = set(occurrence_id_list)
    if (
        len(occurrence_id_list) != len(pool_occurrences)
        or "" in occurrence_ids
        or len(occurrence_ids) != len(occurrence_id_list)
    ):
        raise EvaluationError(
            f"{path} source pool contains a malformed/missing/duplicate occurrence identity")

    try:
        from .truth import _excerpt_spans, _h_candidate_spans

        truth_spans = _excerpt_spans(settings, task_id)
        candidate_spans = _h_candidate_spans(settings, task_id)
    except Exception as exc:
        raise EvaluationError(
            f"{path} support-index universe cannot be reconstructed: "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    truth_by_id: dict[str, dict] = {}
    for span in truth_spans:
        span_id = str(span.get("span_id") or "")
        span_occurrences = set(map(str, span.get("source_occurrence_ids") or ()))
        if (
            not span_id
            or span_id in truth_by_id
            or not span_occurrences
            or not span_occurrences.issubset(occurrence_ids)
        ):
            raise EvaluationError(
                f"{path} reconstructed an invalid/duplicate truth span")
        truth_by_id[span_id] = span

    truth_supports: dict[str, set[str]] = {}
    expected_prechunk: dict[str, list[str]] = {}
    vendor_limit = int(settings.get("week1", "odr", "max_content_length"))
    for atom in atoms:
        atom_id = str(atom["atom_id"])
        span_ids = list(map(str, atom.get("supporting_span_ids") or ()))
        if not span_ids or len(set(span_ids)) != len(span_ids):
            raise EvaluationError(
                f"{path} atom {atom_id} has missing/duplicate supporting spans")
        unknown = set(span_ids) - set(truth_by_id)
        if unknown:
            raise EvaluationError(
                f"{path} atom {atom_id} names spans outside the frozen truth universe")
        truth_supports[atom_id] = set(span_ids)
        prechunk: set[str] = set()
        for span_id in span_ids:
            span = truth_by_id[span_id]
            if int(span["char_end"]) <= vendor_limit:
                prechunk.update(map(str, span.get("source_occurrence_ids") or ()))
        expected_prechunk[atom_id] = sorted(prechunk)

    raw_prechunk = index.get("prechunk_atom_occurrence_ids")
    if not isinstance(raw_prechunk, dict) or set(map(str, raw_prechunk)) != atom_ids:
        raise EvaluationError(
            f"{path} support index atom universe differs from the truth packet")
    observed_prechunk: dict[str, list[str]] = {}
    for atom_id, values in raw_prechunk.items():
        if not isinstance(values, list):
            raise EvaluationError(
                f"{path} support index has a non-list prechunk occurrence set")
        normalized = list(map(str, values))
        if len(set(normalized)) != len(normalized) or not set(normalized).issubset(
            occurrence_ids
        ):
            raise EvaluationError(
                f"{path} support index contains duplicate/unknown prechunk occurrences")
        observed_prechunk[str(atom_id)] = normalized
    if observed_prechunk != expected_prechunk:
        raise EvaluationError(
            f"{path} support index prechunk occurrences do not reconstruct from truth spans")

    expected_candidate_occurrences: dict[str, dict[str, list[str]]] = {}
    candidate_by_chunker: dict[str, dict[str, dict]] = {}
    for chunker, spans in candidate_spans.items():
        by_id: dict[str, dict] = {}
        occurrence_map: dict[str, list[str]] = {}
        for span in spans:
            span_id = str(span.get("span_id") or "")
            span_occurrences = list(map(
                str, span.get("source_occurrence_ids") or ()))
            if (
                not span_id
                or span_id in by_id
                or not span_occurrences
                or len(set(span_occurrences)) != len(span_occurrences)
                or not set(span_occurrences).issubset(occurrence_ids)
            ):
                raise EvaluationError(
                    f"{path} reconstructed an invalid/duplicate {chunker} candidate span")
            by_id[span_id] = span
            occurrence_map[span_id] = sorted(span_occurrences)
        candidate_by_chunker[str(chunker)] = by_id
        expected_candidate_occurrences[str(chunker)] = occurrence_map

    raw_candidate_occurrences = index.get("candidate_span_occurrence_ids")
    if not isinstance(raw_candidate_occurrences, dict):
        raise EvaluationError(
            f"{path} support index has no candidate-span occurrence mapping")
    observed_candidate_occurrences: dict[str, dict[str, list[str]]] = {}
    for chunker, mapping in raw_candidate_occurrences.items():
        if not isinstance(mapping, dict):
            raise EvaluationError(
                f"{path} support index candidate mapping for {chunker} is malformed")
        normalized_mapping: dict[str, list[str]] = {}
        for span_id, values in mapping.items():
            if not isinstance(values, list):
                raise EvaluationError(
                    f"{path} support index occurrence list for {span_id} is malformed")
            normalized = list(map(str, values))
            if len(set(normalized)) != len(normalized):
                raise EvaluationError(
                    f"{path} support index duplicates an occurrence for {span_id}")
            normalized_mapping[str(span_id)] = sorted(normalized)
        observed_candidate_occurrences[str(chunker)] = normalized_mapping
    if observed_candidate_occurrences != expected_candidate_occurrences:
        raise EvaluationError(
            f"{path} support-index candidate span/occurrence universe does not reconstruct")

    chunkers = index.get("chunkers")
    if not isinstance(chunkers, dict) or set(map(str, chunkers)) != set(candidate_by_chunker):
        raise EvaluationError(
            f"{path} support-index chunker universe differs from runnable H variants")
    for chunker, atom_mapping in chunkers.items():
        chunker = str(chunker)
        if not isinstance(atom_mapping, dict) or set(map(str, atom_mapping)) != atom_ids:
            raise EvaluationError(
                f"{path} support-index atom universe differs for chunker {chunker}")
        candidate_by_id = candidate_by_chunker[chunker]
        for atom_id, values in atom_mapping.items():
            atom_id = str(atom_id)
            if not isinstance(values, list):
                raise EvaluationError(
                    f"{path} support spans for {atom_id}/{chunker} are malformed")
            support_ids = list(map(str, values))
            if (
                len(set(support_ids)) != len(support_ids)
                or not set(support_ids).issubset(candidate_by_id)
            ):
                raise EvaluationError(
                    f"{path} support spans for {atom_id}/{chunker} leave the candidate universe")
            for support_id in support_ids:
                candidate = candidate_by_id[support_id]
                candidate_occurrences = set(map(
                    str, candidate.get("source_occurrence_ids") or ()))
                overlaps_truth = any(
                    candidate_occurrences.intersection(
                        map(str, truth_by_id[truth_id].get(
                            "source_occurrence_ids") or ()))
                    and int(candidate["char_start"]) < int(truth_by_id[truth_id]["char_end"])
                    and int(truth_by_id[truth_id]["char_start"]) < int(candidate["char_end"])
                    for truth_id in truth_supports[atom_id]
                )
                if not overlaps_truth:
                    raise EvaluationError(
                        f"{path} support span {support_id} for {atom_id}/{chunker} "
                        "does not overlap a grounded truth span")
    return index


def _assert_record_coordinates(
    record: dict,
    *,
    cell: _FrozenCell,
    run_id: str,
    phase_id: str,
    execution_binding_sha256: str,
    protocol_document_sha256: str,
) -> None:
    """A valid object under the wrong cell key is still the wrong outcome."""
    expected = (cell.block_id, cell.task_id, cell.arm_id, cell.replicate_id)
    if not record:
        raise EvaluationError(f"frozen assignment {expected} has an empty result artifact")
    raw_cell = record.get("cell") or {}
    raw_arm = raw_cell.get("arm") or {}
    observed = (
        str(raw_cell.get("block_id") or ""),
        str(raw_cell.get("task_id") or ""),
        str(raw_arm.get("arm_id") or raw_cell.get("arm_id") or ""),
        str(raw_cell.get("replicate_id") or ""),
    )
    if observed != expected:
        raise EvaluationError(
            f"result artifact coordinates {observed} do not match frozen assignment {expected}"
        )
    if raw_arm:
        variants = (
            str(raw_arm.get("page_variant") or ""),
            str(raw_arm.get("close_variant") or ""),
        )
        expected_variants = (cell.page_variant, cell.close_variant)
        if variants != expected_variants:
            raise EvaluationError(
                f"result artifact for {expected} records variants {variants}, "
                f"not frozen variants {expected_variants}"
            )
    if raw_cell:
        if int(raw_cell.get("seed", -1)) != cell.seed:
            raise EvaluationError(
                f"result artifact for {expected} records seed {raw_cell.get('seed')!r}, "
                f"not frozen seed {cell.seed}"
            )
        if int(raw_cell.get("order_index", -1)) != cell.order_index:
            raise EvaluationError(
                f"result artifact for {expected} records order "
                f"{raw_cell.get('order_index')!r}, not frozen order {cell.order_index}"
            )
    if str(record.get("run_id") or "") != run_id:
        raise EvaluationError(
            f"result artifact for {expected} belongs to run {record.get('run_id')!r}, "
            f"not {run_id!r}"
        )
    if str(record.get("phase_id") or "") != phase_id:
        raise EvaluationError(
            f"result artifact for {expected} belongs to phase {record.get('phase_id')!r}, "
            f"not {phase_id!r}"
        )
    if str(record.get("execution_binding_sha256") or "") != execution_binding_sha256:
        raise EvaluationError(
            f"result artifact for {expected} belongs to execution binding "
            f"{record.get('execution_binding_sha256')!r}, "
            f"not {execution_binding_sha256!r}"
        )
    if str(record.get("protocol_document_sha256") or "") != protocol_document_sha256:
        raise EvaluationError(
            f"result artifact for {expected} belongs to protocol document "
            f"{record.get('protocol_document_sha256')!r}, "
            f"not {protocol_document_sha256!r}"
        )


def _load_frozen_scope(
    directory: Path,
    *,
    run_id: str,
    phase_id: str,
    execution_binding_sha256: str,
    protocol_document_sha256: str,
) -> _FrozenScope:
    """Verify one complete content-addressed campaign root and its exact directory.

    Checking only the files that happen to exist cannot establish an ITT denominator: deleting
    a failed block leaves every remaining per-file digest valid. The root commits to the
    pre-treatment schedule and all terminal outcomes, and the directory must contain exactly
    those files.
    """
    if not directory.exists() or not directory.is_dir():
        raise ValueError(f"frozen block directory does not exist: {directory}")
    root_path = directory / FROZEN_ROOT_FILENAME
    if not root_path.is_file():
        raise ValueError(
            f"frozen campaign root is missing: {root_path}; partial block directories "
            "cannot define the all-offered ITT scope"
        )
    try:
        root = json.loads(root_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid frozen campaign root {root_path}: {exc}") from exc
    root_sha = str(root.get("freeze_root_sha256") or "")
    actual_root_sha = sha256_hex(canonical_json({
        key: value for key, value in root.items() if key != "freeze_root_sha256"
    }))
    if root_sha != actual_root_sha:
        raise ValueError(
            f"frozen campaign root {root_path} was edited: records {root_sha!r}, "
            f"hashes to {actual_root_sha}"
        )
    if root.get("schema_version") != "frozen_campaign_root_v1":
        raise ValueError(f"{root_path} has unsupported frozen root schema")
    if root.get("terminal_frozen") is not True:
        raise ValueError(f"{root_path} is not terminal_frozen")
    if str(root.get("run_id") or "") != run_id:
        raise ValueError(
            f"{root_path} belongs to run {root.get('run_id')!r}, not {run_id!r}")
    if str(root.get("phase_id") or "") != phase_id:
        raise ValueError(
            f"{root_path} belongs to phase {root.get('phase_id')!r}, not {phase_id!r}")
    root_binding = str(root.get("execution_binding_sha256") or "")
    root_protocol = str(root.get("protocol_sha256") or "")
    if root_binding != execution_binding_sha256:
        raise ValueError(
            f"{root_path} belongs to execution binding {root_binding!r}, "
            f"not {execution_binding_sha256!r}"
        )
    if root_protocol != protocol_document_sha256:
        raise ValueError(
            f"{root_path} belongs to protocol document {root_protocol!r}, "
            f"not {protocol_document_sha256!r}"
        )

    schedule = root.get("schedule")
    if not isinstance(schedule, dict):
        raise ValueError(f"{root_path} has no embedded schedule")
    schedule_sha = str(root.get("schedule_sha256") or "")
    recorded_schedule_sha = str(schedule.get("schedule_sha256") or "")
    actual_schedule_sha = sha256_hex(canonical_json({
        key: value for key, value in schedule.items() if key != "schedule_sha256"
    }))
    if (
        not schedule_sha
        or schedule_sha != recorded_schedule_sha
        or schedule_sha != actual_schedule_sha
    ):
        raise ValueError(
            f"{root_path} does not bind one verifiable schedule "
            f"({schedule_sha!r}, {recorded_schedule_sha!r}, {actual_schedule_sha!r})"
        )
    if str(schedule.get("split") or "") != str(root.get("split") or ""):
        raise ValueError(f"{root_path} schedule/root split mismatch")
    if str(schedule.get("protocol_sha") or "") != str(root.get("protocol_sha256") or ""):
        raise ValueError(f"{root_path} schedule/root protocol mismatch")
    if (
        str(schedule.get("execution_binding_sha256") or "")
        != root_binding
    ):
        raise ValueError(f"{root_path} schedule/root execution-binding mismatch")
    notes = schedule.get("notes")
    if not isinstance(notes, dict):
        raise ValueError(f"{root_path} schedule has no pre-treatment notes")
    analysis_design_receipt_sha256 = str(
        notes.get("analysis_design_receipt_sha256") or "")
    task_feature_registry_sha256 = str(
        notes.get("task_feature_registry_sha256") or "")
    eligibility_spec_content_sha256 = str(
        notes.get("eligibility_spec_content_sha256") or "")
    if any(len(value) != 64 for value in (
        analysis_design_receipt_sha256,
        task_feature_registry_sha256,
        eligibility_spec_content_sha256,
    )):
        raise ValueError(
            f"{root_path} schedule does not bind the pre-treatment analysis design")

    scheduled_blocks = schedule.get("blocks")
    root_blocks = root.get("blocks")
    if not isinstance(scheduled_blocks, list) or not scheduled_blocks:
        raise ValueError(f"{root_path} schedule has no offered blocks")
    if not isinstance(root_blocks, list):
        raise ValueError(f"{root_path} has no frozen block index")
    scheduled_by_id: dict[str, dict] = {}
    for scheduled in scheduled_blocks:
        block_id = str((scheduled or {}).get("block_id") or "")
        if not block_id or block_id in scheduled_by_id:
            raise ValueError(f"{root_path} has duplicate/unnamed scheduled block {block_id!r}")
        scheduled_by_id[block_id] = scheduled
    rooted_by_id: dict[str, dict] = {}
    for rooted in root_blocks:
        block_id = str((rooted or {}).get("block_id") or "")
        if not block_id or block_id in rooted_by_id:
            raise ValueError(f"{root_path} has duplicate/unnamed frozen block {block_id!r}")
        rooted_by_id[block_id] = rooted
    if set(scheduled_by_id) != set(rooted_by_id):
        raise ValueError(
            f"{root_path} frozen blocks differ from scheduled blocks: "
            f"missing={sorted(set(scheduled_by_id) - set(rooted_by_id))}, "
            f"extra={sorted(set(rooted_by_id) - set(scheduled_by_id))}"
        )

    expected_entries = {
        FROZEN_ROOT_FILENAME,
        *(f"{block_id}.json" for block_id in scheduled_by_id),
    }
    observed_entries = {path.name for path in directory.iterdir()}
    if observed_entries != expected_entries:
        raise ValueError(
            f"frozen directory does not exactly match root: "
            f"missing={sorted(expected_entries - observed_entries)}, "
            f"extra={sorted(observed_entries - expected_entries)}"
        )

    cells: list[_FrozenCell] = []
    seen: set[tuple[str, str, str, str]] = set()
    blocks: dict[str, _FrozenBlock] = {}
    for scheduled in scheduled_blocks:
        block_id = str(scheduled["block_id"])
        path = directory / f"{block_id}.json"
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as e:
            raise ValueError(f"invalid frozen block {path}: {e}") from e
        recorded = str(body.get("freeze_sha256") or "")
        unsigned = {k: v for k, v in body.items() if k != "freeze_sha256"}
        actual = sha256_hex(canonical_json(unsigned))
        if not recorded or recorded != actual:
            raise ValueError(
                f"frozen block {path} was edited or lacks freeze_sha256: "
                f"records {recorded!r}, hashes to {actual}"
            )
        if str(body.get("execution_binding_sha256") or "") != root_binding:
            raise ValueError(f"frozen block {path} has the wrong execution binding")
        if str(body.get("protocol_document_sha256") or "") != root_protocol:
            raise ValueError(f"frozen block {path} has the wrong protocol document")
        rooted = rooted_by_id[block_id]
        if str(rooted.get("freeze_sha256") or "") != recorded:
            raise ValueError(f"frozen block {path} freeze hash differs from root")
        if rooted.get("terminal_frozen") is not True or body.get("terminal_frozen") is not True:
            raise ValueError(f"frozen block {path} is not terminal_frozen")
        if canonical_json(rooted.get("cells") or []) != canonical_json(body.get("cells") or []):
            raise ValueError(f"frozen block {path} cells differ from the frozen root")

        observed_block_id = str(body.get("block_id") or "")
        task_id = str(body.get("task_id") or "")
        replicate_id = str(body.get("replicate_id") or "")
        if observed_block_id != block_id or not task_id:
            raise ValueError(f"frozen block {path} lacks block_id/task_id")
        if path.name != f"{observed_block_id}.json":
            raise ValueError(f"frozen block filename does not match {observed_block_id}")
        if (
            task_id != str(scheduled.get("task_id") or "")
            or replicate_id != str(scheduled.get("replicate_id") or "")
        ):
            raise ValueError(f"frozen block {path} coordinates differ from schedule")
        block_digest = sha256_hex(canonical_json(scheduled))
        if (
            str(body.get("block_digest") or "") != block_digest
            or str(rooted.get("block_digest") or "") != block_digest
        ):
            raise ValueError(f"frozen block {path} digest differs from embedded schedule")
        validity = rooted.get("valid_for_paired_estimate")
        if (
            not isinstance(validity, bool)
            or body.get("valid_for_paired_estimate") != validity
        ):
            raise ValueError(f"frozen block {path} lacks a consistent paired-validity flag")
        invalid_reason = str(rooted.get("invalid_reason") or "")
        if invalid_reason != str(body.get("invalid_reason") or ""):
            raise ValueError(f"frozen block {path} invalid_reason differs from root")
        if not validity and not invalid_reason:
            raise ValueError(
                f"frozen block {path} is invalid for paired estimation without a reason")
        if validity and invalid_reason:
            raise ValueError(
                f"frozen block {path} is marked paired-valid but also has invalid_reason")
        body_epochs = sorted(map(str, body.get("engine_epochs") or ()))
        root_epochs = sorted(map(str, rooted.get("engine_epochs") or ()))
        if body_epochs != root_epochs or not body_epochs:
            raise ValueError(f"frozen block {path} engine epochs differ from root or are absent")
        if validity and len(body_epochs) != 1:
            raise ValueError(
                f"frozen block {path} spans engine epochs but is marked paired-valid")
        scheduled_cells = scheduled.get("cells") or []
        raw_cells = body.get("cells") or []
        if len(raw_cells) != len(scheduled_cells) or not raw_cells:
            raise ValueError(f"frozen block {path} cells differ in count from schedule")
        scheduled_cell_by_key: dict[tuple[str, str, str, str], dict] = {}
        for raw in scheduled_cells:
            arm = raw.get("arm") or {}
            key = (
                str(raw.get("block_id") or ""),
                str(raw.get("task_id") or ""),
                str(arm.get("arm_id") or ""),
                str(raw.get("replicate_id") or ""),
            )
            if not all(key) or key in scheduled_cell_by_key:
                raise ValueError(f"duplicate/unnamed scheduled cell {key} in {root_path}")
            scheduled_cell_by_key[key] = raw

        epoch_by_arm: dict[str, str] = {}
        for raw in raw_cells:
            arm = raw.get("arm") or {}
            key = (
                str(raw.get("block_id") or block_id),
                str(raw.get("task_id") or task_id),
                str(arm.get("arm_id") or raw.get("arm_id") or ""),
                str(raw.get("replicate_id", replicate_id)),
            )
            expected_raw = scheduled_cell_by_key.get(key)
            if expected_raw is None:
                raise ValueError(f"cell {key} in {path} was not in the frozen schedule")
            expected_fields = {
                field: expected_raw.get(field)
                for field in (
                    "block_id", "task_id", "seed", "replicate_id", "order_index")
            }
            observed_fields = {
                field: raw.get(field)
                for field in (
                    "block_id", "task_id", "seed", "replicate_id", "order_index")
            }
            if observed_fields != expected_fields or arm != (expected_raw.get("arm") or {}):
                raise ValueError(f"cell {key} in {path} differs from the frozen schedule")
            state = str(raw.get("state") or "MISSING")
            from ..experiment.ledger import TERMINAL_STATES

            if state not in TERMINAL_STATES or not str(raw.get("output_ref") or ""):
                raise ValueError(f"cell {key} in {path} is not a frozen terminal outcome")
            engine_epoch = str(raw.get("engine_epoch") or "")
            if not engine_epoch or engine_epoch not in body_epochs:
                raise ValueError(f"cell {key} in {path} lacks its frozen engine epoch")
            arm_key = f"{key[2]}:{key[3]}"
            if arm_key in epoch_by_arm:
                raise ValueError(f"duplicate arm/replicate epoch key {arm_key} in {path}")
            epoch_by_arm[arm_key] = engine_epoch
            cell = _FrozenCell(
                block_id=key[0],
                task_id=key[1],
                replicate_id=key[3],
                arm_id=key[2],
                page_variant=str(arm.get("page_variant") or ""),
                close_variant=str(arm.get("close_variant") or ""),
                seed=int(raw.get("seed")),
                order_index=int(raw.get("order_index")),
                engine_epoch=engine_epoch,
                state=state,
                output_ref=str(raw.get("output_ref") or ""),
            )
            if (cell.block_id, cell.task_id, cell.replicate_id) != (
                block_id, task_id, replicate_id
            ):
                raise ValueError(f"cell coordinates disagree with frozen block {path}")
            key = (cell.block_id, cell.task_id, cell.arm_id, cell.replicate_id)
            if not cell.arm_id or key in seen:
                raise ValueError(f"duplicate or unnamed frozen cell {key} in {path}")
            seen.add(key)
            cells.append(cell)
        if set(scheduled_cell_by_key) != {
            (cell.block_id, cell.task_id, cell.arm_id, cell.replicate_id)
            for cell in cells if cell.block_id == block_id
        }:
            raise ValueError(f"frozen block {path} is missing a scheduled cell")
        blocks[block_id] = _FrozenBlock(
            block_id=block_id,
            task_id=task_id,
            replicate_id=replicate_id,
            block_digest=block_digest,
            freeze_sha256=recorded,
            valid_for_paired_estimate=validity,
            invalid_reason=invalid_reason,
            engine_epochs=tuple(body_epochs),
            engine_epoch_by_arm=epoch_by_arm,
        )
    return _FrozenScope(
        run_id=run_id,
        phase_id=phase_id,
        execution_binding_sha256=root_binding,
        protocol_document_sha256=root_protocol,
        schedule_sha256=schedule_sha,
        freeze_root_sha256=root_sha,
        analysis_design_receipt_sha256=analysis_design_receipt_sha256,
        task_feature_registry_sha256=task_feature_registry_sha256,
        eligibility_spec_content_sha256=eligibility_spec_content_sha256,
        cells=tuple(cells),
        blocks=blocks,
    )


def _write_evaluation_scope_receipt(
    directory: Path,
    *,
    scope: _FrozenScope,
    run_id: str,
    phase_id: str,
    score_records: list[dict],
) -> str:
    """Seal the exact all-offered score set consumed by analysis.

    Score files are independently content-addressed, but that still does not reveal whether one
    is missing. This receipt binds one score to every frozen block and is written only after the
    evaluator has produced the complete set.
    """
    by_id: dict[str, dict] = {}
    for record in score_records:
        block_id = str(record.get("block_id") or "")
        if not block_id or block_id in by_id:
            raise EvaluationError(f"duplicate/unnamed score record {block_id!r}")
        by_id[block_id] = record
    if set(by_id) != set(scope.blocks):
        raise EvaluationError(
            "cannot seal partial evaluation scope: "
            f"missing={sorted(set(scope.blocks) - set(by_id))}, "
            f"extra={sorted(set(by_id) - set(scope.blocks))}"
        )

    directory = Path(directory)
    expected_primary = {f"{block_id}.json" for block_id in scope.blocks}
    auxiliary = {
        EVALUATION_SCOPE_FILENAME,
        *(f"{block_id}.judge.json" for block_id in scope.blocks),
    }
    observed_json = {path.name for path in directory.glob("*.json")}
    unexpected = observed_json - expected_primary - auxiliary
    missing = expected_primary - observed_json
    if unexpected or missing:
        raise EvaluationError(
            "judgment directory differs from the frozen all-offered scope: "
            f"missing={sorted(missing)}, extra={sorted(unexpected)}"
        )
    ordered: list[dict] = []
    for block_id in sorted(scope.blocks):
        path = directory / f"{block_id}.json"
        try:
            score = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise EvaluationError(f"invalid score artifact {path}: {exc}") from exc
        recorded_sha = str(score.get("content_sha256") or "")
        actual_sha = sha256_hex(canonical_json({
            key: value for key, value in score.items() if key != "content_sha256"
        }))
        expected_sha = str(by_id[block_id].get("score_content_sha256") or "")
        if not recorded_sha or recorded_sha != actual_sha or recorded_sha != expected_sha:
            raise EvaluationError(
                f"score artifact {path} does not match its evaluated content")
        ordered.append(dict(by_id[block_id]))

    body = {
        "schema_version": "evaluated_itt_scope_v1",
        "run_id": run_id,
        "phase_id": phase_id,
        "execution_binding_sha256": scope.execution_binding_sha256,
        "protocol_document_sha256": scope.protocol_document_sha256,
        "schedule_sha256": scope.schedule_sha256,
        "freeze_root_sha256": scope.freeze_root_sha256,
        "analysis_design_receipt_sha256":
            scope.analysis_design_receipt_sha256,
        "task_feature_registry_sha256":
            scope.task_feature_registry_sha256,
        "eligibility_spec_content_sha256":
            scope.eligibility_spec_content_sha256,
        "all_offered_blocks": len(scope.blocks),
        "scores": ordered,
    }
    body["evaluation_scope_sha256"] = sha256_hex(canonical_json(body))
    path = directory / EVALUATION_SCOPE_FILENAME
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return str(body["evaluation_scope_sha256"])
    except FileExistsError:
        pass
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid evaluation scope receipt {path}: {exc}") from exc
    existing_sha = str(existing.get("evaluation_scope_sha256") or "")
    actual_existing = sha256_hex(canonical_json({
        key: value for key, value in existing.items()
        if key != "evaluation_scope_sha256"
    }))
    if existing_sha != actual_existing:
        raise EvaluationError(f"evaluation scope receipt {path} was edited")
    if existing_sha != body["evaluation_scope_sha256"]:
        raise EvaluationError(
            f"{path} already seals a different score set; evaluation scope is write-once")
    return existing_sha


def _evidence_context_token_partition(
    *,
    span_kind_by_id: dict[str, str],
    span_token_counts: dict[str, int],
    published_span_ids: set[str],
) -> tuple[int, int, int, int]:
    """Split visible material tokens without promoting context into evidence.

    Returns offered evidence, offered context, published evidence and published context.
    Rendered-token metrics remain over the complete publication, because context still costs
    work; evidence precision/recall denominators use only TOOL_EVIDENCE.
    """
    evidence_ids = {
        span_id for span_id, kind in span_kind_by_id.items()
        if kind == "TOOL_EVIDENCE"
    }
    context_ids = {
        span_id for span_id, kind in span_kind_by_id.items()
        if kind in {
            "TOOL_UNATTRIBUTED_CONTEXT",
            "MODEL_DERIVED_CONTEXT",
            "USER_CONTEXT",
        }
    }
    return (
        sum(span_token_counts.get(span_id, 0) for span_id in evidence_ids),
        sum(span_token_counts.get(span_id, 0) for span_id in context_ids),
        sum(
            span_token_counts.get(span_id, 0)
            for span_id in published_span_ids & evidence_ids
        ),
        sum(
            span_token_counts.get(span_id, 0)
            for span_id in published_span_ids & context_ids
        ),
    )


def _nonnegative_token_count(
    raw: dict,
    field: str,
    invalid: list[str],
    checkpoint_digest: str,
) -> int:
    """Read one frozen token counter without capturing a surrounding checkpoint loop."""

    value = raw.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        invalid.append(
            f"{checkpoint_digest or '<missing checkpoint>'}: {field} is invalid"
        )
        return 0
    return value


def _add_direct_checkpoint_metrics(
    settings: Settings, score, outputs: list[ArmOutput], truth: dict,
    visibility_decider=None,
) -> None:
    """Add C_VISIBLE recall from actual published ids at the exact C checkpoint.

    The old implementation used atoms recovered from the *final report*. That measures what
    the final writer happened to say, not what the reducer retained. Here the numerator is the
    reducer's frozen ``published_span_ids``. The final report never enters this function.
    """
    from ..evaluation.visible_truth_projection import project_visible_truth, visible_recall
    from ..odr.checkpoints import CheckpointStore
    from ..strategies.pipeline import spans_from_visible_view
    from ..strategies.visible_view import build_visible_view

    store = CheckpointStore(settings.path("checkpoints"))
    atom_texts = truth.get("atom_texts") or {}
    atomic_evidence = list(truth["packet"].get("atomic_evidence") or [])
    required = {str(a.get("atom_id") or "") for a in atomic_evidence}
    required.discard("")
    atom_weights: dict[str, float] = {}
    truth_weight_errors: list[str] = []
    seen_atom_ids: set[str] = set()
    for index, atom in enumerate(atomic_evidence):
        atom_id = str(atom.get("atom_id") or "")
        if not atom_id:
            truth_weight_errors.append(f"truth atom {index} has no atom_id")
            continue
        if atom_id in seen_atom_ids:
            truth_weight_errors.append(f"truth atom id {atom_id!r} is duplicated")
            continue
        seen_atom_ids.add(atom_id)
        if "weight" not in atom:
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has no frozen weight")
            continue
        raw_weight = atom.get("weight")
        if isinstance(raw_weight, bool):
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has a boolean weight")
            continue
        try:
            weight = float(raw_weight)
        except (TypeError, ValueError):
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has a non-numeric weight")
            continue
        if not math.isfinite(weight) or weight < 0:
            truth_weight_errors.append(
                f"truth atom {atom_id!r} has an invalid weight {raw_weight!r}")
            continue
        atom_weights[atom_id] = weight
    missing_atom_texts = required - set(map(str, atom_texts))
    if missing_atom_texts:
        truth_weight_errors.append(
            "visible projection lacks frozen text for truth atoms: "
            f"{sorted(missing_atom_texts)}"
        )
    atom_facets = {
        str(a["atom_id"]): str(a.get("facet_id") or "")
        for a in (truth["packet"].get("atomic_evidence") or [])
    }
    contradiction_pairs = {
        (str(p["atom_id_a"]), str(p["atom_id_b"]))
        for p in (truth["packet"].get("contradiction_pairs") or [])
    }
    critical_atom_ids = {
        str(atom.get("atom_id") or "")
        for atom in (truth["packet"].get("atomic_evidence") or ())
        if atom.get("critical")
    }
    negative_atom_ids = {
        str(item.get("atom_id") or "")
        for item in (truth["packet"].get("negative_evidence") or ())
    }
    negative_query_ids = {
        str(item.get("query_attempt_id") or "")
        for item in (truth["packet"].get("negative_evidence") or ())
    }
    known_gap_query_ids = {
        str(item.get("query_attempt_id") or "")
        for item in (truth["packet"].get("known_gaps") or ())
        if isinstance(item, dict)
        and str(item.get("status") or "") in {
            "FAILED", "TIMEOUT", "BLOCKED_BUDGET",
        }
    }
    critical_atom_ids.discard("")
    negative_atom_ids.discard("")
    negative_query_ids.discard("")
    known_gap_query_ids.discard("")
    unknown_negative_atoms = negative_atom_ids - required
    if unknown_negative_atoms:
        truth_weight_errors.append(
            "negative evidence references unknown truth atoms: "
            f"{sorted(unknown_negative_atoms)}"
        )
    guarded_query_ids = negative_query_ids | known_gap_query_ids
    from ..strategies.factory import load_registry

    registry = load_registry(settings.repo / "configs")
    reference_projection_cache: dict[str, object] = {}
    from ..evidence.model_tokenizer import (
        load_frozen_tokenizer,
        tokenizer_sha256,
    )

    tokenizer = load_frozen_tokenizer(settings)
    tokenizer_digest = tokenizer_sha256(tokenizer)

    for output in outputs:
        arm_key = f"{output.arm_id}:{output.replicate_id}"
        per_arm = score.per_arm.get(arm_key)
        if per_arm is None:
            continue
        direct = per_arm.get("direct_node_metrics") or {}
        traces = [
            raw for raw in output.direct_node_records
            if str(raw.get("node") or "").upper().startswith("C")
        ]
        c_normalization = (
            (direct.get("selector_normalization_by_node") or {}).get("C")
            or selector_normalization_metrics(traces, node="C")
        )
        close_specs = [
            registry[part] for part in output.variant_id.split("+")
            if part in registry and registry[part].node in {
                "C_VISIBLE", "C_REGISTRY", "C_FUSED_EXT",
            }
        ]
        if not traces:
            if close_specs:
                direct.setdefault("by_node", {})["C"] = {
                    "applicable": True,
                    "status": "DIRECT_TRACE_UNAVAILABLE",
                    "checkpoint_count": 0,
                    "selector_normalization": c_normalization,
                    "selector_guards_complete": False,
                }
                per_arm["direct_node_metrics"] = direct
            continue
        if len(close_specs) != 1:
            direct["visible_projection_status"] = "C_VARIANT_UNRESOLVED"
            direct.setdefault("by_node", {})["C"] = {
                "applicable": True,
                "status": "C_VARIANT_UNRESOLVED",
                "checkpoint_count": len(traces),
                "selector_normalization": c_normalization,
                "selector_guards_complete": False,
            }
            per_arm["direct_node_metrics"] = direct
            continue
        close_spec = close_specs[0]
        if close_spec.node != "C_VISIBLE":
            direct["visible_projection_status"] = (
                f"NOT_APPLICABLE_{close_spec.node}")
            direct.setdefault("by_node", {})["C"] = {
                "applicable": False,
                "status": f"NOT_APPLICABLE_{close_spec.node}",
                "checkpoint_count": len(traces),
                "selector_normalization": c_normalization,
                "selector_guards_complete": False,
            }
            per_arm["direct_node_metrics"] = direct
            continue
        visible_rows: list[dict] = []
        c_trace_invalid: list[str] = list(truth_weight_errors)
        for raw in traces:
            digest = str(raw.get("checkpoint_hash") or "")
            published = list(map(str, raw.get("published_span_ids") or ()))
            offered_span_ids = set(map(str, raw.get("offered_span_ids") or ()))
            selected_span_ids = set(map(str, raw.get("selected_span_ids") or ()))
            published_span_ids = set(published)
            token_fields = (
                "offered_span_token_counts",
                "offered_evidence_tokens",
                "staged_rendered_tokens",
                "published_rendered_tokens",
            )
            semantic_token_fields = (
                "offered_material_tokens",
                "offered_context_tokens",
            )
            token_trace_present = all(field in raw for field in token_fields)
            if any(field in raw for field in token_fields) and not token_trace_present:
                c_trace_invalid.append(
                    f"{digest or '<missing checkpoint>'}: partial materialization token trace"
                )
            if (
                any(field in raw for field in semantic_token_fields)
                and not all(field in raw for field in semantic_token_fields)
            ):
                c_trace_invalid.append(
                    f"{digest or '<missing checkpoint>'}: partial evidence/context token split"
                )
            span_token_counts: dict[str, int] = {}
            if token_trace_present:
                raw_token_counts = raw.get("offered_span_token_counts")
                if not isinstance(raw_token_counts, list | tuple):
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: "
                        "offered_span_token_counts is not a sequence"
                    )
                    raw_token_counts = ()
                for item in raw_token_counts:
                    if (
                        not isinstance(item, list | tuple)
                        or len(item) != 2
                        or not isinstance(item[0], str)
                        or not item[0]
                        or isinstance(item[1], bool)
                        or not isinstance(item[1], int)
                        or item[1] < 0
                        or item[0] in span_token_counts
                    ):
                        c_trace_invalid.append(
                            f"{digest or '<missing checkpoint>'}: invalid or duplicate "
                            "offered-span token count"
                        )
                        continue
                    span_token_counts[item[0]] = item[1]
                if set(span_token_counts) != offered_span_ids:
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: span token counts do not "
                        "cover exactly the offered set"
                    )

                reported_evidence_tokens = _nonnegative_token_count(
                    raw,
                    "offered_evidence_tokens",
                    c_trace_invalid,
                    digest,
                )
                if all(field in raw for field in semantic_token_fields):
                    offered_material_tokens = _nonnegative_token_count(
                        raw,
                        "offered_material_tokens",
                        c_trace_invalid,
                        digest,
                    )
                    reported_context_tokens = _nonnegative_token_count(
                        raw,
                        "offered_context_tokens",
                        c_trace_invalid,
                        digest,
                    )
                else:
                    # Backward-compatible reconstruction for already frozen traces produced
                    # before origin-aware accounting: their offered_evidence_tokens wire field
                    # meant all candidate material.
                    offered_material_tokens = reported_evidence_tokens
                    reported_context_tokens = None
                staged_rendered_tokens = _nonnegative_token_count(
                    raw,
                    "staged_rendered_tokens",
                    c_trace_invalid,
                    digest,
                )
                published_rendered_tokens = _nonnegative_token_count(
                    raw,
                    "published_rendered_tokens",
                    c_trace_invalid,
                    digest,
                )
                if offered_material_tokens != sum(span_token_counts.values()):
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: offered token total "
                        "does not equal the per-span sum"
                    )
                if raw.get("fell_back") or raw.get("failure"):
                    if published_rendered_tokens != 0:
                        c_trace_invalid.append(
                            f"{digest or '<missing checkpoint>'}: failed/fallback C "
                            "selection claims published tokens"
                        )
                elif published_span_ids and published_rendered_tokens <= 0:
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: published C evidence has "
                        "no rendered-token count"
                    )
                if published_rendered_tokens > staged_rendered_tokens:
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: published rendered tokens "
                        "exceed staged tokens"
                    )
            else:
                offered_material_tokens = None
                reported_evidence_tokens = None
                reported_context_tokens = None
                staged_rendered_tokens = None
                published_rendered_tokens = None
            if not selected_span_ids.issubset(offered_span_ids):
                c_trace_invalid.append(
                    f"{digest or '<missing checkpoint>'}: selected ids outside offered set")
            if not published_span_ids.issubset(selected_span_ids):
                c_trace_invalid.append(
                    f"{digest or '<missing checkpoint>'}: published ids outside selected set")
            for relation in raw.get("published_relations") or ():
                if isinstance(relation, dict):
                    span_id = str(relation.get("span_id") or "")
                    role = str(relation.get("role") or "")
                elif isinstance(relation, list | tuple) and len(relation) >= 3:
                    span_id, _facet_id, role = map(str, relation[:3])
                else:
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: malformed published relation")
                    continue
                if span_id not in published_span_ids:
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: relation references "
                        f"unpublished span {span_id}")
                if role not in {"support", "contradict", "background"}:
                    c_trace_invalid.append(
                        f"{digest or '<missing checkpoint>'}: invalid relation role {role!r}")
            offered_queries = set(map(
                str, raw.get("offered_query_attempt_ids") or ()))
            published_queries = set(map(
                str, raw.get("published_query_attempt_ids")
                or raw.get("selected_query_attempt_ids") or ()))
            for gap in raw.get("published_gaps") or ():
                if isinstance(gap, dict):
                    query_ids = gap.get("query_attempt_ids") or ()
                elif isinstance(gap, list | tuple) and len(gap) >= 2:
                    query_ids = gap[1] or ()
                else:
                    continue
                published_queries.update(map(str, query_ids))
            if not published_queries.issubset(offered_queries):
                c_trace_invalid.append(
                    f"{digest or '<missing checkpoint>'}: published query ids outside offered set"
                )
            offered_negative_queries = negative_query_ids & offered_queries
            retained_negative_queries = offered_negative_queries & published_queries
            offered_known_gaps = known_gap_query_ids & offered_queries
            retained_known_gaps = offered_known_gaps & published_queries
            if not digest:
                visible_rows.append({
                    "checkpoint_hash": "", "visible_recall": None,
                    "note": "DIRECT_TRACE_MISSING_CHECKPOINT",
                })
                continue
            try:
                checkpoint = store.get(digest)
            except (FileNotFoundError, ValueError) as e:
                visible_rows.append({
                    "checkpoint_hash": digest, "visible_recall": None,
                    "note": f"{type(e).__name__}: {e}",
                })
                continue

            view = build_visible_view(checkpoint.researcher_messages)
            spans = spans_from_visible_view(
                view.view_bytes, view_hash=view.view_hash, messages=view.message_segments,
                tokenizer=tokenizer, chunker=close_spec.chunker)
            expected_offered = [
                str(span.get("visible_span_id") or span.get("span_id") or "")
                for span in spans
            ]
            observed_offered = list(map(
                str, raw.get("offered_span_ids") or ()))
            if observed_offered != expected_offered:
                c_trace_invalid.append(
                    f"{digest}: offered candidate set/order does not reconstruct from "
                    "the frozen C checkpoint"
                )
            if str(raw.get("chunker") or "") != close_spec.chunker:
                c_trace_invalid.append(
                    f"{digest}: trace chunker does not match frozen variant")
            if str(raw.get("contract") or "") != close_spec.contract:
                c_trace_invalid.append(
                    f"{digest}: trace contract does not match frozen variant")
            if str(raw.get("aggregation") or "") != close_spec.aggregation:
                c_trace_invalid.append(
                    f"{digest}: trace aggregation does not match frozen variant")
            if list(map(str, raw.get("offered_query_attempt_ids") or ())) != list(
                checkpoint.query_attempt_ids
            ):
                c_trace_invalid.append(
                    f"{digest}: offered query attempts do not match the frozen C checkpoint")

            # Rebuild the selector's immutable offered view, including its prompt bytes. This
            # binds the direct trace to the actual checkpoint rather than trusting a self-
            # reported list of IDs that could have been fabricated after selection.
            try:
                from ..p1.view import CandidateViewRecord

                task_view = json.loads(
                    (settings.path("evaluator_root") / "tasks"
                     / f"{output.task_id}.json").read_text(encoding="utf-8")
                )
                rebuilt_view = CandidateViewRecord.build(
                    spans=spans,
                    tokenizer=tokenizer,
                    namespace="VISIBLE_MESSAGE",
                    topic=str(task_view["original_question"]),
                    contract=close_spec.contract,
                    token_budget=int(settings.get(
                        "week1", "measurement", "selected_token_budget")),
                    query_attempts=[
                        (str(query_id), str(query_id))
                        for query_id in checkpoint.query_attempt_ids
                    ],
                    visible_views={view.view_hash: view.view_bytes},
                    snapshot_texts={},
                )
                if str(raw.get("candidate_view_sha256") or "") != rebuilt_view.view_sha256:
                    c_trace_invalid.append(
                        f"{digest}: candidate view digest does not bind the frozen checkpoint")
                if str(raw.get("tokenizer_sha256") or "") != tokenizer_digest:
                    c_trace_invalid.append(
                        f"{digest}: trace tokenizer does not match the frozen model tokenizer"
                    )
            except (OSError, KeyError, ValueError) as e:
                c_trace_invalid.append(
                    f"{digest}: candidate view reconstruction failed: "
                    f"{type(e).__name__}: {e}"
                )
            span_texts = {
                s.get("visible_span_id") or s.get("span_id", ""):
                view.view_bytes[s["byte_start"]:s["byte_end"]].decode(
                    "utf-8", errors="replace")
                for s in spans
            }
            span_kind_by_id = {
                str(s.get("visible_span_id") or s.get("span_id") or ""):
                str(s.get("kind") or "")
                for s in spans
            }
            allowed_visible_kinds = {
                "TOOL_EVIDENCE",
                "TOOL_UNATTRIBUTED_CONTEXT",
                "MODEL_DERIVED_CONTEXT",
                "USER_CONTEXT",
            }
            unknown_kinds = {
                kind for kind in span_kind_by_id.values()
                if kind not in allowed_visible_kinds
            }
            if unknown_kinds:
                c_trace_invalid.append(
                    f"{digest}: visible spans have unknown origins {sorted(unknown_kinds)}"
                )
            context_span_ids = {
                span_id for span_id, kind in span_kind_by_id.items()
                if kind in {
                    "TOOL_UNATTRIBUTED_CONTEXT",
                    "MODEL_DERIVED_CONTEXT",
                    "USER_CONTEXT",
                }
            }
            for relation in raw.get("published_relations") or ():
                if isinstance(relation, dict):
                    relation_span_id = str(relation.get("span_id") or "")
                    relation_role = str(relation.get("role") or "")
                elif isinstance(relation, list | tuple) and len(relation) >= 3:
                    relation_span_id = str(relation[0])
                    relation_role = str(relation[2])
                else:
                    continue
                if (
                    relation_span_id in context_span_ids
                    and relation_role in {"support", "contradict"}
                ):
                    c_trace_invalid.append(
                        f"{digest}: non-citable context span {relation_span_id} is marked "
                        f"{relation_role}"
                    )
            if token_trace_present:
                expected_token_counts = {
                    span_id: tokenizer.count(text)
                    for span_id, text in span_texts.items()
                }
                if span_token_counts != expected_token_counts:
                    c_trace_invalid.append(
                        f"{digest}: offered span token counts do not reconstruct from "
                        "the frozen C checkpoint"
                    )
                (
                    offered_evidence_tokens,
                    offered_context_tokens,
                    published_evidence_span_tokens,
                    published_context_span_tokens,
                ) = _evidence_context_token_partition(
                    span_kind_by_id=span_kind_by_id,
                    span_token_counts=span_token_counts,
                    published_span_ids=published_span_ids,
                )
                if reported_context_tokens is not None:
                    if reported_evidence_tokens != offered_evidence_tokens:
                        c_trace_invalid.append(
                            f"{digest}: reported evidence tokens do not match "
                            "TOOL_EVIDENCE spans"
                        )
                    if reported_context_tokens != offered_context_tokens:
                        c_trace_invalid.append(
                            f"{digest}: reported context tokens do not match "
                            "model/user context spans"
                        )
                    if (
                        offered_evidence_tokens + offered_context_tokens
                        != offered_material_tokens
                    ):
                        c_trace_invalid.append(
                            f"{digest}: evidence plus context tokens do not equal material"
                        )
            else:
                offered_evidence_tokens = None
                offered_context_tokens = None
                published_evidence_span_tokens = None
                published_context_span_tokens = None
            try:
                # The denominator must not be chosen by the arm's chunker. Rebuild one
                # treatment-independent prechunk view (one span per message), then measure
                # whether the arm's candidate chunks preserved those facts and whether the
                # selector published them.
                reference_projection = reference_projection_cache.get(digest)
                if reference_projection is None:
                    reference_spans = spans_from_visible_view(
                        view.view_bytes,
                        view_hash=view.view_hash,
                        messages=view.message_segments,
                        tokenizer=tokenizer,
                        max_tokens=max(1, len(view.view_bytes) + 1),
                        chunker="fixed_token_v1",
                    )
                    reference_texts = {
                        s.get("visible_span_id") or s.get("span_id", ""):
                        view.view_bytes[s["byte_start"]:s["byte_end"]].decode(
                            "utf-8", errors="replace")
                        for s in reference_spans
                    }
                    reference_projection = project_visible_truth(
                        checkpoint_hash=digest,
                        visible_view_hash=view.view_hash,
                        spans=reference_spans,
                        span_texts=reference_texts,
                        atoms=[(a, atom_texts.get(a, "")) for a in sorted(required)],
                        decider=visibility_decider,
                        projection_prompt_hash=(
                            relation_prompt_sha256()
                            if visibility_decider is not None else ""),
                    )
                    reference_projection_cache[digest] = reference_projection
                candidate_projection = project_visible_truth(
                    checkpoint_hash=digest, visible_view_hash=view.view_hash,
                    spans=spans, span_texts=span_texts,
                    atoms=[(a, atom_texts.get(a, "")) for a in sorted(required)],
                    decider=visibility_decider,
                    projection_prompt_hash=(
                        relation_prompt_sha256() if visibility_decider is not None else ""),
                )
            except Exception as e:
                from ..bench.grading.judge_client import JudgeUnavailable

                if not isinstance(e, JudgeUnavailable):
                    raise
                # Direct projection judgment is a measurement, not treatment execution.
                # Preserve the offered assignment and expose a missing denominator so the
                # ITT sensitivity layer can assign its adverse bound.
                visible_rows.append({
                    "checkpoint_hash": digest,
                    "visible_recall": None,
                    "note": f"JUDGE_UNAVAILABLE: {e}",
                })
                score.human_queue.append({
                    "task_id": output.task_id,
                    "arm_id": output.arm_id,
                    "replicate_id": output.replicate_id,
                    "reason": (
                        "DIRECT_C_PROJECTION_JUDGE_UNAVAILABLE: "
                        f"{type(e).__name__}: {e}"
                    ),
                })
                continue
            reference_projected_ids = [
                atom.truth_atom_id for atom in reference_projection.projected_atoms
            ]
            candidate_projected_ids = [
                atom.truth_atom_id for atom in candidate_projection.projected_atoms
            ]
            projection_errors: list[str] = []
            if (
                len(reference_projected_ids) != len(set(reference_projected_ids))
                or set(reference_projected_ids) != required
            ):
                projection_errors.append(
                    f"{digest}: reference VisibleTruthProjection does not contain "
                    "exactly one row for every frozen truth atom"
                )
            if (
                len(candidate_projected_ids) != len(set(candidate_projected_ids))
                or set(candidate_projected_ids) != required
            ):
                projection_errors.append(
                    f"{digest}: candidate VisibleTruthProjection does not contain "
                    "exactly one row for every frozen truth atom"
                )
            allowed_projection_statuses = {
                "EXPLICITLY_VISIBLE", "NOT_VISIBLE", "AMBIGUOUS",
            }
            if any(
                atom.projection_status not in allowed_projection_statuses
                for atom in (
                    *reference_projection.projected_atoms,
                    *candidate_projection.projected_atoms,
                )
            ):
                projection_errors.append(
                    f"{digest}: VisibleTruthProjection has an unknown status")
            c_trace_invalid.extend(projection_errors)
            projection_valid = not projection_errors and not truth_weight_errors
            retained = [
                atom.atom_id for atom in candidate_projection.projected_atoms
                if set(atom.supporting_visible_span_ids) & set(published)
            ]
            roles_by_atom: dict[str, set[str]] = {}
            for relation in raw.get("published_relations") or ():
                if isinstance(relation, dict):
                    span_id = str(relation.get("span_id") or "")
                    facet_id = str(relation.get("facet_id") or "")
                    role = str(relation.get("role") or "")
                elif isinstance(relation, list | tuple) and len(relation) >= 3:
                    span_id, facet_id, role = map(str, relation[:3])
                else:
                    continue
                for atom in candidate_projection.projected_atoms:
                    if span_id in atom.supporting_visible_span_ids and (
                        not facet_id or facet_id == atom_facets.get(atom.truth_atom_id)
                    ):
                        roles_by_atom.setdefault(atom.truth_atom_id, set()).add(role)
            visible_ids = set(reference_projection.explicitly_visible_ids)
            candidate_ids = set(candidate_projection.explicitly_visible_ids) & visible_ids
            retained_ids = set(retained) & visible_ids
            relevant_published_spans = {
                span_id
                for atom in candidate_projection.projected_atoms
                if atom.truth_atom_id in retained_ids
                for span_id in atom.supporting_visible_span_ids
                if span_id in published_span_ids
            }
            # Selected-token precision is an evidence precision.  Model/user context remains a
            # real rendered cost (and therefore stays in published_rendered_tokens), but it is
            # neither its numerator nor its evidence denominator.
            published_span_tokens = published_evidence_span_tokens
            relevant_published_span_tokens = (
                sum(
                    span_token_counts.get(span_id, 0)
                    for span_id in relevant_published_spans
                )
                if token_trace_present else None
            )
            visible_pairs = {
                pair for pair in contradiction_pairs
                if pair[0] in visible_ids and pair[1] in visible_ids
            }
            retained_pairs = {
                pair for pair in visible_pairs
                if pair[0] in retained_ids and pair[1] in retained_ids
            }
            visible_critical = visible_ids & critical_atom_ids
            retained_critical = retained_ids & visible_critical
            visible_negative_atoms = visible_ids & negative_atom_ids
            retained_negative_atoms = retained_ids & visible_negative_atoms
            visible_truth_weight = sum(
                atom_weights[atom_id] for atom_id in visible_ids
                if atom_id in atom_weights
            )
            retained_truth_weight = sum(
                atom_weights[atom_id] for atom_id in retained_ids
                if atom_id in atom_weights
            )
            visible_negative_atom_weight = sum(
                atom_weights[atom_id] for atom_id in visible_negative_atoms
                if atom_id in atom_weights
            )
            retained_negative_atom_weight = sum(
                atom_weights[atom_id] for atom_id in retained_negative_atoms
                if atom_id in atom_weights
            )
            negative_gap_weight = (
                visible_negative_atom_weight
                + float(len(offered_negative_queries))
                + float(len(offered_known_gaps))
            )
            retained_negative_gap_weight = (
                retained_negative_atom_weight
                + float(len(retained_negative_queries))
                + float(len(retained_known_gaps))
            )
            typed = str(raw.get("contract") or "") in {"P1_TYPED", "P1_BRIDGE"}
            role_pairs = {
                pair for pair in retained_pairs
                if (
                    (
                        "support" in roles_by_atom.get(pair[0], set())
                        and "contradict" in roles_by_atom.get(pair[1], set())
                    )
                    or (
                        "contradict" in roles_by_atom.get(pair[0], set())
                        and "support" in roles_by_atom.get(pair[1], set())
                    )
                )
            }
            visible_rows.append({
                "checkpoint_hash": digest,
                "projection_valid": projection_valid,
                "visible_recall": visible_recall(reference_projection, retained_ids)
                if projection_valid else None,
                "c_total_published_prechunk_recall":
                    visible_recall(reference_projection, retained_ids)
                    if projection_valid else None,
                "weighted_evidence_recall": (
                    visible_recall(
                        reference_projection, retained_ids, weights=atom_weights)
                    if projection_valid else None
                ),
                "candidate_coverage": (
                    len(candidate_ids) / len(visible_ids) if visible_ids else None
                ),
                "selector_conditional_recall": (
                    len(retained_ids & candidate_ids) / len(candidate_ids)
                    if candidate_ids else None
                ),
                "reference_projection_sha256": reference_projection.content_sha256,
                "candidate_projection_sha256": candidate_projection.content_sha256,
                "explicitly_visible_atom_ids": sorted(visible_ids),
                "candidate_visible_atom_ids": sorted(candidate_ids),
                "published_visible_atom_ids": sorted(retained_ids),
                "eligible_denominators": {
                    "candidate_coverage": len(visible_ids),
                    "selector_conditional_recall": len(candidate_ids),
                    "weighted_evidence_recall": (
                        visible_truth_weight if projection_valid else None
                    ),
                    "total_published_prechunk_recall": len(visible_ids),
                    "critical_truth_recall": len(visible_critical),
                    "contradiction_pair_recall": len(visible_pairs),
                    "grounded_negative_atom_recall":
                        len(visible_negative_atoms),
                    "negative_query_trace_recall":
                        len(offered_negative_queries),
                    "unresolved_gap_recall": len(offered_known_gaps),
                    "negative_gap_recall": (
                        negative_gap_weight if projection_valid else None
                    ),
                    "typed_relation_coverage":
                        len(retained_ids) if typed else 0,
                    "contradiction_role_pair_recall":
                        len(visible_pairs) if typed else 0,
                },
                "contradiction_pair_recall": (
                    len(retained_pairs) / len(visible_pairs) if visible_pairs else None
                ),
                "critical_truth_recall": (
                    len(retained_critical) / len(visible_critical)
                    if visible_critical else None
                ),
                "grounded_negative_atom_recall": (
                    len(retained_negative_atoms) / len(visible_negative_atoms)
                    if visible_negative_atoms else None
                ),
                "typed_relation_coverage": (
                    sum(bool(roles_by_atom.get(atom_id)) for atom_id in retained_ids)
                    / len(retained_ids) if typed and retained_ids else None
                ),
                "contradiction_role_pair_recall": (
                    len(role_pairs) / len(visible_pairs)
                    if typed and visible_pairs else None
                ),
                "negative_query_trace_recall": (
                    len(retained_negative_queries) / len(offered_negative_queries)
                    if offered_negative_queries else None
                ),
                "unresolved_gap_recall": (
                    len(retained_known_gaps) / len(offered_known_gaps)
                    if offered_known_gaps else None
                ),
                "negative_gap_recall": (
                    retained_negative_gap_weight / negative_gap_weight
                    if projection_valid and negative_gap_weight > 0 else None
                ),
                "weighted_evidence_numerator": (
                    retained_truth_weight if projection_valid else None
                ),
                "weighted_evidence_denominator": (
                    visible_truth_weight if projection_valid else None
                ),
                "token_trace_status": (
                    "OK" if token_trace_present else "UNAVAILABLE"
                ),
                "offered_material_tokens": offered_material_tokens,
                "offered_evidence_tokens": offered_evidence_tokens,
                "offered_context_tokens": offered_context_tokens,
                "staged_rendered_tokens": staged_rendered_tokens,
                "published_rendered_tokens": published_rendered_tokens,
                "published_span_tokens": published_span_tokens,
                "published_evidence_span_tokens":
                    published_evidence_span_tokens,
                "published_context_span_tokens":
                    published_context_span_tokens,
                "relevant_published_span_tokens":
                    relevant_published_span_tokens,
                "selected_token_precision": (
                    relevant_published_span_tokens / published_span_tokens
                    if (
                        projection_valid
                        and relevant_published_span_tokens is not None
                        and published_span_tokens
                    ) else None
                ),
                "weighted_truth_per_100_rendered_tokens": (
                    100.0 * retained_truth_weight / published_rendered_tokens
                    if (
                        projection_valid
                        and published_rendered_tokens
                    ) else None
                ),
                "materialization_ratio": (
                    published_rendered_tokens / offered_material_tokens
                    if (
                        projection_valid
                        and published_rendered_tokens is not None
                        and offered_material_tokens
                    ) else None
                ),
                "negative_gap_numerator": (
                    retained_negative_gap_weight if projection_valid else None
                ),
                "negative_gap_denominator": (
                    negative_gap_weight if projection_valid else None
                ),
            })
        direct["visible_checkpoint_metrics"] = visible_rows
        projection_complete = bool(visible_rows) and all(
            "note" not in row and row.get("projection_valid") is True
            for row in visible_rows
        )
        direct["visible_projection_status"] = "OK" if projection_complete \
            else "VISIBLE_PROJECTION_INCOMPLETE"
        direct["direct_denominator_status"] = (
            "OK_C_VISIBLE_PROJECTION" if projection_complete
            else "DIRECT_DENOMINATOR_UNAVAILABLE_VISIBLE_PROJECTION"
        )
        values = [r["visible_recall"] for r in visible_rows
                  if r.get("visible_recall") is not None]
        direct["visible_truth_recall_mean"] = (
            sum(values) / len(values) if values else None
        )
        direct["c_visible_truth_recall"] = direct["visible_truth_recall_mean"]
        direct["c_total_published_prechunk_recall"] = (
            direct["visible_truth_recall_mean"])
        direct["h_exact_truth_recall"] = direct.get("exact_truth_recall")
        if c_trace_invalid:
            direct.setdefault("errors", []).extend(c_trace_invalid)
            direct["status"] = "INVALID_DIRECT_TRACE"
        c_query_complete = (
            not guarded_query_ids
            or all(
                "offered_query_attempt_ids" in raw
                and any(key in raw for key in (
                    "published_query_attempt_ids", "published_gaps"))
                for raw in traces
            )
        )
        relation_complete = all(
            str(raw.get("contract") or "") not in {"P1_TYPED", "P1_BRIDGE"}
            or "published_relations" in raw
            for raw in traces
        )
        h_query_complete = (
            bool(direct.get("query_guard_complete"))
            if direct.get("h_checkpoint_count") else True
        )
        h_relation_complete = (
            bool(direct.get("relation_guard_complete"))
            if direct.get("h_checkpoint_count") else True
        )
        h_guard_complete = (
            bool(direct.get("selector_guards_complete"))
            if direct.get("h_checkpoint_count") else True
        )
        direct["h_selector_guards_complete"] = h_guard_complete
        direct["c_query_guard_complete"] = c_query_complete
        direct["c_relation_guard_complete"] = relation_complete
        direct["query_guard_complete"] = h_query_complete and c_query_complete
        direct["relation_guard_complete"] = h_relation_complete and relation_complete
        c_guards_complete = (
            not c_trace_invalid
            and projection_complete
            and c_query_complete
            and relation_complete
        )
        direct["selector_guards_complete"] = (
            h_guard_complete
            and c_guards_complete
        )
        def mean_metric(name: str, rows=visible_rows) -> float | None:
            observed = [
                float(row[name]) for row in rows
                if row.get(name) is not None
            ]
            return sum(observed) / len(observed) if observed else None

        def pooled_metric(
            numerator_name: str, denominator_name: str, rows=visible_rows
        ) -> float | None:
            eligible = [
                row for row in rows
                if row.get(numerator_name) is not None
                and row.get(denominator_name) is not None
            ]
            if len(eligible) != len(rows):
                return None
            numerator = sum(float(row[numerator_name]) for row in eligible)
            denominator = sum(float(row[denominator_name]) for row in eligible)
            return numerator / denominator if denominator > 0 else None

        # Keep the macro checkpoint means above, but also freeze the sufficient counts for
        # a pooled micro view.  The two estimands answer different questions: macro gives an
        # equal vote to each close checkpoint, while micro gives an equal vote to each visible
        # truth-atom opportunity.  Persisting integer counts here avoids the irrecoverable (and
        # generally wrong) practice of multiplying a rounded mean by a later denominator.
        c_micro_counts = {
            "candidate_coverage": {
                "numerator": sum(
                    len(set(map(str, row.get("candidate_visible_atom_ids") or ())))
                    for row in visible_rows
                    if row.get("candidate_coverage") is not None
                ),
                "denominator": sum(
                    len(set(map(str, row.get("explicitly_visible_atom_ids") or ())))
                    for row in visible_rows
                    if row.get("candidate_coverage") is not None
                ),
            },
            "total_published_prechunk_recall": {
                "numerator": sum(
                    len(set(map(str, row.get("published_visible_atom_ids") or ())))
                    for row in visible_rows
                    if row.get("c_total_published_prechunk_recall") is not None
                ),
                "denominator": sum(
                    len(set(map(str, row.get("explicitly_visible_atom_ids") or ())))
                    for row in visible_rows
                    if row.get("c_total_published_prechunk_recall") is not None
                ),
            },
            "weighted_evidence_recall": {
                "numerator": sum(
                    float(row["weighted_evidence_numerator"])
                    for row in visible_rows
                    if row.get("weighted_evidence_numerator") is not None
                ),
                "denominator": sum(
                    float(row["weighted_evidence_denominator"])
                    for row in visible_rows
                    if row.get("weighted_evidence_denominator") is not None
                ),
            },
            "negative_gap_recall": {
                "numerator": sum(
                    float(row["negative_gap_numerator"])
                    for row in visible_rows
                    if row.get("negative_gap_numerator") is not None
                ),
                "denominator": sum(
                    float(row["negative_gap_denominator"])
                    for row in visible_rows
                    if row.get("negative_gap_denominator") is not None
                ),
            },
        }
        for metric, counts in c_micro_counts.items():
            numerator = float(counts["numerator"])
            denominator = float(counts["denominator"])
            if (
                not math.isfinite(numerator)
                or not math.isfinite(denominator)
                or denominator < 0
                or numerator < 0
                or numerator > denominator
            ):
                c_trace_invalid.append(
                    f"pooled {metric} counts are inconsistent: "
                    f"{numerator}/{denominator}"
                )

        c_metrics = {
            "applicable": True,
            "status": (
                "INVALID_DIRECT_TRACE" if c_trace_invalid
                else "OK" if projection_complete
                else "DIRECT_PROJECTION_UNAVAILABLE"
            ),
            "errors": list(c_trace_invalid),
            "checkpoint_count": len(traces),
            "selector_normalization": c_normalization,
            "checkpoint_metrics": visible_rows,
            "micro_counts": c_micro_counts,
            "direct_denominator_status": direct["direct_denominator_status"],
            "eligible_denominators": {
                field: sum(
                    float((row.get("eligible_denominators") or {}).get(field) or 0)
                    for row in visible_rows
                )
                for field in (
                    "candidate_coverage",
                    "selector_conditional_recall",
                    "weighted_evidence_recall",
                    "total_published_prechunk_recall",
                    "critical_truth_recall",
                    "contradiction_pair_recall",
                    "grounded_negative_atom_recall",
                    "negative_query_trace_recall",
                    "unresolved_gap_recall",
                    "negative_gap_recall",
                    "typed_relation_coverage",
                    "contradiction_role_pair_recall",
                )
            },
            "candidate_coverage": mean_metric("candidate_coverage"),
            "selector_conditional_recall": mean_metric(
                "selector_conditional_recall"),
            # Pool frozen weights across close checkpoints. A macro average would let a
            # one-atom checkpoint outweigh a checkpoint carrying many high-weight atoms.
            "weighted_evidence_recall": pooled_metric(
                "weighted_evidence_numerator", "weighted_evidence_denominator"),
            "selected_token_precision": pooled_metric(
                "relevant_published_span_tokens", "published_span_tokens"),
            "weighted_truth_per_100_rendered_tokens": (
                100.0
                * sum(
                    float(row["weighted_evidence_numerator"])
                    for row in visible_rows
                    if row.get("weighted_evidence_numerator") is not None
                )
                / sum(
                    float(row["published_rendered_tokens"])
                    for row in visible_rows
                    if row.get("published_rendered_tokens") is not None
                )
                if (
                    len([
                        row for row in visible_rows
                        if row.get("weighted_evidence_numerator") is not None
                        and row.get("published_rendered_tokens") is not None
                    ]) == len(visible_rows)
                    and sum(
                        float(row["published_rendered_tokens"])
                        for row in visible_rows
                        if row.get("published_rendered_tokens") is not None
                    ) > 0
                ) else None
            ),
            "materialization_ratio": pooled_metric(
                "published_rendered_tokens", "offered_material_tokens"),
            "token_trace_complete": bool(visible_rows) and all(
                row.get("token_trace_status") == "OK" for row in visible_rows
            ),
            "total_published_prechunk_recall": mean_metric(
                "c_total_published_prechunk_recall"),
            "critical_truth_recall": mean_metric("critical_truth_recall"),
            "contradiction_pair_recall": mean_metric(
                "contradiction_pair_recall"),
            "grounded_negative_atom_recall": mean_metric(
                "grounded_negative_atom_recall"),
            "negative_query_trace_recall": mean_metric(
                "negative_query_trace_recall"),
            "unresolved_gap_recall": mean_metric("unresolved_gap_recall"),
            "negative_gap_recall": pooled_metric(
                "negative_gap_numerator", "negative_gap_denominator"),
            "typed_relation_coverage": mean_metric("typed_relation_coverage"),
            "contradiction_role_pair_recall": mean_metric(
                "contradiction_role_pair_recall"),
            "selector_guards_complete": c_guards_complete,
            "query_guard_complete": c_query_complete,
            "relation_guard_complete": relation_complete,
        }
        direct.setdefault("by_node", {})["C"] = c_metrics
        per_arm["direct_node_metrics"] = direct


def _frozen_pool_for(settings: Settings, task_id: str) -> dict:
    from .acquire import runner_pool_path

    path = runner_pool_path(settings, task_id)
    if not path.exists():
        return {}
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvaluationError(f"invalid frozen source pool {path}: {exc}") from exc
    from ..canonical import canonical_json
    from ..hashing import sha256_hex

    recorded = str(body.get("pool_sha256") or "")
    actual = sha256_hex(canonical_json(
        {key: value for key, value in body.items() if key != "pool_sha256"}))
    if recorded != actual:
        raise EvaluationError(
            f"{path} was edited: records pool {recorded}, hashes to {actual}")
    return body


def _citation_supports_for(settings: Settings, task_id: str, pool: dict, truth: dict, relate):
    """A real citation resolver, or an explicit unknown when the world is not readable.

    ``lambda claim_id, label: None`` is what stood here, so every citation was unknown, every
    claim lacked a supporting citation, and every quality metric was 0.0 for every arm.
    """

    occurrences = pool.get("occurrences")
    snapshots = pool.get("snapshots")
    if not isinstance(occurrences, list) or not occurrences:
        raise EvaluationError(
            f"citation measurement for {task_id} has no frozen source occurrences")
    if not isinstance(snapshots, dict) or not snapshots:
        raise EvaluationError(
            f"citation measurement for {task_id} has no frozen source snapshots")

    texts: dict[str, str] = {}
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    for content_hash, snap in snapshots.items():
        if not str(content_hash):
            raise EvaluationError(
                f"citation measurement for {task_id} has an unnamed frozen snapshot")
        if not isinstance(snap, dict):
            raise EvaluationError(
                f"citation snapshot {content_hash!r} for {task_id} is malformed")
        ref = snap.get("object_ref")
        if not ref:
            raise EvaluationError(
                f"citation snapshot {content_hash!r} for {task_id} has no object_ref")
        try:
            texts[str(content_hash)] = objects.get_bytes(str(ref)).decode("utf-8")
        except (
            CorruptObject, FileNotFoundError, KeyError, OSError, UnicodeDecodeError, ValueError,
        ) as exc:
            raise EvaluationError(
                f"citation snapshot {content_hash!r} for {task_id} is unreadable: "
                f"{type(exc).__name__}: {exc}"
            ) from exc

    url_to_content = {}
    occurrence_to_url: dict[str, str] = {}
    from ..evaluation.citation_support import _normalize

    for occurrence in occurrences:
        if not isinstance(occurrence, dict):
            raise EvaluationError(
                f"citation occurrence for {task_id} is malformed: "
                f"expected object, got {type(occurrence).__name__}")
        occurrence_id = str(occurrence.get("occurrence_id") or "")
        url = str(occurrence.get("url") or "")
        content_hash = str(occurrence.get("content_hash") or "")
        if not occurrence_id or not url or not content_hash:
            raise EvaluationError(
                f"citation occurrence {occurrence_id or '<unnamed>'!r} for {task_id} "
                "lacks occurrence_id, url, or content_hash")
        if content_hash not in texts:
            raise EvaluationError(
                f"citation occurrence {occurrence_id!r} for {task_id} "
                f"references missing snapshot {content_hash!r}")
        normalized_url = _normalize(url)
        if not normalized_url:
            raise EvaluationError(
                f"citation occurrence {occurrence_id!r} for {task_id} has an invalid URL")
        previous = url_to_content.get(normalized_url)
        if previous is not None and previous != content_hash:
            raise EvaluationError(
                f"citation URL {normalized_url!r} for {task_id} is ambiguously bound to "
                f"{previous!r} and {content_hash!r}")
        url_to_content[normalized_url] = content_hash
        occurrence_to_url[occurrence_id] = normalized_url
    if not url_to_content:
        raise EvaluationError(
            f"citation measurement for {task_id} has no URL-to-snapshot bindings")

    return _PerReportResolver(
        url_to_content, occurrence_to_url, texts, relate)


class _PerReportResolver:
    """One resolver per (task, report): the label map comes from the report being scored."""

    def __init__(self, url_to_content, occurrence_to_url, texts, relate) -> None:
        self._url_to_content = dict(url_to_content)
        self._occurrence_to_url = dict(occurrence_to_url)
        self._texts = texts
        self._relate = relate
        self._current = None
        self.records: list = []

    def bind(
        self,
        report_text: str,
        claim_texts: dict,
        *,
        allowed_occurrence_ids=(),
    ) -> None:
        from ..evaluation.citation_support import CitationSupportResolver

        if self._current is not None:
            self.records.extend(self._current.record())
        unknown = sorted(
            set(map(str, allowed_occurrence_ids)) - set(self._occurrence_to_url)
        )
        if unknown:
            raise EvaluationError(
                "arm citation lineage names occurrence ids outside the frozen task pool: "
                f"{unknown[:5]}"
            )
        allowed_urls = {
            self._occurrence_to_url[occurrence_id]
            for occurrence_id in map(str, allowed_occurrence_ids)
        }
        visible_bindings = {
            url: content_hash for url, content_hash in self._url_to_content.items()
            if url in allowed_urls
        }
        self._current = CitationSupportResolver(
            url_to_content=visible_bindings,
            content_texts=self._texts,
            judge_relation=self._relate,
            claim_texts=claim_texts,
            citation_map=parse_citation_map(report_text),
        )

    def __call__(self, claim_id: str, label: str):
        if self._current is None:
            return None
        return self._current(claim_id, label)

    def record(self) -> list:
        current = self._current.record() if self._current is not None else []
        return [*self.records, *current]


def _policy_sha(settings: Settings, observed: list) -> str:
    """The scoring policy's identity, including what the judge actually answered as."""
    returned = sorted({o["returned_model"] for o in observed if o.get("returned_model")})
    fingerprints = sorted({o["system_fingerprint"] for o in observed
                           if o.get("system_fingerprint")})
    return judge_policy_digest(
        {**settings.configs["judge"],
         "relation_prompt_sha256": relation_prompt_sha256()},
        requested_model=settings.judge_model(),
        returned_model=",".join(returned),
        system_fingerprint=",".join(fingerprints),
    )


def _write_judge_provenance(scores_dir: Path, artifact_id: str, settings: Settings,
                            observed: list, supports, *, task_id: str = "",
                            run_id: str = "", phase_id: str = "") -> None:
    """Who scored this task, under what, and how each citation resolved."""
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "task_id": task_id,
        "run_id": run_id,
        "phase_id": phase_id,
        "artifact_id": artifact_id,
        "requested_model": settings.judge_model(),
        "returned_models": sorted({o["returned_model"] for o in observed
                                   if o.get("returned_model")}),
        "system_fingerprints": sorted({o["system_fingerprint"] for o in observed
                                       if o.get("system_fingerprint")}),
        "judgments": len(observed),
        "relation_prompt_sha256": relation_prompt_sha256(),
        "judge_policy_sha256": _policy_sha(settings, observed),
        "citation_resolutions": supports.record() if hasattr(supports, "record") else [],
    }
    (scores_dir / f"{artifact_id}.judge.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
