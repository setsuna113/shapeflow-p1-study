"""`evaluate`: score the frozen outputs, as the evaluator, once they cannot change.

Reads committed cell outputs from the runner's ledger, pairs each task's arms with that task's
TruthPacket, and writes one score file per task under the evaluator tree. Nothing here can feed
back into a treatment: it runs after the fact, under a different identity, and its outputs live
in a directory the runner cannot read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

from ..evaluation.citation_support import parse_citation_map
from ..evaluation.judge_client import DeepSeekJudge
from ..evaluation.runner import ArmOutput, judge_policy_digest, score_task, write_scores
from ..object_store import ObjectStore
from ..providers.provider_client import PROVIDER_KEY_PLACEHOLDER
from .screen import open_run_ledger, provider_client_for
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


def relation_prompt_sha256() -> str:
    from ..canonical import canonical_json
    from ..hashing import sha256_hex

    return sha256_hex(canonical_json({
        "version": RELATION_PROMPT_VERSION,
        "system": _RELATION_SYSTEM,
        "user": _RELATION_PROMPT,
        "schema": {"relation": ["entail", "contradict", "unrelated", "uncertain"]},
    }))


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
                          task_limit: Optional[int] = None,
                          run_id: Optional[str] = None,
                          phase_id: Optional[str] = None,
                          frozen_blocks_dir: Optional[Path] = None) -> dict:
    """Score the frozen outputs of one run and phase.

    Scoped on purpose. It used to score every COMMITTED row in the ledger regardless of which
    run or phase produced it, so a diagnostic and a screen ended up in one pooled number, and
    a block that was never frozen -- half a paired comparison -- was scored as if it were.
    """
    import asyncio

    loop = asyncio.get_running_loop()
    ledger, _store = open_run_ledger(settings)
    store = ObjectStore(settings.path("object_store"))
    client = provider_client_for(settings, "evaluator")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_REPORT"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
    )

    complete = _frozen_task_ids(frozen_blocks_dir)
    sql = ("SELECT w.task_id, w.phase_id, a.run_id, a.result_object_ref FROM work_items w"
           " JOIN attempts a ON a.work_key = w.work_key WHERE a.state='COMMITTED'")
    params: list = []
    if phase_id:
        sql += " AND w.phase_id=?"
        params.append(phase_id)
    if run_id:
        sql += " AND a.run_id=?"
        params.append(run_id)
    rows = ledger.raw_connection.execute(sql, params).fetchall()

    by_task: dict[str, list[dict]] = {}
    for row in rows:
        ref = row["result_object_ref"]
        if not (ref and store.verify(ref)):
            continue
        if complete is not None and row["task_id"] not in complete:
            continue
        record = json.loads(store.get_bytes(ref).decode("utf-8"))
        by_task.setdefault(row["task_id"], []).append(record)

    truth_dir = settings.path("truth_packets")
    scores_dir = settings.path("judgments")
    scored: list[str] = []
    skipped: list[str] = []

    for task_id, records in sorted(by_task.items()):
        if task_limit is not None and len(scored) >= task_limit:
            break
        truth_path = truth_dir / f"{task_id}.json"
        if not truth_path.exists():
            skipped.append(f"{task_id}: no truth packet")
            continue
        truth = json.loads(truth_path.read_text(encoding="utf-8"))
        outputs = [
            ArmOutput(
                task_id=task_id, arm_id=r["cell"]["arm"]["arm_id"],
                variant_id=r.get("variant_id", ""), replicate_id=r["cell"]["replicate_id"],
                final_report=r.get("final_report", ""),
                frozen=True,                       # only committed outputs reach this loop
                terminal_failure=bool(r.get("error")),
            )
            for r in records
        ]
        observed: list[dict] = []
        relate = judge_relation_via(judge, loop, observed=observed)
        pool = _frozen_pool_for(settings, task_id)
        supports = _citation_supports_for(settings, task_id, pool, truth, relate)

        score = await asyncio.to_thread(
            score_task,
            truth_body=truth["packet"], outputs=outputs, atom_texts=truth["atom_texts"],
            judge_relation=relate,
            citation_supports=supports,
            judge_policy_sha256=_policy_sha(settings, observed),
            claim_scope=settings.claim_scope,
        )
        _add_visible_recall(settings, score, records, truth)
        write_scores(score, scores_dir / f"{task_id}.json")
        _write_judge_provenance(scores_dir, task_id, settings, observed, supports)
        scored.append(task_id)

    ledger.close()
    return {"scored": scored, "skipped": skipped, "claim_scope": settings.claim_scope,
            "judge_policy_sha256": _policy_sha(settings, []),
            "relation_prompt_sha256": relation_prompt_sha256()}


def _add_visible_recall(settings: Settings, score, records: list, truth: dict) -> None:
    """Score C arms against what their compressor could actually see.

    A C_VISIBLE arm is a reducer. Its input is the exact ``researcher_messages`` bytes at the
    close boundary, and an atom that never entered those bytes was lost upstream -- charging
    the reducer for it attributes an upstream loss to the compressor, which
    visible_truth_projection.py calls the single most likely way to manufacture a false
    C_VISIBLE failure. The projection existed and was imported by nothing but its own tests,
    so every C arm was scored against the full TruthPacket.

    Recorded alongside the full-truth number rather than replacing it: the two answer
    different questions, and a reader needs to see which one a verdict rests on.
    """
    from ..evaluation.visible_truth_projection import project_visible_truth, visible_recall
    from ..odr.checkpoints import CheckpointStore
    from ..strategies.pipeline import spans_from_visible_view
    from ..strategies.visible_view import build_visible_view

    store = CheckpointStore(settings.path("checkpoints"))
    atom_texts = truth.get("atom_texts") or {}
    required = {a["atom_id"] for a in (truth["packet"].get("atomic_evidence") or [])}

    for record in records:
        arm_key = f"{record['cell']['arm']['arm_id']}:{record['cell']['replicate_id']}"
        per_arm = score.per_arm.get(arm_key)
        if per_arm is None or record["cell"]["arm"]["arm_id"] not in ("C_VISIBLE",
                                                                     "H_PLUS_C_VISIBLE"):
            continue
        digest = next((c["digest"] for c in record.get("checkpoints") or []
                       if c.get("kind") in ("C", "CCheckpoint")), "")
        if not digest:
            per_arm["visible_recall"] = None
            per_arm["visible_recall_note"] = "no C checkpoint was stored for this arm"
            continue
        try:
            checkpoint = store.get(digest)
        except (FileNotFoundError, ValueError) as e:
            per_arm["visible_recall"] = None
            per_arm["visible_recall_note"] = f"{type(e).__name__}: {e}"
            continue

        from ..evidence.chunkers import WhitespaceTokenizer

        view = build_visible_view(checkpoint.researcher_messages)
        spans = spans_from_visible_view(
            view.view_bytes, view_hash=view.view_hash, messages=view.message_segments,
            tokenizer=WhitespaceTokenizer())
        span_texts = {
            s.get("visible_span_id") or s.get("span_id", ""):
            view.view_bytes[s["byte_start"]:s["byte_end"]].decode("utf-8", errors="replace")
            for s in spans
        }
        projection = project_visible_truth(
            checkpoint_hash=digest, visible_view_hash=view.view_hash,
            spans=spans, span_texts=span_texts,
            atoms=[(a, atom_texts.get(a, "")) for a in sorted(required)],
        )
        retained = score.matched_atoms_by_arm.get(arm_key, [])
        per_arm["visible_recall"] = visible_recall(projection, retained)
        per_arm["visible_truth_projection_sha256"] = projection.content_sha256
        per_arm["explicitly_visible_atoms"] = len(projection.explicitly_visible_ids)


def _frozen_task_ids(directory: Optional[Path]) -> Optional[set]:
    """Task ids whose blocks were frozen whole, or None when no directory was given.

    Plan §16.3 makes the pre-registered block the unit. Scoring an unfrozen block scores half
    a paired comparison, and half a pair is not an observation.
    """
    if directory is None:
        return None
    directory = Path(directory)
    if not directory.exists():
        return set()
    ids: set = set()
    for path in sorted(directory.glob("*.json")):
        try:
            body = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if body.get("complete"):
            ids.update(c.get("task_id", "") for c in body.get("cells", []))
    ids.discard("")
    return ids


def _frozen_pool_for(settings: Settings, task_id: str) -> dict:
    from .acquire import runner_pool_path

    path = runner_pool_path(settings, task_id)
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _citation_supports_for(settings: Settings, task_id: str, pool: dict, truth: dict, relate):
    """A real citation resolver, or an explicit unknown when the world is not readable.

    ``lambda claim_id, label: None`` is what stood here, so every citation was unknown, every
    claim lacked a supporting citation, and every quality metric was 0.0 for every arm.
    """
    from ..evaluation.citation_support import CitationSupportResolver

    occurrences = pool.get("occurrences") or []
    if not occurrences:
        return lambda claim_id, label: None

    texts: dict[str, str] = {}
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    for content_hash, snap in (pool.get("snapshots") or {}).items():
        ref = snap.get("object_ref")
        if not ref:
            continue
        try:
            texts[content_hash] = objects.get_bytes(ref).decode("utf-8", errors="replace")
        except Exception:  # noqa: BLE001 - an unreadable page is an unknown, not a False
            continue

    url_to_content = {}
    from ..evaluation.citation_support import _normalize

    for occurrence in occurrences:
        url = str(occurrence.get("url") or "")
        content_hash = str(occurrence.get("content_hash") or "")
        if url and content_hash:
            url_to_content.setdefault(_normalize(url), content_hash)

    return _PerReportResolver(url_to_content, texts, relate)


class _PerReportResolver:
    """One resolver per (task, report): the label map comes from the report being scored."""

    def __init__(self, url_to_content, texts, relate) -> None:
        from ..evaluation.citation_support import CitationSupportResolver

        self._make = lambda report, claims: CitationSupportResolver(
            url_to_content=url_to_content, content_texts=texts,
            judge_relation=relate, claim_texts=claims,
            citation_map=parse_citation_map(report),
        )
        self._current = None
        self.records: list = []

    def bind(self, report_text: str, claim_texts: dict) -> None:
        self._current = self._make(report_text, claim_texts)

    def __call__(self, claim_id: str, label: str):
        if self._current is None:
            return None
        return self._current(claim_id, label)

    def record(self) -> list:
        return self._current.record() if self._current is not None else []


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


def _write_judge_provenance(scores_dir: Path, task_id: str, settings: Settings,
                            observed: list, supports) -> None:
    """Who scored this task, under what, and how each citation resolved."""
    scores_dir = Path(scores_dir)
    scores_dir.mkdir(parents=True, exist_ok=True)
    body = {
        "task_id": task_id,
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
    (scores_dir / f"{task_id}.judge.json").write_text(
        json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
