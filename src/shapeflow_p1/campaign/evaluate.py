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

from ..evaluation.judge_client import DeepSeekJudge, JudgeUnavailable
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


def judge_relation_via(judge: DeepSeekJudge):
    """A blind relation classifier. Uncertainty stays uncertain rather than becoming a pass."""
    import asyncio

    def relate(claim: str, fact: str) -> str:
        try:
            response = asyncio.get_event_loop().run_until_complete(
                judge.judge(_RELATION_SYSTEM,
                            _RELATION_PROMPT.format(claim=claim, fact=fact)))
        except JudgeUnavailable:
            raise
        relation = str(response.data.get("relation", "uncertain"))
        return relation if relation in ("entail", "contradict", "unrelated") else "uncertain"

    return relate


async def evaluate_frozen(settings: Settings, *, repo: Path,
                          task_limit: Optional[int] = None) -> dict:
    """Score every task whose arms are all committed."""
    ledger, _store = open_run_ledger(settings)
    store = ObjectStore(settings.path("object_store"))
    client = provider_client_for(settings, "evaluator")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_REPORT"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
    )
    policy_sha = judge_policy_digest(
        settings.configs["judge"], requested_model=settings.judge_model(),
        returned_model=settings.judge_model(),
    )

    rows = ledger.raw_connection.execute(
        "SELECT w.task_id, a.result_object_ref FROM work_items w JOIN attempts a"
        " ON a.work_key = w.work_key WHERE a.state='COMMITTED'"
    ).fetchall()
    by_task: dict[str, list[dict]] = {}
    for row in rows:
        ref = row["result_object_ref"]
        if not (ref and store.verify(ref)):
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
        score = score_task(
            truth_body=truth["packet"], outputs=outputs, atom_texts=truth["atom_texts"],
            judge_relation=judge_relation_via(judge),
            citation_supports=lambda claim_id, label: None,
            judge_policy_sha256=policy_sha, claim_scope=settings.claim_scope,
        )
        write_scores(score, scores_dir / f"{task_id}.json")
        scored.append(task_id)

    ledger.close()
    return {"scored": scored, "skipped": skipped, "claim_scope": settings.claim_scope,
            "judge_policy_sha256": policy_sha}
