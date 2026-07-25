"""`evaluate` end to end, with a real ledger, a real object store and a scripted judge.

There was no test of this path at all, and it could not have passed one: the relation
classifier called ``get_event_loop().run_until_complete`` from inside the loop that was
already running it, so the first judged claim raised "this event loop is already running".
Every quality metric was 0.0 anyway, because ``citation_supports`` returned None for every
citation and ``covered`` requires a supporting one.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from shapeflow_p1.campaign.evaluate import evaluate_frozen
from shapeflow_p1.campaign.settings import Settings
from shapeflow_p1.experiment.ledger import Ledger
from shapeflow_p1.object_store import ObjectStore

REPO = Path(__file__).resolve().parents[2]

_REPORT = """\
The reactor reached 41% efficiency in 2025 [1].

Sources:
[1] Efficiency review -- https://example.com/efficiency
"""

_PAGE = "The reactor reached 41% efficiency in 2025, the review reported."


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


def _truth(task_id: str) -> dict:
    packet = {
        "task_id": task_id,
        "content_sha256": "t" * 64,
        "required_facets": ["f1"],
        "facets": [{"facet_id": "f1", "text": "efficiency"}],
        "atomic_evidence": [
            {"atom_id": "a1", "facet_id": "f1", "critical": False, "weight": 1.0,
             "spans": [{"content_hash": "h1", "byte_start": 0, "byte_end": 10}]},
        ],
    }
    return {"packet": packet, "atom_texts": {"a1": "41% efficiency in 2025"}}


def _world(settings, task_id: str) -> None:
    """One frozen page, published exactly the way acquisition publishes it."""
    objects = ObjectStore(settings.path("frozen_corpus_for_runner") / "objects")
    ref = objects.put_bytes(_PAGE.encode("utf-8"))
    pool = {
        "task_id": task_id,
        "occurrences": [{"occurrence_id": "o1", "url": "https://example.com/efficiency",
                         "title": "Efficiency review", "snippet_content": "41%",
                         "content_hash": "h1", "vendor_visible_order": 0}],
        "snapshots": {"h1": {"object_ref": ref.key, "byte_len": len(_PAGE),
                             "raw_content_format": "exa_text",
                             "normalization_version": "v1",
                             "fetched_at_utc": "2026-07-25T00:00:00Z"}},
    }
    from shapeflow_p1.canonical import canonical_json
    from shapeflow_p1.hashing import sha256_hex

    pool["pool_sha256"] = sha256_hex(canonical_json(pool))
    path = settings.path("frozen_corpus_for_runner") / "pools" / f"{task_id}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(pool), encoding="utf-8")

    truth_dir = settings.path("truth_packets")
    truth_dir.mkdir(parents=True, exist_ok=True)
    (truth_dir / f"{task_id}.json").write_text(json.dumps(_truth(task_id)), encoding="utf-8")


def _committed_cell(settings, ledger: Ledger, task_id: str, arm_id: str) -> None:
    store = ObjectStore(settings.path("object_store"))
    record = {
        "cell": {"arm": {"arm_id": arm_id}, "replicate_id": "0", "task_id": task_id},
        "variant_id": f"{arm_id}-v1",
        "final_report": _REPORT,
        "checkpoints": [],
    }
    from shapeflow_p1.canonical import canonical_json

    ref = store.put_bytes(canonical_json(record))
    key = ledger.ensure_work_item(
        protocol_sha="p", split="FORMATIVE_SCREEN", phase_id="run-screen",
        task_id=task_id, arm_id=arm_id, variant_id=f"{arm_id}-v1",
        replicate_id="0", checkpoint_hash="B1", stage_version="v1",
    )
    attempt = ledger.claim(key, "w", lease_seconds=60)
    ledger.advance(attempt.attempt_id, "MATERIALIZED")
    ledger.advance(attempt.attempt_id, "VALIDATED")
    ledger.commit(attempt.attempt_id, result_object_ref=ref.key)


class _NoClient:
    """The judge is monkeypatched, so this only has to exist to be called."""

    def deepseek_transport(self, **_kw):
        async def transport(_body):  # pragma: no cover - never dispatched
            raise AssertionError("the scripted judge should have intercepted this")

        return transport


class _ScriptedJudge:
    """Answers `entail` for anything mentioning 41%, `unrelated` otherwise."""

    def __init__(self) -> None:
        self.calls = 0

    async def judge(self, system, user, *, validate=None):
        from shapeflow_p1.evaluation.judge_client import JudgeResponse

        self.calls += 1
        relation = "entail" if "41" in user else "unrelated"
        return JudgeResponse(
            data={"relation": relation}, requested_model="deepseek-v4-flash",
            returned_model="deepseek-v4-flash", usage={}, request_id=f"r{self.calls}",
            system_fingerprint="fp_test",
        )


async def test_evaluate_scores_a_frozen_task_end_to_end(settings, tmp_path, monkeypatch):
    task_id = "T-eval-1"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()

    judge = _ScriptedJudge()
    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: judge)
    monkeypatch.setattr(evaluate_module, "provider_client_for",
                        lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    out = await evaluate_frozen(settings, repo=REPO)

    assert out["scored"] == [task_id], out
    assert judge.calls > 0, "no claim was ever judged"

    body = json.loads(
        (settings.path("judgments") / f"{task_id}.json").read_text(encoding="utf-8"))
    arm = body["per_arm"]["P0:0"]
    # The number that was structurally 0.0 for every arm of every task.
    assert arm["weighted_required_atom_recall"] > 0.0, arm
    assert arm["citation_correctness"] > 0.0, arm


async def test_the_judge_that_actually_answered_is_recorded(settings, tmp_path, monkeypatch):
    task_id = "T-eval-2"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()

    judge = _ScriptedJudge()
    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: judge)
    monkeypatch.setattr(evaluate_module, "provider_client_for",
                        lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    await evaluate_frozen(settings, repo=REPO)

    provenance = json.loads(
        (settings.path("judgments") / f"{task_id}.judge.json").read_text(encoding="utf-8"))
    assert provenance["returned_models"] == ["deepseek-v4-flash"]
    assert provenance["system_fingerprints"] == ["fp_test"]
    assert provenance["judgments"] > 0
    assert len(provenance["relation_prompt_sha256"]) == 64
    # Every citation resolution is explicable rather than an unexplained zero.
    assert provenance["citation_resolutions"], provenance


async def test_a_task_whose_block_was_never_frozen_is_not_scored(settings, tmp_path,
                                                                 monkeypatch):
    """Half a paired comparison is not an observation (plan §16.3)."""
    task_id = "T-eval-3"
    _world(settings, task_id)
    settings.path("provider_ledger").parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(settings.path("provider_ledger")))
    _committed_cell(settings, ledger, task_id, "P0")
    ledger.close()

    import shapeflow_p1.campaign.evaluate as evaluate_module

    monkeypatch.setattr(evaluate_module, "DeepSeekJudge", lambda *a, **kw: _ScriptedJudge())
    monkeypatch.setattr(evaluate_module, "provider_client_for",
                        lambda *a, **kw: _NoClient())
    monkeypatch.setattr(
        evaluate_module, "open_run_ledger",
        lambda s: (Ledger(str(s.path("provider_ledger"))), None))

    empty = tmp_path / "frozen-blocks"
    empty.mkdir()
    out = await evaluate_frozen(settings, repo=REPO, frozen_blocks_dir=empty)
    assert out["scored"] == []
