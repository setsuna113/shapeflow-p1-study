"""The canary's own assertions, tested against synthetic runs.

The canary is the gate that decides whether screening starts, so its checks have to fail when
they should. Each test below constructs the exact shape of a P1 arm that *looks* like it worked
-- no selector decode, a relabelled P0 report, a CPU control that quietly called the model --
and requires the corresponding check to go red.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from shapeflow_p1.campaign.canary import CANARY_CHECKS, _p1_differs_from_p0, _verify
from shapeflow_p1.campaign.schedule import ArmSpec, build_blocks, cell_key
from shapeflow_p1.campaign.settings import Settings

REPO = Path(__file__).resolve().parents[2]

ARMS = [
    ArmSpec("P0", "P0", "P0"),
    ArmSpec("H_ID", "H02", "P0"),
    ArmSpec("CPU_LEXICAL", "H00-CPU", "P0"),
    ArmSpec("SHORT_PROSE", "H00-PROSE", "P0"),
]


@pytest.fixture()
def settings(tmp_path):
    return Settings.load(REPO, data_root=tmp_path)


@pytest.fixture()
def proved_repo(tmp_path, settings):
    """A repo whose installer already proved the runner is shut out of both trees.

    Cross-uid isolation cannot be demonstrated from inside one process, so the canary reads
    the proof install_host.sh generates. Staging it here is what makes the healthy case
    healthy; without it the check fails, which is the point.
    """
    import json

    for name in ("evaluator_root", "steward_root"):
        settings.path(name).mkdir(parents=True, exist_ok=True)
    repo = tmp_path / "repo"
    (repo / "reports").mkdir(parents=True)
    (repo / "reports" / "CREDENTIAL_ISOLATION.json").write_text(json.dumps({
        "denied": {"sfrunner": True, "sfinfer": True, "sfsteward": True,
                   "sfevaluator": True},
        "provider_can_read": True,
        "runner_cannot_list": ["evaluator", "steward"],
    }), encoding="utf-8")
    return repo


def _manifest():
    manifest = build_blocks(
        execution_binding_sha256="e" * 64,
        protocol_sha="a" * 64,
        split="FORMATIVE_SCREEN",
        task_ids=["T1"],
        arms=ARMS,
        seeds=[1],
        layer="causal",
        claim_scope="FORMATIVE_ONLY",
    )
    canary_counts = {arm.arm_id: 1 for arm in ARMS}
    screen_counts = {arm.arm_id: 24 for arm in ARMS}
    manifest.notes["gpu_budget_projection_plan"] = {
        "version": "canary_gpu_projection_v2",
        "projection_method": "per_arm_observed_max",
        "canary_cells": len(manifest.cells),
        "screen_cells": 96,
        "planned_total_cells": len(manifest.cells) + 96,
        "canary_cells_by_arm": canary_counts,
        "screen_cells_by_arm": screen_counts,
        "arm_variant_ids": {
            arm.arm_id: f"{arm.page_variant}+{arm.close_variant}" for arm in ARMS
        },
        "screen_task_count": 24,
        "screen_arm_count": 4,
        "screen_task_ids_sha256": "b" * 64,
        "screen_arm_ids_sha256": "c" * 64,
        "screen_assignment_sha256": "d" * 64,
    }
    return manifest


def _outputs(manifest, *, p1_report="P1 report", p0_report="P0 report", counts=None):
    records = {}
    for cell in manifest.cells:
        arm = cell.arm.arm_id
        checkpoint = f"H-{arm}"
        selector_op = (
            "PAGE_P1_SELECTOR_LOCAL"
            if arm in {"H_ID", "SHORT_PROSE"} else None
        )
        default_counts = {
            "page_batches_deferred": 1,
            "page_batches_reduced": 1,
            "page_fallbacks": 0,
            "close_reduced": 1,
            "close_failed": 0,
            "rendered_output_count": 1 if arm == "SHORT_PROSE" else 0,
            "max_rendered_tokens": 128 if arm == "SHORT_PROSE" else 0,
            "total_rendered_tokens": 128 if arm == "SHORT_PROSE" else 0,
        }
        records[cell_key(cell)] = {
            "cell": cell.content(),
            "work_key": f"WK-{arm}",
            "final_report": p0_report if arm == "P0" else p1_report,
            "events": [
                {"kind": "PAGE_BATCH_REDUCED", "checkpoint": checkpoint,
                 "fell_back": False, "failure": None, "siblings": 1},
                {"kind": "TOOL_BATCH_PUBLISHED", "checkpoint": checkpoint,
                 "atomic_publish": True, "sibling_count": 1,
                 "tool_call_ids": [f"call-{arm}"]},
            ],
            "checkpoints": [{"kind": "HCheckpoint", "digest": "d"},
                            {"kind": "CCheckpoint", "digest": "e"}],
            "direct_node_records": (
                [] if arm == "P0" else [{
                    "node": "H",
                    "checkpoint_hash": checkpoint,
                    "stage": "single",
                    "offered_span_ids": [f"span-{arm}"],
                    "offered_source_occurrence_ids": [f"occ-{arm}"],
                }]
            ),
            "work_summary": {
                "telemetry_complete": True,
                "overlap_valid": True,
                "service_seconds": 10.0,
                "by_op": ({
                    selector_op: {"count": 1, "completion_tokens": 120}
                } if selector_op else {}),
            },
            "counts": counts or default_counts,
        }
    return records


def _states(manifest, value="COMMITTED"):
    return {cell_key(c): value for c in manifest.cells}


def _provider_ledger(
    settings,
    *,
    max_tokens=768,
    work_key="WK-H_ID",
    op_class="PAGE_P1_SELECTOR_LOCAL",
    call_id="c1",
    attempt_id="a1",
    state="COMMITTED",
):
    """A real provider ledger with one committed inference attempt.

    The ledger-derived checks read it rather than a stub, because their whole failure mode
    was reading an absent ledger as "nothing was dispatched" and passing.
    """
    import json

    from shapeflow_p1.experiment.ledger import Ledger
    from shapeflow_p1.object_store import ObjectStore

    path = settings.data_root / str(settings.get("week1", "paths", "provider_ledger"))
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(path))
    store = ObjectStore(
        settings.data_root / str(settings.get("week1", "paths", "provider_root")) / "objects")
    ref = store.put_bytes(json.dumps({"model": "m", "max_tokens": max_tokens}).encode())
    with ledger.transaction() as cur:
        cur.execute(
            "INSERT INTO external_calls(call_id, provider, op_class, call_key, work_key,"
            " state, attempt_count, created_at, updated_at)"
            " VALUES (?,?,?,?,?,?,1,1,1)",
            (call_id, "vllm", op_class, f"k-{call_id}", work_key, state))
        cur.execute(
            "INSERT INTO external_call_attempts(attempt_id, call_id, attempt_ordinal, state,"
            " request_object_ref, usage_json, opened_at)"
            " VALUES (?,?,0,?,?,?,1)",
            (attempt_id, call_id, state, ref.key, '{"completion_tokens": 120}'))
    ledger.close()


def _run(settings, monkeypatch, inference):
    import shapeflow_p1.campaign.canary as canary

    manifest = _manifest()
    _provider_ledger(settings)
    monkeypatch.setattr(canary, "_inference_events", lambda _s, **_kw: inference)
    monkeypatch.setattr(canary, "_open_external_calls", lambda _s, **_kw: 0)
    monkeypatch.setattr(canary, "_patched_graph_check",
                        lambda _r: {"name": "patched_graph_invoked", "status": "PASS",
                                    "detail": "stubbed"})
    return manifest, canary


def _good_inference():
    return [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-H_ID",
         "completion_tokens": 120, "prompt_tokens": 900},
        {"op_class": "PAGE_P0_SUMMARY", "work_key": "WK-P0",
         "completion_tokens": 400, "prompt_tokens": 3000},
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-SHORT_PROSE",
         "completion_tokens": 200, "prompt_tokens": 900},
    ]


def _provider_audit(
    outputs: dict,
    *,
    inference=None,
    cap: float = 540_000.0,
    reserved: float = 0.0,
    settled: float = 40.0,
    open_attempts: int = 0,
) -> dict:
    inference = list(_good_inference() if inference is None else inference)
    work_keys = sorted(record["work_key"] for record in outputs.values())
    rows = []
    for work_key in work_keys:
        record = next(
            value for value in outputs.values() if value["work_key"] == work_key)
        work_settled = float(record["work_summary"]["service_seconds"])
        ops = []
        for event in inference:
            if event["work_key"] != work_key:
                continue
            op_class = event["op_class"]
            completion = int(event["completion_tokens"])
            ops.append({
                "op_class": op_class,
                "attempt_count": 1,
                "committed_attempt_count": 1,
                "prompt_tokens": int(event["prompt_tokens"]),
                "completion_tokens": completion,
                "cached_prompt_tokens": None,
                "max_completion_tokens_observed": completion,
                "settled_gpu_seconds": 0.0,
                "selector_request_max_tokens": (
                    [768] if op_class in {
                        "PAGE_P1_SELECTOR_LOCAL",
                        "PAGE_P1_SELECTOR_GLOBAL",
                        "COMPRESSOR_P1_SELECTOR",
                    } else []
                ),
            })
        if not ops:
            # Every synthetic cell still performs ordinary researcher work; only selector
            # events are abbreviated by _good_inference().
            ops.append({
                "op_class": "RESEARCHER_REACT",
                "attempt_count": 1,
                "committed_attempt_count": 1,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "cached_prompt_tokens": None,
                "max_completion_tokens_observed": 0,
                "settled_gpu_seconds": work_settled,
                "selector_request_max_tokens": [],
            })
        else:
            ops[0]["settled_gpu_seconds"] = work_settled
        rows.append({
            "work_key": work_key,
            "open_attempts": open_attempts if work_key == work_keys[0] else 0,
            "settled_gpu_seconds": work_settled,
            "ops": ops,
        })
    return {
        "version": "canary_audit_v1",
        "work_keys": work_keys,
        "work": rows,
        "open_attempts": open_attempts,
        "gpu_budget": {
            "resource": "gpu_seconds",
            "cap": cap,
            "reserved": reserved,
            "settled": settled,
            "remaining": cap - reserved - settled,
        },
    }


def test_a_healthy_run_passes_every_check(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    failed = [c for c in checks if c["status"] != "PASS"]
    assert failed == [], failed
    assert {c["name"] for c in checks} >= set(CANARY_CHECKS)
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["canary_actual_gpu_seconds"] == 40.0
    assert projection["canary_mean_gpu_seconds_per_cell"] == 10.0
    assert projection["planned_total_cells"] == 100
    assert projection["projected_total_gpu_seconds"] == 1000.0
    assert projection["hard_cap_gpu_seconds"] == 150 * 3600
    assert projection["projected_margin_gpu_seconds"] == 150 * 3600 - 1000
    assert projection["gpu_settlement_abs_tolerance_seconds"] == 1e-6
    assert projection["gpu_settlement_rel_tolerance"] == 1e-9
    assert projection["gate_scope"] == "cost_only_no_quality"
    assert projection["hard_budget_authority"] == "provider_budget_ledger"


@pytest.mark.parametrize(
    ("mutation", "detail"),
    [
        (lambda summary: summary.pop("service_seconds"), "service_seconds is not numeric"),
        (lambda summary: summary.__setitem__("service_seconds", 0), "finite and positive"),
        (lambda summary: summary.__setitem__("service_seconds", float("nan")),
         "finite and positive"),
        (lambda summary: summary.__setitem__("telemetry_complete", False),
         "telemetry_complete is not true"),
        (lambda summary: summary.__setitem__("overlap_valid", False),
         "overlap_valid is not true"),
    ],
)
def test_gpu_projection_fails_closed_on_unusable_cell_work(
    settings, monkeypatch, proved_repo, mutation, detail
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    mutation(next(iter(outputs.values()))["work_summary"])
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert detail in projection["detail"]
    assert projection["projected_total_gpu_seconds"] is None


def test_gpu_projection_blocks_a_full_screen_that_exceeds_the_hard_cap(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    for record in outputs.values():
        record["work_summary"]["service_seconds"] = 6000.0
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert projection["projected_total_gpu_seconds"] == 600_000.0
    assert projection["hard_cap_gpu_seconds"] == 540_000.0
    assert projection["projected_margin_gpu_seconds"] == -60_000.0
    assert "Cost-only forecast" in projection["detail"]
    assert "provider ledger remains authoritative" in projection["detail"]


def test_gpu_projection_preserves_the_screen_arm_mix_instead_of_using_a_naive_mean(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    plan = manifest.notes["gpu_budget_projection_plan"]
    plan["screen_task_count"] = 1
    plan["screen_cells_by_arm"] = {
        "P0": 2,
        "H_ID": 90,
        "CPU_LEXICAL": 2,
        "SHORT_PROSE": 2,
    }
    outputs = _outputs(manifest)
    for record in outputs.values():
        record["work_summary"]["service_seconds"] = (
            100.0 if record["cell"]["arm"]["arm_id"] == "H_ID" else 1.0
        )
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    # Actual canary = 103.  The arm-stratified screen forecast is 90*100 + 6*1,
    # not overall-canary-mean * 96, which would badly underweight the expensive arm.
    assert projection["projection_method"] == "per_arm_observed_max"
    assert projection["projected_screen_gpu_seconds"] == 9006.0
    assert projection["projected_total_gpu_seconds"] == 9109.0


def test_live_gpu_projection_uses_remaining_headroom_after_historical_spend(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    # The canary itself used 40 seconds, but historical work has already consumed almost all
    # of the same provider ledger account. A nominal-cap projection would incorrectly pass.
    audit = _provider_audit(outputs, reserved=100.0, settled=539_400.0)
    checks = canary._verify(
        settings,
        manifest,
        _states(manifest),
        outputs,
        repo=proved_repo,
        provider_audit=audit,
        require_provider_audit=True,
    )
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert projection["projected_screen_gpu_seconds"] == 960.0
    assert projection["ledger_remaining_gpu_seconds"] == 500.0
    assert projection["ledger_reserved_gpu_seconds"] == 100.0
    assert projection["ledger_settled_gpu_seconds"] == 539_400.0
    assert projection["projected_total_gpu_seconds"] == 540_460.0
    assert projection["projected_margin_gpu_seconds"] == -460.0


def test_historical_global_settlement_cannot_mask_unsettled_current_work(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    audit = _provider_audit(outputs, settled=539_400.0)
    # The global account looks amply settled because it includes old campaigns. One exact
    # current work item is nevertheless missing its authoritative GPU settlement.
    row = audit["work"][0]
    row["settled_gpu_seconds"] = 0.0
    for op in row["ops"]:
        op["settled_gpu_seconds"] = 0.0
    checks = canary._verify(
        settings,
        manifest,
        _states(manifest),
        outputs,
        repo=proved_repo,
        provider_audit=audit,
        require_provider_audit=True,
    )
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert "does not match output service_seconds" in projection["detail"]


def test_provider_work_and_op_settlement_must_reconcile(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    audit = _provider_audit(outputs)
    audit["work"][0]["ops"][0]["settled_gpu_seconds"] += 1.0
    checks = canary._verify(
        settings,
        manifest,
        _states(manifest),
        outputs,
        repo=proved_repo,
        provider_audit=audit,
        require_provider_audit=True,
    )
    assert _status(checks, "provider_canary_audit") == "FAIL"
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert "work/op settlement mismatch" in projection["detail"]


def test_live_gpu_projection_rejects_budget_cap_drift(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    audit = _provider_audit(outputs, cap=500_000.0, settled=40.0)
    checks = canary._verify(
        settings,
        manifest,
        _states(manifest),
        outputs,
        repo=proved_repo,
        provider_audit=audit,
        require_provider_audit=True,
    )
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert "cap drift" in projection["detail"]


def test_live_gpu_projection_still_fails_closed_on_missing_cell_telemetry(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    next(iter(outputs.values()))["work_summary"]["telemetry_complete"] = False
    checks = canary._verify(
        settings,
        manifest,
        _states(manifest),
        outputs,
        repo=proved_repo,
        provider_audit=_provider_audit(outputs),
        require_provider_audit=True,
    )
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert projection["projected_total_gpu_seconds"] is None
    assert "telemetry_complete is not true" in projection["detail"]


def test_live_canary_never_falls_back_to_opening_provider_files(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    monkeypatch.setattr(
        canary,
        "_inference_events",
        lambda *_args, **_kwargs: pytest.fail("live path opened provider ledger"),
    )
    monkeypatch.setattr(
        canary,
        "_open_external_calls",
        lambda *_args, **_kwargs: pytest.fail("live path opened provider ledger"),
    )
    monkeypatch.setattr(
        canary,
        "_requests_carry_a_completion_cap",
        lambda *_args, **_kwargs: pytest.fail("live path opened provider object store"),
    )
    checks = canary._verify(
        settings,
        manifest,
        _states(manifest),
        _outputs(manifest),
        repo=proved_repo,
        provider_audit=None,
        provider_audit_error="provider unavailable",
        require_provider_audit=True,
    )
    assert _status(checks, "provider_canary_audit") == "FAIL"
    assert _status(checks, "ledger_readable") == "FAIL"
    assert _status(checks, "reservations_closed") == "FAIL"
    assert _status(checks, "generation_cap_is_a_completion_limit") == "FAIL"
    assert _status(checks, "projected_gpu_budget_feasible") == "FAIL"


def test_gpu_projection_requires_a_pre_run_frozen_denominator(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    manifest.notes.pop("gpu_budget_projection_plan")
    checks = canary._verify(
        settings, manifest, _states(manifest), _outputs(manifest), repo=proved_repo)
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert "frozen manifest has no gpu_budget_projection_plan" in projection["detail"]


def test_gpu_projection_rejects_a_denominator_that_does_not_reconcile(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    manifest.notes["gpu_budget_projection_plan"]["planned_total_cells"] += 1
    checks = canary._verify(
        settings, manifest, _states(manifest), _outputs(manifest), repo=proved_repo)
    projection = next(
        check for check in checks
        if check["name"] == "projected_gpu_budget_feasible"
    )
    assert projection["status"] == "FAIL"
    assert "does not equal frozen canary_cells + screen_cells" in projection["detail"]


def test_canary_rejects_h_trace_without_raw_occurrence_denominator(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    h_record = next(
        record for record in outputs.values()
        if record["cell"]["arm"]["arm_id"] == "H_ID")
    h_record["direct_node_records"][0].pop("offered_source_occurrence_ids")
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "direct_denominator_provenance") == "FAIL"


@pytest.mark.asyncio
async def test_run_canary_wires_seed_layer_and_provider_work_summary(
    settings, monkeypatch
):
    import shapeflow_p1.campaign.canary as canary

    captured = {}

    class DummyLedger:
        def create_run(self, *_args, **_kwargs):
            captured["run_created"] = True

        def close(self):
            captured["closed"] = True

        def committed_ref(self, _key):
            return None

    class DummyStore:
        pass

    class DummyClient:
        base_url = "http://provider"
        token = "runner-token"  # noqa: S105 - inert unit-test capability

        async def register_cell(self, **kwargs):
            captured["registration"] = kwargs

        async def work_summary(self, **kwargs):
            captured["work_request"] = kwargs
            return {"telemetry_complete": True, "overlap_valid": True}

        async def canary_audit(self, **kwargs):
            captured["audit_request"] = kwargs
            return {
                "version": "canary_audit_v1",
                "work_keys": [],
                "work": [],
                "open_attempts": 0,
                "gpu_budget": {
                    "resource": "gpu_seconds",
                    "cap": 540_000.0,
                    "reserved": 0.0,
                    "settled": 0.0,
                    "remaining": 540_000.0,
                },
            }

    class DummyRunner:
        def __init__(self, _settings, **kwargs):
            captured.update(kwargs)
            self.execution_binding_sha256 = kwargs["execution_binding_sha256"]
            self.protocol_document_sha256 = kwargs["protocol_document_sha256"]

        def arms_from_config(self, _name):
            return [
                SimpleNamespace(
                    arm_id="P0", page_variant="P0", close_variant="P0")
            ]

        def build_schedule(self, **_kwargs):
            return SimpleNamespace(cells=[], notes={}, digest="e" * 64)

        def freeze_schedule(self, *_args, **_kwargs):
            return "schedule"

        async def run_cells(self, *_args, **_kwargs):
            return []

        def cell_states(self, *_args, **_kwargs):
            return {}

    client = DummyClient()
    monkeypatch.setattr(canary, "open_run_ledger",
                        lambda _settings: (DummyLedger(), DummyStore()))
    monkeypatch.setattr(canary, "provider_client_for",
                        lambda _settings, _role: client)
    monkeypatch.setattr(canary, "available_tasks", lambda _settings, _split: ["T1"])
    monkeypatch.setattr(canary, "questions_for",
                        lambda _settings, _tasks: {"T1": "question"})
    monkeypatch.setattr(canary, "CampaignRunner", DummyRunner)
    monkeypatch.setattr(
        canary,
        "verified_execution_binding",
        lambda _repo, expected_digest=None: SimpleNamespace(
            digest="e" * 64, protocol_sha="d" * 64),
    )
    monkeypatch.setattr(
        canary, "_verify",
        lambda *_args, **_kwargs: [
            {"name": "intentional_stop", "status": "FAIL", "detail": "unit fixture"}
        ],
    )

    await canary.run_canary(settings, repo=REPO, task_limit=1)

    selector = captured["model_call_factory"]("cell-token", seed=47)
    assert selector._seed == 47
    spec = SimpleNamespace(
        cell_token="cell-token",  # noqa: S106 - inert unit-test capability
        run_id="run", task_id="T1", arm_id="H_ID",
        variant_id="H02+P0", replicate_id="r0", work_key="WK",
    )
    await captured["register_cell"](spec)
    expected_layer = str(settings.get("week1", "measurement", "layer"))
    assert captured["registration"]["layer"] == expected_layer
    summary = await captured["fetch_work_summary"]("WK")
    assert summary["telemetry_complete"] is True
    assert captured["work_request"] == {
        "work_key": "WK", "require_isolated": expected_layer == "causal"
    }
    assert "audit_request" not in captured, (
        "an empty manifest must fail locally, not send a minItems-violating audit request"
    )


def test_a_selector_that_never_decoded_fails(settings, monkeypatch, proved_repo):
    """A P1 arm with zero selector decode did not select anything, however green it looks."""
    inference = [e for e in _good_inference() if not e["op_class"].startswith("PAGE_P1")]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "selector_decode") == "FAIL"


def test_p1_bytes_identical_to_p0_fails(settings, monkeypatch, proved_repo):
    """The failure mode this exists for: copy P0's output and change the label."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest, p1_report="same", p0_report="same"),
                            repo=proved_repo)
    assert _status(checks, "p1_bytes_differ_from_p0") == "FAIL"


def test_a_cpu_control_that_called_the_model_fails(settings, monkeypatch, proved_repo):
    """If it decoded, it is not a CPU control, and the comparison it anchors is meaningless."""
    inference = _good_inference() + [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-CPU_LEXICAL",
         "completion_tokens": 64, "prompt_tokens": 500},
    ]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "cpu_control_zero_decode") == "FAIL"


def test_a_prose_control_over_its_cap_fails(settings, monkeypatch, proved_repo):
    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    inference = _good_inference() + [
        {"op_class": "PAGE_P1_SELECTOR_LOCAL", "work_key": "WK-SHORT_PROSE",
         "completion_tokens": cap + 1, "prompt_tokens": 500},
    ]
    manifest, canary = _run(settings, monkeypatch, inference)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "short_prose_same_budget") == "FAIL"


def test_a_prose_control_without_measured_rendered_bytes_fails(
    settings, monkeypatch, proved_repo
):
    """A missing output count used to become zero and pass the same-budget check."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    prose = next(
        value for value in outputs.values()
        if value["cell"]["arm"]["arm_id"] == "SHORT_PROSE"
    )
    prose["counts"].pop("rendered_output_count")
    prose["counts"].pop("max_rendered_tokens")
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo
    )
    assert _status(checks, "short_prose_same_budget") == "FAIL"


def test_a_reduced_batch_without_atomic_publication_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs.values()))
    first["events"] = [
        event for event in first["events"] if event["kind"] != "TOOL_BATCH_PUBLISHED"
    ]
    checks = canary._verify(settings, manifest, _states(manifest),
                            outputs, repo=proved_repo)
    assert _status(checks, "atomic_publication") == "FAIL"


def test_a_multi_sibling_turn_is_one_valid_atomic_publication(
    settings, monkeypatch, proved_repo
):
    """Two sibling tool calls are one reduced turn, not a deferred/reduced count mismatch."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs.values()))
    reduced = next(e for e in first["events"] if e["kind"] == "PAGE_BATCH_REDUCED")
    published = next(e for e in first["events"] if e["kind"] == "TOOL_BATCH_PUBLISHED")
    reduced["siblings"] = 2
    published["sibling_count"] = 2
    published["tool_call_ids"] = ["call-a", "call-b"]
    first["counts"]["page_batches_deferred"] = 2
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "atomic_publication") == "PASS"


def test_two_partial_publications_for_one_checkpoint_fail(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs.values()))
    publication = next(
        e for e in first["events"] if e["kind"] == "TOOL_BATCH_PUBLISHED")
    first["events"].append({**publication, "tool_call_ids": ["call-other"]})
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "atomic_publication") == "FAIL"


def test_an_incomplete_canary_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    states = _states(manifest)
    states[next(iter(states))] = "FAILED_FINAL"
    checks = canary._verify(settings, manifest, states, _outputs(manifest), repo=proved_repo)
    assert _status(checks, "cells_committed") == "FAIL"


def test_an_open_reservation_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    monkeypatch.setattr(canary, "_open_external_calls", lambda _s, **_kw: 3)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "reservations_closed") == "FAIL"


def test_a_replay_miss_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs))
    outputs[first]["events"].append({"kind": "PAGE_BATCH_REDUCED",
                                     "detail": "ReplayMiss: query not captured"})
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "no_live_search_miss") == "FAIL"


def test_a_readable_evaluator_tree_fails(settings, monkeypatch, proved_repo, tmp_path):
    """The in-process probe is evidence in one direction only, so it is applied in that
    direction: if the *runner* can list the tree, no proof outweighs it."""
    monkeypatch.setenv("USER", "runner")
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    (tmp_path / "evaluator").mkdir(parents=True, exist_ok=True)
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "truth_invisible_to_treatment") == "FAIL"


def test_isolation_is_not_proved_by_an_absent_tree(settings, monkeypatch, proved_repo):
    """It passed for exactly as long as there was nothing to protect."""
    import shutil

    manifest, canary = _run(settings, monkeypatch, _good_inference())
    shutil.rmtree(settings.path("evaluator_root"))
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "truth_invisible_to_treatment") == "FAIL"


def test_isolation_without_a_cross_uid_proof_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    (proved_repo / "reports" / "CREDENTIAL_ISOLATION.json").unlink()
    checks = canary._verify(settings, manifest, _states(manifest),
                            _outputs(manifest), repo=proved_repo)
    assert _status(checks, "truth_invisible_to_treatment") == "FAIL"


def test_the_p0_comparison_is_per_task():
    p0 = {"k": {"cell": {"task_id": "T1"}, "final_report": "A"}}
    p1 = {"j": {"cell": {"task_id": "T1"}, "final_report": "B"},
          "l": {"cell": {"task_id": "T2"}, "final_report": "C"}}
    differing, compared = _p1_differs_from_p0(p0, p1)
    assert (differing, compared) == (1, 1), "a P1 cell with no P0 on its task was compared"


def _status(checks: list[dict], name: str) -> str:
    for check in checks:
        if check["name"] == name:
            return check["status"]
    raise AssertionError(f"check {name!r} was not produced; produced {[c['name'] for c in checks]}")


def test_verify_is_the_public_entry_point():
    assert callable(_verify)


# --- the checks that used to pass no matter what ---------------------------------------------


def test_the_fallback_check_can_actually_fail(settings, monkeypatch, proved_repo):
    """It was `_check(..., True, ...)`: it formatted counts into a sentence and passed."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    first = next(iter(outputs.values()))
    first["counts"]["page_fallbacks"] = 3          # a fallback the ledger knows nothing about
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "fallback_and_failures_accounted") == "FAIL"


def test_the_generation_cap_check_reads_requests_not_config(settings, monkeypatch,
                                                            proved_repo):
    """It multiplied two config values into a constant True and never saw a request."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    # Re-write the recorded request without a cap.
    import json

    from shapeflow_p1.object_store import ObjectStore

    store = ObjectStore(settings.data_root
                        / str(settings.get("week1", "paths", "provider_root")) / "objects")
    ref = store.put_bytes(json.dumps({"model": "m"}).encode())
    from shapeflow_p1.experiment.ledger import Ledger

    ledger = Ledger(str(settings.data_root
                        / str(settings.get("week1", "paths", "provider_ledger"))))
    with ledger.transaction() as cur:
        cur.execute("UPDATE external_call_attempts SET request_object_ref=?", (ref.key,))
    ledger.close()

    checks = canary._verify(settings, manifest, _states(manifest), _outputs(manifest),
                            repo=proved_repo)
    assert _status(checks, "generation_cap_is_a_completion_limit") == "FAIL"


def test_a_large_research_request_does_not_violate_the_selector_cap(settings):
    """The selector cap is not the research/final-answer cap."""
    import shapeflow_p1.campaign.canary as canary

    cap = int(settings.get("week1", "measurement", "selector_max_completion_tokens"))
    _provider_ledger(
        settings, max_tokens=cap, work_key="WK-H_ID",
        op_class="PAGE_P1_SELECTOR_LOCAL", call_id="selector", attempt_id="selector-a")
    _provider_ledger(
        settings, max_tokens=8192, work_key="WK-H_ID",
        op_class="RESEARCHER_REACT", call_id="research", attempt_id="research-a")
    _provider_ledger(
        settings, max_tokens=8192, work_key="WK-OLD-RUN",
        op_class="PAGE_P1_SELECTOR_LOCAL", call_id="old-selector",
        attempt_id="old-selector-a")
    ok, detail = canary._requests_carry_a_completion_cap(
        settings, cap, work_keys={"WK-H_ID"})
    assert ok, detail
    assert "1 selector request" in detail


def test_historical_requests_cannot_supply_current_selector_decode(settings):
    import shapeflow_p1.campaign.canary as canary

    _provider_ledger(
        settings, work_key="WK-OLD-RUN", call_id="old", attempt_id="old-a")
    assert canary._inference_events(
        settings, work_keys={"WK-CURRENT-CANARY"}) == []


def test_historical_open_request_does_not_poison_current_canary(settings):
    import shapeflow_p1.campaign.canary as canary

    _provider_ledger(
        settings, work_key="WK-OLD-RUN", call_id="old-open", attempt_id="old-open-a",
        state="SENT")
    assert canary._open_external_calls(
        settings, work_keys={"WK-CURRENT-CANARY"}) == 0
    assert canary._open_external_calls(
        settings, work_keys={"WK-OLD-RUN"}) == 1


def _add_page_fallback(record: dict, *, with_direct_record: bool = True) -> None:
    reduced = next(e for e in record["events"] if e["kind"] == "PAGE_BATCH_REDUCED")
    reduced["fell_back"] = True
    reduced["failure"] = "INVALID_JSON"
    record["counts"]["page_fallbacks"] = 1
    if with_direct_record:
        record["direct_node_records"].append({
            "node": "H",
            "checkpoint_hash": reduced["checkpoint"],
            "offered_span_ids": ["E1"],
            "selected_span_ids": [],
            "published_span_ids": [],
            "fell_back": True,
            "failure": "INVALID_JSON",
        })


def test_strategy_fallback_is_reconciled_to_committed_selector_work(
    settings, monkeypatch, proved_repo
):
    """Invalid JSON is a strategy incident after a committed provider response."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    target = next(
        record for record in outputs.values()
        if record["cell"]["arm"]["arm_id"] == "H_ID")
    _add_page_fallback(target)
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "fallback_and_failures_accounted") == "PASS"


def test_fallback_without_direct_failure_provenance_fails(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    target = next(
        record for record in outputs.values()
        if record["cell"]["arm"]["arm_id"] == "H_ID")
    _add_page_fallback(target, with_direct_record=False)
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "fallback_and_failures_accounted") == "FAIL"


def test_fallback_without_a_reason_fails(settings, monkeypatch, proved_repo):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    target = next(
        record for record in outputs.values()
        if record["cell"]["arm"]["arm_id"] == "H_ID")
    _add_page_fallback(target)
    reduced = next(e for e in target["events"] if e["kind"] == "PAGE_BATCH_REDUCED")
    reduced["failure"] = None
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "fallback_and_failures_accounted") == "FAIL"


def test_fallback_without_complete_provider_work_fails(
    settings, monkeypatch, proved_repo
):
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    target = next(
        record for record in outputs.values()
        if record["cell"]["arm"]["arm_id"] == "H_ID")
    _add_page_fallback(target)
    target["work_summary"]["telemetry_complete"] = False
    checks = canary._verify(
        settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "fallback_and_failures_accounted") == "FAIL"


def test_one_checkpoint_kind_is_not_both(settings, monkeypatch, proved_repo):
    """It was a set intersection, so an H-only run passed a check named 'both boundaries'."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    for record in outputs.values():
        record["checkpoints"] = [c for c in record["checkpoints"]
                                 if c["kind"] != "CCheckpoint"]
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "checkpoints_present") == "FAIL"


def test_an_arm_that_is_byte_identical_to_p0_fails(settings, monkeypatch, proved_repo):
    """One differing cell out of six used to be enough for the whole check."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    p0 = next(v for v in outputs.values() if v["cell"]["arm"]["arm_id"] == "P0")
    for record in outputs.values():
        if record["cell"]["arm"]["arm_id"] != "P0":
            record["final_report"] = p0["final_report"]
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "p1_bytes_differ_from_p0") == "FAIL"


def test_an_arm_that_reduced_nothing_fails_even_beside_one_that_did(settings, monkeypatch,
                                                                    proved_repo):
    """reduce_published_batch fires for any bound bundle, so one working control satisfied
    the old total while a completely inert LLM arm passed behind it."""
    manifest, canary = _run(settings, monkeypatch, _good_inference())
    outputs = _outputs(manifest)
    for record in outputs.values():
        if record["cell"]["arm"]["arm_id"] not in ("P0", "CPU_LEXICAL"):
            record["counts"]["page_batches_reduced"] = 0
            record["counts"]["close_reduced"] = 0
            record["counts"]["page_batches_deferred"] = 0
    checks = canary._verify(settings, manifest, _states(manifest), outputs, repo=proved_repo)
    assert _status(checks, "p1_strategy_invocations") == "FAIL"


def test_an_unreadable_ledger_is_a_failure_not_a_zero(settings, monkeypatch, proved_repo):
    import shapeflow_p1.campaign.canary as canary_module

    manifest, canary = _run(settings, monkeypatch, _good_inference())
    monkeypatch.setattr(canary_module, "_inference_events",
                        lambda _s, **_kw: (_ for _ in ()).throw(
                            canary_module.LedgerUnreadable("gone")))
    checks = canary._verify(settings, manifest, _states(manifest), _outputs(manifest),
                            repo=proved_repo)
    assert _status(checks, "ledger_readable") == "FAIL"
