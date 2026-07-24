"""A GPU engineering diagnostic: prove the P1 path is real on the actual stack.

This is NOT the formative screen and never becomes one. Its corpus is a small hand-built
fixture, because the Tavily credential is rejected and no real world could be frozen (see
reports/BLOCKED_TAVILY_UNAUTHORIZED.md). Every artifact it writes is labelled
DIAGNOSTIC_NON_PROTOCOL. What it establishes is exactly the thing the data blocker does not
touch: that the real patched Open Deep Research graph runs on the leased GPU against the real
Qwen model, that P1 selectors actually decode there, and that a P1 arm's output is not P0's
bytes with a different label.

It reuses the real graph driver, the real provider, the real vLLM engine and the real
strategy factory. The only fabricated input is the page content, and that is stated plainly.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from pathlib import Path

REPO = Path(os.environ.get("SHAPEFLOW_REPO", Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(REPO / "src"))

from shapeflow_p1.acquire.snapshot_store import SnapshotStore  # noqa: E402
from shapeflow_p1.acquire.source_pool import build_source_pool, QueryResponse, RawResult  # noqa: E402
from shapeflow_p1.campaign.canary import _verify  # noqa: E402
from shapeflow_p1.campaign.runner import CampaignRunner, RunnerConfig  # noqa: E402
from shapeflow_p1.campaign.schedule import ArmSpec, cell_key  # noqa: E402
from shapeflow_p1.campaign.selector_client import SelectorModelCall  # noqa: E402
from shapeflow_p1.campaign.settings import Settings  # noqa: E402
from shapeflow_p1.experiment.ledger import Ledger  # noqa: E402
from shapeflow_p1.object_store import ObjectStore  # noqa: E402
from shapeflow_p1.providers.provider_client import ProviderClient, load_role_token  # noqa: E402

DIAG_ROOT = Path(os.environ.get("SHAPEFLOW_DIAG_ROOT", "/storage/nvme/shapeflow-diag"))

# A fixture world: three sources per task, each with substantive markdown so the P1 selectors
# have real spans to choose among. This is the ONLY fabricated input.
_PAGES = {
    "q-alpha": [
        ("https://diag.invalid/harbour",
         "# Harbour Authority Report 2025\n\n"
         "The Harbour Authority recorded 4,821 container movements in Q1 2025, up from 4,102 in "
         "Q1 2024. The figure was independently confirmed by the Regional Freight Board.\n\n"
         "## Methodology\n\nMovements are counted at the seaward gate. Empty repositioning is "
         "excluded, which the 2023 revision changed and which accounts for part of the "
         "year-on-year rise.\n\n## Disputed totals\n\nThe Coastal Union reported 4,410 for the "
         "same quarter, a discrepancy the Authority attributes to differing empty-container "
         "rules."),
        ("https://diag.invalid/freightboard",
         "# Regional Freight Board bulletin\n\n"
         "Board figures put Q1 2025 harbour throughput at 4,821 units, matching the Harbour "
         "Authority. The Board notes that the Coastal Union's lower 4,410 figure omits "
         "transhipment cargo entirely.\n\n## Historical series\n\n2022: 3,980. 2023: 4,050. "
         "2024: 4,102. 2025: 4,821."),
        ("https://diag.invalid/coastalunion",
         "# Coastal Union quarterly note\n\n"
         "The Union's tally for Q1 2025 is 4,410 harbour movements. The Union counts only laden "
         "boxes and excludes transhipment, which it argues gives a truer picture of trade."),
    ],
    "q-beta": [
        ("https://diag.invalid/registry",
         "# National Filings Registry 2024\n\n"
         "The Registry accepted 7,788 enforcement filings in 2024. Of these, 312 were withdrawn "
         "before adjudication. The remaining 7,476 proceeded to a hearing.\n\n## By category\n\n"
         "Environmental: 2,201. Financial: 3,090. Safety: 2,185. The financial category grew "
         "fastest, from 2,640 in 2023."),
        ("https://diag.invalid/oversight",
         "# Oversight Office annual review\n\n"
         "The Office confirms 7,788 filings for 2024 and notes the 312 withdrawals. It flags a "
         "data-quality caveat: 44 filings were double-counted in the Registry's first release "
         "and later removed, so the audited total is 7,744."),
        ("https://diag.invalid/tribunal",
         "# Tribunal caseload statement\n\n"
         "The Tribunal heard 7,476 enforcement matters arising from 2024 filings. It does not "
         "publish a filings total of its own and defers to the Registry."),
    ],
}


def _build_fixture(settings: Settings) -> list[str]:
    """Write question-only task views and non-empty frozen pools into the diagnostic root."""
    store = SnapshotStore(ObjectStore(settings.path("frozen_corpus_for_runner") / "objects"))
    tasks_dir = settings.path("frozen_corpus_for_runner") / "tasks"
    pools_dir = settings.path("frozen_corpus_for_runner") / "pools"
    tasks_dir.mkdir(parents=True, exist_ok=True)
    pools_dir.mkdir(parents=True, exist_ok=True)

    from shapeflow_p1.campaign.acquire import _write_runner_pool

    questions = {
        "DIAG-ALPHA": "Which bodies reported Q1 2025 harbour movement totals, and where do their "
                      "published figures disagree?",
        "DIAG-BETA": "How many enforcement filings did the National Filings Registry accept in "
                     "2024, and how do oversight bodies reconcile the audited total?",
    }
    query_key = {"DIAG-ALPHA": "q-alpha", "DIAG-BETA": "q-beta"}
    task_ids = []
    for task_id, question in questions.items():
        (tasks_dir / f"{task_id}.json").write_text(json.dumps({
            "task_id": task_id, "split": "FORMATIVE_SCREEN", "original_question": question,
            "corpus_tier": "FORMATIVE_MACHINE_AUTHORED", "claim_scope": "FORMATIVE_ONLY",
        }, indent=2, sort_keys=True) + "\n")
        results = tuple(
            RawResult(url=url, title=url.rsplit("/", 1)[-1].title(), rank=i + 1,
                      snippet=text[:120], raw_content=text, score=0.9 - 0.1 * i)
            for i, (url, text) in enumerate(_PAGES[query_key[task_id]])
        )
        pool = build_source_pool(
            task_id, [QueryResponse(query_snapshot_id=f"qs-{task_id}",
                                    query_text=question, results=results)],
            store, fetched_at_utc="2026-07-24T00:00:00Z")
        _write_runner_pool(settings, task_id, pool, pool.snapshots)
        task_ids.append(task_id)
    return task_ids


async def main() -> int:
    settings = Settings.load(REPO, data_root=DIAG_ROOT)
    # A lean ODR config: one react call, one researcher iteration. The formative screen uses the
    # protocol's fuller config; the diagnostic only needs one search and one close to exercise
    # the whole P1 path (defer -> transform -> publish -> close -> select -> report), and a real
    # 14B model left at the protocol's limits runs the loop to its maximum, which is minutes of
    # GPU per cell for no additional engineering signal. This override is legitimate precisely
    # because the run is DIAGNOSTIC_NON_PROTOCOL.
    lean = dict(settings.configs["week1"])
    # Enough react turns that the real model reliably issues a search (one turn is not
    # enough -- it often answers or completes directly), but a single researcher
    # iteration so a cell stays a few minutes rather than fifteen.
    lean["odr"] = {**lean["odr"], "max_react_tool_calls": 3, "max_researcher_iterations": 1,
                   "max_concurrent_research_units": 1}
    settings.configs["week1"] = lean
    settings.ensure_paths("runner_root", "runs", "object_store")
    task_ids = _build_fixture(settings)[:1]     # one task is enough to prove the path

    host = settings.get("week1", "provider", "bind_host")
    port = settings.get("week1", "provider", "bind_port")
    token_dir = str(settings.get("week1", "provider", "token_dir"))
    client = ProviderClient(base_url=f"http://{host}:{port}",
                            token=load_role_token(token_dir, "runner"))

    ledger = Ledger(str(settings.path("runs") / "diag.sqlite"))
    store = ObjectStore(settings.path("object_store"))

    async def register(spec):
        await client.register_cell(
            cell_token=spec.cell_token, run_id=spec.run_id, task_id=spec.task_id,
            arm_id=spec.arm_id, variant_id=spec.variant_id, replicate_id=spec.replicate_id,
            work_key=spec.work_key)

    def model_call_factory(cell_token: str) -> SelectorModelCall:
        return SelectorModelCall(
            client, cell_token=cell_token, repo=REPO,
            temperature=float(settings.get("stack", "sampling", "temperature")),
            top_p=float(settings.get("stack", "sampling", "top_p")),
            max_completion_tokens=int(settings.get("week1", "measurement",
                                                   "selector_max_completion_tokens")),
            guided_decoding=bool(settings.get("week1", "measurement", "guided_decoding")))

    runner = CampaignRunner(
        settings, ledger=ledger, store=store,
        config=RunnerConfig(run_id="gpu-diagnostic", provider_base_url=client.base_url,
                            runner_token=client.token,
                            lease_seconds=float(settings.get("week1", "runtime", "lease_seconds"))),
        model_call_factory=model_call_factory, register_cell=register)

    # A targeted set: P0, one page-P1 arm, one close-P1 arm. Enough to prove the P1 path fires
    # on the real GPU and that the real model produces valid selector output; the full six-arm
    # canary is what the formative screen runs once a real world exists.
    arms = [ArmSpec("P0", "P0", "P0"), ArmSpec("H_ID", "H02", "P0"),
            ArmSpec("C_VISIBLE", "P0", "C01")]
    from shapeflow_p1.campaign.runner import questions_for

    manifest = runner.build_schedule(task_ids=task_ids, arms=arms, split="FORMATIVE_SCREEN")
    runner.freeze_schedule(manifest, settings.path("runs") / "diag_schedule.json")
    await runner.run_cells(manifest, phase_id="diagnostic", split="FORMATIVE_SCREEN",
                           questions=questions_for(settings, task_ids))

    states = runner.cell_states(manifest, phase_id="diagnostic", split="FORMATIVE_SCREEN")
    outputs = {}
    for cell in manifest.cells:
        ref = ledger.committed_ref(
            runner.work_key_for(cell, phase_id="diagnostic", split="FORMATIVE_SCREEN"))
        if ref:
            outputs[cell_key(cell)] = json.loads(store.get_bytes(ref).decode("utf-8"))

    checks = _verify(settings, manifest, states, outputs, repo=REPO)
    ok = all(c["status"] == "PASS" for c in checks)
    report = {
        "label": "DIAGNOSTIC_NON_PROTOCOL",
        "purpose": "prove the P1 path is non-inert on the real GPU; NOT the formative screen",
        "corpus": "hand-built fixture (Tavily is blocked; see BLOCKED_TAVILY_UNAUTHORIZED.md)",
        "ok": ok,
        "checks": checks,
        "cell_states": {k: v for k, v in states.items()},
    }
    out = REPO / "reports" / "GPU_DIAGNOSTIC.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    ledger.close()
    for c in checks:
        print(f"  {c['status']:4}  {c['name']}: {c['detail'][:100]}")
    print(f"\nDIAGNOSTIC_NON_PROTOCOL: {'PASS' if ok else 'FAIL'}  ->  {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
