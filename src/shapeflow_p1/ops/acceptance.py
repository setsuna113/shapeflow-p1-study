"""The acceptance matrix and the campaign preflight: machine gates, not completion claims.

Both answer the same question in different places: is the thing that is about to run actually
the thing that was approved? Neither trusts a statement that work was done -- each re-derives
the fact from an artifact. A gate that could be satisfied by asserting it had been satisfied is
not a gate.

Preflight is the last check before treatment. It runs after acquisition and before any GPU hour,
so a sealed corpus that does not match the approval, or a frozen world whose manifest has been
edited, stops the campaign while nothing is yet attributable to it.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from ..campaign.settings import Settings

__all__ = ["Gate", "run_acceptance", "run_preflight"]

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Gate:
    name: str
    status: str
    detail: str = ""

    def as_dict(self) -> dict:
        return {"name": self.name, "status": self.status, "detail": self.detail}


def _ok(gates: list[Gate]) -> bool:
    """Green means every gate PASSed. A SKIP is not a pass: a check that could not run has not
    been satisfied, and counting it as success is how a gate goes green on a machine that cannot
    perform it."""
    return bool(gates) and all(g.status == PASS for g in gates)


def run_acceptance(settings: Settings, *, repo: Path) -> dict:
    """Everything that must hold before a paid step, derived from artifacts."""
    gates: list[Gate] = []

    # 1. The approval binds the live configuration.
    from ..protocol import ApprovalError, verify_approval_file

    try:
        binding = verify_approval_file(repo)
        gates.append(Gate("approval", PASS, f"binding {binding.digest[:12]}"))
    except ApprovalError as e:
        gates.append(Gate("approval", FAIL, str(e)[:300]))

    # 2. Every registered arm can be built. A campaign that discovered a broken arm mid-run
    #    would already have spent budget on the ones before it.
    try:
        from ..strategies.factory import StrategyFactory, load_registry

        registry = load_registry(repo / "configs")
        factory = StrategyFactory(registry=registry, model_call=_unusable_model_call)
        bundles = factory.build_all()
        gates.append(Gate("variant_registry", PASS, f"{len(bundles)} arms instantiate"))
    except Exception as e:  # noqa: BLE001
        gates.append(Gate("variant_registry", FAIL, f"{type(e).__name__}: {e}"))

    # 3. The canary's arms are all registered.
    try:
        arms = settings.get("week1", "canary", "arms")
        unknown = sorted(
            v for a in arms for v in (a["page_variant"], a["close_variant"])
            if v != "P0" and v not in registry
        )
        gates.append(Gate("canary_arms", PASS if not unknown else FAIL,
                          f"{len(arms)} arms" if not unknown else f"unregistered: {unknown}"))
    except Exception as e:  # noqa: BLE001
        gates.append(Gate("canary_arms", FAIL, f"{type(e).__name__}: {e}"))

    # 4. Schemas are closed, and the treatment path cannot import the evaluator.
    from ..doctor import check_schemas_closed

    schema_check = check_schemas_closed(repo / "schemas")
    gates.append(Gate("schemas", schema_check.status, schema_check.detail))

    # 5. The secret scanner exists. Asserting the scan passed without a scanner is not a scan.
    gates.append(
        Gate("secret_scanner", PASS if shutil.which("gitleaks") else FAIL,
             "gitleaks present" if shutil.which("gitleaks")
             else "gitleaks not installed; the secret gate cannot be satisfied by assertion")
    )

    # 6. The two measurement layers are declared separately, and the causal one is causal.
    causal = settings.get("stack", "isolation", "causal")
    causal_ok = (causal["max_num_seqs"] == 1 and causal["enable_prefix_caching"] is False
                 and causal["gateway_max_upstream_inflight"] == 1)
    gates.append(Gate("causal_layer", PASS if causal_ok else FAIL,
                      "max_num_seqs=1, APC off, single upstream in flight" if causal_ok
                      else f"declared {causal}"))

    # 7. The claim scope travels with the corpus.
    scope_ok = (settings.claim_scope == "FORMATIVE_ONLY"
                and settings.corpus_tier == "FORMATIVE_MACHINE_AUTHORED")
    gates.append(Gate("claim_scope", PASS if scope_ok else FAIL,
                      f"{settings.corpus_tier} / {settings.claim_scope}"))

    return {"ok": _ok(gates), "gates": [g.as_dict() for g in gates]}


async def _unusable_model_call(**_kw):  # pragma: no cover - never invoked
    raise RuntimeError("acceptance builds arms; it does not run them")


def run_preflight(settings: Settings, *, repo: Path,
                  approved_protocol_sha: Optional[str] = None) -> dict:
    """The last check before treatment: approval, corpus, world, and the runner's own view."""
    from ..campaign.acquire import acquired_task_ids
    from ..campaign.prepare import load_sealed_registry
    from ..protocol import ApprovalError, protocol_sha, verify_approval_file

    checks: list[Gate] = []

    live_sha = protocol_sha(repo)
    if approved_protocol_sha and approved_protocol_sha != live_sha:
        checks.append(Gate("protocol_sha", FAIL,
                           f"approved {approved_protocol_sha[:12]} != live {live_sha[:12]}"))
    else:
        checks.append(Gate("protocol_sha", PASS, live_sha[:12]))

    try:
        verify_approval_file(repo)
        checks.append(Gate("approval", PASS, "binds the live configuration"))
    except ApprovalError as e:
        checks.append(Gate("approval", FAIL, str(e)[:300]))

    try:
        registry, digest = load_sealed_registry(settings)
        checks.append(Gate("sealed_registry", PASS,
                           f"{len(registry['tasks'])} tasks, {digest[:12]}"))
    except Exception as e:  # noqa: BLE001
        checks.append(Gate("sealed_registry", FAIL, f"{type(e).__name__}: {e}"))
        registry = {"tasks": []}

    frozen = acquired_task_ids(settings)
    split = str(settings.get("week1", "screen", "split"))
    wanted = [t["task_id"] for t in registry.get("tasks", []) if t.get("split") == split]
    missing = sorted(set(wanted) - set(frozen))
    checks.append(Gate("frozen_world", PASS if wanted and not missing else FAIL,
                       f"{len(frozen)} worlds frozen" if not missing
                       else f"{len(missing)} {split} task(s) unacquired"))

    # The runner's own view must carry the question and nothing else.
    runner_tasks = settings.path("frozen_corpus_for_runner") / "tasks"
    leaked = []
    for path in sorted(runner_tasks.glob("*.json"))[:200]:
        body = json.loads(path.read_text(encoding="utf-8"))
        if set(body) - {"task_id", "split", "original_question", "corpus_tier", "claim_scope",
                        "deployment_pins"}:
            leaked.append(path.name)
    checks.append(Gate("runner_view", PASS if not leaked else FAIL,
                       "question only" if not leaked
                       else f"{len(leaked)} task view(s) carry more than the question"))

    # The evaluator's tree must not be readable from here.
    evaluator = settings.path("evaluator_root")
    readable = evaluator.exists() and _readable(evaluator)
    checks.append(Gate("evaluator_isolation",
                       PASS if not readable else FAIL,
                       "unreadable from this identity" if not readable
                       else f"{evaluator} is readable by the process running preflight"))

    return {"ok": _ok(checks), "checks": [c.as_dict() for c in checks],
            "protocol_sha": live_sha}


def _readable(path: Path) -> bool:
    import os

    try:
        os.listdir(path)
        return True
    except OSError:
        return False
