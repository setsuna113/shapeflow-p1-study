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

    # 8. No report claims something no artifact supports.
    gates.append(check_report_claims(repo))

    return {"ok": _ok(gates), "gates": [g.as_dict() for g in gates]}


#: Phrases a report may not contain, and what is wrong with each. These are not style
#: preferences: every one of them was written about this study while the artifact that
#: would justify it did not exist. A verdict that survives an adversarial reader cannot
#: contain a sentence whose evidence nobody can produce.
UNSUPPORTED_CLAIMS: dict[str, str] = {
    "Block C/D 完成": "no block is complete while the component screen re-runs the full graph",
    "Block C/D complete": "no block is complete while the component screen re-runs the full graph",
    "campaign runner 完成": "the campaign runner cannot start: PhaseStore rejects its first phase",
    "campaign runner complete": "the campaign runner cannot start: PhaseStore rejects its first phase",
    "leased GPU": "the GPU is borrowed on a shared host; nothing is leased",
    "662 tests all green": "cite the count the suite actually collects, from a run you did",
}


def check_report_claims(repo: Path) -> Gate:
    """Refuse to launch while a report asserts something this tree cannot back up."""
    reports = repo / "reports"
    if not reports.exists():
        return Gate("report_claims", PASS, "no reports yet")
    found: list[str] = []
    for path in sorted(reports.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in (".md", ".json", ".txt"):
            continue
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for phrase in UNSUPPORTED_CLAIMS:
            if phrase in text:
                found.append(f"{path.relative_to(repo)}: {phrase!r}")
    if found:
        return Gate("report_claims", FAIL, "; ".join(found[:6]))
    return Gate("report_claims", PASS, "no unsupported claims in reports/")


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

    # Neither the answer key nor the steward's audit graph may be reachable from here.
    ok, detail = tree_isolation(repo, settings, ("evaluator_root", "steward_root"))
    checks.append(Gate("tree_isolation", PASS if ok else FAIL, detail))

    return {"ok": _ok(checks), "checks": [c.as_dict() for c in checks],
            "protocol_sha": live_sha}


def tree_isolation(repo: Path, settings: Settings, names: tuple[str, ...]) -> tuple[bool, str]:
    """Whether the runner is shut out of the trees it must not read.

    The authority is the installer's cross-uid proof, because that is the only thing that
    can answer the question. A same-process ``os.listdir`` says what *this* identity can
    reach, which for the steward reading its own tree is "everything" and for a host where
    the directory does not exist yet is "nothing" -- and the old check read that second
    answer as success, so it passed for exactly as long as there was nothing to protect.

    The in-process probe is still used, but only in the direction where it is evidence: if
    this identity *can* read one of these trees, that is a failure regardless of what any
    proof says.
    """
    proof_path = repo / "reports" / "CREDENTIAL_ISOLATION.json"
    problems: list[str] = []
    for name in names:
        path = settings.path(name)
        if not path.exists():
            problems.append(f"{path} does not exist, so nothing has been shown unreachable")
        elif _readable(path) and _role() == "runner":
            problems.append(f"{path} is readable by the runner identity")
    if not proof_path.exists():
        problems.append(
            f"{proof_path.name} is absent: cross-uid isolation cannot be shown from inside "
            "one process, and this one was never proved")
    else:
        try:
            proved = set(json.loads(proof_path.read_text(encoding="utf-8"))
                         .get("runner_cannot_list") or [])
        except (OSError, json.JSONDecodeError) as e:
            problems.append(f"the isolation proof is unreadable: {e}")
            proved = set()
        for name in names:
            tree = name.replace("_root", "")
            if tree not in proved:
                problems.append(f"the proof does not cover the {tree} tree")
    if problems:
        return False, "; ".join(problems)
    return True, f"runner shut out of {', '.join(n.replace('_root', '') for n in names)}"


def _role() -> str:
    import os

    return os.environ.get("USER") or os.environ.get("USERNAME") or ""


def _readable(path: Path) -> bool:
    import os

    try:
        os.listdir(path)
        return True
    except OSError:
        return False
