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

from ..campaign.settings import Settings
from ..world.pools import acquired_task_ids

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

    # 6. The measurement layers are declared separately, and the active one is causal without
    #    being serialized. Prefix caching off is what makes a layer causal -- one arm's prefill
    #    must not subsidise another's. Admitting one request at a time never was: that was a
    #    requirement of the summed-service metric, and imposing it on a graph that issues
    #    concurrent page summaries produced 212 vendor timeouts on the largest pages, which is
    #    the stratum the study exists to measure. The gate now checks what makes the layer
    #    causal, and separately that the active layer does not serialize the system under test.
    active_name = str(settings.get("week1", "measurement", "layer"))
    active = settings.get("stack", "isolation", active_name)
    active_ok = (
        active["enable_prefix_caching"] is False
        and active["gateway_max_upstream_inflight"] == 0
        and int(active["max_num_seqs"]) > 1
    )
    gates.append(Gate(
        "causal_layer", PASS if active_ok else FAIL,
        f"{active_name}: APC off, max_num_seqs={active.get('max_num_seqs')}, "
        "graph concurrency preserved" if active_ok else f"declared {active}"))

    # 6b. The serialized mechanism layer is still declared, and still serialized. It is a
    #     control, not the product layer, so it must keep the property that makes summed
    #     service seconds arithmetic mean something.
    mechanism_name = str(settings.get("week1", "measurement", "mechanism_layer"))
    mechanism = settings.get("stack", "isolation", mechanism_name)
    mechanism_ok = (
        mechanism["max_num_seqs"] == 1
        and mechanism["enable_prefix_caching"] is False
        and mechanism["gateway_max_upstream_inflight"] == 1
        and mechanism_name != active_name
    )
    gates.append(Gate(
        "mechanism_layer", PASS if mechanism_ok else FAIL,
        f"{mechanism_name}: max_num_seqs=1, APC off, single upstream in flight"
        if mechanism_ok else f"declared {mechanism}"))

    # 7. The claim scope travels with the corpus.
    scope_ok = (settings.claim_scope == "FORMATIVE_ONLY"
                and settings.corpus_tier == "FORMATIVE_MACHINE_AUTHORED")
    gates.append(Gate("claim_scope", PASS if scope_ok else FAIL,
                      f"{settings.corpus_tier} / {settings.claim_scope}"))

    # 8. No report claims something no artifact supports.
    gates.append(check_report_claims(repo))

    return {"ok": _ok(gates), "gates": [g.as_dict() for g in gates]}


#: Phrases a report may not contain, keyed by a short rule id, with what is wrong with each.
#: These are not style preferences: every one of them was written about this study while the
#: artifact that would justify it did not exist. A verdict that survives an adversarial reader
#: cannot contain a sentence whose evidence nobody can produce.
#:
#: Keyed by id, because the failure detail names the *rule* and never quotes the phrase. A
#: scanner that writes the string it searches for into its own report under reports/ poisons
#: itself: the next run finds its own output, and the gate can never pass again no matter what
#: is fixed. That is exactly what happened here.
UNSUPPORTED_CLAIMS: dict[str, tuple[tuple[str, ...], str]] = {
    "block_cd_complete": (
        ("Block C/D 完成", "Block C/D complete"),
        "no block is complete while the component screen re-runs the full graph",
    ),
    "runner_complete": (
        ("campaign runner 完成", "campaign runner complete"),
        "the campaign runner cannot start: PhaseStore rejects its first phase",
    ),
    "gpu_exclusivity": (
        ("leased GPU", "leased gpu"),
        "the GPU is borrowed on a shared host: the flock excludes our own second worker, "
        "not a foreign process, so nothing is leased in the sense a reader would assume",
    ),
    "stale_test_count": (
        ("662 tests all green",),
        "cite the count the suite actually collects, from a run you did",
    ),
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
        for rule_id, (phrases, _why) in UNSUPPORTED_CLAIMS.items():
            if any(phrase in text for phrase in phrases):
                # The rule id, never the phrase -- see the note on UNSUPPORTED_CLAIMS.
                found.append(f"{path.relative_to(repo)}: {rule_id}")
    if found:
        return Gate("report_claims", FAIL, "; ".join(sorted(found)[:6]))
    return Gate("report_claims", PASS, "no unsupported claims in reports/")


async def _unusable_model_call(**_kw):  # pragma: no cover - never invoked
    raise RuntimeError("acceptance builds arms; it does not run them")


def run_preflight(settings: Settings, *, repo: Path,
                  approved_protocol_sha: str | None = None) -> dict:
    """The last check before treatment: approval, corpus, world, and the runner's own view."""
    from ..protocol import ApprovalError, protocol_sha, verify_approval_file

    checks: list[Gate] = []

    live_sha = protocol_sha(repo)
    if not approved_protocol_sha:
        # Absent used to pass. The one caller that omits the flag is a caller that never
        # said which protocol it believes it is running, which is the disagreement this
        # gate exists to catch.
        checks.append(Gate("protocol_sha", FAIL,
                           "no --approved-protocol-sha was named; the caller has not said "
                           "which protocol it believes it is running"))
    elif approved_protocol_sha != live_sha:
        checks.append(Gate("protocol_sha", FAIL,
                           f"approved {approved_protocol_sha[:12]} != live {live_sha[:12]}"))
    else:
        checks.append(Gate("protocol_sha", PASS, live_sha[:12]))

    # P0 parity runs before paid acquisition, so its success is carried forward as a
    # content-addressed receipt rather than a phase-row written into a different UID's ledger.
    parity_path = repo / "reports" / "P0_PARITY_PASSED.json"
    if not parity_path.exists():
        checks.append(Gate("p0_parity", FAIL, "P0 parity receipt is missing"))
    else:
        try:
            from ..canonical import canonical_json
            from ..hashing import sha256_hex

            parity = json.loads(parity_path.read_text(encoding="utf-8"))
            recorded = str(parity.get("content_sha256") or "")
            actual = sha256_hex(canonical_json(
                {k: v for k, v in parity.items() if k != "content_sha256"}))
            parity_ok = (
                parity.get("status") == "PASS"
                and parity.get("protocol_sha") == live_sha
                and len(str(parity.get("probe_inputs_sha256") or "")) == 64
                and recorded == actual
            )
            checks.append(Gate(
                "p0_parity", PASS if parity_ok else FAIL,
                f"receipt {recorded[:12]}" if parity_ok
                else "receipt status/protocol/content hash does not match this tree",
            ))
        except (OSError, json.JSONDecodeError, TypeError) as e:
            checks.append(Gate("p0_parity", FAIL, f"{type(e).__name__}: {e}"))

    try:
        verify_approval_file(repo)
        checks.append(Gate("approval", PASS, "binds the live configuration"))
    except ApprovalError as e:
        checks.append(Gate("approval", FAIL, str(e)[:300]))

    # Every handle the codec can emit, costed on the exact tokenizer this host will serve. The
    # predecessor's cap held in the abstract and failed for every handle production actually
    # produced, so this is enumerated rather than argued, and it runs before the first GPU hour
    # rather than being discovered in the artifacts afterwards.
    try:
        from ..evidence.model_tokenizer import load_frozen_tokenizer
        from ..p1.handle_proof import load_or_prove

        proof = load_or_prove(
            load_frozen_tokenizer(settings),
            receipt_path=repo / "reports" / "HANDLE_DOMAIN_PROOF.json",
        )
        checks.append(Gate(
            "publication_handle_domain", PASS,
            f"{proof['capacity']} handles, max {proof['max_tokens']} tokens "
            f"(cap {proof['max_publication_handle_tokens']}), "
            f"proof {str(proof['content_sha256'])[:12]}",
        ))
    except Exception as e:  # noqa: BLE001 - any failure here must stop the launch
        checks.append(Gate(
            "publication_handle_domain", FAIL, f"{type(e).__name__}: {str(e)[:300]}"))

    # The pre-treatment design receipt, the acquisition manifest and the source-cluster split
    # gate all read artifacts produced by the Week-1 corpus phases, which are gone. Their
    # Freeze-1 replacements are the frozen prereg document and the benchmark split manifest,
    # and they will register gates here the same way. Deliberately absent rather than stubbed
    # PASS: a preflight that reports PASS for a check it did not run is the exact failure this
    # module exists to prevent.
    receipt_tasks = [
        {"task_id": task_id, "split": str(settings.get("week1", "screen", "split"))}
        for task_id in acquired_task_ids(settings)
    ]

    # Every runner-visible world is still re-verified from its published pool and object bytes.
    broken = _runner_world_failures(settings, receipt_tasks)
    checks.append(Gate("frozen_world", PASS if receipt_tasks and not broken else FAIL,
                       f"{len(receipt_tasks)} worlds frozen and verified"
                       if receipt_tasks and not broken
                       else "; ".join(broken[:5]) or "no frozen worlds"))

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


def _runner_world_failures(settings: Settings, task_rows: list[dict]) -> list[str]:
    """Verify only the runner-safe world; never traverse the steward tree."""
    from ..canonical import canonical_json
    from ..hashing import sha256_hex
    from ..object_store import ObjectStore

    root = settings.path("frozen_corpus_for_runner")
    objects = ObjectStore(root / "objects")
    failures: list[str] = []
    seen: set[str] = set()
    for task in task_rows:
        task_id = str(task.get("task_id") or "")
        if not task_id or task_id in seen:
            failures.append(f"duplicate/empty task coordinate {task_id!r}")
            continue
        seen.add(task_id)
        pool_path = root / "pools" / f"{task_id}.json"
        task_path = root / "tasks" / f"{task_id}.json"
        try:
            pool = json.loads(pool_path.read_text(encoding="utf-8"))
            recorded = str(pool.get("pool_sha256") or "")
            actual = sha256_hex(canonical_json({
                key: value for key, value in pool.items() if key != "pool_sha256"
            }))
            if (
                recorded != actual
                or recorded != str(task.get("source_pool_sha256") or "")
                or str(pool.get("task_id") or "") != task_id
            ):
                raise ValueError("pool identity/hash differs from frozen receipt")
            view = json.loads(task_path.read_text(encoding="utf-8"))
            if (
                str(view.get("task_id") or "") != task_id
                or str(view.get("split") or "") != str(task.get("split") or "")
            ):
                raise ValueError("runner task view differs from frozen receipt")
            for snapshot in (pool.get("snapshots") or {}).values():
                if (
                    not isinstance(snapshot, dict)
                    or not objects.verify(str(snapshot.get("object_ref") or ""))
                ):
                    raise ValueError("snapshot object is missing or corrupt")
        except (OSError, json.JSONDecodeError, KeyError, ValueError) as exc:
            failures.append(f"{task_id}: {type(exc).__name__}: {exc}")
    return failures


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
