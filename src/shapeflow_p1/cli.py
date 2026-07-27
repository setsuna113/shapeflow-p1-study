"""The `shapeflow-p1` command surface (plan §19).

The commands map one-to-one to campaign phases, and each asserts the identity it must run as:
the steward acquires and freezes, the runner executes treatments, the evaluator reads truth. A
command that checked nothing but a docstring would leave the whole UID separation resting on
whoever typed it.

Once ``reports/LAUNCH_GATE_PASSED.json`` exists, mutation commands accept only
``--resume --protocol-sha <exact>``: the historically named flag carries the complete approved
execution-binding digest. A diagnostic override after launch would silently alter an approved
run, and any real change to budget, sample or phase order mints a new execution binding and needs
a new approval instead.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import typer

app = typer.Typer(add_completion=False, help="ShapeFlow P1 Week-1 study CLI")

_REPO = Path(__file__).resolve().parents[2]
_CONFIGS = {
    "decision": _REPO / "configs" / "decision.yaml",
    "budget": _REPO / "configs" / "budget_v1.yaml",
}
_SCHEMAS = _REPO / "schemas"
_CFG = typer.Option(None, "--config", help="Campaign config (configs/week1.yaml)")


def _now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _fail(message: str, code: int = 1) -> None:
    typer.echo(message, err=True)
    raise typer.Exit(code=code)


def _require_role(role: str) -> None:
    """Assert the effective identity. The separation is the boundary; this enforces it."""
    from .doctor import check_identity

    result = check_identity(role)
    if result.status == "FAIL":
        _fail(f"  FAIL  {result.name}: {result.detail}")


def _settings():
    from .campaign.settings import Settings

    return Settings.load(_REPO)


def _verified_analysis_binding(
    *,
    execution_binding_sha256: str,
    protocol_document_sha256: str,
):
    """Verify the clean analysis implementation without rewriting treatment identity.

    ``execution_binding_sha256`` is the frozen treatment binding carried by the verified score
    scope. It remains an input identity even when analysis runs from a later, separately
    approved clean commit. The live approval therefore proves a second
    ``analysis_execution_binding_sha256``; requiring it to equal the treatment digest would
    make every legitimate post-campaign analysis-only revision impossible because the binding
    includes ``approved_commit``.

    The protocol itself may not drift: a later analysis implementation can fix code, but it
    cannot silently change the hypotheses or thresholds attached to the frozen run.
    """

    from .protocol import ApprovalError, verified_execution_binding

    frozen_execution = str(execution_binding_sha256)
    if (
        len(frozen_execution) != 64
        or any(character not in "0123456789abcdef" for character in frozen_execution)
    ):
        _fail("frozen treatment execution binding is not a SHA-256 digest", code=2)
    try:
        binding = verified_execution_binding(_REPO)
    except ApprovalError as exc:
        _fail(f"refusing post-outcome analysis under unapproved code: {exc}", code=2)
    if binding.protocol_sha != str(protocol_document_sha256):
        _fail(
            "refusing post-outcome analysis: live protocol document differs from the "
            "frozen evaluation scope",
            code=2,
        )
    return binding


def _launched() -> bool:
    return (_REPO / "reports" / "LAUNCH_GATE_PASSED.json").exists()


def _assert_post_launch_flags(protocol_sha: Optional[str], resume: bool) -> None:
    """After launch, resume only under the exact complete approved execution binding."""
    if not _launched():
        return
    from .protocol import ApprovalError, verified_execution_binding

    if not resume:
        _fail("this run has already launched; mutation requires --resume", code=2)
    if not protocol_sha:
        _fail(
            "this run has already launched; --protocol-sha must carry the exact approved "
            "execution-binding digest",
            code=2,
        )
    try:
        verified_execution_binding(_REPO, expected_digest=protocol_sha)
    except ApprovalError as exc:
        _fail(
            "--protocol-sha is the approved execution-binding digest after launch; "
            f"resume identity did not verify: {exc}",
            code=2,
        )


def _content_sha(body: dict) -> str:
    """The digest of everything except the digest field itself."""
    from .canonical import canonical_json
    from .hashing import sha256_hex

    return sha256_hex(canonical_json(
        {key: value for key, value in body.items() if key != "content_sha256"}))


def _write_once_json(path: Path, body: dict, *, digest_field: str, label: str) -> None:
    """Create, or verify an identical existing artifact. Never replace.

    Exclusive creation rather than exists()-then-write: two resumptions reaching a freeze at once
    is a truncation race, and a pre-registration artifact that can be silently rewritten is not
    pre-registration.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return
    except FileExistsError:
        pass
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get(digest_field) != body.get(digest_field):
        _fail(
            f"{path} already contains a different {label}; it is write-once. "
            f"Existing {existing.get(digest_field, '')[:12]}, "
            f"computed {str(body.get(digest_field, ''))[:12]}."
        )


def _require_approval():
    """Every command that can spend verifies the approval itself.

    Not in the launch script. A shell wrapper checks the approval once, at the top, for a
    sequence of commands each of which can be run on its own -- and each of which spends
    money or GPU hours when it is. `_assert_post_launch_flags` is not this check either: it
    returns immediately until reports/LAUNCH_GATE_PASSED.json exists, and reports/ is
    gitignored, so before the first launch it is a pass-through.
    """
    from .protocol import ApprovalError, verified_execution_binding

    try:
        return verified_execution_binding(_REPO)
    except ApprovalError as e:
        _fail(f"refusing to run: the approval does not bind this configuration.\n{e}")


def _record_phase(phase_name: str, detail: dict) -> None:
    """Record a completed campaign phase.

    Only three phases were ever written -- GPU_SMOKE_PASSED, SCREEN_RUNNING and
    SCREEN_COMPLETE -- and the state machine's first legal edge from NEW is DOCTOR_PASSED.
    So with the earlier gates unrecorded, `begin(GPU_SMOKE_PASSED)` raised
    IllegalPhaseTransition: a *passing* canary crashed on success while a failing one exited
    cleanly, and `run-screen` could not start at all.
    """
    from .campaign.phases import PhaseStore
    from .experiment.ledger import Ledger
    from .experiment.state_machine import Phase
    from .protocol import ApprovalError, verified_execution_binding

    settings = _settings()
    try:
        binding = verified_execution_binding(_REPO)
    except ApprovalError as exc:
        _fail(f"cannot record phase without a verified execution binding: {exc}")
    # Campaign phase ownership is the runner's ledger, the same ledger CampaignRunner reads.
    # Writing gate phases into the provider ledger made the state machine split-brain: the
    # canary saw NEW even though doctor/acquisition/parity had "completed" elsewhere. It also
    # asked non-provider UIDs to open a 0700 provider directory.
    path = settings.path("runs") / "ledger.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    ledger = Ledger(str(path))
    try:
        phases = PhaseStore(ledger, protocol_sha=binding.digest)
        phase = Phase(phase_name)
        phases.begin(phase)
        phases.complete(phase, detail)
    finally:
        ledger.close()


def _provider_client(settings, role: str = "runner"):
    from .providers.provider_client import ProviderClient, load_role_token

    token_dir = str(settings.get("week1", "provider", "token_dir"))
    host = settings.get("week1", "provider", "bind_host")
    port = settings.get("week1", "provider", "bind_port")
    return ProviderClient(base_url=f"http://{host}:{port}",
                          token=load_role_token(token_dir, role))


def _unsigned_content_sha256(body: Mapping[str, Any],
                             field: str = "content_sha256") -> str:
    """Hash a content-addressed JSON wrapper without trusting its claimed digest."""
    from .canonical import canonical_json
    from .hashing import sha256_hex

    return sha256_hex(canonical_json({
        key: value for key, value in body.items() if key != field
    }))


def _load_content_addressed_json(path: Path, *,
                                  hash_field: str = "content_sha256") -> dict:
    """Load one exact JSON object and verify its wrapper digest."""
    try:
        body = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid content-addressed JSON {path}: {exc}") from exc
    if not isinstance(body, dict):
        raise ValueError(f"content-addressed JSON {path} is not an object")
    recorded = str(body.get(hash_field) or "")
    actual = _unsigned_content_sha256(body, hash_field)
    if recorded != actual:
        raise ValueError(
            f"{path} was edited: records {recorded!r}, hashes to {actual}")
    return body


def _load_e2e_analysis_design(settings, scope_receipt: Mapping[str, Any]) -> tuple[
    dict[str, dict], dict, dict
]:
    """Verify the evaluator-side pre-treatment design against the evaluated scope.

    The analysis-design receipt itself is runner-safe and deliberately not readable through
    the evaluator tree.  Its digest is nevertheless frozen into the schedule and then into the
    verified EVALUATION_SCOPE.  The two evaluator-owned objects can be checked more strongly:
    their wrapper digests, every task-feature record digest, the registry index, the spec's
    registry binding, and both scope bindings are all re-derived here.
    """
    from .analysis.design import (
        ELIGIBILITY_SPEC_FILENAME,
        EVALUATOR_RECEIPT_FILENAME,
        REGISTRY_FILENAME,
        build_eligibility_spec,
    )
    from .analysis.e2e_effects import task_feature_registry_sha256

    design_dir = settings.path("evaluator_root") / "analysis_design"
    registry = _load_content_addressed_json(design_dir / REGISTRY_FILENAME)
    spec = _load_content_addressed_json(design_dir / ELIGIBILITY_SPEC_FILENAME)
    design_receipt = _load_content_addressed_json(
        design_dir / EVALUATOR_RECEIPT_FILENAME)
    if registry.get("schema_version") != "frozen_task_feature_registry_v1":
        raise ValueError("unsupported frozen task-feature registry schema")
    if spec.get("schema_version") != "e2e_eligibility_spec_v2":
        raise ValueError("unsupported frozen eligibility-spec schema")
    raw_records = registry.get("records")
    if not isinstance(raw_records, dict) or not raw_records:
        raise ValueError("frozen task-feature registry has no records")
    records: dict[str, dict] = {}
    for raw_task_id, raw_record in raw_records.items():
        task_id = str(raw_task_id)
        if not task_id or not isinstance(raw_record, dict):
            raise ValueError("frozen task-feature registry has an invalid task record")
        record = dict(raw_record)
        if (
            str(record.get("task_id") or "") != task_id
            or str(record.get("content_sha256") or "")
            != _unsigned_content_sha256(record)
        ):
            raise ValueError(
                f"frozen task-feature record {task_id!r} does not verify")
        records[task_id] = record
    derived_registry_sha = task_feature_registry_sha256(records)
    recorded_registry_sha = str(
        registry.get("task_feature_registry_sha256") or "")
    if recorded_registry_sha != derived_registry_sha:
        raise ValueError(
            "task-feature registry index does not address its exact task records")
    if str(spec.get("task_feature_registry_sha256") or "") != derived_registry_sha:
        raise ValueError("eligibility spec does not bind the frozen task-feature registry")
    expected_spec = build_eligibility_spec(settings, registry)
    if spec != expected_spec:
        raise ValueError(
            "frozen eligibility spec differs from the hash-locked decision design"
        )

    analysis_receipt_sha = str(
        scope_receipt.get("analysis_design_receipt_sha256") or "")
    scope_registry_sha = str(
        scope_receipt.get("task_feature_registry_sha256") or "")
    scope_spec_sha = str(
        scope_receipt.get("eligibility_spec_content_sha256") or "")
    if analysis_receipt_sha != str(design_receipt.get("content_sha256") or ""):
        raise ValueError(
            "evaluation scope names a different analysis-design receipt")
    if scope_registry_sha != derived_registry_sha:
        raise ValueError(
            "evaluation scope names a different task-feature registry")
    if scope_spec_sha != str(spec["content_sha256"]):
        raise ValueError(
            "evaluation scope names a different eligibility specification")
    expected_receipt_bindings = {
        "task_feature_registry_sha256": derived_registry_sha,
        "feature_registry_content_sha256": str(registry["content_sha256"]),
        "eligibility_spec_content_sha256": str(spec["content_sha256"]),
        "decision_config_sha256": str(settings.shas["decision"]),
        "variants_config_sha256": str(settings.shas["variants"]),
        "week1_config_sha256": str(settings.shas["week1"]),
    }
    for field, expected in expected_receipt_bindings.items():
        if str(design_receipt.get(field) or "") != expected:
            raise ValueError(
                f"analysis-design receipt {field} does not bind evaluator artifacts/config")
    bindings = {
        "analysis_design_receipt_sha256": analysis_receipt_sha,
        "task_feature_registry_sha256": derived_registry_sha,
        "feature_registry_content_sha256": str(registry["content_sha256"]),
        "eligibility_spec_content_sha256": str(spec["content_sha256"]),
    }
    return records, spec, bindings


def _e2e_semantic_mappings(settings) -> tuple[
    dict[str, str], dict[str, str], dict[str, dict]
]:
    """Resolve factorial and matched-ablation semantics from hash-locked configs."""
    from .analysis.matched import resolve_arm_semantics

    factorial = settings.get("decision", "e2e_analysis", "core_factorial")
    if not isinstance(factorial, dict) or set(factorial) != {"p0", "h", "c", "hc"}:
        raise ValueError("decision core_factorial must define exactly p0/h/c/hc")
    arm_map: dict[str, str] = {}
    expected_variants: dict[str, str] = {}
    for semantic in ("p0", "h", "c", "hc"):
        corner = factorial[semantic]
        if not isinstance(corner, dict):
            raise ValueError(f"decision core_factorial.{semantic} is not an object")
        arm_id = str(corner.get("arm_id") or "")
        page = str(corner.get("page_variant") or "")
        close = str(corner.get("close_variant") or "")
        if not arm_id or not page or not close:
            raise ValueError(
                f"decision core_factorial.{semantic} lacks arm/page/close identity")
        arm_map[semantic] = arm_id
        expected_variants[semantic] = f"{page}+{close}"
    if len(set(arm_map.values())) != 4 or len(set(expected_variants.values())) != 4:
        raise ValueError("core factorial corners must name four distinct arms and variants")

    screen = settings.get("week1", "screen_arms", "arms")
    if not isinstance(screen, list) or not screen:
        raise ValueError("week1 screen arm registry is empty")
    arms: dict[str, dict] = {}
    for raw in screen:
        if not isinstance(raw, dict):
            raise ValueError("week1 screen arm is not an object")
        arm_id = str(raw.get("arm_id") or "")
        if not arm_id or arm_id in arms:
            raise ValueError(f"week1 screen has duplicate/unnamed arm {arm_id!r}")
        arms[arm_id] = dict(raw)
    for semantic, arm_id in arm_map.items():
        configured = arms.get(arm_id)
        if configured is None:
            raise ValueError(f"core factorial arm {arm_id!r} is absent from the screen")
        actual = (
            f"{configured.get('page_variant')}+{configured.get('close_variant')}")
        if actual != expected_variants[semantic]:
            raise ValueError(
                f"decision core_factorial.{semantic} disagrees with screen arm {arm_id}")

    raw_variants = settings.get("variants", "variants")
    if not isinstance(raw_variants, list) or not raw_variants:
        raise ValueError("variant registry is empty")
    variants: dict[str, dict] = {}
    for raw in raw_variants:
        if not isinstance(raw, dict):
            raise ValueError("variant registry entry is not an object")
        variant_id = str(raw.get("variant_id") or "")
        if not variant_id or variant_id in variants:
            raise ValueError(f"duplicate/unnamed variant {variant_id!r}")
        variants[variant_id] = dict(raw)

    matched = settings.get("week1", "matched_contrasts")
    if not isinstance(matched, dict):
        raise ValueError("week1 matched_contrasts is not an object")
    executable_fields = tuple(map(
        str, matched.get("executable_variant_fields") or ()))
    requested_arms = {
        str(pair.get(side) or "")
        for pair in matched.get("pairs") or ()
        if isinstance(pair, dict)
        for side in ("left_arm_id", "right_arm_id")
    }
    if "" in requested_arms:
        raise ValueError("matched contrast has an unnamed arm")
    arm_variants: dict[str, dict] = {}
    for arm_id in sorted(requested_arms):
        arm = arms.get(arm_id)
        if arm is None:
            raise ValueError(f"matched contrast arm {arm_id!r} is absent from the screen")
        arm_variants[arm_id] = resolve_arm_semantics(
            arm_id,
            arm,
            variants,
            executable_fields,
        )
    return arm_map, expected_variants, arm_variants


def _write_e2e_analysis_once(path: Path, body: Mapping[str, Any]) -> str:
    """Exclusively create the controlled analysis artifact, or prove idempotence."""
    expected = dict(body)
    digest = str(expected.get("content_sha256") or "")
    if digest != _unsigned_content_sha256(expected):
        raise ValueError("E2E analysis body is not correctly content addressed")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(expected, indent=2, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.chmod(path, 0o440)
        return "CREATED"
    except FileExistsError:
        existing = _load_content_addressed_json(path)
        if existing != expected:
            raise ValueError(
                f"{path} already seals a different E2E analysis; output is write-once"
            ) from None
        return "EXISTING_IDENTICAL"


# --- read-only ---------------------------------------------------------------------------


@app.command()
def doctor(
    config: Path = _CFG,
    role: str = typer.Option(None, "--role", help="Assert the effective identity for this role"),
) -> None:
    """Verify the environment and stack, read-only. Fails closed on any non-PASS check."""
    from .doctor import check_git_clean, check_stack_manifest, run_pure_checks

    report = run_pure_checks(repo=_REPO, configs=_CONFIGS, schema_dir=_SCHEMAS, role=role)
    report.add(check_git_clean(_REPO))
    report.add(check_stack_manifest(
        _REPO, _REPO / "configs" / "stack.yaml",
        measurement_layer=_settings().measurement_layer,
    ))
    if config is not None and not config.exists():
        _fail(f"  FAIL  config: {config} not found")
    for check in report.checks:
        typer.echo(f"  {check.status:4}  {check.name}: {check.detail}")
    if not report.ok:
        skipped = report.skipped
        if skipped:
            _fail(f"doctor: FAILED -- {len(skipped)} check(s) could not run. A check that did "
                  "not run has not been satisfied; it is not a pass.")
        _fail("doctor: FAILED")
    _record_phase("DOCTOR_PASSED", {"role": role or "unspecified",
                                    "checks": len(report.checks)})
    typer.echo("doctor: ok")


@app.command("verify-approval")
def verify_approval(
    approval: Path = typer.Option(
        ..., "--approval", help="External append-only approval store's current pointer"),
    config: Path = _CFG,
) -> None:
    """Assert the approval binds the whole live configuration, not just three of its hashes.

    The protocol SHA is read from the tracked document, never from the environment: a value the
    caller supplies and then compares against itself is a gate that cannot fail. A caller who
    *claims* a different SHA is a disagreement about which experiment is running, and fatal.
    """
    from .protocol import ApprovalError, verified_execution_binding

    try:
        binding = verified_execution_binding(_REPO, approval_path=approval)
    except ApprovalError as e:
        _fail(f"approval mismatch: {e}")

    claimed = os.environ.get("SHAPEFLOW_PROTOCOL_SHA")
    if claimed and claimed != binding.protocol_sha:
        _fail(f"SHAPEFLOW_PROTOCOL_SHA={claimed[:12]} does not match the protocol document "
              f"({binding.protocol_sha[:12]}); the caller and the repository disagree about "
              "which protocol is being run")
    typer.echo(f"approval ok (protocol={binding.protocol_sha[:12]} binding={binding.digest[:12]})")


@app.command()
def status(status_json: Path = typer.Option(None, help="Path to runs/STATUS.json")) -> None:
    """Print the latest STATUS.json if present."""
    path = status_json or (_REPO / "reports" / "STATUS.json")
    if not path.exists():
        typer.echo("no STATUS.json yet")
        raise typer.Exit(code=0)
    typer.echo(path.read_text(encoding="utf-8"))


@app.command("verify-artifacts")
def verify_artifacts(ledger_path: Path, object_store: Path) -> None:
    """Confirm every committed work item's artifact still verifies in the object store."""
    from .experiment.ledger import Ledger
    from .object_store import ObjectStore

    ledger = Ledger(str(ledger_path))
    store = ObjectStore(object_store)
    counts = ledger.state_counts()
    committed = counts.get("COMMITTED", 0)
    rows = ledger.raw_connection.execute(
        "SELECT result_object_ref FROM attempts WHERE state='COMMITTED'").fetchall()
    bad = sum(1 for r in rows if not (r["result_object_ref"]
                                      and store.verify(r["result_object_ref"])))
    typer.echo(f"committed={committed} verified={committed - bad} corrupt={bad}")
    ledger.close()
    if bad:
        raise typer.Exit(code=1)


@app.command()
def report(
    decision_json: Path = typer.Argument(..., help="Path to a WEEK1_P1_DECISION.json"),
) -> None:
    """Render a decision JSON to Markdown on stdout (both come from one object upstream)."""
    from .analysis.decision import Verdict
    from .analysis.report import DecisionObject, EffectWithCI, NodeDecision, render_markdown

    data = json.loads(decision_json.read_text(encoding="utf-8"))

    def node(key: str) -> NodeDecision:
        n = data[key]
        ws = n.get("work_saving")
        return NodeDecision(
            node=n["node"], verdict=Verdict(n["verdict"]),
            champion_variant=n.get("champion_variant"),
            work_saving=EffectWithCI(**ws) if ws else None,
        )

    decision = DecisionObject(
        webpage_p1=node("WEBPAGE_P1"), c_visible=node("C_VISIBLE"),
        c_registry=node("C_REGISTRY"), h_plus_c_visible=node("H_PLUS_C_VISIBLE"),
        verdict_status=data["verdict_status"],
        confirmatory_power_shortfall=data.get("confirmatory_power_shortfall", False),
        human_audit_status=data.get("human_audit_status", ""),
        protocol_sha=data.get("protocol_sha", ""), freeze_sha=data.get("freeze_sha", ""),
        generated_at_utc=data.get("generated_at_utc", ""),
    )
    typer.echo(render_markdown(decision))


@app.command("authorize-budget-raise")
def authorize_budget_raise(
    config: Path = _CFG,
    reason: str = typer.Option(..., "--reason", help="why the original ceiling was wrong"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Provider-only: apply raised caps from the frozen budget config, on the record.

    ``Budget.ensure_account`` refuses to widen a ceiling, and that refusal is correct -- a cap
    that drifted upward as a side effect of loading a config would let a round spend past what
    its approval was granted against. So a raise cannot happen implicitly, and editing
    ``budget_v1.yaml`` alone changes nothing about what this ledger will allow.

    This is the explicit act that applies one. It requires a verified approval, so the wider
    ceiling is bound to a protocol version someone froze deliberately; it records an incident
    per resource carrying the old value, the new value and how much was already spent; and it
    refuses to lower anything, because a call that moved caps in both directions would just be
    an ordinary write with a longer name.
    """
    from .experiment.budget import Budget
    from .experiment.ledger import Ledger

    _require_role("provider")
    binding = _require_approval()
    settings = _settings()
    ledger_path = settings.path("provider_ledger")
    if not ledger_path.exists():
        _fail(f"no provider ledger at {ledger_path}")

    caps = settings.budget_caps()
    ledger = Ledger(str(ledger_path))
    try:
        budget = Budget(ledger)
        current = {
            r["resource"]: (r["cap"], r["settled_total"])
            for r in ledger.raw_connection.execute(
                "SELECT resource, cap, settled_total FROM budget_accounts")
        }
        changes = []
        for resource, cap in sorted(caps.items()):
            have = current.get(resource)
            if have is None or cap <= have[0]:
                continue
            changes.append((resource, have[0], cap, have[1]))
        if not changes:
            typer.echo("no cap in the frozen config is above the ledger; nothing to raise")
            return
        for resource, was, now, spent in changes:
            typer.echo(f"  {resource}: {was} -> {now}   (already spent {spent})")
        if dry_run:
            typer.echo("dry run; nothing was changed")
            return
        for resource, _was, now, _spent in changes:
            budget.authorize_cap_raise(
                resource, now,
                authorization=f"approval binding {binding.digest[:12]}",
                reason=reason,
            )
    finally:
        ledger.close()
    typer.echo(f"raised {len(changes)} cap(s) under binding {binding.digest[:12]}")


#: Why the first round's treatment artifacts cannot be analysed. Each is independently
#: sufficient; together they mean no cell in that round observed the treatment it is labelled
#: with. They are enumerated rather than summarised because a later reader deciding whether some
#: subset is salvageable needs to see all four.
TREATMENT_INVALIDATION_REASONS = {
    "P1_PUBLISHED_OUTPUT_COUNT_ZERO": (
        "Across 19 arms and 146 cells, not one P1 span was published. Every H batch raised "
        "`publication handle 'H3_1_0_1' costs 9 exact model tokens, exceeding the frozen cap 8` "
        "on its first candidate and fell back to P0 as a whole, so every H cell is a P0 cell "
        "wearing an H label."
    ),
    "H_VIEW_CONSTRUCTION_FALLBACK_ALL": (
        "The failure was total rather than partial: 2,067 recorded view-construction failures "
        "across 87 distinct handles, with no arm publishing anything. A fallback rate is an "
        "outcome; a 100% fallback rate before the first selector call is an apparatus that "
        "never ran the treatment."
    ),
    "P0_NON_VENDOR_OUTPUT_CAP_1024": (
        "summarization_model_max_tokens was 1024 against the pinned vendor default of 8192. "
        "Vendor's summariser falls back to the raw page when truncated, so 263 of 1839 P0 "
        "summaries (14.3%) published whole pages as compressed notes -- worst on the largest "
        "pages, and P1 never reaches that code path. The baseline was handicapped exactly where "
        "P1 was meant to win."
    ),
    "SERIAL_REGIME_NOT_TARGET_ODR": (
        "The engine admitted one upstream request at a time, which was a precondition of the "
        "summed-service metric rather than of causal isolation. The pinned graph summarises a "
        "result set with asyncio.gather, so the serialization queued eight concurrent summaries "
        "behind each other and produced 212 vendor timeouts that exist in no native run. The "
        "measured system is not the system the study is about."
    ),
}


@app.command("invalidate-treatment")
def invalidate_treatment(
    config: Path = _CFG,
    attempt_id: str = typer.Option(..., "--attempt-id", help="e.g. serial-engineering-smoke"),
    state: str = typer.Option("INVALIDATED_SERIAL_ENGINEERING_SMOKE", "--state"),
    note: str = typer.Option("", "--note"),
) -> None:
    """Seal a round of treatment artifacts as unanalysable, without deleting any of them.

    Nothing is removed and nothing is refunded. The point is that a later reader can tell the
    difference between "this round produced no P1 effect" and "this round never executed P1" --
    which the artifacts alone cannot say, because a P0 fallback is a legitimate part of the ITT
    design and a run that fell back 146 times out of 146 looks exactly like a run.

    The new campaign writes to per-lane trees, so the sealed round is not overwritten either.
    """
    _require_role("runner")
    settings = _settings()
    root = settings.data_root / str(settings.get("week1", "paths", "runner_root"))
    marker = root / f"{state}.json"

    inventory: dict[str, int] = {}
    for name in ("runs", "object_store", "checkpoints"):
        directory = settings.data_root / str(settings.get("week1", "paths", name))
        inventory[name] = (
            sum(1 for path in directory.rglob("*") if path.is_file())
            if directory.exists() else 0
        )

    body = {
        "schema_version": "invalidated_treatment_attempt_v1",
        "attempt_id": attempt_id,
        "state": state,
        "reasons": TREATMENT_INVALIDATION_REASONS,
        "note": note,
        "sealed_at_utc": _now(),
        "sealed_tree": str(root),
        "artifact_counts": inventory,
        "protocol_sha_at_sealing": __import__(
            "shapeflow_p1.protocol", fromlist=["protocol_sha"]).protocol_sha(_REPO),
        "analysis_permitted": False,
    }
    body["content_sha256"] = _content_sha(body)
    _write_once_json(marker, body, digest_field="content_sha256",
                     label="treatment invalidation record")
    typer.echo(json.dumps({
        "state": state, "sealed_tree": str(root),
        "artifact_counts": inventory,
        "reasons": sorted(TREATMENT_INVALIDATION_REASONS),
        "content_sha256": body["content_sha256"],
    }, indent=2, sort_keys=True))


@app.command("freeze-corpus-attempt")
def freeze_corpus_attempt(
    config: Path = _CFG,
    attempt_id: str = typer.Option(..., "--attempt-id", help="e.g. attempt-1"),
    corpus_version: str = typer.Option(..., "--corpus-version"),
    state: str = typer.Option("FAILED_AUTHORING_ATTEMPT", "--state"),
    reason: str = typer.Option("", "--reason"),
    note: str = typer.Option("", "--note"),
) -> None:
    """Close out one round of corpus work without deleting or refunding anything.

    Everything this round spent stays spent and stays on the books. The point is not to
    tidy up -- it is to make the failure a recorded fact, so a later round is visibly a new
    attempt rather than a silent continuation of a corpus nobody can reconstruct. The
    ledger is copied through the backup API first, so the frozen state of the round
    survives whatever happens to the live database next.
    """
    from .experiment.ledger import Ledger
    from .protocol import protocol_sha

    _require_role("provider")
    settings = _settings()
    ledger_path = settings.path("provider_ledger")
    if not ledger_path.exists():
        _fail(f"no provider ledger at {ledger_path}")

    ledger = Ledger(str(ledger_path))
    try:
        snapshot_dir = ledger_path.parent / "ledger_snapshots"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        snapshot = snapshot_dir / f"{attempt_id}-{_now().replace(':', '')}.sqlite"
        ledger.backup_to(str(snapshot))
        os.chmod(snapshot, 0o400)

        ledger.record_corpus_attempt(
            attempt_id=attempt_id, corpus_version=corpus_version, state=state,
            reason=reason, protocol_sha=protocol_sha(_REPO), note=note,
        )
        spend = {
            r["resource"]: {"cap": r["cap"], "reserved": r["reserved_total"],
                            "settled": r["settled_total"]}
            for r in ledger.raw_connection.execute(
                "SELECT resource, cap, reserved_total, settled_total FROM budget_accounts")
        }
        calls = [
            dict(r) for r in ledger.raw_connection.execute(
                "SELECT provider, op_class, state, COUNT(*) n FROM external_call_attempts"
                " a JOIN external_calls c USING(call_id) GROUP BY 1,2,3 ORDER BY 1,2,3")
        ]
    finally:
        ledger.close()

    body = {
        "attempt_id": attempt_id, "corpus_version": corpus_version, "state": state,
        "reason": reason, "note": note, "frozen_at_utc": _now(),
        "ledger_snapshot": str(snapshot), "spend": spend, "attempts_by_outcome": calls,
    }
    out = _REPO / "reports" / f"{state}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo(f"{state} recorded as {attempt_id}; ledger snapshot at {snapshot}")
    for resource, totals in sorted(spend.items()):
        if totals["settled"]:
            typer.echo(f"  kept on the books: {resource} = {totals['settled']}")


# --- steward -----------------------------------------------------------------------------


@app.command("freeze-approval")
def freeze_approval(
    approved_commit: str = typer.Option("", "--approved-commit"),
    approval: Path = typer.Option(None, "--approval"),
) -> None:
    """Steward-only: record approval outside the clean Git execution tree."""
    from .protocol import ApprovalError, write_approval_file

    _require_role("steward")
    try:
        binding = write_approval_file(_REPO, approved_at_utc=_now(),
                                      approved_commit=approved_commit, approval_path=approval)
    except ApprovalError as e:
        _fail(f"cannot write approval: {e}")
    typer.echo(f"approval recorded (binding={binding.digest[:12]})")


@app.command("freeze-stack")
def freeze_stack(
    config: Path = _CFG,
    engine_pid: int = typer.Option(None, "--engine-pid", help="Running vLLM pid, for its flags"),
    engine_log: Path = typer.Option(None, "--engine-log",
                                    help="The engine's startup log, for its attention backend"),
) -> None:
    """Steward-only: resolve every @STEWARD_FREEZES@ field into protocol/stack_manifest.json."""
    from .campaign.evaluate import relation_prompt_sha256
    from .campaign.truth import truth_prompt_sha256
    from .evaluation.atomizer import atomize_protocol_sha256
    from .ops.live_stack import (
        StackError,
        attention_backend_from_log,
        freeze_stack as do_freeze,
        observe,
    )

    _require_role("steward")
    settings = _settings()
    stack = settings.configs["stack"]
    observation = observe(
        gpu_uuid=str(stack["host"]["gpu_uuid"]),
        model_dir=Path(str(stack["model"]["path"])),
        vllm_python=Path(str(stack["engine"]["vllm_venv"])) / "bin" / "python",
        engine_pid=engine_pid,
    )
    # The attention backend is only knowable from a served engine: vLLM picks it at startup.
    # Recorded from the engine's own log when it is running, and left unresolved otherwise so
    # the freeze refuses rather than writing down a plausible guess.
    backend = attention_backend_from_log(engine_log) if engine_log else ""
    if backend:
        observation.values["attention_backend"] = backend
    observation.values["truth_prompt_sha256"] = truth_prompt_sha256()
    observation.values["atomize_prompt_sha256"] = atomize_protocol_sha256()
    observation.values["report_prompt_sha256"] = relation_prompt_sha256()
    try:
        body = do_freeze(_REPO, stack, observation, frozen_at_utc=_now())
    except StackError as e:
        _fail(f"freeze-stack failed: {e}")
    typer.echo(f"stack frozen (manifest={body['manifest_sha256'][:12]}, "
               f"unavailable={len(body['unavailable'])})")


@app.command()
def prepare(config: Path = _CFG,
            protocol_sha: str = typer.Option(None, "--protocol-sha"),
            resume: bool = typer.Option(False, "--resume")) -> None:
    """Steward-only: author, audit, split and seal the task registry."""
    from .campaign.prepare import prepare_corpus
    from .evaluation.judge_client import DeepSeekJudge
    from .providers.provider_client import PROVIDER_KEY_PLACEHOLDER

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("steward")
    _require_approval()
    settings = _settings()
    client = _provider_client(settings, "steward")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="TASK_AUTHOR", work_key="prepare"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
        sampling=settings.authoring_sampling(),
    )
    result = asyncio.run(prepare_corpus(
        settings, judge=judge, authored_at_utc=_now(),
        target_model=str(settings.get("stack", "model", "repo")),
    ))
    typer.echo(f"sealed {len(result.registry.tasks)} tasks "
               f"(registry={result.registry_sha256[:12]}, "
               f"steward={result.steward_task_count}, runner={result.runner_task_count})")


@app.command()
def acquire(config: Path = _CFG,
            protocol_sha: str = typer.Option(None, "--protocol-sha"),
            resume: bool = typer.Option(False, "--resume")) -> None:
    """Steward-only: call the search provider once per task and freeze the world."""
    from .acquire.exa_client import ExaCaptureClient
    from .campaign.acquire import acquire_all, exa_params_from
    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("steward")
    _require_approval()
    settings = _settings()
    client = _provider_client(settings, "steward")
    params = exa_params_from(settings)

    def factory(task_id: str) -> ExaCaptureClient:
        return ExaCaptureClient(client.exa_transport(task_id=task_id), params)

    outcome = asyncio.run(acquire_all(settings, client_factory=factory, fetched_at_utc=_now()))
    typer.echo(f"acquired={outcome.tasks_acquired} skipped={outcome.tasks_skipped} "
               f"queries ok/empty/failed={outcome.queries_ok}/{outcome.queries_empty}/"
               f"{outcome.queries_failed} root={outcome.campaign_sha256[:12]}")
    if outcome.blocked_budget:
        _fail("acquisition stopped on a refused budget reservation; already-frozen worlds kept")


@app.command("build-truth")
def build_truth(config: Path = _CFG) -> None:
    """Steward-only: build evaluator-readable TruthPackets from the frozen sources.

    The host permission boundary deliberately makes the steward the owner of both the sealed
    acquisition inputs and the evaluator tree.  The evaluator is read-only on that tree and
    scores the resulting packets later; requiring the evaluator here would leave it unable to
    read the steward inputs or create the output files.
    """
    from .campaign.acquire import acquired_task_ids
    from .campaign.truth import build_truth_for_task
    from .evaluation.judge_client import DeepSeekJudge
    from .providers.provider_client import PROVIDER_KEY_PLACEHOLDER

    _require_role("steward")
    settings = _settings()
    client = _provider_client(settings, "steward")
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_TRUTH"),
        settings.judge_model(), PROVIDER_KEY_PLACEHOLDER,
        max_retries=settings.judge_max_retries(),
        sampling=settings.judge_sampling(),
    )
    evaluator_tasks = settings.path("evaluator_root") / "tasks"
    packets = settings.path("truth_packets")
    built = skipped = 0
    for task_id in acquired_task_ids(settings):
        # An existing packet is already-frozen truth, so it is skipped rather than rebuilt --
        # the same rule `acquire` applies to an already-frozen world.
        #
        # Rebuilding is not merely wasteful, it cannot succeed: a packet is write-once, and the
        # build is not reproducible run to run (batching adapts to what truncates, a retry
        # advances the judge's seed, and only calls with byte-identical bodies replay from the
        # ledger). So a second pass re-derived a *different* artifact and the write-once guard
        # refused it -- correctly. The consequence was that any interrupted build could never be
        # resumed: every later run died on the first task that had already succeeded, and 44
        # unbuilt packets stayed unbuilt behind it.
        #
        # Skipping keeps the guard meaningful. It still fires if something rewrites a packet in
        # place; it no longer fires merely because the work was done.
        if (packets / f"{task_id}.json").exists():
            skipped += 1
            continue
        record = json.loads(
            (evaluator_tasks / f"{task_id}.json").read_text(encoding="utf-8"))
        asyncio.run(build_truth_for_task(
            settings, judge=judge, task_id=task_id,
            question=record["original_question"],
            required_facets=record["authored_facets"],
        ))
        built += 1
    typer.echo(f"truth packets built: {built} (skipped {skipped} already frozen)")


@app.command("freeze-analysis-design")
def freeze_analysis_design_command(config: Path = _CFG) -> None:
    """Steward-only: freeze outcome-blind features and the eligibility spec once."""
    from .analysis.design import freeze_analysis_design

    _require_role("steward")
    body = freeze_analysis_design(_settings())
    registry = body["registry"]
    spec = body["eligibility_spec"]
    typer.echo(json.dumps({
        "ok": True,
        "tasks": len(registry["records"]),
        "task_feature_registry_sha256":
            registry["task_feature_registry_sha256"],
        "eligibility_spec_sha256": spec["content_sha256"],
        "analysis_design_receipt_sha256": body["receipt"]["content_sha256"],
        "registry_path": body["registry_path"],
        "eligibility_spec_path": body["eligibility_spec_path"],
        "receipt_path": body["receipt_path"],
    }, indent=2, sort_keys=True))


# --- gates --------------------------------------------------------------------------------


@app.command("test-p0-parity")
def test_p0_parity(config: Path = _CFG) -> None:
    """Assert the patched hooks-off graph reproduces vendor, and that explicit P0 does too."""
    import subprocess

    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "tests/integration/test_p0_parity.py"],
        cwd=str(_REPO), capture_output=True, text=True, timeout=3600,
    )
    typer.echo(result.stdout[-4000:])
    if result.returncode != 0:
        typer.echo(result.stderr[-4000:], err=True)
        _fail("P0 parity failed; all GPU screening is barred")
    from .canonical import canonical_json
    from .hashing import sha256_hex
    from .protocol import protocol_sha

    probe_paths = (
        _REPO / "tests" / "integration" / "test_p0_parity.py",
        _REPO / "tests" / "integration" / "parity_probe.py",
        _REPO / "patches" / "odr_p1_hooks.patch",
    )
    report = {
        "status": "PASS",
        "protocol_sha": protocol_sha(_REPO),
        "probe": "tests/integration/test_p0_parity.py",
        "probe_inputs_sha256": sha256_hex(canonical_json({
            str(path.relative_to(_REPO)): sha256_hex(path.read_bytes())
            for path in probe_paths
        })),
    }
    report["content_sha256"] = sha256_hex(canonical_json(report))
    path = _REPO / "reports" / "P0_PARITY_PASSED.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    typer.echo("p0 parity: ok")


@app.command()
def accept(config: Path = _CFG) -> None:
    """Run the acceptance matrix and write reports/ACCEPTANCE.json."""
    from .ops.acceptance import run_acceptance

    settings = _settings()
    body = run_acceptance(settings, repo=_REPO)
    path = _REPO / "reports" / "ACCEPTANCE.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for gate in body["gates"]:
        typer.echo(f"  {gate['status']:4}  {gate['name']}: {gate['detail']}")
    if not body["ok"]:
        _fail("acceptance: FAILED")
    typer.echo("acceptance: ok")


@app.command()
def preflight(
    config: Path = _CFG,
    approved_protocol_sha: str = typer.Option(None, "--approved-protocol-sha"),
) -> None:
    """Campaign preflight: the approval, the sealed corpus, the frozen world, the schedule."""
    from .ops.acceptance import run_preflight

    settings = _settings()
    body = run_preflight(settings, repo=_REPO, approved_protocol_sha=approved_protocol_sha)
    for check in body["checks"]:
        typer.echo(f"  {check['status']:4}  {check['name']}: {check['detail']}")
    if not body["ok"]:
        _fail("preflight: FAILED")
    # Preflight is run by the runner after acquisition. It has just re-derived the sealed
    # registry, every frozen world, the acquisition manifest and the P0 parity receipt, so it
    # is the one identity that can safely advance the runner-owned phase spine without giving
    # the truth-holding steward write access to treatment state.
    parity = json.loads(
        (_REPO / "reports" / "P0_PARITY_PASSED.json").read_text(encoding="utf-8"))
    _record_phase("ACQUISITION_COMPLETE", {
        "preflight_protocol_sha": body["protocol_sha"],
        "campaign_manifest": next(
            c["detail"] for c in body["checks"]
            if c["name"] == "campaign_manifest_receipt"),
        "analysis_design_receipt_sha256":
            body["analysis_design_receipt_sha256"],
    })
    _record_phase("SNAPSHOTS_FROZEN", {
        "frozen_world": next(
            c["detail"] for c in body["checks"] if c["name"] == "frozen_world"),
    })
    _record_phase("P0_PARITY_PASSED", {
        "receipt_sha256": parity["content_sha256"],
        "probe_inputs_sha256": parity["probe_inputs_sha256"],
    })
    typer.echo("preflight: ok")


@app.command()
def smoke(config: Path = _CFG,
          tasks: int = typer.Option(None, "--tasks"),
          protocol_sha: str = typer.Option(None, "--protocol-sha"),
          resume: bool = typer.Option(False, "--resume")) -> None:
    """Runner-only: the real GPU canary. Engineering correctness only, never a quality gate."""
    from .campaign.canary import run_canary

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    binding = _require_approval()
    settings = _settings()
    body = asyncio.run(run_canary(
        settings,
        repo=_REPO,
        task_limit=tasks,
        execution_binding_sha256=binding.digest,
    ))
    path = _REPO / "reports" / "GPU_SMOKE_REPORT.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body["markdown"], encoding="utf-8")
    (_REPO / "reports" / "GPU_SMOKE.json").write_text(
        json.dumps(body["json"], indent=2, sort_keys=True) + "\n", encoding="utf-8")
    for check in body["json"]["checks"]:
        typer.echo(f"  {check['status']:4}  {check['name']}: {check['detail']}")
    if not body["json"]["ok"]:
        _fail("gpu smoke: FAILED")
    typer.echo("gpu smoke: ok")


# --- the campaign ---------------------------------------------------------------------------


@app.command("freeze-shards")
def freeze_shards(config: Path = _CFG) -> None:
    """Steward-only: freeze which GPU lane runs which task, before any of them runs.

    Task-atomic. Every replicate and every arm of one task stays on one lane, because a block is
    paired-valid only under a single engine epoch and the paired contrast is P1 against P0 on
    the same task -- P0 on one card and P1 on another would fold that card's clocks and thermals
    into the treatment effect with nothing afterwards able to separate them.

    Balanced on the frozen corpus's own byte size, which exists before any arm runs. Written
    write-once: a lane chosen after a result is seen is not a partition, it is a selection.
    """
    from .campaign.acquire import acquired_task_ids
    from .campaign.schedule import build_blocks
    from .campaign.screen import available_tasks
    from .campaign.sharding import SHARD_MANIFEST_FILENAME, Lane, build_shard_manifest

    _require_role("steward")
    settings = _settings()
    binding = _require_approval()
    split = str(settings.get("week1", "screen", "split"))
    tasks = available_tasks(settings, split)
    if not tasks:
        _fail(f"freeze-shards: no {split} task has a frozen world")

    from .campaign.runner import CampaignRunner  # arm resolution lives with the runner

    arms = CampaignRunner.arms_from_config_static(
        settings, str(settings.get("week1", "screen", "arms_block")))
    manifest = build_blocks(
        execution_binding_sha256=binding.digest,
        protocol_sha=binding.protocol_sha,
        split=split,
        task_ids=tasks,
        arms=arms,
        seeds=[int(s) for s in settings.get("week1", "screen", "seeds")],
        layer=settings.measurement_layer,
        claim_scope=settings.claim_scope,
        second_seed_fraction=float(settings.get("week1", "screen", "second_seed_fraction")),
    )

    shards = settings.get("week1", "measurement", "shards")
    lane_count = int(shards["lane_count"])
    pool = list(settings.get("stack", "host", "gpu_uuid_pool"))
    if len(pool) < lane_count:
        _fail(f"freeze-shards: {lane_count} lanes but only {len(pool)} GPUs in the frozen pool")
    lanes = [
        Lane(
            shard_id=index,
            gpu_uuid=str(pool[index]),
            vllm_port=int(shards["vllm_base_port"]) + index,
            provider_port=int(shards["provider_base_port"]) + index,
            runner_root=str(shards["runner_root_template"]).format(lane=index),
            serves_paid_upstreams=(index == int(shards["paid_upstream_lane"])),
        )
        for index in range(lane_count)
    ]

    # Pre-treatment only: the frozen corpus's bytes for a task exist before any arm runs.
    costs: dict[str, float] = {}
    del acquired_task_ids
    pool_dir = settings.path("frozen_corpus_for_runner") / "pools"
    for task_id in tasks:
        try:
            costs[task_id] = float((pool_dir / f"{task_id}.json").stat().st_size)
        except OSError as exc:
            _fail(f"freeze-shards: no frozen pool for {task_id}: {exc}")

    stack_manifest = _REPO / "protocol" / "stack_manifest.json"
    try:
        stack_sha = str(json.loads(
            stack_manifest.read_text(encoding="utf-8")).get("manifest_sha256") or "")
    except (OSError, json.JSONDecodeError) as exc:
        _fail(f"freeze-shards: stack manifest unreadable ({exc}); run freeze-stack first")
    body = build_shard_manifest(
        manifest, lanes=lanes, task_costs=costs,
        stack_manifest_sha256=stack_sha, protocol_sha256=binding.protocol_sha,
    )
    directory = settings.path("shards")
    directory.mkdir(parents=True, exist_ok=True)
    _write_once_json(directory / SHARD_MANIFEST_FILENAME, body,
                     digest_field="shard_manifest_sha256", label="shard manifest")
    typer.echo(json.dumps({
        "shard_manifest_sha256": body["shard_manifest_sha256"],
        "lane_count": lane_count,
        "tasks": len(tasks),
        "cells_by_shard": {k: len(v) for k, v in body["cells_by_shard"].items()},
    }, indent=2, sort_keys=True))


@app.command("merge-shards")
def merge_shards(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
) -> None:
    """Steward-only: reconstitute one campaign from the lanes, or refuse.

    Four separately-valid lanes are not a valid campaign. A task silently run twice, a lane's
    blocks quietly missing, or one task's arms split across two GPUs each leave every individual
    lane internally consistent -- the error exists only in the union, so this is the last place
    it can be caught. Nothing downstream reads a lane's root directly.
    """
    from .campaign.schedule import FROZEN_ROOT_FILENAME, build_blocks
    from .campaign.screen import available_tasks
    from .campaign.sharding import (
        SHARD_MANIFEST_FILENAME,
        ShardMergeError,
        merge_shard_freeze_roots,
    )

    _require_role("steward")
    settings = _settings()
    binding = _require_approval()
    split = str(settings.get("week1", "screen", "split"))

    from .campaign.runner import CampaignRunner

    manifest = build_blocks(
        execution_binding_sha256=binding.digest,
        protocol_sha=binding.protocol_sha,
        split=split,
        task_ids=available_tasks(settings, split),
        arms=CampaignRunner.arms_from_config_static(
            settings, str(settings.get("week1", "screen", "arms_block"))),
        seeds=[int(s) for s in settings.get("week1", "screen", "seeds")],
        layer=settings.measurement_layer,
        claim_scope=settings.claim_scope,
        second_seed_fraction=float(settings.get("week1", "screen", "second_seed_fraction")),
    )
    shard_dir = settings.path("shards")
    body = json.loads((shard_dir / SHARD_MANIFEST_FILENAME).read_text(encoding="utf-8"))

    template = str(settings.get("week1", "measurement", "shards", "runner_root_template"))
    suffix = str(settings.get("week1", "paths", "runs"))[
        len(str(settings.get("week1", "paths", "runner_root"))):
    ].lstrip("/")
    roots: dict[int, dict] = {}
    for lane in body["lanes"]:
        shard_id = int(lane["shard_id"])
        path = (
            settings.data_root / template.format(lane=shard_id) / suffix
            / "e2e_blocks" / run_id / FROZEN_ROOT_FILENAME
        )
        try:
            roots[shard_id] = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            _fail(f"merge-shards: lane {shard_id} freeze root unreadable at {path}: {exc}")
    try:
        merged = merge_shard_freeze_roots(body, manifest, roots)
    except ShardMergeError as exc:
        _fail(f"merge-shards: BLOCKED: {exc}")
    _write_once_json(shard_dir / f"MERGED_ROOT_{run_id}.json", merged,
                     digest_field="merged_root_sha256", label="merged campaign root")
    typer.echo(json.dumps({
        "merged_root_sha256": merged["merged_root_sha256"],
        "lane_count": merged["lane_count"],
        "total_cells": merged["total_cells"],
        "blocks": len(merged["blocks"]),
    }, indent=2, sort_keys=True))


@app.command("run-screen")
def run_screen(config: Path = _CFG,
               resume: bool = typer.Option(False, "--resume"),
               protocol_sha: str = typer.Option(None, "--protocol-sha"),
               max_cells: int = typer.Option(None, "--max-cells")) -> None:
    """Runner-only: causal screening over the FORMATIVE_SCREEN split, in paired blocks."""
    from .campaign.screen import run_screening

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    binding = _require_approval()
    body = asyncio.run(run_screening(
        _settings(),
        repo=_REPO,
        max_cells=max_cells,
        execution_binding_sha256=binding.digest,
    ))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    if not body.get("ok", False):
        raise typer.Exit(code=1)


@app.command("run-week1")
def run_week1(config: Path = _CFG,
              resume: bool = typer.Option(False, "--resume"),
              protocol_sha: str = typer.Option(None, "--protocol-sha")) -> None:
    """Runner-only: the Week-1 campaign. Idempotent; a restart fills gaps."""
    from .campaign.screen import run_screening

    _assert_post_launch_flags(protocol_sha, resume)
    _require_role("runner")
    binding = _require_approval()
    body = asyncio.run(run_screening(
        _settings(),
        repo=_REPO,
        max_cells=None,
        execution_binding_sha256=binding.digest,
    ))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))
    if not body.get("ok", False):
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    phase_id: str = typer.Option(..., "--phase-id"),
    frozen_blocks_dir: Path = typer.Option(..., "--frozen-blocks-dir"),
) -> None:
    """Evaluator-only: score one exact run/phase from its immutable block manifests."""
    from .campaign.evaluate import evaluate_frozen

    _require_role("evaluator")
    binding = _require_approval()
    body = asyncio.run(evaluate_frozen(
        _settings(), repo=_REPO, run_id=run_id,
        phase_id=phase_id, frozen_blocks_dir=frozen_blocks_dir,
        execution_binding_sha256=binding.digest,
        protocol_document_sha256=binding.protocol_sha,
    ))
    typer.echo(json.dumps(body, indent=2, sort_keys=True))


@app.command("prepare-human-audit")
def prepare_human_audit_command(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    phase_id: str = typer.Option(..., "--phase-id"),
) -> None:
    """Evaluator-only: freeze the content-addressed human-review queue."""
    from .evaluation.human_audit_workflow import prepare_human_audit

    _require_role("evaluator")
    try:
        body = prepare_human_audit(
            _settings(), run_id=run_id, phase_id=phase_id)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"prepare-human-audit failed: {exc}", code=2)
    typer.echo(json.dumps(body, indent=2, sort_keys=True))


@app.command("finalize-human-audit")
def finalize_human_audit_command(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    phase_id: str = typer.Option(..., "--phase-id"),
    results_json: Path = typer.Option(..., "--results-json"),
) -> None:
    """Evaluator-only: verify reviewer results and emit an audit-gate receipt."""
    from .evaluation.human_audit_workflow import finalize_human_audit

    _require_role("evaluator")
    try:
        results = json.loads(results_json.read_text(encoding="utf-8"))
        if not isinstance(results, dict):
            raise ValueError("human-audit results JSON must be an object")
        body = finalize_human_audit(
            _settings(),
            run_id=run_id,
            phase_id=phase_id,
            results=results,
        )
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
        _fail(f"finalize-human-audit failed: {exc}", code=2)
    typer.echo(json.dumps(body, indent=2, sort_keys=True))


@app.command("analyze-itt")
def analyze_itt(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    phase_id: str = typer.Option(..., "--phase-id"),
) -> None:
    """Build the frozen all-offered ITT estimands for one exact run/phase.

    Output location and bootstrap policy come from the hash-locked protocol; neither is an
    analyst-selected post-outcome option.
    """
    from .analysis.estimands import (
        build_verdict_inputs,
        load_scoped_scores,
        write_verdict_inputs,
    )
    from .evaluation.human_audit_workflow import load_human_audit_receipt
    from .scoped_paths import resolve_scoped_path

    _require_role("evaluator")
    settings = _settings()
    records = load_scoped_scores(
        settings.path("judgments"), run_id=run_id, phase_id=phase_id)

    scope_receipt = getattr(records, "scope_receipt", None)
    if not isinstance(scope_receipt, Mapping):
        _fail("verified all-offered evaluation scope is absent", code=2)
    analysis_binding = _verified_analysis_binding(
        execution_binding_sha256=str(
            scope_receipt.get("execution_binding_sha256") or ""
        ),
        protocol_document_sha256=str(
            scope_receipt.get("protocol_document_sha256") or ""
        ),
    )
    try:
        # Cluster ownership is a pre-treatment feature.  It must come from the exact registry
        # whose digest is bound by EVALUATION_SCOPE and the evaluator-side design receipt, not
        # from mutable task-view JSON that merely happens to carry a cluster_id field.
        task_features, _eligibility_spec, _design_bindings = (
            _load_e2e_analysis_design(settings, scope_receipt)
        )
        cluster_by_task = {
            task_id: str(record.get("cluster_id") or "")
            for task_id, record in task_features.items()
        }
        if not cluster_by_task or any(not value for value in cluster_by_task.values()):
            raise ValueError(
                "frozen task-feature registry has a task without cluster_id")
        truth_sha_by_task: dict[str, str] = {}
        score_sha_by_block: dict[str, str] = {}
        for record in records:
            task_id = str(record.get("task_id") or "")
            block_id = str(record.get("block_id") or "")
            truth_sha = str(record.get("truth_packet_sha256") or "")
            score_sha = str(record.get("content_sha256") or "")
            if (
                not task_id
                or not block_id
                or len(truth_sha) != 64
                or len(score_sha) != 64
            ):
                raise ValueError(
                    "evaluated score lacks task/block/truth/score content identity")
            previous = truth_sha_by_task.setdefault(task_id, truth_sha)
            if previous != truth_sha:
                raise ValueError(
                    f"task {task_id!r} was scored against multiple truth packets")
            if block_id in score_sha_by_block:
                raise ValueError(f"duplicate evaluated block {block_id!r}")
            score_sha_by_block[block_id] = score_sha
        human_audit_receipt = load_human_audit_receipt(
            settings,
            run_id=run_id,
            phase_id=phase_id,
            evaluation_scope_sha256=str(
                scope_receipt["evaluation_scope_sha256"]),
            execution_binding_sha256=str(
                scope_receipt["execution_binding_sha256"]),
            protocol_document_sha256=str(
                scope_receipt["protocol_document_sha256"]),
            truth_packet_sha256_by_task=truth_sha_by_task,
            score_content_sha256_by_block=score_sha_by_block,
        )
        destination = resolve_scoped_path(
            settings.path("judgments"),
            run_id=run_id,
            phase_id=phase_id,
            tail=("analysis", "ITT_VERDICT_INPUTS.json"),
        )
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"analyze-itt failed: {exc}", code=2)

    body = build_verdict_inputs(
        records,
        cluster_by_task=cluster_by_task,
        n_boot=int(settings.get("decision", "e2e_analysis", "n_boot")),
        seed_namespace=str(
            settings.get("decision", "e2e_analysis", "seed_namespace")),
        human_audit_receipt=human_audit_receipt,
    )
    body = dict(body)
    body["analysis_execution_binding_sha256"] = analysis_binding.digest
    body["analysis_approved_commit"] = analysis_binding.approved_commit
    body["content_sha256"] = _unsigned_content_sha256(body)
    digest = write_verdict_inputs(body, destination)
    typer.echo(json.dumps({
        "ok": True,
        "run_id": run_id,
        "phase_id": phase_id,
        "blocks_offered": body["blocks_offered"],
        "content_sha256": digest,
        "output": str(destination),
        "verdict_ready_arms": sorted(
            arm for arm, value in body["arms"].items() if value["verdict_ready"]),
    }, indent=2, sort_keys=True))


@app.command("analyze-e2e")
def analyze_e2e(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    phase_id: str = typer.Option(..., "--phase-id"),
) -> None:
    """Evaluator-only: build frozen factorial, eligibility, and matched-ablation results.

    There is intentionally no output-path option.  An analyst cannot redirect a post-outcome
    result over an arbitrary file; one exact run/phase owns one write-once
    ``analysis/E2E_ANALYSIS.json`` under the evaluator judgment tree.
    """
    from .analysis.e2e_effects import build_e2e_effects
    from .analysis.estimands import load_scoped_scores
    from .analysis.matched import build_matched_contrasts
    from .canonical import canonical_json
    from .hashing import sha256_hex
    from .scoped_paths import resolve_scoped_path

    _require_role("evaluator")
    settings = _settings()
    records = load_scoped_scores(
        settings.path("judgments"), run_id=run_id, phase_id=phase_id)
    scope_receipt = getattr(records, "scope_receipt", None)
    if not isinstance(scope_receipt, Mapping):
        _fail("verified all-offered evaluation scope is absent", code=2)
    analysis_binding = _verified_analysis_binding(
        execution_binding_sha256=str(
            scope_receipt.get("execution_binding_sha256") or ""
        ),
        protocol_document_sha256=str(
            scope_receipt.get("protocol_document_sha256") or ""
        ),
    )

    try:
        task_features, eligibility_spec, design_bindings = (
            _load_e2e_analysis_design(settings, scope_receipt)
        )
        arm_map, expected_variants, arm_variants = (
            _e2e_semantic_mappings(settings)
        )
        analysis_cfg = settings.get("decision", "e2e_analysis")
        n_boot = int(analysis_cfg["n_boot"])
        seed_namespace = str(analysis_cfg["seed_namespace"])
        quality_view = str(analysis_cfg["primary_quality_view"])
        quality_metric = str(analysis_cfg["primary_quality_metric"])
        if quality_view not in set(map(str, analysis_cfg["quality_views"])):
            raise ValueError(
                "primary quality view is absent from e2e_analysis")
        if quality_metric not in set(map(str, analysis_cfg["quality_metrics"])):
            raise ValueError(
                "primary quality metric is absent from e2e_analysis")

        e2e = build_e2e_effects(
            records,
            task_features=task_features,
            eligibility_spec=eligibility_spec,
            scope_receipt=scope_receipt,
            arm_map=arm_map,
            expected_variant_ids=expected_variants,
            quality_view=quality_view,
            quality_metric=quality_metric,
            task_level_joint_outcomes_policy=settings.get(
                "decision", "task_level_joint_outcomes"
            ),
            n_boot=n_boot,
            seed_namespace=seed_namespace,
        )
        if str(e2e.get("content_sha256") or "") != _unsigned_content_sha256(e2e):
            raise ValueError("E2E effects builder returned a non-verifying artifact")

        cluster_by_task = {
            task_id: str(record.get("cluster_id") or "")
            for task_id, record in task_features.items()
        }
        seed_digest = sha256_hex(canonical_json({
            "namespace": seed_namespace,
            "run_id": run_id,
            "phase_id": phase_id,
            "week1_config_sha256": settings.shas["week1"],
        }))
        matched_seed = int(seed_digest[:8], 16)
        matched = build_matched_contrasts(
            records,
            scope_receipt=scope_receipt,
            cluster_by_task=cluster_by_task,
            matched_contrasts=settings.get("week1", "matched_contrasts"),
            arm_variants=arm_variants,
            quality_views=analysis_cfg["quality_views"],
            quality_metrics=analysis_cfg["quality_metrics"],
            structured_increment_policy=settings.get(
                "decision", "structured_increment"
            ),
            n_boot=n_boot,
            seed=matched_seed,
        )
        if (
            str(matched.get("content_sha256") or "")
            != _unsigned_content_sha256(matched)
        ):
            raise ValueError(
                "matched-contrast builder returned a non-verifying artifact")

        body: dict[str, Any] = {
            "schema_version": "e2e_analysis_bundle_v1",
            "run_id": run_id,
            "phase_id": phase_id,
            "execution_binding_sha256":
                str(scope_receipt["execution_binding_sha256"]),
            "protocol_document_sha256":
                str(scope_receipt["protocol_document_sha256"]),
            "evaluation_scope_sha256":
                str(scope_receipt["evaluation_scope_sha256"]),
            "design_bindings": design_bindings,
            "config_sha256": {
                "decision": settings.shas["decision"],
                "variants": settings.shas["variants"],
                "week1": settings.shas["week1"],
            },
            "semantic_mapping": {
                "core_arm_map": arm_map,
                "core_expected_variant_ids": expected_variants,
                "matched_arm_variants": arm_variants,
            },
            "analysis_policy": {
                "n_boot": n_boot,
                "seed_namespace": seed_namespace,
                "matched_seed": matched_seed,
                "analysis_execution_binding_sha256": analysis_binding.digest,
                "analysis_approved_commit": analysis_binding.approved_commit,
                "trajectory_checkpoint_divergence":
                    "mediated_end_to_end_outcome_not_pairing_error",
            },
            "e2e_effects": e2e,
            "matched_contrasts": matched,
        }
        body["content_sha256"] = sha256_hex(canonical_json(body))
        destination = resolve_scoped_path(
            settings.path("judgments"),
            run_id=run_id,
            phase_id=phase_id,
            tail=("analysis", "E2E_ANALYSIS.json"),
        )
        write_status = _write_e2e_analysis_once(destination, body)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(f"analyze-e2e failed: {exc}", code=2)

    typer.echo(json.dumps({
        "ok": True,
        "run_id": run_id,
        "phase_id": phase_id,
        "blocks_offered": e2e["all_offered_blocks"],
        "tasks": e2e["tasks"],
        "clusters": e2e["clusters"],
        "matched_contrasts": len(matched["contrasts"]),
        "eligibility_status": e2e["eligibility"]["status"],
        "content_sha256": body["content_sha256"],
        "output": str(destination),
        "write_status": write_status,
    }, indent=2, sort_keys=True))


@app.command("finalize-decision")
def finalize_decision(
    config: Path = _CFG,
    run_id: str = typer.Option(..., "--run-id"),
    phase_id: str = typer.Option(..., "--phase-id"),
) -> None:
    """Evaluator-only: render one provenance-bound P1 decision at fixed scoped paths.

    There are intentionally no caller-selected input, output, threshold, or operational paths.
    The command reads the two write-once analyses owned by this run/phase and writes JSON and
    Markdown derived from one typed object.  It is an offline analysis command: it neither
    spends resources nor treats the currently-invalidated launch approval as permission to run.
    """
    from .analysis.finalize import build_final_decision, write_final_decision
    from .protocol import protocol_sha
    from .scoped_paths import resolve_scoped_path

    _require_role("evaluator")
    settings = _settings()
    analysis_dir = resolve_scoped_path(
        settings.path("judgments"),
        run_id=run_id,
        phase_id=phase_id,
        tail=("analysis",),
    )
    itt_path = analysis_dir / "ITT_VERDICT_INPUTS.json"
    e2e_path = analysis_dir / "E2E_ANALYSIS.json"
    json_path = analysis_dir / "WEEK1_P1_DECISION.json"
    markdown_path = analysis_dir / "WEEK1_P1_DECISION.md"
    audit_path = resolve_scoped_path(
        settings.path("judgments"),
        run_id=run_id,
        phase_id=phase_id,
        tail=("human_audit", "AUDITED_RECEIPT.json"),
    )
    # A future operational campaign has its own run/phase. Its fixed parent-binding receipt is
    # placed beside the causal analysis it extends; the finalizer verifies the independent
    # operational scope plus both parent causal hashes before using it.
    operational_path = analysis_dir / "OPERATIONAL_EVIDENCE.json"
    try:
        itt = _load_content_addressed_json(itt_path)
        e2e = _load_content_addressed_json(e2e_path)
        audit = (
            _load_content_addressed_json(audit_path)
            if audit_path.is_file()
            else None
        )
        operational = (
            _load_content_addressed_json(operational_path)
            if operational_path.is_file()
            else None
        )
        analysis_binding = _verified_analysis_binding(
            execution_binding_sha256=str(
                itt.get("execution_binding_sha256") or ""
            ),
            protocol_document_sha256=str(
                itt.get("protocol_document_sha256") or ""
            ),
        )
        policy = e2e.get("analysis_policy")
        if (
            not isinstance(policy, Mapping)
            or str(itt.get("analysis_execution_binding_sha256") or "")
            != analysis_binding.digest
            or str(itt.get("analysis_approved_commit") or "")
            != analysis_binding.approved_commit
            or str(policy.get("analysis_execution_binding_sha256") or "")
            != analysis_binding.digest
            or str(policy.get("analysis_approved_commit") or "")
            != analysis_binding.approved_commit
            or str(e2e.get("execution_binding_sha256") or "")
            != str(itt.get("execution_binding_sha256") or "")
            or str(e2e.get("protocol_document_sha256") or "")
            != analysis_binding.protocol_sha
        ):
            raise ValueError(
                "ITT/E2E analysis artifacts are not bound to the same live approved "
                "analysis implementation and frozen treatment/protocol"
            )
        if markdown_path.exists() and not json_path.exists():
            raise ValueError(
                "partial final decision exists (Markdown without JSON); generated identity "
                "cannot be recovered safely"
            )
        if json_path.exists():
            previous = _load_content_addressed_json(json_path)
            generated_at_utc = str(previous.get("generated_at_utc") or "")
        else:
            generated_at_utc = _now()
        decision = build_final_decision(
            itt,
            e2e,
            decision_config=settings.configs["decision"],
            week1_config=settings.configs["week1"],
            variants_config=settings.configs["variants"],
            operational_evidence=operational,
            human_audit_receipt=audit,
            protocol_sha=protocol_sha(_REPO),
            generated_at_utc=generated_at_utc,
        )
        write_status = write_final_decision(
            decision,
            json_path=json_path,
            markdown_path=markdown_path,
        )
    except (KeyError, OSError, TypeError, ValueError) as exc:
        _fail(f"finalize-decision failed: {exc}", code=2)
    typer.echo(json.dumps({
        "ok": True,
        "run_id": run_id,
        "phase_id": phase_id,
        "content_sha256": decision.to_json_obj()["content_sha256"],
        "verdict_status": decision.verdict_status,
        "operational_status": decision.operational_status,
        "claim_scope": decision.claim_scope,
        "json_output": str(json_path),
        "markdown_output": str(markdown_path),
        "write_status": write_status,
    }, indent=2, sort_keys=True))


@app.command("stop-safely")
def stop_safely(config: Path = _CFG) -> None:
    """Stop admission without modifying the protocol. Finished work is kept."""
    settings = _settings()
    sentinel = settings.data_root / str(settings.get("week1", "runtime", "stop_sentinel"))
    sentinel.parent.mkdir(parents=True, exist_ok=True)
    sentinel.write_text(_now() + "\n", encoding="utf-8")
    typer.echo(f"stop requested at {sentinel}")


@app.command("release-holdout")
def release_holdout(config: Path = _CFG) -> None:
    """Holdout gate: materialize the holdout corpus for the runner, once.

    Fail-closed, and this round it must fail: a formative machine-authored corpus has no
    confirmatory holdout, and the campaign never reaches POLICY_FROZEN. This is a gate that
    refuses, not a stub -- it checks the real preconditions and reports which one is unmet.
    """
    from .campaign.phases import PhaseStore
    from .experiment.ledger import Ledger
    from .experiment.state_machine import Phase

    settings = _settings()
    if not bool(settings.get("week1", "campaign", "open_holdout")):
        _fail("configs/week1.yaml sets open_holdout: false -- this corpus is FORMATIVE_ONLY and "
              "has no confirmatory holdout to release")
    binding = _require_approval()
    ledger = Ledger(str(settings.data_root / str(settings.get("week1", "paths", "runs"))
                        / "ledger.sqlite"))
    phases = PhaseStore(ledger, protocol_sha=binding.digest)
    if not phases.is_complete(Phase.POLICY_FROZEN):
        ledger.close()
        _fail("holdout release requires POLICY_FROZEN; releasing earlier would let the holdout "
              "be seen before the policy that it validates was fixed")
    ledger.close()
    _fail("holdout release is not authorized under this approval")


@app.command("serve-provider")
def serve_provider(config: Path = _CFG) -> None:  # pragma: no cover - process entry point
    """Run the provider. The only process that reads a credential, and never as root."""
    from .runtime.provider_main import run

    _require_role("provider")
    if config is not None and not config.exists():
        _fail(f"  FAIL  config: {config} not found")
    run(_settings())


def main() -> None:  # pragma: no cover
    app()


if __name__ == "__main__":  # pragma: no cover
    main()
