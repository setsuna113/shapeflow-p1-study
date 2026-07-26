"""The Week-1 decision, rendered to Markdown and JSON from one typed object.

Both outputs derive from the same :class:`DecisionObject`, so the human-readable verdict and the
machine verdict cannot disagree -- a property the plan (§20.3) requires and a test enforces. The
renderer also enforces the provisional-verdict discipline: while the run is machine-only
(``verdict_status == PROVISIONAL_MACHINE``), the quality-dependent verdicts are shown as
``PROVISIONAL_*`` and may not be read as final. ``KILL_STRUCTURAL`` is exempt because a verified
contract/lineage failure does not rest on a quality judgment. ``KILL_NO_HEADROOM`` is not
exempt: it is only available after the quality family passes.

The Markdown follows the fixed §20.3 section order and never reduces to "P1 was faster on
average": every node reports whether it helped, how much, when, over how many tasks, at what
quality cost, the worst case, and whether the conclusion has been human-audited.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..canonical import canonical_json, canonical_str
from ..hashing import sha256_hex
from .decision import Verdict

__all__ = [
    "EffectWithCI",
    "NodeDecision",
    "DecisionObject",
    "render_json",
    "render_markdown",
    "PROVISIONAL_STATUS",
    "AUDITED_STATUS",
]

PROVISIONAL_STATUS = "PROVISIONAL_MACHINE"
AUDITED_STATUS = "HUMAN_AUDIT_COMPLETE"

# Verdicts that depend on the (un-audited) quality judgments and must carry
# PROVISIONAL_ until audit.
_QUALITY_DEPENDENT = {
    Verdict.KEEP,
    Verdict.THESIS_GRADE,
    Verdict.CONDITIONAL,
    Verdict.MECHANISM_ONLY,
    Verdict.KILL_HARM,
    Verdict.KILL_NO_HEADROOM,
}


@dataclass(frozen=True)
class EffectWithCI:
    point: float
    lower: float | None
    upper: float | None
    unit: str = ""
    interval_type: str = ""
    direction: str = ""

    def obj(self) -> dict:
        return {
            "point": self.point,
            "lower": self.lower,
            "upper": self.upper,
            "unit": self.unit,
            "interval_type": self.interval_type,
            "direction": self.direction,
        }

    def fmt(self) -> str:
        suffix = f" {self.unit}" if self.unit else ""
        lower = f"{self.lower:.3g}" if self.lower is not None else "−∞"
        upper = f"{self.upper:.3g}" if self.upper is not None else "+∞"
        metadata = (
            f" ({'; '.join(item for item in (self.interval_type, self.direction) if item)})"
            if self.interval_type or self.direction
            else ""
        )
        return f"{self.point:.3g} [{lower}, {upper}]{suffix}{metadata}"


@dataclass(frozen=True)
class NodeDecision:
    node: str
    verdict: Verdict
    champion_variant: str | None
    work_saving: EffectWithCI | None
    coverage: EffectWithCI | None = None
    quality_effects: dict[str, EffectWithCI] = field(default_factory=dict)
    critical_harm_rate: float | None = None
    critical_harm_absolute: dict = field(default_factory=dict)
    where_it_works: tuple[str, ...] = ()
    where_it_fails: tuple[str, ...] = ()
    candidate_variants: tuple[str, ...] = ()
    reasons: tuple[str, ...] = ()
    limitations: tuple[str, ...] = ()
    arm_results: dict[str, dict] = field(default_factory=dict)
    claim_level: str = "UNSPECIFIED"

    def display_verdict(self, status: str) -> str:
        if status == PROVISIONAL_STATUS and self.verdict in _QUALITY_DEPENDENT:
            return f"PROVISIONAL_{self.verdict.value}"
        return self.verdict.value

    def obj(self, status: str) -> dict:
        return {
            "node": self.node,
            "verdict": self.verdict.value,
            "display_verdict": self.display_verdict(status),
            "champion_variant": self.champion_variant,
            "work_saving": self.work_saving.obj() if self.work_saving else None,
            "coverage": self.coverage.obj() if self.coverage else None,
            "quality_effects": {k: v.obj() for k, v in self.quality_effects.items()},
            "critical_harm_rate": self.critical_harm_rate,
            "critical_harm_absolute": self.critical_harm_absolute,
            "where_it_works": list(self.where_it_works),
            "where_it_fails": list(self.where_it_fails),
            "candidate_variants": list(self.candidate_variants),
            "reasons": list(self.reasons),
            "limitations": list(self.limitations),
            "arm_results": self.arm_results,
            "claim_level": self.claim_level,
        }


@dataclass(frozen=True)
class DecisionObject:
    webpage_p1: NodeDecision
    c_visible: NodeDecision
    c_registry: NodeDecision
    h_plus_c_visible: NodeDecision
    verdict_status: str
    confirmatory_power_shortfall: bool | None
    human_audit_status: str
    protocol_sha: str
    freeze_sha: str
    generated_at_utc: str
    schema_version: str = "week1_p1_decision_v3"
    run_id: str = ""
    phase_id: str = ""
    evaluation_scope_sha256: str = ""
    execution_binding_sha256: str = ""
    protocol_document_sha256: str = ""
    claim_scope: str = ""
    corpus_tier: str = ""
    operational_status: str = "UNSPECIFIED"
    eligibility_status: str = "UNSPECIFIED"
    exploratory_task_level_eligibility: dict = field(default_factory=dict)
    input_sha256: dict[str, str] = field(default_factory=dict)
    config_sha256: dict[str, str] = field(default_factory=dict)
    limitations: tuple[str, ...] = ()
    blockers: tuple[str, ...] = ()
    confirmatory_status: str = "UNSPECIFIED"
    champion_selection_rule: str = "UNSPECIFIED"
    trajectory_outcomes: dict[str, dict] = field(default_factory=dict)
    blocks_offered: int = 0
    stack_status: str = "NOT_ESTABLISHED"
    executed_design: dict[str, dict] = field(default_factory=dict)
    matched_variant_outcomes: dict = field(default_factory=dict)
    distribution_outcomes: dict = field(default_factory=dict)
    work_outcomes: dict = field(default_factory=dict)
    sensitivity_outcomes: dict = field(default_factory=dict)
    reproduction_commands: tuple[str, ...] = ()
    artifact_locations: dict[str, str] = field(default_factory=dict)

    @property
    def nodes(self) -> list[NodeDecision]:
        return [self.webpage_p1, self.c_visible, self.c_registry, self.h_plus_c_visible]

    def to_json_obj(self) -> dict:
        body = {
            "schema_version": self.schema_version,
            "WEBPAGE_P1": self.webpage_p1.obj(self.verdict_status),
            "C_VISIBLE": self.c_visible.obj(self.verdict_status),
            "C_REGISTRY": self.c_registry.obj(self.verdict_status),
            "H_PLUS_C_VISIBLE": self.h_plus_c_visible.obj(self.verdict_status),
            "verdict_status": self.verdict_status,
            "confirmatory_power_shortfall": self.confirmatory_power_shortfall,
            "human_audit_status": self.human_audit_status,
            "protocol_sha": self.protocol_sha,
            "freeze_sha": self.freeze_sha,
            "generated_at_utc": self.generated_at_utc,
            "run_id": self.run_id,
            "phase_id": self.phase_id,
            "evaluation_scope_sha256": self.evaluation_scope_sha256,
            "execution_binding_sha256": self.execution_binding_sha256,
            "protocol_document_sha256": self.protocol_document_sha256,
            "claim_scope": self.claim_scope,
            "corpus_tier": self.corpus_tier,
            "operational_status": self.operational_status,
            "eligibility_status": self.eligibility_status,
            "exploratory_task_level_eligibility": self.exploratory_task_level_eligibility,
            "input_sha256": dict(sorted(self.input_sha256.items())),
            "config_sha256": dict(sorted(self.config_sha256.items())),
            "limitations": list(self.limitations),
            "blockers": list(self.blockers),
            "confirmatory_status": self.confirmatory_status,
            "champion_selection_rule": self.champion_selection_rule,
            "trajectory_outcomes": self.trajectory_outcomes,
            "blocks_offered": self.blocks_offered,
            "stack_status": self.stack_status,
            "executed_design": self.executed_design,
            "matched_variant_outcomes": self.matched_variant_outcomes,
            "distribution_outcomes": self.distribution_outcomes,
            "work_outcomes": self.work_outcomes,
            "sensitivity_outcomes": self.sensitivity_outcomes,
            "reproduction_commands": list(self.reproduction_commands),
            "artifact_locations": dict(sorted(self.artifact_locations.items())),
        }
        body["content_sha256"] = sha256_hex(canonical_json(body))
        return body


def render_json(decision: DecisionObject) -> str:
    return canonical_str(decision.to_json_obj())


def _node_block(n: NodeDecision, status: str) -> str:
    lines = [
        f"### {n.node}: **{n.display_verdict(status)}**",
        f"- Champion variant: `{n.champion_variant or '(none advanced)'}`",
        f"- Complete work saving: {n.work_saving.fmt() if n.work_saving else 'n/a'}",
    ]
    if n.coverage:
        lines.append(f"- Eligibility coverage: {n.coverage.fmt()}")
    if n.critical_harm_rate is not None:
        adverse = n.critical_harm_absolute.get("all_offered_adverse")
        ci = adverse.get("ci") if isinstance(adverse, dict) else None
        lines.append(
            f"- Absolute all-offered adverse critical-harm rate: "
            f"{n.critical_harm_rate:.3g} "
            f"[{ci.get('lower') if isinstance(ci, dict) else 'n/a'}, "
            f"{ci.get('upper') if isinstance(ci, dict) else 'n/a'}], "
            f"pairs={adverse.get('n_pairs') if isinstance(adverse, dict) else 'n/a'}, "
            f"tasks={adverse.get('n_tasks') if isinstance(adverse, dict) else 'n/a'}"
        )
        lines.append(
            "    - This is an absolute treatment rate from the ITT artifact, not reconstructed "
            "from a treatment-minus-P0 effect."
        )
    elif n.critical_harm_absolute:
        lines.append("- Absolute all-offered critical-harm rate: `NOT_ESTABLISHED`.")
    if n.quality_effects:
        lines.append("- Quality effects (NI):")
        for name, eff in n.quality_effects.items():
            lines.append(f"    - {name}: {eff.fmt()}")
    if n.where_it_works:
        lines.append("- Where it works: " + "; ".join(n.where_it_works))
    if n.where_it_fails:
        lines.append("- Where it fails: " + "; ".join(n.where_it_fails))
    if n.candidate_variants:
        lines.append(
            "- Executed non-control candidates: "
            + ", ".join(f"`{variant}`" for variant in n.candidate_variants)
        )
    if n.claim_level:
        lines.append(f"- Claim level: `{n.claim_level}`")
    for arm_id, arm in sorted(n.arm_results.items()):
        structured = arm.get("structured_increment")
        if isinstance(structured, dict):
            lines.append(
                f"- `{arm_id}` bounded-policy increment over CPU/prose controls: "
                f"`{structured.get('status', 'NOT_ESTABLISHED')}`"
            )
            lines.append(
                "    - Attribution scope: "
                f"`{structured.get('attribution_scope', 'NOT_ESTABLISHED')}`"
            )
            lines.append(
                "    - Pointer-only attribution: "
                f"`{structured.get('pointer_only_attribution_status', 'NOT_ESTABLISHED')}`"
            )
            components = structured.get("component_gates")
            if isinstance(components, dict):
                for control_type in ("LLM_vs_CPU", "structured_selection_vs_prose"):
                    component = components.get(control_type)
                    component_status = (
                        component.get("status", "NOT_ESTABLISHED")
                        if isinstance(component, dict)
                        else "NOT_ESTABLISHED"
                    )
                    lines.append(f"    - {control_type}: " f"`{component_status}`")
            if structured.get("reason"):
                lines.append(f"    - {structured['reason']}")
        efficiency = arm.get("selector_efficiency")
        if isinstance(efficiency, dict):
            for node, summary in sorted(efficiency.items()):
                if not isinstance(summary, dict):
                    continue
                lines.append(
                    f"- `{arm_id}` / {node} selector efficiency: "
                    f"`{summary.get('decision_status', 'NOT_ESTABLISHED')}` "
                    "(secondary/descriptive)"
                )
                metrics = summary.get("metrics")
                if summary.get("decision_status") == "ESTIMABLE_DESCRIPTIVE" and isinstance(
                    metrics, dict
                ):
                    for metric, value in sorted(metrics.items()):
                        ci = value.get("ci") if isinstance(value, dict) else None
                        lines.append(
                            f"    - {metric}: point="
                            f"{ci.get('point') if isinstance(ci, dict) else 'NOT_ESTABLISHED'}, "
                            f"CI=[{ci.get('lower') if isinstance(ci, dict) else 'n/a'}, "
                            f"{ci.get('upper') if isinstance(ci, dict) else 'n/a'}]"
                        )
                    published = summary.get("published_rendered_tokens")
                    offered = summary.get("offered_evidence_tokens")
                    published_total = (
                        published.get("total") if isinstance(published, dict) else "NOT_ESTABLISHED"
                    )
                    offered_total = (
                        offered.get("total") if isinstance(offered, dict) else "NOT_ESTABLISHED"
                    )
                    lines.append(
                        "    - exact token totals: published="
                        f"{published_total}, offered={offered_total}"
                    )
                elif summary.get("reason"):
                    lines.append(f"    - {summary['reason']}")
    if n.reasons:
        lines.append("- Decision reasons:")
        lines.extend(f"    - {reason}" for reason in n.reasons)
    if n.limitations:
        lines.append("- Node limitations:")
        lines.extend(f"    - {limitation}" for limitation in n.limitations)
    return "\n".join(lines)


def _trajectory_block(decision: DecisionObject) -> str:
    lines = [
        (
            "These are descriptive mediated end-to-end outcomes. Post-treatment query, "
            "reasoning, and checkpoint differences are measured here; trajectory equality is "
            "never a pairing or acceptance condition."
        )
    ]
    if not decision.trajectory_outcomes:
        lines.append("- Trajectory contrasts unavailable.")
        return "\n".join(lines)
    for endpoint, result in sorted(decision.trajectory_outcomes.items()):
        status = str(result.get("status") or "UNKNOWN")
        lines.append(f"- `{endpoint}`: `{status}`")
        for contrast in ("h_simple", "c_simple", "joint", "interaction"):
            ci = result.get(contrast)
            if not isinstance(ci, dict):
                continue
            point = ci.get("point")
            lower = ci.get("lower")
            upper = ci.get("upper")
            lines.append(f"    - {contrast}: point={point!r}, lower={lower!r}, upper={upper!r}")
    return "\n".join(lines)


def _frozen_scope_block(decision: DecisionObject) -> str:
    return "\n".join(
        (
            f"- run_id / phase_id: `{decision.run_id}` / `{decision.phase_id}`",
            f"- blocks offered: `{decision.blocks_offered}`",
            f"- evaluation scope: `{decision.evaluation_scope_sha256}`",
            f"- execution binding: `{decision.execution_binding_sha256}`",
            f"- protocol document: `{decision.protocol_document_sha256}`",
            f"- frozen campaign root: `{decision.freeze_sha}`",
            f"- protocol: `{decision.protocol_sha}`",
            f"- stack status: `{decision.stack_status}`",
            f"- claim scope / corpus tier: `{decision.claim_scope}` / `{decision.corpus_tier}`",
        )
    )


def _actually_run_block(decision: DecisionObject) -> str:
    if not decision.executed_design:
        return "- `NOT_ESTABLISHED`: no executed-design index is bound."
    lines = []
    for arm_id, item in sorted(decision.executed_design.items()):
        lines.append(
            f"- `{arm_id}`: page=`{item.get('page_variant', '')}`, "
            f"close=`{item.get('close_variant', '')}`, "
            f"control=`{item.get('is_control')}`, "
            f"role=`{item.get('analysis_role', 'NOT_ESTABLISHED')}`, "
            f"measurement=`{item.get('measurement_status', 'NOT_ESTABLISHED')}`"
        )
    return "\n".join(lines)


def _variant_comparison_block(decision: DecisionObject) -> str:
    contrasts = decision.matched_variant_outcomes.get("contrasts")
    if not isinstance(contrasts, list) or not contrasts:
        return (
            "- `NOT_ESTABLISHED`: no bound matched one-factor contrast artifact is available. "
            "No structured-mechanism attribution is available."
        )
    lines = [
        f"- Primary-arm rule: `{decision.champion_selection_rule}`",
        "- Every ordinary matched contrast and every non-core H/C arm is "
        "`SECONDARY_EXPLORATORY_NO_MULTIPLICITY_ADJUSTED_CLAIM`; it cannot select, replace, "
        "or upgrade a primary verdict.",
        f"- Matched-artifact decision use: "
        f"`{decision.matched_variant_outcomes.get('decision_use', 'NOT_ESTABLISHED')}`",
    ]
    for item in contrasts:
        if not isinstance(item, dict):
            continue
        pairing_status = str(item.get("pairing_status") or "NOT_ESTABLISHED")
        endpoint_counts = item.get("endpoint_status_counts")
        endpoint_summary = endpoint_counts if isinstance(endpoint_counts, dict) else {}
        lines.append(
            f"- `{item.get('contrast_id', 'unnamed')}`: "
            f"pairing_status=`{pairing_status}`, "
            f"endpoint_status_counts=`{endpoint_summary}`"
        )
    return "\n".join(lines)


def _distribution_block(decision: DecisionObject) -> str:
    lines = [
        "- The primary complete-work effect is the paired task-mean log-ratio saving shown in "
        "Sections 5–8; its confidence bounds are not relabeled as a raw arithmetic mean.",
    ]
    work = decision.work_outcomes.get("service_work_seconds")
    work_distribution = work.get("task_effect_distribution") if isinstance(work, dict) else None
    if isinstance(work_distribution, dict):
        lines.append(
            "- Descriptive task-level complete-work saving distribution "
            "(cluster bootstrap above remains primary inference):"
        )
        for name in ("h_simple", "c_simple", "joint"):
            value = work_distribution.get(name)
            if isinstance(value, dict):
                lines.append(
                    f"    - {name}: median={value.get('median')!r}, "
                    f"bad lower tail p05={value.get('p05')!r}, "
                    f"n_tasks={value.get('n_tasks')!r}, scale={value.get('scale')!r}"
                )
    else:
        lines.append("- Task-level work-saving median/bad-tail distribution: `NOT_ESTABLISHED`.")

    latency = decision.distribution_outcomes.get("e2e_latency_seconds")
    if isinstance(latency, dict):
        lines.append(f"- E2E latency factorial status: `{latency.get('status', 'UNKNOWN')}`")
        for name in ("h_simple", "c_simple", "joint", "interaction"):
            value = latency.get(name)
            if isinstance(value, dict):
                lines.append(
                    f"    - {name}: point={value.get('point')!r}, "
                    f"lower={value.get('lower')!r}, upper={value.get('upper')!r}"
                )
        latency_distribution = latency.get("task_effect_distribution")
        if isinstance(latency_distribution, dict):
            lines.append(
                "- Descriptive task-level latency-difference distribution "
                "(positive is slower; cluster bootstrap remains primary inference):"
            )
            for name in ("h_simple", "c_simple", "joint"):
                value = latency_distribution.get(name)
                if isinstance(value, dict):
                    lines.append(
                        f"    - {name}: median={value.get('median')!r}s, "
                        f"bad upper tail p95={value.get('p95')!r}s, "
                        f"n_tasks={value.get('n_tasks')!r}"
                    )
        else:
            lines.append("- Task-level latency median/bad-tail distribution: `NOT_ESTABLISHED`.")
    else:
        lines.append("- Average E2E latency effect: `NOT_ESTABLISHED`.")
        lines.append("- Task-level latency median/bad-tail distribution: `NOT_ESTABLISHED`.")
    joint_outcomes = decision.distribution_outcomes.get("task_level_joint_outcomes")
    if isinstance(joint_outcomes, dict) and joint_outcomes.get("status") == "OK":
        lines.append(
            "- Joint task-level work/quality outcomes for the exact primary H/C/H+C arms "
            "(descriptive proportions with source-topic cluster CIs):"
        )
        for arm_name in ("h_simple", "c_simple", "joint"):
            arm = (joint_outcomes.get("arms") or {}).get(arm_name)
            if not isinstance(arm, dict):
                lines.append(f"    - {arm_name}: `NOT_ESTABLISHED`")
                continue
            lines.append(f"    - {arm_name}:")
            for threshold, value in sorted((arm.get("saving_threshold_proportions") or {}).items()):
                lines.append(
                    f"        - {threshold}: {value.get('point')!r} "
                    f"[{value.get('lower')!r}, {value.get('upper')!r}]"
                )
            for label in (
                "quality_qualified_pareto_win_proportion",
                "slower_and_quality_harmed_proportion",
            ):
                value = arm.get(label)
                lines.append(
                    f"        - {label}: "
                    f"{value.get('point') if isinstance(value, dict) else 'NOT_ESTABLISHED'} "
                    f"[{value.get('lower') if isinstance(value, dict) else 'n/a'}, "
                    f"{value.get('upper') if isinstance(value, dict) else 'n/a'}]"
                )
    else:
        reason = (
            joint_outcomes.get("reason", "TASK_LEVEL_JOINT_OUTCOMES_MISSING")
            if isinstance(joint_outcomes, dict)
            else "TASK_LEVEL_JOINT_OUTCOMES_MISSING"
        )
        lines.append(
            "- Joint task-level saving-threshold/Pareto/adverse proportions: "
            f"`NOT_ESTABLISHED` ({reason})."
        )
    lines.append(
        "- Raw operational p95 latency *ratio* across live concurrent blocks: "
        "`NOT_ESTABLISHED` in the causal task distribution; it requires the separately bound "
        "operational artifact and is not imputed from task-level latency differences."
    )
    lines.append(
        "- Selector token efficiency is secondary/descriptive (not the primary causal utility):"
    )
    for node in decision.nodes:
        for arm_id, arm in sorted(node.arm_results.items()):
            family = arm.get("selector_efficiency")
            if not isinstance(family, dict):
                continue
            for selector_node, summary in sorted(family.items()):
                if not isinstance(summary, dict):
                    continue
                status = summary.get("decision_status", "NOT_ESTABLISHED")
                lines.append(f"    - {arm_id}/{selector_node}: `{status}`")
                if status != "ESTIMABLE_DESCRIPTIVE":
                    lines.append(
                        "      token_trace_complete=false or estimand incomplete; "
                        "selector-efficiency conclusion is NOT_ESTABLISHED."
                    )
                    continue
                for metric in (
                    "selected_token_precision",
                    "weighted_truth_per_100_rendered_tokens",
                    "materialization_ratio",
                ):
                    raw = (summary.get("metrics") or {}).get(metric)
                    ci = raw.get("ci") if isinstance(raw, dict) else None
                    lines.append(
                        f"      {metric}="
                        f"{ci.get('point') if isinstance(ci, dict) else 'NOT_ESTABLISHED'}"
                    )
    return "\n".join(lines)


def _quality_block(decision: DecisionObject) -> str:
    lines = []
    for node in decision.nodes:
        if not node.quality_effects:
            lines.append(f"- `{node.node}`: strict quality/harm family `NOT_ESTABLISHED`.")
            continue
        lines.append(f"- `{node.node}` strict all-offered one-sided guards:")
        lines.extend(
            f"    - {name}: {effect.fmt()}" for name, effect in sorted(node.quality_effects.items())
        )
    return "\n".join(lines)


def _work_balance_block(decision: DecisionObject) -> str:
    lines = [
        f"- Operational evidence status: `{decision.operational_status}`.",
    ]
    if decision.operational_status == "UNAVAILABLE_PRODUCER_NOT_IMPLEMENTED":
        lines.append(
            "- The repository has no verified live operational-evidence producer. Full "
            "concurrent batching/APC makespan, throughput, energy, and tail-cost questions "
            "remain unanswered; causal results cannot exceed MECHANISM_ONLY."
        )
    for node in decision.nodes:
        lines.append(
            f"- `{node.node}` complete service-work saving: "
            f"{node.work_saving.fmt() if node.work_saving else 'NOT_ESTABLISHED'}"
        )
    for endpoint in (
        "service_work_seconds",
        "prompt_tokens",
        "completion_tokens",
        "cached_prompt_tokens",
    ):
        raw = decision.work_outcomes.get(endpoint)
        lines.append(
            f"- Core 2x2 `{endpoint}`: "
            f"`{raw.get('status', 'UNKNOWN') if isinstance(raw, dict) else 'NOT_ESTABLISHED'}`"
        )
    lines.append(
        "- API spend, GPU time, retries, failed and fallback work stay included in the frozen "
        "service-work/accounting artifacts; unavailable balances are not imputed."
    )
    return "\n".join(lines)


def _eligibility_block(decision: DecisionObject) -> str:
    lines = [
        f"- Overall status: `{decision.eligibility_status}`.",
        "- Eligibility is an exploratory, formative, whole-task analysis using only frozen "
        "pre-treatment features. It is not an invocation-level or same-checkpoint policy; "
        "task coverage is not invocation coverage.",
        "- These findings never change a primary verdict, champion, structured-mechanism "
        "attribution, or deployment envelope. They are hypotheses for a later confirmatory "
        "study, including when an untouched formative holdout was evaluated.",
        "- Every frozen target is reported; null, unstable, and not-estimable targets are not "
        "discarded in favor of a best-looking rule.",
    ]
    snapshot = decision.exploratory_task_level_eligibility
    if not isinstance(snapshot, dict) or not snapshot:
        lines.append(
            "- Frozen target findings: `NOT_ESTABLISHED`; no valid v3 exploratory family was "
            "bound to this decision."
        )
        return "\n".join(lines)
    lines.extend(
        (
            f"- Bound eligibility schema: `{snapshot.get('schema_version', 'UNKNOWN')}`.",
            f"- Analysis scope: `{snapshot.get('analysis_scope', 'UNKNOWN')}`.",
        )
    )
    target_results = snapshot.get("target_results")
    if not isinstance(target_results, list) or not target_results:
        lines.append("- Frozen target findings: `NOT_ESTABLISHED`.")
        return "\n".join(lines)
    lines.append("- Frozen target findings (complete canonical snapshots):")
    for target in target_results:
        target_id = (
            str(target.get("target_id") or "UNNAMED_TARGET")
            if isinstance(target, dict)
            else "MALFORMED_TARGET"
        )
        status = (
            str(target.get("status") or target.get("finding_status") or "UNKNOWN")
            if isinstance(target, dict)
            else "MALFORMED"
        )
        payload = canonical_str(target)
        lines.append(f"    - `{target_id}`: `{status}` — `{payload}`")
    return "\n".join(lines)


def _reliability_block(decision: DecisionObject) -> str:
    lines = []
    for node in decision.nodes:
        if not node.arm_results:
            lines.append(f"- `{node.node}`: fallback/failure effects `NOT_ESTABLISHED`.")
            continue
        for arm_id, arm in sorted(node.arm_results.items()):
            quality = arm.get("quality_effects") or {}
            terminal = quality.get("terminal_failure_risk_difference")
            fallback = quality.get("fallback_risk_difference")
            lines.append(
                f"- `{arm_id}`: terminal UCB="
                f"{terminal.get('upper') if isinstance(terminal, dict) else 'NOT_ESTABLISHED'}, "
                f"fallback UCB="
                f"{fallback.get('upper') if isinstance(fallback, dict) else 'NOT_ESTABLISHED'}"
            )
    return "\n".join(lines)


def _sensitivity_block(decision: DecisionObject) -> str:
    views = decision.sensitivity_outcomes
    lines = [
        "- Strict/no-repair, fallback-assisted, worst-case and best-case results remain "
        "separate; no sensitivity view is substituted for the strict decision family.",
        f"- Live operational sensitivity: `{decision.operational_status}`.",
    ]
    arm_sensitivity_seen = False
    for node in decision.nodes:
        for arm_id, arm in sorted(node.arm_results.items()):
            validity = arm.get("selector_output_validity")
            no_repair = arm.get("no_repair_quality")
            if isinstance(validity, dict):
                arm_sensitivity_seen = True
                hard = validity.get("invalid_id_hard_gate")
                strict = validity.get("strict_valid_rate_lcb")
                repair = validity.get("repair_rate_ucb")
                hard_status = (
                    hard.get("status", "NOT_ESTABLISHED")
                    if isinstance(hard, dict)
                    else "NOT_ESTABLISHED"
                )
                strict_status = (
                    strict.get("status", "NOT_ESTABLISHED")
                    if isinstance(strict, dict)
                    else "NOT_ESTABLISHED"
                )
                repair_status = (
                    repair.get("status", "NOT_ESTABLISHED")
                    if isinstance(repair, dict)
                    else "NOT_ESTABLISHED"
                )
                lines.append(
                    f"- `{arm_id}` normalization: invalid-ID gate="
                    f"`{hard_status}`, strict-valid status=`{strict_status}`, "
                    f"repair status=`{repair_status}`."
                )
            if isinstance(no_repair, dict) and no_repair:
                arm_sensitivity_seen = True
                primary = no_repair.get("weighted_required_atom_recall")
                primary_status = (
                    primary.get("status", "NOT_ESTABLISHED")
                    if isinstance(primary, dict)
                    else "NOT_ESTABLISHED"
                )
                lines.append(
                    f"    - no-repair adverse weighted-required-atom: " f"`{primary_status}`"
                )
                if isinstance(primary, dict) and isinstance(primary.get("effect"), dict):
                    effect = primary["effect"]
                    lines.append(
                        f"      point={effect.get('point')!r}, lower={effect.get('lower')!r}, "
                        f"upper={effect.get('upper')!r}, dirty_pairs="
                        f"{primary.get('dirty_pairs')!r}"
                    )
    if not arm_sensitivity_seen:
        lines.append("- Per-arm normalization/no-repair sensitivity: `NOT_ESTABLISHED`.")
    if not views:
        lines.append("- Bound quality sensitivity views: `NOT_ESTABLISHED`.")
    for view, metrics in sorted(views.items()):
        primary = (
            metrics.get("weighted_required_atom_recall") if isinstance(metrics, dict) else None
        )
        primary_status = (
            primary.get("status", "RECORDED") if isinstance(primary, dict) else "NOT_ESTABLISHED"
        )
        lines.append(f"- `{view}` weighted-required-atom sensitivity: " f"`{primary_status}`")
        if isinstance(primary, dict):
            for contrast in ("h_simple", "c_simple", "joint", "interaction"):
                ci = primary.get(contrast)
                if isinstance(ci, dict):
                    lines.append(
                        f"    - {contrast}: point={ci.get('point')!r}, "
                        f"lower={ci.get('lower')!r}, upper={ci.get('upper')!r}"
                    )
    lines.append(_trajectory_block(decision))
    return "\n".join(lines)


def _reproduction_block(decision: DecisionObject) -> str:
    if not decision.reproduction_commands:
        return "- `NOT_ESTABLISHED`: no fixed reproduction command sequence is bound."
    return "\n".join(f"- `{command}`" for command in decision.reproduction_commands)


def _hash_block(decision: DecisionObject) -> str:
    lines = [
        f"- decision JSON content SHA-256: `{decision.to_json_obj()['content_sha256']}`",
    ]
    lines.extend(
        f"- input `{name}`: `{digest}`" for name, digest in sorted(decision.input_sha256.items())
    )
    lines.extend(
        f"- config `{name}`: `{digest}`" for name, digest in sorted(decision.config_sha256.items())
    )
    lines.extend(
        f"- artifact `{name}`: `{location}`"
        for name, location in sorted(decision.artifact_locations.items())
    )
    return "\n".join(lines)


def render_markdown(decision: DecisionObject) -> str:
    status = decision.verdict_status
    power_shortfall = (
        decision.confirmatory_power_shortfall
        if decision.confirmatory_power_shortfall is not None
        else "NOT_ASSESSED_FORMATIVE_ONLY"
    )
    provisional_banner = ""
    if status == PROVISIONAL_STATUS:
        provisional_banner = (
            "> **PROVISIONAL — machine verdicts only.** Quality-dependent verdicts carry a "
            "`PROVISIONAL_` prefix and are NOT final until the human audit gate is discharged.\n\n"
        )

    sections = [
        f"# Week-1 P1 Decision\n\n{provisional_banner}"
        f"- verdict_status: `{status}`\n"
        f"- human_audit_status: `{decision.human_audit_status}`\n"
        f"- confirmatory_power_shortfall: `{power_shortfall}`\n"
        f"- confirmatory_status: `{decision.confirmatory_status}`\n"
        f"- claim_scope: `{decision.claim_scope or 'UNSPECIFIED'}`\n"
        f"- corpus_tier: `{decision.corpus_tier or 'UNSPECIFIED'}`\n"
        f"- operational_status: `{decision.operational_status}`\n"
        f"- eligibility_status: `{decision.eligibility_status}`\n"
        f"- champion_selection_rule: `{decision.champion_selection_rule}`\n"
        f"- protocol_sha: `{decision.protocol_sha}`  freeze_sha: `{decision.freeze_sha}`\n"
        f"- generated_at_utc: `{decision.generated_at_utc}`\n"
        f"- run/phase: `{decision.run_id or 'UNSPECIFIED'}` / "
        f"`{decision.phase_id or 'UNSPECIFIED'}`\n"
        f"- evaluation_scope_sha256: `{decision.evaluation_scope_sha256}`\n"
        f"- execution_binding_sha256: `{decision.execution_binding_sha256}`\n"
        f"- protocol_document_sha256: `{decision.protocol_document_sha256}`",
        "## 1. Executive verdict\n"
        + "\n".join(
            f"- **{n.node}**: {n.display_verdict(status)}"
            f"{' — champion `' + n.champion_variant + '`' if n.champion_variant else ''}"
            for n in decision.nodes
        ),
        "## 2. Frozen scope and stack\n" + _frozen_scope_block(decision),
        "## 3. What actually ran\n" + _actually_run_block(decision),
        "## 4. Variant and control comparison\n" + _variant_comparison_block(decision),
        "## 5. WEBPAGE-P1 verdict\n" + _node_block(decision.webpage_p1, status),
        "## 6. C_VISIBLE verdict\n" + _node_block(decision.c_visible, status),
        "## 7. C_REGISTRY extension verdict\n" + _node_block(decision.c_registry, status),
        "## 8. H×C interaction\n" + _node_block(decision.h_plus_c_visible, status),
        "## 9. Average, median, and tail behavior\n" + _distribution_block(decision),
        "## 10. Quality and harm\n" + _quality_block(decision),
        "## 11. Complete work and cost balance\n" + _work_balance_block(decision),
        "## 12. Eligibility, when useful, and coverage\n" + _eligibility_block(decision),
        "## 13. Fallback, restart, and terminal failure\n" + _reliability_block(decision),
        "## 14. Sensitivity and mediated trajectory outcomes\n" + _sensitivity_block(decision),
        "## 15. Human-audit status\n"
        + (
            f"`{decision.human_audit_status}`. "
            + (
                "Verdicts are provisional until this completes."
                if status == PROVISIONAL_STATUS
                else "Audit complete; verdicts are final."
            )
        ),
        "## 16. Decision for the ShapeFlow proposal\n" + _proposal_line(decision),
        "## 17. Bound inputs and limitations\n"
        + (
            "- Input hashes:\n"
            + "\n".join(
                f"    - {name}: `{digest}`"
                for name, digest in sorted(decision.input_sha256.items())
            )
            if decision.input_sha256
            else "- Input hashes: n/a"
        )
        + (
            "\n- Global blockers:\n" + "\n".join(f"    - {item}" for item in decision.blockers)
            if decision.blockers
            else ""
        )
        + (
            "\n- Global limitations:\n"
            + "\n".join(f"    - {item}" for item in decision.limitations)
            if decision.limitations
            else ""
        ),
        "## 18. Reproduction commands\n" + _reproduction_block(decision),
        "## 19. Artifact and configuration hashes\n" + _hash_block(decision),
    ]
    return "\n\n".join(sections) + "\n"


def _proposal_line(decision: DecisionObject) -> str:
    if decision.verdict_status == PROVISIONAL_STATUS:
        candidates = [
            n.node
            for n in decision.nodes
            if n.verdict in {Verdict.KEEP, Verdict.THESIS_GRADE, Verdict.CONDITIONAL}
        ]
        return (
            "NO FINAL GO / proposal NO-GO: all quality-dependent verdicts are "
            "machine-provisional until the frozen human audit completes. "
            + (
                "Machine-provisional positive candidates: "
                + ", ".join(candidates)
                + "."
                if candidates
                else "No machine-provisional positive candidate was observed."
            )
            + " NOT_ESTABLISHED is not a demonstrated absence of effect."
        )
    # A NOT_ESTABLISHED or MECHANISM_ONLY node is a proposal NO-GO; report it as such without
    # claiming a proven absence of effect.
    go_blockers = [
        n.node
        for n in decision.nodes
        if n.verdict
        in {
            Verdict.NOT_ESTABLISHED,
            Verdict.MECHANISM_ONLY,
            Verdict.KILL_STRUCTURAL,
            Verdict.KILL_HARM,
            Verdict.KILL_NO_HEADROOM,
        }
    ]
    if any(
        n.verdict in {Verdict.KEEP, Verdict.THESIS_GRADE, Verdict.CONDITIONAL}
        for n in decision.nodes
    ):
        return "At least one node reached KEEP/CONDITIONAL; see per-node envelopes for scope."
    return (
        "No node reached KEEP/CONDITIONAL — proposal NO-GO for now. "
        f"Nodes not established / mechanism-only / killed: {', '.join(go_blockers)}. "
        "NOT_ESTABLISHED is a NO-GO but is not a demonstrated absence of effect."
    )
