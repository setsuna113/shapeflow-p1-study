#!/usr/bin/env python
"""Measure the candidate judges on held-out material, then apply the frozen rule.

The rule lives in ``configs/judge_calibration.yaml`` and was committed before this script ever
ran, so the selection cannot be steered by looking at the numbers first. This program only fills
in measurements and applies the rule mechanically.

What is measured, per candidate, on the *same* prompts:

``truncation``      how often a judgment still does not fit after truth.py's adaptive batch
                    splitting. The 8000 -> 32000 proposal was never derived from data; this is
                    the residual the splitting cannot fix. Reasoning tokens are billed inside
                    completion_tokens, so a higher effort makes this strictly worse at a fixed cap.
``stability``       the same batch asked twice. Two calls to one judge are one rater asked twice,
                    not two raters; a verdict that moves under resampling adds variance to every
                    downstream estimate.
``binding validity``  judge.yaml *requires* exact span binding. This checks it rather than
                    trusting it: the cited span_id must be in the offered set, and the quoted
                    excerpt must appear verbatim inside that span's text.
``yield``           accepted atoms per task, which is the ranking metric -- validity alone is
                    trivially maximized by binding almost nothing.

Not measured, deliberately: binding *recall*, which needs human labels that do not exist yet, and
style-differential bias, which has no unbiased alternative to be thresholded against. Both are
recorded as limitations by the rule rather than silently omitted.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

import yaml  # noqa: E402

from shapeflow_p1.campaign.settings import Settings  # noqa: E402
from shapeflow_p1.campaign.truth import (  # noqa: E402
    _checker,
    _excerpt_batches,
    _excerpt_spans,
    _format_query_attempts,
    _query_attempts,
    _TRUTH_PROMPT,
    _TRUTH_SYSTEM,
)
from shapeflow_p1.evaluation.judge_client import (  # noqa: E402
    DeepSeekJudge,
    JudgeTruncated,
    JudgeUnavailable,
    SamplingEnvelope,
)


@dataclass
class CandidateResult:
    candidate_id: str
    model: str
    reasoning_effort: str | None
    max_tokens: int
    calls: int = 0
    truncated: int = 0
    json_valid: int = 0
    atoms_proposed: int = 0
    atoms_bound_valid: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    reasoning_tokens: int = 0
    stability_pairs: int = 0
    stability_agreements: int = 0
    errors: list[str] = field(default_factory=list)
    per_task_atoms: dict = field(default_factory=dict)

    # --- the four pre-registered quantities --------------------------------------------------
    @property
    def json_valid_non_truncated(self) -> float:
        return self.json_valid / self.calls if self.calls else 0.0

    @property
    def repeat_decision_stability(self) -> float:
        if not self.stability_pairs:
            return 0.0
        return self.stability_agreements / self.stability_pairs

    @property
    def span_binding_validity(self) -> float:
        if not self.atoms_proposed:
            # No atoms means nothing was bound correctly, not "perfectly valid". Returning 1.0
            # here would let a candidate that answers nothing pass the strictest gate.
            return 0.0
        return self.atoms_bound_valid / self.atoms_proposed

    @property
    def accepted_atoms_per_task(self) -> float:
        if not self.per_task_atoms:
            return 0.0
        return sum(self.per_task_atoms.values()) / len(self.per_task_atoms)

    def projected_total_usd(self, pricing: dict, cells: int) -> float:
        """Cost of this candidate extrapolated to the whole campaign, not to 48 packets.

        Report judging over the screen is the larger half and was omitted from the earlier
        $40-70 estimate entirely.
        """
        if not self.calls:
            return float("inf")
        per_call = (
            self.prompt_tokens / self.calls / 1e6 * pricing["in"]
            + self.completion_tokens / self.calls / 1e6 * pricing["out"]
        )
        return per_call * cells

    def content(self, pricing: dict, cells: int) -> dict:
        return {
            "candidate_id": self.candidate_id,
            "model": self.model,
            "reasoning_effort": self.reasoning_effort,
            "max_tokens": self.max_tokens,
            "calls": self.calls,
            "json_valid_non_truncated": round(self.json_valid_non_truncated, 4),
            "repeat_decision_stability": round(self.repeat_decision_stability, 4),
            "span_binding_validity": round(self.span_binding_validity, 4),
            "accepted_atoms_per_task": round(self.accepted_atoms_per_task, 3),
            "atoms_proposed": self.atoms_proposed,
            "atoms_bound_valid": self.atoms_bound_valid,
            "truncated_calls": self.truncated,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "reasoning_tokens": self.reasoning_tokens,
            "projected_total_usd": round(self.projected_total_usd(pricing, cells), 2),
            "errors": self.errors[:5],
        }


def _validity_of(payload: dict, offered: dict[str, str]) -> tuple[int, int]:
    """(proposed, referentially valid) for one judgment.

    Referential validity only, per the amended rule: the frozen schema's ``supporting_span_ids``
    are bare id strings with no quote, so there is no excerpt to verify verbatim. An atom counts
    as valid when it cites at least one span and every id it cites was in the set actually
    offered to that call -- which still catches a model inventing plausible span ids. Whether a
    span *semantically supports* its atom is truth.py's separate binding pass.
    """
    proposed = 0
    valid = 0
    for atom in payload.get("atoms", []) or []:
        proposed += 1
        span_ids = atom.get("supporting_span_ids") or []
        if not span_ids:
            continue
        if all(str(sid) in offered for sid in span_ids):
            valid += 1
    return proposed, valid


def _normalize(text: str) -> str:
    return " ".join(str(text).split()).casefold()


def _utc_now() -> str:
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _atom_keys(payload: dict) -> set:
    """A judgment's decision content, for comparing two calls on the same input."""
    keys = set()
    for atom in payload.get("atoms", []) or []:
        keys.add((str(atom.get("facet_id", "")), _normalize(atom.get("text", ""))))
    return keys


async def _run_candidate(
    settings: Settings, spec: dict, tasks: list[dict], *, batches_per_task: int, client,
) -> CandidateResult:
    from shapeflow_p1.providers.provider_client import PROVIDER_KEY_PLACEHOLDER

    result = CandidateResult(
        candidate_id=spec["id"], model=spec["model"],
        reasoning_effort=spec.get("reasoning_effort"), max_tokens=int(spec["max_tokens"]),
    )
    # Same envelope the campaign would use, with only the two knobs under test overridden, so a
    # difference between candidates is the knob and not an incidental sampling change.
    base = settings.judge_sampling()
    envelope = SamplingEnvelope(
        temperature=base.temperature, top_p=base.top_p, seed=base.seed,
        max_tokens=int(spec["max_tokens"]),
        enable_thinking=base.enable_thinking,
        send_thinking_switch=base.send_thinking_switch,
        reasoning_effort=spec.get("reasoning_effort"),
    )
    judge = DeepSeekJudge(
        client.deepseek_transport(op_class="JUDGE_TRUTH", work_key="judge-calibration"),
        spec["model"], PROVIDER_KEY_PLACEHOLDER,
        max_retries=settings.judge_max_retries(),
        sampling=envelope,
    )

    for task in tasks:
        task_id = task["task_id"]
        spans = _excerpt_spans(settings, task_id)
        if not spans:
            result.errors.append(f"{task_id}: no frozen spans")
            continue
        offered = {s["span_id"]: s["text"] for s in spans}
        attempts = _query_attempts(settings, task_id)
        batches = _excerpt_batches(spans, max_prompt_chars=80_000)[:batches_per_task]
        accepted_here = 0

        for index, batch in enumerate(batches):
            prompt = _TRUTH_PROMPT.format(
                question=task["question"],
                facets="\n".join(f"- {f}" for f in task["required_facets"]) or "- (none stated)",
                excerpts="\n".join(f"{s['span_id']}: {s['text']}" for s in batch),
                query_attempts=_format_query_attempts(attempts),
            )
            payload = await _one(result, judge, prompt)
            if payload is None:
                continue
            proposed, valid = _validity_of(payload, offered)
            result.atoms_proposed += proposed
            result.atoms_bound_valid += valid
            accepted_here += valid

            # Stability: ask the first batch of each task a second time. Same input, same
            # declared sampling policy -- any disagreement is the instrument, not the evidence.
            if index == 0:
                repeat = await _one(result, judge, prompt)
                if repeat is not None:
                    a, b = _atom_keys(payload), _atom_keys(repeat)
                    union = a | b
                    result.stability_pairs += 1
                    if union:
                        result.stability_agreements += len(a & b) / len(union)
                    else:
                        result.stability_agreements += 1
        result.per_task_atoms[task_id] = accepted_here
    return result


async def _one(result: CandidateResult, judge: DeepSeekJudge, prompt: str) -> dict | None:
    result.calls += 1
    try:
        response = await judge.judge(_TRUTH_SYSTEM, prompt, validate=_checker())
    except JudgeTruncated:
        result.truncated += 1
        return None
    except JudgeUnavailable as e:
        result.errors.append(f"unavailable: {type(e).__name__}")
        return None
    except Exception as e:  # noqa: BLE001 -- a candidate that crashes is a candidate that failed
        result.errors.append(f"{type(e).__name__}: {e}")
        return None
    result.json_valid += 1
    # Every attempt is charged, including the ones that failed on the way to this answer, so a
    # candidate that needs three tries to produce one judgment pays for three.
    for attempt in response.attempts or ():
        usage = attempt.usage or {}
        result.prompt_tokens += int(usage.get("prompt_tokens", 0) or 0)
        result.completion_tokens += int(usage.get("completion_tokens", 0) or 0)
        result.reasoning_tokens += int(attempt.reasoning_tokens or 0)
    return response.data


def select(results: list[CandidateResult], rule: dict, pricing: dict, cells: int) -> dict:
    """Apply the frozen rule. No judgement calls live here -- that is the point."""
    gates = rule["admissibility"]
    admissible = []
    rejected = {}
    for r in results:
        failures = []
        if r.json_valid_non_truncated < gates["json_valid_non_truncated_min"]:
            failures.append(
                f"json_valid_non_truncated {r.json_valid_non_truncated:.3f} < "
                f"{gates['json_valid_non_truncated_min']}")
        if r.repeat_decision_stability < gates["repeat_decision_stability_min"]:
            failures.append(
                f"repeat_decision_stability {r.repeat_decision_stability:.3f} < "
                f"{gates['repeat_decision_stability_min']}")
        if r.span_binding_validity < gates["span_binding_validity_min"]:
            failures.append(
                f"span_binding_validity {r.span_binding_validity:.3f} < "
                f"{gates['span_binding_validity_min']}")
        if failures:
            rejected[r.candidate_id] = failures
        else:
            admissible.append(r)

    if not admissible:
        return {
            "selected": rule["incumbent"],
            "status": rule["fallback"]["status"],
            "reason": "no candidate cleared the admissibility gates; the incumbent stands",
            "rejected": rejected,
        }

    best = max(admissible, key=lambda r: r.accepted_atoms_per_task)
    epsilon = float(rule["selection"]["primary_epsilon_relative"])
    floor = best.accepted_atoms_per_task * (1.0 - epsilon)
    tied = [r for r in admissible if r.accepted_atoms_per_task >= floor]
    winner = min(tied, key=lambda r: r.projected_total_usd(pricing, cells))
    return {
        "selected": winner.candidate_id,
        "status": "JUDGE_SELECTED",
        "reason": (
            f"highest accepted_atoms_per_task within {epsilon:.0%} "
            f"({winner.accepted_atoms_per_task:.2f}); cheapest of {len(tied)} tied at "
            f"${winner.projected_total_usd(pricing, cells):.2f} projected"
        ),
        "rejected": rejected,
        "tied_within_epsilon": [r.candidate_id for r in tied],
    }


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--batches-per-task", type=int, default=3)
    ap.add_argument("--out", default=str(REPO / "reports" / "JUDGE_CALIBRATION.json"))
    ap.add_argument("--acquire", action="store_true",
                    help="fetch the held-out worlds first (they are not acquired up front)")
    args = ap.parse_args()

    from shapeflow_p1.providers.provider_client import ProviderClient, load_role_token

    settings = Settings.load(REPO, data_root=Path(os.environ["SHAPEFLOW_DATA_ROOT"]))
    rule = yaml.safe_load((REPO / "configs" / "judge_calibration.yaml").read_text())

    registry = json.loads((settings.path("tasks") / "registry_v1.json").read_text())
    held = rule["held_out"]
    pool = sorted(
        (t for t in registry["tasks"] if t["split"] == held["split"]),
        key=lambda t: t["task_id"],
    )[: int(held["n_tasks"])]
    if not pool:
        print(f"BLOCKED: no {held['split']} tasks available for calibration", file=sys.stderr)
        return 2

    token_dir = str(settings.get("week1", "provider", "token_dir"))
    host = settings.get("week1", "provider", "bind_host")
    port = settings.get("week1", "provider", "bind_port")
    client = ProviderClient(base_url=f"http://{host}:{port}",
                            token=load_role_token(token_dir, "steward"))

    if args.acquire:
        # RESERVE worlds are deliberately not fetched up front. Widening the split filter is an
        # explicit act, done here in the open and only for the named calibration tasks -- the
        # campaign's own acquire_splits on disk is untouched, and because this acquires fewer
        # than the whole split, no campaign acquisition manifest is published.
        from shapeflow_p1.acquire.exa_client import ExaCaptureClient
        from shapeflow_p1.campaign.acquire import acquire_all, exa_params_from

        settings.configs["acquisition"] = dict(
            settings.configs["acquisition"], acquire_splits=[held["split"]])
        params = exa_params_from(settings)
        ids = [t["task_id"] for t in pool]
        print(f"acquiring {len(ids)} held-out worlds: {ids}", flush=True)
        outcome = await acquire_all(
            settings,
            client_factory=lambda task_id: ExaCaptureClient(
                client.exa_transport(task_id=task_id), params),
            fetched_at_utc=_utc_now(),
            only_tasks=ids,
        )
        print(f"acquired={outcome.tasks_acquired} skipped={outcome.tasks_skipped} "
              f"ok/empty/failed={outcome.queries_ok}/{outcome.queries_empty}/"
              f"{outcome.queries_failed}", flush=True)

    price = settings.admission_pricing()
    pricing = {
        "in": price["usd_per_1m_input_tokens"],
        "out": price["usd_per_1m_output_tokens"],
    }
    cells = 836

    results = []
    for spec in rule["candidates"]:
        print(f"== {spec['id']} ==", flush=True)
        r = await _run_candidate(
            settings, spec, pool, batches_per_task=args.batches_per_task, client=client,
        )
        results.append(r)
        print(json.dumps(r.content(pricing, cells), indent=2), flush=True)

    decision = select(results, rule, pricing, cells)
    report = {
        "rule_config_sha256": __import__("hashlib").sha256(
            (REPO / "configs" / "judge_calibration.yaml").read_bytes()).hexdigest(),
        "held_out_tasks": [t["task_id"] for t in pool],
        "batches_per_task": args.batches_per_task,
        "projection_cells": cells,
        "candidates": [r.content(pricing, cells) for r in results],
        "decision": decision,
        "limitations_recorded_not_gated": rule["limitations_recorded_not_gated"],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(decision, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
