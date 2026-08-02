"""The analysis, rendered so a reader can disagree with it.

Evaluator-only, because it reads an artifact built from the answer key and because putting a
renderer on the treatment side is how the answer key eventually gets imported "just to format
something".

Two rules the tables here follow, both learned from what goes wrong in write-ups of this shape:

- **Costs and goods are never put in one column.** Prompt tokens and completion tokens move in
  opposite directions under P1 -- that is the whole trade -- so a single "tokens" figure with an
  arrow next to it would hide exactly the thing being measured.
- **A number with no interval is not reported.** Every contrast carries its paired 95% bootstrap
  interval, and whether that interval excludes zero is printed rather than left to the reader's
  eye. An effect whose interval spans zero is written as "no detectable difference", never as a
  small effect in the direction the point estimate happens to fall.
"""

from __future__ import annotations

from typing import Mapping, Optional, Sequence

__all__ = ["EVALUATOR_ONLY", "render_markdown"]

EVALUATOR_ONLY = True

#: How each metric is named in prose, and whether it is a cost or a good.
_LABELS: Mapping[str, tuple[str, str]] = {
    "accuracy": ("Accuracy (official grader)", "good"),
    "evidence_recall": ("Evidence recall (agent-level)", "good"),
    "gold_recall": ("Gold recall (agent-level)", "good"),
    "interval_union_seconds": ("GPU-busy seconds (primary work)", "cost"),
    "prompt_tokens": ("Prompt tokens (co-primary)", "cost"),
    "completion_tokens": ("Completion tokens (co-primary)", "cost"),
    "cached_prompt_tokens": ("Cached prompt tokens", "cost"),
    "e2e_latency_seconds": ("End-to-end seconds", "cost"),
    "search_queries": ("Search queries issued", "neutral"),
}

#: Reported first, because they are the three the verdict turns on.
_HEADLINE = ("accuracy", "evidence_recall", "interval_union_seconds",
             "prompt_tokens", "completion_tokens", "e2e_latency_seconds")


def _binding_lines(context: Mapping) -> list[str]:
    """The binding block: one line normally, two when the analysis ran under a different one.

    Analysis normally runs under the binding the run committed under, the two digests are the
    same string, and "execution binding" names it unambiguously. When `--ran-under-binding` moved
    it they differ, and the live digest is then the tree that *graded* the run rather than the one
    that produced any of its cells -- the reading a reader would otherwise take.

    In that case the label changes rather than merely gaining a sibling. Everywhere else -- in
    `P1_FINDINGS.md`, in the approval chain -- "execution binding" means the binding a *cell* was
    committed under, so leaving the analysis-time digest under that name would leave one phrase
    denoting two different hashes across documents a reader is meant to cross-reference.
    """
    live = str(context.get("execution_binding_sha256", ""))
    cells = str(context.get("cells_committed_under_sha256", ""))
    if not cells or cells == live:
        return [f"- Execution binding: `{live[:16]}`"]
    return [
        f"- Analysis binding: `{live[:16]}` (the tree that graded the run)",
        f"- Execution binding: `{cells[:16]}` -- the cells were committed under this; "
        f"re-run with `--ran-under-binding {cells}`",
    ]


def _n(value, digits: int = 3) -> str:
    if value is None:
        return "--"
    if isinstance(value, bool):
        return str(value)
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if abs(number) >= 1000:
        return f"{number:,.0f}"
    return f"{number:.{digits}f}".rstrip("0").rstrip(".") or "0"


def _verdict_word(metric: Mapping) -> str:
    """One phrase per contrast. 'No detectable difference' when the interval spans zero."""
    if not metric.get("reportable"):
        return "not measured"
    if not metric.get("excludes_zero"):
        return "no detectable difference"
    return "better" if metric.get("direction") == "better" else "WORSE"


def render_markdown(report: Mapping, *, title: str = "BrowseComp-Plus: P1 against P0") -> str:
    lines: list[str] = [f"# {title}", ""]
    context = report.get("context") or {}
    arms = report.get("arms") or {}
    contrasts = report.get("contrasts") or {}
    baseline = report.get("baseline_arm", "P0")

    lines += [
        "## What this is",
        "",
        f"- Baseline arm: **{baseline}**",
        f"- Cells: {report.get('cells_committed')} committed of {report.get('cells_total')}",
        f"- Tasks: {len(report.get('tasks') or ())}",
        f"- Run: `{context.get('run_id', '')}` layer `{context.get('layer', '')}`"
        f" lanes `{context.get('lanes')}`",
        *_binding_lines(context),
        f"- Answer key: `{str(context.get('evaluator_source_sha256', ''))[:16]}`",
        f"- Bootstrap: {(report.get('bootstrap') or {}).get('resamples')} resamples, "
        f"seed {(report.get('bootstrap') or {}).get('seed')}",
        "",
    ]

    grading = context.get("grading") or {}
    if grading.get("skipped"):
        lines += ["> Accuracy was not computed for this report (`--skip-grading`).", ""]
    elif grading.get("ungraded"):
        lines += [f"> {grading['ungraded']} of {grading.get('graded')} cells have no judgment "
                  "and are excluded from accuracy -- they are not counted as wrong.", ""]

    lines += _arm_table(arms)
    lines += _liveness_table(arms, baseline=baseline)
    for arm in sorted(contrasts):
        lines += _contrast_section(arm, contrasts[arm], baseline=baseline)
    lines += _reading(contrasts, arms, baseline=baseline)
    # Every section builder ends with "" so the next one starts after a blank line, which leaves
    # a trailing empty on the last section and a blank line at end of file. The repository's
    # whitespace gate rejects that, so a generated report could not be committed as generated --
    # and an artifact that has to be hand-edited before it can ship is no longer the artifact.
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n"


def _arm_table(arms: Mapping) -> list[str]:
    header = ("| arm | cells | committed | accuracy | evidence recall | GPU-busy s | "
              "prompt tok | completion tok | e2e s |")
    rows = [header, "|---|---|---|---|---|---|---|---|---|"]
    for arm_id in sorted(arms):
        a = arms[arm_id]
        rows.append(
            f"| `{arm_id}` | {a.get('cells')} | {a.get('committed')} | "
            f"{_n(a.get('accuracy'))} | {_n(a.get('evidence_recall_mean'))} | "
            f"{_n(a.get('interval_union_seconds_mean'), 1)} | "
            f"{_n(a.get('prompt_tokens_mean'), 0)} | "
            f"{_n(a.get('completion_tokens_mean'), 0)} | "
            f"{_n(a.get('e2e_latency_seconds_mean'), 1)} |")
    return ["## Arms", "", *rows, ""]


def _liveness_table(arms: Mapping, *, baseline: str) -> list[str]:
    """Whether each treatment arm actually fired. Placed before the contrasts deliberately.

    An arm that fell back on every batch has the baseline's numbers under the treatment's name,
    and every contrast below it would be a comparison of P0 with P0. That has happened in this
    codebase before -- nineteen arms, every check green, not one P1 span published -- so the
    reader is shown the publication counts before being shown any effect.
    """
    rows = ["| arm | H published / offered | C published / offered | fallbacks | "
            "close failures | overall rate |",
            "|---|---|---|---|---|---|"]
    any_treatment = False
    dead_h, dead_c = [], []
    for arm_id in sorted(arms):
        if arm_id == baseline:
            continue
        a = arms[arm_id]
        if not a.get("p1_opportunities") and not a.get("p1_publications"):
            continue
        any_treatment = True
        h_n, h_d = a.get("h_publications") or 0, a.get("h_opportunities") or 0
        c_n, c_d = a.get("c_publications") or 0, a.get("c_opportunities") or 0
        treats_h = (a.get("page_variant") or "P0") != "P0"
        treats_c = (a.get("close_variant") or "P0") != "P0"
        if treats_h and not h_n:
            dead_h.append(arm_id)
        if treats_c and not c_n:
            dead_c.append(arm_id)
        rows.append(
            f"| `{arm_id}` | {_fraction(h_n, h_d, treats_h)} | {_fraction(c_n, c_d, treats_c)} | "
            f"{a.get('page_fallbacks')} | {a.get('close_failures')} | "
            f"{_n(a.get('publication_rate'))} |")
    if not any_treatment:
        return ["## Did the treatment fire?", "",
                "**No arm published a single P1 output.** Every contrast below is a comparison "
                "of the baseline with itself.", ""]
    # The boundaries are reported apart because summing them hides exactly the failure this
    # study found: an arm running both can look healthy on a merged rate while one of its two
    # halves has never emitted a span.
    notes = []
    for arms_dead, where in ((dead_h, "H (WEBPAGE_P1)"), (dead_c, "C (RESEARCHER_CLOSE)")):
        if arms_dead:
            notes.append(
                f"**Nothing was published at {where}** by {', '.join(f'`{a}`' for a in arms_dead)}"
                ": every opportunity fell back to P0. For those arms the contrast below measures "
                "the cost of attempting P1, not the effect of receiving it.")
    return ["## Did the treatment fire?", "", *rows, "",
            *([n for note in notes for n in (note, "")] if notes else [])]


def _fraction(numerator: int, denominator: int, treated: bool) -> str:
    """`--` where the arm has no treatment at this boundary: absent is not the same as zero.

    ``treated`` comes from the arm's variant, not from the denominator. Reading it off a zero
    denominator would print the same `--` for "this arm does not touch this boundary" and for
    "this arm treats this boundary and never got a single opportunity" -- and the second is a
    finding.
    """
    if not treated:
        return "--"
    if not denominator:
        return "0 / 0 (no opportunity)"
    return f"{numerator} / {denominator} ({numerator / denominator:.0%})"


def _contrast_section(arm: str, contrast: Mapping, *, baseline: str) -> list[str]:
    metrics = contrast.get("metrics") or {}
    pairing = contrast.get("pairing") or {}
    lines = [f"## `{arm}` vs `{baseline}`", "",
             f"Paired on {contrast.get('n_pairs')} tasks where both arms committed."]
    if pairing and not pairing.get("reportable", True):
        lines.append(
            f"**Survivorship warning:** only {pairing.get('survival', 0):.0%} of the smaller "
            "arm's committed tasks survived the pairing, so this contrast is computed on a task "
            "set that the arms' own failures selected.")
    lines += ["",
              "| metric | baseline | treatment | paired diff | 95% CI | wins/losses | verdict |",
              "|---|---|---|---|---|---|---|"]
    ordered = [k for k in _HEADLINE if k in metrics] + [
        k for k in sorted(metrics) if k not in _HEADLINE]
    for key in ordered:
        metric = metrics[key]
        label = _LABELS.get(key, (key, "neutral"))[0]
        if not metric.get("reportable"):
            lines.append(f"| {label} | -- | -- | -- | -- | -- | not measured |")
            continue
        lines.append(
            f"| {label} | {_n(metric.get('baseline_mean'))} | "
            f"{_n(metric.get('treatment_mean'))} | "
            f"{_n(metric.get('mean_paired_difference'))} "
            f"({_n((metric.get('relative_change') or 0) * 100, 1)}%) | "
            f"[{_n(metric.get('ci95_low'))}, {_n(metric.get('ci95_high'))}] | "
            f"{metric.get('wins')}/{metric.get('losses')} | {_verdict_word(metric)} |")
    return [*lines, ""]


def _reading(contrasts: Mapping, arms: Mapping, *, baseline: str) -> list[str]:
    """The plain-language summary, derived from the intervals rather than written by hand."""
    lines = ["## Reading", ""]
    if not contrasts:
        return [*lines, "No treatment arm ran.", ""]
    for arm in sorted(contrasts):
        metrics = contrasts[arm].get("metrics") or {}
        works, fails, flat = [], [], []
        # Iterate the label registry, not the caller's dict. The JSON artifact is written with
        # sorted keys and the markdown is rendered from the in-memory report, so reading the
        # JSON back and re-rendering used to produce a different ordering from the same data --
        # a report that cannot be re-derived byte-for-byte from its own record. Any metric the
        # registry does not name is appended, so a new endpoint still appears.
        ordered = [k for k in _LABELS if k in metrics]
        ordered += [k for k in metrics if k not in _LABELS]
        for key in ordered:
            metric = metrics[key]
            if not metric.get("reportable"):
                continue
            label = _LABELS.get(key, (key, "neutral"))[0]
            if not metric.get("excludes_zero"):
                flat.append(label)
            elif metric.get("direction") == "better":
                works.append(label)
            else:
                fails.append(label)
        published = (arms.get(arm) or {}).get("p1_publications", 0)
        lines.append(f"**`{arm}`** -- {published} P1 publications.")
        lines.append(f"- Improves: {', '.join(works) if works else 'nothing measurably'}")
        lines.append(f"- Costs: {', '.join(fails) if fails else 'nothing measurably'}")
        lines.append(f"- Unchanged within the interval: "
                     f"{', '.join(flat) if flat else 'nothing'}")
        lines.append("")
    return lines
