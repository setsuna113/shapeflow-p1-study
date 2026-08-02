"""The paired analysis must not turn an absence into a number.

Three failure modes are tested here rather than argued about, because each one produces a
publishable-looking table:

- an unavailable judgment counted as a wrong answer, which makes a judge outage look like a
  quality regression in whichever arm was being graded at the time;
- a contrast whose pairing dropped a third of the tasks reported with no warning, which makes a
  fragile arm look accurate because the tasks it failed left the denominator;
- a treatment arm that published nothing reported as an enormous work saving, which is exactly
  what an inert P1 looks like from the outside.
"""

from __future__ import annotations

from shapeflow.bench.bcplus.analysis import (
    CellRecord,
    build_report,
    paired_contrast,
    summarize_arm,
)
from shapeflow.bench.bcplus.render import render_markdown


def _runner_label(page_variant: str, close_variant: str) -> str:
    """The ``variant_id`` a real cell carries: the treated halves only, joined by ``+``.

    Reproduced here because the encoding is the trap. A both-boundary arm stamps ``"H02+C01"``
    and splits cleanly into a page half and a close half; a C-only arm stamps a bare ``"C01"``,
    whose first ``+``-separated field is a *close* variant sitting where a page variant would be.
    Analysis that parsed this label credited every C-only arm with a perfect publication rate at
    H -- the boundary it does not touch -- and printed ``--`` at C, the boundary under test.
    """
    treated = [v for v in (page_variant, close_variant) if v != "P0"]
    return "+".join(treated) if treated else "P0"


def _cell(task: str, arm: str, *, work: float, prompt: int, completion: int,
          correct=None, recall=None, state: str = "COMMITTED", counts=None,
          page_variant: str = "P0", close_variant: str = "P0",
          variant_id: str | None = None) -> CellRecord:
    return CellRecord(
        task_id=task, arm_id=arm,
        # Defaults to the runner's own lossy label: the two variants joined, with a ``P0`` half
        # dropped. Tests pass the boundary variants separately and on purpose, because the label
        # is not what the analysis may read.
        variant_id=variant_id if variant_id is not None else _runner_label(
            page_variant, close_variant),
        page_variant=page_variant, close_variant=close_variant,
        replicate_id="0", seed=1, state=state,
        final_report="x", error="", counts=counts or {"search_queries": 3},
        work={"interval_union_seconds": work,
              "tokens": {"prompt_tokens": prompt, "completion_tokens": completion,
                         "cached_prompt_tokens": 0}},
        e2e_latency_seconds=work + 1.0, energy_joules=None, retrieval_trace=[],
        pages_registered=5, output_ref=f"ref-{task}-{arm}",
        evidence_recall=recall, correct=correct)


def _pair(n: int, *, treatment_correct, baseline_correct, treatment_work, baseline_work):
    records = []
    for i in range(n):
        task = f"t{i}"
        records.append(_cell(task, "P0", work=baseline_work, prompt=1000, completion=800,
                             correct=baseline_correct, recall=0.5))
        records.append(_cell(task, "H", work=treatment_work, prompt=1400, completion=200,
                             correct=treatment_correct, recall=0.5, page_variant="H02",
                             counts={"search_queries": 3, "page_batches_reduced": 2,
                                     "page_fallbacks": 0}))
    return records


def test_an_unavailable_grade_is_not_a_wrong_answer():
    cells = [_cell("t1", "P0", work=1, prompt=1, completion=1, correct=True),
             _cell("t2", "P0", work=1, prompt=1, completion=1, correct=None),
             _cell("t3", "P0", work=1, prompt=1, completion=1, correct=False)]
    summary = summarize_arm(cells)
    assert summary.accuracy_n == 2, "the ungraded cell must leave the denominator, not enter it"
    assert summary.accuracy == 0.5


def test_a_real_work_saving_is_detected_and_signed_as_a_saving():
    records = _pair(30, treatment_correct=True, baseline_correct=True,
                    treatment_work=4.0, baseline_work=10.0)
    contrast = paired_contrast([r for r in records if r.arm_id == "P0"],
                               [r for r in records if r.arm_id == "H"], resamples=400)
    work = contrast.metrics["interval_union_seconds"]
    assert work["excludes_zero"] and work["direction"] == "better"
    assert work["mean_paired_difference"] < 0, "less GPU time is a negative difference"
    # The trade must stay visible as two columns, never one.
    assert contrast.metrics["prompt_tokens"]["direction"] == "worse"
    assert contrast.metrics["completion_tokens"]["direction"] == "better"


def test_no_difference_is_reported_as_no_difference():
    records = _pair(30, treatment_correct=True, baseline_correct=True,
                    treatment_work=10.0, baseline_work=10.0)
    contrast = paired_contrast([r for r in records if r.arm_id == "P0"],
                               [r for r in records if r.arm_id == "H"], resamples=400)
    work = contrast.metrics["interval_union_seconds"]
    assert not work["excludes_zero"]


def test_survivorship_is_named_when_the_pairing_drops_tasks():
    records = _pair(20, treatment_correct=True, baseline_correct=False,
                    treatment_work=4.0, baseline_work=10.0)
    # The treatment arm dies on a quarter of the tasks. Those are exactly the tasks whose
    # absence flatters it.
    survivors = []
    dropped = 0
    for record in records:
        if record.arm_id == "H" and dropped < 5:
            dropped += 1
            record.state = "FAILED_FINAL"
        survivors.append(record)
    report = build_report(survivors, resamples=200)
    pairing = report["contrasts"]["H"]["pairing"]
    assert pairing["n_pairs"] == 15
    assert not pairing["reportable"], "a 75% survival must not be reported as a clean contrast"
    assert pairing["dropped_from_baseline"], "the dropped task ids must be nameable"


def test_an_inert_arm_shows_zero_publications_before_any_effect_is_shown():
    """The publication table is rendered above the contrasts, on purpose."""
    records = []
    for i in range(5):
        records.append(_cell(f"t{i}", "P0", work=10, prompt=1000, completion=800, correct=True,
                             recall=0.5))
        records.append(_cell(f"t{i}", "H", work=2, prompt=1000, completion=10, correct=True,
                             recall=0.5, page_variant="H02",
                             counts={"search_queries": 3, "page_batches_reduced": 2,
                                     "page_fallbacks": 2}))
    report = build_report(records, resamples=100)
    assert report["arms"]["H"]["p1_publications"] == 0
    markdown = render_markdown(report)
    assert markdown.index("Did the treatment fire?") < markdown.index("vs `P0`")
    assert "| `H` | 0 / 10 (0%) | -- |" in markdown, (
        "the H boundary ran and published nothing; C never ran, and the two must not read alike")
    assert "Nothing was published at H (WEBPAGE_P1)" in markdown


def test_the_render_never_sums_prompt_and_completion_tokens():
    records = _pair(10, treatment_correct=True, baseline_correct=True,
                    treatment_work=4.0, baseline_work=10.0)
    markdown = render_markdown(build_report(records, resamples=100))
    assert "Prompt tokens" in markdown and "Completion tokens" in markdown
    assert "Total tokens" not in markdown


def _hc_counts(*, h_batches: int, h_fallbacks: int, c_reduced: int, c_failed: int) -> dict:
    return {"search_queries": 3, "page_batches_reduced": h_batches,
            "page_fallbacks": h_fallbacks, "close_reduced": c_reduced,
            "close_failed": c_failed}


def test_a_dead_h_boundary_is_not_covered_by_a_live_c_boundary():
    """The failure that made this split necessary.

    An arm running both boundaries -- H_PLUS_C -- can post a healthy merged publication rate
    while its H half has never emitted a single span, because the C half's successes are summed
    into the same numerator. On BrowseComp-Plus that is not hypothetical: the C selector
    publishes and the H selector falls back every time.
    """
    counts = _hc_counts(h_batches=6, h_fallbacks=6, c_reduced=3, c_failed=1)
    cell = _cell("t1", "H_PLUS_C", work=1.0, prompt=10, completion=2, counts=counts,
                 page_variant="H02", close_variant="C01")
    summary = summarize_arm([cell])
    assert summary.h_publications == 0 and summary.h_opportunities == 6
    assert summary.c_publications == 3 and summary.c_opportunities == 4
    assert summary.h_publication_rate == 0.0
    assert summary.c_publication_rate == 0.75
    # The merged number is the one that would have hidden it.
    assert summary.publication_rate == 0.3, "3 of 10 -- reads as a working arm"


def test_an_arm_with_no_treatment_at_a_boundary_reports_none_not_zero():
    """Absent and never-published produce the same count and mean opposite things."""
    cell = _cell("t1", "H_MARKDOWN_ID", work=1.0, prompt=10, completion=2, page_variant="H02",
                 counts=_hc_counts(h_batches=4, h_fallbacks=4, c_reduced=0, c_failed=0))
    summary = summarize_arm([cell])
    assert summary.h_publication_rate == 0.0, "ran at H and published nothing"
    assert summary.c_publication_rate is None, "never ran at C at all"


def test_the_report_names_the_boundary_that_published_nothing():
    records = []
    for i in range(5):
        records += [
            _cell(f"t{i}", "P0", work=2.0 + i, prompt=100 + i, completion=20 + i),
            _cell(f"t{i}", "H_MARKDOWN_ID", work=3.0 + i, prompt=150 + i, completion=22 + i,
                  page_variant="H02",
                  counts=_hc_counts(h_batches=4, h_fallbacks=4, c_reduced=0, c_failed=0)),
            _cell(f"t{i}", "C_ID", work=2.0 + i, prompt=98 + i, completion=19 + i,
                  close_variant="C01",
                  counts=_hc_counts(h_batches=2, h_fallbacks=0, c_reduced=3, c_failed=1)),
        ]
    text = render_markdown(build_report(records, baseline_arm="P0", resamples=100))
    assert "Nothing was published at H (WEBPAGE_P1)" in text
    assert "`H_MARKDOWN_ID`" in text
    assert "Nothing was published at C" not in text, "C published; it must not be named"
    assert "0 / 20 (0%)" in text, "five cells x four batches, none published"
    assert "15 / 20 (75%)" in text
    assert "| `C_ID` | -- |" in text, (
        "C_ID's page half is plain P0; it must not be credited with publishing at H")


def _variant_cell(arm: str, counts: dict, **variants) -> CellRecord:
    return _cell("t1", arm, work=1.0, prompt=10, completion=2, counts=counts, **variants)


def test_a_p0_page_half_is_not_credited_with_publishing_at_h():
    """page_batches_reduced counts vendor batches too, and C_ID's page half is plain P0.

    Left ungated, a C-only arm records batches reduced with no fallbacks and is reported as
    publishing at H with a perfect rate -- while having no H treatment whatsoever. That would
    contradict the study's central finding in the study's own table.
    """
    summary = summarize_arm([_variant_cell(
        "C_ID", _hc_counts(h_batches=2, h_fallbacks=0, c_reduced=3, c_failed=1),
        close_variant="C01")])
    assert summary.h_opportunities == 0 and summary.h_publications == 0
    assert summary.h_publication_rate is None, "no H treatment is not a zero rate"
    assert summary.c_publication_rate == 0.75


def test_a_p0_close_half_is_not_credited_with_publishing_at_c():
    summary = summarize_arm([_variant_cell(
        "H_MARKDOWN_ID", _hc_counts(h_batches=4, h_fallbacks=1, c_reduced=2, c_failed=0),
        page_variant="H02")])
    assert summary.c_opportunities == 0 and summary.c_publication_rate is None
    assert summary.h_publications == 3 and summary.h_opportunities == 4


def test_the_boundary_is_read_from_the_variant_not_from_the_stamped_label():
    """The regression the campaign's own report was printed with.

    Every test above passed while the live analysis credited `C_ID` and `C_CPU_CONTROL` with a
    100% publication rate at H and `--` at C, because they fed the parser ``"P0+C01"`` -- a
    string no runner emits. A C-only cell is stamped with a bare ``"C01"``, which the parser read
    as a *page* variant. So the label is pinned to the real one here, and the boundary counts
    must come out the same as the test above.
    """
    cell = _variant_cell(
        "C_ID", _hc_counts(h_batches=2, h_fallbacks=0, c_reduced=3, c_failed=1),
        close_variant="C01")
    assert cell.variant_id == "C01", "this is what the runner actually stamps on a C-only cell"
    summary = summarize_arm([cell])
    assert summary.h_publication_rate is None, (
        "a bare close-variant label must not be read as a page variant")
    assert summary.c_publications == 3 and summary.c_opportunities == 4
    text = render_markdown({"arms": {"C_ID": summary.content()},
                            "contrasts": {}, "baseline_arm": "P0", "cells": []})
    assert "| `C_ID` | -- | 3 / 4 (75%) |" in text


def test_a_run_id_that_escapes_its_root_is_refused():
    """run_id and layer come off the command line and are then joined into paths.

    The schedule directory, the frozen block directory, the STATUS file and the report filename
    are all built from them. The grammar that guards that boundary lived in shapeflow.scoped_paths
    with no caller at all, so "../.." would have written a campaign's schedule outside the data
    root the whole role isolation depends on.
    """
    import pytest

    from shapeflow.scoped_paths import safe_scope_component

    assert safe_scope_component("campaign1", name="run_id") == "campaign1"
    assert safe_scope_component("bcplus-b1_select-x.2", name="run_id") == "bcplus-b1_select-x.2"
    for bad in ("../..", "a/b", "", ".hidden", "x" * 129, "with space", "/abs"):
        with pytest.raises(ValueError, match="one safe path component"):
            safe_scope_component(bad, name="run_id")
