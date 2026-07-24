"""Variant elimination, promotion ordering, tie-breaks, and node-level verdicts."""

from __future__ import annotations

from shapeflow_p1.experiment.promotion import (
    ScreenGates,
    ScreenResult,
    node_verdict_when_none_advance,
    promote,
    screen_eliminate,
    simplicity_key,
)

GATES = ScreenGates(min_selector_quality=0.90, max_crash_rate=0.10, indifference_band=0.02,
                    max_per_node=2)


def _r(vid, node="WEBPAGE_P1", *, structural=True, quality=0.95, harm_ok=True, saving=0.2,
       crash=0.0, contract="id", agg="stable_union", close="separate", control=False,
       det_struct=False):
    return ScreenResult(
        variant_id=vid, node=node, structural_pass=structural, selector_quality=quality,
        harm_screen_pass=harm_ok, work_saving=saving, crash_rate=crash, contract=contract,
        aggregation=agg, close_mode=close, is_control=control,
        deterministic_structural_failure=det_struct,
    )


# --- elimination ----------------------------------------------------------------------


def test_structural_failure_eliminates():
    assert screen_eliminate(_r("v", structural=False), GATES) is not None


def test_harm_failure_eliminates():
    assert "harm" in screen_eliminate(_r("v", harm_ok=False), GATES)


def test_low_quality_eliminates():
    assert "quality" in screen_eliminate(_r("v", quality=0.80), GATES)


def test_high_crash_eliminates():
    assert "crash" in screen_eliminate(_r("v", crash=0.5), GATES)


def test_work_increase_without_quality_eliminates():
    assert "work increased" in screen_eliminate(_r("v", saving=-0.1, quality=0.95), GATES)


def test_clean_variant_survives():
    assert screen_eliminate(_r("v"), GATES) is None


# --- promotion ordering ---------------------------------------------------------------


def test_promotes_at_most_two_per_node():
    results = [_r(f"v{i}", saving=0.1 * i) for i in range(1, 6)]
    promoted = promote(results, GATES)
    assert len(promoted["WEBPAGE_P1"]) == 2
    # highest savings win when well outside the band
    assert set(promoted["WEBPAGE_P1"]) == {"v5", "v4"}


def test_controls_never_promote():
    results = [_r("p0", control=True, saving=0.9), _r("real", saving=0.2)]
    promoted = promote(results, GATES)
    assert promoted["WEBPAGE_P1"] == ["real"]


def test_indifference_band_prefers_simpler():
    # Two variants within 0.02 saving: the simpler contract (id < bridge) wins the top slot.
    results = [
        _r("bridge_v", saving=0.205, contract="bridge"),
        _r("id_v", saving=0.200, contract="id"),
    ]
    promoted = promote(results, ScreenGates(indifference_band=0.02, max_per_node=1))
    assert promoted["WEBPAGE_P1"] == ["id_v"]


def test_outside_band_higher_saving_wins_even_if_complex():
    # A clearly higher saving (well beyond the band) beats simplicity.
    results = [
        _r("bridge_v", saving=0.40, contract="bridge"),
        _r("id_v", saving=0.20, contract="id"),
    ]
    promoted = promote(results, ScreenGates(indifference_band=0.02, max_per_node=1))
    assert promoted["WEBPAGE_P1"] == ["bridge_v"]


def test_simplicity_key_orders_contract_agg_close():
    assert simplicity_key(_r("v", contract="id")) < simplicity_key(_r("v", contract="bridge"))
    assert simplicity_key(_r("v", agg="stable_union")) < simplicity_key(_r("v", agg="global_rerank"))
    assert simplicity_key(_r("v", close="separate")) < simplicity_key(_r("v", close="fused_with_fallback"))


# --- node verdict when nothing advances -----------------------------------------------


def test_kill_structural_only_when_whole_family_deterministically_fails():
    results = [
        _r("v1", structural=False, det_struct=True),
        _r("v2", structural=False, det_struct=True),
    ]
    assert node_verdict_when_none_advance("WEBPAGE_P1", results) == "KILL_STRUCTURAL"


def test_not_established_when_only_some_fail():
    # One variant merely underperformed (not a deterministic structural failure) -> not KILL.
    results = [
        _r("v1", structural=False, det_struct=True),
        _r("v2", quality=0.5, det_struct=False),
    ]
    assert node_verdict_when_none_advance("WEBPAGE_P1", results) == "NOT_ESTABLISHED"


def test_controls_do_not_force_kill_structural():
    results = [
        _r("p0", control=True, det_struct=False),
        _r("v1", structural=False, det_struct=True),
    ]
    # Only the real variant matters, and it deterministically fails -> KILL_STRUCTURAL.
    assert node_verdict_when_none_advance("WEBPAGE_P1", results) == "KILL_STRUCTURAL"
