from __future__ import annotations

from pathlib import Path

import pytest

from shapeflow_p1.analysis.estimands import load_scoped_scores
from shapeflow_p1.scoped_paths import resolve_scoped_path, safe_scope_component


@pytest.mark.parametrize(
    "value",
    [
        "",
        ".",
        "..",
        "../../escape",
        "/outside-judgments/escape",
        r"C:\tmp\escape",
        r"run\..\escape",
        "contains space",
        "x" * 129,
    ],
)
def test_scope_component_rejects_paths_and_non_protocol_names(value: str):
    with pytest.raises(ValueError, match="one safe path component"):
        safe_scope_component(value, name="run_id")


def test_resolved_scoped_path_stays_under_the_configured_root(tmp_path: Path):
    result = resolve_scoped_path(
        tmp_path,
        run_id="run-01",
        phase_id="FORMATIVE_SCREEN",
        tail=("analysis", "ITT_VERDICT_INPUTS.json"),
    )
    assert result == (
        tmp_path.resolve()
        / "run-01"
        / "FORMATIVE_SCREEN"
        / "analysis"
        / "ITT_VERDICT_INPUTS.json"
    )


def test_existing_scope_symlink_cannot_redirect_a_write(tmp_path: Path):
    root = tmp_path / "judgments"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "run-1").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="escapes the configured root"):
        resolve_scoped_path(
            root,
            run_id="run-1",
            phase_id="phase-1",
            tail=("analysis", "E2E_ANALYSIS.json"),
        )


@pytest.mark.parametrize(
    ("run_id", "phase_id"),
    [
        ("../../outside", "phase"),
        ("/outside-judgments", "phase"),
        ("run", "../../outside"),
        ("run", "/outside-judgments"),
    ],
)
def test_score_loader_rejects_scope_escape_before_reading(
    tmp_path: Path, run_id: str, phase_id: str,
):
    with pytest.raises(ValueError, match="one safe path component"):
        load_scoped_scores(tmp_path, run_id=run_id, phase_id=phase_id)
