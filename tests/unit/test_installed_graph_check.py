"""A stale installed graph must be a refusal, not a quiet empty result.

The failure this guards against already happened. ``open_deep_research`` installs as a *copy*
into site-packages, so re-materializing the patch changes nothing until the copy is reinstalled
-- and the approval cannot see the difference, because ``patched_tree_sha`` binds the recorded
hash *file*, not the running bytes.

After the package rename the host went on running hooks that did
``from shapeflow_p1.odr import vendor_hooks``. Vendor's supervisor catches every exception out
of that block (``if is_token_limit_exceeded(e, ...) or True:``) and returns an empty note set,
so an ImportError presented as "the agent decided not to research": every cell committed, every
report was written from nothing, and no gate said a word.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from shapeflow.doctor import check_installed_graph
from shapeflow.treehash import tree_sha256


def _repo_with_expectation(tmp_path: Path, digest: str) -> Path:
    (tmp_path / "patches").mkdir(parents=True, exist_ok=True)
    (tmp_path / "patches" / "patched_tree.sha256").write_text(digest + "\n", encoding="utf-8")
    return tmp_path


def _installed_digest() -> str:
    import open_deep_research

    return tree_sha256(Path(open_deep_research.__path__[0]))


def test_a_matching_tree_passes():
    result = check_installed_graph(_repo_with_expectation(
        Path(pytest.importorskip("tempfile").mkdtemp()), _installed_digest()))
    assert result.status == "PASS"
    assert result.name == "installed_graph"


def test_a_stale_installed_tree_fails_and_says_how_to_fix_it(tmp_path):
    result = check_installed_graph(_repo_with_expectation(tmp_path, "0" * 64))
    assert result.status == "FAIL"
    assert "materialize_vendor.sh" in result.detail, (
        "the message has to name the fix: the failure mode is silent and the operator will "
        "otherwise reach for the approval, which cannot see this")


def test_a_missing_expectation_is_a_failure_not_a_skip(tmp_path):
    """Nothing pinning the graph is worse than a mismatch, not better."""
    result = check_installed_graph(tmp_path)
    assert result.status == "FAIL"


def test_the_check_is_wired_into_the_pure_checks():
    """It has to run where it cannot be forgotten -- doctor, not a script somebody remembers."""
    from shapeflow.doctor import run_pure_checks

    repo = Path(__file__).resolve().parents[2]
    report = run_pure_checks(repo=repo, configs={}, schema_dir=repo / "schemas", role=None)
    assert any(c.name == "installed_graph" for c in report.checks)
