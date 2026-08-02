"""The cited-artifact gate must fail on the thing it was written for.

`reports/*` is gitignored with per-file exceptions and the finals are force-added by hand, so a
newly generated artifact is skipped by ``git add -A`` in silence: the commit succeeds and the
report ships citing evidence nobody who clones the repository receives. That happened to
`reports/BCPLUS_campaign1.json`, which `P1_FINDINGS.md` names as the paired analysis behind every
number in it.

A gate that cannot fail is not a gate, so the cases here are the ones that must be rejected --
an untracked citation and a missing one -- rather than only the clean path.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

CHECK_PATH = Path(__file__).resolve().parents[2] / "tools" / "ci" / "check_cited_artifacts.py"


def _load_check():
    """Import the CI script by path.

    ``tools/ci`` is deliberately not a package -- the checks are run as scripts by
    ``scripts/verify_local.sh`` and must not become importable from the library. Registering the
    module in ``sys.modules`` before executing it is required: ``@dataclass`` looks its own module
    up there while it builds the class and raises if the entry is missing.
    """
    spec = importlib.util.spec_from_file_location("shapeflow_ci_cited_artifacts", CHECK_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _repo(tmp_path: Path) -> Path:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.email", "t@example.invalid"],
                   cwd=tmp_path, check=True)
    subprocess.run(["git", "config", "user.name", "t"], cwd=tmp_path, check=True)
    (tmp_path / "reports").mkdir()
    return tmp_path


def _run(module, repo: Path, monkeypatch) -> int:
    monkeypatch.setattr(module, "REPO", repo)
    return module.main()


def test_a_citation_of_an_untracked_file_is_refused(tmp_path, monkeypatch, capsys):
    """The exact shape of the bug: the artifact is on disk, and only on disk."""
    repo = _repo(tmp_path)
    (repo / "reports" / "FINDINGS.md").write_text(
        "The analysis is `reports/ANALYSIS.json`.\n", encoding="utf-8")
    (repo / "reports" / "ANALYSIS.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "reports/FINDINGS.md"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "report only"], cwd=repo, check=True)

    module = _load_check()
    assert _run(module, repo, monkeypatch) == 1
    err = capsys.readouterr().err
    assert "reports/ANALYSIS.json" in err
    assert "not tracked" in err, "the message must say why, not just that it failed"


def test_a_citation_of_a_file_that_does_not_exist_is_refused(tmp_path, monkeypatch, capsys):
    repo = _repo(tmp_path)
    (repo / "reports" / "FINDINGS.md").write_text(
        "See `reports/GONE.json`.\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "dangling citation"], cwd=repo, check=True)

    module = _load_check()
    assert _run(module, repo, monkeypatch) == 1
    assert "does not exist" in capsys.readouterr().err


def test_a_tracked_citation_passes(tmp_path, monkeypatch):
    repo = _repo(tmp_path)
    (repo / "reports" / "FINDINGS.md").write_text(
        "The analysis is `reports/ANALYSIS.json`.\n", encoding="utf-8")
    (repo / "reports" / "ANALYSIS.json").write_text("{}\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "both"], cwd=repo, check=True)

    module = _load_check()
    assert _run(module, repo, monkeypatch) == 0


def test_a_placeholder_is_not_a_citation(tmp_path, monkeypatch):
    """`reports/BCPLUS_<run-id>.{json,md}` names a family. Demanding it be tracked would make
    the gate impossible to satisfy and train people to delete it."""
    repo = _repo(tmp_path)
    (repo / "reports" / "FINDINGS.md").write_text(
        "A run writes `reports/BCPLUS_<run-id>.{json,md}` and `reports/gates/`.\n",
        encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(["git", "commit", "-qm", "template only"], cwd=repo, check=True)

    module = _load_check()
    assert _run(module, repo, monkeypatch) == 0


def test_a_non_repository_fails_cleanly_rather_than_with_a_traceback(tmp_path, monkeypatch):
    """The check shells out to git. Outside a checkout that used to raise CalledProcessError,
    which reads as the gate being broken rather than as the gate being inapplicable.

    `git ls-files` walks *up* from its -C directory, so "not a repository" is a property of the
    whole ancestor chain rather than of this directory. Asserting it here keeps the test honest
    on a machine where the temp root happens to sit inside some checkout: without it the test
    would silently pass for the wrong reason.
    """
    probe = subprocess.run(["git", "-C", str(tmp_path), "rev-parse", "--git-dir"],
                           capture_output=True, text=True)
    if probe.returncode == 0:
        pytest.skip(f"{tmp_path} is inside a git repository; nothing to test")

    module = _load_check()
    monkeypatch.setattr(module, "REPO", tmp_path)      # no git init
    with pytest.raises(SystemExit) as caught:
        module.main()
    assert "cannot run" in str(caught.value), "the exit must say why, not just exit"
