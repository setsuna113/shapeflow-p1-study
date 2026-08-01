"""Grading a finished run must not require the tree to still be at the run's commit.

A work key is namespaced by the execution binding, and the binding contains ``approved_commit``,
so every commit renames every work key ever written. That is correct for the runner -- a cell
committed under different bytes is a different cell -- and a trap for the evaluator, which reads
a finished run and writes no cell. With the namespace pinned to HEAD, a bug in the judge parser
could only be fixed by re-running the campaign that exposed it.

So ``grade-bcplus --ran-under-binding`` exists, and the whole of its safety is that the digest
must appear in the repository's own append-only approval chain.
"""

from __future__ import annotations

import json
from pathlib import Path

from shapeflow.protocol import approval_chain_digests


def _repo_and_approval(tmp_path: Path) -> tuple[Path, Path]:
    """The approval must live outside the tree it approves, or it approves itself."""
    repo = tmp_path / "repo"
    repo.mkdir(parents=True, exist_ok=True)
    return repo, tmp_path / "approvals" / "launch_approval.json"


def _write_index(approval: Path, links) -> None:
    history = approval.parent / "history"
    history.mkdir(parents=True, exist_ok=True)
    (history / "INDEX.json").write_text(json.dumps({"approvals": links}), encoding="utf-8")


def _chain(tmp_path: Path, digests) -> tuple[Path, Path]:
    repo, approval = _repo_and_approval(tmp_path)
    _write_index(approval, [
        {"sequence": i + 1, "binding_sha256": d, "file": f"a{i}.json"}
        for i, d in enumerate(digests)
    ])
    return repo, approval


def test_the_chain_yields_every_binding_ever_approved(tmp_path):
    repo, approval = _chain(tmp_path, ["a" * 64, "b" * 64, "c" * 64])
    assert approval_chain_digests(repo, approval) == {"a" * 64, "b" * 64, "c" * 64}


def test_an_absent_chain_authorises_nothing(tmp_path):
    """No index means no recorded approval, so no past namespace may be named."""
    repo, approval = _repo_and_approval(tmp_path)
    assert approval_chain_digests(repo, approval) == frozenset()


def test_an_unrecorded_digest_is_not_in_the_chain(tmp_path):
    """The check that makes --ran-under-binding an audit question, not a free parameter."""
    repo, approval = _chain(tmp_path, ["a" * 64])
    assert "d" * 64 not in approval_chain_digests(repo, approval)


def test_links_without_a_digest_are_skipped_rather_than_admitted_as_none(tmp_path):
    repo, approval = _repo_and_approval(tmp_path)
    _write_index(approval, [{"sequence": 1}, {"sequence": 2, "binding_sha256": "e" * 64}])
    digests = approval_chain_digests(repo, approval)
    assert digests == {"e" * 64}
    assert "None" not in digests
