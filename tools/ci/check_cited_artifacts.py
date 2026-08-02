#!/usr/bin/env python3
"""Every artifact a tracked report points at must itself be tracked.

`.gitignore` carries ``reports/*`` on purpose -- the generated ones stay local and a human
force-adds the finals. The failure mode that design creates is silent: ``git add -A`` skips the
new artifact without a word, the commit succeeds, the report ships citing a file that is not in
the repository, and nothing anywhere says so. That happened to `reports/BCPLUS_campaign1.json`,
which `P1_FINDINGS.md` names as the paired analysis behind its numbers.

A report whose evidence is missing from the repository is the same class of problem as a gate
that can be satisfied by asserting it was satisfied: the claim is there and the thing that backs
it is not. So the citation is checked rather than trusted.

Scope is deliberately narrow -- repo-relative paths under ``reports/`` and ``protocol/``, named
inside tracked Markdown under ``reports/``. Prose about a directory is not a citation, and a
glob or a shell placeholder (``<run-id>``) names a family rather than a file.

Exit 0 clean, 1 on a violation. Wired into `scripts/verify_local.sh`.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Directories whose files are evidence rather than commentary.
CITED_ROOTS = ("reports/", "protocol/")

#: A citation is a repo-relative path with a file extension. Backticked or bare, both appear.
_CITATION = re.compile(
    r"(?:^|[\s`(\[])((?:reports|protocol)/[A-Za-z0-9_./-]+\.[A-Za-z0-9]{1,6})")

#: Placeholders stand for a family of artifacts, not one file. `<run-id>` and `*` are the two
#: forms the reports use; neither can be resolved to something that could be tracked.
_PLACEHOLDER = re.compile(r"[<>*{}]")


def tracked_files() -> set[str]:
    out = subprocess.run(["git", "-C", str(REPO), "ls-files"],
                         capture_output=True, text=True, check=True).stdout
    return set(out.split("\n")) - {""}


def main() -> int:
    tracked = tracked_files()
    reports = sorted(p for p in tracked
                     if p.startswith("reports/") and p.endswith(".md"))
    failures: list[str] = []
    checked = 0
    for rel in reports:
        text = (REPO / rel).read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), 1):
            for cited in _CITATION.findall(line):
                if _PLACEHOLDER.search(cited):
                    continue
                checked += 1
                if cited in tracked:
                    continue
                exists = (REPO / cited).exists()
                why = ("exists on disk but is not tracked -- `git add -A` skips it, so it will "
                       "not reach anyone who clones this" if exists else "does not exist")
                failures.append(f"{rel}:{lineno}: cites {cited}, which {why}")

    if failures:
        print("CITED ARTIFACT CHECK FAILED", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print("\nEither `git add -f` the artifact, or stop citing it. A report that names its\n"
              "evidence and ships without it is not reproducible by the person reading it.",
              file=sys.stderr)
        return 1

    print(f"cited artifacts: {checked} citation(s) across {len(reports)} tracked report(s), "
          f"every one tracked")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
