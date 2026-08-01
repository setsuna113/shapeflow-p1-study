#!/usr/bin/env python3
"""Static half of the observability enforcement.

`ClosedFeatureMap` raises on an unregistered key, but only on paths that execute. A decision
branch taken once a month under load is exactly the branch a test suite misses, so this walks the
AST of every module allowed to make broker decisions and rejects any feature-shaped string
literal that is not in the registry.

Scope is deliberately narrow. It scans the modules that decide -- broker, predictor, cost model
-- and not the whole tree, because the point is to constrain what a *decision* may read. Widening
it to every module would produce false positives on documentation and test fixtures and would
train people to ignore the check, which is worse than not having it.

Exit 0 clean, 1 on a violation. Wired into `scripts/verify_local.sh`.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))

from shapeflow.contracts.observability import OBSERVABLE  # noqa: E402

#: Modules whose string literals are constrained. Anything that decides, predicts or prices.
SCANNED = (
    "src/shapeflow/broker",
    "src/shapeflow/predictor",
    "src/shapeflow/work",
)

#: Roots from SCANNED that do not exist yet, named one by one. A missing root used to be skipped
#: by a bare ``if not root.exists(): continue``, so this gate scanned ``broker`` alone and printed
#: the same green line it prints at full coverage -- two thirds of its declared scope absent and
#: nothing said so. Silence about what was not checked is the failure mode a CI gate exists to
#: prevent, so an unexpected absence is now an error and an expected one has to be written down
#: here, where deleting a live package fails the build instead of quietly shrinking the check.
NOT_BUILT_YET = {
    "src/shapeflow/predictor",  # Phase 1: the cost model that prices a form at the tick.
    "src/shapeflow/work",       # Phase 1: W components and the microbench that fits alpha/beta.
}

#: A literal is treated as a feature reference if it looks like one. Prefixes come from the
#: registry itself, so adding a feature family to the contract extends this automatically.
PREFIXES = tuple(sorted({name.split(".", 1)[0] + "." for name in OBSERVABLE}))

#: The registry module names every feature by definition, and the contract test names the
#: banned ones on purpose.
EXEMPT_FILES = {"observability.py"}


def _is_feature_shaped(value: str) -> bool:
    return value.startswith(PREFIXES) and " " not in value


def violations_in(path: Path) -> list[tuple[int, str]]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if _is_feature_shaped(node.value) and node.value not in OBSERVABLE:
                found.append((node.lineno, node.value))
    return found


def main() -> int:
    scanned = 0
    failures: list[str] = []
    pending: list[str] = []
    for rel in SCANNED:
        root = REPO / rel
        if not root.exists():
            if rel in NOT_BUILT_YET:
                pending.append(rel)
                continue
            failures.append(
                f"{rel} is in SCANNED, is not listed in NOT_BUILT_YET, and does not exist. "
                "Either it was deleted -- in which case this gate just stopped covering it -- or "
                "the path is wrong and it never covered it at all.")
            continue
        if rel in NOT_BUILT_YET:
            failures.append(
                f"{rel} exists but is still listed in NOT_BUILT_YET, so a reader is told it is "
                "unbuilt while its literals go unscanned. Remove it from NOT_BUILT_YET.")
            continue
        for path in sorted(root.rglob("*.py")):
            if path.name in EXEMPT_FILES or "__pycache__" in path.parts:
                continue
            scanned += 1
            for lineno, value in violations_in(path):
                failures.append(
                    f"{path.relative_to(REPO)}:{lineno}: {value!r} is not in the observability "
                    "contract")

    if failures:
        print("OBSERVABILITY CHECK FAILED", file=sys.stderr)
        for line in failures:
            print(f"  {line}", file=sys.stderr)
        print(
            "\nA broker decision may only read features the broker could have known at the tick.\n"
            "Add the feature to OBSERVABILITY_v1.md and contracts/observability.py first -- which\n"
            "changes contracts_sha and requires re-approval, deliberately.",
            file=sys.stderr,
        )
        return 1

    # The pending roots are printed on the success line, not swallowed. A gate that reports what
    # it did not look at is the difference between "clean" and "clean over two thirds of itself".
    not_yet = f"; not built yet: {', '.join(sorted(pending))}" if pending else ""
    print(f"observability: {scanned} module(s) scanned across "
          f"{len(SCANNED) - len(pending)} of {len(SCANNED)} root(s), "
          f"{len(OBSERVABLE)} registered features, no unregistered feature reference{not_yet}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
