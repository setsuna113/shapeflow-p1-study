#!/usr/bin/env bash
# Rebuild both ODR trees from the pinned submodule. Deterministic: the patched tree's hash is
# pinned in patches/patched_tree.sha256 and asserted here, so "the patch applied" is a fact
# rather than an exit status.
#
# Two trees, not one. Parity needs to run vendor and patched in separate interpreters -- they
# share the module name `open_deep_research`, so a single process would hand the second run the
# first one's modules and compare a tree against itself.
set -euo pipefail
REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"
PIN="408da442a661ea5e40a6163329f82e3f22628949"

GOT="$(git -C vendor/open_deep_research rev-parse HEAD)"
[ "$GOT" = "$PIN" ] || { echo "vendor at $GOT, expected $PIN" >&2; exit 1; }

rm -rf .build/odr-pristine .build/open_deep_research-patched
mkdir -p .build/odr-pristine .build/open_deep_research-patched
git -C vendor/open_deep_research archive HEAD | tar -x -C .build/odr-pristine
git -C vendor/open_deep_research archive HEAD | tar -x -C .build/open_deep_research-patched
git apply --directory=.build/open_deep_research-patched patches/odr_p1_hooks.patch

ACTUAL="$(python -c "
from pathlib import Path; import sys
sys.path.insert(0, 'src')
from shapeflow_p1.treehash import tree_sha256
print(tree_sha256(Path('.build/open_deep_research-patched/src/open_deep_research')))
")"
EXPECTED="$(tr -d ' \n' < patches/patched_tree.sha256)"
[ "$ACTUAL" = "$EXPECTED" ] || {
  echo "patched tree hashes to $ACTUAL, expected $EXPECTED" >&2; exit 1; }
echo "materialized; patched tree $ACTUAL"
