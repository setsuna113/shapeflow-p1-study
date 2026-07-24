#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
echo "status (empty = clean):"
git status --short
echo "commits: $(git rev-list --count HEAD)"
echo "test functions: $(grep -rho 'def test_[a-zA-Z0-9_]*' tests | sort -u | wc -l)"
echo "src python lines: $(find src -name '*.py' -exec cat {} + | wc -l)"
echo "src modules: $(find src -name '*.py' | wc -l)"
