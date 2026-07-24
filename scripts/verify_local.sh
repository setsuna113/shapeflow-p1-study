#!/usr/bin/env bash
# Fast local gate: import every module, run tests, and scan for real credentials.
# Not a substitute for the launch gate on sjtu -- just enough to keep each commit clean.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

echo "== import every module =="
PYTHONPATH=src "$PY" - <<'PYEOF'
import importlib, pkgutil, shapeflow_p1
ok = 0
for m in pkgutil.walk_packages(shapeflow_p1.__path__, "shapeflow_p1."):
    importlib.import_module(m.name); ok += 1
print(f"imported {ok} submodules cleanly")
PYEOF

echo "== unit + property tests =="
PYTHONPATH=src "$PY" -m pytest tests -q -p no:cacheprovider

echo "== secret scan (real credentials must not appear; FAKE placeholders allowed) =="
# Mirrors .gitleaks.toml patterns. Excludes gitignored dirs and the __pycache__.
hits=$(grep -rEn 'tvly-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,}' \
        --include='*.py' --include='*.md' --include='*.toml' --include='*.json' \
        --include='*.yaml' --include='*.sh' \
        src tests schemas scripts configs protocol ./*.md ./*.toml 2>/dev/null \
      | grep -v 'FAKEFAKEFAKE' || true)
if [ -n "$hits" ]; then
  echo "SECRET SCAN FAILED:"; echo "$hits"; exit 1
fi
echo "secret scan clean"
echo "ALL LOCAL CHECKS PASSED"
