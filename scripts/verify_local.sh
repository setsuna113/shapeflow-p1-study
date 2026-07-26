#!/usr/bin/env bash
# Fast local gate: import every module, run tests, and scan for real credentials.
# Not a substitute for the launch gate on sjtu -- just enough to keep each commit clean.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=.venv/bin/python

# Deliberately NO PYTHONPATH=src fallback. The package is installed editable by
# scripts/dev_setup.sh, so "can this be imported the way the run host imports it" is a real
# signal here rather than something the harness papers over.
echo "== package is installed (not just on sys.path) =="
"$PY" - <<'PYEOF'
import pathlib, sys
try:
    import shapeflow_p1
except ModuleNotFoundError:
    sys.exit("shapeflow_p1 is not installed in .venv -- run scripts/dev_setup.sh")
print("shapeflow_p1 <-", pathlib.Path(shapeflow_p1.__file__).parent)
PYEOF

echo "== import every module =="
"$PY" - <<'PYEOF'
import importlib, pkgutil, shapeflow_p1
ok = 0
for m in pkgutil.walk_packages(shapeflow_p1.__path__, "shapeflow_p1."):
    importlib.import_module(m.name); ok += 1
print(f"imported {ok} submodules cleanly")
PYEOF

echo "== shell syntax =="
# The launch gate, the host installer and the supervisor are shell. A syntax error in any of
# them is only discovered on the run host, as root, mid-launch.
for script in scripts/*.sh; do
  bash -n "$script" || { echo "SHELL SYNTAX FAILED: $script"; exit 1; }
done

echo "== whitespace =="
git diff --check HEAD -- . || { echo "WHITESPACE ERRORS"; exit 1; }

echo "== lint (defect rules) =="
# Deliberately not the full rule set. The repository carries several hundred stylistic
# diagnostics (UP007 Optional -> | None, UP035, I001) whose mass rewrite right before a freeze
# would be a large untested diff for no correctness gain. These families are the ones that
# catch defects rather than style, and they pass today -- so this gate is real and enforced,
# instead of aspirational and skipped. Run `ruff check .` for the full picture.
"$PY" -m ruff check . --select E9,F63,F7,F82,F811,F841,B006,B023,S102,S307,S608 \
  || { echo "LINT FAILED"; exit 1; }

echo "== unit + property tests =="
"$PY" -m pytest tests -q -p no:cacheprovider

echo "== secret scan (real credentials must not appear; FAKE placeholders allowed) =="
# Mirrors .gitleaks.toml patterns. Excludes gitignored dirs and the __pycache__.
hits=$(grep -rEn 'tvly-[A-Za-z0-9_-]{12,}|sk-[A-Za-z0-9_-]{12,}|(EXA_API_KEY|x-api-key)["'"'"'[:space:]:=]{1,4}[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}' \
        --include='*.py' --include='*.md' --include='*.toml' --include='*.json' \
        --include='*.yaml' --include='*.sh' \
        src tests schemas scripts configs protocol ./*.md ./*.toml 2>/dev/null \
      | grep -vE '(tvly|sk|exa)-[A-Z0-9-]*FAKE[A-Z0-9-]*' || true)
if [ -n "$hits" ]; then
  echo "SECRET SCAN FAILED:"; echo "$hits"; exit 1
fi
echo "secret scan clean"

echo "== no fabricated user decisions =="
# Protocol v0.1 is USER_EXPLICIT_AUTO_LAUNCH. A coding agent may materialize and hash the
# protocol's values; it may never invent a different launch policy and write it in as fact.
# This guard exists because exactly that happened once (commit b655164 shipped
# USER_EXPLICIT_GATE_GREEN_THEN_PAUSE as a default).
#
# Scope is the production surface only. tests/unit/test_config_freeze.py names the forged
# string on purpose -- it asserts that build_launch_approval and verify_launch_approval REJECT
# it -- so scanning tests here would make the regression guard delete its own proof.
forged=$(grep -rEn 'GATE_GREEN_THEN_PAUSE|requires_human_launch|AWAITING_HUMAN_GO_AHEAD' \
          src configs schemas protocol scripts/bootstrap_and_run.sh 2>/dev/null || true)
if [ -n "$forged" ]; then
  echo "FABRICATED DECISION FOUND (protocol v0.1 mode is USER_EXPLICIT_AUTO_LAUNCH):"
  echo "$forged"; exit 1
fi
echo "no fabricated decisions"
echo "ALL LOCAL CHECKS PASSED"
