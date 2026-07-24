#!/usr/bin/env bash
# Minimal DEV environment for pure-Python unit/property tests on the authoring host.
# This is NOT the campaign environment: the frozen, ODR-including env is built by
# bootstrap on sjtu via `uv sync --frozen` after the patched vendor is materialized.
# Here we only need enough to exercise the non-GPU modules.
set -euo pipefail
cd "$(dirname "$0")/.."

uv venv --python 3.12 .venv >/dev/null 2>&1 || true
# Dependencies actually imported by the pure-Python modules and their tests.
uv pip install --python .venv/bin/python -q \
  pytest pytest-asyncio hypothesis jsonschema pydantic PyYAML orjson httpx tenacity zstandard \
  numpy scipy

echo "dev venv ready: $(.venv/bin/python --version)"
