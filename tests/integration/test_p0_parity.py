"""P0 parity: the patched graph must be indistinguishable from vendor when nothing is bound.

This runs the **real** researcher subgraph, twice, in two subprocesses whose ``PYTHONPATH``
points at two different ODR source trees. The isolation is not decoration: both trees call
themselves ``open_deep_research``, so importing one after the other in a single interpreter
would silently hand the second run the first one's modules and produce a "parity" result that
compares a tree against itself.

Two gates, and the second is the one that earns its keep:

- **pristine vendor vs patched hooks-off** proves the patch is inert when unbound.
- **patched hooks-off vs patched with an explicit P0 strategy** proves the new code path --
  defer, checkpoint, reduce, refill -- introduces no difference of its own. The first gate
  cannot see that, because with hooks off the new path never executes.

Parity failure blocks all GPU screening. An effect measured against a baseline that is not
vendor is an effect confounded with our own harness.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]
PROBE = REPO / "tests" / "integration" / "parity_probe.py"
PRISTINE = REPO / ".build" / "odr-pristine" / "src"
PATCHED = REPO / ".build" / "open_deep_research-patched" / "src"

SCENARIOS = [
    "single_search",
    "multi_query_one_call",
    "two_search_siblings",
    "mixed_search_and_think",
    "empty_raw_content",
    "duplicate_url",
    "no_results",
    "research_complete",
    "no_tool_call",
    "max_react",
    "tool_exception",
    "summarization_timeout",
]

pytestmark = pytest.mark.skipif(
    not (PRISTINE.exists() and PATCHED.exists()),
    reason="ODR trees not materialized; run scripts/materialize_vendor.sh",
)


def _trace(odr_root: Path, scenario: str, strategy: str | None = None) -> dict:
    env = dict(os.environ)
    env["PYTHONPATH"] = f"{odr_root}:{REPO / 'src'}"
    env["PYTHONHASHSEED"] = "0"
    env["TZ"] = "UTC"
    cmd = [sys.executable, str(PROBE), scenario]
    if strategy:
        cmd += ["--strategy", strategy]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=300, cwd=REPO)
    assert "<<<TRACE>>>" in proc.stdout, (
        f"probe produced no trace for {scenario} under {odr_root}\n"
        f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-4000:]}"
    )
    return json.loads(proc.stdout.split("<<<TRACE>>>", 1)[1])


def _comparable(trace: dict) -> dict:
    """Everything except which tree produced it."""
    return {k: v for k, v in trace.items() if k not in ("odr_root", "strategy")}


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_pristine_vendor_matches_patched_hooks_off(scenario):
    vendor = _trace(PRISTINE, scenario)
    patched = _trace(PATCHED, scenario)
    # The probe reports which tree it loaded; if these were equal the isolation failed and the
    # comparison below would be vacuous.
    assert vendor["odr_root"] != patched["odr_root"]
    assert str(PRISTINE) in vendor["odr_root"]
    assert str(PATCHED) in patched["odr_root"]

    v, p = _comparable(vendor), _comparable(patched)
    assert v["model_requests"] == p["model_requests"], (
        f"{scenario}: model requests differ -- prompt bytes, tool signature or sampling changed"
    )
    assert v["publish_batches"] == p["publish_batches"], (
        f"{scenario}: publish batches differ -- bytes, batching, goto or state update changed"
    )
    assert v["exceptions"] == p["exceptions"], f"{scenario}: exception behaviour differs"
    assert v["final_report_sha256"] == p["final_report_sha256"], f"{scenario}: output differs"
    assert v["raw_notes_sha256"] == p["raw_notes_sha256"], f"{scenario}: raw_notes differ"
    assert v == p, f"{scenario}: traces differ"


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_patched_hooks_off_matches_explicit_p0_strategy(scenario):
    """The deferred/checkpoint/reduce/refill path must introduce no difference of its own.

    Gate A cannot see this: with hooks off, that code never executes. Here it does -- the tool
    defers, the batch is checkpointed and reduced, and vendor's own closures are replayed --
    and the result still has to be vendor's, down to the batch boundary.
    """
    hooks_off = _trace(PATCHED, scenario)
    explicit_p0 = _trace(PATCHED, scenario, strategy="p0")
    a, b = _comparable(hooks_off), _comparable(explicit_p0)
    assert a["publish_batches"] == b["publish_batches"], (
        f"{scenario}: the P1 code path changed what or how the batch was published"
    )
    assert a["exceptions"] == b["exceptions"], f"{scenario}: exception behaviour differs"
    assert a["final_report_sha256"] == b["final_report_sha256"], f"{scenario}: output differs"
    assert a["raw_notes_sha256"] == b["raw_notes_sha256"], f"{scenario}: raw_notes differ"
    # Model requests are compared as a multiset: the P0 strategy issues vendor's summarization
    # calls from its own gather, so the interleaving with the researcher's calls can differ
    # while every request itself is identical. What must not differ is which calls were made.
    assert sorted(map(str, a["model_requests"])) == sorted(map(str, b["model_requests"])), (
        f"{scenario}: model requests differ"
    )


def test_the_two_trees_are_actually_different():
    """Guard against the gate passing because the patch was never applied."""
    from shapeflow_p1.treehash import tree_sha256

    a = tree_sha256(PRISTINE / "open_deep_research")
    b = tree_sha256(PATCHED / "open_deep_research")
    assert a != b, "pristine and patched trees are identical; the patch is not applied"


def test_the_installed_package_is_the_patched_tree():
    """The bytes the campaign will run must be the bytes the patch produced.

    ODR installs as a copy and is a namespace package, so `__file__` is None and a path check
    is impossible. Comparing tree hashes is both possible and stronger: it proves the running
    bytes are the patched bytes rather than that some path resolves somewhere.
    """
    from shapeflow_p1.treehash import tree_sha256

    site = Path(sys.prefix) / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" \
        / "site-packages" / "open_deep_research"
    if not site.exists():
        pytest.skip("open_deep_research not installed in this interpreter")
    assert tree_sha256(site) == tree_sha256(PATCHED / "open_deep_research")
