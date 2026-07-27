"""Every vendor knob we move is declared, and an undeclared one fails.

``summarization_model_max_tokens`` sat at 1024 against a vendor default of 8192 for the whole
design. Nothing recorded why, and nothing checked. Vendor's summariser falls back to the *raw
page* when its completion is cut off, so the value silently degraded 263 of 1839 P0 summaries
(14.3%) into whole pages published as "compressed" notes -- worst on the largest pages, which is
precisely the high-evidence-volume stratum where P1 is supposed to win. P1 never reaches that
code path, so the handicap fell on the baseline alone.

The defect was not the number. It was that a knob could differ from vendor with no written
reason and no check, so nobody had to notice it bound only one arm.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs" / "week1.yaml"
VENDOR = ROOT / ".build" / "odr-pristine" / "src" / "open_deep_research" / "configuration.py"

# Knobs we set that are not vendor Configuration fields; they are protocol of our own.
_NOT_VENDOR_FIELDS = {"search_api", "summarization_timeout_seconds"}


def _vendor_defaults() -> dict[str, object]:
    """Read the pinned vendor's declared defaults straight from its source.

    From the pristine tree rather than the patched one: the patch is ours, so reading defaults
    through it would let a patched default certify itself.
    """
    text = VENDOR.read_text(encoding="utf-8")
    defaults: dict[str, object] = {}
    for match in re.finditer(
        r"^\s{4}(\w+):\s*(int|str|bool|float)\s*=\s*Field\(\s*\n?\s*default=([^,\n]+)",
        text,
        re.M,
    ):
        name, kind, raw = match.group(1), match.group(2), match.group(3).strip()
        if kind == "int":
            defaults[name] = int(raw)
        elif kind == "bool":
            defaults[name] = raw == "True"
        elif kind == "str":
            defaults[name] = raw.strip('"').strip("'")
    return defaults


def _ours() -> dict:
    return yaml.safe_load(CONFIG.read_text(encoding="utf-8"))


def test_the_vendor_source_is_readable_and_declares_the_knobs_we_set():
    vendor = _vendor_defaults()
    assert vendor, "could not read vendor defaults; run scripts/materialize_vendor.sh"
    unknown = {
        knob for knob in _ours()["odr"]
        if knob not in vendor and knob not in _NOT_VENDOR_FIELDS
    }
    assert not unknown, f"we set knobs the pinned vendor does not declare: {sorted(unknown)}"


def test_every_deviation_from_vendor_is_declared_with_a_reason():
    """No silent deviation, in either direction.

    Both halves matter. An undeclared deviation is the 1024 bug. A declared deviation that no
    longer deviates is a stale justification that makes the config look more considered than it
    is.
    """
    ours = _ours()
    vendor = _vendor_defaults()
    declared = {entry["knob"]: entry for entry in ours["odr_deviations"]}

    actual = {
        knob: value
        for knob, value in ours["odr"].items()
        if knob in vendor and vendor[knob] != value
    }
    # summarization_timeout_seconds is vendor behaviour hard-coded in utils.py rather than a
    # Configuration field, so it cannot be read from the model; it is declared explicitly.
    actual["summarization_timeout_seconds"] = ours["odr"]["summarization_timeout_seconds"]

    assert set(declared) == set(actual), (
        f"undeclared deviations {sorted(set(actual) - set(declared))}; "
        f"stale declarations {sorted(set(declared) - set(actual))}"
    )
    for knob, entry in declared.items():
        assert entry["ours"] == actual[knob], f"{knob}: declaration does not match the config"
        assert entry["binds"] in {"BOTH_ARMS", "P0_ONLY", "P1_ONLY"}, knob
        assert len(str(entry["reason"]).split()) >= 12, f"{knob}: reason is not a reason"


def test_the_summarization_cap_is_back_at_the_vendor_default():
    """The one knob whose deviation was an asymmetric handicap on the baseline."""
    ours = _ours()
    assert ours["odr"]["summarization_model_max_tokens"] == _vendor_defaults()[
        "summarization_model_max_tokens"
    ]
    assert "summarization_model_max_tokens" not in {
        entry["knob"] for entry in ours["odr_deviations"]
    }


def test_only_one_knob_binds_a_single_arm_and_it_is_the_documented_one():
    """A knob that binds one arm is a confound unless the asymmetry is vendor's own.

    ``summarization_timeout_seconds`` qualifies: vendor's page summariser only runs in P0's path
    at all, because P1's hook returns from ``defer_page_batch`` before it is awaited. Any *other*
    single-arm knob would be ours, and would be a confound.
    """
    one_sided = [
        entry["knob"]
        for entry in _ours()["odr_deviations"]
        if entry["binds"] != "BOTH_ARMS"
    ]
    assert one_sided == ["summarization_timeout_seconds"]
