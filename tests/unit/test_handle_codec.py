"""The publication handle: one grammar, a proven bound, and no silent fallback.

Three separate defects produced the same outcome -- zero P1 output in every cell of every H arm,
with nothing failing loudly:

1. the encoding cost 9-10 tokens against a frozen cap of 8, so view construction raised before
   the first selector call;
2. the trajectory validator carried its own three-group regex while the producer emitted four
   groups, so fixing (1) alone would have moved the failure one step downstream;
3. both failures degraded to a P0 fallback that is a legitimate part of the ITT design, so a
   100% failure rate looked exactly like a run.

These tests pin each one.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

from shapeflow_p1.p1 import handle_codec
from shapeflow_p1.p1.handle_codec import (
    FROZEN_RADICES,
    HandleDomainError,
    HandleFormatError,
    HandleRadices,
)

CONFIG = Path(__file__).resolve().parents[2] / "configs" / "week1.yaml"


# --- the encoding is a bijection over the frozen domain --------------------------------------


def test_small_domain_is_enumerated_collision_free_and_round_trips():
    """Exhaustive over a reduced domain: every tuple encodes uniquely and decodes back.

    Reduced rather than frozen so the *logic* is proven by enumeration in the unit suite; the
    frozen domain's 26 million handles are enumerated against the real tokenizer by
    ``handle_proof`` at preflight, where the tokenizer that decides the cost actually exists.
    """
    radices = HandleRadices(
        researcher_iteration=3,
        child_ordinal=2,
        assistant_turn=4,
        toolset=3,
        evidence=7,
    )
    seen: dict[str, tuple] = {}
    for structural in range(radices.structural):
        for turn in range(radices.assistant_turn):
            for toolset in range(radices.toolset):
                for evidence in range(1, radices.evidence + 1):
                    coordinates = (structural, turn, toolset, evidence)
                    handle = handle_codec.encode(*coordinates, radices=radices)
                    assert handle not in seen, (
                        f"{handle} names both {seen.get(handle)} and {coordinates}"
                    )
                    seen[handle] = coordinates
                    assert handle_codec.decode(handle, radices=radices) == coordinates
                    assert handle_codec.validate(handle, radices=radices)
    assert len(seen) == radices.capacity


def test_handles_are_lowercase_letters_only():
    """No digits and no separators: both are what made the predecessor expensive."""
    for evidence in (1, 2, 26, 27, 700, 24000):
        handle = handle_codec.encode(0, 0, 0, evidence)
        assert re.fullmatch(r"H[a-z]+", handle), handle


def test_absent_researcher_coordinate_is_distinct_from_zero_zero():
    """The predecessor dropped the field when absent and changed the handle's arity.

    That arity change is precisely what the validator's grammar then failed to match.
    """
    absent = handle_codec.structural_ordinal(None)
    present = handle_codec.structural_ordinal((0, 0))
    assert absent == 0 and present == 1
    assert handle_codec.encode(absent, 0, 0, 1) != handle_codec.encode(present, 0, 0, 1)


def test_structural_ordinal_packs_without_the_cantor_waste():
    """Mixed radix over the real bounds, not a pairing function over an unbounded plane.

    ``Cantor(3, 0)`` is 6, so four reachable nested-researcher positions used to consume seven
    structural slots; every wasted slot costs a letter in the typical handle.
    """
    values = [
        handle_codec.structural_ordinal((iteration, child))
        for iteration in range(FROZEN_RADICES.researcher_iteration)
        for child in range(FROZEN_RADICES.child_ordinal)
    ]
    assert values == list(range(1, FROZEN_RADICES.structural))


# --- out-of-domain fails closed, it does not widen or fall back -------------------------------


@pytest.mark.parametrize(
    "coordinates",
    [
        (FROZEN_RADICES.structural, 0, 0, 1),
        (0, FROZEN_RADICES.assistant_turn, 0, 1),
        (0, 0, FROZEN_RADICES.toolset, 1),
        (0, 0, 0, FROZEN_RADICES.evidence + 1),
        (0, 0, 0, 0),
        (-1, 0, 0, 1),
    ],
)
def test_out_of_domain_coordinates_raise(coordinates):
    with pytest.raises(HandleDomainError):
        handle_codec.encode(*coordinates)


def test_researcher_coordinate_outside_its_radix_raises():
    with pytest.raises(HandleDomainError):
        handle_codec.structural_ordinal((FROZEN_RADICES.researcher_iteration, 0))
    with pytest.raises(HandleDomainError):
        handle_codec.structural_ordinal((0, FROZEN_RADICES.child_ordinal))


@pytest.mark.parametrize("handle", ["", "H", "h1", "H3_1_0_1", "HA", "H1", "Ha_b", "X"])
def test_malformed_handles_are_rejected_not_coerced(handle):
    assert handle_codec.validate(handle) is False
    with pytest.raises((HandleFormatError, HandleDomainError)):
        handle_codec.decode(handle)


def test_the_predecessor_format_no_longer_validates():
    """The exact handles that appeared 2,067 times in the invalidated canary."""
    for legacy in ("H3_1_0_1", "H3_1_0_mq", "H3_1_0_1481", "H0_0_1"):
        assert handle_codec.validate(legacy) is False


def test_a_handle_beyond_the_domain_decodes_to_nothing():
    """Bijective base 26 has no unreachable-but-parseable region inside the domain."""
    beyond = "H" + "z" * (handle_codec.max_encoded_width() + 1)
    assert handle_codec.validate(beyond) is False


# --- the domain the protocol froze is the domain the code uses --------------------------------


def test_config_and_codec_agree_on_the_frozen_domain():
    """Two registries that can disagree eventually do.

    The op-class registries drifted the same way -- a class listed in one place and absent from
    another -- and the disagreement was invisible until an arm was silently refused. The handle
    domain is protocol-hashed in week1.yaml and enforced in the codec, so they are asserted equal
    rather than kept equal by convention.
    """
    declared = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["p1_publication_handle"]
    assert declared["format_version"] == handle_codec.FORMAT_VERSION
    assert (
        declared["max_publication_handle_tokens"]
        == handle_codec.MAX_PUBLICATION_HANDLE_TOKENS
    )
    radices = declared["radices"]
    assert radices == {
        "researcher_iteration": FROZEN_RADICES.researcher_iteration,
        "child_ordinal": FROZEN_RADICES.child_ordinal,
        "assistant_turn": FROZEN_RADICES.assistant_turn,
        "toolset": FROZEN_RADICES.toolset,
        "evidence": FROZEN_RADICES.evidence,
    }


def test_frozen_radices_cover_the_configured_graph_bounds():
    """A radix that the vendor configuration can exceed is a cell failure waiting to happen."""
    odr = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))["odr"]
    assert FROZEN_RADICES.researcher_iteration > odr["max_researcher_iterations"]
    assert FROZEN_RADICES.child_ordinal > odr["max_concurrent_research_units"]
    assert FROZEN_RADICES.assistant_turn > odr["max_react_tool_calls"]


def test_frozen_evidence_radix_covers_the_worst_frozen_batch():
    """23,782 spans is the frozen corpus's worst atomic publication batch, measured.

    The eight largest pages (the acquisition config takes eight results per query) chunked by
    markdown_structure_v1 at 320 tokens. Median page is 97 spans and p95 is 656, so this bound
    binds three pages in the whole corpus -- but it must bind them without silently truncating.
    """
    assert FROZEN_RADICES.evidence > 23_782


def test_the_frozen_domain_stays_within_six_letters():
    """Width is what the token cost tracks; the proof at preflight costs the exact strings."""
    assert handle_codec.max_encoded_width() == 6


# --- one grammar, everywhere --------------------------------------------------------------


def test_no_second_handle_grammar_exists_in_the_tree():
    """A regex for handles written next to a caller is how producer and validator diverged."""
    root = Path(__file__).resolve().parents[2]
    offenders = []
    for path in (root / "src").rglob("*.py"):
        if path.name in {"handle_codec.py", "handle_proof.py"}:
            continue
        text = path.read_text(encoding="utf-8")
        if re.search(r"""["']\^?H\[0-9a-z""", text) or "H[0-9a-z]+_" in text:
            offenders.append(str(path.relative_to(root)))
    assert not offenders, f"handle grammar duplicated in {offenders}"


# --- the proof mechanism itself ---------------------------------------------------------


def test_the_domain_proof_enumerates_and_reports_the_worst_case(tmp_path):
    """Proven by enumeration, not argued from the encoding's shape.

    The predecessor's cap held for most handles and failed for every one production actually
    emitted, so a sampled check would have passed while P1 was completely inert.
    """
    from shapeflow_p1.evidence.chunkers import WhitespaceTokenizer
    from shapeflow_p1.p1.handle_proof import HandleProofError, load_or_prove, prove_handle_domain

    radices = HandleRadices(
        researcher_iteration=2,
        child_ordinal=1,
        assistant_turn=2,
        toolset=2,
        evidence=5,
    )
    tokenizer = WhitespaceTokenizer()
    proof = prove_handle_domain(tokenizer, radices=radices)
    assert proof["capacity"] == radices.capacity
    assert sum(proof["token_histogram"].values()) == radices.capacity
    assert proof["max_tokens"] <= handle_codec.MAX_PUBLICATION_HANDLE_TOKENS

    # A cached proof is honoured only when format, radices, cap and tokenizer all still match.
    receipt = tmp_path / "proof.json"
    first = load_or_prove(tokenizer, receipt_path=receipt, radices=radices)
    assert receipt.exists()
    again = load_or_prove(tokenizer, receipt_path=receipt, radices=radices)
    assert again["content_sha256"] == first["content_sha256"]
    wider = HandleRadices(
        researcher_iteration=2, child_ordinal=1, assistant_turn=2, toolset=2, evidence=6
    )
    assert (
        load_or_prove(tokenizer, receipt_path=receipt, radices=wider)["capacity"]
        == wider.capacity
    )

    class _Expensive:
        """A tokenizer that charges more than the cap for every handle."""

        identity_sha256 = "e" * 64

        def encode_offsets(self, text):  # pragma: no cover - not used by the proof
            return [(i, i + 1) for i in range(len(text))]

        def count(self, text):
            return handle_codec.MAX_PUBLICATION_HANDLE_TOKENS + 1

    with pytest.raises(HandleProofError):
        prove_handle_domain(_Expensive(), radices=radices)
