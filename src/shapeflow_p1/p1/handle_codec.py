"""The one encoding of an H publication handle, owned in one place.

A published handle is the pointer P1 spends tokens on. Its cost is capped
(:data:`MAX_PUBLICATION_HANDLE_TOKENS`) because a bounded pointer is the form claim itself: a
handle that costs as much as the prose it replaces is not a cheaper representation of anything.

The first encoding spent that budget on punctuation. ``H3_1_0_1`` renders four structural
coordinates with underscores between them, and Qwen's tokenizer charges one token per
underscore, one per digit, and never merges a digit with a letter -- nine tokens for four small
numbers, against a cap of eight. Every H arm therefore failed view construction before its
first selector call and fell back to P0, silently, in all 146 canary cells.

Two properties of the tokenizer decide the encoding, both measured on the frozen
``tokenizer.json`` rather than assumed:

- digits cost one token each and never merge (``[H12345]`` is 7 tokens),
- runs of lowercase letters merge hard (``[Habcd]`` is 3 tokens, ``[Habcdef]`` is 3).

So the coordinates are packed into a single integer by mixed radix and rendered in *bijective*
base 26 -- lowercase only, no separators, no leading-zero ambiguity, and no fixed width, so the
common case (a page of ~100 chunks) costs three or four tokens and only the corpus's largest
pages reach the worst case.

The radices are frozen protocol, not a tuning knob. A coordinate outside them raises
:class:`HandleDomainError`, which the caller must surface as a cell failure: silently widening
the domain would let one handle mean two spans, and silently falling back to P0 is exactly the
failure that hid this bug the first time.

Producer, trajectory validator, preflight and evaluator all call :func:`encode` and
:func:`validate` here. The predecessor kept a second, independent grammar in the validator -- a
three-group regex that could not match the four-group handles the producer emitted -- so fixing
only the token cap would have moved the failure one step downstream instead of removing it.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "FORMAT_VERSION",
    "MAX_PUBLICATION_HANDLE_TOKENS",
    "HandleRadices",
    "FROZEN_RADICES",
    "HandleDomainError",
    "HandleFormatError",
    "encode",
    "decode",
    "validate",
    "structural_ordinal",
    "max_encoded_width",
    "iter_domain",
]

FORMAT_VERSION = "h_publication_ordinal_bijective_base26_v1"

#: The exact-token ceiling on one rendered handle, ``[H...]``, under the frozen tokenizer.
#: This is a protocol bound on P1's published pointer cost. Raising it would weaken the
#: bounded-pointer claim, so the encoding moves and this does not.
MAX_PUBLICATION_HANDLE_TOKENS = 8

_ALPHABET = "abcdefghijklmnopqrstuvwxyz"
_VALUE = {ch: index + 1 for index, ch in enumerate(_ALPHABET)}
_PREFIX = "H"


class HandleDomainError(ValueError):
    """A coordinate lies outside the frozen handle domain.

    Raised at publication time, before any model call. The caller must fail the cell: widening
    the domain in place would silently re-map handles that a previous batch already published.
    """


class HandleFormatError(ValueError):
    """A handle is not a well-formed structural ordinal in this format version."""


@dataclass(frozen=True)
class HandleRadices:
    """The frozen bound on each structural coordinate.

    ``evidence`` is the one that matters in practice: publication ordinals are allocated across
    every span of every page in one atomic publication batch, so the bound is the whole batch's
    chunk count, not one page's. It is derived from the frozen corpus, not from what a canary
    happened to reach.

    The nested-researcher position is two frozen radices rather than one, because the
    predecessor Cantor-paired ``(iteration, child)`` into a single integer and a pairing function
    that must stay injective over an unbounded plane wastes most of a bounded rectangle --
    ``Cantor(3, 0)`` is 6, so four reachable positions consumed seven slots. Mixed radix over the
    real bounds is exact, and every wasted slot here costs a letter in the *typical* handle.
    """

    researcher_iteration: int
    child_ordinal: int
    assistant_turn: int
    toolset: int
    evidence: int

    def __post_init__(self) -> None:
        for name in (
            "researcher_iteration",
            "child_ordinal",
            "assistant_turn",
            "toolset",
            "evidence",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"handle radix {name} must be a positive integer")

    @property
    def structural(self) -> int:
        """Positions the nested-researcher coordinate can take, plus one for "not nested"."""
        return 1 + self.researcher_iteration * self.child_ordinal

    @property
    def capacity(self) -> int:
        """How many distinct handles this domain can name."""
        return self.structural * self.assistant_turn * self.toolset * self.evidence

    def content(self) -> dict:
        return {
            "format_version": FORMAT_VERSION,
            "researcher_iteration": self.researcher_iteration,
            "child_ordinal": self.child_ordinal,
            "assistant_turn": self.assistant_turn,
            "toolset": self.toolset,
            "evidence": self.evidence,
        }


#: Frozen for this protocol round. Each bound is measured against the frozen world rather than
#: guessed, and each is deliberately larger than what the configuration can reach, because a
#: coordinate that overflows fails its cell:
#:
#: - ``researcher_iteration`` 8 against ``max_researcher_iterations: 3``.
#: - ``child_ordinal`` 2 against ``max_concurrent_research_units: 1``.
#: - ``assistant_turn`` 16 against ``max_react_tool_calls: 5`` plus structured-output retries;
#:   the canary's observed maximum was 3.
#: - ``toolset`` 4 for tool-call sets per assistant turn; the canary's observed maximum was 0.
#: - ``evidence`` 24000: the frozen corpus's worst atomic publication batch is the eight largest
#:   pages chunked by ``markdown_structure_v1`` at 320 tokens, which is 23,782 spans. The median
#:   page is 97 spans and p95 is 656, so this bound binds three pages in the whole corpus.
#:
#: Capacity is 26,112,000, whose bijective base-26 rendering never exceeds six letters. Every
#: six-letter body was enumerated against the frozen tokenizer to establish the cap; see
#: ``tests/unit/test_handle_codec.py``.
FROZEN_RADICES = HandleRadices(
    researcher_iteration=8,
    child_ordinal=2,
    assistant_turn=16,
    toolset=4,
    evidence=24000,
)


def structural_ordinal(
    researcher_coordinate: tuple[int, int] | None,
    *,
    radices: HandleRadices = FROZEN_RADICES,
) -> int:
    """Fold an optional ``(supervisor iteration, child ordinal)`` into one coordinate.

    Zero means "not inside a nested researcher". A present coordinate is packed by mixed radix
    and offset by one, so ``(0, 0)`` stays distinguishable from absence -- the predecessor
    dropped the field entirely when it was absent and produced handles with a different arity,
    which is what the validator's grammar then disagreed with.
    """
    if researcher_coordinate is None:
        return 0
    iteration, child_ordinal = researcher_coordinate
    for name, value, radix in (
        ("researcher iteration", iteration, radices.researcher_iteration),
        ("child ordinal", child_ordinal, radices.child_ordinal),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise HandleDomainError(f"{name} must be a non-negative integer")
        if value >= radix:
            raise HandleDomainError(
                f"{name} {value} exceeds the frozen domain (radix {radix})"
            )
    return 1 + iteration * radices.child_ordinal + child_ordinal


def _bijective_encode(value: int) -> str:
    """Render ``value >= 1`` in bijective base 26.

    Bijective rather than positional: every positive integer has exactly one representation and
    no representation has a leading-zero twin, so a variable-width handle is still uniquely
    decodable without a separator or a length prefix.
    """
    out: list[str] = []
    while value > 0:
        value, remainder = divmod(value - 1, 26)
        out.append(_ALPHABET[remainder])
    return "".join(reversed(out))


def _bijective_decode(text: str) -> int:
    value = 0
    for ch in text:
        digit = _VALUE.get(ch)
        if digit is None:
            raise HandleFormatError(
                f"handle body {text!r} contains {ch!r}, which is not a base-26 digit"
            )
        value = value * 26 + digit
    return value


def _pack(
    structural: int,
    assistant_turn: int,
    toolset: int,
    evidence: int,
    radices: HandleRadices,
) -> int:
    coordinates = (
        ("structural ordinal", structural, radices.structural, 0),
        ("assistant turn index", assistant_turn, radices.assistant_turn, 0),
        ("toolset ordinal", toolset, radices.toolset, 0),
        ("evidence ordinal", evidence, radices.evidence, 1),
    )
    for name, value, radix, low in coordinates:
        if isinstance(value, bool) or not isinstance(value, int):
            raise HandleDomainError(f"{name} must be an integer")
        if value < low:
            raise HandleDomainError(f"{name} {value} is below its minimum {low}")
        if value - low >= radix:
            raise HandleDomainError(
                f"{name} {value} exceeds the frozen domain (radix {radix}). The publication "
                "domain is protocol: widening it here would re-map handles a previous batch "
                "already published, so this cell fails instead."
            )
    index = structural
    index = index * radices.assistant_turn + assistant_turn
    index = index * radices.toolset + toolset
    index = index * radices.evidence + (evidence - 1)
    return index


def encode(
    structural: int,
    assistant_turn: int,
    toolset: int,
    evidence: int,
    *,
    radices: HandleRadices = FROZEN_RADICES,
) -> str:
    """The handle for one span's structural position. ``evidence`` is 1-based."""
    index = _pack(structural, assistant_turn, toolset, evidence, radices)
    return _PREFIX + _bijective_encode(index + 1)


def decode(
    handle: str,
    *,
    radices: HandleRadices = FROZEN_RADICES,
) -> tuple[int, int, int, int]:
    """Recover ``(structural, assistant_turn, toolset, evidence)``; the inverse of :func:`encode`."""
    if not isinstance(handle, str) or not handle.startswith(_PREFIX) or len(handle) < 2:
        raise HandleFormatError(
            f"{handle!r} is not an H publication handle in {FORMAT_VERSION}"
        )
    index = _bijective_decode(handle[len(_PREFIX):]) - 1
    if index < 0 or index >= radices.capacity:
        raise HandleFormatError(
            f"{handle!r} decodes outside the frozen handle domain"
        )
    index, evidence = divmod(index, radices.evidence)
    index, toolset = divmod(index, radices.toolset)
    structural, assistant_turn = divmod(index, radices.assistant_turn)
    return structural, assistant_turn, toolset, evidence + 1


def validate(handle: str, *, radices: HandleRadices = FROZEN_RADICES) -> bool:
    """True when ``handle`` is a structural ordinal this format can have produced.

    The trajectory validator, preflight and the evaluator all ask this question, and they must
    all ask it here. A second grammar written next to one of those callers is how the producer
    and the validator came to disagree about a handle's arity.
    """
    try:
        coordinates = decode(handle, radices=radices)
    except (HandleFormatError, HandleDomainError):
        return False
    return encode(*coordinates, radices=radices) == handle


def max_encoded_width(radices: HandleRadices = FROZEN_RADICES) -> int:
    """Length of the longest handle body this domain can produce."""
    return len(_bijective_encode(radices.capacity))


def iter_domain(radices: HandleRadices = FROZEN_RADICES):
    """Every handle the domain can produce, in packed order.

    Used by the exhaustive proof: the token cap is a claim about *all* reachable handles, and a
    sampled check of a cap that is already known to have been violated in production is not
    evidence.
    """
    for index in range(radices.capacity):
        yield _PREFIX + _bijective_encode(index + 1)
