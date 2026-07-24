"""Canonical JSON must be stable and must refuse ambiguous input."""

from __future__ import annotations

import math

import pytest

from shapeflow_p1.canonical import CanonicalizationError, canonical_json


def test_key_order_does_not_change_bytes():
    a = {"z": 1, "a": {"n": 2, "m": 3}}
    b = {"a": {"m": 3, "n": 2}, "z": 1}
    assert canonical_json(a) == canonical_json(b)


def test_no_insignificant_whitespace():
    assert canonical_json({"a": [1, 2]}) == b'{"a":[1,2]}'


def test_non_ascii_survives_as_utf8():
    # Escaping would make the bytes depend on the encoder's policy, not the content.
    assert canonical_json({"q": "研究"}) == '{"q":"研究"}'.encode("utf-8")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -math.inf])
def test_rejects_non_finite_floats(bad):
    with pytest.raises(CanonicalizationError, match="non-finite"):
        canonical_json({"score": bad})


def test_rejects_non_string_keys_because_they_collide():
    # This is the whole reason for the check: json would render both as {"1":"a"},
    # so two distinct objects would hash to one ID.
    with pytest.raises(CanonicalizationError, match="not str"):
        canonical_json({1: "a"})


@pytest.mark.parametrize("bad", [(1, 2), {1, 2}, frozenset({1}), b"x", bytearray(b"x")])
def test_rejects_ambiguous_containers_and_bytes(bad):
    with pytest.raises(CanonicalizationError):
        canonical_json({"v": bad})


def test_error_names_the_path():
    with pytest.raises(CanonicalizationError, match=r"events\[1\]\.usage"):
        canonical_json({"events": [{}, {"usage": float("nan")}]})


def test_rejects_unencodable_surrogate():
    with pytest.raises(CanonicalizationError, match="surrogate"):
        canonical_json({"s": "\ud800"})


def test_rejects_arbitrary_objects():
    class Thing:
        pass

    with pytest.raises(CanonicalizationError, match="not canonically serializable"):
        canonical_json({"o": Thing()})
