"""Redaction. These fakes are allowlisted in .gitleaks.toml so proving redaction
works does not itself trip the secret scan."""

from __future__ import annotations

import io
import logging

import pytest

from shapeflow_p1.secrets import RedactingFormatter, SecretRedactor

FAKE_TAVILY = "tvly-FAKEFAKEFAKEFAKEFAKE"
FAKE_DEEPSEEK = "sk-FAKEFAKEFAKEFAKEFAKE"


def test_registered_secret_is_removed_whatever_its_shape():
    r = SecretRedactor()
    # A credential with no recognizable prefix: only exact registration can catch it.
    r.register("Zx91ppQvNn02bLkeRt77", label="tavily")
    out = r.redact("connecting with key=Zx91ppQvNn02bLkeRt77 ...")
    assert "Zx91ppQvNn02bLkeRt77" not in out
    assert "[REDACTED:tavily:" in out


def test_unregistered_key_still_caught_by_shape():
    r = SecretRedactor()
    assert FAKE_TAVILY not in r.redact(f"tavily said {FAKE_TAVILY}")
    assert FAKE_DEEPSEEK not in r.redact(f"judge key {FAKE_DEEPSEEK}")


def test_bearer_header_keeps_prefix_but_drops_credential():
    r = SecretRedactor()
    out = r.redact("Authorization: Bearer abcdef0123456789abcdef")
    assert "abcdef0123456789abcdef" not in out
    # The header name survives so an incident log still says which header leaked.
    assert "Authorization: Bearer" in out


def test_query_string_credential_is_redacted():
    r = SecretRedactor()
    out = r.redact("GET https://api.example.com/s?q=x&api_key=abcdef0123456789")
    assert "abcdef0123456789" not in out
    assert "q=x" in out


def test_traceback_is_redacted_because_that_is_where_keys_escape():
    r = SecretRedactor()
    r.register(FAKE_TAVILY, label="tavily")

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(RedactingFormatter(logging.Formatter("%(message)s"), r))
    logger = logging.getLogger("redaction-traceback-test")
    logger.handlers = [handler]
    logger.propagate = False
    logger.setLevel(logging.ERROR)

    try:
        # Exactly the real failure mode: a client raises with the full request URL.
        raise RuntimeError(f"POST https://api.tavily.com/search?key={FAKE_TAVILY} failed")
    except RuntimeError:
        logger.exception("provider call failed")

    rendered = stream.getvalue()
    assert FAKE_TAVILY not in rendered
    assert "provider call failed" in rendered


def test_is_clean_detects_both_registered_and_shaped_secrets():
    r = SecretRedactor()
    r.register("Zx91ppQvNn02bLkeRt77", label="tavily")
    assert r.is_clean("nothing sensitive here")
    assert not r.is_clean("Zx91ppQvNn02bLkeRt77")
    assert not r.is_clean(FAKE_DEEPSEEK)


def test_refuses_to_register_a_value_too_short_to_be_a_key():
    r = SecretRedactor()
    with pytest.raises(ValueError, match="too short"):
        r.register("abc", label="tavily")


def test_longest_secret_wins_so_overlap_cannot_leave_a_fragment():
    r = SecretRedactor()
    r.register("SHORTSECRET01", label="a")
    r.register("SHORTSECRET01_EXTENDED_TAIL", label="b")
    out = r.redact("value=SHORTSECRET01_EXTENDED_TAIL")
    assert "EXTENDED_TAIL" not in out
    assert "SHORTSECRET01" not in out


def test_empty_and_none_registration_are_noops():
    r = SecretRedactor()
    r.register(None, label="x")
    r.register("", label="x")
    assert r.redact("plain text") == "plain text"
