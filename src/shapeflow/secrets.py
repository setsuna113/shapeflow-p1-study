"""Secret redaction.

Two independent mechanisms, because either alone leaks:

**Exact-value redaction.** Live credential values are registered at startup and removed
by exact substring match. This catches a key regardless of its shape — including a
provider that changes its prefix, or a key that appears base64'd inside a serialized
request. It is the strong guarantee, but only for values we know about.

**Shape patterns.** Regexes for ``tvly-``/``sk-`` keys, bearer headers and credential
query parameters. This catches secrets we were never told about — a key pasted into a
task fixture, or one echoed back inside a provider error body.

Redaction is applied to the *rendered* log line, not to the log record's arguments,
so it also covers exception tracebacks, which is where credentials usually escape: a
provider client raises with the full request URL in the message and the default
traceback renderer prints it.

Placeholders carry a short fingerprint (``[REDACTED:tavily:a1b2c3]``) so two occurrences
of the same credential are correlatable in logs without disclosing it. The fingerprint
is 24 bits of a SHA-256 over a high-entropy key: it cannot be inverted, and it makes
"is this the rotated key or the old one?" answerable during an incident.

Note the deliberate asymmetry: this module is a backstop, not the boundary. The real
control is that only the provider identity ever holds a credential, and that logging
uses a field allowlist rather than dumping a request and filtering afterwards.
"""

from __future__ import annotations

import hashlib
import logging
import re
from typing import Iterable

__all__ = ["SecretRedactor", "REDACTOR", "redact", "register_secret", "RedactingFormatter"]

#: Registering a short value would redact common substrings out of ordinary text
#: (imagine a 4-character "secret"), so refuse anything too short to be a real key.
_MIN_REGISTERABLE_LEN = 8

_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("tavily", re.compile(r"tvly-[A-Za-z0-9_\-]{12,}")),
    ("openai-compatible", re.compile(r"sk-[A-Za-z0-9_\-]{12,}")),
    (
        "bearer",
        re.compile(r"(?i)(authorization\s*[:=]\s*[\"']?bearer\s+)([A-Za-z0-9._\-]{12,})"),
    ),
    (
        "query-credential",
        re.compile(r"(?i)([?&](?:api_?key|access_?token|token|key)=)([^&\s\"']{8,})"),
    ),
)


def _fingerprint(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8", "replace")).hexdigest()[:6]


class SecretRedactor:
    """Removes credentials from text. Safe to share across threads."""

    def __init__(self) -> None:
        # Kept longest-first so a key that contains another registered value as a
        # substring is replaced whole, instead of being partially rewritten into
        # something the later pass no longer recognizes.
        self._secrets: list[tuple[str, str]] = []

    def register(self, value: str | None, *, label: str = "secret") -> None:
        """Register a live credential for exact redaction. Never logs ``value``."""
        if not value:
            return
        value = value.strip()
        if len(value) < _MIN_REGISTERABLE_LEN:
            raise ValueError(
                f"refusing to register a {len(value)}-character {label} for redaction: "
                "too short to distinguish from ordinary text"
            )
        if any(existing == value for existing, _ in self._secrets):
            return
        self._secrets.append((value, f"[REDACTED:{label}:{_fingerprint(value)}]"))
        self._secrets.sort(key=lambda pair: len(pair[0]), reverse=True)

    def register_all(self, values: Iterable[tuple[str | None, str]]) -> None:
        for value, label in values:
            self.register(value, label=label)

    def redact(self, text: str) -> str:
        if not text:
            return text
        for secret, placeholder in self._secrets:
            if secret in text:
                text = text.replace(secret, placeholder)
        for label, pattern in _PATTERNS:
            if pattern.groups == 2:
                # Keep the identifying prefix ("Authorization: Bearer ") so the log still
                # shows which header leaked, and replace only the credential itself.
                text = pattern.sub(
                    lambda m, _label=label: f"{m.group(1)}[REDACTED:{_label}:{_fingerprint(m.group(2))}]",
                    text,
                )
            else:
                text = pattern.sub(
                    lambda m, _label=label: f"[REDACTED:{_label}:{_fingerprint(m.group(0))}]",
                    text,
                )
        return text

    def is_clean(self, text: str) -> bool:
        """True if ``text`` contains no registered secret and matches no pattern.

        Used by the launch gate and the leak tests to assert over an artifact's bytes.
        """
        return self.redact(text) == text


#: Process-wide instance. Provider clients register into this at startup.
REDACTOR = SecretRedactor()


def redact(text: str) -> str:
    return REDACTOR.redact(text)


def register_secret(value: str | None, *, label: str = "secret") -> None:
    REDACTOR.register(value, label=label)


class RedactingFormatter(logging.Formatter):
    """Wraps another formatter and redacts its fully rendered output.

    Wrapping the *output* rather than the record is what makes this cover tracebacks and
    ``%``-interpolated arguments, which is where credentials actually escape.
    """

    def __init__(self, inner: logging.Formatter, redactor: SecretRedactor | None = None) -> None:
        super().__init__()
        self._inner = inner
        self._redactor = redactor or REDACTOR

    def format(self, record: logging.LogRecord) -> str:
        return self._redactor.redact(self._inner.format(record))

    def formatException(self, ei) -> str:  # noqa: ANN001 - matches logging's signature
        return self._redactor.redact(self._inner.formatException(ei))

    def formatStack(self, stack_info: str) -> str:
        return self._redactor.redact(self._inner.formatStack(stack_info))
