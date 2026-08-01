"""Load the exact tokenizer that defines every evidence and publication token budget.

Whitespace token counts are useful for unit tests, but they are not a conservative proxy for
Qwen's subword tokenizer: either direction can differ, and therefore a 512-"token" output can
silently exceed the actual model budget.  Production has no fallback.  It loads the frozen
``tokenizer.json`` from the exact model directory whose Merkle root is checked by the stack
gate, and every direct-node trace records that file's digest.
"""

from __future__ import annotations

import os
from pathlib import Path

from ..hashing import sha256_hex
from .chunkers import Tokenizer, WhitespaceTokenizer

__all__ = [
    "FrozenModelTokenizer",
    "load_frozen_tokenizer",
    "tokenizer_sha256",
]

_TEST_TOKENIZER_ENV = "SHAPEFLOW_TEST_TOKENIZER"
_WHITESPACE_ID = sha256_hex(b"shapeflow:whitespace-tokenizer:v1")


class FrozenModelTokenizer:
    """Offset-preserving wrapper over one content-addressed Hugging Face tokenizer JSON."""

    def __init__(self, tokenizer_json: Path | str) -> None:
        path = Path(tokenizer_json)
        try:
            raw = path.read_bytes()
        except OSError as exc:
            raise RuntimeError(f"frozen tokenizer file {path} is unreadable: {exc}") from exc
        try:
            from tokenizers import Tokenizer as BackendTokenizer
        except ImportError as exc:  # pragma: no cover - packaging gate exercises this on host
            raise RuntimeError(
                "the pinned 'tokenizers' runtime is absent; refusing to substitute "
                "whitespace counts for the model tokenizer"
            ) from exc
        try:
            self._backend = BackendTokenizer.from_file(str(path))
        except Exception as exc:  # noqa: BLE001 - normalize native parser failures
            raise RuntimeError(f"frozen tokenizer file {path} cannot be loaded: {exc}") from exc
        self.path = path
        self.identity_sha256 = sha256_hex(raw)

    def _encode(self, text: str):
        encoded = self._backend.encode(text, add_special_tokens=False)
        offsets = list(encoded.offsets)
        if len(offsets) != len(encoded.ids):
            raise RuntimeError("tokenizer returned different id and offset counts")
        previous_start = 0
        for start, end in offsets:
            if not (
                isinstance(start, int)
                and isinstance(end, int)
                and 0 <= start < end <= len(text)
                and start >= previous_start
            ):
                raise RuntimeError(
                    f"tokenizer returned an unusable character offset {(start, end)!r}"
                )
            previous_start = start
        return encoded

    def encode_offsets(self, text: str) -> list[tuple[int, int]]:
        return [tuple(map(int, pair)) for pair in self._encode(text).offsets]

    def count(self, text: str) -> int:
        return len(self._encode(text).ids)


def tokenizer_sha256(tokenizer: Tokenizer) -> str:
    identity = str(getattr(tokenizer, "identity_sha256", "") or "")
    if identity:
        return identity
    if isinstance(tokenizer, WhitespaceTokenizer):
        return _WHITESPACE_ID
    raise RuntimeError(
        f"tokenizer {type(tokenizer).__name__} has no frozen identity digest"
    )


def load_frozen_tokenizer(settings) -> Tokenizer:
    """Load production tokenizer; a pytest-only override must be explicit and cannot leak live."""
    test_override = os.environ.get(_TEST_TOKENIZER_ENV, "")
    if test_override:
        if not os.environ.get("PYTEST_CURRENT_TEST"):
            raise RuntimeError(
                f"{_TEST_TOKENIZER_ENV} is test-only and is forbidden outside pytest"
            )
        if test_override != "whitespace-v1":
            raise RuntimeError(f"unknown test tokenizer override {test_override!r}")
        return WhitespaceTokenizer()

    model_dir = Path(str(settings.get("stack", "model", "path")))
    relative = str(settings.get("stack", "model", "tokenizer_file"))
    tokenizer_file = model_dir / relative
    if tokenizer_file.resolve(strict=False).parent != model_dir.resolve(strict=False):
        raise RuntimeError("stack.model.tokenizer_file escapes the frozen model directory")
    return FrozenModelTokenizer(tokenizer_file)
