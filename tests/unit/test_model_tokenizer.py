"""The live evidence budget uses the frozen model tokenizer, never whitespace by accident."""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace

from shapeflow_p1.evidence.model_tokenizer import (
    FrozenModelTokenizer,
    tokenizer_sha256,
)
from shapeflow_p1.hashing import sha256_hex


def _tokenizer_file(tmp_path: Path) -> Path:
    backend = Tokenizer(WordLevel(
        {"[UNK]": 0, "alpha": 1, "beta": 2, "gamma": 3},
        unk_token="[UNK]",
    ))
    backend.pre_tokenizer = Whitespace()
    path = tmp_path / "tokenizer.json"
    backend.save(str(path))
    return path


def test_frozen_model_tokenizer_counts_and_offsets_from_exact_file(tmp_path):
    path = _tokenizer_file(tmp_path)
    tokenizer = FrozenModelTokenizer(path)

    assert tokenizer.count("alpha beta gamma") == 3
    assert tokenizer.encode_offsets("alpha beta gamma") == [
        (0, 5), (6, 10), (11, 16),
    ]
    assert tokenizer_sha256(tokenizer) == sha256_hex(path.read_bytes())


def test_tokenizer_identity_changes_when_the_frozen_bytes_change(tmp_path):
    first = _tokenizer_file(tmp_path)
    first_digest = tokenizer_sha256(FrozenModelTokenizer(first))
    first.write_text(first.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    second_digest = tokenizer_sha256(FrozenModelTokenizer(first))
    assert second_digest != first_digest
