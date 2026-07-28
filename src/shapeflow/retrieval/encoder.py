"""The query encoder, reproducing exactly how the published index was built.

Torch and transformers are imported **lazily, inside the loader**. This module must stay
importable in an environment that has neither, because the study package's dependency closure is
pinned to the vendor's own lock and a deep-learning stack has no business in it. The encoder runs
in its own interpreter; this file is the recipe and the boundary.

The recipe is not ours and must not be improved. It reproduces the Tevatron commands that
produced the shipped vectors, and every parameter below is load-bearing:

| | queries | passages |
|---|---|---|
| prefix | ``Instruct: ...\\nQuery:`` | **empty** |
| max length | 512 | 4096 |
| pooling | last token | last token |
| normalise | yes | yes |

The asymmetry is the trap. Encoding a passage *with* the query instruction produces a vector that
is confidently wrong: cosine against the stored one drops far enough to look like a broken model,
a mismatched revision, or a corrupt index -- anything except the one-line cause. The conformance
check exists to catch that before a GPU hour is spent, and it can only catch it if this table is
right.

Left padding is why last-token pooling works: with left padding the final position is always the
final real token, so pooling never has to find it. Right padding would silently pool a pad token
for every sequence shorter than the batch maximum.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

__all__ = ["EncoderSpec", "QueryEncoder", "QUERY_PREFIX", "PASSAGE_PREFIX"]

#: Verbatim from the benchmark's own encode command. The literal ``\n`` matters.
QUERY_PREFIX = (
    "Instruct: Given a web search query, retrieve relevant passages that answer the query\nQuery:"
)
PASSAGE_PREFIX = ""


@dataclass(frozen=True)
class EncoderSpec:
    """Everything that decides what a vector comes out as."""

    model: str = "Qwen/Qwen3-Embedding-0.6B"
    revision: str = ""
    query_max_len: int = 512
    passage_max_len: int = 4096
    pooling: str = "eos"
    normalize: bool = True
    dtype: str = "float32"
    padding_side: str = "left"

    def content(self) -> dict:
        from ..hashing import sha256_hex

        return {
            "model": self.model,
            "revision": self.revision,
            "query_max_len": self.query_max_len,
            "passage_max_len": self.passage_max_len,
            "pooling": self.pooling,
            "normalize": self.normalize,
            "dtype": self.dtype,
            "padding_side": self.padding_side,
            "query_prefix_sha256": sha256_hex(QUERY_PREFIX.encode("utf-8")),
            "passage_prefix_sha256": sha256_hex(PASSAGE_PREFIX.encode("utf-8")),
        }


class QueryEncoder:
    """Loads the model once and embeds text. CPU by default, and by design.

    Never place this on a GPU. Each device already serves a vLLM engine at high memory
    utilisation, and the sustainable arrival rate on that engine is the headline measurement --
    an encoder sharing the device spends SM time the measurement would attribute to serving.
    """

    def __init__(self, spec: EncoderSpec = EncoderSpec(), *, device: str = "cpu",
                 threads: int | None = None) -> None:
        self.spec = spec
        self.device = device
        self._threads = threads
        self._model = None
        self._tokenizer = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        if self._threads:
            torch.set_num_threads(self._threads)
        kwargs = {"revision": self.spec.revision} if self.spec.revision else {}
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.spec.model, padding_side=self.spec.padding_side, **kwargs)
        self._model = AutoModel.from_pretrained(
            self.spec.model, torch_dtype=getattr(torch, self.spec.dtype), **kwargs)
        self._model.eval().to(self.device)

    def encode(self, texts: Sequence[str], *, is_query: bool):
        """Embed ``texts``. Returns a float32 array, one L2-normalised row per input."""
        import numpy as np
        import torch

        self._load()
        prefix = QUERY_PREFIX if is_query else PASSAGE_PREFIX
        max_len = self.spec.query_max_len if is_query else self.spec.passage_max_len
        batch = self._tokenizer(
            [prefix + t for t in texts],
            padding=True, truncation=True, max_length=max_len, return_tensors="pt",
        ).to(self.device)

        with torch.no_grad():
            hidden = self._model(**batch).last_hidden_state
        if self.spec.pooling != "eos":
            raise ValueError(f"unsupported pooling {self.spec.pooling!r}; the index is eos-pooled")
        # Left padding puts the final real token last, for every sequence in the batch.
        pooled = hidden[:, -1, :]
        if self.spec.normalize:
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        return pooled.to(torch.float32).cpu().numpy().astype(np.float32)

    def encode_query(self, text: str):
        return self.encode([text], is_query=True)[0]
