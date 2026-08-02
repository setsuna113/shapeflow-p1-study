"""The retrieval freeze: one record naming everything that decides what a query returns.

Freeze-1 §3.1 reserves a slot for "retriever and index commit, top-k, slicing rule, full-document
fetch rule, token-aware shared truncation". This is that slot, filled as a single content-addressed
object and hashed into the execution binding, so a change to any part of it invalidates the
approval instead of quietly changing what every arm retrieved.

Two properties beyond recording values.

**It is not effective until a pilot says so.** ``effective_after`` holds the digest of the P0
competence pilot artifact and is empty until that pilot passes. A freeze that took effect on
being written would let a retriever nobody had validated become the frozen one by default, and
the first evidence of a problem would be a whole campaign's worth of answers nobody can trust.

**Its parameters are the ones that changed the vectors, not the ones that describe them.** The
dtype is in here because the encoder is served in that dtype; the conformance digest is in here
because a freeze whose encoder was never checked against the index is a filename, not a fact.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

from ..canonical import canonical_json
from ..hashing import sha256_hex

__all__ = ["RetrievalFreeze", "FreezeNotEffective", "write_freeze", "load_freeze"]


class FreezeNotEffective(RuntimeError):
    """The retrieval freeze exists but has not been validated by the competence pilot."""


@dataclass(frozen=True)
class RetrievalFreeze:
    # --- what turns text into a vector ---
    encoder_repo: str
    encoder_revision: str
    encoder_dtype: str
    pooling: str
    normalize: bool
    query_prefix_sha256: str
    passage_prefix_sha256: str
    query_max_len: int
    passage_max_len: int

    # --- what the vector is searched against ---
    index_subset: str
    index_dim: int
    index_num_docs: int
    index_shard_sha256: tuple[str, ...]

    # --- what comes back ---
    top_k: int
    full_document: bool = True

    # --- provenance ---
    corpus_shard_sha256: tuple[str, ...] = ()
    bench_repo_commit: str = ""
    #: Digest of the conformance report proving this encoder reproduces this index.
    conformance_sha256: str = ""
    #: Digest of the competence pilot artifact. Empty until the pilot passes; see the module
    #: docstring. Nothing may retrieve under this freeze while it is empty.
    effective_after: str = ""
    notes: Mapping[str, object] = field(default_factory=dict)

    @property
    def effective(self) -> bool:
        return bool(self.effective_after)

    def require_effective(self) -> None:
        if not self.effective:
            raise FreezeNotEffective(
                f"the retrieval freeze for {self.encoder_repo} has no competence-pilot digest, so "
                "it is not in force. Run the pilot and record its artifact before retrieving "
                "under this freeze: an unvalidated retriever becoming the frozen one by default "
                "is exactly what this field prevents.")
        if not self.conformance_sha256:
            raise FreezeNotEffective(
                "the retrieval freeze records no conformance digest; an encoder that was never "
                "checked against its index is a filename, not a fact")

    def content(self) -> dict:
        return {
            "encoder": {
                "repo": self.encoder_repo,
                "revision": self.encoder_revision,
                "dtype": self.encoder_dtype,
                "pooling": self.pooling,
                "normalize": self.normalize,
                "query_prefix_sha256": self.query_prefix_sha256,
                "passage_prefix_sha256": self.passage_prefix_sha256,
                "query_max_len": self.query_max_len,
                "passage_max_len": self.passage_max_len,
            },
            "index": {
                "subset": self.index_subset,
                "dim": self.index_dim,
                "num_docs": self.index_num_docs,
                "shard_sha256": list(self.index_shard_sha256),
            },
            "results": {"top_k": self.top_k, "full_document": self.full_document},
            "provenance": {
                "corpus_shard_sha256": list(self.corpus_shard_sha256),
                "bench_repo_commit": self.bench_repo_commit,
                "conformance_sha256": self.conformance_sha256,
            },
            "effective_after": self.effective_after,
            "notes": dict(self.notes),
        }

    @property
    def digest(self) -> str:
        return sha256_hex(canonical_json(self.content()))


def write_freeze(freeze: RetrievalFreeze, path: Path) -> dict:
    """Write once. Re-writing an identical freeze is fine; a different one is an error.

    ``effective_after`` is part of the digest, so recording the pilot result *changes* the
    freeze rather than annotating it. That is intended: the validated retriever and the
    unvalidated one are different objects, and the binding should be able to tell them apart.
    """
    body = freeze.content()
    body["digest"] = freeze.digest
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with path.open("x", encoding="utf-8") as handle:
            handle.write(json.dumps(body, indent=2, sort_keys=True) + "\n")
        return body
    except FileExistsError:
        pass
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing.get("digest") != body["digest"]:
        raise RuntimeError(
            f"{path} already holds a different retrieval freeze "
            f"({existing.get('digest', '')[:12]} vs {body['digest'][:12]}). Two retrievers under "
            "one freeze cannot both be the one every arm searched.")
    return existing


def load_freeze(path: Path) -> RetrievalFreeze:
    body = json.loads(Path(path).read_text(encoding="utf-8"))
    enc, idx, res, prov = (body["encoder"], body["index"], body["results"], body["provenance"])
    freeze = RetrievalFreeze(
        encoder_repo=enc["repo"], encoder_revision=enc["revision"], encoder_dtype=enc["dtype"],
        pooling=enc["pooling"], normalize=enc["normalize"],
        query_prefix_sha256=enc["query_prefix_sha256"],
        passage_prefix_sha256=enc["passage_prefix_sha256"],
        query_max_len=enc["query_max_len"], passage_max_len=enc["passage_max_len"],
        index_subset=idx["subset"], index_dim=idx["dim"], index_num_docs=idx["num_docs"],
        index_shard_sha256=tuple(idx["shard_sha256"]),
        top_k=res["top_k"], full_document=res["full_document"],
        corpus_shard_sha256=tuple(prov["corpus_shard_sha256"]),
        bench_repo_commit=prov["bench_repo_commit"],
        conformance_sha256=prov["conformance_sha256"],
        effective_after=body["effective_after"], notes=body.get("notes", {}),
    )
    if freeze.digest != body.get("digest"):
        raise RuntimeError(f"{path} has been edited: records {body.get('digest', '')[:12]}, "
                           f"hashes to {freeze.digest[:12]}")
    return freeze
